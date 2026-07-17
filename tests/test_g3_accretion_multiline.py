"""G3 §7: multiline L_acc — combined limit = most restrictive, detections
compatibility, Mdot Monte-Carlo; WP-10 own vs config M,R + V5."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.constants import MSUN_OVER_MJUP, RJUP_CM, RSUN_CM
from musepipe.models.accretion import combine_accretion, line_lacc, mdot_from_lacc, mdot_mc
from musepipe.stages.stage_g3_accretion import (
    _own_mr_from_derived, compute_stage_g3_accretion, stage_g3_paths,
)

REL = {"a": 1.13, "b": 1.74, "scatter_dex": 0.30, "citation": "Alcala+2017", "validity_range": [-5, -1]}


def _make_run(tmp, run_id="syn", derived=None):
    rundir = Path(tmp) / "runs" / run_id
    (rundir / "tables").mkdir(parents=True)
    (rundir / "stages").mkdir(parents=True)
    with (rundir / "tables" / "g2_line_measurements.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "rest_A", "status", "flux_upper_limit_5sigma", "flux_direct"])
        w.writerow(["Halpha", "6562.80", "upper_limit", "5.0e3", ""])
    if derived is not None:
        (rundir / "stages" / "g3_rows_derived.json").write_text(json.dumps(derived))
    return Path(tmp)


def _cfg():
    return {"run_id": "syn", "h03_distance_pc": 138.6, "h03_av": 1.8, "h03_av_err": 0.5,
            "h03_rv_extinction": 3.1, "h03_extinction_law_citation": "Cardelli+1989",
            "h03_flux_unit_cgs": 1e-20, "h03_companion_mass_msun": 0.0167,
            "h03_companion_radius_rsun": 0.135, "h03_companion_mass_err_msun": 0.0014,
            "h03_companion_radius_err_rsun": 0.017, "g3_seed": 0, "g3_n_mc": 500,
            "h03_relation_scatter_dex": 0.30,
            "g3_lacc_relations": {"Halpha": {"a": 1.13, "b": 1.74, "scatter_dex": 0.41,
                                             "citation": "Alcala+2017", "validity_range": "x"}}}


class AccretionMultilineTests(unittest.TestCase):
    def test_relation_requires_citation(self):
        with self.assertRaises(RuntimeError):
            line_lacc(1e-16, distance_pc=138.6, av=1.8, a_lambda_over_av=0.818,
                      relation={"a": 1.13, "b": 1.74, "scatter_dex": 0.30, "citation": None})

    def test_combined_upper_limit_is_most_restrictive(self):
        per_line = [
            {"name": "Ha", "status": "upper_limit", "l_acc_lsun": 2.0e-6},
            {"name": "Hb", "status": "upper_limit", "l_acc_lsun": 5.0e-6},
            {"name": "HeI", "status": "upper_limit", "l_acc_lsun": 9.0e-6},
        ]
        out = combine_accretion(per_line)
        self.assertEqual(out["kind"], "upper_limit")
        self.assertEqual(out["l_acc_lsun"], 2.0e-6)  # smallest = most restrictive
        self.assertEqual(out["from_line"], "Ha")

    def test_compatible_detections_combine(self):
        per_line = [
            {"name": "Ha", "status": "detected", "l_acc_lsun": 1.0e-4, "l_acc_err_lsun": 2e-5, "scatter_dex": 0.3},
            {"name": "Hb", "status": "detected", "l_acc_lsun": 1.2e-4, "l_acc_err_lsun": 3e-5, "scatter_dex": 0.3},
        ]
        out = combine_accretion(per_line)
        self.assertEqual(out["kind"], "detection")
        self.assertTrue(out["compatible"])
        self.assertTrue(1.0e-4 <= out["l_acc_lsun"] <= 1.2e-4)

    def test_incompatible_detections_flagged_discrepant(self):
        per_line = [
            {"name": "Ha", "status": "detected", "l_acc_lsun": 1.0e-4, "l_acc_err_lsun": 1e-6, "scatter_dex": 0.02},
            {"name": "Hb", "status": "detected", "l_acc_lsun": 1.0e-2, "l_acc_err_lsun": 1e-4, "scatter_dex": 0.02},
        ]
        out = combine_accretion(per_line)
        self.assertEqual(out["status"], "discrepant")
        self.assertIsNone(out["l_acc_lsun"])

    def test_mdot_scales_with_lacc(self):
        a = mdot_from_lacc(1e-6, 0.0167, 0.135)
        b = mdot_from_lacc(2e-6, 0.0167, 0.135)
        self.assertAlmostEqual(b / a, 2.0, places=6)

    def test_mdot_mc_returns_ordered_percentiles(self):
        out = mdot_mc(2e-6, 0.0167, 0.0014, 0.135, 0.017, 0.30, n_mc=1000, seed=0)
        self.assertLess(out["p16"], out["p50"])
        self.assertLess(out["p50"], out["p84"])


class OwnVsConfigMRTests(unittest.TestCase):
    def test_own_mr_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.json"
            p.write_text(json.dumps([
                {"property": "mass", "value": 20.0, "err_stat_lo": 3.0, "err_stat_hi": 4.0},
                {"property": "radius", "value": 1.5, "err_stat_lo": 0.2, "err_stat_hi": 0.2}]))
            m, me, r, re_ = _own_mr_from_derived(p)
            self.assertAlmostEqual(m, 20.0 / MSUN_OVER_MJUP, places=9)
            self.assertAlmostEqual(r, 1.5 * RJUP_CM / RSUN_CM, places=9)
            self.assertIsNone(_own_mr_from_derived(Path(tmp) / "nope.json"))

    def test_stage_without_derived_uses_config_mr(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = stage_g3_paths("syn", project_root=_make_run(tmp))
            rows, qc = compute_stage_g3_accretion(_cfg(), paths)
            mdot = next(r for r in rows if r["property"] == "mdot")
            self.assertIn("config", mdot["assumptions"])
            lacc = next(r for r in rows if r["property"] == "l_acc_combined")["value"]
            md = mdot_mc(float(lacc), 0.0167, 0.0014, 0.135, 0.017, 0.30, n_mc=500, seed=0)
            self.assertAlmostEqual(float(mdot["value"]), md["p50"], places=12)  # reproduces
            self.assertIn("g3_mdot_config_mr_p50", qc["halpha_h03_consistency_v5"])

    def test_stage_with_derived_uses_own_mr(self):
        with tempfile.TemporaryDirectory() as tmp:
            derived = [{"property": "mass", "value": 20.0, "err_stat_lo": 3.0, "err_stat_hi": 4.0},
                       {"property": "radius", "value": 1.5, "err_stat_lo": 0.2, "err_stat_hi": 0.2}]
            paths = stage_g3_paths("syn", project_root=_make_run(tmp, derived=derived))
            rows, qc = compute_stage_g3_accretion(_cfg(), paths)
            mdot = next(r for r in rows if r["property"] == "mdot")
            self.assertIn("derived", mdot["assumptions"])
            self.assertIn("evolutionary_model", mdot["depends_on"])
            # differs from the config-M,R value used for V5
            self.assertNotAlmostEqual(
                float(mdot["value"]),
                qc["halpha_h03_consistency_v5"]["g3_mdot_config_mr_p50"], places=20)


if __name__ == "__main__":
    unittest.main()
