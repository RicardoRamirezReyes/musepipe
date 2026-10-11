"""WP-G3R-12: G4 real T3/T4/T6 from the G3 QC + ambiguity quantification."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.constants import MSUN_OVER_MJUP
from musepipe.stages.stage_g4_classify import (
    _ambiguity_quantification, _combine_verdicts, _combined_label, _mass_probabilities,
    _mass_verdicts, _t3_gravity, build_matrix, compute_stage_g4, stage_g4_paths,
)

_GRAV = {"best_class": "young",
         "dchi2_by_class": {"young": 0.0, "field": 98.7, "nonstellar": 4653.0}}


class HelperTests(unittest.TestCase):
    def test_t3_gravity_verdicts(self):
        gv = _t3_gravity(_GRAV, 9.0, 4.0)
        self.assertEqual(gv["substellar_companion"][0], "supports")     # young, dchi2 0
        self.assertEqual(gv["m_star_associated"][0], "supports")        # young
        self.assertEqual(gv["m_star_background"][0], "excludes")        # field, 98.7 > 9
        self.assertEqual(gv["contaminant"][0], "excludes")             # nonstellar, 4653
        self.assertIsNone(_t3_gravity({}, 9.0, 4.0))

    def test_combine_most_unfavorable(self):
        self.assertEqual(_combine_verdicts(("supports", 0.0), ("excludes", 0.01))[0], "excludes")
        self.assertEqual(_combine_verdicts(("supports", 0.0), None)[0], "supports")
        self.assertEqual(_combine_verdicts(("neutral", 1.0), ("disfavors", 0.1))[0], "disfavors")
        self.assertEqual(_combine_verdicts(None, ("not_available", None))[0], "not_available")

    def test_mass_probabilities_and_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mp.npz"
            np.savez(p, combined=np.full(1000, 5.0 / MSUN_OVER_MJUP))  # 5 M_Jup -> <13
            probs = _mass_probabilities(p, [13.0, 75.0])
            self.assertAlmostEqual(probs["planet_lt13"], 1.0)
            mv = _mass_verdicts(probs, {"supports": 0.6, "neutral": 0.2, "disfavors": 0.05})
            self.assertEqual(mv["substellar_companion"][0], "supports")  # P(<13)=1
            self.assertEqual(mv["brown_dwarf"][0], "excludes")           # P(13-75)=0
            self.assertEqual(mv["m_star_associated"][0], "excludes")     # P(>75)=0


def _make_run(tmp, *, syslim, acceptance=None, atmo_qc=None):
    rd = Path(tmp) / "runs" / "syn"
    (rd / "stages").mkdir(parents=True)
    (rd / "tables").mkdir(parents=True)
    if atmo_qc is not None:
        (rd / "stages" / "g3_atmo_fit_qc.json").write_text(json.dumps(atmo_qc))
    (rd / "stages" / "stage01c_qc.json").write_text(json.dumps(
        {"astrometry": {"expected_sep_arcsec": 0.1, "sep_deviation_sigma": 1.0},
         "companion": {"snr_detection": 20.0}}))
    (rd / "stages" / "stage_h02_qc.json").write_text(json.dumps({"overall": "survives"}))
    with (rd / "tables" / "g2_line_measurements.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "rest_A", "status"])
        w.writerow(["Halpha", "6562.8", "upper_limit"])
    g3 = {"spt": {"gravity_classes": _GRAV}}
    if acceptance is not None:
        g3["spt"]["acceptance"] = acceptance
    if syslim:
        g3["systematics_limited"] = {"accepted": True, "reason": "C3",
                                     "robust_results": {"gravity_class": _GRAV}}
    (rd / "stages" / "stage_g3_qc.json").write_text(json.dumps(g3))
    np.savez(rd / "stages" / "g3_mass_posterior.npz",
             combined=np.full(1000, 0.02), family_BHAC15=np.full(500, 0.02))  # ~21 M_Jup -> BD
    with (rd / "tables" / "g3_physical_properties.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["property", "value", "label", "err_stat_lo",
                                          "err_stat_hi", "err_sys"])
        w.writeheader()
        lbl = "not_constrained" if syslim else "atmospheric_model_dependent"
        w.writerow({"property": "radius", "value": "" if syslim else "1.5", "label": lbl,
                    "err_stat_lo": "0.1", "err_stat_hi": "0.1", "err_sys": ""})
        w.writerow({"property": "a_v_spectral", "value": "" if syslim else "1.9", "label": lbl,
                    "err_stat_lo": "0.2", "err_stat_hi": "0.2", "err_sys": ""})
    return Path(tmp)


def _cfg():
    return {"run_id": "syn", "g4_cpm_confirmed": True, "g4_t3_dchi2_excl": 9.0,
            "g4_t3_dchi2_disfavor": 4.0, "g4_mass_boundaries_mjup": [13.0, 75.0],
            "g4_mass_prob_verdicts": {"supports": 0.6, "neutral": 0.2, "disfavors": 0.05},
            "g4_t4_radius_rjup_range": [0.5, 3.0], "g4_t6_av_sigma_support": 2.0,
            "g4_t6_av_sigma_background": 3.0, "h03_av": 1.8, "h03_av_err": 0.5,
            "g4_pointsource_snr_min": 5.0}


class BuildMatrixTests(unittest.TestCase):
    def test_systematics_limited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_run(tmp, syslim=True)
            paths = stage_g4_paths("syn", project_root=root)
            matrix, _ = build_matrix(_cfg(), paths)
            self.assertEqual(matrix["T3"]["substellar_companion"]["verdict"], "supports")
            self.assertEqual(matrix["T3"]["m_star_background"]["verdict"], "excludes")
            self.assertEqual(matrix["T4"]["substellar_companion"]["verdict"], "not_available")
            self.assertEqual(matrix["T6"]["substellar_companion"]["verdict"], "not_available")
            g3 = json.loads((root / "runs" / "syn" / "stages" / "stage_g3_qc.json").read_text())
            amb = _ambiguity_quantification(_cfg(), paths, g3)
            self.assertEqual(amb["status"], "not_available")
            self.assertEqual(amb["dchi2_gravity_classes"]["field"], 98.7)

    def test_mass_resolves_when_trustworthy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_run(tmp, syslim=False)
            paths = stage_g4_paths("syn", project_root=root)
            matrix, _ = build_matrix(_cfg(), paths)
            # mass ~21 M_Jup -> BD; gravity supports young. Combined (most unfavorable):
            self.assertEqual(matrix["T3"]["brown_dwarf"]["verdict"], "supports")      # both agree
            self.assertEqual(matrix["T3"]["substellar_companion"]["verdict"], "excludes")  # mass P(<13)=0
            self.assertEqual(matrix["T4"]["substellar_companion"]["verdict"], "supports")  # R 1.5 in range
            g3 = json.loads((root / "runs" / "syn" / "stages" / "stage_g3_qc.json").read_text())
            amb = _ambiguity_quantification(_cfg(), paths, g3)
            self.assertEqual(amb["status"], "computed")
            self.assertAlmostEqual(amb["mass_prob_combined"]["bd_13_75"], 1.0)

    def test_binned_gof_is_the_acceptance_gate_of_the_gravity(self):
        """2026-09-24: tipo y gravedad del nativo; la puerta es la bondad binada."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_run(tmp, syslim=True, acceptance={"gate": "binned_gof", "pass": False,
                                                           "chi2_red": 5.0, "threshold": 1.5})
            matrix, _ = build_matrix(_cfg(), stage_g4_paths("syn", project_root=root))
            cell = matrix["T3"]["m_star_background"]
            self.assertEqual(cell["verdict"], "not_available")
            self.assertIn("binned GOF failed", cell["source"])
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_run(tmp, syslim=True, acceptance={"gate": "binned_gof", "pass": True})
            matrix, _ = build_matrix(_cfg(), stage_g4_paths("syn", project_root=root))
            self.assertEqual(matrix["T3"]["m_star_background"]["verdict"], "excludes")


class AtmoFitNotUsableTests(unittest.TestCase):
    """2026-10-10: G4 usaba la masa, el radio y la A_V de un ajuste atmosferico
    que habia fallado sus puertas. La marca `systematics_limited` estaba en el
    config de los dos runs del paper, pero no en el QC de G3 (solo la escribe
    `finalize_systematics_limited`, que ningun lanzamiento llama), y G4 solo
    miraba el QC. Tampoco leia el sello `usable_for_downstream` del ajuste.

    En cada caso el QC de G3 NO trae el bloque `systematics_limited` y la tabla
    trae radio y A_V con valor, como en los runs del 2026-09-26: lo unico que
    puede apagar T4/T6 y la masa es la declaracion o el sello.
    """

    def _assert_atmo_off(self, cfg, atmo_qc, motivo):
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_run(tmp, syslim=False, atmo_qc=atmo_qc)
            paths = stage_g4_paths("syn", project_root=root)
            matrix, _ = build_matrix(cfg, paths)
            # sin masa, T3 es solo la gravedad: joven -> supports, aunque la masa (~21 M_Jup) la excluiria
            self.assertEqual(matrix["T3"]["substellar_companion"]["verdict"], "supports")
            self.assertIn(motivo, matrix["T3"]["substellar_companion"]["source"])
            for t in ("T4", "T6"):
                self.assertEqual(matrix[t]["substellar_companion"]["verdict"], "not_available", t)
                self.assertIn(motivo, matrix[t]["substellar_companion"]["source"], t)
            g3 = json.loads((root / "runs" / "syn" / "stages" / "stage_g3_qc.json").read_text())
            amb = _ambiguity_quantification(cfg, paths, g3)
            self.assertEqual(amb["status"], "not_available")
            self.assertIn(motivo, amb["reason"])
            _, qc = compute_stage_g4(cfg, paths)
            self.assertIn(motivo, " ".join(i["issue"] for i in qc["open_issues"]))

    def test_config_declaration_alone_switches_it_off(self):
        cfg = dict(_cfg(), g3_atmosphere_systematics_limited=True)
        self._assert_atmo_off(cfg, None, "declared in config")

    def test_failed_seal_alone_switches_it_off(self):
        atmo = {"stops": ["chi2_red=12.26 > 3 (atmo fit, 25 A bins)"], "usable_for_downstream": False}
        self._assert_atmo_off(_cfg(), atmo, "chi2_red=12.26")

    def test_atmo_qc_without_seal_is_not_usable(self):
        """La misma regla que la rodaja de acrecion: sin sello, no consta que pasara."""
        self._assert_atmo_off(_cfg(), {"best": {"chi2_red": 1.0}}, "sin el sello")

    def test_declaration_false_and_passed_seal_keep_the_mass(self):
        cfg = dict(_cfg(), g3_atmosphere_systematics_limited=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_run(tmp, syslim=False, atmo_qc={"stops": [], "usable_for_downstream": True})
            paths = stage_g4_paths("syn", project_root=root)
            matrix, _ = build_matrix(cfg, paths)
            self.assertEqual(matrix["T3"]["substellar_companion"]["verdict"], "excludes")  # P(<13)=0
            self.assertEqual(matrix["T4"]["substellar_companion"]["verdict"], "supports")
            self.assertNotEqual(matrix["T6"]["substellar_companion"]["verdict"], "not_available")
            g3 = json.loads((root / "runs" / "syn" / "stages" / "stage_g3_qc.json").read_text())
            self.assertEqual(_ambiguity_quantification(cfg, paths, g3)["status"], "computed")


class CombinedLabelTests(unittest.TestCase):
    """2026-10-10 (spec G4 §4.5): la etiqueta de un empate mira el conjunto, no el top-2."""

    def test_labels(self):
        self.assertIsNone(_combined_label(["brown_dwarf"]))
        self.assertEqual(_combined_label(["substellar_companion", "brown_dwarf"]),
                         "companion_substellar_or_planetary")
        self.assertEqual(_combined_label(["planet_forming", "substellar_companion", "brown_dwarf"]),
                         "companion_substellar_or_planetary")
        # ROXs 42B b sin masa: el top-2 son subestelares, pero la M asociada tambien empata
        self.assertEqual(_combined_label(["substellar_companion", "brown_dwarf", "m_star_associated"]),
                         "bound_companion_mass_unresolved")
        self.assertEqual(_combined_label(["m_star_associated", "m_star_background"]), "unresolved")

    def test_stage_without_mass_reports_the_bound_tie(self):
        """Sin masa, la gravedad joven no separa a las ligadas: empate y `ambiguous`."""
        cfg = dict(_cfg(), g3_atmosphere_systematics_limited=True, g4_combined_class_margin=0.5,
                   g4_background_density_per_arcsec2=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_run(tmp, syslim=False)
            _, qc = compute_stage_g4(cfg, stage_g4_paths("syn", project_root=root))
            fc = qc["final_class"]
            self.assertEqual(fc["label"], "bound_companion_mass_unresolved")
            self.assertIsNone(fc["leader"])
            # Halpha es upper_limit en el fixture: T8 deja a planet_forming en −1, como en 42B b
            self.assertEqual(set(fc["tied_at_top"]), {"substellar_companion", "brown_dwarf",
                                                      "m_star_associated"})
            self.assertEqual(fc["robustness"], "ambiguous")


if __name__ == "__main__":
    unittest.main()
