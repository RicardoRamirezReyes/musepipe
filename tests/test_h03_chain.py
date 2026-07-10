import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.stages.stage_h03_limits import (
    G_CGS,
    L_SUN_ERG_S,
    M_SUN_G,
    PC_CM,
    R_SUN_CM,
    YR_S,
    compute_stage_h03_products,
    limit_conversion_chain,
    stage_h03_paths,
    write_stage_h03_products,
)


def h03_physical_config(run_id, root):
    return {
        "run_id": run_id,
        "project_root": str(root),
        "h03_distance_pc": 10.0,
        "h03_distance_err_pc": 0.0,
        "h03_distance_source": "synthetic distance",
        "h03_av": 0.0,
        "h03_av_err": 0.0,
        "h03_av_source": "synthetic extinction",
        "h03_rv_sys_kms": 0.0,
        "h03_rv_source": "synthetic rv",
        "h03_a_halpha_over_av": 1.0,
        "h03_extinction_law_citation": "synthetic law",
        "h03_lacc_lha_relation": "identity",
        "h03_lacc_lha_citation": "synthetic relation",
        "h03_lacc_lha_a": 1.0,
        "h03_lacc_lha_b": 0.0,
        "h03_relation_scatter_dex": 0.3,
        "h03_companion_mass_msun": 1.0,
        "h03_companion_radius_rsun": 1.0,
        "h03_mass_source": "synthetic mass",
        "h03_radius_source": "synthetic radius",
        "h03_canonical_method": "psffit",
        "h03_template_width_factors": [1.0, 2.0],
        "h03_matched_sigma_by_method": {"psffit": {"1": 2.0e-17, "2": 3.0e-17}},
        "h03_separation_arcsec": 1.7,
    }


class H03ChainTests(unittest.TestCase):
    def test_conversion_chain_matches_analytic_values(self):
        result = limit_conversion_chain(
            z_threshold=3.0,
            z_5sigma_extrap=5.0,
            matched_sigma=2.0e-17,
            throughput=0.5,
            throughput_err=0.0,
            distance_pc=10.0,
            distance_err_pc=0.0,
            av=0.0,
            av_err=0.0,
            a_halpha_over_av=1.0,
            lacc_lha_slope=1.0,
            lacc_lha_intercept=0.0,
            relation_scatter_dex=0.3,
            mass_msun=1.0,
            radius_rsun=1.0,
        )

        f_stat = 3.0 * 2.0e-17
        f_obs = f_stat / 0.5
        lha = 4.0 * math.pi * (10.0 * PC_CM) ** 2 * f_obs
        lacc_lsun = lha / L_SUN_ERG_S
        mdot = (lacc_lsun * L_SUN_ERG_S) * R_SUN_CM / (G_CGS * M_SUN_G) * YR_S / M_SUN_G

        self.assertAlmostEqual(result["f_stat_99"], f_stat)
        self.assertAlmostEqual(result["f_lim_observed"], f_obs)
        self.assertAlmostEqual(result["f_lim_dereddened"], f_obs)
        self.assertAlmostEqual(result["l_halpha_erg_s"], lha)
        self.assertAlmostEqual(result["l_acc_lsun"], lacc_lsun)
        self.assertAlmostEqual(result["mdot_msun_yr"], mdot)
        self.assertAlmostEqual(result["mdot_err_dex"], 0.3)

    def test_stage_h03_writes_upper_limit_table_qc_and_context_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_h03"
            paths = stage_h03_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            paths["stage_h01_qc_json"].write_text(
                json.dumps(
                    {
                        "verdict": {"verdict": "non_detection"},
                        "line": {"rest_A": 6562.8, "rv_sys_kms": 0.0},
                        "templates": {"lsf_fwhm_A": 2.5},
                    }
                ),
                encoding="utf-8",
            )
            paths["stage_h02_qc_json"].write_text(json.dumps({"overall": "survives"}), encoding="utf-8")
            paths["stage_h04_qc_json"].write_text(
                json.dumps({"regression_historic": {"verdict": "pass"}, "throughput": {"psf_perturbation_pct": 0.0}}),
                encoding="utf-8",
            )
            null_maxima = np.linspace(-1.0, 2.5, 120)
            np.savez(paths["h01_null_maxima_npz"], psffit_null_maxima=null_maxima)
            with paths["halpha_detection_csv"].open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["method", "template_factor", "matched_sigma"])
                writer.writeheader()
                writer.writerow({"method": "psffit", "template_factor": 1.0, "matched_sigma": 2.0e-17})
            with paths["injection_throughput_csv"].open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["method", "template_factor", "snr", "throughput", "throughput_err"],
                )
                writer.writeheader()
                for factor in (1.0, 2.0):
                    writer.writerow({"method": "psffit", "template_factor": factor, "snr": 1.0, "throughput": 0.4, "throughput_err": 0.04})
                    writer.writerow({"method": "psffit", "template_factor": factor, "snr": 5.0, "throughput": 0.8, "throughput_err": 0.08})

            cfg = h03_physical_config(run_id, root)
            product = compute_stage_h03_products(cfg, paths)
            written = write_stage_h03_products(product, cfg, paths)

            self.assertTrue(paths["halpha_upper_limits_csv"].exists())
            self.assertTrue(paths["stage_h03_qc_json"].exists())
            self.assertTrue(paths["context_plot"].exists())
            self.assertEqual(written["qc"]["prerequisites"]["e4_v1"], "pass")
            self.assertEqual(written["qc"]["canonical_method"], "psffit")
            self.assertTrue(any(row["row_kind"] == "combined_final" for row in product.rows))


if __name__ == "__main__":
    unittest.main()
