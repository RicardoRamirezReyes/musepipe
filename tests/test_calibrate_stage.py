import json
import tempfile
import unittest
from pathlib import Path

from musepipe.extraction.product import SpectrumProduct
from musepipe.stages.stage_x11_calibrate import compute_stage_x11_products, stage_x11_paths, write_stage_x11_products
from tests.test_calibrate_no_double import calibration_product


class CalibrateStageTests(unittest.TestCase):
    def test_stage_x11_writes_final_product_qc_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_x11"
            paths = stage_x11_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            for method, key in {
                "aperture": "spec_aperture_object",
                "optimal_ls": "spec_optimal_object",
                "optimal_psfsub": "spec_optimal_psfsub_object",
                "psffit": "spec_psffit_object",
            }.items():
                calibration_product(method=method).write(paths[key])
            paths["stage00q_qc_json"].write_text(
                json.dumps(
                    {
                        "stage": "00q_cube_qc",
                        "run_id": run_id,
                        "cube": {"wavelength_frame": "barycentric", "vbary_kms": 4.0},
                        "m1_wavelength": {"status": "green", "offset_median_A": 0.0},
                        "m3_flux": {"status": "green", "scale_factor": 1.5, "scale_err_frac": 0.10},
                    }
                ),
                encoding="utf-8",
            )
            paths["stage_e01_qc_json"].write_text(
                json.dumps({"companion_ring_metric": {"residual_pct_median": 2.0}}),
                encoding="utf-8",
            )
            cfg = {"run_id": run_id, "project_root": str(root), "x11_canonical_method": "psffit"}

            product = compute_stage_x11_products(cfg, paths)
            written = write_stage_x11_products(product, cfg, paths)

            self.assertEqual(written["qc"]["canonical_method"], "psffit")
            self.assertTrue(paths["spec_final_object"].exists())
            self.assertTrue(paths["stage_x11_qc_json"].exists())
            self.assertTrue(paths["stage_x11_error_budget_png"].exists())
            final = SpectrumProduct.read(paths["spec_final_object"])
            self.assertIn("flux_err_total", final.extra_columns)
            self.assertIn("cont_runmed", final.extra_columns)


if __name__ == "__main__":
    unittest.main()
