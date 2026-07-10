import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.stages.stage_h02_artifacts import compute_stage_h02_products, stage_h02_paths, write_stage_h02_products
from musepipe.stages.stage_x10_compare import METHOD_ORDER
from tests.test_h01_helpers import h01_controls, h01_product


class H02StageTests(unittest.TestCase):
    def test_stage_h02_writes_qc_and_figures_with_unavailable_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_h02"
            paths = stage_h02_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            for method in METHOD_ORDER:
                h01_product(method).write(paths[f"spec_calibrated_{method}_object"])
                np.savez(paths[f"controls_calibrated_{method}_npz"], control_spectra=h01_controls())
            paths["stage_h01_qc_json"].write_text(
                json.dumps({"verdict": {"verdict": "non_detection"}}),
                encoding="utf-8",
            )
            with paths["halpha_detection_csv"].open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["method", "peak_wave_A", "matched_z", "global_empirical_fap"])
                writer.writeheader()
                writer.writerow({"method": "psffit", "peak_wave_A": 6562.8, "matched_z": 0.0, "global_empirical_fap": 1.0})
            paths["stage00q_qc_json"].write_text(
                json.dumps(
                    {
                        "m2_lsf": {"status": "green", "fwhm_at_halpha_A": 2.5},
                        "m1_wavelength": {"skyline_waves_A": [6577.0]},
                        "excluded_windows_A": [[5790.0, 6040.0]],
                    }
                ),
                encoding="utf-8",
            )
            paths["stage02_qc_json"].write_text(
                json.dumps({"stripe_metric": {"dirty_channels": [10]}}),
                encoding="utf-8",
            )
            cfg = {
                "run_id": run_id,
                "project_root": str(root),
                "h01_rv_sys_kms": 0.0,
                "h01_expected_controls": 100,
                "h02_placebo_centers_A": [6400.0],
            }

            product = compute_stage_h02_products(cfg, paths)
            written = write_stage_h02_products(product, cfg, paths)

            self.assertTrue(paths["stage_h02_qc_json"].exists())
            self.assertTrue(paths["t1_plot"].exists())
            self.assertIn(written["qc"]["overall"], {"survives", "mixed", "fails"})
            self.assertEqual(written["qc"]["t3"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
