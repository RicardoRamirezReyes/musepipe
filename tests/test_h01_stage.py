import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.stages.stage_h01_detect import compute_stage_h01_products, stage_h01_paths, write_stage_h01_products
from musepipe.stages.stage_x10_compare import METHOD_ORDER
from tests.test_h01_helpers import h01_controls, h01_product


class H01StageTests(unittest.TestCase):
    def test_stage_h01_writes_table_qc_and_summary_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_h01"
            paths = stage_h01_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            for method in METHOD_ORDER:
                signal = 8.0 if method in {"psffit", "aperture"} else 0.0
                product = h01_product(method, signal_flux=signal, lsf_fwhm_A=2.5)
                product.write(paths[f"spec_calibrated_{method}_object"])
                np.savez(paths[f"controls_calibrated_{method}_npz"], control_spectra=h01_controls())
            paths["stage00q_qc_json"].write_text(
                json.dumps({"m2_lsf": {"status": "green", "fwhm_at_halpha_A": 2.5}}),
                encoding="utf-8",
            )
            cfg = {
                "run_id": run_id,
                "project_root": str(root),
                "h01_rv_sys_kms": 0.0,
                "h01_expected_controls": 100,
            }

            product = compute_stage_h01_products(cfg, paths)
            written = write_stage_h01_products(product, cfg, paths)

            self.assertEqual(written["qc"]["verdict"]["verdict"], "detection")
            self.assertTrue(paths["halpha_detection_csv"].exists())
            self.assertTrue(paths["stage_h01_qc_json"].exists())
            self.assertTrue(paths["summary_plot"].exists())


if __name__ == "__main__":
    unittest.main()
