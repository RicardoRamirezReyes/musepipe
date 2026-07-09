import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.stages.stage_x10_compare import compute_stage_x10_products, stage_x10_paths, write_stage_x10_products
from tests.test_compare_verdicts import clean_controls, make_products


class CompareStageTests(unittest.TestCase):
    def test_stage_x10_writes_tables_qc_and_overview(self):
        products = make_products()
        controls = clean_controls()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_x10"
            paths = stage_x10_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            products["aperture"].write(paths["spec_aperture_object"])
            products["optimal_ls"].write(paths["spec_optimal_object"])
            products["optimal_psfsub"].write(paths["spec_optimal_psfsub_object"])
            products["psffit"].write(paths["spec_psffit_object"])
            np.savez(paths["controls_aperture_npz"], control_spectra=controls["aperture"])
            np.savez(paths["controls_optimal_ls_npz"], control_spectra=controls["optimal_ls"])
            np.savez(paths["controls_optimal_psfsub_npz"], control_spectra=controls["optimal_psfsub"])
            np.savez(paths["controls_psffit_npz"], control_spectra=controls["psffit"])

            cfg = {"run_id": run_id, "project_root": str(root), "x10_sigma_smooth_channels": 1}
            product = compute_stage_x10_products(cfg, paths)
            written = write_stage_x10_products(product, cfg, paths)

            self.assertEqual(written["qc"]["verdict"], "consistent")
            self.assertTrue(paths["method_comparison_csv"].exists())
            self.assertTrue(paths["method_comparison_controls_csv"].exists())
            self.assertTrue(paths["stage_x10_qc_json"].exists())
            self.assertTrue(paths["stage_x10_overview_png"].exists())


if __name__ == "__main__":
    unittest.main()
