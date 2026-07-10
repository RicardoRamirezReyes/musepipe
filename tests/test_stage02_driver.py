import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.stages.stage02_xcorr import (
    compute_stage02_products,
    stage02_paths,
    write_stage02_products,
)


class Stage02DriverTests(unittest.TestCase):
    def test_driver_writes_stage02_contract_with_stat_and_metric_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_stage02"
            stage_dir = root / "runs" / run_id / "stages"
            stage_dir.mkdir(parents=True)
            stage01 = stage_dir / "stage01_cropped_cube_stack.fits"

            wavelengths = 5000.0 + np.arange(16, dtype=np.float64) * 2.0
            cube = np.full((1, 16, 8, 8), 100.0, dtype=np.float32)
            cube[:, :, :, 4:] += 0.5
            stat = np.ones_like(cube, dtype=np.float32)
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(cube, name="CUBES"),
                    fits.ImageHDU(wavelengths, name="WAVELENGTH"),
                    fits.ImageHDU(stat, name="STAT"),
                ]
            ).writeto(stage01)

            cfg = {
                "run_id": run_id,
                "target_name": "synthetic",
                "stage01_cube_fits": str(stage01),
                "xcorr_nstripes": 2,
                "xcorr_stripe_orientation": "vertical",
                "xcorr_wmin_A": 5000.0,
                "xcorr_wmax_A": 5030.0,
                "xcorr_max_lag_ch": 2,
                "xcorr_apply_mode": "integer",
                "xcorr_metric_excluded_windows_A": [],
                "stage02_save_metric_plot": True,
                "overwrite_existing_stage_files": True,
            }
            product = compute_stage02_products(cfg)
            paths = stage02_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            write_stage02_products(product, cfg, paths)

            with fits.open(paths["stage02_cube_fits"]) as hdul:
                self.assertIn("CUBES", hdul)
                self.assertIn("WAVELENGTH", hdul)
                self.assertIn("STAT", hdul)
                np.testing.assert_array_equal(hdul["CUBES"].data, cube)
                np.testing.assert_array_equal(hdul["STAT"].data, stat)

            qc = json.loads(paths["stage02_qc_json"].read_text())
            self.assertIn("stripe_metric", qc)
            self.assertIn("stat", qc)
            self.assertIn("equivalence", qc)
            self.assertTrue(qc["stat"]["present"])
            self.assertEqual(qc["equivalence"]["verdict"], "not_checked")
            self.assertTrue(paths["stage02_metric_csv"].exists())
            self.assertTrue(paths["stage02_metric_plot"].exists())


if __name__ == "__main__":
    unittest.main()
