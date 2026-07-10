import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.stages.stage01_align import compute_stage01_products, write_stage01_products, stage01_paths


class Stage01EquivalenceContractTests(unittest.TestCase):
    def test_stage01_product_names_shapes_and_qc_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_stage01"
            run_dir = root / "runs" / run_id
            (run_dir / "config").mkdir(parents=True)
            cube_path = root / "input.fits"
            cube = np.zeros((20, 25, 25), dtype=np.float32)
            cube[:, 12, 12] = 100.0
            stat = np.ones_like(cube, dtype=np.float32)
            hdr = fits.Header()
            hdr["CRVAL3"] = 5000.0
            hdr["CRPIX3"] = 1.0
            hdr["CDELT3"] = 1.0
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(cube, header=hdr),
                    fits.ImageHDU(stat, name="STAT"),
                ]
            ).writeto(cube_path)
            cfg = {
                "run_id": run_id,
                "target_name": "synthetic",
                "cube_files": [str(cube_path)],
                "data_ext": 1,
                "crop_npix": 10,
                "drop_wave_min_A": 5780.0,
                "drop_wave_max_A": 6050.0,
                "centering_method": "peak",
                "stage01_profile": "peak",
                "spatial_shift_mode": "none",
                "spectral_grid_mode": "common_grid",
                "stage01_initial_crop_npix": None,
            }
            product = compute_stage01_products(cfg)
            paths = stage01_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            write_stage01_products(product, cfg, paths)
            with fits.open(paths["stage01_cube_fits"]) as hdul:
                self.assertIn("CUBES", hdul)
                self.assertIn("WAVELENGTH", hdul)
                self.assertIn("STAT", hdul)
                self.assertEqual(hdul["CUBES"].data.shape, (1, 20, 10, 10))
            qc = json.loads(paths["stage01_qc_json"].read_text())
            self.assertIn("shifts", qc)
            self.assertIn("stat", qc)
            self.assertTrue(qc["stat"]["propagated"])


if __name__ == "__main__":
    unittest.main()
