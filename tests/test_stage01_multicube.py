import unittest

import numpy as np

from musepipe.stages.stage01_align import compute_stage01_products


def _cube_with_star(y, x, *, shape=(12, 30, 30)):
    nz, ny, nx = shape
    cube = np.zeros(shape, dtype=np.float32)
    yy, xx = np.indices((ny, nx), dtype=float)
    star = np.exp(-0.5 * (((yy - y) ** 2 + (xx - x) ** 2) / 1.5**2))
    cube += star[None, :, :].astype(np.float32)
    cube += np.linspace(0, 0.1, nz, dtype=np.float32)[:, None, None]
    return cube


class Stage01MulticubeTests(unittest.TestCase):
    def test_multicube_shifts_recover_offsets_with_peak_centering(self):
        cubes = [_cube_with_star(15, 15), _cube_with_star(17, 14), _cube_with_star(14, 18)]
        # Write synthetic FITS to temp files because Stage01 is intentionally
        # file-oriented like the real pipeline.
        import tempfile
        from pathlib import Path
        from astropy.io import fits

        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for i, cube in enumerate(cubes):
                path = Path(tmp) / f"cube_{i}.fits"
                hdr = fits.Header()
                hdr["CRVAL3"] = 5000.0
                hdr["CRPIX3"] = 1.0
                hdr["CDELT3"] = 1.0
                fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(cube, header=hdr)]).writeto(path)
                files.append(str(path))
            cfg = {
                "run_id": "synthetic",
                "target_name": "synthetic",
                "cube_files": files,
                "data_ext": 1,
                "crop_npix": 16,
                "drop_wave_min_A": 5780.0,
                "drop_wave_max_A": 6050.0,
                "centering_method": "peak",
                "stage01_profile": "peak",
                "spatial_shift_mode": "integer",
                "spectral_grid_mode": "common_grid",
                "stage01_initial_crop_npix": None,
            }
            product = compute_stage01_products(cfg)
        shifts = product.qc["spatial_shifts"]
        self.assertAlmostEqual(shifts[0]["shift_y"], 0.0, places=1)
        self.assertLess(shifts[1]["shift_y"], 0.0)
        self.assertGreater(shifts[2]["shift_x"], -5.0)
        self.assertEqual(product.cubes.shape[:2], (3, 12))


if __name__ == "__main__":
    unittest.main()
