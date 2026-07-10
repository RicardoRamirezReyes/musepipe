import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.stages.stage01c_localize import (
    compute_stage01c_products,
    sep_pa_from_positions,
    stage01c_paths,
    write_stage01c_products,
)


def gaussian2d(ny, nx, y0, x0, amp, sigma):
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    return amp * np.exp(-0.5 * (((yy - y0) ** 2 + (xx - x0) ** 2) / sigma**2))


def write_stack(path, cube, wavelengths):
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(cube[None].astype(np.float32), name="CUBES"),
            fits.ImageHDU(np.asarray(wavelengths, dtype=np.float64), name="WAVELENGTH"),
        ]
    ).writeto(path)


def synthetic_cube(*, companion_yx=(30.0, 70.0), field_yx=(70.0, 25.0), drift_px=0.0):
    rng = np.random.default_rng(2)
    wavelengths = np.linspace(6000.0, 9000.0, 48)
    ny, nx = 96, 104
    primary_yx = (45.0, 45.0)
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    halo = 90.0 * np.exp(-np.hypot(yy - primary_yx[0], xx - primary_yx[1]) / 22.0)
    planes = []
    for i, wave in enumerate(wavelengths):
        frac = i / (wavelengths.size - 1)
        cy = companion_yx[0]
        cx = companion_yx[1] + drift_px * (frac - 0.5)
        image = 3.0 + halo
        image += gaussian2d(ny, nx, primary_yx[0], primary_yx[1], 600.0, 2.0)
        image += gaussian2d(ny, nx, cy, cx, 80.0 * (0.7 + 0.6 * frac), 1.7)
        image += gaussian2d(ny, nx, field_yx[0], field_yx[1], 65.0, 1.8)
        image += rng.normal(0.0, 0.03, size=(ny, nx))
        planes.append(image)
    return np.asarray(planes, dtype=np.float32), wavelengths, primary_yx, companion_yx, field_yx


class Stage01cSyntheticTests(unittest.TestCase):
    def test_scene_positions_astrometry_and_outputs(self):
        cube, waves, primary_yx, companion_yx, field_yx = synthetic_cube()
        pixel_scale = 0.05
        sep, pa = sep_pa_from_positions(primary_yx, companion_yx, pixel_scale)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "stage02.fits"
            write_stack(input_path, cube, waves)
            cfg = {
                "run_id": "synthetic_stage01c",
                "project_root": str(root),
                "stage01c_input_cube_fits": str(input_path),
                "pixel_scale_arcsec": pixel_scale,
                "expected_sep_arcsec": sep,
                "expected_pa_deg": pa,
                "expected_sep_err": 0.05,
                "expected_pa_err": 3.0,
                "stage01c_companion_approx_yx": companion_yx,
                "stage01c_field_source_approx_yx": field_yx,
                "stage01c_psf_fwhm_px": 4.0,
                "stage01c_search_radius_px": 5.0,
                "stage01c_detection_snr_min": 5.0,
                "stage01c_validate_legacy": False,
                "stage01c_legacy_companion_xy": [companion_yx[1], companion_yx[0]],
            }
            product = compute_stage01c_products(cfg, input_path=input_path)
            paths = stage01c_paths(cfg["run_id"], project_root=root)
            written = write_stage01c_products(product, cfg, paths)

            self.assertTrue(written["qc_json"].exists())
            qc = product.qc
            self.assertLess(np.linalg.norm(np.asarray(qc["primary"]["pos_yx"]) - primary_yx), 0.2)
            self.assertLess(np.linalg.norm(np.asarray(qc["companion"]["pos_yx"]) - companion_yx), 0.2)
            self.assertLess(np.linalg.norm(np.asarray(qc["field_source"]["pos_yx"]) - field_yx), 0.2)
            self.assertLess(abs(qc["astrometry"]["sep_arcsec"] - sep), 0.02)
            self.assertIn("pos_yx", qc["companion"])
            self.assertIn("pos_xy", qc["companion"])


if __name__ == "__main__":
    unittest.main()
