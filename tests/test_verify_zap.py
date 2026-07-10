import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.reduction.sky_zap import aperture_pixel_mask
from musepipe.reduction.verify_zap import (
    verify_eigenspectra_not_stellar,
    verify_halpha_intact,
    verify_rms_reduction_skylines,
    verify_sky_residual_symmetry,
    verify_source_continuum_intact,
    verify_stat_untouched,
)


def _write_cube(path, data, stat, history=None):
    primary = fits.PrimaryHDU()
    if history:
        primary.header.add_history(history)
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(data=np.asarray(data, dtype=np.float32), name="DATA"),
            fits.ImageHDU(data=np.asarray(stat, dtype=np.float32), name="STAT"),
        ]
    ).writeto(path)


class VerifyZapTests(unittest.TestCase):
    def test_v1_detects_skyline_rms_reduction(self):
        rng = np.random.default_rng(7)
        wave = np.array([5200.0, 5577.3, 6300.3, 6650.0])
        pre = rng.normal(0, 1, size=(wave.size, 20, 20))
        post = pre.copy()
        pre[1] *= 8.0
        pre[2] *= 6.0
        post[1] *= 2.0
        post[2] *= 2.0
        sky_mask = aperture_pixel_mask((20, 20), [(5, 5), (14, 14)], radius_px=3)
        result = verify_rms_reduction_skylines(pre, post, wave, sky_mask, min_reduction=2.0)
        self.assertTrue(result.passed)

    def test_v2_detects_three_percent_companion_flux_theft(self):
        wave = np.array([5200.0, 5400.0, 6100.0, 6200.0, 6650.0, 6750.0])
        pre = np.ones((wave.size, 12, 12), dtype=float)
        pre[:, 6, 6] = 100.0
        post = pre.copy()
        post[:, 6, 6] *= 0.97
        result = verify_source_continuum_intact(
            pre,
            post,
            wave,
            {"companion": (6, 6)},
            aperture_radius_px=0.5,
        )
        self.assertFalse(result.passed)
        self.assertGreater(result.value, 2.5)

    def test_v3_detects_halpha_change(self):
        wave = np.array([6500.0, 6520.0, 6550.0, 6563.0, 6610.0, 6620.0])
        pre = np.ones((wave.size, 10, 10), dtype=float)
        post = pre.copy()
        post[2:4, 5, 5] -= 5.0
        result = verify_halpha_intact(pre, post, wave, (5, 5), aperture_radius_px=0.5)
        self.assertFalse(result.passed)

    def test_v4_sky_residual_symmetry(self):
        rng = np.random.default_rng(3)
        cube = rng.normal(0, 1, size=(5, 12, 12))
        sky_mask = np.ones((12, 12), dtype=bool)
        self.assertTrue(verify_sky_residual_symmetry(cube, sky_mask).passed)

    def test_v5_rejects_stellar_eigenspectrum(self):
        star = np.array([1, 2, 3, 4, 5], dtype=float)
        result = verify_eigenspectra_not_stellar(np.array([[1, 2, 3, 4, 5]], dtype=float), star)
        self.assertFalse(result.passed)

    def test_v6_stat_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            pre_path = Path(tmp) / "pre.fits"
            post_path = Path(tmp) / "post.fits"
            data = np.ones((3, 4, 4), dtype=float)
            stat = np.arange(data.size, dtype=float).reshape(data.shape)
            _write_cube(pre_path, data, stat)
            _write_cube(post_path, data * 2, stat, history="A2 ZAP correction")
            self.assertTrue(verify_stat_untouched(pre_path, post_path).passed)


if __name__ == "__main__":
    unittest.main()
