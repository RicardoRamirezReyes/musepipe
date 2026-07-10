import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.reduction.telluric import (
    HALPHA_PROTECTED,
    apply_transmission_to_arrays,
    apply_transmission_to_cube_file,
    enforce_protected_transmission,
    validate_transmission_physical,
    verify_halpha_untouched,
    verify_outside_bands_unchanged,
    verify_stat_scaled,
)


def _header():
    hdr = fits.Header()
    hdr["CRVAL3"] = 6500.0
    hdr["CRPIX3"] = 1.0
    hdr["CDELT3"] = 50.0
    return hdr


def _write_cube(path, data, stat):
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.asarray(data, dtype=np.float32), header=_header(), name="DATA"),
            fits.ImageHDU(data=np.asarray(stat, dtype=np.float32), name="STAT"),
        ]
    ).writeto(path)


class TelluricApplyTests(unittest.TestCase):
    def test_apply_transmission_scales_data_and_stat_and_protects_halpha(self):
        wave = np.array([6500.0, 6550.0, 6563.0, 6900.0, 7200.0])
        data = np.ones((wave.size, 2, 2)) * 10.0
        stat = np.ones_like(data) * 4.0
        transmission = np.array([0.8, 0.5, 0.5, 0.8, 0.9])
        data_corr, stat_corr, trans = apply_transmission_to_arrays(data, stat, wave, transmission)
        self.assertEqual(trans[1], 1.0)
        self.assertEqual(trans[2], 1.0)
        self.assertTrue(np.allclose(data_corr[0], 12.5))
        self.assertTrue(np.allclose(stat_corr[0], 6.25))
        self.assertTrue(np.allclose(data_corr[2], 10.0))
        self.assertTrue(np.allclose(stat_corr[2], 4.0))

    def test_enforce_protected_transmission_sets_windows_to_one(self):
        wave = np.array([5800.0, 6563.0, 7000.0])
        trans = enforce_protected_transmission(wave, np.array([0.2, 0.2, 0.8]))
        self.assertEqual(trans[0], 1.0)
        self.assertEqual(trans[1], 1.0)
        self.assertEqual(trans[2], 0.8)

    def test_validate_transmission_physical(self):
        self.assertTrue(validate_transmission_physical([1.0, 0.9, 0.1]))
        self.assertFalse(validate_transmission_physical([1.1, 0.9]))
        self.assertFalse(validate_transmission_physical([0.0, 0.9]))

    def test_verify_outside_bands_detects_five_percent_overcorrection(self):
        wave = np.array([6500.0, 6550.0, 6900.0, 7200.0, 9000.0])
        pre = np.ones_like(wave)
        post = pre.copy()
        post[-1] *= 1.05
        self.assertFalse(verify_outside_bands_unchanged(pre, post, wave))

    def test_verify_halpha_untouched(self):
        wave = np.array([6500.0, 6550.0, 6563.0, 6900.0])
        pre = np.ones((wave.size, 5, 5))
        post = pre.copy()
        post[3, 2, 2] *= 2.0
        self.assertTrue(verify_halpha_untouched(pre, post, wave, (2, 2), aperture_radius_px=0.5))
        post[2, 2, 2] *= 1.1
        self.assertFalse(verify_halpha_untouched(pre, post, wave, (2, 2), aperture_radius_px=0.5))

    def test_verify_stat_scaled(self):
        wave = np.array([6500.0, 6550.0, 6900.0])
        stat = np.ones((wave.size, 2, 2)) * 4.0
        transmission = np.array([0.8, 0.5, 0.8])
        _, stat_post, trans = apply_transmission_to_arrays(np.ones_like(stat), stat, wave, transmission)
        self.assertTrue(verify_stat_scaled(stat, stat_post, wave, trans))

    def test_apply_transmission_to_cube_file_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "input.fits"
            out = Path(tmp) / "telcorr.fits"
            trans_path = Path(tmp) / "trans.fits"
            data = np.ones((4, 3, 3)) * 10.0
            stat = np.ones_like(data) * 4.0
            _write_cube(inp, data, stat)
            apply_transmission_to_cube_file(
                inp,
                out,
                [1.0, 1.0, 0.8, 0.9],
                transmission_output=trans_path,
            )
            self.assertTrue(out.exists())
            self.assertTrue(trans_path.exists())
            with fits.open(out) as hdul:
                self.assertIn("telluric", " ".join(str(x) for x in hdul[0].header.get("HISTORY", "")))


if __name__ == "__main__":
    unittest.main()
