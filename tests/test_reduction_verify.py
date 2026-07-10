import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.reduction.verify import (
    VerificationError,
    best_integer_shift_correlation,
    verify_sky_mask_clean,
    verify_standard_response,
    verify_star_spectrum_vs_adp,
    verify_stat,
    verify_wcs_headers,
    verify_whitelight_vs_adp,
)


def _cube_header():
    header = fits.Header()
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = -20.0
    header["CRPIX1"] = 3.0
    header["CRPIX2"] = 3.0
    header["CDELT1"] = 0.2
    header["CDELT2"] = 0.2
    header["CRVAL3"] = 4800.0
    header["CRPIX3"] = 1.0
    header["CDELT3"] = 1.25
    return header


def _write_cube(path, data, stat=None, header=None):
    hdus = [fits.PrimaryHDU()]
    hdus.append(fits.ImageHDU(data=np.asarray(data, dtype=np.float32), header=header or _cube_header(), name="DATA"))
    if stat is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(stat, dtype=np.float32), name="STAT"))
    fits.HDUList(hdus).writeto(path)


class ReductionVerifyTests(unittest.TestCase):
    def test_verify_stat_passes_good_data_and_stat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cube.fits"
            data = np.ones((4, 5, 5))
            stat = np.ones_like(data) * 0.25
            _write_cube(path, data, stat)
            result = verify_stat(path)
            self.assertTrue(result.passed)
            self.assertEqual(result.name, "v1_stat_present")

    def test_verify_stat_fails_missing_stat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cube.fits"
            _write_cube(path, np.ones((4, 5, 5)), stat=None)
            with self.assertRaises(VerificationError):
                verify_stat(path)

    def test_verify_stat_excludes_laser_region(self):
        # Channels that fall in the NaLGS laser range are NaN in every spaxel;
        # V1 must exclude them and still pass instead of counting them as defects.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cube.fits"
            header = _cube_header()
            header["CRVAL3"] = 5700.0
            header["CRPIX3"] = 1.0
            header["CDELT3"] = 100.0  # channels: 5700, 5800, 5900, 6000, 6100, 6200
            data = np.ones((6, 5, 5))
            data[1:4] = np.nan  # 5800, 5900, 6000 all inside [5780, 6050]
            stat = np.ones_like(data) * 0.25
            stat[1:4] = np.nan
            _write_cube(path, data, stat, header)
            result = verify_stat(path)
            self.assertTrue(result.passed)
            self.assertAlmostEqual(result.value, 0.0, places=6)
            self.assertIn("laser_channels_excluded=3", result.message)

    def test_verify_standard_response_excludes_bad_ranges(self):
        wave = np.array([5000.0, 5900.0, 6500.0])
        reference = np.ones(3)
        response = np.array([1.02, 9.0, 0.98])
        result = verify_standard_response(wave, response, reference, max_rms=0.05)
        self.assertTrue(result.passed)

    def test_verify_wcs_headers_passes_matching_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "cube.fits"
            adp = Path(tmp) / "adp.fits"
            data = np.ones((4, 5, 5))
            _write_cube(cube, data, np.ones_like(data), _cube_header())
            _write_cube(adp, data, np.ones_like(data), _cube_header())
            self.assertTrue(verify_wcs_headers(cube, adp).passed)

    def test_best_integer_shift_correlation_finds_shifted_scene(self):
        image = np.zeros((10, 10))
        image[3:6, 4:7] = 1.0
        reference = np.zeros((10, 10))
        reference[4:7, 3:6] = 1.0
        corr, shift = best_integer_shift_correlation(image, reference, max_shift=2)
        self.assertGreater(corr, 0.99)
        self.assertEqual(shift, (-1, 1))

    def test_verify_whitelight_vs_adp_passes_same_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "cube.fits"
            adp = Path(tmp) / "adp.fits"
            data = np.zeros((4, 8, 8))
            data[:, 2:5, 3:6] = 5.0
            _write_cube(cube, data, np.ones_like(data), _cube_header())
            _write_cube(adp, data.copy(), np.ones_like(data), _cube_header())
            result = verify_whitelight_vs_adp(cube, adp)
            self.assertTrue(result.passed)
            self.assertGreater(result.value, 0.99)

    def test_verify_whitelight_vs_adp_handles_different_shapes(self):
        # The re-reduced cube and the ADP have different array sizes; V4 must
        # register on the primary star instead of requiring identical shapes.
        yy, xx = np.mgrid[-2:3, -2:3]
        blob = 10.0 * np.exp(-(yy**2 + xx**2) / 2.0)
        with tempfile.TemporaryDirectory() as tmp:
            cube_path = Path(tmp) / "cube.fits"
            adp_path = Path(tmp) / "adp.fits"
            cube = np.zeros((4, 8, 8))
            cube[:, 2:7, 2:7] = blob  # star peak at (4, 4)
            adp = np.zeros((4, 10, 10))
            adp[:, 4:9, 1:6] = blob  # same star scene, peak at (6, 3)
            _write_cube(cube_path, cube, np.ones_like(cube))
            _write_cube(adp_path, adp, np.ones_like(adp))
            result = verify_whitelight_vs_adp(cube_path, adp_path)
            self.assertTrue(result.passed)
            self.assertGreater(result.value, 0.99)

    def test_verify_star_spectrum_vs_adp_passes_smooth_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "cube.fits"
            adp = Path(tmp) / "adp.fits"
            data = np.ones((6, 9, 9))
            data[:, 4, 4] = 10.0
            _write_cube(adp, data, np.ones_like(data), _cube_header())
            _write_cube(cube, data * 1.03, np.ones_like(data), _cube_header())
            result = verify_star_spectrum_vs_adp(cube, adp, star_yx=(4, 4), radius=1.5)
            self.assertTrue(result.passed)

    def test_verify_sky_mask_clean_requires_mask(self):
        with self.assertRaises(VerificationError):
            verify_sky_mask_clean(None, [(2, 2)], radius_px=1)

    def test_verify_sky_mask_clean_fails_overlap(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[2, 2] = True
        result = verify_sky_mask_clean(mask, [(2, 2)], radius_px=1)
        self.assertFalse(result.passed)

    def test_verify_sky_mask_clean_passes_no_overlap(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[0, 0] = True
        result = verify_sky_mask_clean(mask, [(4, 4)], radius_px=1)
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
