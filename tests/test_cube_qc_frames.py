import unittest

import numpy as np
from astropy.io import fits

from musepipe.qc.cube_qc import (
    C_KMS,
    Skyline,
    detect_wavelength_frame,
    expected_skyline_wave,
    fit_wavelength_offsets,
    measure_skylines,
)


def _gaussian(wave, center, sigma, amp):
    return amp * np.exp(-0.5 * ((wave - center) / sigma) ** 2)


class CubeQcFrameTests(unittest.TestCase):
    def test_barycentric_skyline_expectation_uses_vbary(self):
        lab = 6300.304
        vbary = 25.0
        expected = expected_skyline_wave(lab, frame="barycentric", vbary_kms=vbary)
        self.assertAlmostEqual(expected, lab * (1.0 - vbary / C_KMS))

    def test_header_detection_reads_barycentric_velocity(self):
        header = fits.Header()
        header["HIERARCH ESO DRS MUSE RVCORR"] = 25.0
        frame, vbary = detect_wavelength_frame(header)
        self.assertEqual(frame, "barycentric")
        self.assertEqual(vbary, 25.0)

    def test_header_detection_reads_specsys_barycent_without_velocity(self):
        header = fits.Header()
        header["SPECSYS"] = "BARYCENT"
        frame, vbary = detect_wavelength_frame(header)
        self.assertEqual(frame, "barycentric")
        self.assertIsNone(vbary)

    def test_m1_corrects_barycentric_frame(self):
        line = Skyline("OI_6300", 6300.304)
        vbary = 25.0
        expected = expected_skyline_wave(line.wave_A, frame="barycentric", vbary_kms=vbary)
        wave = np.arange(expected - 5.0, expected + 5.0, 0.05)
        spec = 1.0 + _gaussian(wave, expected, 0.45, 100.0)
        measurements = measure_skylines(
            wave,
            spec,
            [line],
            frame="barycentric",
            vbary_kms=vbary,
            min_snr=1.0,
        )
        fit = fit_wavelength_offsets(measurements)
        self.assertAlmostEqual(fit["offset_median_A"], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
