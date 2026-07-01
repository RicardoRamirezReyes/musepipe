"""Tests for the spectral metric helpers extracted in O2 (spectrum_to_snr,
integrated_line_flux)."""

import unittest

import numpy as np

from musepipe.spectral import integrated_line_flux, spectrum_to_snr


class SpectrumToSnr(unittest.TestCase):
    def test_divides_by_noise(self):
        spec = np.array([1.0, 2.0, 4.0])
        np.testing.assert_allclose(spectrum_to_snr(spec, 2.0), [0.5, 1.0, 2.0])

    def test_invalid_noise_returns_all_nan(self):
        spec = np.array([1.0, 2.0, 3.0])
        for bad in (0.0, -1.0, np.nan):
            out = spectrum_to_snr(spec, bad)
            self.assertEqual(out.shape, spec.shape)
            self.assertTrue(np.all(np.isnan(out)))

    def test_preserves_nan_in_spectrum(self):
        spec = np.array([1.0, np.nan, 3.0])
        out = spectrum_to_snr(spec, 1.0)
        self.assertTrue(np.isnan(out[1]))
        np.testing.assert_allclose(out[[0, 2]], [1.0, 3.0])


class IntegratedLineFlux(unittest.TestCase):
    def test_sum_over_indices_times_dispersion(self):
        spec = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(integrated_line_flux(spec, [1, 2, 3], 0.5), 3.0)

    def test_ignores_nan_channels(self):
        spec = np.array([1.0, np.nan, 3.0])
        self.assertAlmostEqual(integrated_line_flux(spec, [0, 1, 2], 2.0), 8.0)

    def test_indices_cast_to_int(self):
        spec = np.arange(10.0)
        self.assertAlmostEqual(integrated_line_flux(spec, np.array([2.0, 5.0]), 1.0), 7.0)


if __name__ == "__main__":
    unittest.main()
