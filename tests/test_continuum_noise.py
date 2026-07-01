"""Tests for musepipe.stats.continuum_noise_from_spectrum (O2 reconciliation:
robust MAD by default, simple std as an explicit option)."""

import unittest

import numpy as np

from musepipe.stats import continuum_noise_from_spectrum, robust_sigma


class ContinuumNoise(unittest.TestCase):
    def test_default_is_robust_mad(self):
        rng = np.random.default_rng(0)
        spec = rng.normal(size=200)
        mask = np.ones(200, dtype=bool)
        self.assertAlmostEqual(
            continuum_noise_from_spectrum(spec, mask),
            robust_sigma(spec[mask]),
        )

    def test_std_option_matches_nanstd(self):
        spec = np.arange(10.0)
        mask = np.ones(10, dtype=bool)
        self.assertAlmostEqual(
            continuum_noise_from_spectrum(spec, mask, method="std"),
            float(np.nanstd(spec)),
        )

    def test_std_option_returns_nan_below_five_channels(self):
        spec = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
        mask = np.ones(5, dtype=bool)
        self.assertTrue(np.isnan(continuum_noise_from_spectrum(spec, mask, method="std")))

    def test_robust_downweights_a_spike(self):
        rng = np.random.default_rng(1)
        spiked = rng.normal(scale=1.0, size=401)
        mask = np.ones(401, dtype=bool)
        spiked[200] = 1000.0
        # MAD-based sigma stays near the real noise (~1); std is inflated by the spike
        self.assertLess(continuum_noise_from_spectrum(spiked, mask), 3.0)
        self.assertGreater(continuum_noise_from_spectrum(spiked, mask, method="std"), 40.0)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            continuum_noise_from_spectrum(np.zeros(5), np.ones(5, dtype=bool), method="nope")


if __name__ == "__main__":
    unittest.main()
