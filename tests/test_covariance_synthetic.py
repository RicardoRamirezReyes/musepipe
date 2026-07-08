"""G1 §7: correlated synthetic noise with known rho is recovered to <10%."""

import unittest

import numpy as np

from musepipe.covariance import (
    correlation_by_lag,
    correlation_length_channels,
    n_eff_over_n,
    spatial_inflation_by_box,
)


def ar1_noise(n_rows, n_ch, rho, rng):
    """First-order autoregressive rows: corr(lag) = rho**lag."""
    x = np.zeros((n_rows, n_ch))
    x[:, 0] = rng.normal(size=n_rows)
    s = np.sqrt(1.0 - rho * rho)
    for k in range(1, n_ch):
        x[:, k] = rho * x[:, k - 1] + s * rng.normal(size=n_rows)
    return x


class CovarianceSyntheticTests(unittest.TestCase):
    def test_ar1_lag1_correlation_recovered(self):
        rng = np.random.default_rng(1)
        rho = 0.6
        noise = ar1_noise(40, 4000, rho, rng)
        est = correlation_by_lag(noise, max_lag=5)
        self.assertAlmostEqual(est[0], 1.0, places=6)
        self.assertLess(abs(est[1] - rho) / rho, 0.10)  # lag-1 within 10%
        self.assertLess(abs(est[2] - rho**2) / rho**2, 0.15)  # lag-2 ~ rho^2

    def test_white_noise_has_unit_length_and_full_neff(self):
        rng = np.random.default_rng(2)
        white = rng.normal(size=(30, 4000))
        rho = correlation_by_lag(white, max_lag=8)
        self.assertLess(correlation_length_channels(rho), 1.3)
        self.assertGreater(n_eff_over_n(rho), 0.75)

    def test_correlated_noise_reduces_neff(self):
        rng = np.random.default_rng(3)
        rho = correlation_by_lag(ar1_noise(40, 4000, 0.6, rng), max_lag=10)
        self.assertGreater(correlation_length_channels(rho), 2.0)
        self.assertLess(n_eff_over_n(rho), 0.5)

    def test_spatial_inflation_white_image_is_unity(self):
        rng = np.random.default_rng(4)
        img = rng.normal(size=(120, 120))
        factors = spatial_inflation_by_box(img, boxes=(1, 3, 5), n_samples=800, seed=0)
        self.assertEqual(factors[1], 1.0)
        for N in (3, 5):
            self.assertLess(abs(factors[N] - 1.0), 0.2)  # independent pixels ~ 1


if __name__ == "__main__":
    unittest.main()
