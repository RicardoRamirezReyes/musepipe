import math
import unittest

import numpy as np

from musepipe.stats import (
    empirical_z,
    finite_percentile,
    finite_values,
    median_finite,
    robust_limits,
    robust_sigma,
    robust_sigma_axis0,
)


class StatsTests(unittest.TestCase):
    def test_finite_values_filters_nan_and_inf(self):
        values = finite_values([1.0, np.nan, np.inf, -np.inf, 2.5])
        np.testing.assert_allclose(values, [1.0, 2.5])

    def test_robust_sigma_uses_mad_with_std_fallback(self):
        self.assertAlmostEqual(robust_sigma([1.0, 1.0, 2.0, 100.0, np.nan]), 0.7413, places=4)
        self.assertEqual(robust_sigma([5.0, 5.0, 5.0]), 0.0)
        self.assertTrue(math.isnan(robust_sigma([np.nan, np.inf])))

    def test_robust_sigma_axis0_falls_back_per_column(self):
        values = np.array(
            [
                [1.0, 10.0],
                [1.0, 12.0],
                [2.0, 14.0],
            ]
        )
        sigma = robust_sigma_axis0(values)
        np.testing.assert_allclose(sigma, [np.std([1.0, 1.0, 2.0]), 2.9652], rtol=1e-6)

    def test_finite_summary_helpers(self):
        values = [1.0, 3.0, np.nan, 5.0]
        self.assertEqual(finite_percentile(values, 50), 3.0)
        self.assertEqual(median_finite(values), 3.0)
        self.assertTrue(math.isnan(finite_percentile([np.nan], 50)))
        self.assertTrue(math.isnan(median_finite([np.nan])))

    def test_empirical_z_and_robust_limits(self):
        refs = [1.0, 1.0, 2.0, 3.0]
        self.assertAlmostEqual(empirical_z(3.0, refs), (3.0 - 1.5) / 0.7413, places=4)
        self.assertTrue(math.isnan(empirical_z(3.0, [1.0, 1.0, 1.0])))

        lo, hi = robust_limits([0.0, 1.0, 2.0, 3.0, np.nan], p_lo=0, p_hi=100)
        self.assertEqual((lo, hi), (0.0, 3.0))
        lo, hi = robust_limits([-2.0, -1.0, 3.0], p_hi=100, symmetric=True)
        self.assertEqual((lo, hi), (-3.0, 3.0))


if __name__ == "__main__":
    unittest.main()
