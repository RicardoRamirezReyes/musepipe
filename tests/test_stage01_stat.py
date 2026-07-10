import unittest

import numpy as np

from musepipe.stages.stage01_align import bilinear_shift_variance, propagate_stat_alignment


class Stage01StatTests(unittest.TestCase):
    def test_fractional_shift_uses_squared_bilinear_weights(self):
        var = np.ones((1, 4, 4), dtype=float)
        shifted = bilinear_shift_variance(var, 0.5, 0.5)
        # Interior pixels receive four weights of 0.25, so variance is
        # 4 * 0.25^2 = 0.25. A wrong DATA-like interpolation would return 1.
        self.assertAlmostEqual(float(shifted[0, 2, 2]), 0.25)

    def test_integer_shift_preserves_variance_values(self):
        var = np.arange(9, dtype=float).reshape(1, 3, 3)
        shifted, kernel = propagate_stat_alignment(var, 1.0, -1.0, "integer")
        self.assertEqual(kernel, "integer_roll")
        self.assertEqual(shifted.shape, var.shape)
        self.assertCountEqual(shifted.ravel().tolist(), var.ravel().tolist())


if __name__ == "__main__":
    unittest.main()
