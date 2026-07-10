import unittest

import numpy as np

from musepipe.stripes import apply_stripe_spectral_shifts, shift_spectral_variance


class Stage02StatShiftTests(unittest.TestCase):
    def test_integer_stat_channel_moves_with_data_stripe(self):
        data = np.zeros((8, 4, 6), dtype=np.float32)
        stat = np.zeros_like(data)
        data[2, :, :3] = 10.0
        stat[2, :, :3] = 99.0

        shifted_data, data_kernel = apply_stripe_spectral_shifts(
            data,
            [2.0, 0.0],
            nstripes=2,
            stripe_orientation="vertical",
            apply_mode="integer",
        )
        shifted_stat, stat_kernel = apply_stripe_spectral_shifts(
            stat,
            [2.0, 0.0],
            nstripes=2,
            stripe_orientation="vertical",
            apply_mode="integer",
            is_variance=True,
        )

        self.assertEqual(data_kernel, "integer,none")
        self.assertEqual(stat_kernel, "integer,none")
        self.assertTrue(np.allclose(shifted_data[4, :, :3], 10.0))
        self.assertTrue(np.allclose(shifted_stat[4, :, :3], 99.0))
        self.assertTrue(np.all(np.isnan(shifted_data[:2, :, :3])))
        self.assertTrue(np.allclose(shifted_data[2, :, 3:], 0.0))

    def test_fractional_variance_shift_uses_squared_linear_weights(self):
        var = np.zeros((6, 1, 1), dtype=np.float32)
        var[2, 0, 0] = 16.0

        shifted, kernel = shift_spectral_variance(var, 0.5, apply_mode="subpixel")

        self.assertEqual(kernel, "linear_kernel_squared")
        self.assertAlmostEqual(float(shifted[2, 0, 0]), 4.0)
        self.assertAlmostEqual(float(shifted[3, 0, 0]), 4.0)


if __name__ == "__main__":
    unittest.main()
