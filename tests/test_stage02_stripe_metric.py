import unittest

import numpy as np

from musepipe.stripes import stripe_channel_amplitudes, stripe_metric_summary


class Stage02StripeMetricTests(unittest.TestCase):
    def test_metric_recovers_synthetic_relative_stripe_amplitude(self):
        nz, ny, nx = 6, 10, 12
        amplitude = 0.12
        half_swing = amplitude / 1.4826
        pattern = np.r_[
            np.full(nx // 2, -half_swing),
            np.full(nx - nx // 2, half_swing),
        ]
        cube = 100.0 * (1.0 + pattern[None, None, :])
        cube = np.broadcast_to(cube, (nz, ny, nx)).astype(np.float32)

        metric = stripe_channel_amplitudes(
            cube,
            nstripes=2,
            stripe_orientation="vertical",
            normalization="median_abs",
        )

        self.assertTrue(np.all(np.isfinite(metric)))
        self.assertAlmostEqual(float(np.nanmedian(metric)), amplitude, delta=1e-4)

    def test_metric_is_zero_for_flat_stripe_free_cube(self):
        cube = np.full((5, 8, 8), 100.0, dtype=np.float32)
        metric = stripe_channel_amplitudes(
            cube,
            nstripes=2,
            stripe_orientation="vertical",
            normalization="median_abs",
        )
        self.assertLess(float(np.nanmax(metric)), 1e-10)

    def test_summary_ignores_excluded_windows_and_marks_dirty_channels(self):
        wave = np.array([5000.0, 5790.0, 6100.0, 6200.0])
        pre = np.array([0.2, 99.0, 0.2, 0.2])
        post = np.array([0.1, 99.0, 0.1, 0.5])
        summary = stripe_metric_summary(
            pre,
            post,
            wave,
            excluded_windows_A=((5780.0, 6050.0),),
        )

        self.assertAlmostEqual(summary["reduction_factor"], 2.0)
        self.assertEqual(summary["dirty_channels"], [3])


if __name__ == "__main__":
    unittest.main()
