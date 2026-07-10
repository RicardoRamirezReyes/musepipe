import unittest

import numpy as np

from musepipe.stages.stage_x11_calibrate import conservative_stat_error


class CalibrateMaxRuleTests(unittest.TestCase):
    def test_channelwise_max_then_median_smoothing(self):
        flux_err = np.array([1.0, 10.0, 1.0, 10.0, 1.0])
        flux_err_emp = np.array([5.0, 2.0, 5.0, 2.0, 5.0])

        smoothed = conservative_stat_error(flux_err, flux_err_emp, smooth_channels=3)

        np.testing.assert_allclose(smoothed, np.array([7.5, 5.0, 10.0, 5.0, 7.5]))


if __name__ == "__main__":
    unittest.main()
