import unittest

import numpy as np

from musepipe.stages.stage_x11_calibrate import CalibrationCorrections, calibrate_spectrum_product
from tests.test_calibrate_no_double import calibration_product


class CalibrateBudgetTests(unittest.TestCase):
    def test_known_systematics_give_analytic_total_error(self):
        wave = np.linspace(6299.0, 6301.0, 9)
        product = calibration_product(
            wave=wave,
            flux=np.full(wave.size, 100.0),
            flux_err=np.full(wave.size, 3.0),
            flux_err_emp=np.full(wave.size, 4.0),
        )
        corrections = CalibrationCorrections(
            frame_final="barycentric",
            flux_scale=2.0,
            flux_scale_err_frac=0.10,
            psf_frac=0.05,
            sky_frac=0.02,
            sky_windows_A=((6298.0, 6302.0),),
        )

        calibrated = calibrate_spectrum_product(
            product,
            corrections,
            method="psffit",
            error_smooth_channels=1,
        )

        extra = calibrated.product.extra_columns
        expected_stat = 8.0
        expected_total = np.sqrt(expected_stat**2 + 20.0**2 + 10.0**2 + 4.0**2)
        np.testing.assert_allclose(extra["flux_err_stat"], expected_stat)
        np.testing.assert_allclose(extra["sys_fluxcal"], 20.0)
        np.testing.assert_allclose(extra["sys_psf"], 10.0)
        np.testing.assert_allclose(extra["sys_sky"], 4.0)
        np.testing.assert_allclose(extra["flux_err_total"], expected_total)
        np.testing.assert_allclose(calibrated.product.flux_err, expected_total)


if __name__ == "__main__":
    unittest.main()
