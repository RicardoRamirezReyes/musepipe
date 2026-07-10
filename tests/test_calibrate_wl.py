import unittest

import numpy as np

from musepipe.stages.stage_x11_calibrate import (
    CalibrationCorrections,
    apply_wavelength_correction,
    verify_wavelength_residuals,
)


class CalibrateWavelengthTests(unittest.TestCase):
    def test_constant_positive_offset_is_subtracted(self):
        true_wave = np.linspace(6500.0, 6510.0, 8)
        observed_wave = true_wave + 0.08
        corrections = CalibrationCorrections(
            wavelength_status="green",
            wavelength_apply=True,
            wavelength_offset_A=0.08,
            frame_final="barycentric",
        )

        corrected = apply_wavelength_correction(observed_wave, corrections)
        v1 = verify_wavelength_residuals([0.08, 0.10, 0.04], corrections)

        np.testing.assert_allclose(corrected, true_wave)
        self.assertTrue(v1["ok"])
        self.assertLess(v1["max_abs_residual_A"], 0.05)


if __name__ == "__main__":
    unittest.main()
