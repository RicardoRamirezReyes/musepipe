"""H04 recovery mask must match stage_h01's BAD_DETECTION_FLAGS, not `flags == 0`.

FLAG_CLIPPED / FLAG_INTERPOLATED are informational and NOT disqualifying in E1.
The optimal extractors set FLAG_CLIPPED on most channels for a messy (binary-
primary) halo; a `flags == 0` recovery mask would reject every channel and yield
null throughput. This locks in that clipped channels still recover, while a
skyline/bad-window flag on the line does reject it (as in h01).
"""

import unittest

import numpy as np

from musepipe.extraction.aperture import (
    FLAG_BAD_WINDOW,
    FLAG_CLIPPED,
    FLAG_INTERPOLATED,
    FLAG_SKYLINE,
)
from musepipe.stages.stage_h04_injection import measure_recovery_with_h01_estimator


def _line_product(flags_value_in_window):
    wave = np.arange(6500.0, 6620.0, 1.25)
    center = 6562.8
    fwhm = 2.3
    sigma = fwhm / 2.3548
    flux = 100.0 * np.exp(-0.5 * ((wave - center) / sigma) ** 2)  # a clear emission line
    flux_err = np.full(wave.size, 5.0)
    flags = np.zeros(wave.size, dtype=np.int32)
    window = (wave > center - 6) & (wave < center + 6)
    flags[window] = flags_value_in_window
    return {"wave_A": wave, "flux": flux, "flux_err": flux_err, "flags": flags}, center, fwhm


class H04RecoveryFlagMaskTests(unittest.TestCase):
    def test_clipped_channels_still_recover(self):
        prod, center, fwhm = _line_product(FLAG_CLIPPED | FLAG_INTERPOLATED)
        meas = measure_recovery_with_h01_estimator(prod, line_center_A=center, line_fwhm_A=fwhm)
        self.assertTrue(np.isfinite(meas["recovered_flux"]))
        self.assertTrue(np.isfinite(meas["recovered_snr"]))
        self.assertGreater(meas["recovered_snr"], 3.0)

    def test_skyline_flag_on_line_masks_the_peak(self):
        # A skyline/bad-window flag DOES disqualify the line channels (unlike
        # clipped), so the matched filter loses the peak and recovers far less.
        clipped, center, fwhm = _line_product(FLAG_CLIPPED)
        skyline, _, _ = _line_product(FLAG_SKYLINE | FLAG_BAD_WINDOW)
        snr_clipped = measure_recovery_with_h01_estimator(
            clipped, line_center_A=center, line_fwhm_A=fwhm)["recovered_snr"]
        snr_skyline = measure_recovery_with_h01_estimator(
            skyline, line_center_A=center, line_fwhm_A=fwhm)["recovered_snr"]
        self.assertGreater(snr_clipped, 3.0)
        self.assertLess(snr_skyline, 0.5 * snr_clipped)


if __name__ == "__main__":
    unittest.main()
