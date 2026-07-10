import unittest

import numpy as np

from musepipe.qc.cube_qc import Skyline, fit_wavelength_offsets, measure_skylines


def _gaussian(wave, center, sigma, amp):
    return amp * np.exp(-0.5 * ((wave - center) / sigma) ** 2)


class CubeQcM1Tests(unittest.TestCase):
    def test_wavelength_offset_plus_005A_is_recovered(self):
        wave = np.arange(5500.0, 6400.0, 0.1)
        lines = [Skyline("OI_5577", 5577.338), Skyline("OI_6300", 6300.304), Skyline("OI_6363", 6363.776)]
        spec = np.ones_like(wave)
        for line in lines:
            spec += _gaussian(wave, line.wave_A + 0.05, 0.55, 100.0)
        measurements = measure_skylines(wave, spec, lines, min_snr=1.0, half_width_A=3.0)
        fit = fit_wavelength_offsets(measurements)
        self.assertEqual(fit["n_lines"], 3)
        self.assertAlmostEqual(fit["offset_median_A"], 0.05, places=2)
        self.assertEqual(fit["status"], "green")


if __name__ == "__main__":
    unittest.main()
