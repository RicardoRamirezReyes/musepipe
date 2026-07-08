"""G2 §4/§7: edge cases — line at range edge, bad_ranges, NaN window, fit failure."""

import unittest

import numpy as np

from musepipe.lines import measure_line


class LinesEdgeTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.linspace(6400.0, 6720.0, 320)
        self.flux = np.full_like(self.wave, 100.0)
        self.ferr = np.full_like(self.wave, 2.0)

    def test_line_at_range_edge_is_not_measurable(self):
        # rest wavelength beyond the blue edge: no blue continuum window
        m = measure_line(self.wave, self.flux, self.ferr, {"name": "edge", "wave_A": 6402.0},
                         lsf_fwhm_A=2.5, n_mc=0, min_continuum_pixels=6)
        self.assertEqual(m.status, "not_measurable")
        self.assertIn("edge", m.reason)

    def test_line_inside_bad_range_loses_continuum(self):
        m = measure_line(self.wave, self.flux, self.ferr, {"name": "x", "wave_A": 6562.8},
                         lsf_fwhm_A=2.5, n_mc=0, bad_ranges=[(6480.0, 6640.0)], min_continuum_pixels=6)
        self.assertEqual(m.status, "not_measurable")

    def test_nan_in_window_sets_flag_but_measures(self):
        flux = self.flux.copy()
        # inject an emission and NaN ~half the line window
        sig = 1.8
        flux += 400.0 / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((self.wave - 6562.8) / sig) ** 2)
        line_win = np.abs(self.wave - 6562.8) <= 6.0
        idx = np.where(line_win)[0]
        flux[idx[: len(idx) // 2 + 2]] = np.nan
        m = measure_line(self.wave, flux, self.ferr, {"name": "x", "wave_A": 6562.8}, lsf_fwhm_A=2.5, n_mc=0)
        self.assertIn("nan_window", m.flags)
        self.assertNotEqual(m.status, "not_measurable")


if __name__ == "__main__":
    unittest.main()
