"""G2 §7: EW sign convention (emission negative, absorption positive) + continuum<=0."""

import unittest

import numpy as np

from musepipe.constants import fwhm_to_sigma
from musepipe.lines import measure_line


class LinesEwSignTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.linspace(6400.0, 6720.0, 320)
        self.lsf = 2.5
        self.sig = np.hypot(fwhm_to_sigma(self.lsf), 1.5)

    def _line(self, cont, amp_total):
        amp = amp_total / (self.sig * np.sqrt(2.0 * np.pi))
        flux = cont + amp * np.exp(-0.5 * ((self.wave - 6562.8) / self.sig) ** 2)
        return flux, np.full_like(self.wave, 2.0)

    def test_emission_has_negative_ew(self):
        flux, ferr = self._line(300.0, +600.0)
        m = measure_line(self.wave, flux, ferr, {"name": "Ha", "wave_A": 6562.8}, lsf_fwhm_A=self.lsf, n_mc=0)
        self.assertLess(m.ew_A, 0.0)

    def test_absorption_has_positive_ew(self):
        flux, ferr = self._line(300.0, -600.0)
        m = measure_line(self.wave, flux, ferr, {"name": "Ha", "wave_A": 6562.8}, lsf_fwhm_A=self.lsf, n_mc=0)
        self.assertGreater(m.ew_A, 0.0)

    def test_nonpositive_continuum_makes_ew_nan(self):
        flux, ferr = self._line(-50.0, +600.0)  # negative continuum (residual)
        m = measure_line(self.wave, flux, ferr, {"name": "Ha", "wave_A": 6562.8}, lsf_fwhm_A=self.lsf, n_mc=0)
        self.assertTrue(np.isnan(m.ew_A))
        self.assertIn("continuum_nonpositive", m.flags)


if __name__ == "__main__":
    unittest.main()
