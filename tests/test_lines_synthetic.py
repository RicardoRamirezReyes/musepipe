"""G2 §6 V1: fabricated spectrum with known line params → measured within tolerance."""

import unittest

import numpy as np

from musepipe.constants import fwhm_to_sigma, sigma_to_fwhm
from musepipe.lines import measure_line


def emission(wave, center, total_flux, sigma_tot):
    amp = total_flux / (sigma_tot * np.sqrt(2.0 * np.pi))
    return amp * np.exp(-0.5 * ((wave - center) / sigma_tot) ** 2)


class LinesSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.linspace(6400.0, 6720.0, 320)
        self.lsf_fwhm = 2.5
        self.sig_lsf = fwhm_to_sigma(self.lsf_fwhm)
        self.cont = 200.0
        self.err = 2.0

    def _spectrum(self, center, total_flux, sig_int, seed=0):
        sig_tot = np.hypot(self.sig_lsf, sig_int)
        rng = np.random.default_rng(seed)
        flux = self.cont + emission(self.wave, center, total_flux, sig_tot)
        flux = flux + rng.normal(0.0, self.err, self.wave.size)
        return flux, np.full_like(self.wave, self.err)

    def test_strong_emission_recovered_within_tolerance(self):
        flux, ferr = self._spectrum(6562.8, 800.0, 1.5, seed=1)
        m = measure_line(self.wave, flux, ferr, {"name": "Ha", "wave_A": 6562.8}, lsf_fwhm_A=self.lsf_fwhm, n_mc=300, seed=2)
        self.assertEqual(m.status, "detected")
        self.assertEqual(m.label, "direct_measurement")
        self.assertLess(abs(m.flux_direct - 800.0), 4.0 * m.flux_direct_err)
        self.assertLess(abs(m.continuum_density - self.cont), 3.0)
        self.assertAlmostEqual(m.ew_A, -800.0 / self.cont, delta=0.5)  # emission -> negative EW

    def test_unresolved_line_reports_zero_intrinsic_fwhm(self):
        flux, ferr = self._spectrum(6562.8, 600.0, 0.0, seed=3)  # intrinsic = 0 (only LSF)
        m = measure_line(self.wave, flux, ferr, {"name": "Ha", "wave_A": 6562.8}, lsf_fwhm_A=self.lsf_fwhm, n_mc=200, seed=4)
        self.assertLess(m.fwhm_intrinsic_A, 1.0)  # ~unresolved

    def test_resolved_line_recovers_intrinsic_fwhm(self):
        flux, ferr = self._spectrum(6562.8, 1200.0, 3.0, seed=5)
        m = measure_line(self.wave, flux, ferr, {"name": "Ha", "wave_A": 6562.8}, lsf_fwhm_A=self.lsf_fwhm, n_mc=100, seed=6)
        self.assertAlmostEqual(m.fwhm_intrinsic_A, sigma_to_fwhm(3.0), delta=1.0)

    def test_line_below_noise_is_upper_limit(self):
        flux, ferr = self._spectrum(6562.8, 5.0, 1.0, seed=7)  # negligible line
        m = measure_line(self.wave, flux, ferr, {"name": "Ha", "wave_A": 6562.8}, lsf_fwhm_A=self.lsf_fwhm, n_mc=100, seed=8)
        self.assertIn(m.status, ("upper_limit", "marginal"))
        self.assertGreater(m.flux_upper_limit_5sigma, 0.0)


if __name__ == "__main__":
    unittest.main()
