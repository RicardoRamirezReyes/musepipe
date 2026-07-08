"""G2 §7: intrinsic FWHM deconvolution — resolved and unresolved analytic cases."""

import unittest

import numpy as np

from musepipe.constants import fwhm_to_sigma, sigma_to_fwhm
from musepipe.lines import measure_line


def make(wave, total, sig_int, sig_lsf, err=1.0, cont=200.0, seed=0):
    sig_tot = np.hypot(sig_int, sig_lsf)
    amp = total / (sig_tot * np.sqrt(2 * np.pi))
    rng = np.random.default_rng(seed)
    flux = cont + amp * np.exp(-0.5 * ((wave - 6562.8) / sig_tot) ** 2) + rng.normal(0, err, wave.size)
    return flux, np.full_like(wave, err)


class LinesLsfDeconvTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.linspace(6400.0, 6720.0, 640)  # 0.5 A/ch, well sampled
        self.lsf = 2.5
        self.sig_lsf = fwhm_to_sigma(self.lsf)

    def test_unresolved_line_intrinsic_fwhm_near_zero(self):
        # intrinsic width << LSF -> deconvolved intrinsic FWHM ~ 0 (unresolved)
        flux, ferr = make(self.wave, 1500.0, 0.1, self.sig_lsf, err=1.0, seed=1)
        m = measure_line(self.wave, flux, ferr, {"name": "x", "wave_A": 6562.8}, lsf_fwhm_A=self.lsf, n_mc=0)
        self.assertLess(m.fwhm_intrinsic_A, 0.8)

    def test_resolved_line_intrinsic_fwhm_recovered(self):
        sig_int = 4.0
        flux, ferr = make(self.wave, 3000.0, sig_int, self.sig_lsf, err=1.0, seed=2)
        m = measure_line(self.wave, flux, ferr, {"name": "x", "wave_A": 6562.8}, lsf_fwhm_A=self.lsf, n_mc=0)
        self.assertAlmostEqual(m.fwhm_intrinsic_A, sigma_to_fwhm(sig_int), delta=0.8)

    def test_intrinsic_is_deconvolved_below_total_width(self):
        # deconvolving the LSF must leave intrinsic FWHM below the total observed
        # gaussian width sqrt(sig_int^2 + sig_lsf^2); the fit recovers sig_int.
        sig_int = 4.0
        flux, ferr = make(self.wave, 3000.0, sig_int, self.sig_lsf, err=1.0, seed=3)
        m = measure_line(self.wave, flux, ferr, {"name": "x", "wave_A": 6562.8}, lsf_fwhm_A=self.lsf, n_mc=0)
        total_fwhm = sigma_to_fwhm(np.hypot(sig_int, self.sig_lsf))
        self.assertTrue(np.isfinite(m.fwhm_obs_A) and m.fwhm_obs_A > 0)
        self.assertLess(m.fwhm_intrinsic_A, total_fwhm)
        self.assertGreater(m.fwhm_intrinsic_A, 0.0)


if __name__ == "__main__":
    unittest.main()
