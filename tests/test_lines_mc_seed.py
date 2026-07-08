"""G2 §7: MC reproducibility with a seed; MC error ~ analytic for a simple gaussian."""

import unittest

import numpy as np

from musepipe.constants import fwhm_to_sigma
from musepipe.lines import measure_line


def spectrum(wave, cont, total, sig_tot, err, seed):
    amp = total / (sig_tot * np.sqrt(2 * np.pi))
    rng = np.random.default_rng(seed)
    flux = cont + amp * np.exp(-0.5 * ((wave - 6562.8) / sig_tot) ** 2) + rng.normal(0, err, wave.size)
    return flux, np.full_like(wave, err)


class LinesMcSeedTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.linspace(6400.0, 6720.0, 320)
        self.lsf = 2.5
        self.sig = np.hypot(fwhm_to_sigma(self.lsf), 1.5)

    def test_same_seed_reproduces_errors(self):
        flux, ferr = spectrum(self.wave, 200.0, 700.0, self.sig, 3.0, seed=10)
        ld = {"name": "Ha", "wave_A": 6562.8}
        a = measure_line(self.wave, flux, ferr, ld, lsf_fwhm_A=self.lsf, n_mc=200, seed=42)
        b = measure_line(self.wave, flux, ferr, ld, lsf_fwhm_A=self.lsf, n_mc=200, seed=42)
        self.assertEqual(a.flux_direct_err, b.flux_direct_err)
        self.assertEqual(a.centroid_err_A, b.centroid_err_A)

    def test_different_seed_changes_errors_slightly(self):
        flux, ferr = spectrum(self.wave, 200.0, 700.0, self.sig, 3.0, seed=11)
        ld = {"name": "Ha", "wave_A": 6562.8}
        a = measure_line(self.wave, flux, ferr, ld, lsf_fwhm_A=self.lsf, n_mc=300, seed=1)
        b = measure_line(self.wave, flux, ferr, ld, lsf_fwhm_A=self.lsf, n_mc=300, seed=2)
        self.assertNotEqual(a.flux_direct_err, b.flux_direct_err)

    def test_mc_flux_error_matches_analytic(self):
        # flat spectrum, pure noise: MC error on integrated flux ~ sqrt(sum (err*dl)^2)
        err = 3.0
        flux, ferr = spectrum(self.wave, 200.0, 0.0, self.sig, err, seed=12)
        m = measure_line(self.wave, flux, ferr, {"name": "Ha", "wave_A": 6562.8}, lsf_fwhm_A=self.lsf, n_mc=800, seed=3)
        dl = np.gradient(self.wave)
        line = np.abs(self.wave - 6562.8) <= 6.0
        analytic = np.sqrt(np.sum((err * dl[line]) ** 2))
        self.assertLess(abs(m.flux_direct_err - analytic) / analytic, 0.25)


if __name__ == "__main__":
    unittest.main()
