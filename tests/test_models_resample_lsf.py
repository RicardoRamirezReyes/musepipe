"""G3 §7: LSF degradation + resampling conserve integrated flux (analytic)."""

import unittest

import numpy as np

from musepipe.models.prep import degrade_to_lsf, resample_conserve_flux


class ResampleLsfTests(unittest.TestCase):
    def test_resample_conserves_integral_flat(self):
        win = np.linspace(6000.0, 7000.0, 1000)
        flux = np.full_like(win, 3.0)
        wout = np.linspace(6050.0, 6950.0, 137)
        rout = resample_conserve_flux(win, flux, wout)
        # flat density -> resampled density unchanged
        self.assertTrue(np.allclose(rout, 3.0, atol=1e-6))

    def test_resample_conserves_integral_boxcar(self):
        win = np.linspace(6000.0, 7000.0, 2001)  # 0.5 A/ch
        flux = np.where((win >= 6490) & (win <= 6510), 10.0, 1.0)
        wout = np.linspace(6010.0, 6990.0, 197)
        rout = resample_conserve_flux(win, flux, wout)
        dl_in = np.gradient(win)
        dl_out = np.gradient(wout)
        integral_in = np.sum(flux * dl_in)
        integral_out = np.nansum(rout * dl_out)
        self.assertLess(abs(integral_out - integral_in) / integral_in, 0.02)

    def test_lsf_conserves_flux_and_broadens(self):
        wave = np.linspace(6500.0, 6620.0, 480)
        line = np.exp(-0.5 * ((wave - 6560.0) / 0.8) ** 2)
        out = degrade_to_lsf(wave, line, lsf_fwhm_A=2.5)
        dl = np.gradient(wave)
        self.assertLess(abs(np.sum(out * dl) - np.sum(line * dl)) / np.sum(line * dl), 0.02)
        self.assertLess(out.max(), line.max())  # broadened -> lower peak


if __name__ == "__main__":
    unittest.main()
