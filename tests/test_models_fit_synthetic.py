"""G3 §7 V1: recover Teff/A_V/scale from a synthetic spectrum (mini 3x3 grid)."""

import unittest

import numpy as np

from musepipe.models.extinction import CCMExtinction
from musepipe.models.fit import fit_grid
from musepipe.models.prep import prepare_template
from musepipe.models.synthetic import SyntheticGridLibrary


class FitSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.lib = SyntheticGridLibrary(teff_axis=(2600.0, 2800.0, 3000.0))
        self.law = CCMExtinction(rv=3.1, citation="Cardelli+1989")
        self.wave = np.linspace(6100.0, 8900.0, 400)

    def _make(self, teff, av, scale, err=0.02, seed=0):
        t = self.lib.get(teff=teff)
        model = prepare_template(t, self.wave, lsf_fwhm_A=2.5, extinction=self.law, av=av, scale=scale)
        rng = np.random.default_rng(seed)
        return model + rng.normal(0.0, err, self.wave.size), np.full_like(self.wave, err)

    def test_recovers_grid_point(self):
        flux, ferr = self._make(2800.0, 1.0, 5.0, err=0.02, seed=1)
        out = fit_grid(self.wave, flux, ferr, self.lib, self.law,
                       teff_axis=(2600.0, 2800.0, 3000.0), av_axis=(0.0, 1.0, 2.0), lsf_fwhm_A=2.5)
        self.assertEqual(out["teff_best"], 2800.0)
        self.assertEqual(out["av_best"], 1.0)
        self.assertAlmostEqual(out["scale_best"], 5.0, delta=0.5)
        self.assertLess(out["chi2_red"], 3.0)

    def test_flat_grid_reports_not_constrained(self):
        # pure noise: no template preferred -> Teff not constrained
        rng = np.random.default_rng(2)
        flux = rng.normal(0.0, 1.0, self.wave.size)
        ferr = np.full_like(self.wave, 1.0)
        out = fit_grid(self.wave, flux, ferr, self.lib, self.law,
                       teff_axis=(2600.0, 2800.0, 3000.0), av_axis=(0.0, 1.0, 2.0),
                       lsf_fwhm_A=2.5, delta_chi2_confidence=1.0)
        # with only noise the delta-chi2 spread is tiny -> at least one axis unconstrained
        self.assertTrue(out["teff_interval"] == "not_constrained" or out["av_interval"] == "not_constrained")

    def test_line_mask_excludes_channels(self):
        flux, ferr = self._make(3000.0, 0.0, 4.0, err=0.02, seed=3)
        line_mask = np.abs(self.wave - 6562.8) <= 5.0
        out = fit_grid(self.wave, flux, ferr, self.lib, self.law,
                       teff_axis=(2600.0, 2800.0, 3000.0), av_axis=(0.0, 1.0, 2.0),
                       lsf_fwhm_A=2.5, line_mask=line_mask)
        self.assertEqual(out["teff_best"], 3000.0)


if __name__ == "__main__":
    unittest.main()
