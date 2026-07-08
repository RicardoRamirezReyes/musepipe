"""G3 §7: the 3 adapter protocols work with synthetic mini-libraries."""

import unittest

import numpy as np

from musepipe.models import (
    EvolutionaryModel,
    ExtinctionLaw,
    SpectralLibrary,
    TemplateSpectrum,
)
from musepipe.models.extinction import CCMExtinction
from musepipe.models.synthetic import SyntheticGridLibrary, SyntheticTracks


class AdapterContractTests(unittest.TestCase):
    def test_spectral_library_protocol(self):
        lib = SyntheticGridLibrary()
        self.assertIsInstance(lib, SpectralLibrary)
        axes = lib.grid()
        self.assertIn("teff", axes)
        t = lib.get(teff=axes["teff"][0])
        self.assertIsInstance(t, TemplateSpectrum)
        self.assertEqual(t.wave_A.shape, t.flux.shape)

    def test_extinction_protocol(self):
        law = CCMExtinction(rv=3.1, citation="Cardelli+1989")
        self.assertIsInstance(law, ExtinctionLaw)
        a = law.a_lambda_over_av(np.array([6562.8, 8446.0]))
        self.assertEqual(np.asarray(a).shape, (2,))

    def test_evolutionary_protocol(self):
        tr = SyntheticTracks()
        self.assertIsInstance(tr, EvolutionaryModel)
        out = tr.lookup(1e-3, 5.0)
        for key in ("mass_msun", "radius_rsun", "logg"):
            self.assertIn(key, out)

    def test_library_axes_monotone_effect(self):
        lib = SyntheticGridLibrary()
        cool = lib.get(teff=2600.0).flux
        warm = lib.get(teff=3000.0).flux
        self.assertGreater(np.nanmedian(warm), np.nanmedian(cool))  # warmer -> brighter


if __name__ == "__main__":
    unittest.main()
