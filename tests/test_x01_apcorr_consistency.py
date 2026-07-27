"""V4(b) de C2: ¿dan lo mismo box3 y box5 tras la corrección de apertura?

La spec pide dos cosas en V4 y el código solo implementaba la primera (que la
magnitud de `apcorr` esté en un rango). La segunda es la que prueba algo
físico: dos aperturas distintas, cada una con SU corrección, tienen que dar el
mismo flujo total; si no, el que falla es el modelo de PSF de C1, no la
fotometría.

El rango, además, nunca podía pasar en NFM: una box3 recoge ~2% de la PSF, así
que `apcorr` vale decenas frente al ≤1.8 que exige el código.
"""
import unittest
from types import SimpleNamespace

import numpy as np

from musepipe.stages.stage_x01_aperture import (
    APCORR_CONSISTENCY_BAND_A,
    _apcorr_consistency,
)
from tests.test_calibrate_no_double import calibration_product


def _extraction(flux, *, wave, err=1.0):
    product = calibration_product(
        method="aperture", wave=wave,
        flux=np.asarray(flux, dtype=np.float64),
        flux_err_emp=np.full(np.size(wave), float(err)),
    )
    return SimpleNamespace(product=product)


def _pair(f3, f5, *, err=1.0, n=200):
    wave = np.linspace(APCORR_CONSISTENCY_BAND_A[0], APCORR_CONSISTENCY_BAND_A[1], n)
    return {
        "box3": _extraction(np.full(n, f3) if np.isscalar(f3) else f3, wave=wave, err=err),
        "box5": _extraction(np.full(n, f5) if np.isscalar(f5) else f5, wave=wave, err=err),
    }


class ApcorrConsistencyTests(unittest.TestCase):
    def test_two_apertures_that_agree_within_errors_pass(self):
        out = _apcorr_consistency(_pair(100.0, 100.5, err=10.0), {})
        self.assertTrue(out["ok"])
        self.assertAlmostEqual(out["box5_over_box3_median"], 1.005, places=6)
        self.assertEqual(out["fraction_channels_agree"], 1.0)

    def test_a_systematic_offset_larger_than_the_errors_fails(self):
        # Es el caso real de ROXs 42B b: box5 da un 40% menos que box3 y eso no
        # lo explica el ruido.
        out = _apcorr_consistency(_pair(100.0, 60.0, err=1.0), {})
        self.assertFalse(out["ok"])
        self.assertAlmostEqual(out["box5_over_box3_median"], 0.6, places=6)
        self.assertEqual(out["fraction_channels_agree"], 0.0)

    def test_the_verdict_is_about_the_error_bars_not_the_ratio(self):
        """La misma discrepancia con errores grandes es compatible con cero."""
        estrecho = _apcorr_consistency(_pair(100.0, 60.0, err=1.0), {})
        ancho = _apcorr_consistency(_pair(100.0, 60.0, err=100.0), {})
        self.assertFalse(estrecho["ok"])
        self.assertTrue(ancho["ok"])
        self.assertAlmostEqual(estrecho["box5_over_box3_median"],
                               ancho["box5_over_box3_median"], places=9)

    def test_only_the_band_where_the_companion_is_detected_counts(self):
        # Fuera de la banda las dos aperturas miden ruido y "concuerdan" siempre:
        # incluir esos canales diluiría la discrepancia real.
        n = 200
        wave = np.linspace(4800.0, 9300.0, n)
        dentro = (wave >= APCORR_CONSISTENCY_BAND_A[0]) & (wave <= APCORR_CONSISTENCY_BAND_A[1])
        f3 = np.full(n, 100.0)
        f5 = np.where(dentro, 60.0, 100.0)
        ext = {"box3": _extraction(f3, wave=wave), "box5": _extraction(f5, wave=wave)}
        out = _apcorr_consistency(ext, {})
        self.assertEqual(out["n_channels"], int(dentro.sum()))
        self.assertEqual(out["fraction_channels_agree"], 0.0)
        self.assertFalse(out["ok"])

    def test_the_band_and_the_threshold_are_configurable(self):
        ext = _pair(100.0, 60.0, err=1.0)
        self.assertTrue(_apcorr_consistency(ext, {"x01_apcorr_consistency_threshold": 0.0})["ok"])
        out = _apcorr_consistency(ext, {"x01_apcorr_consistency_band_A": [7500.0, 8000.0]})
        self.assertEqual(out["band_A"], [7500.0, 8000.0])
        self.assertLess(out["n_channels"], 200)

    def test_without_box5_there_is_nothing_to_compare(self):
        ext = _pair(100.0, 100.0)
        del ext["box5"]
        self.assertIsNone(_apcorr_consistency(ext, {}))

    def test_the_note_points_at_c1_not_at_the_photometry(self):
        out = _apcorr_consistency(_pair(100.0, 100.0), {})
        self.assertIn("C1", out["note"])
        self.assertIn("V4(b)", out["note"])


if __name__ == "__main__":
    unittest.main()
