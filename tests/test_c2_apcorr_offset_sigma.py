"""V4(b) de C2: la fraccion premia al dato ruidoso, `offset_sigma` no.

`fraction_channels_agree` cuenta canales donde `|f3-f5| <= sigma_combinada`, asi
que mide el sistematico RELATIVO al ruido. Medido el 2026-09-02 sobre los tres
runs: el deficit de box5 es el mismo (~9-10 %) en los tres, y sin embargo
ROXs 42B b concuerda en el 81 % de los canales y la noche buena de ROXs 12 b solo
en el 12 %, porque su sigma es la mitad. Quien compare la fraccion entre objetos
concluye lo contrario de lo que pasa.
"""
from __future__ import annotations

import unittest

import numpy as np

from musepipe.stages.stage_x01_aperture import _apcorr_consistency


class _Prod:
    def __init__(self, wave, flux, err):
        self.wave_A = wave
        self.flux = flux
        self.flux_err_emp = err
        self.apcorr = np.full_like(flux, 5.0)


class _Extr:
    def __init__(self, p):
        self.product = p


def _par(deficit=0.10, sigma=1.0, n=1200):
    """box3 plano a 100; box5 con un deficit fijo; ruido controlado."""
    w = np.linspace(7500.0, 9000.0, n)
    f3 = np.full(n, 100.0)
    f5 = np.full(n, 100.0 * (1.0 - deficit))
    e = np.full(n, sigma)
    return {"box3": _Extr(_Prod(w, f3, e)), "box5": _Extr(_Prod(w, f5, e))}


class ApcorrOffsetSigmaTests(unittest.TestCase):
    def test_publica_las_tres_cifras(self):
        r = _apcorr_consistency(_par(), {})
        for k in ("offset_sigma", "median_abs_difference", "median_combined_sigma"):
            self.assertIn(k, r)

    def test_el_mismo_sistematico_da_fracciones_opuestas_segun_el_ruido(self):
        """El caso real: mismo deficit, sigma distinta, veredictos opuestos."""
        preciso = _apcorr_consistency(_par(deficit=0.10, sigma=3.0), {})
        ruidoso = _apcorr_consistency(_par(deficit=0.10, sigma=30.0), {})
        self.assertLess(preciso["fraction_channels_agree"], 0.5)
        self.assertGreater(ruidoso["fraction_channels_agree"], 0.9)
        self.assertFalse(preciso["ok"])
        self.assertTrue(ruidoso["ok"], "el dato ruidoso PASA con el mismo sistematico")

    def test_offset_sigma_no_se_deja_enganar_por_el_ruido(self):
        """Es la cifra que hay que comparar entre runs: crece cuando mejora el S/N."""
        preciso = _apcorr_consistency(_par(deficit=0.10, sigma=3.0), {})
        ruidoso = _apcorr_consistency(_par(deficit=0.10, sigma=30.0), {})
        self.assertGreater(preciso["offset_sigma"], ruidoso["offset_sigma"])
        self.assertAlmostEqual(preciso["offset_sigma"], 10 / (3 * np.sqrt(2)), places=6)

    def test_el_tamano_del_sistematico_no_depende_del_ruido(self):
        """`box5_over_box3_median` mide el sistematico, no su significancia."""
        a = _apcorr_consistency(_par(deficit=0.10, sigma=3.0), {})
        b = _apcorr_consistency(_par(deficit=0.10, sigma=30.0), {})
        self.assertAlmostEqual(a["box5_over_box3_median"], b["box5_over_box3_median"], places=9)
        self.assertAlmostEqual(a["box5_over_box3_median"], 0.90, places=9)

    def test_sin_box5_no_hay_comprobacion(self):
        e = _par(); del e["box5"]
        self.assertIsNone(_apcorr_consistency(e, {}))


if __name__ == "__main__":
    unittest.main()
