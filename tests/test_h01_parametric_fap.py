"""La cola parametrica de E1: que hace, y que se niega a hacer.

El criterio de deteccion pide FAP < 0.01 y `empirical_fap` no puede bajar de
`1/(n+1)`: con los 33 controles que caben a la separacion de la companera su
suelo es 0.029, asi que el criterio es inalcanzable por construccion. Estos
tests fijan el contrato de la salida (`docs/spec_E1_v2_codex_halpha_detection.md`).
"""
from __future__ import annotations

import unittest

import numpy as np

from musepipe.stages.stage08c_look_elsewhere import empirical_fap, parametric_fap
from musepipe.stages.stage_h01_detect import FAP_ESTIMATORS, classify_h01_verdict


def _nula(n=33, seed=7):
    rng = np.random.default_rng(seed)
    # maximos de bloque: Gumbel es la familia correcta por construccion
    return rng.gumbel(loc=2.7, scale=1.1, size=n)


class ParametricFapTests(unittest.TestCase):
    def test_rompe_el_suelo_del_contador(self):
        nula = _nula()
        z = float(nula.max()) + 6.0
        suelo = 1.0 / (nula.size + 1)
        self.assertAlmostEqual(empirical_fap(z, nula), suelo)
        p = parametric_fap(z, nula)
        self.assertLess(p["fap"], suelo,
                        "la cola tiene que poder bajar del suelo del contador")

    def test_publica_el_diagnostico_con_el_numero(self):
        p = parametric_fap(float(_nula().max()) + 4.0, _nula())
        for clave in ("fap", "family", "n_null", "loc", "scale", "ks_p",
                      "extrapolation_sd", "ci95", "reason"):
            self.assertIn(clave, p)
        self.assertGreater(p["ks_p"], 0.01, "el ajuste deberia describir su propia familia")
        self.assertGreater(p["extrapolation_sd"], 0.0)

    def test_nula_degenerada_no_inventa_un_numero(self):
        p = parametric_fap(10.0, np.full(33, 2.0))
        self.assertTrue(np.isnan(p["fap"]))
        self.assertEqual(p["reason"], "degenerate_null_zero_spread")

    def test_muestra_corta_no_se_ajusta(self):
        p = parametric_fap(10.0, _nula(n=5))
        self.assertTrue(np.isnan(p["fap"]))
        self.assertEqual(p["reason"], "insufficient_null_sample")

    def test_familia_desconocida_es_error(self):
        with self.assertRaises(ValueError):
            parametric_fap(10.0, _nula(), family="lognormal")

    def test_bootstrap_devuelve_intervalo_que_contiene_el_valor(self):
        nula = _nula()
        z = float(nula.max()) + 4.0
        p = parametric_fap(z, nula, n_boot=300, seed=3)
        self.assertIsNotNone(p["ci95"])
        lo, hi = p["ci95"]
        self.assertLessEqual(lo, p["fap"])
        self.assertLessEqual(p["fap"], hi)


class VerdictEstimatorTests(unittest.TestCase):
    """El estimador que manda cambia el veredicto, y por eso se declara."""

    def _filas(self):
        # las dos admisibles pasan por la cola pero NO por el contador
        return [
            {"method": m, "rv_consistent": True,
             "global_empirical_fap": 0.029, "global_parametric_fap": 1e-6}
            for m in ("psffit", "aperture")
        ]

    def test_empirico_no_detecta_por_el_suelo(self):
        v = classify_h01_verdict(self._filas(), fap_estimator="empirical")
        self.assertEqual(v["verdict"], "non_detection")
        self.assertEqual(v["fap_estimator"], "empirical")

    def test_parametrico_si_detecta_con_los_mismos_datos(self):
        v = classify_h01_verdict(self._filas(), fap_estimator="parametric")
        self.assertEqual(v["verdict"], "detection")
        self.assertEqual(v["fap_estimator"], "parametric")

    def test_estimador_desconocido_es_error(self):
        with self.assertRaises(ValueError):
            classify_h01_verdict(self._filas(), fap_estimator="bayesiano")

    def test_las_columnas_declaradas_existen(self):
        self.assertEqual(set(FAP_ESTIMATORS), {"empirical", "parametric"})


if __name__ == "__main__":
    unittest.main()
