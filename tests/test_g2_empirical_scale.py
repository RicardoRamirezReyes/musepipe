"""G2 etiqueta contra los controles, no contra el error propagado (spec G2 v2).

El error propagado canal a canal describe el presupuesto de UN espectro; no
describe cuánto dispersa la medida cuando se repite a la misma separación de una
estrella brillante. En `ROXs12b_realigned` esa diferencia era ×14 en Hα y
convertía una no detección en `detected`, con E1 diciendo lo contrario sobre los
mismos datos.

Lo que se fija aquí: que la escala salga de los controles, que el suelo en 1
impida fabricar significancia, que sin controles se declare en vez de callar, y
que objeto y controles midan con las MISMAS ventanas.
"""
from __future__ import annotations

import unittest

import numpy as np

from musepipe.lines import (
    Z_GAUSS_99,
    build_windows,
    empirical_scale,
    measure_catalog,
    measure_line,
    null_z_from_controls,
)

LINEA = {"name": "Halpha", "wave_A": 6562.8, "family": "Balmer", "kind": "accretion"}


def _malla(n=1200, paso=1.25):
    return 6562.8 + (np.arange(n) - n // 2) * paso


def _gauss(wave, centro, fwhm, amp):
    sig = fwhm / 2.3548
    return amp * np.exp(-0.5 * ((wave - centro) / sig) ** 2)


def _controles(wave, n, escala, semilla=0):
    """N controles de ruido blanco con la desviación pedida."""
    rng = np.random.default_rng(semilla)
    return rng.normal(0.0, escala, size=(n, wave.size))


class EscalaEmpiricaTests(unittest.TestCase):
    def setUp(self):
        self.wave = _malla()
        self.ferr = np.full(self.wave.size, 1.0)
        self.windows = build_windows(self.wave, LINEA)

    def test_controls_as_noisy_as_the_budget_give_inflation_about_one(self):
        """Si los controles dispersan lo que dice el error, no hay que corregir."""
        nulls = null_z_from_controls(self.wave, _controles(self.wave, 200, 1.0),
                                     self.ferr, center_A=6562.8, lsf_fwhm_A=2.4,
                                     windows=self.windows)
        esc = empirical_scale(nulls)
        self.assertAlmostEqual(esc["sigma_inflation"], 1.0, delta=0.45)

    def test_controls_ten_times_noisier_are_measured_as_such(self):
        """El factor se MIDE: controles ×10 -> inflación ~10."""
        nulls = null_z_from_controls(self.wave, _controles(self.wave, 200, 10.0),
                                     self.ferr, center_A=6562.8, lsf_fwhm_A=2.4,
                                     windows=self.windows)
        esc = empirical_scale(nulls)
        self.assertGreater(esc["sigma_inflation"], 6.0)
        self.assertLess(esc["sigma_inflation"], 15.0)

    def test_a_line_significant_only_against_the_budget_is_not_detected(self):
        """El caso que motivó la v2, en pequeño.

        Una línea a 12 sigmas del error propagado, con controles que dispersan
        ×10: contra los controles no destaca y NO puede salir `detected`.
        """
        flujo = _gauss(self.wave, 6562.8, 2.4, 12.0)
        nulls = null_z_from_controls(self.wave, _controles(self.wave, 200, 10.0),
                                     self.ferr, center_A=6562.8, lsf_fwhm_A=2.4,
                                     windows=self.windows)
        sin_escala = measure_line(self.wave, flujo, self.ferr, LINEA, lsf_fwhm_A=2.4,
                                  n_mc=0, windows=self.windows)
        con_escala = measure_line(self.wave, flujo, self.ferr, LINEA, lsf_fwhm_A=2.4,
                                  n_mc=0, windows=self.windows, null_z=nulls)
        self.assertEqual(sin_escala.status, "detected")      # el comportamiento viejo
        self.assertEqual(con_escala.status, "upper_limit")   # el de la v2
        self.assertGreater(con_escala.sigma_inflation, 5.0)

    def test_a_genuinely_strong_line_survives_the_empirical_scale(self):
        """La corrección no mata todo: una línea muy por encima de los controles sigue detectada."""
        flujo = _gauss(self.wave, 6562.8, 2.4, 400.0)
        nulls = null_z_from_controls(self.wave, _controles(self.wave, 200, 10.0),
                                     self.ferr, center_A=6562.8, lsf_fwhm_A=2.4,
                                     windows=self.windows)
        m = measure_line(self.wave, flujo, self.ferr, LINEA, lsf_fwhm_A=2.4,
                         n_mc=0, windows=self.windows, null_z=nulls)
        self.assertEqual(m.status, "detected")
        self.assertEqual(m.fap_empirical, 0.0)

    def test_quiet_controls_never_boost_significance(self):
        """El suelo en 1: controles tranquilos NO pueden inflar el z del objeto.

        Sin suelo, `Ca II 8662` salía `detected` con z nominal 1.40 porque su
        inflación medida era 0.27.
        """
        flujo = _gauss(self.wave, 6562.8, 2.4, 1.4)
        nulls = null_z_from_controls(self.wave, _controles(self.wave, 200, 0.05),
                                     self.ferr, center_A=6562.8, lsf_fwhm_A=2.4,
                                     windows=self.windows)
        m = measure_line(self.wave, flujo, self.ferr, LINEA, lsf_fwhm_A=2.4,
                         n_mc=0, windows=self.windows, null_z=nulls)
        self.assertLess(m.sigma_inflation, 1.0)              # medido: los controles callan
        self.assertEqual(m.sigma_inflation_applied, 1.0)     # aplicado: no se usa para inflar
        self.assertAlmostEqual(m.z_emp, m.z_score, places=9)
        self.assertIn("null_quieter_than_budget", m.flags)
        self.assertNotEqual(m.status, "detected")

    def test_the_upper_limit_grows_with_the_measured_inflation(self):
        """Un límite calculado con el sigma propagado sería optimista en ese factor."""
        flujo = _gauss(self.wave, 6562.8, 2.4, 1.0)
        nulls = null_z_from_controls(self.wave, _controles(self.wave, 200, 8.0),
                                     self.ferr, center_A=6562.8, lsf_fwhm_A=2.4,
                                     windows=self.windows)
        sin_ = measure_line(self.wave, flujo, self.ferr, LINEA, lsf_fwhm_A=2.4,
                            n_mc=0, windows=self.windows)
        con_ = measure_line(self.wave, flujo, self.ferr, LINEA, lsf_fwhm_A=2.4,
                            n_mc=0, windows=self.windows, null_z=nulls)
        razon = con_.flux_upper_limit_5sigma / sin_.flux_upper_limit_5sigma
        self.assertAlmostEqual(razon, con_.sigma_inflation_applied, places=6)

    def test_without_controls_it_says_so_instead_of_pretending(self):
        flujo = _gauss(self.wave, 6562.8, 2.4, 12.0)
        m = measure_line(self.wave, flujo, self.ferr, LINEA, lsf_fwhm_A=2.4,
                         n_mc=0, windows=self.windows)
        self.assertEqual(m.null_source, "none")
        self.assertEqual(m.n_controls, 0)
        self.assertTrue(np.isnan(m.fap_empirical))

    def test_object_and_controls_share_the_same_windows(self):
        """Si cada uno eligiera sus ventanas, no sería la misma medida.

        Se comprueba pidiendo el catálogo entero: `measure_catalog` construye
        las ventanas una vez por línea y las reparte.
        """
        catalogo = [LINEA, {"name": "vecina", "wave_A": 6600.0, "family": "x", "kind": "x"}]
        flujo = _gauss(self.wave, 6562.8, 2.4, 5.0)
        ms = measure_catalog(self.wave, flujo, self.ferr, catalogo, lsf_fwhm_A=2.4,
                             n_mc=0, control_fluxes=_controles(self.wave, 40, 1.0))
        for m in ms:
            with self.subTest(linea=m.name):
                self.assertEqual(m.n_controls, 40)
                self.assertNotEqual(m.null_source, "none")
        # Y son las ventanas del CATALOGO (con el vecino excluido del continuo),
        # no unas por defecto: la exclusión hace un hueco dentro de la ventana,
        # así que se comprueba canal a canal y no por sus extremos.
        line_mask, blue, red = build_windows(self.wave, LINEA, catalog=catalogo)
        vecino = np.abs(self.wave - 6600.0) <= 6.0
        self.assertTrue(np.any(vecino))
        self.assertFalse(np.any((blue | red) & vecino),
                         "el continuo de Halpha no debe incluir canales de la línea vecina")
        # La nula se calcula sobre esas mismas máscaras.
        nulls = null_z_from_controls(self.wave, _controles(self.wave, 40, 1.0), self.ferr,
                                     center_A=6562.8, lsf_fwhm_A=2.4,
                                     windows=(line_mask, blue, red))
        self.assertEqual(np.isfinite(nulls).sum(), 40)

    def test_a_control_grid_that_does_not_match_is_refused(self):
        """Mallas distintas compararían canales distintos: error, no aviso."""
        with self.assertRaises(ValueError):
            measure_catalog(self.wave, np.zeros(self.wave.size), self.ferr, [LINEA],
                            lsf_fwhm_A=2.4, n_mc=0,
                            control_fluxes=np.zeros((10, self.wave.size - 5)))

    def test_the_quantile_is_the_same_estimator_E3_uses(self):
        """Dos etapas no deben medir lo mismo de dos maneras."""
        from musepipe.stages.stage_h03_limits import empirical_upper_quantile

        nulls = null_z_from_controls(self.wave, _controles(self.wave, 60, 3.0),
                                     self.ferr, center_A=6562.8, lsf_fwhm_A=2.4,
                                     windows=self.windows)
        esc = empirical_scale(nulls)
        finitos = np.asarray(nulls)[np.isfinite(nulls)]
        self.assertAlmostEqual(esc["z_quantile"],
                               empirical_upper_quantile(finitos, fap=0.01), places=12)
        self.assertAlmostEqual(esc["sigma_inflation"], esc["z_quantile"] / Z_GAUSS_99, places=12)

    def test_n_controls_sets_the_fap_floor(self):
        nulls = null_z_from_controls(self.wave, _controles(self.wave, 33, 1.0),
                                     self.ferr, center_A=6562.8, lsf_fwhm_A=2.4,
                                     windows=self.windows)
        esc = empirical_scale(nulls, object_z=0.0)
        self.assertEqual(esc["n_controls"], 33)
        self.assertAlmostEqual(esc["min_resolvable_fap"], 1.0 / 34.0)


if __name__ == "__main__":
    unittest.main()
