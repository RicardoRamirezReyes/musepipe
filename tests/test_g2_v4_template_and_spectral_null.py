"""Spec G2 v4: continuo de plantilla en el rojo, nula espectral y límite al 99 %.

La v4 existe porque en el rojo de un M tardío la recta de la v1 no sigue las bandas de
TiO/VO (O I 8446 salía a z = −9 en ROXs 42B b) y los controles, que no tienen fotosfera,
no podían destapar ese error. Estos tests fijan las cuatro piezas con espectros sintéticos:

- el continuo de plantilla quita una banda que la recta convierte en «línea»;
- la nula espectral ve el error del continuo que la espacial no ve, y solo puede BAJAR z_emp;
- los placebos respetan las líneas del catálogo, las bandas telúricas fuertes y el láser;
- el límite al 99 % es la regla de E3 v2 sobre el filtro adaptado.

Y que sin pedir nada la librería sigue siendo la v3: lo que la v4 añade es opcional.
"""
from __future__ import annotations

import unittest

import numpy as np

from musepipe.lines import (
    Z_GAUSS_99,
    measure_catalog,
    measure_line,
    null_z_spectral,
    placebo_centers,
    template_fit_mask,
)

LSF = 2.3


def _banda(wave, cabeza, profundidad, escala):
    """Banda molecular con cabeza: un salto hacia el rojo que se recupera exponencialmente."""
    d = wave - cabeza
    return np.where(d >= 0, 1.0 - profundidad * np.exp(-d / escala), 1.0)


class G2V4Base(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261026)
        self.wave = np.arange(7600.0, 9300.0, 1.25)
        w = self.wave
        # «fotosfera»: pendiente suave con cabezas de banda cada ~60-90 A, una justo al lado
        # de la linea de 8446 A
        self.tmpl = (1.0 + 0.0004 * (w - 8400.0)) * _banda(w, 8443.0, 0.35, 12.0) \
            * _banda(w, 8205.0, 0.25, 20.0) * _banda(w, 8660.0, 0.2, 15.0) \
            * _banda(w, 8860.0, 0.3, 18.0) * _banda(w, 9050.0, 0.25, 15.0)
        self.cont_true = 100.0 * self.tmpl * (1.0 + 0.05 * (w - 8400.0) / 500.0)
        self.ferr = np.full(w.size, 1.0)
        self.linea = {"name": "O I 8446", "wave_A": 8446.36, "family": "O I", "kind": "emission"}
        self.catalogo = [self.linea, {"name": "Ca II 8542", "wave_A": 8542.09},
                         {"name": "Pa 9", "wave_A": 9229.01}]

    def _ruido(self, n=None):
        return self.rng.normal(0.0, 1.0, self.wave.size if n is None else (n, self.wave.size))


class TemplateContinuumTests(G2V4Base):
    def test_a_straight_line_reads_the_band_as_a_line_and_the_template_does_not(self):
        flux = self.cont_true + self._ruido()
        recta = measure_line(self.wave, flux, self.ferr, self.linea, lsf_fwhm_A=LSF, n_mc=0,
                             catalog=self.catalogo)
        plantilla = measure_line(self.wave, flux, self.ferr, self.linea, lsf_fwhm_A=LSF, n_mc=0,
                                 catalog=self.catalogo, continuum_template=self.tmpl,
                                 template_name="sintetica")
        self.assertGreater(abs(recta.z_score), 5.0)          # la banda sale como «linea»
        self.assertLess(abs(plantilla.z_score), 3.0)         # con plantilla, ruido
        self.assertEqual(plantilla.continuum_model, "template:sintetica")
        self.assertEqual(recta.continuum_model, "linear")

    def test_the_template_continuum_recovers_an_injected_line(self):
        flujo = 60.0
        sig = LSF / 2.3548200450309493
        linea = flujo / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((self.wave - 8446.36) / sig) ** 2)
        flux = self.cont_true + linea + self._ruido()
        m = measure_line(self.wave, flux, self.ferr, self.linea, lsf_fwhm_A=LSF, n_mc=200, seed=1,
                         catalog=self.catalogo, continuum_template=self.tmpl)
        self.assertAlmostEqual(m.flux_direct, flujo, delta=4.0 * max(m.flux_direct_err, 1.0))
        self.assertAlmostEqual(m.flux_mf, flujo, delta=5.0)
        self.assertEqual(m.status, "detected")

    def test_too_few_fit_pixels_is_not_measurable_never_measured_with_what_is_left(self):
        m = measure_line(self.wave, self.cont_true, self.ferr, self.linea, lsf_fwhm_A=LSF, n_mc=0,
                         continuum_template=self.tmpl,
                         template_mask=np.zeros(self.wave.size, dtype=bool))
        self.assertEqual(m.status, "not_measurable")
        self.assertTrue(m.reason.startswith("template_fit_pixels="))

    def test_the_fit_mask_leaves_out_core_neighbours_flags_and_excluded_bands(self):
        bad = np.zeros(self.wave.size, dtype=bool)
        bad[np.abs(self.wave - 8480.0) < 2.0] = True
        excl = (self.wave > 8420.0) & (self.wave < 8425.0)
        mask = template_fit_mask(self.wave, 8446.36, self.tmpl,
                                 catalog=self.catalogo + [{"wave_A": 8467.25}],
                                 bad_flags=bad, exclude_mask=excl)
        d = np.abs(self.wave - 8446.36)
        self.assertFalse(np.any(mask & (d <= 6.0)))                           # nucleo
        self.assertFalse(np.any(mask & (np.abs(self.wave - 8467.25) <= 6.0)))  # vecina
        self.assertFalse(np.any(mask & bad))
        self.assertFalse(np.any(mask & excl))
        self.assertFalse(np.any(mask & (d > 60.0)))
        self.assertGreater(int(np.count_nonzero(mask)), 30)

    def test_only_lines_redward_of_the_threshold_change(self):
        """Hα y el azul no se tocan: con y sin plantilla, sus filas son identicas."""
        wave = np.arange(6400.0, 9300.0, 1.25)
        tmpl = np.interp(wave, self.wave, self.tmpl, left=1.0)
        flux = 100.0 * tmpl + self.rng.normal(0.0, 1.0, wave.size)
        catalogo = [{"name": "Halpha", "wave_A": 6562.8}, self.linea]
        ferr = np.ones(wave.size)
        sin = measure_catalog(wave, flux, ferr, catalogo, lsf_fwhm_A=LSF, n_mc=50, seed=3)
        con = measure_catalog(wave, flux, ferr, catalogo, lsf_fwhm_A=LSF, n_mc=50, seed=3,
                              continuum_template=tmpl, template_min_A=7000.0)
        self.assertEqual(sin[0].to_row(), con[0].to_row())
        self.assertNotEqual(sin[1].continuum_model, con[1].continuum_model)


class SpectralNullTests(G2V4Base):
    def setUp(self):
        super().setUp()
        # un continuo con ondulacion que la recta no sigue: los controles (sin fotosfera)
        # no la tienen, y por eso la nula espacial no la ve
        w = self.wave
        self.flux = 100.0 + 4.0 * np.sin(2 * np.pi * (w - 7600.0) / 23.0) + self._ruido()
        self.controles = self._ruido(33)

    def test_the_spectral_null_sees_what_the_controls_do_not(self):
        ms = measure_catalog(self.wave, self.flux, self.ferr, [self.linea], lsf_fwhm_A=LSF, n_mc=0,
                             control_fluxes=self.controles, spectral_null=True)
        m = ms[0]
        self.assertLess(m.sigma_inflation, 1.6)                    # los controles: ruido
        self.assertGreater(m.sigma_inflation_spectral, 2.0)        # el espectro: la ondulacion
        self.assertEqual(m.sigma_inflation_applied, m.sigma_inflation_spectral)
        self.assertIn("spectral_null_dominates", m.flags)
        self.assertAlmostEqual(m.z_emp, m.z_score / m.sigma_inflation_spectral, places=12)
        self.assertGreaterEqual(m.n_placebos, 10)

    def test_the_spectral_null_never_raises_z_emp(self):
        flux = 100.0 + self._ruido()
        controles = 3.0 * self._ruido(33)                          # controles MAS ruidosos
        base = measure_catalog(self.wave, flux, self.ferr, [self.linea], lsf_fwhm_A=LSF, n_mc=0,
                               control_fluxes=controles)[0]
        v4 = measure_catalog(self.wave, flux, self.ferr, [self.linea], lsf_fwhm_A=LSF, n_mc=0,
                             control_fluxes=controles, spectral_null=True)[0]
        self.assertEqual(v4.z_emp, base.z_emp)
        self.assertEqual(v4.sigma_inflation_applied, base.sigma_inflation_applied)
        self.assertNotIn("spectral_null_dominates", v4.flags)

    def test_too_few_placebos_is_said_not_used(self):
        m = measure_line(self.wave, self.flux, self.ferr, self.linea, lsf_fwhm_A=LSF, n_mc=0,
                         null_z_spectral=np.array([0.1, -0.3, 0.5]))
        self.assertIn("spectral_null_unavailable", m.flags)
        self.assertTrue(np.isnan(m.sigma_inflation_spectral))
        self.assertEqual(m.n_placebos, 3)

    def test_placebos_keep_away_from_lines_bands_and_edges(self):
        catalogo = self.catalogo + [{"wave_A": 8498.02}, {"wave_A": 8662.14}]
        banda = (8900.0, 9000.0, 6.0)
        centros = placebo_centers(self.wave, 8446.36, catalogo, exclude_ranges=(banda,))
        self.assertGreater(len(centros), 10)
        for lp in centros:
            self.assertGreaterEqual(abs(lp - 8446.36), 20.0 - 1e-9)
            self.assertLessEqual(abs(lp - 8446.36), 300.0 + 1e-9)
            self.assertTrue(all(abs(lp - c["wave_A"]) > 10.0 for c in catalogo))
            self.assertFalse(8894.0 <= lp <= 9006.0)
            self.assertTrue(self.wave[0] + 50.0 <= lp <= self.wave[-1] - 50.0)
        resto = np.diff(sorted(centros)) % 12.5
        self.assertTrue(np.all(np.isclose(resto, 0.0, atol=1e-6) | np.isclose(resto, 12.5, atol=1e-6)))

    def test_template_placebos_use_the_template_continuum(self):
        flux = self.cont_true + self._ruido()
        z_recta, _ = null_z_spectral(self.wave, flux, self.ferr, center_A=8446.36, lsf_fwhm_A=LSF,
                                     catalog=self.catalogo)
        z_tmpl, _ = null_z_spectral(self.wave, flux, self.ferr, center_A=8446.36, lsf_fwhm_A=LSF,
                                    catalog=self.catalogo, template=self.tmpl)
        self.assertGreater(np.nanmax(np.abs(z_recta)), 5.0)    # la recta: las bandas son «lineas»
        self.assertLess(np.nanmax(np.abs(z_tmpl)), 4.5)         # la plantilla: ruido


class UpperLimit99Tests(G2V4Base):
    def test_the_99_percent_limit_is_the_E3_v2_rule_on_the_matched_filter(self):
        flux = 100.0 + self._ruido()
        controles = 2.0 * self._ruido(33)
        m = measure_catalog(self.wave, flux, self.ferr, [self.linea], lsf_fwhm_A=LSF, n_mc=0,
                            control_fluxes=controles, throughput=0.8)[0]
        sigma_prop = m.flux_mf / m.z_score
        self.assertAlmostEqual(m.flux_mf_sigma_emp, sigma_prop * m.sigma_inflation_applied, places=10)
        esperado = (max(m.flux_mf, 0.0) + Z_GAUSS_99 * m.flux_mf_sigma_emp) / 0.8
        self.assertAlmostEqual(m.flux_upper_limit_99, esperado, places=10)

    def test_a_negative_measurement_does_not_shrink_the_limit_below_the_margin(self):
        flux = 100.0 - 3.0 * np.exp(-0.5 * ((self.wave - 8446.36) / 1.0) ** 2) + self._ruido()
        m = measure_line(self.wave, flux, self.ferr, self.linea, lsf_fwhm_A=LSF, n_mc=0)
        self.assertLess(m.flux_mf, 0.0)
        self.assertAlmostEqual(m.flux_upper_limit_99, Z_GAUSS_99 * m.flux_mf_sigma_emp, places=10)


class StageTemplateResolutionTests(unittest.TestCase):
    def test_an_undeclared_template_is_said_not_invented(self):
        from musepipe.stages.stage_g2_measure_lines import _resolve_continuum_template

        flux, info = _resolve_continuum_template({}, np.arange(10.0), 2.3)
        self.assertIsNone(flux)
        self.assertIsNone(info["declared"])
        self.assertIn("not declared", info["reason"])

    def test_a_declared_template_that_does_not_exist_is_an_error(self):
        from unittest import mock

        from musepipe.stages import stage_g2_measure_lines as g2

        cfg = {"g2_continuum_template": {"library": "no_existe", "object": "LM601"}, "project_root": "."}
        with mock.patch("musepipe.stages.stage_g3_template_fit._build_libraries", return_value=[]):
            with self.assertRaises(RuntimeError):
                g2._resolve_continuum_template(cfg, np.arange(10.0), 2.3)

    def test_the_table_carries_the_v4_columns(self):
        from musepipe.stages.stage_g2_measure_lines import TABLE_FIELDS

        for c in ("continuum_model", "sigma_inflation_spectral", "n_placebos", "flux_mf",
                  "flux_mf_sigma_emp", "flux_upper_limit_99"):
            self.assertIn(c, TABLE_FIELDS)


if __name__ == "__main__":
    unittest.main()
