"""La ventana de las líneas en los ajustes de T_eff (pregunta del autor del 2026-10-28).

Tres piezas: `tipo_ir_plantillas.ventanas_de_lineas` / `ventana_extra` (la ventana de las líneas de G2 en una copia del
config, una para Hα y otra para el resto), las opciones `--ventana-lineas` / `--ventana-halpha` de
`scripts/teff_bayes.py` (etiqueta, json, tablas) y la fuga de `scripts/ventanas_lineas.py`. Las primeras clases no leen
nada de disco. `DatosRealesTests` (marca `external_data`) lee los runs del paper de `MUSE_PROJECT_ROOT`; si faltan, se
salta.
"""
import argparse
import copy
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROYECTO = Path(os.environ.get("MUSE_PROJECT_ROOT", ROOT))
C_KMS = 299792.458


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _args(**kw):
    base = dict(slug="X", rejilla="btsettl", modo="libre", tracks="ATMO2020", av="primaria", plantilla=None,
                av_fija=None, inyecta_bt=None, datos="todo", r250=False, degrada="cuadratura", semilla=1,
                etiqueta=None, mascara_muse="paper")
    base.update(kw)
    return argparse.Namespace(**base)


class VentanasDeLineasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import tipo_ir_plantillas as tp
        cls.tp = tp

    def test_sin_ventanas_devuelve_el_mismo_config(self):
        cfg = {"g3_fit_masks": {"line_window_kms": 300.0}}
        out, extra = self.tp.ventanas_de_lineas(cfg)
        self.assertIs(out, cfg)
        self.assertEqual(extra, [])

    def test_una_ventana_global(self):
        cfg = {"g3_fit_masks": {"line_window_kms": 300.0, "telluric_bands_A": [[6860.0, 6960.0]]}}
        antes = copy.deepcopy(cfg)
        out, extra = self.tp.ventanas_de_lineas(cfg, 120.0)
        self.assertEqual(out["g3_fit_masks"]["line_window_kms"], 120.0)
        self.assertEqual(out["g3_fit_masks"]["telluric_bands_A"], [[6860.0, 6960.0]])
        self.assertEqual(extra, [])
        self.assertEqual(cfg, antes)                       # el original no se toca

    def test_halpha_mayor_que_el_resto(self):
        out, extra = self.tp.ventanas_de_lineas({}, 50.0, 100.0)
        self.assertEqual(out["g3_fit_masks"]["line_window_kms"], 50.0)
        self.assertEqual(extra, [(True, 100.0)])

    def test_resto_mayor_que_halpha(self):
        out, extra = self.tp.ventanas_de_lineas({"g3_fit_masks": {"line_window_kms": 300.0}}, None, 100.0)
        self.assertEqual(out["g3_fit_masks"]["line_window_kms"], 100.0)
        self.assertEqual(extra, [(False, 300.0)])        # la del run para las demás

    def test_negativa_no_vale(self):
        with self.assertRaises(ValueError):
            self.tp.ventanas_de_lineas({}, -1.0)

    def test_ventana_extra_con_la_regla_de_build_fit_masks(self):
        wave = np.arange(6500.0, 6800.0, 1.25)
        rests = [6562.8, 6678.15]
        m_ha = self.tp.ventana_extra(wave, rests, [(True, 100.0)])
        half = 6562.8 * 100.0 / C_KMS
        np.testing.assert_array_equal(m_ha, (wave >= 6562.8 - half) & (wave <= 6562.8 + half))
        m_resto = self.tp.ventana_extra(wave, rests, [(False, 100.0)])
        half = 6678.15 * 100.0 / C_KMS
        np.testing.assert_array_equal(m_resto, (wave >= 6678.15 - half) & (wave <= 6678.15 + half))


class OpcionVentanaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = _carga("teff_bayes")

    def test_la_etiqueta_del_paper_no_cambia(self):
        self.assertEqual(self.tb.etiqueta_de(_args()), "X_libre_avprimaria_s1")

    def test_la_etiqueta_lleva_las_ventanas(self):
        a = _args(mascara_muse="sin_o2b", ventana_lineas=50.0, ventana_halpha=100.0)
        self.assertEqual(self.tb.etiqueta_de(a), "X_libre_avprimaria_mascara-sin_o2b_ventana-50_ventanaha-100_s1")
        self.assertEqual(self.tb.etiqueta_de(_args(ventana_lineas=100.0)), "X_libre_avprimaria_ventana-100_s1")

    def test_las_cadenas_con_ventana_no_entran_en_las_tablas(self):
        d = dict(modo="libre", av="primaria", datos="todo", rejilla="btsettl", r_analisis=500)
        self.assertTrue(self.tb._de_la_tabla(d))
        self.assertTrue(self.tb._de_rejillas(d))
        for k in ("ventana_lineas_kms", "ventana_halpha_kms"):
            self.assertFalse(self.tb._de_la_tabla({**d, k: 100.0}))
            self.assertFalse(self.tb._de_rejillas({**d, k: 100.0}))


class FugaSinteticaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import ventanas_lineas as vl
        cls.vl = vl

    def test_fraccion_fuera_de_una_gaussiana(self):
        from scipy.special import erfc
        x = np.arange(0, 10.0001, 0.001)
        sig = 1.0
        f = self.vl.fraccion_fuera(x, np.exp(-0.5 * (x / sig) ** 2), 10.0)
        for d in (0.5, 1.0, 2.0):
            self.assertAlmostEqual(float(f(d)), float(erfc(d / (sig * np.sqrt(2)))), places=5)

    def test_la_linea_en_canales_conserva_el_flujo(self):
        wave = np.arange(6400.0, 6700.0, 1.25)
        x = np.arange(-10, 10.0001, 0.01)
        lin = self.vl.linea_en_canales(wave, 123.0, 6562.4, x, np.exp(-0.5 * x ** 2))
        self.assertAlmostEqual(float(np.sum(lin * np.diff(self.vl.bordes(wave)))), 123.0, places=9)

    def _inputs(self, w_kms):
        wave = np.arange(6300.0, 6900.0, 1.25)
        rest = 6562.8
        half = rest * w_kms / C_KMS
        mask = (wave >= rest - half) & (wave <= rest + half)
        cov = {"block_bounds": np.array([[0, wave.size]]), "n_eff_over_n_by_block": np.array([0.7])}
        return dict(wave=wave, flux=np.ones(wave.size), err=np.full(wave.size, 0.1), mask=mask, bad_mask=None,
                    cov=cov, mask_prov={}, err_column="flux_err_total")

    def test_la_fuga_es_lineal_y_baja_con_la_ventana(self):
        x = np.arange(-10, 10.0001, 0.01)
        p = np.exp(-0.5 * (x / 1.0) ** 2)
        cfg = {"g3_fit_wave_range_A": [6350.0, 6850.0]}
        kern = lambda w: np.sqrt(np.clip((w / 500.0) ** 2 - 2.3 ** 2, 0, None))  # noqa: E731
        fugas = []
        for w in (30.0, 300.0):
            inp = self._inputs(w)
            wb, fu, eb = self.vl.fuga_unitaria(inp, cfg, kern, 6562.8, x, p)
            # lineal: el doble de flujo da el doble de fuga
            inp2 = dict(inp, flux=2 * np.where(np.isfinite(inp["flux"]),
                                                  self.vl.linea_en_canales(inp["wave"], 1.0, 6562.8, x, p), np.nan))
            fs2 = self.vl._fs(inp2, cfg, kern)
            np.testing.assert_allclose(fs2.flux_bin, 2 * fu, rtol=1e-12, atol=1e-15)
            fugas.append(np.max(np.abs(fu)))
        self.assertGreater(fugas[0], fugas[1])
        self.assertLess(fugas[1], 1e-6)              # a ±300 km/s, una gaussiana de σ = 1 Å no deja nada

    def test_w_minima(self):
        por_w = {"50": {"todas": {"z_max": 0.3}}, "100": {"todas": {"z_max": 0.05}},
                 "150": {"todas": {"z_max": 0.12}}, "200": {"todas": {"z_max": 0.01}}}
        self.assertEqual(self.vl.w_minima(por_w, None, 0.1), 200.0)   # la de 100 no vale: 150 vuelve a pasar
        self.assertEqual(self.vl.w_minima(por_w, None, 0.2), 100.0)
        self.assertIsNone(self.vl.w_minima({"300": {"todas": {"z_max": 1.0}}}, None, 0.1))


def _runs_del_paper():
    fichas = [json.loads(f.read_text()) for f in sorted((ROOT / "targets").glob("*.json"))]
    return {d["slug"]: d["paper"]["run"] for d in fichas if "paper" in d}


@pytest.mark.external_data
class DatosRealesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = _runs_del_paper()
        if not cls.runs or not all((PROYECTO / "runs" / r / "stages" / "spec_final_object.fits").exists()
                                   for r in cls.runs.values()):
            raise unittest.SkipTest("sin los runs del paper (MUSE_PROJECT_ROOT)")
        cls.tb = _carga("teff_bayes")
        cls.raiz_antes = cls.tb.ROOT
        cls.tb.fija_raiz(PROYECTO)
        from scripts import tipo_ir_plantillas as tp
        cls.tp = tp

    @classmethod
    def tearDownClass(cls):
        cls.tb.fija_raiz(cls.raiz_antes)

    def test_la_ventana_del_run_explicita_da_el_mismo_segmento(self):
        for slug, run in self.runs.items():
            a = self.tp.segmento_muse(run)
            for vl, vh in ((300.0, None), (300.0, 300.0), (None, 300.0)):
                b = self.tp.segmento_muse(run, ventana_kms=vl, ventana_ha_kms=vh)
                for k in ("w", "f", "e", "bueno"):
                    np.testing.assert_array_equal(a[k], b[k], err_msg=f"{slug} {k} {vl} {vh}")

    def test_menos_ventana_mas_bins_y_halpha_aparte(self):
        for slug, run in self.runs.items():
            n = lambda vl, vh=None: int(self.tp.segmento_muse(run, ventana_kms=vl, ventana_ha_kms=vh)["bueno"].sum())  # noqa: E731,E501
            self.assertEqual(n(None), 84, slug)
            self.assertGreaterEqual(n(100.0), n(300.0))
            self.assertGreater(n(50.0), n(300.0))
            self.assertGreaterEqual(n(50.0, 100.0), n(100.0))   # Hα a 100 y el resto a 50: al menos los de 100
            self.assertEqual(self.tb.bins_muse(slug, 100.0)["paper"]["n"], n(100.0))


if __name__ == "__main__":
    unittest.main()
