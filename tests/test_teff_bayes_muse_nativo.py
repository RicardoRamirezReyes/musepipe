"""`scripts/teff_bayes.py --muse-nativo` (el espectro MUSE a su resolución, canal a canal) y `--ens-vr` (con un
ensanchamiento y una velocidad libres), 2026-10-30.

Lo que se prueba son invariantes, no cifras del run: sin la opción todo es lo de siempre (la etiqueta, el json, los
modelos de `ajuste_conjunto.modelos_btsettl`); el núcleo es la LSF; las combinaciones sin sentido fallan antes de leer
nada y una rejilla de paso basto, al medir su paso; `modelo_de_cadena` lee la opción del json. Con los datos reales:
el segmento nativo es, canal a canal, el espectro del ajuste nativo de G3, y el modelo nativo deja el IR como estaba.
Con `--ens-vr`: sin ensanchamiento ni velocidad, el modelo es el de los canales sin tocar (bit a bit) y, en los datos,
el nativo salvo el orden de interpolar y promediar; una línea se mueve λ·v/c y se ensancha en cuadratura.
"""
import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from tests._datos import carga_script, modelo_teff, raiz_temporal, requiere, runs_del_paper

#: un json de cadena del paper, con las claves que llevan (las de `tests/test_teff_bayes_variantes.py`)
PAPER = dict(etiqueta="ROXs12b_libre_avprimaria_s1", slug="ROXs12b", run="ROXs12b_invvar", modo="libre",
             av="primaria", av_fija=None, rejilla="btsettl", datos="todo", r_analisis=500, nodos_interpolados=[],
             ejes={}, tracks=None, nlive=500, semilla=1, ndim=12, nombres=[], segundos=1.0, ncall=1, logz=-1.0,
             logz_err=0.1, logl_max=-1.0, chi2nu_mejor={}, chi2nu_mejor_inflado={}, mejor={}, verdad=None,
             posterior={}, n_puntos={})


class ValidaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_sin_la_opcion_no_falla_nada(self):
        self.tb.valida_muse_nativo(False, datos="ir", rejilla="sphinx", r250=True, plantilla="TWA26")

    def test_combinaciones_sin_sentido(self):
        for kw in (dict(datos="ir"), dict(r250=True), dict(rejilla="sphinx"), dict(rejilla="atmo_ceq"),
                   dict(rejilla="btsettl_bin"), dict(plantilla="TWA26")):
            with self.subTest(**kw), self.assertRaises(ValueError):
                self.tb.valida_muse_nativo(True, **kw)

    def test_combinaciones_validas(self):
        # DRIFT-PHOENIX pasa aquí: la rechaza después `comprueba_muestreo_nativo`, con el paso de sus npz
        for kw in (dict(datos="muse"), dict(datos="todo"), dict(rejilla="bt_dusty"), dict(rejilla="diamondback_f2"),
                   dict(rejilla="drift")):
            with self.subTest(**kw):
                self.tb.valida_muse_nativo(True, **kw)

    def test_modelo_falla_antes_de_leer_nada(self):
        # un slug que no existe: si validara después de leer, el error sería otro
        with self.assertRaises(ValueError):
            self.tb.Modelo("no_existe", datos="ir", muse_nativo=True)

    def test_un_modelo_de_new_no_es_nativo(self):
        self.assertFalse(self.tb.Modelo.__new__(self.tb.Modelo).muse_nativo)


class MuestreoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_paso_fino_pasa_y_basto_no(self):
        lsf = 2.285
        self.tb.comprueba_muestreo_nativo(0.02, lsf, "BT-Settl")
        self.tb.comprueba_muestreo_nativo(self.tb.MUESTREO_MAX_NATIVO * lsf, lsf, "en el tope")
        for paso in (2.0, math.nan):
            with self.subTest(paso=paso), self.assertRaises(ValueError):
                self.tb.comprueba_muestreo_nativo(paso, lsf, "DRIFT-PHOENIX")


class DegradaLsfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    @staticmethod
    def _linea(fwhm_in, paso=0.2):
        w = np.arange(7000.0, 7200.0, paso)
        s = fwhm_in / (2 * math.sqrt(2 * math.log(2)))
        return w, 1.0 + np.exp(-0.5 * ((w - 7100.0) / s) ** 2)

    @staticmethod
    def _fwhm(w, f):
        p = f - 1.0
        m = np.sum(w * p) / np.sum(p)
        return 2 * math.sqrt(2 * math.log(2)) * math.sqrt(np.sum((w - m) ** 2 * p) / np.sum(p))

    def test_el_nucleo_es_la_lsf_y_conserva_el_flujo(self):
        w, f = self._linea(0.5)
        g = self.tb.degrada_lsf(w, f, 2.285)
        self.assertAlmostEqual(self._fwhm(w, g), math.hypot(0.5, 2.285), delta=0.005)
        self.assertAlmostEqual(float(np.sum(g - 1.0)), float(np.sum(f - 1.0)), delta=1e-6 * float(np.sum(f - 1.0)))

    def test_un_hueco_vuelve_a_nan_solo_cerca(self):
        w, f = self._linea(0.5)
        f[(w > 7050.0) & (w < 7052.0)] = np.nan
        g = self.tb.degrada_lsf(w, f, 2.285)
        self.assertTrue(np.all(np.isnan(g[(w > 7050.5) & (w < 7051.5)])))
        self.assertTrue(np.all(np.isfinite(g[(w < 7045.0) | (w > 7057.0)])))


class ModelosBtsettlTests(unittest.TestCase):
    """`ajuste_conjunto.modelos_btsettl` con una rejilla de mentira en un directorio temporal."""

    @classmethod
    def setUpClass(cls):
        cls.aj = carga_script("ajuste_conjunto")
        cls.dir = tempfile.TemporaryDirectory()
        w = np.arange(4000.0, 25500.0, 0.05)
        for teff in (2500, 2600):
            flujo = 1e5 * (w / 1e4) ** -2 * (1 + 0.1 * np.sin(w / 7.0)) * teff / 2500
            meta = dict(teff_k=float(teff), logg=4.0)
            np.savez(Path(cls.dir.name) / f"teff{teff}_logg4.0.npz", wave_A=w, flux=flujo, meta_json=json.dumps(meta))

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_sin_paso_vis_lo_de_siempre(self):
        a = self.aj.modelos_btsettl(self.dir.name)
        b = self.aj.modelos_btsettl(self.dir.name, paso_vis_A=None)
        self.assertEqual(len(a), 2)
        for x, y in zip(a, b):
            self.assertEqual(set(x), set(y))
            self.assertNotIn("muestreo_vis_A", x)
            for brazo in ("vis", "nir"):
                np.testing.assert_array_equal(x[brazo][0], y[brazo][0])
                np.testing.assert_array_equal(x[brazo][1], y[brazo][1])

    def test_paso_vis_solo_cambia_el_visible(self):
        a = self.aj.modelos_btsettl(self.dir.name)
        b = self.aj.modelos_btsettl(self.dir.name, paso_vis_A=0.2)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x["nir"][0], y["nir"][0])
            np.testing.assert_array_equal(x["nir"][1], y["nir"][1])
            np.testing.assert_allclose(np.diff(y["vis"][0]), 0.2, atol=1e-9)
            self.assertLess(y["vis"][0][-1], 10500.0)
            self.assertAlmostEqual(y["muestreo_vis_A"], 0.05, places=9)


class EtiquetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def _a(self, **kw):
        base = dict(etiqueta=None, slug="ROXs12b", rejilla="btsettl", modo="libre", tracks="ATMO2020", av="primaria",
                    plantilla=None, av_fija=None, inyecta_bt=None, datos="muse", r250=False, semilla=1)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_sin_la_opcion_la_de_siempre(self):
        for a in (self._a(), self._a(muse_nativo=False)):
            self.assertEqual(self.tb.etiqueta_de(a), "ROXs12b_libre_avprimaria_solomuse_s1")

    def test_la_pieza_va_detras_de_la_resolucion(self):
        self.assertEqual(self.tb.etiqueta_de(self._a(muse_nativo=True)),
                         "ROXs12b_libre_avprimaria_solomuse_muse-nativo_s1")
        self.assertEqual(self.tb.etiqueta_de(self._a(muse_nativo=True, av_fija=3.0, mascara_muse="g3")),
                         "ROXs12b_libre_avprimaria_avfija3_solomuse_muse-nativo_mascara-g3_s1")
        self.assertEqual(self.tb.etiqueta_de(self._a(muse_nativo=True, ens_vr=True)),
                         "ROXs12b_libre_avprimaria_solomuse_muse-nativo_ens-vr_s1")
        self.assertEqual(self.tb.etiqueta_de(self._a(muse_nativo=True, ens_vr=False)),
                         "ROXs12b_libre_avprimaria_solomuse_muse-nativo_s1")


class VarianteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_nativo_es_variante_y_no_entra_en_las_tablas(self):
        d = {**PAPER, "muse_nativo": True}
        self.assertTrue(self.tb._es_variante(d))
        self.assertFalse(self.tb._de_la_tabla(d))
        self.assertFalse(self.tb._de_rejillas(d))

    def test_ens_vr_es_variante(self):
        for d in ({**PAPER, "muse_nativo": True, "ens_vr": [10.0, 100.0]}, {**PAPER, "ens_vr": [10.0, 100.0]}):
            self.assertTrue(self.tb._es_variante(d))
            self.assertFalse(self.tb._de_la_tabla(d))
        self.assertFalse(self.tb._es_variante({**PAPER, "ens_vr": None}))

    def test_sin_la_clave_o_en_false_es_la_del_paper(self):
        for d in (PAPER, {**PAPER, "muse_nativo": False}):
            self.assertFalse(self.tb._es_variante(d))
            self.assertTrue(self.tb._de_la_tabla(d))


class ReconstruyeTests(unittest.TestCase):
    """`modelo_de_cadena` le pasa al `Modelo` lo que dice el json (con un `Modelo` de mentira que lo apunta)."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def _kw(self, d):
        visto = {}

        class Falso:
            nombres = ["teff"]

            def __init__(self, *args, **kw):
                visto.update(kw)

        with mock.patch.object(self.tb, "Modelo", Falso):
            self.tb.modelo_de_cadena({**PAPER, "mejor": {"teff": 2600.0}, **d})
        return visto

    def test_lee_muse_nativo(self):
        self.assertIs(self._kw({"muse_nativo": True})["muse_nativo"], True)
        self.assertIs(self._kw({})["muse_nativo"], False)

    def test_lee_ens_vr(self):
        self.assertEqual(self._kw({"muse_nativo": True, "ens_vr": [10.0, 100.0]})["ens_vr"], (10.0, 100.0))
        self.assertIsNone(self._kw({})["ens_vr"])


class ValidaEnsVrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_pide_muse_nativo_y_topes_validos(self):
        tb = self.tb
        tb.valida_muse_nativo(True, ens_vr=(tb.ENS_MAX_A, tb.VR_MAX_KMS))
        for nativo, ens_vr in ((False, (10.0, 100.0)), (True, (0.0, 100.0)), (True, (10.0, math.nan)),
                               (True, (10.0,)), (True, (60.0, 100.0))):
            with self.subTest(nativo=nativo, ens_vr=ens_vr), self.assertRaises(ValueError):
                tb.valida_muse_nativo(nativo, ens_vr=ens_vr)

    def test_modelo_falla_antes_de_leer_nada(self):
        with self.assertRaises(ValueError):
            self.tb.Modelo("no_existe", datos="muse", ens_vr=(10.0, 100.0))

    def test_un_modelo_de_new_no_lo_lleva(self):
        self.assertIsNone(self.tb.Modelo.__new__(self.tb.Modelo).ens_vr)


class EnsanchaDesplazaTests(unittest.TestCase):
    """`ensancha_y_desplaza` sobre una línea de absorción gaussiana en 7100 Å (malla de 0.2 Å, canales de 1.25 Å)."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")
        cls.tp = carga_script("tipo_ir_plantillas")
        cls.w = np.arange(7000.0, 7200.0 + 0.1, 0.2)
        s = 2.3 / (2 * math.sqrt(2 * math.log(2)))
        cls.F = 1.0 - 0.5 * np.exp(-0.5 * ((cls.w - 7100.0) / s) ** 2)
        cls.lo = np.arange(7050.0, 7150.0, 1.25)
        cls.hi = cls.lo + 1.25
        cls.nfft = 2048
        cls.f2 = np.fft.rfftfreq(cls.nfft) ** 2

    def _canales(self, ens, vr):
        return self.tb.ensancha_y_desplaza(self.w, self.F, self.lo, self.hi, ens, vr, self.nfft, self.f2)

    def _momentos(self, m):
        x = 0.5 * (self.lo + self.hi)
        p = 1.0 - m
        c = float(np.sum(x * p) / np.sum(p))
        return c, 2 * math.sqrt(2 * math.log(2)) * math.sqrt(float(np.sum((x - c) ** 2 * p) / np.sum(p)))

    def test_sin_nada_es_el_promedio_en_los_canales_bit_a_bit(self):
        np.testing.assert_array_equal(self._canales(0.0, 0.0),
                                      self.tp.media_en_intervalos(self.w, self.F, self.lo, self.hi))

    def test_la_velocidad_mueve_la_linea_lambda_v_sobre_c(self):
        c0, _ = self._momentos(self._canales(0.0, 0.0))
        for vr in (-40.0, 25.0):
            with self.subTest(vr=vr):
                c, _ = self._momentos(self._canales(0.0, vr))
                self.assertAlmostEqual(c - c0, 7100.0 * vr / self.tb.C_KMS, delta=0.01)

    def test_el_ensanchamiento_va_en_cuadratura(self):
        _, f0 = self._momentos(self._canales(0.0, 0.0))
        for ens in (1.5, 4.0):
            with self.subTest(ens=ens):
                _, f = self._momentos(self._canales(ens, 0.0))
                self.assertAlmostEqual(f ** 2 - f0 ** 2, ens ** 2, delta=0.02 * ens ** 2)


class ParametrosEnsVrTests(unittest.TestCase):
    """Los parámetros de un `Modelo` hecho con `__new__` (solo MUSE, sin leer disco)."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def _modelo(self, ens_vr):
        m = self.tb.Modelo.__new__(self.tb.Modelo)
        m.modo, m.ens_vr = "libre", ens_vr
        m.T0, m.dT, m.nT, m.G0, m.dG, m.nG = 2000.0, 100.0, 3, 3.0, 0.5, 2
        m.d_pc, m.d_err, m.av_fija, m.av_modo, m.av0, m.av_err = 140.0, 5.0, None, "primaria", 1.8, 0.3
        m.segs = [dict(nombre="MUSE", banda="vis", sigma=0.20)]
        m.bandas = ["vis"]
        m._prepara_priores()
        return m

    def test_sin_ens_vr_los_de_siempre(self):
        self.assertEqual(self._modelo(None).nombres, ["teff", "logg", "r_rj", "d_pc", "av", "s_MUSE", "lnf_vis"])

    def test_ens_y_vr_detras_de_av_y_uniformes(self):
        m = self._modelo((10.0, 100.0))
        self.assertEqual(m.nombres, ["teff", "logg", "r_rj", "d_pc", "av", "ens_A", "vr_kms", "s_MUSE", "lnf_vis"])
        ie, iv = m.pos["ens_A"], m.pos["vr_kms"]
        u = np.full(m.ndim, 0.5)
        for x, e, v in ((0.0, 0.0, -100.0), (0.25, 2.5, -50.0), (1.0, 10.0, 100.0)):
            u[ie] = u[iv] = x
            th = m.prior_transform(u)
            self.assertEqual((th[ie], th[iv]), (e, v))


# ------------------------------------------------------------------ datos reales
@pytest.mark.external_data
@requiere("runs")
class SegmentoDatosTests(unittest.TestCase):
    """Sin rejillas: el segmento nativo de MUSE de los runs del paper."""

    def test_es_el_espectro_del_ajuste_nativo_de_g3(self):
        from musepipe.config import load_run_config
        from musepipe.models.observed import fit_spectrum
        from musepipe.paths import RunPaths
        from scripts import tipo_ir_plantillas as tp
        with raiz_temporal() as raiz:
            for slug, run in runs_del_paper().items():
                with self.subTest(slug=slug):
                    s = tp.segmento_muse(run, h2o=False, nativo=True)
                    cfg = load_run_config(run, project_root=raiz).config
                    g3 = fit_spectrum(cfg, RunPaths.from_project_root(run, raiz), n_channels=1)
                    np.testing.assert_array_equal(s["w"], g3.wave_bin)
                    np.testing.assert_array_equal(s["f"], g3.flux_bin)
                    np.testing.assert_array_equal(s["e"], g3.err_bin)
                    np.testing.assert_allclose(s["hi"] - s["lo"], np.median(np.diff(g3.wave_bin)), rtol=1e-9)

    def test_las_bandas_de_h2o_quedan_fuera(self):
        from scripts import tipo_ir_plantillas as tp
        with raiz_temporal():
            for slug, run in runs_del_paper().items():
                s = tp.segmento_muse(run, nativo=True)
                w = s["w"][s["bueno"]]
                for lo, hi in tp.H2O_MUSE_A:
                    self.assertFalse(np.any((w >= lo) & (w <= hi)), (slug, lo, hi))


@pytest.mark.external_data
@pytest.mark.slow
@requiere("runs", "ir", "bibliotecas")
class ModeloDatosTests(unittest.TestCase):
    """Con las rejillas de /mnt/2TB: el `Modelo` nativo frente al de R = 500, solo MUSE."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")
        cls.slug = next(iter(runs_del_paper()))
        try:
            cls.base = modelo_teff(cls.slug, datos="muse")
            cls.nat = modelo_teff(cls.slug, datos="muse", muse_nativo=True)
        except (OSError, SystemExit) as exc:   # sin las rejillas de /mnt/2TB
            raise unittest.SkipTest(f"no se puede construir el modelo: {exc}")

    def test_mismos_parametros_y_el_ir_como_estaba(self):
        self.assertEqual(self.nat.nombres, self.base.nombres)
        np.testing.assert_array_equal(self.nat.lref, self.base.lref)

    def test_canal_a_canal_y_modelo_finito(self):
        s, b = self.nat.segs[0], self.base.segs[0]
        np.testing.assert_allclose(np.diff(s["w"]).min(), 1.25, rtol=1e-3)
        self.assertGreater(s["d"].size, b["d"].size)
        self.assertTrue(np.isfinite(s["lg"]).all())

    def test_una_rejilla_de_2_angstrom_no_vale(self):
        with raiz_temporal(), self.assertRaises(ValueError):
            self.tb.Modelo(self.slug, datos="muse", rejilla="drift", muse_nativo=True)

    def test_ens_vr_sin_ensanchar_ni_mover_es_el_nativo(self):
        lib = modelo_teff(self.slug, datos="muse", muse_nativo=True, ens_vr=(10.0, 100.0))
        self.assertEqual([k for k in lib.nombres if k not in ("ens_A", "vr_kms")], self.nat.nombres)
        np.testing.assert_array_equal(lib.lref, self.nat.lref)
        u = np.full(self.nat.ndim, 0.5)
        th = self.nat.prior_transform(u)
        v = dict(zip(self.nat.nombres, th), ens_A=0.0, vr_kms=0.0)
        m_lib = lib.modelos(np.array([v[k] for k in lib.nombres]))[0]
        m_nat = self.nat.modelos(th)[0]
        # interpolar en log F antes o después de promediar en el canal: < 1 % en el peor canal
        rel = np.abs(m_lib / m_nat - 1)
        self.assertLess(float(rel.max()), 0.01)
        self.assertLess(float(np.median(rel)), 1e-3)
