"""`scripts/teff_bayes.py --ruido-de` (2026-10-30): los datos de prueba con la cobertura de un compañero y la S/N por
píxel del otro, para separar, en la prueba de estabilidad de la calibración con plantillas, la cobertura del ruido.

Lo que se prueba son invariantes, no cifras del run: sin la opción todo es lo de siempre (la etiqueta, el json, el
`Modelo`); las combinaciones sin sentido fallan antes de leer nada; `snr_cruzada` toma la S/N del mismo instrumento o,
si falta, la mediana de la banda, y no cambia la cobertura; `modelo_de_cadena` lee la opción del json. Con los datos
reales: la cobertura no cambia, el MUSE de un compañero lleva la S/N del otro tal cual (los bins son los mismos) y,
con la misma semilla, el error de cada píxel cambia exactamente en el cociente de las dos S/N.
"""
import argparse
import json
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


def _seg(nombre, w, sn, bueno=None, brazo="nir"):
    """Un segmento con flujo 1 y la S/N `sn` (e = 1/sn; sn ≤ 0 o NaN → e NaN)."""
    w, sn = np.asarray(w, float), np.asarray(sn, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(sn > 0, 1.0 / sn, np.nan)
    return dict(nombre=nombre, brazo=brazo, w=w, f=np.ones(w.size), e=e,
                bueno=np.ones(w.size, bool) if bueno is None else np.asarray(bueno, bool))


class ValidaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_sin_la_opcion_no_falla_nada(self):
        self.tb.valida_ruido_de(None, "ROXs12b")

    def test_pide_datos_de_prueba_y_otro_objeto(self):
        for kw in (dict(), dict(plantilla=None, inyecta_bt=None)):
            with self.subTest(**kw), self.assertRaises(ValueError):
                self.tb.valida_ruido_de("ROXs42Bb", "ROXs12b", **kw)
        with self.assertRaises(ValueError):
            self.tb.valida_ruido_de("ROXs12b", "ROXs12b", plantilla="TWA26")
        self.tb.valida_ruido_de("ROXs42Bb", "ROXs12b", plantilla="TWA26")
        self.tb.valida_ruido_de("ROXs42Bb", "ROXs12b", inyecta_bt=(2600.0, 4.0, 1.8, 2.0))

    def test_modelo_falla_antes_de_leer_nada(self):
        # un slug que no existe: si validara después de leer, el error sería otro
        with self.assertRaises(ValueError):
            self.tb.Modelo("no_existe", ruido_de="ROXs42Bb")

    def test_un_modelo_de_new_no_cruza_el_ruido(self):
        self.assertIsNone(self.tb.Modelo.__new__(self.tb.Modelo).ruido_de)


class SnrCruzadaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_el_mismo_instrumento_manda(self):
        w = np.linspace(1.0e4, 1.1e4, 11)
        a = [_seg("X_J", w, np.full(11, 50.0))]
        b = [_seg("X_J", w, np.full(11, 20.0)), _seg("Y_J", w, np.full(11, 90.0))]
        (sn,) = self.tb.snr_cruzada(a, ["J"], b, ["J", "J"])
        np.testing.assert_array_equal(sn, 20.0)

    def test_sin_el_instrumento_la_mediana_de_la_banda(self):
        w = np.linspace(1.0e4, 1.1e4, 11)
        a = [_seg("Z_J", w, np.full(11, 50.0))]
        b = [_seg("X_J", w, np.full(11, 20.0)), _seg("Y_J", w, np.full(11, 90.0)), _seg("X_H", w, np.full(11, 1.0))]
        (sn,) = self.tb.snr_cruzada(a, ["J"], b, ["J", "J", "H"])
        np.testing.assert_allclose(sn, 55.0)

    def test_fuera_de_la_cobertura_del_otro(self):
        """Donde solo uno de los del otro cubre, el suyo; donde ninguno, el más cercano (constante en su borde)."""
        w = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        a = [_seg("Z_J", w, np.full(7, 50.0))]
        b = [_seg("X_J", [1.0, 2.0], [10.0, 20.0]), _seg("Y_J", [2.0, 3.0], [40.0, 60.0])]
        (sn,) = self.tb.snr_cruzada(a, ["J"], b, ["J", "J"])
        np.testing.assert_allclose(sn, [10.0, 10.0, 30.0, 60.0, 60.0, 60.0, 60.0])

    def test_interpola_en_lambda(self):
        a = [_seg("X_J", [1.5, 2.5], [1.0, 1.0])]
        b = [_seg("X_J", [1.0, 2.0, 3.0], [10.0, 20.0, 40.0])]
        (sn,) = self.tb.snr_cruzada(a, ["J"], b, ["J"])
        np.testing.assert_allclose(sn, [15.0, 30.0])

    def test_la_cobertura_no_cambia(self):
        """Los píxeles que este no usa (S/N NaN o ≤ 0) siguen fuera; los malos del otro no cuentan."""
        w = np.arange(5.0)
        a = [_seg("X_J", w, [50.0, np.nan, 50.0, 0.0, 50.0])]
        b = [_seg("X_J", w, [10.0, 10.0, 999.0, 10.0, 10.0], bueno=[True, True, False, True, True])]
        (sn,) = self.tb.snr_cruzada(a, ["J"], b, ["J"])
        np.testing.assert_array_equal(np.isfinite(sn), [True, False, True, False, True])
        np.testing.assert_allclose(sn[[0, 2, 4]], 10.0)

    def test_consigo_mismo_es_su_propia_snr(self):
        rng = np.random.default_rng(3)
        segs = [_seg("X_J", np.linspace(1.1e4, 1.3e4, 40), rng.uniform(20, 80, 40)),
                _seg("MUSE", np.linspace(6300, 9200, 84), rng.uniform(2, 30, 84), brazo="vis")]
        for s, sn in zip(segs, self.tb.snr_cruzada(segs, ["J", "vis"], segs, ["J", "vis"])):
            np.testing.assert_array_equal(sn, np.abs(s["f"]) / s["e"])

    def test_una_banda_que_el_otro_no_tiene(self):
        a = [_seg("X_K", [2.0e4, 2.1e4], [50.0, 50.0])]
        b = [_seg("X_J", [1.1e4, 1.2e4], [50.0, 50.0])]
        with self.assertRaises(ValueError):
            self.tb.snr_cruzada(a, ["K"], b, ["J"])


class EtiquetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def _a(self, **kw):
        base = dict(etiqueta=None, slug="ROXs12b", rejilla="atmo_ceq1500", modo="libre", tracks="ATMO2020",
                    av="primaria", plantilla="TWA26", av_fija=None, inyecta_bt=None, datos="todo", r250=False,
                    semilla=0)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_sin_la_opcion_la_de_siempre(self):
        for a in (self._a(), self._a(ruido_de=None)):
            self.assertEqual(self.tb.etiqueta_de(a), "ROXs12b_atmo_ceq1500_libre_avprimaria_pl-TWA26_s0")

    def test_la_pieza_va_detras_de_los_datos_de_prueba(self):
        self.assertEqual(self.tb.etiqueta_de(self._a(ruido_de="ROXs42Bb")),
                         "ROXs12b_atmo_ceq1500_libre_avprimaria_pl-TWA26_ruido-ROXs42Bb_s0")
        self.assertEqual(self.tb.etiqueta_de(self._a(ruido_de="ROXs42Bb", datos="ir")),
                         "ROXs12b_atmo_ceq1500_libre_avprimaria_pl-TWA26_ruido-ROXs42Bb_soloir_s0")


class VarianteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_es_variante_y_no_entra_en_las_tablas(self):
        d = {**PAPER, "ruido_de": "ROXs42Bb"}
        self.assertTrue(self.tb._es_variante(d))
        self.assertFalse(self.tb._de_la_tabla(d))
        self.assertFalse(self.tb._de_rejillas(d))
        self.assertFalse(self.tb._es_variante({**PAPER, "ruido_de": None}))


class ReconstruyeTests(unittest.TestCase):
    """`modelo_de_cadena` le pasa al `Modelo` lo que dice el json (con un `Modelo` de mentira que lo apunta)."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_lee_ruido_de(self):
        visto = {}

        class Falso:
            nombres = ["teff"]

            def __init__(self, *args, **kw):
                visto.update(kw)

        with mock.patch.object(self.tb, "Modelo", Falso):
            self.tb.modelo_de_cadena({**PAPER, "mejor": {"teff": 2600.0}})
            self.assertIsNone(visto["ruido_de"])
            self.tb.modelo_de_cadena({**PAPER, "mejor": {"teff": 2600.0}, "ruido_de": "ROXs42Bb"})
            self.assertEqual(visto["ruido_de"], "ROXs42Bb")


class _ModeloFalso:
    """Lo que `ajusta` lee de un `Modelo`, sin datos; apunta los argumentos por nombre."""
    visto = {}

    def __init__(self, *args, **kw):
        type(self).visto = dict(kw)
        self.run, self.r_analisis, self.rellenados, self.degrada = "RUN", 500, [], None
        self.rk_fijo, self.td_rango = None, None
        self.T0, self.dT, self.nT, self.G0, self.dG, self.nG = 2000.0, 100.0, 13, 2.5, 0.5, 6
        self.nombres, self.ndim = ["teff", "logg", "av"], 3
        self.dir_rejilla = Path("/rejilla")
        self.segs = [dict(nombre="MUSE", d=np.zeros(84))]
        self.verdad = dict(plantilla="TWA26", spt="M9", lib="manara", av=1.8)

    def chi2(self, th, inflado=False):
        return {"MUSE": 1.0}


def _corre_falso(mod, nlive, semilla):
    m = np.ones((10, mod.ndim))
    return dict(segundos=0.0, ncall=1, logz=-1.0, logz_err=0.1, muestras=m, derivadas=np.ones((10, 5)),
                mejor=m[0], logl_max=-1.0)


class AjustaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def _json(self, *opciones):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(self.tb, "Modelo", _ModeloFalso), \
                mock.patch.object(self.tb, "corre", _corre_falso):
            self.tb.main(["ajusta", "ROXs12b", "--plantilla", "TWA26", "--salida", t, *opciones])
            (f,) = Path(t).glob("*.json")
            return json.loads(f.read_text())

    def test_sin_la_opcion_ni_clave_ni_argumento(self):
        d = self._json()
        self.assertNotIn("ruido_de", d)
        self.assertNotIn("ruido_de", _ModeloFalso.visto)

    def test_con_la_opcion_clave_etiqueta_y_argumento(self):
        d = self._json("--ruido-de", "ROXs42Bb")
        self.assertEqual(d["ruido_de"], "ROXs42Bb")
        self.assertEqual(d["etiqueta"], "ROXs12b_libre_avprimaria_pl-TWA26_ruido-ROXs42Bb_s0")
        self.assertEqual(_ModeloFalso.visto["ruido_de"], "ROXs42Bb")
        self.assertLessEqual(set(d), self.tb.CLAVES_JSON | set(self.tb.VARIANTES_JSON))


# ------------------------------------------------------------------ datos reales
def _bandas(slug, segs):
    from scripts import tipo_ir_indices as tir
    ir = tir.perfil(slug)["ir_spectra"]["spectra"]
    return ["vis" if s["brazo"] == "vis" else ir[s["nombre"]]["band"] for s in segs]


@pytest.mark.external_data
@requiere("runs", "ir", "bibliotecas")
class SnrDatosTests(unittest.TestCase):
    """Sin rejillas: los segmentos de los dos compañeros del paper (`_segmentos`)."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")
        slugs = list(runs_del_paper())
        if len(slugs) < 2:
            raise unittest.SkipTest("hacen falta dos compañeros")
        cls.a, cls.b = slugs[:2]
        with raiz_temporal():
            cls.sa = cls.tb._segmentos(cls.a)[1]
            cls.sb = cls.tb._segmentos(cls.b)[1]
        cls.ba, cls.bb = _bandas(cls.a, cls.sa), _bandas(cls.b, cls.sb)

    def _propia(self, s):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(s["f"]) / s["e"]

    def test_la_cobertura_no_cambia(self):
        for (x, bx), (y, by) in (((self.sa, self.ba), (self.sb, self.bb)), ((self.sb, self.bb), (self.sa, self.ba))):
            for s, sn in zip(x, self.tb.snr_cruzada(x, bx, y, by)):
                p = self._propia(s)
                np.testing.assert_array_equal(np.isfinite(sn), np.isfinite(p) & (p > 0), s["nombre"])

    def test_el_mismo_instrumento_queda_en_su_rango(self):
        cruz = self.tb.snr_cruzada(self.sa, self.ba, self.sb, self.bb)
        otros = {o["nombre"]: o for o in self.sb}
        n = 0
        for s, sn in zip(self.sa, cruz):
            if s["nombre"] not in otros:
                continue
            o = otros[s["nombre"]]
            po = self._propia(o)[o["bueno"]]
            ok = np.isfinite(sn)
            self.assertGreaterEqual(float(sn[ok].min()), float(np.nanmin(po)) * (1 - 1e-12), s["nombre"])
            self.assertLessEqual(float(sn[ok].max()), float(np.nanmax(po)) * (1 + 1e-12), s["nombre"])
            n += 1
        self.assertGreater(n, 0)

    def test_muse_lleva_la_snr_del_otro(self):
        """Los bins de 15 Å de MUSE son los mismos en los dos (los centros, a 0.2 Å): la S/N es la del otro, salvo
        lo que mueve interpolar en ese desfase."""
        (ka,) = [k for k, s in enumerate(self.sa) if s["brazo"] == "vis"]
        (kb,) = [k for k, s in enumerate(self.sb) if s["brazo"] == "vis"]
        a, b = self.sa[ka], self.sb[kb]
        self.assertEqual(a["w"].size, b["w"].size)
        self.assertLess(float(np.abs(a["w"] - b["w"]).max()), 1.0)
        sn = self.tb.snr_cruzada(self.sa, self.ba, self.sb, self.bb)[ka]
        ok = np.isfinite(sn) & b["bueno"]
        self.assertGreater(int(ok.sum()), 0)
        self.assertLess(float(np.max(np.abs(sn[ok] / self._propia(b)[ok] - 1))), 0.05)


@pytest.mark.external_data
@pytest.mark.slow
@requiere("runs", "ir", "bibliotecas")
class ModeloDatosTests(unittest.TestCase):
    """Con BT-Settl de /mnt/2TB: una plantilla con el ruido de su compañero y con el del otro, la misma semilla."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")
        slugs = list(runs_del_paper())
        if len(slugs) < 2:
            raise unittest.SkipTest("hacen falta dos compañeros")
        cls.a, cls.b = slugs[:2]
        try:
            cls.propio = modelo_teff(cls.a, plantilla="TWA26", semilla=1)
            cls.cruzado = modelo_teff(cls.a, plantilla="TWA26", semilla=1, ruido_de=cls.b)
        except (OSError, SystemExit) as exc:   # sin las rejillas de /mnt/2TB
            raise unittest.SkipTest(f"no se puede construir el modelo: {exc}")

    def test_misma_cobertura_y_parametros(self):
        self.assertEqual(self.cruzado.nombres, self.propio.nombres)
        self.assertEqual(self.cruzado.verdad, self.propio.verdad)
        for s, c in zip(self.propio.segs, self.cruzado.segs):
            np.testing.assert_array_equal(c["w"], s["w"])
            np.testing.assert_array_equal(c["lg"], s["lg"])

    def test_misma_plantilla_mismas_normales_y_el_error_de_la_otra_snr(self):
        """e = |plantilla| / S/N, así que la plantilla sin ruido es e · S/N: la misma en las dos. Con la misma semilla,
        las dos llevan las mismas normales tipificadas, (d − plantilla) / e."""
        with raiz_temporal():
            segs_a = self.tb._segmentos(self.a)[1]
            segs_b = self.tb._segmentos(self.b)[1]
        cruz = dict(zip((s["nombre"] for s in segs_a),
                        self.tb.snr_cruzada(segs_a, _bandas(self.a, segs_a), segs_b, _bandas(self.b, segs_b))))
        for s, c in zip(self.propio.segs, self.cruzado.segs):
            o = next(x for x in segs_a if x["nombre"] == s["nombre"])
            i = np.searchsorted(o["w"], s["w"])
            np.testing.assert_array_equal(o["w"][i], s["w"])
            with np.errstate(divide="ignore", invalid="ignore"):
                propia = np.abs(o["f"][i]) / o["e"][i]
            ep, ec = np.sqrt(s["e2"]), np.sqrt(c["e2"])
            pl_p, pl_c = ep * propia, ec * cruz[s["nombre"]][i]
            np.testing.assert_allclose(pl_c, pl_p, rtol=1e-12, err_msg=s["nombre"])
            np.testing.assert_allclose((c["d"] - pl_p) / ec, (s["d"] - pl_p) / ep, rtol=1e-6, atol=1e-9,
                                       err_msg=s["nombre"])


if __name__ == "__main__":
    unittest.main()
