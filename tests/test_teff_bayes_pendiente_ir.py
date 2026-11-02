"""`scripts/teff_bayes.py --pendiente-ir` (2026-11-01): una inclinación libre por segmento del IR, para que el ajuste
use la forma de cada segmento y no su inclinación (dos espectros de la misma banda difieren en inclinación un 7–11 %).

Lo que se prueba son invariantes, no cifras del run: sin la opción todo es lo de siempre (la etiqueta, el json, el
`Modelo`); los anchos sin sentido fallan antes de leer nada; `modelo_de_cadena` lee la opción del json. Con los datos
reales: los parámetros nuevos son uno por segmento del IR (MUSE no lleva), detrás de las escalas; con todas las
inclinaciones en cero el modelo y la verosimilitud son, bit a bit, los de sin la opción; con otras, el modelo de cada
segmento es el de sin la opción por 1 + p·xp; el prior cubre ±ancho; un ancho que anularía el factor falla.
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

from tests._datos import carga_script, modelo_teff, requiere, runs_del_paper

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
        self.tb.valida_pendiente_ir(None)
        self.tb.valida_pendiente_ir(None, datos="muse")

    def test_anchos_sin_sentido_y_sin_ir(self):
        for p in (0.0, -0.1, math.nan, math.inf):
            with self.subTest(p=p), self.assertRaises(ValueError):
                self.tb.valida_pendiente_ir(p)
        with self.assertRaises(ValueError):
            self.tb.valida_pendiente_ir(0.1, datos="muse")
        self.tb.valida_pendiente_ir(0.1, datos="ir")

    def test_modelo_falla_antes_de_leer_nada(self):
        # un slug que no existe: si validara después de leer, el error sería otro
        for kw in (dict(pendiente_ir=0.0), dict(pendiente_ir=0.1, datos="muse")):
            with self.subTest(**kw), self.assertRaises(ValueError):
                self.tb.Modelo("no_existe", **kw)

    def test_un_modelo_de_new_no_lleva_inclinacion(self):
        self.assertIsNone(self.tb.Modelo.__new__(self.tb.Modelo).pendiente_ir)


class EtiquetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def _a(self, **kw):
        base = dict(etiqueta=None, slug="ROXs42Bb", rejilla="atmo_ceq1500", modo="libre", tracks="ATMO2020",
                    av="primaria", plantilla=None, av_fija=1.8, inyecta_bt=None, datos="ir", r250=False,
                    semilla=1)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_sin_la_opcion_la_de_siempre(self):
        for a in (self._a(), self._a(pendiente_ir=None)):
            self.assertEqual(self.tb.etiqueta_de(a), "ROXs42Bb_atmo_ceq1500_libre_avprimaria_avfija1.8_soloir_s1")

    def test_la_pieza_va_detras_de_los_quitados(self):
        self.assertEqual(self.tb.etiqueta_de(self._a(pendiente_ir=0.1)),
                         "ROXs42Bb_atmo_ceq1500_libre_avprimaria_avfija1.8_pendir0.1_soloir_s1")
        self.assertEqual(self.tb.etiqueta_de(self._a(pendiente_ir=0.1, quita_ir=["NIFS_H"])),
                         "ROXs42Bb_atmo_ceq1500_libre_avprimaria_avfija1.8_quitair-NIFS_H_pendir0.1_soloir_s1")


class VarianteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_es_variante_y_no_entra_en_las_tablas(self):
        d = {**PAPER, "pendiente_ir": 0.1}
        self.assertTrue(self.tb._es_variante(d))
        self.assertFalse(self.tb._de_la_tabla(d))
        self.assertFalse(self.tb._de_rejillas(d))
        self.assertFalse(self.tb._es_variante({**PAPER, "pendiente_ir": None}))


class ReconstruyeTests(unittest.TestCase):
    """`modelo_de_cadena` le pasa al `Modelo` lo que dice el json (con un `Modelo` de mentira que lo apunta)."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_lee_pendiente_ir(self):
        visto = {}

        class Falso:
            nombres = ["teff"]

            def __init__(self, *args, **kw):
                visto.update(kw)

        with mock.patch.object(self.tb, "Modelo", Falso):
            self.tb.modelo_de_cadena({**PAPER, "mejor": {"teff": 2600.0}})
            self.assertIsNone(visto["pendiente_ir"])
            self.tb.modelo_de_cadena({**PAPER, "mejor": {"teff": 2600.0}, "pendiente_ir": 0.1})
            self.assertEqual(visto["pendiente_ir"], 0.1)


class _ModeloFalso:
    """Lo que `ajusta` lee de un `Modelo`, sin datos; apunta los argumentos por nombre."""
    visto = {}

    def __init__(self, *args, **kw):
        type(self).visto = dict(kw)
        self.run, self.r_analisis, self.rellenados, self.degrada = "RUN", 500, [], None
        self.rk_fijo, self.td_rango = None, None
        self.pendiente_ir = kw.get("pendiente_ir")
        self.T0, self.dT, self.nT, self.G0, self.dG, self.nG = 2000.0, 100.0, 13, 2.5, 0.5, 6
        self.nombres, self.ndim = ["teff", "logg", "av"], 3
        self.dir_rejilla = Path("/rejilla")
        self.segs = [dict(nombre="OSIRIS_K", d=np.zeros(84))]

    def chi2(self, th, inflado=False):
        return {"OSIRIS_K": 1.0}


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
            self.tb.main(["ajusta", "ROXs42Bb", "--datos", "ir", "--salida", t, *opciones])
            (f,) = Path(t).glob("*.json")
            return json.loads(f.read_text())

    def test_sin_la_opcion_ni_clave_ni_argumento(self):
        d = self._json()
        self.assertNotIn("pendiente_ir", d)
        self.assertNotIn("pendiente_ir", _ModeloFalso.visto)

    def test_con_la_opcion_clave_etiqueta_y_argumento(self):
        d = self._json("--pendiente-ir", "0.1")
        self.assertEqual(d["pendiente_ir"], 0.1)
        self.assertEqual(d["etiqueta"], "ROXs42Bb_libre_avprimaria_pendir0.1_soloir_s0")
        self.assertEqual(_ModeloFalso.visto["pendiente_ir"], 0.1)
        self.assertLessEqual(set(d), self.tb.CLAVES_JSON | set(self.tb.VARIANTES_JSON))


# ------------------------------------------------------------------ datos reales
@pytest.mark.external_data
@pytest.mark.slow
@requiere("runs", "ir", "bibliotecas")
class ModeloDatosTests(unittest.TestCase):
    """Con BT-Settl de /mnt/2TB: una plantilla con MUSE y el IR, sin y con la inclinación."""

    ANCHO = 0.1

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")
        slugs = list(runs_del_paper())
        if not slugs:
            raise unittest.SkipTest("hace falta un compañero")
        cls.a = slugs[0]
        try:
            cls.sin = modelo_teff(cls.a, plantilla="TWA26", semilla=1)
            cls.con = modelo_teff(cls.a, plantilla="TWA26", semilla=1, pendiente_ir=cls.ANCHO)
        except (OSError, SystemExit) as exc:   # sin las rejillas de /mnt/2TB
            raise unittest.SkipTest(f"no se puede construir el modelo: {exc}")
        cls.ir = [s["nombre"] for s in cls.sin.segs if s["banda"] != "vis"]

    def _par(self, rng, p=None):
        """Un punto del prior del modelo con la opción (inclinaciones `p`, o del prior) y el mismo sin ellas."""
        xc = self.con.prior_transform(rng.uniform(0.05, 0.95, self.con.ndim))
        if p is not None:
            for n in self.ir:
                xc[self.con.pos[f"p_{n}"]] = p
        xs = np.array([xc[self.con.pos[n]] for n in self.sin.nombres])
        return xc, xs

    def test_una_inclinacion_por_segmento_del_ir_detras_de_las_escalas(self):
        self.assertTrue(self.ir)
        self.assertTrue(any(s["banda"] == "vis" for s in self.sin.segs))   # MUSE está y no lleva
        nuevos = [f"p_{n}" for n in self.ir]
        self.assertEqual([n for n in self.con.nombres if n not in self.sin.nombres], nuevos)
        ultima_escala = max(i for i, n in enumerate(self.con.nombres) if n.startswith("s_"))
        primera_f = min(i for i, n in enumerate(self.con.nombres) if n.startswith("lnf_"))
        self.assertEqual(self.con.nombres[ultima_escala + 1:primera_f], nuevos)

    def test_los_datos_son_los_mismos(self):
        for s, c in zip(self.sin.segs, self.con.segs):
            for k in ("w", "d", "e2", "lg"):
                np.testing.assert_array_equal(c[k], s[k], err_msg=f"{s['nombre']} {k}")
            self.assertEqual("xp" in c, s["banda"] != "vis", s["nombre"])

    def test_la_abscisa_va_centrada_en_el_segmento(self):
        for s in self.con.segs:
            if "xp" in s:
                self.assertAlmostEqual(s["xp"].min(), -s["xp"].max(), places=9)
                np.testing.assert_allclose(np.diff(s["xp"]), np.diff(s["w"]) / 1000.0)

    def test_con_inclinacion_cero_es_bit_a_bit_lo_de_siempre(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            xc, xs = self._par(rng, p=0.0)
            for mc, ms in zip(self.con.modelos(xc), self.sin.modelos(xs)):
                np.testing.assert_array_equal(mc, ms)
            self.assertEqual(self.con.loglike(xc), self.sin.loglike(xs))

    def test_con_inclinacion_el_modelo_se_multiplica_por_el_factor(self):
        rng = np.random.default_rng(1)
        for _ in range(5):
            xc, xs = self._par(rng)
            for s, mc, ms in zip(self.con.segs, self.con.modelos(xc), self.sin.modelos(xs)):
                f = 1.0 + xc[self.con.pos[f"p_{s['nombre']}"]] * s["xp"] if "xp" in s else 1.0
                np.testing.assert_allclose(mc, ms * f, rtol=1e-12, err_msg=s["nombre"])

    def test_el_prior_cubre_mas_menos_el_ancho(self):
        for u, esperado in ((0.0, -self.ANCHO), (0.5, 0.0), (1.0, self.ANCHO)):
            x = self.con.prior_transform(np.full(self.con.ndim, u))
            for n in self.ir:
                self.assertAlmostEqual(x[self.con.pos[f"p_{n}"]], esperado, places=12)

    def test_un_ancho_que_anula_el_factor_falla(self):
        with self.assertRaises(ValueError):
            modelo_teff(self.a, plantilla="TWA26", semilla=1, pendiente_ir=5.0)


if __name__ == "__main__":
    unittest.main()
