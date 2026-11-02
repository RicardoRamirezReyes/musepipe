"""`scripts/teff_bayes.py --banda-una-vez` (2026-11-02): cada banda del IR cuenta una vez en la verosimilitud (el
término de cada segmento del IR por 1/N, con N los segmentos de su banda), sin elegir cuál de dos espectros de la
misma banda se usa.

Lo que se prueba son invariantes, no cifras del run: sin la opción todo es lo de siempre (la etiqueta, el json, el
`Modelo`); sin IR falla antes de leer nada; `modelo_de_cadena` lee la opción del json. Con los datos reales: los
parámetros, los datos y el modelo son los mismos; MUSE no lleva peso y los pesos de cada banda del IR suman 1; la
verosimilitud es la suma de los términos de cada segmento por su peso; si cada banda tiene ya un solo segmento (con
`quita_ir`), la verosimilitud es, bit a bit, la de sin la opción.
"""
import argparse
import json
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
#: lo que hay que quitar a cada compañero para que cada banda del IR tenga un solo segmento
UNO_POR_BANDA = {"ROXs12b": ("OSIRIS_J",), "ROXs42Bb": ("NIFS_H", "SINFONI_K")}


class ValidaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_sin_la_opcion_no_falla_nada(self):
        self.tb.valida_banda_una_vez(False)
        self.tb.valida_banda_una_vez(False, datos="muse")

    def test_sin_ir_falla(self):
        with self.assertRaises(ValueError):
            self.tb.valida_banda_una_vez(True, datos="muse")
        self.tb.valida_banda_una_vez(True, datos="ir")
        self.tb.valida_banda_una_vez(True, datos="todo")

    def test_modelo_falla_antes_de_leer_nada(self):
        # un slug que no existe: si validara después de leer, el error sería otro
        with self.assertRaises(ValueError):
            self.tb.Modelo("no_existe", banda_una_vez=True, datos="muse")

    def test_un_modelo_de_new_no_pesa_las_bandas(self):
        self.assertFalse(self.tb.Modelo.__new__(self.tb.Modelo).banda_una_vez)


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
        for a in (self._a(), self._a(banda_una_vez=False)):
            self.assertEqual(self.tb.etiqueta_de(a), "ROXs42Bb_atmo_ceq1500_libre_avprimaria_avfija1.8_soloir_s1")

    def test_la_pieza_va_detras_de_la_inclinacion(self):
        self.assertEqual(self.tb.etiqueta_de(self._a(banda_una_vez=True)),
                         "ROXs42Bb_atmo_ceq1500_libre_avprimaria_avfija1.8_bandaunavez_soloir_s1")
        self.assertEqual(self.tb.etiqueta_de(self._a(banda_una_vez=True, pendiente_ir=0.3, quita_ir=["NIFS_H"])),
                         "ROXs42Bb_atmo_ceq1500_libre_avprimaria_avfija1.8_quitair-NIFS_H_pendir0.3_bandaunavez"
                         "_soloir_s1")


class VarianteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_es_variante_y_no_entra_en_las_tablas(self):
        d = {**PAPER, "banda_una_vez": True}
        self.assertTrue(self.tb._es_variante(d))
        self.assertFalse(self.tb._de_la_tabla(d))
        self.assertFalse(self.tb._de_rejillas(d))
        self.assertFalse(self.tb._es_variante({**PAPER, "banda_una_vez": False}))


class ReconstruyeTests(unittest.TestCase):
    """`modelo_de_cadena` le pasa al `Modelo` lo que dice el json (con un `Modelo` de mentira que lo apunta)."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_lee_banda_una_vez(self):
        visto = {}

        class Falso:
            nombres = ["teff"]

            def __init__(self, *args, **kw):
                visto.update(kw)

        with mock.patch.object(self.tb, "Modelo", Falso):
            self.tb.modelo_de_cadena({**PAPER, "mejor": {"teff": 2600.0}})
            self.assertIs(visto["banda_una_vez"], False)
            self.tb.modelo_de_cadena({**PAPER, "mejor": {"teff": 2600.0}, "banda_una_vez": True})
            self.assertIs(visto["banda_una_vez"], True)


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
        self.assertNotIn("banda_una_vez", d)
        self.assertNotIn("banda_una_vez", _ModeloFalso.visto)

    def test_con_la_opcion_clave_etiqueta_y_argumento(self):
        d = self._json("--banda-una-vez")
        self.assertIs(d["banda_una_vez"], True)
        self.assertEqual(d["etiqueta"], "ROXs42Bb_libre_avprimaria_bandaunavez_soloir_s0")
        self.assertIs(_ModeloFalso.visto["banda_una_vez"], True)
        self.assertLessEqual(set(d), self.tb.CLAVES_JSON | set(self.tb.VARIANTES_JSON))


# ------------------------------------------------------------------ datos reales
def _terminos(mod, th):
    """El término de la verosimilitud de cada segmento, sin peso (la fórmula de `Modelo.loglike`)."""
    out = []
    for s, m in zip(mod.segs, mod.modelos(th)):
        fb = np.exp(th[mod.pos[f"lnf_{s['banda']}"]])
        var = s["e2"] + (fb * m) ** 2
        r = s["d"] - m
        out.append(-0.5 * float(np.sum(r * r / var + np.log(var))))
    return out


@pytest.mark.external_data
@pytest.mark.slow
@requiere("runs", "ir", "bibliotecas")
class ModeloDatosTests(unittest.TestCase):
    """Con BT-Settl de /mnt/2TB: una plantilla con MUSE y el IR, sin y con el peso por banda."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")
        slugs = [s for s in runs_del_paper() if s in UNO_POR_BANDA]
        if not slugs:
            raise unittest.SkipTest("hace falta un compañero con dos espectros de una banda")
        cls.a = slugs[0]
        q = UNO_POR_BANDA[cls.a]
        try:
            cls.sin = modelo_teff(cls.a, plantilla="TWA26", semilla=1)
            cls.con = modelo_teff(cls.a, plantilla="TWA26", semilla=1, banda_una_vez=True)
            cls.sin_q = modelo_teff(cls.a, plantilla="TWA26", semilla=1, quita_ir=q)
            cls.con_q = modelo_teff(cls.a, plantilla="TWA26", semilla=1, quita_ir=q, banda_una_vez=True)
        except (OSError, SystemExit) as exc:   # sin las rejillas de /mnt/2TB
            raise unittest.SkipTest(f"no se puede construir el modelo: {exc}")

    def _puntos(self, mod, n, semilla):
        rng = np.random.default_rng(semilla)
        return [mod.prior_transform(rng.uniform(0.05, 0.95, mod.ndim)) for _ in range(n)]

    def test_los_parametros_los_datos_y_el_modelo_son_los_mismos(self):
        self.assertEqual(self.con.nombres, self.sin.nombres)
        for s, c in zip(self.sin.segs, self.con.segs):
            for k in ("w", "d", "e2", "lg"):
                np.testing.assert_array_equal(c[k], s[k], err_msg=f"{s['nombre']} {k}")
        for th in self._puntos(self.con, 3, 0):
            for mc, ms in zip(self.con.modelos(th), self.sin.modelos(th)):
                np.testing.assert_array_equal(mc, ms)

    def test_muse_sin_peso_y_cada_banda_suma_uno(self):
        self.assertTrue(any(s["banda"] == "vis" for s in self.con.segs))   # MUSE está
        suma = {}
        for s in self.con.segs:
            self.assertEqual("peso" in s, s["banda"] != "vis", s["nombre"])
            if "peso" in s:
                suma[s["banda"]] = suma.get(s["banda"], 0.0) + s["peso"]
        self.assertEqual(set(suma), {b for b in self.con.bandas if b != "vis"})
        for b, v in suma.items():
            self.assertAlmostEqual(v, 1.0, places=12, msg=b)
        # este compañero tiene alguna banda con dos espectros, y sin la opción nadie lleva peso
        self.assertTrue(any(s.get("peso", 1.0) < 1.0 for s in self.con.segs))
        self.assertFalse(any("peso" in s for s in self.sin.segs))

    def test_la_verosimilitud_es_la_suma_pesada_de_los_terminos(self):
        for th in self._puntos(self.con, 5, 1):
            t = _terminos(self.sin, th)
            self.assertAlmostEqual(self.sin.loglike(th), sum(t), delta=1e-9 * abs(sum(t)))
            pesada = sum(x * s.get("peso", 1.0) for x, s in zip(t, self.con.segs))
            self.assertAlmostEqual(self.con.loglike(th), pesada, delta=1e-9 * abs(pesada))
            self.assertNotEqual(self.con.loglike(th), self.sin.loglike(th))

    def test_con_una_por_banda_es_bit_a_bit_lo_de_siempre(self):
        self.assertTrue(all(s["peso"] == 1.0 for s in self.con_q.segs if "peso" in s))
        for th in self._puntos(self.con_q, 5, 2):
            self.assertEqual(self.con_q.loglike(th), self.sin_q.loglike(th))


if __name__ == "__main__":
    unittest.main()
