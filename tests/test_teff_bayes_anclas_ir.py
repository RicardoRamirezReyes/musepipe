"""`scripts/teff_bayes.py --anclas-ir` (2026-10-30): en qué bandas del IR va la escala anclada a la fotometría, para
separar, en la prueba de estabilidad de la calibración con plantillas, qué segmentos lleva cada compañero de qué
escalas lleva ancladas.

Lo que se prueba son invariantes, no cifras del run: sin la opción todo es lo de siempre (la etiqueta, el json, el
`Modelo`); las combinaciones sin sentido fallan antes de leer nada; `anclas_de` ancla y suelta lo que se pide, deja
MUSE como está y, sobre los datos reales, solo suelta; `modelo_de_cadena` lee la opción del json. Con los datos
reales: soltar las anclas cambia solo el prior de esas escalas, no los datos ni la verosimilitud.
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


def _segs(sigmas):
    """Segmentos con solo lo que lee `anclas_de`: nombre y `sigma_escala`."""
    return [dict(nombre=n, sigma_escala=x) for n, x in sigmas]


class ValidaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_sin_la_opcion_no_falla_nada(self):
        self.tb.valida_anclas_ir(None)
        self.tb.valida_anclas_ir(None, datos="muse")

    def test_una_clave_que_no_existe_y_sin_ir(self):
        with self.assertRaises(ValueError):
            self.tb.valida_anclas_ir("JH")
        with self.assertRaises(ValueError):
            self.tb.valida_anclas_ir("K", datos="muse")
        for k in self.tb.ANCLAS_IR:
            self.tb.valida_anclas_ir(k, datos="ir")

    def test_modelo_falla_antes_de_leer_nada(self):
        # un slug que no existe: si validara después de leer, el error sería otro
        for kw in (dict(anclas_ir="JH"), dict(anclas_ir="K", datos="muse")):
            with self.subTest(**kw), self.assertRaises(ValueError):
                self.tb.Modelo("no_existe", **kw)

    def test_un_modelo_de_new_lleva_las_de_siempre(self):
        self.assertIsNone(self.tb.Modelo.__new__(self.tb.Modelo).anclas_ir)


class AnclasDeTests(unittest.TestCase):
    """Un compañero como ROXs 42B b (J, H y K anclados) y otro como ROXs 12 b (solo K), con MUSE."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")
        cls.aj = carga_script("ajuste_conjunto")
        ir, mu = cls.aj.SIGMA_IR, cls.aj.SIGMA_MUSE
        cls.todo = _segs([("MUSE", mu), ("NIFS_J", ir), ("NIFS_H", ir), ("OSIRIS_K", ir), ("SINFONI_K", ir)])
        cls.b_todo = ["vis", "J", "H", "K", "K"]
        cls.solo_k = _segs([("MUSE", mu), ("NIFS_J", None), ("OSIRIS_H", None), ("OSIRIS_K", ir)])
        cls.b_solo_k = ["vis", "J", "H", "K"]

    def test_sin_la_opcion_las_de_siempre(self):
        for segs, b in ((self.todo, self.b_todo), (self.solo_k, self.b_solo_k)):
            self.assertEqual(self.tb.anclas_de(segs, b, None), [s["sigma_escala"] for s in segs])

    def test_suelta_lo_que_no_es_k(self):
        ir, mu = self.aj.SIGMA_IR, self.aj.SIGMA_MUSE
        self.assertEqual(self.tb.anclas_de(self.todo, self.b_todo, "K"), [mu, None, None, ir, ir])

    def test_ninguna_suelta_todo_el_ir_y_muse_queda(self):
        mu = self.aj.SIGMA_MUSE
        self.assertEqual(self.tb.anclas_de(self.todo, self.b_todo, "ninguna"), [mu, None, None, None, None])
        self.assertEqual(self.tb.anclas_de(self.solo_k, self.b_solo_k, "ninguna"), [mu, None, None, None])

    def test_anclar_pide_datos_de_prueba(self):
        ir, mu = self.aj.SIGMA_IR, self.aj.SIGMA_MUSE
        with self.assertRaises(ValueError):
            self.tb.anclas_de(self.solo_k, self.b_solo_k, "JHK")
        self.assertEqual(self.tb.anclas_de(self.solo_k, self.b_solo_k, "JHK", prueba=True), [mu, ir, ir, ir])

    def test_las_de_siempre_con_la_opcion_fallan(self):
        for segs, b, k in ((self.todo, self.b_todo, "JHK"), (self.solo_k, self.b_solo_k, "K")):
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.tb.anclas_de(segs, b, k, prueba=True)

    def test_no_muta_los_segmentos(self):
        antes = [dict(s) for s in self.todo]
        self.tb.anclas_de(self.todo, self.b_todo, "ninguna")
        self.assertEqual(self.todo, antes)


class EtiquetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def _a(self, **kw):
        base = dict(etiqueta=None, slug="ROXs42Bb", rejilla="atmo_ceq1500", modo="libre", tracks="ATMO2020",
                    av="primaria", plantilla="TWA26", av_fija=None, inyecta_bt=None, datos="todo", r250=False,
                    semilla=0)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_sin_la_opcion_la_de_siempre(self):
        for a in (self._a(), self._a(anclas_ir=None)):
            self.assertEqual(self.tb.etiqueta_de(a), "ROXs42Bb_atmo_ceq1500_libre_avprimaria_pl-TWA26_s0")

    def test_la_pieza_va_detras_del_ruido(self):
        self.assertEqual(self.tb.etiqueta_de(self._a(anclas_ir="K", datos="ir")),
                         "ROXs42Bb_atmo_ceq1500_libre_avprimaria_pl-TWA26_anclas-K_soloir_s0")
        self.assertEqual(self.tb.etiqueta_de(self._a(anclas_ir="ninguna", ruido_de="ROXs12b")),
                         "ROXs42Bb_atmo_ceq1500_libre_avprimaria_pl-TWA26_ruido-ROXs12b_anclas-ninguna_s0")


class VarianteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_es_variante_y_no_entra_en_las_tablas(self):
        for k in self.tb.ANCLAS_IR:
            d = {**PAPER, "anclas_ir": k}
            self.assertTrue(self.tb._es_variante(d), k)
            self.assertFalse(self.tb._de_la_tabla(d), k)
            self.assertFalse(self.tb._de_rejillas(d), k)
        self.assertFalse(self.tb._es_variante({**PAPER, "anclas_ir": None}))


class ReconstruyeTests(unittest.TestCase):
    """`modelo_de_cadena` le pasa al `Modelo` lo que dice el json (con un `Modelo` de mentira que lo apunta)."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_lee_anclas_ir(self):
        visto = {}

        class Falso:
            nombres = ["teff"]

            def __init__(self, *args, **kw):
                visto.update(kw)

        with mock.patch.object(self.tb, "Modelo", Falso):
            self.tb.modelo_de_cadena({**PAPER, "mejor": {"teff": 2600.0}})
            self.assertIsNone(visto["anclas_ir"])
            self.tb.modelo_de_cadena({**PAPER, "mejor": {"teff": 2600.0}, "anclas_ir": "ninguna"})
            self.assertEqual(visto["anclas_ir"], "ninguna")


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
        self.verdad = dict(plantilla="TWA26", spt="M9", lib="manara", av=1.8)

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
            self.tb.main(["ajusta", "ROXs42Bb", "--plantilla", "TWA26", "--datos", "ir", "--salida", t, *opciones])
            (f,) = Path(t).glob("*.json")
            return json.loads(f.read_text())

    def test_sin_la_opcion_ni_clave_ni_argumento(self):
        d = self._json()
        self.assertNotIn("anclas_ir", d)
        self.assertNotIn("anclas_ir", _ModeloFalso.visto)

    def test_con_la_opcion_clave_etiqueta_y_argumento(self):
        d = self._json("--anclas-ir", "K")
        self.assertEqual(d["anclas_ir"], "K")
        self.assertEqual(d["etiqueta"], "ROXs42Bb_libre_avprimaria_pl-TWA26_anclas-K_soloir_s0")
        self.assertEqual(_ModeloFalso.visto["anclas_ir"], "K")
        self.assertLessEqual(set(d), self.tb.CLAVES_JSON | set(self.tb.VARIANTES_JSON))

    def test_una_clave_que_no_existe_no_pasa_de_argparse(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            self._json("--anclas-ir", "JH")


# ------------------------------------------------------------------ datos reales
@pytest.mark.external_data
@pytest.mark.slow
@requiere("runs", "ir", "bibliotecas")
class ModeloDatosTests(unittest.TestCase):
    """Con BT-Settl de /mnt/2TB: una plantilla solo con el IR, con las anclas de siempre y con todas sueltas."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")
        cls.aj = carga_script("ajuste_conjunto")
        slugs = list(runs_del_paper())
        if not slugs:
            raise unittest.SkipTest("hace falta un compañero")
        cls.a = slugs[0]
        try:
            cls.propio = modelo_teff(cls.a, plantilla="TWA26", semilla=1, datos="ir")
            cls.suelto = modelo_teff(cls.a, plantilla="TWA26", semilla=1, datos="ir", anclas_ir="ninguna")
        except (OSError, SystemExit) as exc:   # sin las rejillas de /mnt/2TB
            raise unittest.SkipTest(f"no se puede construir el modelo: {exc}")

    def test_mismos_datos_y_parametros(self):
        self.assertEqual(self.suelto.nombres, self.propio.nombres)
        self.assertEqual(self.suelto.verdad, self.propio.verdad)
        for s, c in zip(self.propio.segs, self.suelto.segs):
            for k in ("w", "d", "e2", "lg"):
                np.testing.assert_array_equal(c[k], s[k], err_msg=f"{s['nombre']} {k}")

    def test_suelta_todas_y_habia_alguna(self):
        self.assertTrue(all(s["sigma"] is None for s in self.suelto.segs))
        self.assertTrue(any(s["sigma"] == self.aj.SIGMA_IR for s in self.propio.segs))

    def test_solo_cambia_el_prior_de_las_escalas_sueltas(self):
        rng = np.random.default_rng(0)
        sueltas = {self.propio.pos[f"s_{s['nombre']}"] for s in self.propio.segs if s["sigma"] is not None}
        for _ in range(5):
            u = rng.uniform(0.05, 0.95, self.propio.ndim)
            xp, xs = self.propio.prior_transform(u), self.suelto.prior_transform(u)
            iguales = [i for i in range(self.propio.ndim) if i not in sueltas]
            np.testing.assert_array_equal(xs[iguales], xp[iguales])
            self.assertTrue(np.all(xs[sorted(sueltas)] != xp[sorted(sueltas)]))
            self.assertEqual(self.suelto.loglike(xp), self.propio.loglike(xp))


if __name__ == "__main__":
    unittest.main()
