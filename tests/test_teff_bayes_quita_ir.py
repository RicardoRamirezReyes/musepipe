"""`scripts/teff_bayes.py --quita-ir` (2026-10-31): qué segmentos del IR salen de la verosimilitud, para saber, en la
prueba de estabilidad de la calibración con plantillas, cuál de los que lleva un compañero y no el otro mueve la T_eff.

Lo que se prueba son invariantes, no cifras del run: sin la opción todo es lo de siempre (la etiqueta, el json, el
`Modelo`); las combinaciones sin sentido fallan (antes de leer nada, las que no piden datos); `quitados_de` acepta
solo segmentos del IR del objeto y deja al menos uno; `modelo_de_cadena` lee la opción del json. Con los datos
reales: los segmentos que quedan llevan los mismos datos que sin la opción, y la verosimilitud pierde justo el
término del segmento quitado.
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


def _segs(*nombres):
    """Segmentos con solo lo que lee `quitados_de`: nombre y brazo (MUSE es el visible)."""
    return [dict(nombre=n, brazo="vis" if n == "MUSE" else "nir") for n in nombres]


class ValidaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_sin_la_opcion_no_falla_nada(self):
        self.tb.valida_quita_ir(None)
        self.tb.valida_quita_ir(None, datos="muse")

    def test_vacia_y_sin_ir(self):
        with self.assertRaises(ValueError):
            self.tb.valida_quita_ir([])
        with self.assertRaises(ValueError):
            self.tb.valida_quita_ir(["NIFS_H"], datos="muse")
        self.tb.valida_quita_ir(["NIFS_H"], datos="ir")

    def test_modelo_falla_antes_de_leer_nada(self):
        # un slug que no existe: si validara después de leer, el error sería otro
        for kw in (dict(quita_ir=[]), dict(quita_ir=["NIFS_H"], datos="muse")):
            with self.subTest(**kw), self.assertRaises(ValueError):
                self.tb.Modelo("no_existe", **kw)

    def test_un_modelo_de_new_los_lleva_todos(self):
        self.assertIsNone(self.tb.Modelo.__new__(self.tb.Modelo).quita_ir)


class QuitadosDeTests(unittest.TestCase):
    """Un compañero como ROXs 42B b: MUSE y cinco segmentos del IR."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")
        cls.segs = _segs("MUSE", "NIFS_J", "NIFS_H", "OSIRIS_K", "OSIRIS_H", "SINFONI_K")

    def test_sin_la_opcion_ninguno(self):
        self.assertEqual(self.tb.quitados_de(self.segs, None), frozenset())

    def test_quita_los_que_se_piden(self):
        self.assertEqual(self.tb.quitados_de(self.segs, ["NIFS_H"]), {"NIFS_H"})
        self.assertEqual(self.tb.quitados_de(self.segs, ("SINFONI_K", "NIFS_H", "NIFS_H")), {"NIFS_H", "SINFONI_K"})

    def test_solo_segmentos_del_ir_del_objeto(self):
        for q in (["OSIRIS_J"], ["MUSE"], ["NIFS_H", "nifs_h"]):
            with self.subTest(q=q), self.assertRaises(ValueError):
                self.tb.quitados_de(self.segs, q)

    def test_deja_al_menos_uno_del_ir(self):
        with self.assertRaises(ValueError):
            self.tb.quitados_de(self.segs, ["NIFS_J", "NIFS_H", "OSIRIS_K", "OSIRIS_H", "SINFONI_K"])
        self.assertEqual(len(self.tb.quitados_de(self.segs, ["NIFS_J", "NIFS_H", "OSIRIS_K", "OSIRIS_H"])), 4)


class EtiquetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def _a(self, **kw):
        base = dict(etiqueta=None, slug="ROXs42Bb", rejilla="atmo_ceq1500", modo="libre", tracks="ATMO2020",
                    av="primaria", plantilla="TWA26", av_fija=None, inyecta_bt=None, datos="ir", r250=False,
                    semilla=0)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_sin_la_opcion_la_de_siempre(self):
        for a in (self._a(), self._a(quita_ir=None)):
            self.assertEqual(self.tb.etiqueta_de(a), "ROXs42Bb_atmo_ceq1500_libre_avprimaria_pl-TWA26_soloir_s0")

    def test_la_pieza_va_detras_de_las_anclas_y_en_orden(self):
        self.assertEqual(self.tb.etiqueta_de(self._a(quita_ir=["SINFONI_K", "NIFS_H"])),
                         "ROXs42Bb_atmo_ceq1500_libre_avprimaria_pl-TWA26_quitair-NIFS_H-SINFONI_K_soloir_s0")
        self.assertEqual(self.tb.etiqueta_de(self._a(quita_ir=["NIFS_H"], anclas_ir="K")),
                         "ROXs42Bb_atmo_ceq1500_libre_avprimaria_pl-TWA26_anclas-K_quitair-NIFS_H_soloir_s0")


class VarianteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_es_variante_y_no_entra_en_las_tablas(self):
        for q in (["NIFS_H"], ["NIFS_H", "SINFONI_K"]):
            d = {**PAPER, "quita_ir": q}
            self.assertTrue(self.tb._es_variante(d), q)
            self.assertFalse(self.tb._de_la_tabla(d), q)
            self.assertFalse(self.tb._de_rejillas(d), q)
        self.assertFalse(self.tb._es_variante({**PAPER, "quita_ir": None}))


class ReconstruyeTests(unittest.TestCase):
    """`modelo_de_cadena` le pasa al `Modelo` lo que dice el json (con un `Modelo` de mentira que lo apunta)."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_lee_quita_ir(self):
        visto = {}

        class Falso:
            nombres = ["teff"]

            def __init__(self, *args, **kw):
                visto.update(kw)

        with mock.patch.object(self.tb, "Modelo", Falso):
            self.tb.modelo_de_cadena({**PAPER, "mejor": {"teff": 2600.0}})
            self.assertIsNone(visto["quita_ir"])
            self.tb.modelo_de_cadena({**PAPER, "mejor": {"teff": 2600.0}, "quita_ir": ["OSIRIS_J"]})
            self.assertEqual(visto["quita_ir"], ["OSIRIS_J"])


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
        self.assertNotIn("quita_ir", d)
        self.assertNotIn("quita_ir", _ModeloFalso.visto)

    def test_con_la_opcion_clave_etiqueta_y_argumento(self):
        d = self._json("--quita-ir", "SINFONI_K", "NIFS_H")
        self.assertEqual(d["quita_ir"], ["NIFS_H", "SINFONI_K"])
        self.assertEqual(d["etiqueta"], "ROXs42Bb_libre_avprimaria_pl-TWA26_quitair-NIFS_H-SINFONI_K_soloir_s0")
        self.assertEqual(_ModeloFalso.visto["quita_ir"], ["SINFONI_K", "NIFS_H"])
        self.assertLessEqual(set(d), self.tb.CLAVES_JSON | set(self.tb.VARIANTES_JSON))

    def test_sin_ningun_segmento_no_pasa_de_argparse(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            self._json("--quita-ir")


# ------------------------------------------------------------------ datos reales
@pytest.mark.external_data
@pytest.mark.slow
@requiere("runs", "ir", "bibliotecas")
class ModeloDatosTests(unittest.TestCase):
    """Con BT-Settl de /mnt/2TB: una plantilla solo con el IR, con todos los segmentos y sin el primero."""

    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")
        slugs = list(runs_del_paper())
        if not slugs:
            raise unittest.SkipTest("hace falta un compañero")
        cls.a = slugs[0]
        try:
            cls.todos = modelo_teff(cls.a, plantilla="TWA26", semilla=1, datos="ir")
            cls.quita = cls.todos.segs[0]["nombre"]
            cls.sin = modelo_teff(cls.a, plantilla="TWA26", semilla=1, datos="ir", quita_ir=(cls.quita,))
        except (OSError, SystemExit) as exc:   # sin las rejillas de /mnt/2TB
            raise unittest.SkipTest(f"no se puede construir el modelo: {exc}")

    def test_los_que_quedan_llevan_los_mismos_datos(self):
        quedan = [s for s in self.todos.segs if s["nombre"] != self.quita]
        self.assertEqual([s["nombre"] for s in self.sin.segs], [s["nombre"] for s in quedan])
        self.assertEqual(self.sin.verdad, self.todos.verdad)
        for s, c in zip(quedan, self.sin.segs):
            self.assertEqual(c["sigma"], s["sigma"], s["nombre"])
            for k in ("w", "d", "e2", "lg"):
                np.testing.assert_array_equal(c[k], s[k], err_msg=f"{s['nombre']} {k}")

    def test_pierde_su_escala_y_nada_mas(self):
        self.assertNotIn(f"s_{self.quita}", self.sin.nombres)
        bandas = sorted({s["banda"] for s in self.sin.segs})
        esperados = [n for n in self.todos.nombres if n != f"s_{self.quita}"
                     and not (n.startswith("lnf_") and n[4:] not in bandas)]
        self.assertEqual(self.sin.nombres, esperados)

    def test_la_verosimilitud_pierde_justo_ese_termino(self):
        k = next(i for i, s in enumerate(self.todos.segs) if s["nombre"] == self.quita)
        s = self.todos.segs[k]
        rng = np.random.default_rng(0)
        for _ in range(5):
            xt = self.todos.prior_transform(rng.uniform(0.05, 0.95, self.todos.ndim))
            xs = np.array([xt[self.todos.pos[n]] for n in self.sin.nombres])
            m = self.todos.modelos(xt)[k]
            var = s["e2"] + (math.exp(xt[self.todos.pos[f"lnf_{s['banda']}"]]) * m) ** 2
            termino = -0.5 * float(np.sum((s["d"] - m) ** 2 / var + np.log(var)))
            self.assertAlmostEqual(self.todos.loglike(xt) - self.sin.loglike(xs), termino,
                                   delta=1e-9 * abs(termino) + 1e-6)

    def test_un_segmento_que_no_tiene_falla(self):
        with self.assertRaises(ValueError):
            modelo_teff(self.a, plantilla="TWA26", semilla=1, datos="ir", quita_ir=("NO_EXISTE_K",))


if __name__ == "__main__":
    unittest.main()
