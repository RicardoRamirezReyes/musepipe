"""Qué cadenas de `scripts/teff_bayes.py` entran en las tablas del paper, y lo que `ajusta` guarda (2026-10-29).

`_de_la_tabla` y `_de_rejillas` filtraban las variantes por lista negra (`mascara_muse`, ventanas), así que una
opción nueva entraba en la Tabla B.6, la B.7, `disco_cotas` y el corner si nadie la añadía a mano. Ahora lo decide
`_es_variante`, una lista blanca. Se prueba que deja fuera cualquier opción nueva y que, sobre las cadenas que hay,
elige exactamente lo mismo que el filtro de antes (reimplementado aquí). Y que `ajusta` no sobrescribe una
etiqueta y guarda su procedencia.
"""
import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from tests._datos import cadenas_teff, carga_script, requiere

#: un json de cadena del paper, con las claves que llevan (valores de muestra)
PAPER = dict(etiqueta="ROXs12b_libre_avprimaria_s1", slug="ROXs12b", run="ROXs12b_invvar", modo="libre",
             av="primaria", av_fija=None, rejilla="btsettl", datos="todo", r_analisis=500, nodos_interpolados=[],
             ejes={}, tracks=None, nlive=500, semilla=1, ndim=12, nombres=[], segundos=1.0, ncall=1, logz=-1.0,
             logz_err=0.1, logl_max=-1.0, chi2nu_mejor={}, chi2nu_mejor_inflado={}, mejor={}, verdad=None,
             posterior={}, n_puntos={})


def _de_la_tabla_antes(tb, d):
    """El filtro de `_de_la_tabla` hasta el 2026-10-29 (lista negra)."""
    ventanas = d.get("ventana_lineas_kms") is not None or d.get("ventana_halpha_kms") is not None
    return (d["modo"] == "libre" and d.get("av") == "primaria" and not d.get("verdad")
            and d.get("datos", "todo") == "todo" and d.get("rejilla", "btsettl") in tb.NOMBRE_REJILLA
            and d.get("r_analisis", 500) == 500 and d.get("mascara_muse", "paper") == "paper" and not ventanas)


def _de_rejillas_antes(tb, d):
    ventanas = d.get("ventana_lineas_kms") is not None or d.get("ventana_halpha_kms") is not None
    return (d["modo"] == "libre" and d.get("av") == "primaria" and not d.get("verdad")
            and d.get("datos", "todo") == "todo" and d.get("rejilla", "btsettl") in tb.REJILLAS_TABLA
            and d.get("mascara_muse", "paper") == "paper" and not ventanas)


class ListaBlancaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = carga_script("teff_bayes")

    def test_la_cadena_del_paper_entra(self):
        self.assertFalse(self.tb._es_variante(PAPER))
        self.assertTrue(self.tb._de_la_tabla(PAPER))
        self.assertTrue(self.tb._de_rejillas(PAPER))
        # las claves que añaden los cargadores (`_npz`, `_json`, `_grupo`) no cuentan, ni el valor del paper
        # escrito explícitamente, ni la procedencia de las cadenas nuevas
        for extra in ({"_npz": "x.npz", "_grupo": "A1"}, {"mascara_muse": "paper"}, {"procedencia": {"commit": "x"}},
                      {"degrada": "cuadratura"}):
            self.assertFalse(self.tb._es_variante({**PAPER, **extra}), extra)

    def test_cada_variante_queda_fuera(self):
        valores = {"mascara_muse": "sin_o2b", "ventana_lineas_kms": 50.0, "ventana_halpha_kms": 100.0,
                   "rk_fijo": 0.0, "td_rango": [300.0, 1500.0], "quita_muse": "extremos",
                   "inyecta_polvo": [0.2, 900.0]}
        self.assertEqual(set(valores), set(self.tb.VARIANTES_JSON))
        for k, v in valores.items():
            d = {**PAPER, k: v}
            self.assertTrue(self.tb._es_variante(d), k)
            self.assertFalse(self.tb._de_la_tabla(d), k)
            self.assertFalse(self.tb._de_rejillas(d), k)

    def test_una_clave_desconocida_queda_fuera(self):
        """Una opción futura que nadie añada a `VARIANTES_JSON` no entra en las tablas."""
        d = {**PAPER, "opcion_nueva": 1}
        self.assertTrue(self.tb._es_variante(d))
        self.assertFalse(self.tb._de_la_tabla(d))
        # con el filtro de antes, una cadena con r_K fija entraba en la tabla
        self.assertTrue(_de_la_tabla_antes(self.tb, {**PAPER, "rk_fijo": 0.0}))


class _ModeloFalso:
    """Lo que `ajusta` lee de un `Modelo`, sin datos."""

    def __init__(self, slug, modo, av, tracks, inyecta_bt, plantilla, av_inyectada, semilla, av_fija, rejilla,
                 dir_rejilla, datos, r250, degrada, mascara_muse="paper", ventana_lineas=None, ventana_halpha=None,
                 rk_fijo=None, td_rango=None, quita_muse=None, inyecta_polvo=None):
        self.run, self.r_analisis, self.rellenados, self.degrada = "RUN", 500, [], None
        self.rk_fijo, self.td_rango = rk_fijo, (None if td_rango is None else tuple(td_rango))
        self.T0, self.dT, self.nT, self.G0, self.dG, self.nG = 2000.0, 100.0, 13, 2.5, 0.5, 6
        self.nombres = ["teff", "logg", "av"]
        self.ndim = 3
        self.dir_rejilla = Path("/rejilla")
        self.segs = [dict(nombre="MUSE", d=np.zeros(84))]
        if inyecta_bt:
            self.verdad = dict(teff=inyecta_bt[0])

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
            self.tb.main(["ajusta", "ROXs12b", "--salida", t, *opciones])
            (f,) = Path(t).glob("*.json")
            return json.loads(f.read_text())

    def test_el_json_de_ajusta_solo_lleva_claves_conocidas(self):
        """Toda clave que escribe `ajusta` está en `CLAVES_JSON` o en `VARIANTES_JSON`: si no, sus propias cadenas
        del paper serían variantes. Sin opciones, la corrida no es una variante; con cada una, sí."""
        conocidas = self.tb.CLAVES_JSON | set(self.tb.VARIANTES_JSON)
        d = self._json()
        self.assertLessEqual(set(d), conocidas)
        self.assertFalse(self.tb._es_variante(d))
        self.assertTrue(self.tb._de_la_tabla(d))
        for opciones, clave in ((["--rk-fijo", "0"], "rk_fijo"), (["--td-libre", "300", "1500"], "td_rango"),
                                (["--quita-muse", "extremos"], "quita_muse"),
                                (["--inyecta-bt", "2600", "4", "1.8", "2", "--inyecta-polvo", "0.2", "900"],
                                 "inyecta_polvo")):
            d = self._json(*opciones)
            self.assertLessEqual(set(d), conocidas, opciones)
            self.assertIn(clave, d)
            self.assertTrue(self.tb._es_variante(d), opciones)
            self.assertFalse(self.tb._de_la_tabla(d), opciones)
        self.assertEqual(self._json("--td-libre", "300", "1500")["td_rango"], [300.0, 1500.0])
        self.assertEqual(self._json("--rk-fijo", "0")["etiqueta"], "ROXs12b_libre_avprimaria_rk0_s0")
        self.assertIn("procedencia", d)

    def test_no_sobrescribe_una_etiqueta(self):
        with tempfile.TemporaryDirectory() as t:
            et = "ROXs12b_libre_avprimaria_rk0_s0"
            (Path(t) / f"{et}.npz").write_bytes(b"")
            with self.assertRaises(SystemExit) as cm:
                self.tb.main(["ajusta", "ROXs12b", "--rk-fijo", "0", "--salida", t])
            self.assertIn(et, str(cm.exception))
            self.assertIn("no se sobrescribe", str(cm.exception))
            self.assertEqual(sorted(os.listdir(t)), [f"{et}.npz"])   # no escribió nada

    def test_procedencia(self):
        p = self.tb.procedencia(a=argparse.Namespace(slug="ROXs12b", salida=Path("/x"), rk_fijo=0.0))
        import musepipe
        import numpy
        import scipy
        self.assertEqual((p["musepipe"], p["numpy"], p["scipy"]), (musepipe.__version__, numpy.__version__,
                                                                   scipy.__version__))
        self.assertEqual(p["muestreo"], self.tb.MUESTREO)
        self.assertEqual(p["argumentos"], {"slug": "ROXs12b", "salida": "/x", "rk_fijo": 0.0})
        self.assertIn("commit", p)
        self.assertIn("dynesty", p)
        json.dumps(p)   # serializable


# ------------------------------------------------------------------ datos reales
def _cadenas():
    out = []
    for raiz in cadenas_teff():
        for r, _, fs in os.walk(raiz, followlinks=True):
            for f in sorted(fs):
                if f.endswith(".json"):
                    try:
                        d = json.loads((Path(r) / f).read_text())
                    except (OSError, ValueError):
                        continue
                    if isinstance(d, dict) and "etiqueta" in d and "modo" in d:
                        out.append(d)
    return out


@pytest.mark.external_data
@requiere("cadenas")
class CadenasRealesTests(unittest.TestCase):
    def test_elige_lo_mismo_que_el_filtro_de_antes(self):
        tb = carga_script("teff_bayes")
        ds = _cadenas()
        self.assertTrue(ds)
        for d in ds:
            self.assertEqual(tb._de_la_tabla(d), _de_la_tabla_antes(tb, d), d["etiqueta"])
            self.assertEqual(tb._de_rejillas(d), _de_rejillas_antes(tb, d), d["etiqueta"])

    def test_ninguna_clave_desconocida(self):
        tb = carga_script("teff_bayes")
        conocidas = tb.CLAVES_JSON | set(tb.VARIANTES_JSON)
        for d in _cadenas():
            self.assertLessEqual(set(d) - conocidas, set(), d["etiqueta"])


if __name__ == "__main__":
    unittest.main()
