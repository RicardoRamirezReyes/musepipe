"""El material suplementario del apéndice B (`scripts/teff_suplemento.py`), con cadenas sintéticas en disco.

Las cadenas de verdad viven fuera de runs/ (`teff_bayes.py ajusta`); aquí se prueba lo que el PDF decide: qué
cadenas entran, qué columnas lleva cada página y que una página se dibuja. El margen Δ usa una entrada sintética
(lo que `entrada_margen` lee de los runs).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENTRADA = dict(lacc0=1e-6, av0=1.8, a=1.0, ahav=0.8, lbol_lit=1e-3)


def _modulo():
    spec = importlib.util.spec_from_file_location("teff_suplemento", ROOT / "scripts" / "teff_suplemento.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["teff_suplemento"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cadena(d, nombre, modo="libre", av_fija=None, datos="todo", verdad=None, npz=True):
    """Un json y su npz como los de `teff_bayes.py ajusta`, con A_V fija si `av_fija` (muestreada, como allí, en
    un perfil de ±0.01 mag)."""
    rng = np.random.default_rng(0)
    n = 400
    if modo == "evo":
        nombres = ["log_edad_myr", "log_lbol", "d_pc", "av", "rk"]
    else:
        nombres = ["teff", "logg", "r_rj", "d_pc", "av"] + ([] if datos == "muse" else ["rk"])
    m = rng.normal(1.0, 0.1, (n, len(nombres)))
    m[:, nombres.index("av")] = (av_fija + 0.02 * (rng.random(n) - 0.5) if av_fija is not None
                                 else rng.normal(3.0, 0.2, n))
    der = np.column_stack([rng.normal(2600, 20, n), rng.normal(4.0, 0.1, n), rng.normal(1.3e10, 1e8, n),
                           np.full(n, np.nan), rng.normal(3e-3, 1e-4, n)])
    if npz:
        np.savez(d / f"{nombre}.npz", muestras=m, derivadas=der, nombres=np.array(nombres))
    meta = dict(etiqueta=nombre, slug="X", modo=modo, av="primaria", av_fija=av_fija, rejilla="btsettl",
                datos=datos, r_analisis=500, tracks="ATMO2020" if modo == "evo" else None, nlive=100, semilla=1,
                logz=10.0, logz_err=0.3, verdad=verdad)
    (d / f"{nombre}.json").write_text(json.dumps(meta))


class SuplementoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ts = _modulo()

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.d = Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def test_secciones_en_orden_sin_evo_solo(self):
        _cadena(self.d, "b_prior")
        _cadena(self.d, "a_fija", av_fija=1.8)
        (self.d / "c_evo_solo.json").write_text(json.dumps(dict(etiqueta="c", slug="X", modo="evo_solo")))
        s = self.ts.carga_secciones([("Uno", [str(self.d / "b_*.json")]), ("Dos", [str(self.d / "[ac]_*.json")])])
        self.assertEqual([(t, [d["etiqueta"] for d in cs]) for t, cs in s],
                         [("Uno", ["b_prior"]), ("Dos", ["a_fija"])])

    def test_falla_sin_coincidencias_sin_npz_o_con_una_cadena_dos_veces(self):
        _cadena(self.d, "a")
        _cadena(self.d, "sin", npz=False)
        for secciones in ([("S", [str(self.d / "nada_*.json")])], [("S", [str(self.d / "sin.json")])],
                          [("S", [str(self.d / "a.json")]), ("T", [str(self.d / "*.json")])]):
            with self.assertRaises(SystemExit):
                self.ts.carga_secciones(secciones)

    def _columnas(self, nombre, **kw):
        _cadena(self.d, nombre, **kw)
        d = json.loads((self.d / f"{nombre}.json").read_text())
        d["_npz"] = str(self.d / f"{nombre}.npz")
        return self.ts.columnas(d, ENTRADA)

    def test_columnas_segun_la_cadena(self):
        etq, m, verdad, nd = self._columnas("prior")
        self.assertEqual(len(etq), 6)                               # T, g, R, A_V, r_K y Δ
        self.assertEqual((m.shape, nd), ((400, 6), [0, 2, 2, 2, 2, 2]))
        etq, m, _, _ = self._columnas("fija", av_fija=1.8)
        self.assertNotIn(r"$A_V$ (mag)", etq)                       # la A_V fija no es un parámetro
        etq, _, _, _ = self._columnas("muse", datos="muse")
        self.assertNotIn(r"$r_K$", etq)
        etq, _, verdad, _ = self._columnas("pl", verdad={"plantilla": "TWA26", "spt": "M9", "lib": "manara", "av": 1.8})
        self.assertNotIn(r"$\Delta$ (dex)", etq)                    # una plantilla no es el compañero
        self.assertEqual(verdad[etq.index(r"$A_V$ (mag)")], 1.8)
        etq, m, _, _ = self._columnas("evo", modo="evo")
        self.assertEqual(etq[:2], [r"$\log$ age (Myr)", r"$T_{\rm eff}$ (K)"])
        self.assertAlmostEqual(float(np.median(m[:, 3])), 1.3e10 / self.ts.tb.RJUP_CM, delta=0.05)

    def test_titulo_con_los_decimales_de_cada_parametro(self):
        x = np.array([2580.0, 2600.0, 2630.0] * 100)
        self.assertEqual(self.ts.titulo("T", np.sort(x), 0), "T = $2600^{+30}_{-20}$")

    @unittest.skipUnless(importlib.util.find_spec("corner"), "corner no está instalado (environment.yml lo fija)")
    def test_una_pagina_y_su_cabecera(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        _cadena(self.d, "inj", verdad={"teff": 2550.0, "logg": 3.75, "av": 1.8, "r_rj": 2.2})
        d = json.loads((self.d / "inj.json").read_text())
        d["_npz"] = str(self.d / "inj.npz")
        lineas = self.ts.descripcion(d, "Obj X")
        self.assertEqual(lineas[0], "Obj X: BT-Settl, MUSE + near-IR")
        self.assertIn("Data replaced by a BT-Settl model", lineas[2])
        with PdfPages(self.d / "p.pdf") as pdf:
            self.ts.pagina(pdf, d, ENTRADA, "Obj X", "1. Tests", "page 2 of 2")
            self.assertEqual(pdf.get_pagecount(), 1)
        self.assertGreater((self.d / "p.pdf").stat().st_size, 1000)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
