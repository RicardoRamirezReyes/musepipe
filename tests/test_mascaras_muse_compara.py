"""Comparación de las cadenas de T_eff con otras máscaras del espectro MUSE (`scripts/mascaras_muse_compara.py`).

Las primeras clases trabajan con cadenas sintéticas (json + npz en un directorio temporal) y un margen inventado, sin
leer nada del proyecto. `DatosRealesTests` (marca `external_data`) reproduce la Tabla D.2 del paper con las cadenas
de la máquina del autor (`MUSE_TEFF_CADENAS`, `MUSE_ATMO_CADENAS`, `MUSE_MASCARAS_CADENAS`, separadas por `:` las
de las variantes) y los runs de `MUSE_PROJECT_ROOT`; si falta algo, se salta.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


mc = _carga("mascaras_muse_compara")

#: un margen de juguete: L_acc fija, sin reescalar con A_V (a = 0), así Δ depende solo de T_eff y L_bol.
ENTRADA = dict(lacc0=1e-6, tipo="detection", av0=1.8, a=0.0, ahav=0.8, lbol_lit=1e-3)


def _cadena(directorio, slug, rejilla="btsettl", av_fija=None, mascara=None, ventanas=None, teff=2700.0, logg=4.5,
            av=3.0, lbol=1e-3, chi=10.0, bins=84, nlive=800, n=400, semilla=1):
    """Escribe una cadena con la forma de `teff_bayes.py ajusta`: posterior, n_puntos, chi2nu_mejor y el npz."""
    partes = [slug] + ([] if rejilla == "btsettl" else [rejilla]) + ["libre", "avprimaria"]
    partes += [] if av_fija is None else [f"avfija{av_fija:g}"]
    partes += [] if mascara is None else [f"mascara-{mascara}"]
    partes += [] if ventanas is None else [f"ventana-{ventanas[0]:g}", f"ventanaha-{ventanas[1]:g}"]
    etiqueta = "_".join(partes + [f"s{semilla}"])
    u = np.linspace(-2.0, 2.0, n)   # muestras simétricas: la mediana es exactamente el valor pedido
    t = teff + 20.0 * u
    a = np.full(n, av) if av_fija is not None else av + 0.1 * u
    der = np.column_stack([t, np.full(n, logg), np.ones(n), np.ones(n), np.full(n, lbol)])
    np.savez(Path(directorio) / f"{etiqueta}.npz", muestras=np.column_stack([t, np.full(n, logg), a]),
             derivadas=der, nombres=np.array(["teff", "logg", "av"]))

    def q(x):
        return dict(p16=float(np.percentile(x, 16)), p50=float(np.median(x)), p84=float(np.percentile(x, 84)))
    d = dict(etiqueta=etiqueta, slug=slug, rejilla=rejilla, modo="libre", av="primaria", av_fija=av_fija,
             r_analisis=500, datos="todo", semilla=semilla, nlive=nlive, n_puntos={"MUSE": bins},
             chi2nu_mejor={"MUSE": chi},
             posterior={"der_teff": q(t), "der_logg": q(np.full(n, logg)), "av": q(a)})
    if mascara is not None:
        d["mascara_muse"] = mascara
    if ventanas is not None:
        d["ventana_lineas_kms"], d["ventana_halpha_kms"] = ventanas
    (Path(directorio) / f"{etiqueta}.json").write_text(json.dumps(d))
    return etiqueta


class ParejasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.ref, self.var = base / "ref", base / "var"
        self.ref.mkdir()
        self.var.mkdir()
        for av_fija in (None, 1.8):
            _cadena(self.ref, "A", av_fija=av_fija, teff=2700.0, av=3.0)
            _cadena(self.ref, "A", rejilla="drift", av_fija=av_fija, teff=2800.0, av=3.2)
        # sin_o2b: la de BT-Settl con prior se mueve 40 K, sube A_V 0.5 y empeora el χ²; la de A_V fija, 10 K
        _cadena(self.var, "A", mascara="sin_o2b", teff=2740.0, av=3.5, chi=12.0, bins=92)
        _cadena(self.var, "A", mascara="sin_o2b", av_fija=1.8, teff=2690.0, chi=9.0, bins=92)
        # una ventana sin máscara nueva
        _cadena(self.var, "A", ventanas=(50.0, 100.0), teff=2702.0, av=3.0, bins=91)
        # una sin referencia (no hay DRIFT con esa semilla)
        _cadena(self.var, "A", rejilla="drift", mascara="g3", semilla=2, bins=142)

    def tearDown(self):
        self.tmp.cleanup()

    def _calcula(self):
        return mc.calcula(mc.carga([self.ref]), mc.carga([self.var]), {"A": ENTRADA})

    def test_variante_y_nombre(self):
        self.assertEqual(mc.variante_de({}), ("paper", None))
        v = mc.variante_de({"ventana_lineas_kms": 50, "ventana_halpha_kms": 100, "mascara_muse": "sin_o2b"})
        self.assertEqual(mc.nombre_variante(v), "sin_o2b+ventana50/100")
        self.assertEqual([mc.nombre_variante(v) for v in mc.ORDEN][:3],
                         ["paper+ventana50/100", "sin_o2b", "sin_o2b+ventana50/100"])

    def test_pareja_por_configuracion(self):
        """Cada variante se compara con la referencia de su misma rejilla, A_V y semilla; la que no la tiene se dice."""
        res = self._calcula()
        self.assertEqual(len(res["parejas"]), 3)
        self.assertEqual(res["sin_pareja"], ["A_drift_libre_avprimaria_mascara-g3_s2"])
        for p in res["parejas"]:
            self.assertEqual(p["rejilla"], "btsettl")
            self.assertEqual(p["av_fija"], None if "avfija" not in p["cadena"]["etiqueta"] else 1.8)
            self.assertNotIn("mascara", p["referencia"]["etiqueta"])

    def test_maximos_y_av_solo_con_prior(self):
        res = self._calcula()
        r = res["resumen"]["sin_o2b"]["A"]
        self.assertEqual(r["n"], 2)
        self.assertEqual(r["bins"], [92])
        self.assertIn("mascara-sin_o2b_s1", r["dT"]["cadena"])
        self.assertAlmostEqual(r["dT"]["valor"], 40.0, delta=1.0)
        # A_V: solo la del prior (la de A_V fija no cambia por construcción)
        self.assertAlmostEqual(r["dav"]["valor"], 0.5, delta=0.05)
        self.assertNotIn("avfija", r["dav"]["cadena"])
        self.assertEqual(r["chi2_sube"], 1)
        # Δ baja al subir T_eff con L_bol fija (el nivel cromosférico sube con T_eff)
        self.assertLess(r["dD"]["valor"], 0)

    def test_filas_y_tex(self):
        """La tabla redondeada y su comparación con un tex que la tiene igual, y con uno que difiere en una cifra."""
        res = self._calcula()
        filas = mc.filas_tabla(res, ["A"])
        self.assertEqual(filas[0]["bins"], [91])
        self.assertEqual(filas[1]["ajustes"], [2])
        self.assertIsNone(filas[3])   # g3 sin pareja
        self.assertEqual(filas[1]["cambios"][0], round(res["resumen"]["sin_o2b"]["A"]["dT"]["max_abs"]))

        def fila_tex(f):
            celdas = [", ".join(f"\\({b}\\)" for b in f["bins"]), ", ".join(f"\\({n}\\)" for n in f["ajustes"])]
            return " & ".join(["Masks"] + celdas + [f"\\({x:g}\\)" for x in f["cambios"]]) + " \\\\"
        cuerpo = [fila_tex(f) if f else "No data & \\(0\\) & \\(0\\) & \\(0\\) & \\(0\\) & \\(0\\) & \\(0\\) \\\\"
                  for f in filas]
        tex = ("\\begin{table*}\n\\label{tab:masks}\n\\begin{tabular}{l c c c c c c}\n\\hline\\hline\n"
               "Masks & Bins & Fits & a & b & c & d \\\\\n\\hline\n" + "\n".join(cuerpo) + "\n\\hline\n\\end{tabular}")
        dif = mc.compara_tex(tex, filas)
        self.assertEqual(len(dif), 4)   # las cuatro variantes sin cadenas (g3 no tiene pareja)
        self.assertTrue(all("faltan cadenas" in x for x in dif))
        mal = tex.replace(fila_tex(filas[1]), fila_tex(dict(filas[1], cambios=[filas[1]["cambios"][0] + 1]
                                                             + filas[1]["cambios"][1:])))
        self.assertEqual(len(mc.compara_tex(mal, filas)), 5)
        self.assertEqual(mc.compara_tex("sin tabla", filas), ["no hay \\label{tab:masks} en el tex"])

    def test_signo_y_repetidas(self):
        """Un cambio de signo de Δ se lista; una etiqueta repetida en dos directorios se usa una vez y se avisa."""
        otra = Path(self.tmp.name) / "otra"
        otra.mkdir()
        _cadena(otra, "A", teff=2700.0, av=3.0)
        sombreadas = []
        refs = mc.carga([self.ref, otra], sombreadas)
        self.assertEqual(len(sombreadas), 1)
        self.assertTrue(sombreadas[0][0].startswith(str(self.ref)))
        # con L_acc/L_bol 0.02 dex sobre el nivel a 2700 K, la referencia queda encima y la variante (2740 K) debajo
        nivel = 10 ** mc.tb.suelo(2700.0)
        e = dict(ENTRADA, lacc0=nivel * 1e-3 * 1.05)
        res = mc.calcula(refs, mc.carga([self.var]), {"A": e})
        self.assertEqual([c["cadena"] for c in res["cambios_de_signo"]], ["A_libre_avprimaria_mascara-sin_o2b_s1"])


# ------------------------------------------------------------------ datos reales
PROYECTO = Path(os.environ.get("MUSE_PROJECT_ROOT", ROOT))
CADENAS = Path(os.environ.get("MUSE_TEFF_CADENAS", "~/muse_noche_20261024_c4/cadenas")).expanduser()
ATMO = Path(os.environ.get("MUSE_ATMO_CADENAS", "~/muse_sesion_20261026/claudio/atmo/res/atmo1500")).expanduser()
VARIANTES = [Path(p).expanduser() for p in os.environ.get(
    "MUSE_MASCARAS_CADENAS",
    "~/muse_sesion_20261028/mascaras_muse/lote/res:~/muse_sesion_20261028/ventanas_lineas/lote/res").split(":")]
#: la Tabla D.2 del paper (rama del 2026-10-29), fila a fila: bins, ajustes y los ocho máximos.
TABLA_D2 = [
    ([91, 90], [2, 2], [2, 0.02, 0.12, 0.03, 15, 0.02, 0.10, 0.00]),
    ([92], [4, 6], [15, 0.09, 0.18, 0.04, 30, 0.04, 0.18, 0.01]),
    ([99, 98], [2, 2], [48, 0.06, 0.53, 0.08, 44, 0.05, 0.30, 0.01]),
    ([142], [4, 6], [49, 0.12, 0.65, 0.14, 62, 0.25, 0.50, 0.07]),
    ([101], [4, 6], [53, 0.97, 0.27, 0.15, 232, 0.54, 1.01, 0.12]),
    ([159], [4, 6], [78, 1.00, 0.69, 0.10, 94, 0.47, 0.42, 0.06]),
]


@pytest.mark.external_data
class DatosRealesTests(unittest.TestCase):
    def test_tabla_d2(self):
        fichas = [json.loads(f.read_text()) for f in sorted((ROOT / "targets").glob("*.json"))]
        runs = [d["paper"]["run"] for d in fichas if "paper" in d]
        if not runs or not all((PROYECTO / "runs" / r).exists() for r in runs):
            self.skipTest("sin los runs del paper (MUSE_PROJECT_ROOT)")
        if not all(p.is_dir() for p in [CADENAS, ATMO] + VARIANTES):
            self.skipTest("sin las cadenas del paper o de las variantes")
        antes = mc.tb.ROOT
        try:
            raiz = mc.tb.fija_raiz(PROYECTO)
            variantes = mc.carga(VARIANTES)
            slugs = sorted({d["slug"] for d in variantes.values()})
            res = mc.calcula(mc.carga([CADENAS, ATMO]), variantes,
                             {s: mc.tb.entrada_margen(s, root=raiz) for s in slugs})
        finally:
            mc.tb.fija_raiz(antes)
        self.assertEqual(res["sin_pareja"], [])
        filas = mc.filas_tabla(res, slugs)
        for k, (f, (bins, ajustes, cambios)) in enumerate(zip(filas, TABLA_D2)):
            self.assertIsNotNone(f, k)
            self.assertEqual((f["bins"], f["ajustes"]), (bins, ajustes), k)
            self.assertEqual([round(x, 2) for x in f["cambios"]], cambios, k)
        self.assertEqual(sorted(c["cadena"] for c in res["cambios_de_signo"]), [
            "ROXs12b_libre_avprimaria_avfija1.8_mascara-g3_s1",
            "ROXs42Bb_atmo_ceq1500_libre_avprimaria_mascara-sin_o2_s1",
            "ROXs42Bb_atmo_ceq1500_libre_avprimaria_mascara-sin_telurico_s1"])
