"""Δ entre rejillas de atmósferas (`scripts/delta_entre_rejillas.py`).

Las primeras clases no leen nada de disco fuera de un directorio temporal: los cuantiles con pesos, la mezcla de
posteriores (con la ley de la varianza total), los pesos iguales y por evidencia, la cuadratura, la lectura del
cuerpo de la Tabla B.7 de un LaTeX y su comparación, y la Tabla B.7 escrita desde cadenas sintéticas (json + npz
como los de `teff_bayes.py ajusta`) con la misma lectura que usa el script. `DatosRealesTests` (marca
`external_data`) reproduce la Tabla B.7 del paper fila a fila desde las cadenas c0 y reconstruye un modelo con su
`logl_max`; lee `MUSE_PROJECT_ROOT` (la raíz con `runs/` y `paper/`, por defecto el repo) y `MUSE_TEFF_CADENAS`
(la carpeta de cadenas, por defecto la del paper); si falta algo se salta, no falla.
"""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import delta_entre_rejillas as dr  # noqa: E402
from scripts import teff_bayes as tb  # noqa: E402
from scripts import teff_cifras_texto as tc  # noqa: E402


def setUpModule():
    # los objetos del paper salen de targets/ (el script no nombra objetos)
    tc.fija_objetos(ROOT)


def _normal(mu, sd, n):
    """n muestras deterministas de N(mu, sd): los cuantiles (i + ½)/n."""
    return mu + sd * stats.norm.ppf((np.arange(n) + 0.5) / n)


class CuantilesTests(unittest.TestCase):
    def test_con_pesos_iguales_es_hazen(self):
        x = np.random.default_rng(3).normal(size=501)
        w = np.ones(x.size)
        for q in (2.5, 16, 50, 84, 97.5):
            self.assertAlmostEqual(float(dr.cuantiles_pesados(x, w, [q])[0]),
                                   float(np.percentile(x, q, method="hazen")), places=12)

    def test_la_cdf_va_en_el_punto_medio_de_cada_peso(self):
        # ordenados: -1.0 (peso 3), 0.3 (1), 0.7 (4), 2.0 (2); total 10: CDF en 0.15, 0.35, 0.60 y 0.90
        x = np.array([0.3, -1.0, 2.0, 0.7])
        w = np.array([1, 3, 2, 4])
        q = dr.cuantiles_pesados(x, w, [15, 35, 60, 90, 47.5, 5, 99])
        np.testing.assert_allclose(q, [-1.0, 0.3, 0.7, 2.0, 0.5, -1.0, 2.0], atol=1e-12)
        # no depende de la escala de los pesos ni del orden
        o = np.array([2, 0, 3, 1])
        np.testing.assert_allclose(dr.cuantiles_pesados(x[o], 7.0 * w[o], [15, 47.5, 90]), q[[0, 4, 3]], atol=1e-12)


class MezclaTests(unittest.TestCase):
    def test_cada_rejilla_pesa_lo_mismo_aunque_tenga_mas_muestras(self):
        a, b = _normal(0.0, 1.0, 1000), _normal(3.0, 1.0, 4000)
        x, w = dr.mezcla([a, b], [0.5, 0.5])
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        self.assertAlmostEqual(float(w[:1000].sum()), 0.5, places=12)
        r = dr.resume(x, w)
        # ley de la varianza total: media de las varianzas + varianza (poblacional) de las medias
        self.assertAlmostEqual(r["media"], 1.5, places=6)
        self.assertAlmostEqual(r["sd"] ** 2, (a.var() + b.var()) / 2 + np.var([a.mean(), b.mean()]), places=6)
        self.assertAlmostEqual(r["p50"], 1.5, places=2)
        self.assertAlmostEqual(r["p_positivo"], 0.5 * 0.5 + 0.5 * stats.norm.cdf(3.0), places=3)

    def test_una_lista_de_pesos_por_cadena(self):
        with self.assertRaises(ValueError):
            dr.mezcla([np.zeros(3), np.ones(3)], [1.0])

    def test_pesos_iguales_con_una_rejilla_de_seis_modelos(self):
        grupos = {"bt": ["b"], "drift": ["d"], "ames": ["a"], "cond": ["c"], "db": [f"f{i}" for i in range(6)]}
        p = dr.pesos_iguales(grupos)
        self.assertAlmostEqual(sum(p.values()), 1.0, places=12)
        self.assertAlmostEqual(p["b"], 0.2, places=12)
        self.assertAlmostEqual(p["f3"], 0.2 / 6, places=12)

    def test_pesos_por_evidencia(self):
        grupos = {"bt": ["b"], "drift": ["d"], "db": ["f1", "f2"]}
        # con la misma evidencia, los pesos son los iguales
        lnz = {"b": 10.0, "d": 10.0, "f1": 10.0, "f2": 10.0}
        self.assertEqual({k: round(v, 12) for k, v in dr.pesos_evidencia(grupos, lnz).items()},
                         {k: round(v, 12) for k, v in dr.pesos_iguales(grupos).items()})
        # ln Z mayor en 50: esa cadena se lleva todo
        lnz = {"b": 0.0, "d": -60.0, "f1": 50.0, "f2": 42.0}
        p = dr.pesos_evidencia(grupos, lnz)
        self.assertGreater(p["f1"], 0.999)
        self.assertAlmostEqual(p["f2"] / p["f1"], math.exp(-8.0), places=12)

    def test_cuadratura(self):
        c = dr.cuadratura([0.03, 0.05, 0.04], [0.0, 0.1, 0.2])
        self.assertAlmostEqual(c["anchura_tipica"], 0.04, places=12)
        self.assertAlmostEqual(c["dispersion"], 0.1, places=12)
        self.assertAlmostEqual(c["total"], math.hypot(0.04, 0.1), places=12)
        self.assertEqual(dr.cuadratura([0.05], [0.3])["dispersion"], 0.0)

    def test_estadisticas_de_un_conjunto(self):
        medias = {"bt": -0.10, "drift": 0.00, "ames": 0.05, "cond": 0.25}
        sd = 0.03
        deltas = {g: _normal(m, sd, 4000 + 1000 * i) for i, (g, m) in enumerate(medias.items())}
        grupos = {g: [g] for g in medias}
        lnz = {"bt": 0.0, "drift": -30.0, "ames": -40.0, "cond": -50.0}
        st = dr.estadisticas(grupos, deltas, lnz)
        m = np.array(list(medias.values()))
        self.assertEqual(st["n_rejillas"], 4)
        self.assertAlmostEqual(st["a"]["mediana"], float(np.median(m)), places=4)
        self.assertEqual((st["a"]["rejilla_min"], st["a"]["rejilla_max"]), ("bt", "cond"))
        self.assertAlmostEqual(st["b"]["sd"], float(np.std(m, ddof=1)), places=4)
        self.assertAlmostEqual(st["b"]["sd_poblacional"], float(np.std(m)), places=4)
        # la que, quitada, más reduce la dispersión: la más alejada (+0.25)
        self.assertEqual(st["b"]["rejilla_que_la_fija"], "cond")
        self.assertAlmostEqual(st["b"]["sd_sin"]["cond"], float(np.std(m[:3], ddof=1)), places=4)
        # (c): la mezcla a pesos iguales cumple la ley de la varianza total
        self.assertAlmostEqual(st["c"]["sd"], math.sqrt(sd ** 2 + np.var(m)), places=4)
        # (d): ln Z 30 por encima de la siguiente: BT-Settl se lo lleva todo y su intervalo es el suyo
        self.assertTrue(st["d"]["domina"])
        self.assertEqual(st["d"]["rejilla_dominante"], "bt")
        self.assertAlmostEqual(st["d"]["p50"], -0.10, places=3)
        # (e): la cuadratura de la anchura típica (semiancho 16–84 % de una normal = sd) y la dispersión
        self.assertAlmostEqual(st["e"]["anchura_tipica"], sd, places=3)
        self.assertAlmostEqual(st["e"]["total"], math.hypot(st["e"]["anchura_tipica"], float(np.std(m, ddof=1))),
                               places=12)
        self.assertIsNone(dr.estadisticas(grupos, deltas, None)["d"])

    def test_diamondback_es_una_rejilla(self):
        self.assertEqual(dr.familia("diamondback1600_f1"), "diamondback")
        self.assertEqual(dr.familia("diamondback1600_nc"), "diamondback")
        self.assertEqual(dr.familia("atmo_ceq1500"), "atmo")
        self.assertEqual(dr.familia("ames_cond"), "ames_cond")
        # dos cadenas de una rejilla pesan como una: la rejilla «b» no pesa el doble que «a»
        deltas = {"a": _normal(0.0, 0.01, 3000), "b1": _normal(1.0, 0.01, 3000), "b2": _normal(1.0, 0.01, 3000)}
        st = dr.estadisticas({"a": ["a"], "b": ["b1", "b2"]}, deltas)
        self.assertAlmostEqual(st["c"]["media"], 0.5, places=6)


class TablaTexTests(unittest.TestCase):
    TEX = "\n".join([
        r"\begin{table*}", r"\label{tab:teff_grids}", r"\begin{tabular}{l c c c c c c c}", r"\hline\hline",
        r"Grid & \(T_{\rm eff}\) (K) & \(\log g\) & \(A_V\) (mag) & \multicolumn{3}{c}{\(\Delta\) (dex)} & \(\Delta\ln Z\) \\",
        r"\hline", r"\multicolumn{8}{l}{Uno} \\",
        r"\quad BT-Settl & $2719^{+28}_{-30}$ & $4.55^{+0.10}_{-0.09}$ & $3.74^{+0.31}_{-0.35}$ & $-0.03$ & $+0.13$ & $+0.23$ & $0.0$ \\",
        r"\quad DRIFT-PHOENIX & $2841^{+23}_{-26}$ & $5.22^{+0.09}_{-0.09}$ & $3.76^{+0.28}_{-0.25}$ & $-0.13$ & $+0.13$ & $+0.23$ & $-52.5$ \\",
        r"\hline", r"\end{tabular}", r"\end{table*}"])

    def test_lee_el_cuerpo(self):
        filas = dr.filas_tabla_tex(self.TEX)
        self.assertEqual(len(filas), 3)
        self.assertTrue(filas[0].startswith(r"\multicolumn{8}{l}{Uno}"))
        self.assertIsNone(dr.celdas_delta(filas[0]))
        self.assertEqual(dr.celdas_delta(filas[1]), ["$-0.03$", "$+0.13$", "$+0.23$"])

    def test_compara(self):
        filas = dr.filas_tabla_tex(self.TEX)
        self.assertTrue(dr.compara_tabla(filas, filas)["iguales"])
        otra = list(filas)
        otra[2] = otra[2].replace("$-0.13$", "$-0.12$")
        c = dr.compara_tabla(otra, filas)
        self.assertFalse(c["iguales"])
        self.assertEqual([i for i, *_ in c["delta_distintas"]], [2])
        # otra celda distinta (no Δ): fila distinta, Δ iguales
        otra = list(filas)
        otra[1] = otra[1].replace("$0.0$", "$0.1$")
        c = dr.compara_tabla(otra, filas)
        self.assertEqual(len(c["distintas"]), 1)
        self.assertEqual(c["delta_distintas"], [])
        self.assertFalse(dr.compara_tabla(filas[:2], filas)["iguales"])


def _escribe_cadena(carpeta, slug, rejilla, av, r250, teff_mu, logz, n=3000):
    """Una cadena sintética como las de `teff_bayes.py ajusta` (json + npz), con T_eff normal y R, A_V fijos."""
    et = tc.etiqueta(slug, rejilla, av, r250=r250)
    teff = _normal(teff_mu, 30.0, n)
    logg = np.full(n, 4.0) + 0.05 * stats.norm.ppf((np.arange(n)[::-1] + 0.5) / n)
    avv = np.full(n, 2.2 if av == "prior" else float(av))
    r_cm = np.full(n, 1.5 * 7.1492e9)
    lbol = 4 * math.pi * r_cm ** 2 * 5.670374e-5 * teff ** 4 / 3.828e33
    nombres = ["teff", "logg", "r_rj", "d_pc", "av"]
    muestras = np.column_stack([teff, logg, np.full(n, 1.5), np.full(n, 140.0), avv])
    derivadas = np.column_stack([teff, logg, r_cm, np.full(n, np.nan), lbol])
    np.savez(carpeta / f"{et}.npz", muestras=muestras, derivadas=derivadas, nombres=np.array(nombres))
    post = {"teff": tb.percentiles(teff), "logg": tb.percentiles(logg), "av": tb.percentiles(avv)}
    d = dict(etiqueta=et, slug=slug, modo="libre", av="primaria", av_fija=None if av == "prior" else float(av),
             rejilla=rejilla, datos="todo", r_analisis=250 if r250 else 500, verdad=None, semilla=1, nlive=800,
             logz=logz, logl_max=logz + 10, posterior=post)
    (carpeta / f"{et}.json").write_text(json.dumps(d))
    return et


class TablaDesdeCadenasTests(unittest.TestCase):
    """La Tabla B.7 desde cadenas sintéticas: lo que lee el script (json + npz → Δ por muestra → filas) da la
    mediana de Δ de cada cadena, calculada aparte con `teff_bayes.margen`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.slug = tc.SLUGS[0]
        teffs = {("btsettl", False): 2600.0, ("drift", False): 2750.0, ("ames_cond", False): 2700.0,
                 ("btsettl", True): 2620.0}
        lnz = {("btsettl", False): 100.0, ("drift", False): 50.0, ("ames_cond", False): 40.0, ("btsettl", True): 30.0}
        for (rj, r), t in teffs.items():
            for i, a in enumerate(tc.AVS):
                _escribe_cadena(self.dir, self.slug, rj, a, r, t + 20 * i, lnz[(rj, r)])
        # ATMO en la misma carpeta: no entra en el cálculo principal
        for a in tc.AVS:
            _escribe_cadena(self.dir, self.slug, "atmo_ceq", a, False, 2400.0, 120.0)
        self.E = {s: dict(lacc0=3e-7, av0=1.8, a=1.13, ahav=0.79, lbol_lit=1e-3) for s in tc.SLUGS}

    def tearDown(self):
        self.tmp.cleanup()

    def _delta(self, rj, a, r250=False):
        z = np.load(self.dir / f"{tc.etiqueta(self.slug, rj, a, r250=r250)}.npz")
        e = self.E[self.slug]
        return tb.margen(z["derivadas"][:, 0], z["muestras"][:, 4], z["derivadas"][:, 4], e["lacc0"], e["av0"], e["a"],
                         e["ahav"])

    def test_filas_y_conjuntos(self):
        K = tc.Cadenas([self.dir])
        res = dr.calcula(K, ROOT, E=self.E)
        filas = res["tabla_b7"]["filas"]
        # cabecera, R = 500 (BT-Settl, DRIFT, AMES-Cond) y R = 250 (BT-Settl); sin ATMO
        self.assertEqual(len(filas), 5)
        self.assertNotIn("ATMO", " ".join(filas))
        orden = [("btsettl", False), ("drift", False), ("ames_cond", False), ("btsettl", True)]
        for fila, (rj, r) in zip(filas[1:], orden):
            esperado = [f"${float(np.percentile(self._delta(rj, a, r), 50)):+.2f}$" for a in tc.AVS]
            self.assertEqual(dr.celdas_delta(fila), esperado)
        # el cuerpo comparado consigo mismo a través de un LaTeX
        tex = "\\label{tab:teff_grids}\n" + "\n".join(filas) + "\n\\end{tabular}\n"
        self.assertTrue(dr.compara_tabla(filas, dr.filas_tabla_tex(tex))["iguales"])
        # (c) del conjunto r500 con prior: la mezcla a pesos iguales de las tres rejillas
        c = res["conjuntos"][f"{self.slug}|r500|prior"]
        self.assertEqual(c["n_rejillas"], 3)
        x, w = dr.mezcla([self._delta(rj, "prior") for rj in ("btsettl", "drift", "ames_cond")], [1 / 3] * 3)
        self.assertAlmostEqual(c["c"]["p50"], dr.resume(x, w)["p50"], places=12)
        self.assertAlmostEqual(c["por_rejilla"]["drift"]["p50"], float(np.percentile(self._delta("drift", "prior"), 50)),
                               places=12)
        # (d): BT-Settl tiene ln Z 50 más
        self.assertEqual(c["d"]["rejilla_dominante"], "btsettl")
        self.assertTrue(c["d"]["domina"])
        # el conjunto r250 va aparte; sin SPHINX no hay `siete`; sin --cadenas-atmo no hay variantes de ATMO
        self.assertIn(f"{self.slug}|r250|1.8", res["conjuntos"])
        self.assertNotIn(f"{self.slug}|siete|1.8", res["conjuntos"])
        self.assertFalse(any("atmo" in k for k in res["conjuntos"]))

    def test_atmo_como_variante(self):
        K = tc.Cadenas([self.dir])
        res = dr.calcula(K, ROOT, KA=tc.Cadenas([self.dir]), E=self.E)
        v = res["conjuntos"][f"{self.slug}|r500_atmo|prior"]
        self.assertEqual(v["n_rejillas"], 4)
        self.assertIn("atmo", v["por_rejilla"])
        self.assertNotIn("atmo", res["conjuntos"][f"{self.slug}|r500|prior"]["por_rejilla"])


# ------------------------------------------------------------------ datos reales
PROYECTO = Path(os.environ.get("MUSE_PROJECT_ROOT", ROOT))
CADENAS = Path(os.environ.get("MUSE_TEFF_CADENAS", "~/muse_noche_20261024_c4/cadenas")).expanduser()


@pytest.mark.external_data
class DatosRealesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tex = PROYECTO / "paper" / "main.tex"
        runs = [PROYECTO / "runs" / json.loads(f.read_text())["paper"]["run"]
                for f in sorted((ROOT / "targets").glob("*.json")) if "paper" in json.loads(f.read_text())]
        if not CADENAS.is_dir() or not tex.exists() or not runs or not all(p.exists() for p in runs):
            raise unittest.SkipTest("faltan las cadenas, el paper o los runs (MUSE_PROJECT_ROOT, MUSE_TEFF_CADENAS)")
        cls.raiz_antes = tb.ROOT
        tb.fija_raiz(PROYECTO)
        cls.K = tc.Cadenas([CADENAS])
        cls.tex = tex.read_text()
        cls.res = dr.calcula(cls.K, PROYECTO, tex=cls.tex)

    @classmethod
    def tearDownClass(cls):
        tb.fija_raiz(cls.raiz_antes)

    def test_la_tabla_b7_del_paper_sale_de_las_cadenas(self):
        comp = self.res["tabla_b7"]["comparacion"]
        self.assertEqual(comp["distintas"], [], "la Tabla B.7 no se reproduce fila a fila")
        self.assertGreater(comp["n_impresas"], 0)

    def test_la_mezcla_queda_entre_las_rejillas(self):
        for k, v in self.res["conjuntos"].items():
            self.assertLessEqual(v["a"]["min"], v["c"]["p50"] + 1e-12, k)
            self.assertLessEqual(v["c"]["p50"], v["a"]["max"] + 1e-12, k)

    def test_un_modelo_reconstruido_da_el_logl_de_su_cadena(self):
        try:
            d = self.K.por_etiqueta[tc.etiqueta(tc.SLUGS[0])]
            fila = dr.residuos_cadena(d)
        except (OSError, SystemExit, RuntimeError) as exc:  # sin las rejillas de /mnt/2TB
            self.skipTest(f"no se puede reconstruir el modelo: {exc}")
        self.assertTrue(fila["reproduce"], (fila["loglike"], fila["logl_max"]))
        self.assertAlmostEqual(sum(b["fraccion_chi2"] for b in fila["bandas"] if b["n"]), 1.0, places=9)
        self.assertEqual(sum(b["n"] for b in fila["bandas"]), fila["n"])


if __name__ == "__main__":
    unittest.main()
