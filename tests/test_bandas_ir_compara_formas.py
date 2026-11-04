"""Los espectros del IR repetidos en una banda (`scripts/bandas_ir_compara.py`, apéndice D.12; 2026-11-03).

Las clases sintéticas no leen disco (salvo `targets/`, que es del clon): cómo se clasifica una cadena por su json
(forma, objeto, anclas y ruido normalizados), qué cadenas entran, el error con dos cadenas de la misma clave, el
cruce de signo de Δ, lo que se mueve frente a «ref», y la lectura de una tabla del tex.
`DatosRealesTests` (marca `external_data` y `slow`) lee las cadenas de referencia y las de los lotes del D.12
(`MUSE_BANDAS_IR_REFERENCIAS` y `MUSE_BANDAS_IR_LOTES`, carpetas separadas por `:`; por defecto las de la máquina del
autor) y comprueba invariantes, no cifras: que cada celda de las tablas tiene su cadena, que la forma de cada cadena es
la que declara su json, y que en las plantillas cambiar el ruido más cambiar los píxeles da el cambio total. Si el
`paper/main.tex` de `MUSE_PROJECT_ROOT` ya lleva el D.12, compara además sus Tablas D.3 y D.4.
"""
import os
import unittest
from pathlib import Path

import pytest

from tests._datos import cadenas_teff, carga_script, raiz_proyecto, raiz_temporal, requiere

#: las cadenas de referencia que no son del paper y las de los lotes del D.12 (máquina del autor, 2026-10-26 a 11-03)
REFERENCIAS_POR_DEFECTO = (
    "~/muse_sesion_20261026/claudio/atmo/res/atmo", "~/muse_sesion_20261026/claudio/atmo/res/atmo1500",
    "~/muse_sesion_20261026/claudio/atmo/res/atmo_datos", "~/muse_sesion_20261026/claudio/atmo/res/plantillas",
    "~/muse_noche_20261027/atmo_s2/res/atmo_s2", "~/muse_noche_20261027/atmo_s2/res/atmo1500_s2",
    "~/muse_noche_20261027/atmo_s2/res/atmo_datos_s2", "~/muse_noche_20261027/atmo_s2/res/plantillas_s1",
    "~/muse_sesion_20261018/teff_datos_logg5/res")
LOTES_POR_DEFECTO = (
    "~/muse_noche_20261030_plantillas_atmo/res", "~/muse_noche_20261030_anclas_ir/res", "~/muse_noche_20261031_quita_ir/res",
    "~/muse_noche_20261031_quita_ir_reales/res", "~/muse_noche_20261101_pendiente_ir/res",
    "~/muse_noche_20261102_banda_una_vez/res", "~/muse_noche_20261102_drift_ir/res", "~/muse_noche_20261103_rejillas_b7/res")
PROPIAS = {"ROXs12b": "K", "ROXs42Bb": "JHK"}


def _q(p50):
    return dict(p2_5=p50 - 20, p16=p50 - 10, p50=p50, p84=p50 + 10, p97_5=p50 + 20)


def _json(etiqueta="x", **kw):
    d = dict(etiqueta=etiqueta, slug="ROXs12b", modo="libre", av="primaria", av_fija=None, rejilla="btsettl",
             datos="todo", r_analisis=500, nlive=800, semilla=1, logz=100.0, logz_err=0.2,
             posterior=dict(teff=_q(2700.0), logg=_q(4.5), av=_q(3.5)), _json=f"{etiqueta}.json")
    d.update(kw)
    return d


class _C:
    def __init__(self, *ds):
        self.por_etiqueta = {d["etiqueta"]: d for d in ds}


class ClasificaTests(unittest.TestCase):
    def setUp(self):
        self.m = carga_script("bandas_ir_compara")

    def test_anclas_propias_de_targets(self):
        for slug, a in PROPIAS.items():
            self.assertEqual(self.m.anclas_propias(slug), a)

    def test_forma_y_opciones_son_inversas(self):
        for f in self.m.FORMAS:
            self.assertEqual(self.m.forma(*self.m.opciones(f)), f)
        self.assertEqual(self.m.opciones("incl", 0.1), (False, 0.1))

    def test_entra(self):
        e = self.m.entra
        self.assertTrue(e(_json(), de_variantes=False))
        self.assertFalse(e(_json(banda_una_vez=True), de_variantes=False))     # una variante no es una referencia
        self.assertTrue(e(_json(banda_una_vez=True, pendiente_ir=0.3), de_variantes=True))
        self.assertTrue(e(_json(quita_ir=["NIFS_J"]), de_variantes=True))
        self.assertFalse(e(_json(banda_una_vez=True, cov_banda=True), de_variantes=True))   # otra variante
        self.assertFalse(e(_json(mascara_muse="sin_o2b"), de_variantes=True))
        self.assertFalse(e(_json(opcion_nueva=1), de_variantes=True))                       # clave desconocida
        self.assertFalse(e(_json(modo="evo_solo"), de_variantes=True))
        self.assertFalse(e(_json(av="libre"), de_variantes=True))
        v = dict(plantilla="TWA26", spt="M9")
        self.assertTrue(e(_json(verdad=v, nlive=250, semilla=0), de_variantes=False))
        self.assertFalse(e(_json(verdad=v, nlive=400, semilla=0), de_variantes=False))      # otros puntos vivos
        self.assertFalse(e(_json(verdad=dict(teff=2550.0, logg=3.75, av=1.8, r_rj=2.2), nlive=250), de_variantes=False))

    def test_clave_normaliza_anclas_y_ruido(self):
        c = self.m.clave
        v = dict(plantilla="TWA26", spt="M9")
        a = c(_json(verdad=v, nlive=250), PROPIAS)
        self.assertEqual(a, c(_json(verdad=v, nlive=250, anclas_ir="K", ruido_de="ROXs12b"), PROPIAS))
        self.assertNotEqual(a, c(_json(verdad=v, nlive=250, ruido_de="ROXs42Bb"), PROPIAS))
        self.assertNotEqual(a, c(_json(verdad=v, nlive=250, anclas_ir="JHK"), PROPIAS))
        d = c(_json(quita_ir=["OSIRIS_J", "NIFS_J"]), PROPIAS)
        self.assertEqual(d, c(_json(quita_ir=["NIFS_J", "OSIRIS_J"]), PROPIAS))   # el orden no cuenta
        self.assertEqual(d[0], "datos")
        # los puntos vivos no son parte de la clave: dos cadenas iguales con otro nlive chocan
        self.assertEqual(c(_json(nlive=500), PROPIAS), c(_json(nlive=800), PROPIAS))

    def test_indice_rechaza_dos_cadenas_con_la_misma_clave(self):
        idx = self.m.indice(_C(_json("a")), _C(_json("b", banda_una_vez=True)), PROPIAS)
        self.assertEqual(len(idx), 2)
        with self.assertRaises(SystemExit):
            self.m.indice(_C(_json("a"), _json("b", nlive=500)), _C(), PROPIAS)


class CruceYMueveTests(unittest.TestCase):
    def setUp(self):
        self.m = carga_script("bandas_ir_compara")

    def test_cruce_lineal(self):
        self.assertAlmostEqual(self.m.cruce(-0.1, 0.2), 1.8 + 1.2 / 3)
        self.assertAlmostEqual(self.m.cruce(-0.3, -0.1), 3.6)         # extrapolado
        self.assertIsNone(self.m.cruce(0.1, 0.1))

    def _c(self, deltas, avs=(3.6, 3.5, 3.4, 3.3)):
        """{av: {forma: fila}} con Δ por A_V y forma; T = 2700 K, y 100 K más en «las dos»; A_V con el prior `avs`."""
        out = {}
        for av, ds in zip((1.8, 3.0, None), deltas):
            out[av] = {f: dict(t=2700.0 + (100.0 if f == "las dos" else 0.0), delta=d,
                               av=(a if av is None else av)) for f, d, a in zip(self.m.FORMAS, ds, avs)}
        return out

    def test_mueve_y_cruces(self):
        c = self._c([(-0.10, -0.10, 0.05, -0.20), (0.20, 0.10, 0.25, -0.05), (0.30, 0.20, 0.35, 0.10)])
        m = self.m.mueve(c)
        self.assertAlmostEqual(m["max_dT"], 100.0)
        self.assertAlmostEqual(m["max_dD"], 0.25)
        self.assertEqual(m["n_sobre_umbral"], 0)
        self.assertEqual(m["delta_prior"], [0.30, 0.20, 0.35, 0.10])
        cr = self.m.cruces(c)
        self.assertTrue(cr["ref"]["dentro"] and cr["peso"]["dentro"])
        self.assertFalse(cr["incl"]["dentro"])        # positiva ya a 1.8
        self.assertFalse(cr["las dos"]["dentro"])     # negativa todavía a 3.0
        self.assertAlmostEqual(cr["ref"]["av_prior_menos_cruce"], 3.6 - (1.8 + 1.2 * 0.1 / 0.3))


class TexTests(unittest.TestCase):
    TABLA = r"""\begin{table*}
\label{tab:x}
\begin{tabular}{l l c}
\hline\hline
 & & \multicolumn{1}{c}{T} \\
Grid & \(A_V\) & Ref. \\
\hline
\multicolumn{3}{l}{ROXs~12~b} \\
\quad ATMO~2020 & \(1.8\) & \(2354\) \\
 & prior & \(-0.03\) \\
\quad Diamondback, \(f_{\rm sed} = 1\)\tablefootmark{a} & (\(3.2\)) & -- \\
\hline
\end{tabular}"""

    def test_filas_y_numeros(self):
        m = carga_script("bandas_ir_compara")
        filas = m._filas_tex(self.TABLA, "tab:x")
        self.assertEqual(len(filas), 3)                 # sin la cabecera ni el título del objeto
        celdas = [c.strip() for c in filas[1].rstrip("\\ ").split("&")]
        self.assertEqual(m._num(celdas[1]), [])         # «prior»
        self.assertEqual(m._num(celdas[2]), [-0.03])
        self.assertEqual(m._num(r"\(f_{\rm sed} = 1\)\tablefootmark{a}"), [1.0])
        self.assertIsNone(m._filas_tex(self.TABLA, "tab:otra"))


def _dirs(var, defecto):
    v = os.environ.get(var)
    partes = [p for p in v.split(os.pathsep) if p] if v else list(defecto)
    return [Path(p).expanduser() for p in partes]


@pytest.mark.external_data
@pytest.mark.slow
@requiere("runs", "cadenas")
@unittest.skipUnless(all(d.is_dir() for d in _dirs("MUSE_BANDAS_IR_LOTES", LOTES_POR_DEFECTO))
                     and all(d.is_dir() for d in _dirs("MUSE_BANDAS_IR_REFERENCIAS", REFERENCIAS_POR_DEFECTO)),
                     "sin las cadenas del D.12 (define MUSE_BANDAS_IR_LOTES y MUSE_BANDAS_IR_REFERENCIAS)")
class DatosRealesTests(unittest.TestCase):
    """Las cadenas de las Tablas B.6/B.7 y las de los lotes del D.12: invariantes, no cifras."""

    @classmethod
    def setUpClass(cls):
        cls.m = carga_script("bandas_ir_compara")
        tc = carga_script("teff_cifras_texto")
        refs = cadenas_teff() + _dirs("MUSE_BANDAS_IR_REFERENCIAS", REFERENCIAS_POR_DEFECTO)
        with raiz_temporal() as raiz:
            cls.res = cls.m.calcula(tc.Cadenas(refs), tc.Cadenas(_dirs("MUSE_BANDAS_IR_LOTES", LOTES_POR_DEFECTO)), raiz)

    def test_cada_celda_tiene_su_cadena_y_su_forma(self):
        m = self.m
        for slug, fams in self.res["tabla_d3"].items():
            for fam, avs in fams.items():
                for av, formas in avs.items():
                    self.assertEqual(list(formas), list(m.FORMAS))
                    self.assertEqual(len({x["etiqueta"] for x in formas.values()}), 4)
                    for f, x in formas.items():
                        with self.subTest(slug=slug, rejilla=fam, av=av, forma=f):
                            e = x["etiqueta"]
                            self.assertEqual("bandaunavez" in e, f in ("peso", "las dos"))
                            self.assertEqual("pendir" in e, f in ("incl", "las dos"))

    def test_tablas_completas(self):
        self.assertEqual(len(self.res["filas_d3"]["filas"]), 2 * 3 * 3)
        self.assertEqual(len(self.res["filas_d4"]), sum(len(v) for v in self.m.D4.values()))
        for f in self.res["filas_d4"]:
            self.assertGreaterEqual(f[0], 0)
            self.assertGreaterEqual(f[1], 0)

    def test_las_formas_sin_peso_ni_inclinacion_son_las_de_referencia(self):
        """«ref» no lleva ninguna opción: su A_V con el prior es la de la cadena de las Tablas B.6/B.7."""
        for slug, fams in self.res["tabla_d3"].items():
            for fam, avs in fams.items():
                e = avs[None]["ref"]["etiqueta"]
                self.assertNotIn("bandaunavez", e)
                self.assertNotIn("pendir", e)

    def test_ruido_mas_pixeles_es_el_total(self):
        """Δ_ruido + Δ_píxeles = Δ_total en cada plantilla (las dos mitades de los dos caminos)."""
        for s in (0, 1):
            filas = self.res["plantillas"][f"atmo_todo_s{s}"]["por_plantilla"]
            self.assertEqual(len(filas), 12)
            for f in filas:
                with self.subTest(semilla=s, plantilla=f["plantilla"]):
                    self.assertAlmostEqual(f["ruido"] + f["pixeles"], f["total"], places=9)

    def test_contra_el_paper_si_lleva_el_d12(self):
        tex = raiz_proyecto() / "paper" / "main.tex"
        if not tex.exists() or "\\label{tab:bands_grids}" not in tex.read_text():
            self.skipTest(f"{tex} no lleva el D.12")
        dif = self.m.compara_tex(tex.read_text(), self.res)
        self.assertEqual(dif["tablas"], [])


if __name__ == "__main__":
    unittest.main()
