"""Cifras de la prosa del paper desde las cadenas de T_eff (`scripts/teff_cifras_texto.py`) y la regla del tipo
equivalente (`scripts/teff_bayes.py`, `tipo_equivalente`).

Las primeras clases no leen nada de disco: la regla del tipo equivalente con plantillas sintéticas (y con las T_eff
de las plantillas de las cadenas c0, tecleadas, que dan los tipos impresos), el solape de intervalos, los redondeos
del texto, las frases que se componen (el solape del 68 %, la lista de rejillas por debajo del nivel, la dispersión
entre rejillas), la búsqueda de una frase en el LaTeX y la carga de cadenas de varias carpetas. `CadenasRealesTests`
(marca `external_data`) corre el script entero sobre las cadenas del paper y el `main.tex` del clon que tiene
`runs/` y `paper/` (`MUSE_PROJECT_ROOT`, por defecto el repo; `MUSE_TEFF_CADENAS`, las carpetas separadas por `:`,
por defecto las del paper); si falta algo se salta, no falla.
"""
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

from scripts import teff_bayes as tb  # noqa: E402
from scripts import teff_cifras_texto as tc  # noqa: E402


def setUpModule():
    # los dos objetos del paper salen de targets/ (el script no nombra objetos en el código)
    tc.fija_objetos(ROOT)


def _pl(spt, teff, nombre=None):
    from musepipe.constants import spt_code
    return dict(plantilla=nombre or spt, spt=spt, codigo=spt_code(spt), teff=float(teff))


class TipoEquivalenteTests(unittest.TestCase):
    def test_un_par_que_encierra_da_el_par(self):
        cal = [_pl("M7", 2800), _pl("M7.5", 2700), _pl("M8", 2600), _pl("M9", 2400)]
        t = tb.tipo_equivalente(cal, 2650.0)
        self.assertEqual((t["clase"], t["tipos"]), ("par", ("M7.5", "M8")))
        self.assertEqual(tc.tipo_tabla(t), "M7.5--M8")
        self.assertEqual(tc.tipo_prosa(t), "M7.5--M8")

    def test_dos_plantillas_del_mismo_tipo_dan_un_tipo(self):
        cal = [_pl("M8", 2600), _pl("M9", 2500, "a"), _pl("M9", 2300, "b"), _pl("L0", 2200)]
        t = tb.tipo_equivalente(cal, 2400.0)
        self.assertEqual((t["clase"], t["tipos"]), ("una", ("M9",)))

    def test_los_empates_conservan_el_orden_de_entrada(self):
        # dos M9: el par consecutivo depende de cuál va primero (el orden de las etiquetas)
        a, b = _pl("M9", 2500, "a"), _pl("M9", 2300, "b")
        cal = [_pl("M8", 2600), a, b, _pl("L0", 2200)]
        self.assertEqual(tb.pares_que_encierran(cal, 2550.0), [(cal[0], a)])
        self.assertEqual(tb.pares_que_encierran([cal[0], b, a, cal[3]], 2550.0), [(cal[0], b)])

    def test_saturada_contra_el_borde_se_lee_la_mas_cercana(self):
        # DRIFT-PHOENIX llega a 3000 K: con el ruido de ROXs 12 b las M6.5-M8 se amontonan en 2960-2990 K
        cal = [_pl("M6.5", 2988), _pl("M7", 2973), _pl("M7.5", 2958), _pl("M8", 2985), _pl("M8.5", 2447)]
        self.assertEqual(tb.tipo_equivalente(cal, 2841.0)["tipos"], ("M8", "M8.5"))  # sin rejilla: el par
        t = tb.tipo_equivalente(cal, 2841.0, rejilla="drift")
        self.assertEqual((t["clase"], t["tipos"]), ("una", ("M8",)))
        self.assertIn("M8", t["saturadas"])

    def test_mas_fria_que_todas_es_posterior_al_tipo_mas_tardio(self):
        cal = [_pl("M9", 2214, "USco"), _pl("L0", 2307), _pl("L1", 2199, "2MASS1207"), _pl("L1", 2256, "WISE0528")]
        t = tb.tipo_equivalente(cal, 2191.0, rejilla="btsettl")
        self.assertEqual((t["clase"], t["tipos"]), ("posterior", ("L1",)))
        self.assertEqual(tc.tipo_tabla(t), r"\(\geq\)L1")
        self.assertEqual(tc.tipo_prosa(t), "L1 or later")

    def test_mas_caliente_que_todas_es_anterior_al_mas_temprano(self):
        cal = [_pl("M6.5", 2800), _pl("M7", 2750)]
        t = tb.tipo_equivalente(cal, 2900.0)
        self.assertEqual((t["clase"], t["tipos"]), ("anterior", ("M6.5",)))
        self.assertEqual(tc.tipo_prosa(t), "M6.5 or earlier")

    def test_dos_pares_es_ambiguo(self):
        cal = [_pl("M7", 2700), _pl("M8", 2500), _pl("M8.5", 2650), _pl("M9", 2400)]
        with self.assertRaises(ValueError):
            tb.tipo_equivalente(cal, 2600.0)

    def test_con_las_plantillas_de_las_cadenas_c0_da_lo_impreso(self):
        # T_eff ajustadas (p50) de las 12 plantillas con las cadenas c0 del 2026-10-24, en el orden de sus etiquetas,
        # y la T_eff con prior (s1) de cada compañero: la Tabla B.5 imprime M7.5--M8, M8, ≥L1 y ≥L1
        nombres = ["2MASS0046", "2MASS1207", "CHSM17173", "J1119", "LM601", "LM717", "TWA26", "TWA29", "USco1607",
                   "USco1610", "VISTA", "WISE0528"]
        tipos = ["L0", "L1", "M8", "M7", "M7.5", "M6.5", "M9", "M9.5", "M9", "M8.5", "M8.5", "L1"]
        T = {("ROXs12b", "btsettl"): [2238, 2165, 2591, 2751, 2738, 2789, 2419, 2349, 2188, 2440, 2425, 2212],
             ("ROXs12b", "drift"): [2215, 2230, 2985, 2973, 2958, 2988, 2648, 2405, 2152, 2447, 2479, 2188],
             ("ROXs42Bb", "btsettl"): [2307, 2199, 2609, 2782, 2721, 2754, 2559, 2444, 2214, 2510, 2526, 2256],
             ("ROXs42Bb", "drift"): [2254, 2231, 2769, 2959, 2874, 2908, 2646, 2421, 2158, 2531, 2581, 2219]}
        comp = {("ROXs12b", "btsettl"): 2718.7, ("ROXs12b", "drift"): 2841.3, ("ROXs42Bb", "btsettl"): 2190.7,
                ("ROXs42Bb", "drift"): 2144.6}
        esperado = {("ROXs12b", "btsettl"): "M7.5--M8", ("ROXs12b", "drift"): "M8",
                    ("ROXs42Bb", "btsettl"): r"\(\geq\)L1", ("ROXs42Bb", "drift"): r"\(\geq\)L1"}
        for k, ts in T.items():
            cal = [_pl(s, t, n) for s, t, n in zip(tipos, ts, nombres)]
            self.assertEqual(tc.tipo_tabla(tb.tipo_equivalente(cal, comp[k], rejilla=k[1])), esperado[k], k)


class SolapeTests(unittest.TestCase):
    def test_solape_de_intervalos(self):
        self.assertTrue(tc.se_solapan((1.0, 2.0), (1.5, 3.0)))
        self.assertTrue(tc.se_solapan((1.0, 2.0), (2.0, 3.0)))  # tocarse cuenta
        self.assertTrue(tc.se_solapan((1.0, 4.0), (2.0, 3.0)))  # uno dentro del otro
        self.assertFalse(tc.se_solapan((1.0, 2.0), (2.1, 3.0)))
        for a, b in (((1, 2), (1.5, 3)), ((1, 2), (2.1, 3)), ((5, 6), (1, 2))):
            self.assertEqual(tc.se_solapan(a, b), tc.se_solapan(b, a))

    def test_frase_del_68_por_ciento(self):
        sol = {("ROXs12b", "teff"): False, ("ROXs12b", "logg"): False, ("ROXs42Bb", "teff"): True,
               ("ROXs42Bb", "logg"): True}
        self.assertEqual(tc.frase_solape68(sol), "do not overlap in temperature or gravity for ROXs~12~b, and they "
                                                 "overlap in both for ROXs~42B~b")
        with self.assertRaises(ValueError):  # solo en una de las dos: otra frase
            tc.frase_solape68({**sol, ("ROXs42Bb", "teff"): False})
        with self.assertRaises(ValueError):  # los dos igual: otra frase
            tc.frase_solape68({**sol, ("ROXs12b", "teff"): True, ("ROXs12b", "logg"): True})


class FormatoTests(unittest.TestCase):
    def test_redondeos(self):
        self.assertEqual(tc.r_mitad(2125, 50), 2150)  # la mitad hacia arriba
        self.assertEqual(tc.k50(2589.3), "2600")
        self.assertEqual(tc.k5(4.0), "5")
        self.assertEqual(tc.k5(128.0), "130")
        self.assertEqual(tc.arriba(10)(216.3), "220")
        self.assertEqual(tc.arriba(10)(220.0), "220")  # un múltiplo exacto no sube
        self.assertEqual(tc.abajo(10)(136.0), "130")
        self.assertEqual(tc.cota_mag(0.93), "1")
        self.assertEqual(tc.cota_mag(1.0), "1")
        self.assertEqual(tc.cota_mag(1.0657), "1.1")

    def test_signos_y_ceros(self):
        self.assertEqual(tc.dec(-0.001, 2), "0.00")
        self.assertEqual(tc.s2(-0.004), "0.00")
        self.assertEqual(tc.s2(0.128), "+0.13")
        self.assertEqual(tc.s2(-0.13), "-0.13")

    def test_rangos(self):
        self.assertEqual(tc.rango(2.018, 2.399, tc.d1), r"\(2.0\)--\(2.4\)")
        self.assertEqual(tc.rango(0.78, 0.82, tc.d1), r"\(0.8\)")  # los dos extremos iguales: un valor
        self.assertEqual(tc.rango(0.128, 0.23, tc.d1, tc.d2), r"\(0.1\)--\(0.23\)")
        self.assertEqual(tc.hasta(-0.13, 0.001, tc.s2), r"\(-0.13\) to \(0.00\)")
        self.assertEqual(tc.por_ciento(16.2, 25.9), r"\(16\)--\(26\%\)")

    def test_cruce_lineal(self):
        self.assertAlmostEqual(tc.cruce_lineal([(1.8, -0.1), (3.0, 0.1)]), 2.4)
        self.assertAlmostEqual(tc.cruce_lineal([(1.8, -0.1), (2.5, -0.05), (3.0, 0.05)]), 2.75)
        self.assertIsNone(tc.cruce_lineal([(1.8, 0.05), (3.0, 0.26)]))


class FrasesCompuestasTests(unittest.TestCase):
    def test_lista_de_rejillas_por_debajo_del_nivel(self):
        neg = [("BT-Settl", "1.8"), ("BT-Settl", "1.8"), ("BT-Settl", "prior"), ("BT-Settl", "prior"),
               ("AMES-Cond", "1.8"), ("AMES-Cond", "3")]
        self.assertEqual(tc.lista_por_debajo(neg), r"with BT-Settl at \(A_V = 1.8\)~mag or with the prior on \(A_V\), "
                                                   r"and with AMES-Cond at \(1.8\) and \(3.0\)~mag")
        self.assertEqual(tc.lista_por_debajo([("BT-Settl", "prior")]), r"with BT-Settl with the prior on \(A_V\)")

    def _c(self, disp):
        return {"rejillas": {"dispersion": {k: dict(dispersion=v, rejilla_extrema=g, teff_extrema=t)
                                            for k, (v, g, t) in disp.items()}}}

    def test_dispersion_entre_rejillas(self):
        c = self._c({"ROXs12b_1.8": (136.0, "x", 0), "ROXs12b_3": (159.0, "x", 0), "ROXs12b_prior": (213.0, "x", 0),
                     "ROXs42Bb_1.8": (239.0, "x", 0), "ROXs42Bb_3": (742.0, "ames_cond", 2888.0),
                     "ROXs42Bb_prior": (420.4, "diamondback1600_f1", 1923.0)})
        self.assertEqual(tc.dispersion(c), r"\(130\)--\(240\)~K (\(420\) and \(740\)~K for ROXs~42B~b with the prior "
                                           r"on \(A_V\) and at \(A_V = 3.0\)~mag, where AMES-Cond gives \(2890\)~K)")
        sin = self._c({k: (v, "x", 0) for k, v in (("ROXs12b_1.8", 141.0), ("ROXs42Bb_3", 295.0))})
        self.assertEqual(tc.dispersion(sin), r"\(140\)--\(300\)~K")
        with self.assertRaises(ValueError):  # una por compañero: otra frase
            tc.dispersion(self._c({"ROXs12b_1.8": (400.0, "x", 0), "ROXs42Bb_3": (742.0, "x", 0),
                                   "ROXs42Bb_1.8": (200.0, "x", 0)}))


class BuscaEnElTexTests(unittest.TestCase):
    TEX = ("% cabecera: 2150-2330 K\n\\section{A}\nFor ROXs~42B~b, both grids give\n"
           "    \\(T_{\\rm eff} = 2150\\)--\\(2330\\)~K for \\(A_V = 1.8\\)--\\(3.0\\)~mag. % nota\n"
           "It is \\(16\\)--\\(26\\%\\), against\n\\(30\\)--\\(42\\%\\).\n")

    def test_normaliza_quita_comentarios_y_colapsa_blancos(self):
        n, pos = tc.normaliza(self.TEX)
        self.assertNotIn("cabecera", n)
        self.assertNotIn("nota", n)
        self.assertIn(r"both grids give \(T_{\rm eff} = 2150\)", n)
        self.assertIn(r"\(16\)--\(26\%\), against \(30\)", n)  # \% no es un comentario
        self.assertEqual(len(n), len(pos))
        i = n.index("For ROXs")
        self.assertEqual(self.TEX.count("\n", 0, pos[i]) + 1, 3)

    def test_evalua_encuentra_falla_y_salta(self):
        C = tc.Cifra
        c = {"x": {"T": [2151.0, 2334.0]}, "falta": {"faltan": ["una"]}}
        cs = [C("ok", "Ap. B", r"both grids give \(T_{\rm eff} = «T»~K for", dict(T=lambda c: tc.rango(*c["x"]["T"], tc.k10)[2:]),
                guarda=lambda c: True),
              C("otra", "Ap. B", r"both grids give \(T_{\rm eff} = «T»~K for", dict(T=lambda c: "2160\\)--\\(2330\\)")),
              C("guarda", "Ap. B", r"It is «a», against", dict(a=lambda c: tc.por_ciento(16, 26)), guarda=lambda c: False),
              C("sin", "Ap. B", "x «a»", dict(a=lambda c: tc.parte(c, "falta"))),
              C("conocida", "Ap. B", "no está «a»", dict(a=lambda c: "1"), discrepancia="no sale")]
        r = {x["id"]: x for x in tc.evalua(c, cs, self.TEX)}
        self.assertEqual((r["ok"]["estado"], r["ok"]["linea"], r["ok"]["guarda"]), ("en el paper", 3, True))
        self.assertEqual(r["otra"]["estado"], "NO está en el paper")
        self.assertEqual((r["guarda"]["estado"], r["guarda"]["guarda"]), ("en el paper", False))
        self.assertEqual(r["sin"]["estado"], "sin cadenas")
        self.assertEqual([k for k, x in r.items() if tc.falla(x)], ["otra", "guarda", "sin"])
        self.assertEqual(tc.evalua(c, cs[:1])[0]["estado"], "calculada")  # sin tex: solo la escribe


class CargaDeCadenasTests(unittest.TestCase):
    def _cadena(self, carpeta, etiqueta, npz=True, **extra):
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / f"{etiqueta}.json").write_text(json.dumps(dict(etiqueta=etiqueta, slug="ROXs12b", **extra)))
        if npz:
            np.savez(carpeta / f"{etiqueta}.npz", x=np.zeros(2))

    def test_gana_la_primera_carpeta_y_se_pide_por_etiqueta(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "nueva", Path(td) / "vieja"
            self._cadena(a / "A1" / "res", "c1", datos="todo")
            self._cadena(b, "c1", datos="todo")
            self._cadena(b, "c2", datos="ir")
            self._cadena(b, "c3", npz=False)  # sin npz: no es una cadena
            (b / "otro.json").write_text(json.dumps(dict(etiqueta="no_coincide", slug="ROXs12b")))
            np.savez(b / "otro.npz", x=np.zeros(1))
            K = tc.Cadenas([a, b])
            self.assertEqual(sorted(K.por_etiqueta), ["c1", "c2"])
            self.assertTrue(K.por_etiqueta["c1"]["_json"].startswith(str(a)))
            self.assertEqual([s["etiqueta"] for s in K.sombreadas], ["c1"])
            with self.assertRaises(tc.Faltan) as cm:
                K.exige(["c1", "c3"])
            self.assertEqual(cm.exception.args[0], ["c3"])
            self.assertEqual(len(K.exige(["c1", "c2"])), 2)
            self.assertEqual(K.procedencia(), {str(a): 1, str(b): 1})
            self.assertEqual(K.fuera_de_la_primera(), {"solo IR": 1})

    def test_etiquetas_como_las_de_ajusta(self):
        self.assertEqual(tc.etiqueta("ROXs12b"), "ROXs12b_libre_avprimaria_s1")
        self.assertEqual(tc.etiqueta("ROXs12b", "drift", "1.8"), "ROXs12b_drift_libre_avprimaria_avfija1.8_s1")
        self.assertEqual(tc.etiqueta("ROXs42Bb", "sphinx", "3", r250=True), "ROXs42Bb_sphinx_libre_avprimaria_avfija3_r250_s1")
        self.assertEqual(tc.etiqueta("ROXs42Bb", "ames_cond", "prior", datos="muse"),
                         "ROXs42Bb_ames_cond_libre_avprimaria_solomuse_s1")
        self.assertEqual(tc.etiqueta("ROXs12b", av="2.5", semilla=3), "ROXs12b_libre_avprimaria_avfija2.5_s3")


# ---------------------------------------------------------------------------------------------- datos reales
PROYECTO = Path(os.environ.get("MUSE_PROJECT_ROOT", ROOT))
CADENAS = [Path(p).expanduser() for p in os.environ.get("MUSE_TEFF_CADENAS", ":".join((
    "~/muse_noche_20261024_c4/cadenas", "~/muse_sesion_20261018/teff_datos_logg5/res",
    "~/muse_sesion_20261016/teff_datos/res_drift", "~/muse_noche_20261021/res",
    "~/muse_sesion_20261014/teff_mcmc/res"))).split(":")]


@pytest.mark.external_data
class CadenasRealesTests(unittest.TestCase):
    """Las 167 cadenas c0 + 22 que no leen MUSE contra el `main.tex` del paper (2026-10-26)."""

    @classmethod
    def setUpClass(cls):
        tex = PROYECTO / "paper" / "main.tex"
        runs = [PROYECTO / "runs" / r / "stages" / "stage_g3_qc.json" for r in ("ROXs12b_invvar", "ROXs42Bb_realigned")]
        if not all(p.exists() for p in runs) or not tex.exists() or not all(d.is_dir() for d in CADENAS):
            raise unittest.SkipTest("faltan los runs del paper, paper/main.tex o las carpetas de cadenas "
                                    "(MUSE_PROJECT_ROOT, MUSE_TEFF_CADENAS)")
        cls.K = tc.Cadenas(CADENAS)
        cls.calc = tc.Calculo(cls.K, PROYECTO)
        cls.tex = tex.read_text()
        cls.res = tc.evalua(cls.calc.c, tc.cifras(), cls.tex)

    def test_todas_las_frases_estan_en_el_paper(self):
        no = {r["id"]: r.get("error") or r["estado"] for r in self.res if r["estado"] != "en el paper"}
        self.assertEqual(no, {})
        self.assertEqual([r["id"] for r in self.res if tc.falla(r)], [])
        self.assertTrue(all(r["guarda"] is not False for r in self.res))

    def test_las_cadenas_usadas(self):
        if CADENAS[0].name != "cadenas":
            self.skipTest("otras carpetas de cadenas")
        self.assertEqual(len(self.K.usadas), 189)
        self.assertEqual(self.K.fuera_de_la_primera(), {"sin espectro": 4, "solo IR": 18})

    def test_una_cifra_cambiada_se_detecta(self):
        tex = self.tex.replace(r"\(T_{\rm eff} = 2150\)--\(2330\)~K for", r"\(T_{\rm eff} = 2160\)--\(2330\)~K for")
        self.assertNotEqual(tex, self.tex)
        res = {r["id"]: r for r in tc.evalua(self.calc.c, tc.cifras(), tex)}
        self.assertEqual(sorted(k for k, r in res.items() if tc.falla(r)), ["apB_42b", "s42_T"])


if __name__ == "__main__":
    unittest.main()
