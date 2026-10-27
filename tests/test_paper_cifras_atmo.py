"""Las cifras del párrafo de ATMO 2020 del apéndice B (`scripts/paper_cifras_atmo.py`), sin leer nada de disco: qué
corrida es cuál, qué rejilla de ATMO se usa, el tipo equivalente por las dos reglas y la dispersión de la misma
plantilla entre el ruido de un objeto y el del otro. Las cifras en sí salen de las cadenas (json + npz) y de los
runs del paper; se comprobaron contra las de la sesión del agente `atmo` (93 iguales, sin tolerancia).
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _p(m, lo, hi, p2=None, p97=None):
    return dict(p2_5=lo if p2 is None else p2, p16=lo, p50=m, p84=hi, p97_5=hi if p97 is None else p97)


def _corrida(slug, rejilla, av_fija=None, teff=(2400.0, 2390.0, 2410.0), logg=(4.0, 3.9, 4.1), plantilla=None,
             r=500, etiqueta=None):
    d = dict(slug=slug, modo="libre", rejilla=rejilla, r_analisis=r, datos="todo", av="primaria", av_fija=av_fija,
             etiqueta=etiqueta or f"{slug}_{rejilla}_{av_fija}_{plantilla}_{r}", logz=0.0,
             posterior=dict(teff=_p(*teff), logg=_p(*logg)))
    if rejilla.startswith("atmo") or rejilla == "btsettl_bin":
        d["degrada"] = "cuadratura"
    if plantilla:
        d["verdad"] = dict(plantilla=plantilla, spt=plantilla.split("_")[0], av=1.8)
    return d


class CorridasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pc = _carga("paper_cifras_atmo")

    def test_busca_distingue_rejilla_av_y_resolucion(self):
        cs = [_corrida("ROXs12b", "atmo_ceq", av) for av in (1.8, 3.0, None)]
        cs += [_corrida("ROXs12b", "atmo_ceq", 1.8, r=250), _corrida("ROXs12b", "btsettl", 1.8)]
        idx = self.pc.indice(cs)
        self.assertIs(self.pc.busca(idx, "ROXs12b", "atmo_ceq", 3.0), cs[1])
        self.assertIs(self.pc.busca(idx, "ROXs12b", "atmo_ceq"), cs[2])
        self.assertIs(self.pc.busca(idx, "ROXs12b", "atmo_ceq", 1.8, r=250), cs[3])
        self.assertIs(self.pc.busca(idx, "ROXs12b", "btsettl", 1.8), cs[4])
        self.assertIsNone(self.pc.busca(idx, "ROXs42Bb", "atmo_ceq"))

    def test_el_degradado_ingenuo_no_es_el_de_las_cifras(self):
        a = _corrida("ROXs12b", "atmo_ceq")
        b = dict(_corrida("ROXs12b", "atmo_ceq", etiqueta="ingenua"), degrada="ingenua")
        idx = self.pc.indice([a, b])
        self.assertIs(self.pc.busca(idx, "ROXs12b", "atmo_ceq"), a)

    def test_dos_corridas_iguales_son_un_error(self):
        with self.assertRaises(SystemExit):
            self.pc.indice([_corrida("ROXs12b", "atmo_ceq"), _corrida("ROXs12b", "atmo_ceq", etiqueta="otra")])

    def test_la_ampliada_solo_si_el_prior_toca_2000_k(self):
        # 42B b: con 2000–3000 K el 2.5 % del prior cae a < 50 K de 2000 K; 12 b toca el borde de log g, no de T
        borde = _corrida("ROXs42Bb", "atmo_ceq", teff=(2007.0, 2002.0, 2016.0), etiqueta="b")
        amp = _corrida("ROXs42Bb", "atmo_ceq1500", teff=(1982.0, 1968.0, 1998.0), etiqueta="a")
        g_borde = _corrida("ROXs12b", "atmo_ceq", teff=(2474.0, 2453.0, 2492.0), logg=(2.53, 2.51, 2.58))
        idx = self.pc.indice([borde, amp, g_borde])
        self.assertEqual(self.pc.rejilla_atmo(idx, "ROXs42Bb"), "atmo_ceq1500")
        self.assertEqual(self.pc.rejilla_atmo(idx, "ROXs12b"), "atmo_ceq")
        idx2 = self.pc.indice([_corrida("ROXs42Bb", "atmo_ceq", teff=(2250.0, 2200.0, 2300.0)), amp])
        self.assertEqual(self.pc.rejilla_atmo(idx2, "ROXs42Bb"), "atmo_ceq")


class TipoEquivalenteTests(unittest.TestCase):
    """Las dos reglas con las T_eff de las plantillas ajustadas con BT-Settl y el ruido de ROXs 12 b (cadenas c0)."""

    PLANTILLAS_12B_BT = {"LM717": ("M6.5", 2789.0), "J11195652": ("M7", 2751.0), "LM601": ("M7.5", 2738.0),
                         "CHSM17173": ("M8", 2591.0), "USco_J1610": ("M8.5", 2440.0), "VISTA": ("M8.5", 2425.0),
                         "USco_J1607": ("M9", 2188.0), "TWA26": ("M9", 2419.0), "TWA29": ("M9.5", 2349.0),
                         "2MASS0046": ("L0", 2238.0), "2MASS1207": ("L1", 2165.0), "WISE0528": ("L1", 2212.0)}

    @classmethod
    def setUpClass(cls):
        cls.pc = _carga("paper_cifras_atmo")

    def test_bt_settl_12b_es_m75_m8_por_la_mediana(self):
        t = self.pc.tipo_equivalente(_p(2719.0, 2689.0, 2746.0), self.PLANTILLAS_12B_BT)
        self.assertEqual(t["mediana"], "M7.5–M8")      # la Tabla B.5
        self.assertEqual(t["p16_84"], "M7–M8")         # LM601 (2738 K) cae dentro del 16–84 %
        self.assertEqual(t["cercana"]["plantilla"], "LM601")

    def test_mas_frio_que_la_l1_mas_fria_es_al_menos_l1(self):
        t = self.pc.tipo_equivalente(_p(2150.0, 2140.0, 2160.0), self.PLANTILLAS_12B_BT)
        self.assertEqual((t["mediana"], t["p16_84"]), ("≥L1", "≥L1"))
        # entre las dos L1 (2165 y 2212 K): L1, no «≥L1»; y bajo la L0 (2238 K) con una M9 más fría (2188 K):
        # la secuencia no es monótona, así que el intervalo va del tipo más temprano de los dos al más tardío
        t = self.pc.tipo_equivalente(_p(2180.0, 2170.0, 2190.0), self.PLANTILLAS_12B_BT)
        self.assertEqual(t["mediana"], "L1")
        t = self.pc.tipo_equivalente(_p(2230.0, 2225.0, 2235.0), self.PLANTILLAS_12B_BT)
        self.assertEqual(t["mediana"], "M9–L0")

    def test_un_tipo_solo_cuando_los_vecinos_coinciden(self):
        pls = {"a": ("M7", 2800.0), "b": ("M8", 2600.0), "c": ("M8", 2500.0), "d": ("M9", 2300.0)}
        self.assertEqual(self.pc.tipo_equivalente(_p(2550.0, 2540.0, 2560.0), pls)["mediana"], "M8")
        self.assertEqual(self.pc.tipo_equivalente(_p(2900.0, 2890.0, 2910.0), pls)["mediana"], "≤M7")

    def test_la_misma_plantilla_con_dos_ruidos(self):
        a = {"x": ("M7", 2500.0), "y": ("M8", 2400.0), "z": ("L1", 2000.0)}
        b = {"x": ("M7", 2540.0), "y": ("M8", 2100.0), "z": ("L1", 2010.0), "solo_b": ("M9", 2200.0)}
        d = self.pc.dispersion_plantillas(a, b)
        self.assertEqual((d["n"], d["mediana"], d["maximo"], d["plantilla_maximo"]), (3, 40.0, 300.0, "y"))
        self.assertIsNone(self.pc.dispersion_plantillas(a, {"otra": ("M7", 1.0)}))


if __name__ == "__main__":
    unittest.main()
