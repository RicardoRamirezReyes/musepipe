"""Las otras lineas fuera de G2: `scripts/otras_lineas_lacc.py` y `scripts/otras_lineas_variantes.py`.

El primero convierte los limites al 99 % de G2 v4 en L_acc (Alcala et al. 2017); el segundo es el
metodo del agente `lineas` (otras plantillas, otras extracciones, la sola dispersion de las nulas, la
suma de Paschen con caso B y la luz de la primaria), portado del 2026-10-26.

Las clases puras no tocan `runs/`: la conversion flujo -> L_acc con un caso calculado a mano y su
inversa, L/L_bol de las Clase III de Manara et al. (2013), los rangos que cita el paper, la medida
sobre un espectro sintetico y la suma de Paschen con lineas sinteticas de caso B.
`CifrasDelPaperTests` lee `runs/` (solo lectura, < 1 s) y comprueba las cifras del paper que salen de
las conversiones; se salta si el run no esta. `VariantesDelPaperTests` necesita ademas las bibliotecas
de G3 (`external_data`, ~12 s) y comprueba las de la «check outside the pipeline». Las dos fijan las
cifras tal como las imprime el paper (Sect. 4.1, 4.2, Ap. C), no los decimales del script.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
#: raiz que contiene `runs/` y la biblioteca de G3: el repo (los tests de `runs/` se saltan sin ella)
PROYECTO = ROOT


def _carga(nombre):
    if nombre in sys.modules:
        return sys.modules[nombre]
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _ccm():
    from musepipe.models.extinction import CCMExtinction
    return CCMExtinction(3.1, citation="Cardelli, Clayton & Mathis 1989")


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("otras_lineas_lacc")
        cls.ext = _ccm()

    def test_caso_a_mano_hbeta(self):
        # F = 1e-17 erg/s/cm2 a 100 pc sin extincion: 4 pi (3.0857e20 cm)^2 = 1.19650e42 cm2,
        # L = 1.19650e25 erg/s = 3.12564e-9 L_sol (L_sol = 3.828e33), log = -8.50506;
        # Hbeta (Alcala 2017): log L_acc = 1.14 (-8.50506) + 2.59 = -7.10577 -> 7.8385e-8 L_sol.
        c = self.m.lacc_de_flujo(1e-17, 4861.33, av=0.0, distancia_pc=100.0, nombre="Hbeta", ext=self.ext)
        self.assertAlmostEqual(c["l_line_lsun"] / 3.12564e-9, 1.0, delta=2e-6)
        self.assertAlmostEqual(c["l_acc_lsun"] / 7.8385e-8, 1.0, delta=2e-5)
        # He I 6678: log L_acc = 1.25 (-8.50506) + 4.70 = -5.93133 -> 1.17131e-6 L_sol
        c = self.m.lacc_de_flujo(1e-17, 6678.15, av=0.0, distancia_pc=100.0, nombre="He I 6678", ext=self.ext)
        self.assertAlmostEqual(c["l_acc_lsun"] / 1.17131e-6, 1.0, delta=2e-5)

    def test_la_unidad_del_producto_multiplica(self):
        a = self.m.lacc_de_flujo(1000.0, 8542.09, av=1.8, distancia_pc=138.6, nombre="Ca II 8542",
                                 ext=self.ext, unidad_cgs=1e-20)
        b = self.m.lacc_de_flujo(1e-17, 8542.09, av=1.8, distancia_pc=138.6, nombre="Ca II 8542", ext=self.ext)
        self.assertAlmostEqual(a["l_acc_lsun"] / b["l_acc_lsun"], 1.0, places=12)

    def test_la_extincion_desenrojece(self):
        k = self.m.k_lambda(self.ext, 9229.01)
        con = self.m.lacc_de_flujo(1e-17, 9229.01, av=1.8, distancia_pc=138.6, nombre="Pa 9", ext=self.ext)
        sin = self.m.lacc_de_flujo(1e-17 * 10 ** (0.4 * k * 1.8), 9229.01, av=0.0, distancia_pc=138.6,
                                   nombre="Pa 9", ext=self.ext)
        self.assertAlmostEqual(con["l_line_lsun"] / sin["l_line_lsun"], 1.0, places=12)
        self.assertGreater(con["l_line_lsun"],
                           self.m.lacc_de_flujo(1e-17, 9229.01, av=0.0, distancia_pc=138.6, nombre="Pa 9",
                                                ext=self.ext)["l_line_lsun"])

    def test_la_prediccion_es_la_inversa(self):
        for nombre, lam in (("Hbeta", 4861.33), ("He I 6678", 6678.15), ("Ca II 8542", 8542.09), ("Pa 9", 9229.01)):
            c = self.m.lacc_de_flujo(3.3e-18, lam, av=1.9, distancia_pc=146.447, nombre=nombre, ext=self.ext)
            f = self.m.flujo_predicho(c["l_acc_lsun"], lam, av=1.9, distancia_pc=146.447, nombre=nombre, ext=self.ext)
            self.assertAlmostEqual(f / 3.3e-18, 1.0, places=12, msg=nombre)

    def test_la_relacion_de_halpha_es_la_del_run(self):
        self.m._comprueba_relacion_halpha({"h03_lacc_lha_a": 1.13, "h03_lacc_lha_b": 1.74})
        with self.assertRaises(SystemExit):
            self.m._comprueba_relacion_halpha({"h03_lacc_lha_a": 0.95, "h03_lacc_lha_b": 1.61})

    def test_las_clase3_de_manara_en_l_sobre_lbol(self):
        # TWA 26, Ca II 854.2 < 1.69e-16 a 42 pc, log L* = -2.70: 4 pi (42 pc)^2 F / L_sol -> -5.3307
        man = json.loads(self.m.MANARA13.read_text())
        t26 = man["objetos"]["TWA26"]
        v = self.m.log_l_sobre_lbol(t26["lineas"]["Ca II 8542"]["flujo_cgs"], t26["d_pc"], t26["log_lstar"])
        self.assertAlmostEqual(v, -5.3307, delta=1e-4)
        # las cifras que el agente tabulaba (tablas.md), a dos decimales
        esperado = {"TWA26": {"Halpha": -4.84, "Ca II 8498": -5.52, "Ca II 8542": -5.33, "Ca II 8662": -5.63},
                    "TWA29": {"Halpha": -4.62, "Ca II 8498": -5.49, "Ca II 8542": -5.26, "Ca II 8662": -5.46}}
        for est, lineas in esperado.items():
            d = man["objetos"][est]
            for ln, val in lineas.items():
                got = self.m.log_l_sobre_lbol(d["lineas"][ln]["flujo_cgs"], d["d_pc"], d["log_lstar"])
                self.assertEqual(round(got, 2), val, msg=f"{est} {ln}")
            self.assertTrue(all(d["lineas"][ln]["limite"] for ln in self.m.CA_II))

    def _conv(self):
        def fila(fam, lacc, pred, lline):
            return {"family": fam, "lacc_over_ha": lacc, "ul_over_pred": pred, "log_lline_lbol": lline}
        return {"Hbeta": fila("Balmer", 6.1, 4.9, -4.4), "He I 5016": fila("He I", 570.0, 610.0, -4.5),
                "He I 6678": fila("He I", 12.1, 7.4, -5.2), "Ca II 8542": fila("Ca II IRT", 15.2, 16.5, -5.14),
                "Ca II 8662": fila("Ca II IRT", 29.1, 37.6, -5.03), "Pa 9": fila("Paschen", 28.7, 17.2, -4.6)}

    def test_los_rangos_del_texto(self):
        c = self.m.cifras(self._conv(), deteccion=True)
        r = c["lacc_sobre_ha"]
        self.assertEqual((r["linea_min"], r["linea_max"]), ("Hbeta", "Ca II 8662"))
        self.assertEqual((r["min"], r["max"]), (6.1, 29.1))
        self.assertIn("He I 5016", c["fuera_del_rango"])
        rc = c["lacc_sobre_ha_conclusiones"]  # sin Hbeta ni Paschen
        self.assertEqual((rc["min"], rc["linea_min"]), (12.1, "He I 6678"))
        p = c["limite_sobre_prediccion"]  # solo Hbeta, He I 6678, Ca II 8542 y Pa 9
        self.assertEqual((p["min"], p["max"]), (4.9, 17.2))
        self.assertEqual(c["factor_He I 6678"], 7.4)
        lim = self.m.cifras(self._conv(), deteccion=False)
        self.assertNotIn("limite_sobre_prediccion", lim)
        self.assertEqual(lim["lacc_sobre_ha"]["min"], 6.1)

    def test_la_comparacion_con_manara(self):
        man = json.loads(self.m.MANARA13.read_text())
        conv = {"Ca II 8498": {"log_lline_lbol": -5.0}, "Ca II 8542": {"log_lline_lbol": -5.0},
                "Ca II 8662": {"log_lline_lbol": -5.0}}
        c = self.m.comparacion_manara(conv, man)
        # -5.0 frente a -5.517, -5.331, -5.629 (TWA 26) y -5.487, -5.259, -5.464 (TWA 29)
        self.assertAlmostEqual(c["rango"]["min"], 0.259, delta=2e-3)
        self.assertEqual(c["rango"]["en_min"], "TWA29 Ca II 8542")
        self.assertAlmostEqual(c["rango"]["max"], 0.629, delta=2e-3)
        self.assertEqual(c["rango"]["en_max"], "TWA26 Ca II 8662")


class MedidaTests(unittest.TestCase):
    """La medida del agente sobre un espectro sintetico sin ruido."""

    @classmethod
    def setUpClass(cls):
        cls.v = _carga("otras_lineas_variantes")
        cls.wave = np.arange(4750.0, 9350.0, 1.25)
        cls.dl = cls.v._channel_widths(cls.wave)
        cls.err = np.ones_like(cls.wave)
        cls.good = np.ones(cls.wave.shape, bool)

    def _linea(self, centro, flujo, fwhm=2.3):
        return flujo * self.v.plantilla_filtro(self.wave, [(centro, 1.0)], fwhm, self.dl)

    def test_el_filtro_tiene_area_uno(self):
        for comps in ([(6562.8, 1.0)], [(7771.94, 10 ** 0.369), (7774.17, 10 ** 0.223), (7775.39, 1.0)]):
            t = self.v.plantilla_filtro(self.wave, comps, 2.3, self.dl)
            self.assertAlmostEqual(float(np.sum(t * self.dl)), 1.0, places=12)

    def test_recta_y_filtro_recuperan_la_linea(self):
        centro = 6562.80 * (1.0 - 6.09 / self.v.C_KMS)
        y = 100.0 + 0.01 * (self.wave - 6562.8) + self._linea(centro, 50.0)
        otras = self.v.otras_lineas(self.v.CATALOGO, 6562.80)
        r = self.v.medir_uno("lin", self.wave, y, self.err, self.good, centro, [(centro, 1.0)], 2.3, otras, self.dl)
        self.assertAlmostEqual(r["f_mf"], 50.0, places=6)
        self.assertAlmostEqual(r["f_dir"], 50.0, places=4)
        self.assertAlmostEqual(r["cont"], 100.0 + 0.01 * (self.wave[np.argmin(np.abs(self.wave - centro))] - 6562.8),
                               places=6)
        self.assertLess(r["ew"], 0.0)  # emision: EW negativa

    def test_plantilla_por_polinomio_recupera_la_linea(self):
        T = 1.0 + 0.3 * np.sin(self.wave / 7.0)  # «bandas» que una recta no sigue
        x = (self.wave - 8542.09) / 60.0
        y = T * (5.0 + 0.4 * x - 0.2 * x ** 2) + self._linea(8542.09, 3.0)
        otras = self.v.otras_lineas(self.v.CATALOGO, 8542.09)
        r = self.v.medir_uno("tmpl:X", self.wave, y, self.err, self.good, 8542.09, [(8542.09, 1.0)], 2.3,
                             otras, self.dl, T=T)
        self.assertAlmostEqual(r["f_mf"], 3.0, places=6)
        # la mezcla (cada plantilla x grado 1) tambien la recupera si el polinomio es de grado 1
        y1 = T * (5.0 + 0.4 * x) + self._linea(8542.09, 3.0)
        r1 = self.v.medir_uno("tmpl:mix", self.wave, y1, self.err, self.good, 8542.09, [(8542.09, 1.0)], 2.3,
                              otras, self.dl, T=np.vstack([T, np.ones_like(T)]))
        self.assertAlmostEqual(r1["f_mf"], 3.0, places=6)

    def test_los_placebos_respetan_las_exclusiones(self):
        centro = 6300.30
        pl = self.v.placebos(self.wave, self.good, centro, self.v.CATALOGO)
        self.assertGreater(len(pl), 10)
        for lp in pl:
            d = lp - centro
            self.assertGreaterEqual(abs(d), 20.0)
            self.assertLessEqual(abs(d), 300.0)
            self.assertAlmostEqual(d / 12.5, round(d / 12.5), places=9)
            self.assertTrue(all(abs(lp - o["wave_A"]) > 10.0 for o in self.v.CATALOGO))
            self.assertFalse(5730.0 <= lp <= 6100.0)  # hueco del laser +- 50 A
            self.assertFalse(6854.0 <= lp <= 6966.0)  # O2 B +- 6 A
        T = np.where(self.wave < 6400.0, np.nan, 1.0)  # plantilla que no cubre el azul
        con = self.v.placebos(self.wave, self.good, centro, self.v.CATALOGO, T)
        self.assertTrue(con and set(con) < set(pl))
        self.assertTrue(all(np.all(np.isfinite(T[np.abs(self.wave - lp) <= 60.0])) for lp in con))

    def test_la_nula_no_tiene_suelo(self):
        v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        r = self.v.resumen_nula(v)
        self.assertAlmostEqual(r["std"], float(np.std(v, ddof=1)), places=12)
        self.assertEqual(r["n"], 6)
        self.assertTrue(math.isnan(self.v.resumen_nula([1.0, 2.0, 3.0, 4.0])["std"]))


class PaschenTests(unittest.TestCase):
    """La suma de Paschen con lineas sinteticas de caso B."""

    @classmethod
    def setUpClass(cls):
        cls.v = _carga("otras_lineas_variantes")
        cls.rest = {n: next(o["wave_A"] for o in cls.v.CATALOGO if o["name"] == n) for n in cls.v.PASCHEN_STACK}

    def _filas(self, A, av, *, sig=None, factores=None, placebos=None, seed=0):
        rng = np.random.default_rng(seed)
        w = self.v.pesos_caso_b(self.rest, av)
        sig = sig if sig is not None else np.array([5.0, 3.7, 3.8, 3.8, 1.5, 2.6])
        filas = []
        for i, n in enumerate(self.v.PASCHEN_STACK):
            f = A * w[i] * (factores[i] if factores is not None else 1.0)
            pl, pv = placebos(i, n) if placebos is not None else ([], [])
            filas.append({"line": n, "method": "psffit", "continuum": "tmpl:X", "rest_A": self.rest[n],
                          "center_A": self.rest[n], "f_mf": f, "f_mf_sig": float(sig[i]),
                          "f_mf_sig_spec": float(sig[i]), "_ctrl_f_mf": list(rng.normal(0, sig[i], 33)),
                          "_plac_l": pl, "_plac_f_mf": pv})
        return filas, w

    def test_sin_extincion_los_pesos_son_el_caso_b(self):
        from musepipe.models.case_b import case_b_ratio
        w = self.v.pesos_caso_b(self.rest, 0.0)
        for n, wi in zip(self.v.PASCHEN_STACK, w):
            self.assertAlmostEqual(wi, case_b_ratio(n, "Pa 9"), places=12)
        self.assertEqual(w[0], 1.0)
        # caso B: las Paschen bajan de Pa 9 a Pa 17
        self.assertTrue(np.all(np.diff(w) < 0))

    def test_la_extincion_diferencial(self):
        w0, w2 = self.v.pesos_caso_b(self.rest, 0.0), self.v.pesos_caso_b(self.rest, 2.0)
        k9 = self.v.a_lambda(1.0, 9229.01)
        for n, a, b in zip(self.v.PASCHEN_STACK, w0, w2):
            self.assertAlmostEqual(b / a, 10 ** (-0.4 * 2.0 * (self.v.a_lambda(1.0, self.rest[n]) - k9)), places=12)
        self.assertTrue(np.all(w2[1:] < w0[1:]))  # las mas azules se apagan mas

    def test_lineas_de_caso_b_dan_su_amplitud(self):
        filas, w = self._filas(8.0, 1.8)
        a = self.v.apilar_paschen(filas, "psffit", "tmpl:X", 1.8)
        self.assertAlmostEqual(a["A_Pa9"], 8.0, places=10)
        self.assertEqual(list(a["weights_rel_Pa9"].values()), list(map(float, w)))
        self.assertAlmostEqual(a["z"], a["A_Pa9"] / max(a["sig_sp"], a["sig_spec"]), places=12)
        self.assertAlmostEqual(a["ul99_Pa9"], 8.0 + self.v.Z99 * a["sig"], places=10)
        r = self.v.razones_caso_b(filas, "psffit", "tmpl:X", 1.8)
        self.assertAlmostEqual(r["chi2"], 0.0, places=18)
        for n, v in r["por_linea"].items():
            self.assertAlmostEqual(v["residuo_sigma"], 0.0, places=10, msg=n)
            self.assertAlmostEqual(v["medido_sobre_Pa 14"], v["caso_b_sobre_Pa 14"], places=10, msg=n)

    def test_una_linea_fuera_del_caso_b_se_ve_en_su_residuo(self):
        factores = [1.0, 1.0, 1.0, 1.0, 3.0, 1.0]  # Pa 14 tres veces lo que pide el caso B
        filas, _w = self._filas(8.0, 1.8, factores=factores)
        r = self.v.razones_caso_b(filas, "psffit", "tmpl:X", 1.8)
        res = {n: v["residuo_sigma"] for n, v in r["por_linea"].items()}
        self.assertEqual(max(res, key=res.get), "Pa 14")
        self.assertGreater(r["chi2"], 0.0)

    def test_la_nula_espectral_correlada(self):
        shifts = [d for d in np.arange(-300.0, 300.0 + 1e-6, 12.5) if abs(d) >= 20.0]
        c = np.random.default_rng(1).normal(0.0, 2.0, len(shifts))
        w = self.v.pesos_caso_b(self.rest, 1.8)

        def placebos(i, n):  # el mismo desajuste c_D en todas las lineas, en proporcion a su peso
            pl = [self.rest[n] + d for d in shifts]
            pv = [c[j] * w[i] for j in range(len(shifts))]
            if i < 3:  # a tres lineas les falta el placebo del primer desplazamiento
                pl, pv = pl[1:], pv[1:]
            return pl, pv

        filas, _ = self._filas(8.0, 1.8, placebos=placebos)
        a = self.v.apilar_paschen(filas, "psffit", "tmpl:X", 1.8)
        # con 3 de 6 ese desplazamiento no entra; los demas dan exactamente c_D
        self.assertEqual(a["n_spec_corr"], len(shifts) - 1)
        self.assertAlmostEqual(a["sig_spec_corr"], float(np.std(c[1:], ddof=1)), places=10)
        self.assertAlmostEqual(a["z_corr"], 8.0 / max(a["sig_sp"], a["sig_spec_corr"]), places=10)

    def test_la_av_de_la_amplitud_invierte_el_caso_b(self):
        f_ha = 2.4e-17
        for av in (0.5, 1.8, 11.0):
            self.assertAlmostEqual(self.v.av_para_paschen(f_ha, self.v.pa9_caso_b(f_ha, av))[0], av, places=9)
        # el Pa 9 de caso B a A_V = 0 es Halpha / R, con R = Halpha/Pa 9 de Storey & Hummer (112.6 en el nodo)
        R = self.v.av_para_paschen(f_ha, f_ha)[1]
        self.assertAlmostEqual(R, 112.6, delta=0.1)
        self.assertAlmostEqual(self.v.pa9_caso_b(f_ha, 0.0) * R / f_ha, 1.0, places=12)


def _runs_del_paper():
    out = {}
    for f in sorted((ROOT / "targets").glob("*.json")):
        ficha = json.loads(f.read_text())
        if (ficha.get("paper") or {}).get("run"):
            out[f.stem] = ficha["paper"]["run"]
    return out


def _hay_runs(*nombres):
    return all((PROYECTO / "runs" / r / "tables" / "g2_line_measurements.csv").exists() for r in nombres)


class CifrasDelPaperTests(unittest.TestCase):
    """Las cifras de las conversiones tal como las imprime el paper (lee `runs/`)."""

    @classmethod
    def setUpClass(cls):
        runs = _runs_del_paper()
        if not runs or not _hay_runs(*runs.values()):
            raise unittest.SkipTest("los runs del paper no estan en runs/")
        cls.m = _carga("otras_lineas_lacc")
        import scripts.paper_figures as pf
        pf.PROYECTO = PROYECTO
        manara = json.loads(cls.m.MANARA13.read_text())
        cls.res = {o.slug: cls.m.objeto(o, manara) for o in pf.objetos_del_paper()}

    def test_la_deteccion(self):
        res = self.res["ROXs12b"]
        c = res["cifras"]
        r = c["lacc_sobre_ha"]                     # «6 to 29 times» (Sect. 4.1, Ap. C)
        self.assertEqual((round(r["min"]), round(r["max"])), (6, 29))
        self.assertTrue(100 < c["fuera_del_rango"]["He I 5016"]["lacc_over_ha"] < 1000)  # «several hundred»
        p = c["limite_sobre_prediccion"]           # «5-17 times below our limits» (Ap. C)
        self.assertEqual((round(p["min"]), round(p["max"])), (5, 17))
        self.assertEqual(round(c["factor_He I 6678"]), 7)   # «a factor of 7» (Sect. 4.1)
        m = res["manara2013"]["rango"]             # «0.1-0.6 dex deeper» (Sect. 4.1)
        self.assertEqual((round(m["min"], 1), round(m["max"], 1)), (0.1, 0.6))

    def test_el_limite(self):
        r = self.res["ROXs42Bb"]["cifras"]["lacc_sobre_ha"]   # «at least 18 times» (Sect. 4.2, Ap. C)
        self.assertEqual(math.floor(r["min"]), 18)
        self.assertEqual(self.res["ROXs42Bb"]["cifras"]["referencia_halpha"], "limite")


@pytest.mark.external_data
class VariantesDelPaperTests(unittest.TestCase):
    """La «check outside the pipeline» de la Sect. 4.1 y el Ap. C (lee `runs/` y la biblioteca de G3)."""

    @classmethod
    def setUpClass(cls):
        runs = _runs_del_paper()
        if "ROXs12b" not in runs or not _hay_runs(runs["ROXs12b"]):
            raise unittest.SkipTest("el run de ROXs 12 b no esta en runs/")
        cfg = json.loads((PROYECTO / "runs" / runs["ROXs12b"] / "config" / "config.json").read_text())["config"]
        if not (PROYECTO / cfg.get("g3_libraries_root", "") / "templates_young").is_dir():
            raise unittest.SkipTest("la biblioteca templates_young de G3 no esta descargada")
        cls.v = _carga("otras_lineas_variantes")
        import scripts.paper_figures as pf
        pf.PROYECTO = PROYECTO
        cls.filas, cls.todo = cls.v.analiza(cls.v.objeto_del_paper("ROXs12b"), PROYECTO, ["CHSM17173", "TWA26"])

    def test_o_i_8446_depende_de_la_plantilla(self):
        z = self.todo["cifras"]["z_por_variante"]["O I 8446"]["psffit"]
        self.assertEqual(round(z["tmpl:TWA26"], 1), 3.5)   # «3.5 sigma with an M9 template»
        self.assertEqual(round(z["tmpl:LM601"], 1), 1.0)   # «1.0 sigma with the M7.5 template»

    def test_la_suma_de_paschen(self):
        c = self.todo["cifras"]
        self.assertEqual(round(c["paschen_z_corr_max"]["z_corr"], 1), 3.3)    # «up to 3.3 sigma»
        self.assertEqual(round(c["paschen_z_corr"]["LM601"]["min"], 1), 1.8)  # el 1.8 del informe
        self.assertEqual(round(c["paschen_av"]["av"]), 11)                    # «A_V ~ 11 mag»

    def test_la_luz_de_la_primaria(self):
        f = self.todo["cifras"]["fuga"]                     # «by 0.3 sigma at most» (Ap. C)
        self.assertEqual(round(abs(f["max_sobre_sigma"]), 1), 0.3)
        self.assertTrue(f["linea_max"].startswith("Ca II"))   # «through the Ca II absorption of the star»
        self.assertLess(abs(f["max_sobre_sigma_g2"]), abs(f["max_sobre_sigma"]))


if __name__ == "__main__":
    unittest.main()
