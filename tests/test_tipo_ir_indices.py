"""El tipo espectral IR por índices de agua (`scripts/tipo_ir_indices.py`).

Da las cifras «M9.9 ± 1.0» de ROXs 42B b y «M7.5 ± 1.0» de ROXs 12 b en el paper
(docs/2026-10-02_tipo_ir_42bb.md) y vivió fuera
del repo hasta el 2026-10-02. Las tres primeras clases no leen nada de disco: fijan la escala de los
polinomios de Allers & Liu, el signo del desenrojecimiento y cómo se combinan los índices.
`CifrasPublicadasTests` lee el catálogo CDS J/A+A/708/A312 en /mnt/2TB (solo lectura, ~1 s), con los
espectros que declara `targets/<slug>.json` (`ir_spectra`), y comprueba las cifras del paper y del control
a media unidad del último dígito publicado. Si el catálogo o el run
no están, su caso se salta, no falla (como los tests que leen `runs/`; `external_data` es para las
bibliotecas de G3).
"""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _rejilla(lo, hi, R=500):
    """Rejilla con λ/Δλ = R por píxel, como los espectros del catálogo."""
    n = int(math.log(hi / lo) * R) + 1
    return lo * np.exp(np.arange(n) / R)


class EscalaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("tipo_ir_indices")

    def test_el_h2o2_de_currie_da_el_L0_que_dice_currie(self):
        # Currie+2014b: «H2O-2 ~ 0.87; el polinomio de Allers 2013 da L0». Fija coeficientes y M0 = 0, L0 = 10.
        self.assertAlmostEqual(self.m.spt_from("H2O-2", 0.87), 9.79, delta=0.01)

    def test_etiquetas(self):
        self.assertEqual(self.m.spt_label(7.5), "M7.5")
        self.assertEqual(self.m.spt_label(10.0), "L0.0")
        self.assertEqual(self.m.spt_label(11.3), "L1.3")

    def test_la_pendiente_es_la_derivada_del_polinomio(self):
        for name in self.m.INDICES:
            x = 0.9
            num = (self.m.spt_from(name, x + 1e-5) - self.m.spt_from(name, x - 1e-5)) / 2e-5
            self.assertAlmostEqual(self.m.spt_slope(name, x), num, places=4)


class IndicesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("tipo_ir_indices")

    def test_un_espectro_plano_da_indices_uno(self):
        for lo, hi, names in ((1.15, 1.36, ["H2O-1"]), (1.48, 1.80, ["H2O"]), (1.95, 2.40, ["H2O-2"])):
            w = _rejilla(lo, hi); f = np.ones_like(w)
            for n in names:
                self.assertAlmostEqual(self.m.ratio_index(w, f, 0.01 * f, n)["index"], 1.0, places=12)
        w = _rejilla(1.15, 1.36); f = np.ones_like(w)
        self.assertAlmostEqual(self.m.kij_index(w, f), 1.0, places=12)

    def test_una_ley_de_potencias_da_su_cociente(self):
        w = _rejilla(1.95, 2.40); f = w ** -2.0
        self.assertAlmostEqual(self.m.ratio_index(w, f, 0.01 * f, "H2O-2")["index"], (2.04 / 2.15) ** -2, places=4)

    def test_la_media_de_ventana_es_la_del_espectro_interpolado(self):
        w = _rejilla(1.95, 2.40); f = np.sin(7 * w) + 2
        g = np.linspace(2.035, 2.045, 201)
        self.assertAlmostEqual(self.m.window_mean(w, f, 2.035, 2.045), float(np.mean(np.interp(g, w, f))), places=12)

    def test_una_ventana_en_el_borde_no_cuenta(self):
        # H2OD necesita 1.951 µm: un K que empieza en 1.953 (el SINFONI de 42B b) no lo cubre
        w = _rejilla(1.953, 2.446); f = np.ones_like(w)
        self.assertIsNone(self.m.ratio_index(w, f, 0.01 * f, "H2OD"))
        # una ventana que solo cubre el último píxel tampoco (el OSIRIS J de 12 b acaba en 1.346 µm)
        w = np.append(_rejilla(1.15, 1.340), 1.346); f = np.ones_like(w)
        self.assertGreaterEqual(w[-1], 1.345)
        self.assertIsNone(self.m.ratio_index(w, f, 0.01 * f, "H2O-1"))

    def test_desenrojecer_sube_h2o2_y_baja_h2o1(self):
        # Slesnick+2004: la extinción hace más tardío el tipo por H2O-2 y más temprano por H2O-1
        w = _rejilla(1.15, 2.40); f = np.ones_like(w); fd = self.m.deredden(w, f, 2.0)
        self.assertGreater(self.m.ratio_index(w, fd, 0.01 * fd, "H2O-2")["index"], 1.0)
        self.assertLess(self.m.ratio_index(w, fd, 0.01 * fd, "H2O-1")["index"], 1.0)

    def test_el_error_conservador_no_baja_del_independiente(self):
        rng = np.random.default_rng(1)
        w = _rejilla(1.95, 2.40); e = np.full_like(w, 0.02); f = 1 + rng.normal(0, 0.02, w.size)
        r = self.m.ratio_index(w, f, e, "H2O-2")
        self.assertGreater(r["sigma"], r["sigma_indep"])


def _fila(m, name, value, sigma):
    return {"index": value, "sigma": sigma, "spt": m.spt_from(name, value)}


class CombinaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("tipo_ir_indices")

    def test_el_rms_se_suma_una_vez_por_indice(self):
        # dos instrumentos idénticos casi sin ruido: el error del índice es el rms, no rms/√2
        rows = {"A": {"H2O-2": _fila(self.m, "H2O-2", 0.87, 1e-6)}, "B": {"H2O-2": _fila(self.m, "H2O-2", 0.87, 1e-6)}}
        r = self.m.combina(rows, {"H2O-2": ["A", "B"]})
        self.assertAlmostEqual(r["per_index"]["H2O-2"]["sigma"], self.m.INDICES["H2O-2"]["rms"], places=4)

    def test_birge_infla_si_los_instrumentos_discrepan(self):
        rows = {"A": {"H2O-2": _fila(self.m, "H2O-2", 0.85, 0.005)}, "B": {"H2O-2": _fila(self.m, "H2O-2", 0.90, 0.005)}}
        r = self.m.combina(rows, {"H2O-2": ["A", "B"]})["per_index"]["H2O-2"]
        self.assertAlmostEqual(r["birge"], math.sqrt(0.05**2 / (2 * 0.005**2)), places=6)
        self.assertAlmostEqual(r["index_err"], 0.005 / math.sqrt(2) * r["birge"], places=9)

    def test_el_error_final_tiene_el_suelo_de_un_subtipo(self):
        rows = {"A": {"H2O": _fila(self.m, "H2O", 1.15, 1e-6)}, "B": {"H2O-2": _fila(self.m, "H2O-2", 0.874, 1e-6)}}
        r = self.m.combina(rows, {"H2O": ["A"], "H2O-2": ["B"]})
        self.assertLess(r["formal"], 1.0)
        self.assertEqual(r["sigma_stat"], 1.0)

    def test_cada_indice_usa_los_espectros_que_lo_cubren(self):
        rows = {"J": {"H2O-1": 1, "KI_J": 1}, "H1": {"H2O": 1}, "H2": {"H2O": 1}, "K": {"H2O-2": 1}}
        self.assertEqual(self.m.grupos(rows), {"H2O": ["H1", "H2"], "H2O-1": ["J"], "H2O-2": ["K"]})
        self.assertEqual(self.m.grupos(rows, only=["H1", "K"]), {"H2O": ["H1"], "H2O-2": ["K"]})

    def test_un_indice_fuera_de_su_rango_no_entra(self):
        rows = {"A": {"H2O-2": _fila(self.m, "H2O-2", 0.60, 0.01)}, "B": {"H2O": _fila(self.m, "H2O", 1.15, 0.01)}}
        r = self.m.combina(rows, {"H2O-2": ["A"], "H2O": ["B"]})
        self.assertFalse(r["per_index"]["H2O-2"]["valid"])
        self.assertEqual(r["used"], ["H2O"])


@unittest.skipUnless(Path("/mnt/2TB/IR_Data/publicados_Hayoz2026_JAA708A312").is_dir(), "catálogo CDS no disponible")
class CifrasPublicadasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("tipo_ir_indices")
        cls.cds = cls.m.DEFAULT_CDS
        cls.p42 = cls.m.perfil("ROXs42Bb")
        cls.p12 = cls.m.perfil("ROXs12b")

    def test_los_espectros_del_cds_no_estan_desenrojecidos(self):
        # fotometría OBSERVADA del propio catálogo (phot.dat: Kraus+2014 en 12 b, Kraus/Currie en 42B b)
        phot = {("ROXs12b", "NIFS_J"): 6.738e-16, ("ROXs12b", "OSIRIS_H"): 8.506e-16, ("ROXs12b", "OSIRIS_K"): 8.694e-16,
                ("ROXs42Bb", "NIFS_J"): 4.881e-16, ("ROXs42Bb", "NIFS_H"): 5.318e-16, ("ROXs42Bb", "OSIRIS_K"): 4.605e-16}
        bands = {"J": (1.17, 1.33), "H": (1.49, 1.78), "K": (2.03, 2.36)}
        for (obj, s), fp in phot.items():
            spec = {"ROXs12b": self.p12, "ROXs42Bb": self.p42}[obj]["ir_spectra"]["spectra"][s]
            w, f, _ = self.m.load(self.cds, spec["file"])
            lo, hi = bands[s[-1]]
            r = float(np.mean(f[(w >= lo) & (w <= hi)])) / fp
            with self.subTest(obj=obj, espectro=s):
                self.assertLess(abs(r - 1), 0.07)   # desenrojecidos con A_V ≈ 1.8 saldrían ×1.2–1.5

    def test_la_lectura_de_42bb_da_la_cifra_del_paper(self):
        r = self.m.lectura(self.cds, self.p42["ir_spectra"]["spectra"], 1.9, 0.3)
        self.assertAlmostEqual(r["spt"], 9.9, delta=0.05)          # «M9.9»
        self.assertAlmostEqual(r["sigma_total"], 1.0, delta=0.05)  # «± 1.0»
        self.assertAlmostEqual(r["formal"], 0.6, delta=0.05)       # «formal error 0.6 subtypes»
        self.assertLess(r["sigma_av_sys"], 0.1)                    # «less than 0.1 subtype»
        self.assertLess(r["chi2nu"], 1.0)                          # «the three indices give consistent types»
        self.assertEqual(sorted(r["used"]), ["H2O", "H2O-1", "H2O-2"])

    def test_la_lectura_de_12b_da_la_cifra_del_paper(self):
        r = self.m.lectura(self.cds, self.p12["ir_spectra"]["spectra"], 1.8, 0.5)
        self.assertAlmostEqual(r["spt"], 7.5, delta=0.05)           # «M7.5»
        self.assertAlmostEqual(r["sigma_total"], 1.0, delta=0.05)  # «± 1.0»
        self.assertAlmostEqual(r["formal"], 0.6, delta=0.05)       # «formal error 0.6 subtypes»
        self.assertAlmostEqual(r["sigma_av_sys"], 0.1, delta=0.05)  # «0.1 subtype»
        self.assertLess(r["chi2nu"], 1.0)
        self.assertEqual(sorted(r["used"]), ["H2O", "H2O-1", "H2O-2"])

    def test_el_control_reproduce_el_doc(self):
        c42 = self.m.control(self.cds, self.p42)
        pub = {p["ref"]: p for p in c42["publicados"]}
        self.assertAlmostEqual(pub["Currie+2014b"]["index"], 0.906, delta=0.0005)
        self.assertAlmostEqual(pub["Currie+2014b"]["index_av0"], 0.890, delta=0.0005)
        self.assertFalse(pub["Currie+2014b"]["dentro"])                # P1 falla
        self.assertAlmostEqual(pub["Bowler+2014"]["spt"], 10.4, delta=0.05)
        self.assertTrue(pub["Bowler+2014"]["dentro"])
        t = {(x["a"], x["b"]): x.get("t") for x in c42["instrumentos"]}
        self.assertAlmostEqual(t[("NIFS_H", "OSIRIS_H")], 0.46, delta=0.01)
        self.assertAlmostEqual(t[("OSIRIS_K", "SINFONI_K")], 0.99, delta=0.01)
        c12 = self.m.control(self.cds, self.p12)
        self.assertAlmostEqual(c12["publicados"][0]["spt"], 7.5, delta=0.05)
        self.assertTrue(c12["publicados"][0]["dentro"])
        self.assertEqual(c12["instrumentos"][0].get("nota"), "no cubierto")   # OSIRIS J acaba en 1.346 µm

    @unittest.skipUnless((ROOT / "runs" / "ROXs42Bb_realigned" / "config" / "config.json").exists(), "run no disponible")
    def test_la_av_es_la_del_run_y_la_de_la_tabla_1(self):
        self.assertEqual(self.m.av_del_run("ROXs42Bb_realigned"), (1.9, 0.3))

    @unittest.skipUnless((ROOT / "runs" / "ROXs12b_invvar" / "config" / "config.json").exists(), "run no disponible")
    def test_la_av_de_12b_es_la_del_run_y_la_de_la_tabla_1(self):
        self.assertEqual(self.m.av_del_run("ROXs12b_invvar"), (1.8, 0.5))


if __name__ == "__main__":
    unittest.main()
