"""T_eff por muestreo anidado (`scripts/teff_bayes.py`).

Da la T_eff, el tipo equivalente y el margen sobre el suelo cromosférico del apéndice B del paper
(docs/2026-10-14_teff_muestreo_anidado.md). Las corridas de dynesty que dan esas cifras tardan de 20 a 140 min cada
una, así que aquí no se repiten: se prueba lo que las sostiene. Las primeras clases no leen nada de disco: el suelo
y su umbral, el margen con la L_acc reescalada a la A_V, y el modelo físico sobre una rejilla sintética lineal en
log F (donde la interpolación bilineal es exacta). `UmbralDelPaperTests` lee los QC de los runs del paper y
`MaquinariaTests` (marca `slow`) construye el modelo sobre los datos y las rejillas de /mnt/2TB; si falta algo se
salta, no falla.
"""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


class SueloYMargenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = _carga("teff_bayes")

    def test_suelo_es_la_ecuacion_2_de_manara(self):
        sc = _carga("suelo_cromosferico")
        for t in (1806.0, 2270.0, 2460.0, 3100.0):
            self.assertAlmostEqual(float(self.tb.suelo(t)), 6.17 * math.log10(t) - 24.54, places=12)
            self.assertAlmostEqual(float(self.tb.suelo(t)), sc.suelo(t), places=12)

    def test_el_umbral_invierte_el_suelo(self):
        for r in (-4.446, -3.618, -3.0):
            self.assertAlmostEqual(float(self.tb.suelo(self.tb.teff_umbral(r))), r, places=10)
        self.assertAlmostEqual(self.tb.teff_umbral(-3.618), 2460.0, delta=1.0)
        self.assertAlmostEqual(self.tb.teff_umbral(-4.446), 1806.0, delta=1.0)

    def test_margen_con_la_av_adoptada_es_el_cociente_menos_el_suelo(self):
        m = self.tb.margen(2600.0, 1.8, 1e-3, 3.7e-7, 1.8, 1.13, 0.8177)
        self.assertAlmostEqual(float(m), math.log10(3.7e-7 / 1e-3) - (6.17 * math.log10(2600.0) - 24.54), places=12)

    def test_la_lacc_sube_0370_dex_por_magnitud(self):
        # pendiente de Alcalá et al. 2017 (1.13) por A_Hα/A_V de CCM89 (0.8177): 0.370 dex/mag, la del paper
        a, ahav = 1.13, 0.8177
        m0, m1 = (float(self.tb.margen(2600.0, av, 1e-3, 3.7e-7, 1.8, a, ahav)) for av in (1.8, 2.8))
        self.assertAlmostEqual(m1 - m0, 0.4 * a * ahav, places=12)
        self.assertAlmostEqual(m1 - m0, 0.370, delta=0.001)

    def test_margen_vectorial_y_mas_lbol_menos_margen(self):
        m = self.tb.margen(np.array([2500.0, 2500.0]), np.array([1.8, 1.8]), np.array([1e-3, 2e-3]), 3.7e-7, 1.8,
                           1.13, 0.8177)
        self.assertEqual(m.shape, (2,))
        self.assertAlmostEqual(float(m[0] - m[1]), math.log10(2.0), places=12)


class AuxiliaresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = _carga("teff_bayes")

    def test_normal_truncada_respeta_el_suelo(self):
        self.assertGreaterEqual(self.tb._ppf_trunc_normal(1e-9, 1.0, 0.5), 0.0)
        self.assertGreaterEqual(self.tb._ppf_trunc_normal(1e-9, 1.0, 0.2, lo=0.05), 0.05)
        self.assertAlmostEqual(self.tb._ppf_trunc_normal(0.5, 1.8, 0.1), 1.8, places=6)

    def test_percentiles_ignora_nan_y_vacio_es_none(self):
        p = self.tb.percentiles([1.0, 2.0, np.nan, 3.0])
        self.assertEqual(p["p50"], 2.0)
        self.assertIsNone(self.tb.percentiles([np.nan]))

    def test_las_dos_rejillas_declaran_sus_ejes(self):
        bt, dr = self.tb.REJILLAS["btsettl"], self.tb.REJILLAS["drift"]
        self.assertEqual((bt["teff"], bt["logg"]), ((2000.0, 100.0, 13), (2.5, 0.5, 5)))
        self.assertEqual((dr["teff"], dr["logg"]), ((2000.0, 100.0, 11), (3.0, 0.5, 7)))


def _modelo_sintetico(tb):
    """Un `Modelo` sin leer disco: una banda K y una rejilla 3×2 con log F lineal en T_eff, log g y λ."""
    m = tb.Modelo.__new__(tb.Modelo)
    m.modo = "libre"
    m.T0, m.dT, m.nT, m.G0, m.dG, m.nG = 2000.0, 100.0, 3, 3.0, 0.5, 2
    w = np.linspace(20500.0, 23500.0, 40)

    def logf(T, g, lam):
        return -2.0 + 1e-3 * T + 0.1 * g - 1e-5 * (lam - 20000.0)
    lg = np.array([[logf(m.T0 + a * m.dT, m.G0 + b * m.dG, w) for b in range(m.nG)] for a in range(m.nT)])
    m.lref = np.array([[logf(m.T0 + a * m.dT, m.G0 + b * m.dG, tb.LAM_K) for b in range(m.nG)] for a in range(m.nT)])
    bb = tb._planck(w, tb.T_D) / tb._planck(tb.LAM_K, tb.T_D)
    e = np.full(w.size, 1e-17)
    m.segs = [dict(nombre="K", banda="K", w=w, d=np.full(w.size, 1e-15), e2=e ** 2, lg=lg,
                   a=np.full(w.size, 0.12), bb=bb, sigma=0.05)]
    m.bandas = ["K"]
    m.nombres = ["teff", "logg", "r_rj", "d_pc", "av", "rk", "s_K", "lnf_K"]
    m.pos = {k: i for i, k in enumerate(m.nombres)}
    return m, logf


class ModeloFisicoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = _carga("teff_bayes")

    def test_flujo_absoluto_extincion_y_exceso_en_k(self):
        from musepipe.constants import PC_CM, RJUP_CM
        m, logf = _modelo_sintetico(self.tb)
        th = np.array([2130.0, 3.2, 2.0, 140.0, 1.5, 0.3, 1.1, math.log(0.05)])
        (out,) = m.modelos(th)
        s = m.segs[0]
        om = (2.0 * RJUP_CM / (140.0 * PC_CM)) ** 2
        fref = 10 ** logf(2130.0, 3.2, self.tb.LAM_K)
        esperado = 1.1 * om * (10 ** logf(2130.0, 3.2, s["w"]) + 0.3 * fref * s["bb"]) * 10 ** (-0.4 * 0.12 * 1.5)
        np.testing.assert_allclose(out, esperado, rtol=1e-12)

    def test_verosimilitud_con_el_error_de_modelo(self):
        m, _ = _modelo_sintetico(self.tb)
        th = np.array([2050.0, 3.4, 1.7, 140.0, 1.8, 0.1, 0.9, math.log(0.1)])
        (mod,) = m.modelos(th)
        s = m.segs[0]
        var = s["e2"] + (0.1 * mod) ** 2
        esperado = -0.5 * float(np.sum((s["d"] - mod) ** 2 / var + np.log(var)))
        self.assertAlmostEqual(m.loglike(th), esperado, places=6)

    def test_lbol_derivada_es_stefan_boltzmann(self):
        from musepipe.constants import LSUN_ERG_S, RJUP_CM, SIGMA_SB_CGS
        m, _ = _modelo_sintetico(self.tb)
        th = np.array([2200.0, 3.5, 2.1, 140.0, 1.8, 0.0, 1.0, math.log(0.01)])
        teff, logg, r_cm, _, lbol = m.fisica_de(th)
        self.assertEqual((teff, logg), (2200.0, 3.5))
        self.assertAlmostEqual(r_cm, 2.1 * RJUP_CM)
        self.assertAlmostEqual(lbol, 4 * math.pi * r_cm ** 2 * SIGMA_SB_CGS * 2200.0 ** 4 / LSUN_ERG_S, places=15)


def _hay_runs():
    import json
    for slug in ("ROXs12b", "ROXs42Bb"):
        perfil = json.loads((ROOT / "targets" / f"{slug}.json").read_text())
        rd = ROOT / "runs" / perfil["paper"]["run"] / "stages"
        if not all((rd / f).exists() for f in ("stage_g3_qc.json", "stage_h03_qc.json", "stage_x11_qc.json")):
            return False
    return True


class TablaDeLasCadenasTests(unittest.TestCase):
    """El cuerpo LaTeX de la tabla de las cadenas del apéndice B (`filas_tabla`), sin leer nada de disco."""

    @classmethod
    def setUpClass(cls):
        cls.tb = _carga("teff_bayes")

    @staticmethod
    def _p(m, lo, hi):
        return dict(p16=lo, p50=m, p84=hi)

    def _corrida(self, etiqueta, rejilla, av_fija, logz, modo="libre", av="primaria", verdad=None):
        p = self._p
        return dict(etiqueta=etiqueta, slug="X", modo=modo, av=av, av_fija=av_fija, rejilla=rejilla, logz=logz,
                    verdad=verdad, posterior=dict(
                        teff=p(2500.0, 2480.0, 2530.0), logg=p(4.0, 3.9, 4.1), r_rj=p(1.5, 1.4, 1.6),
                        av=p(2.5, 2.3, 2.6), rk=p(0.5, 0.4, 0.6),
                        lnf_vis=p(math.log(0.15), math.log(0.13), math.log(0.17))))

    def test_mediana_e_intervalo_16_84(self):
        self.assertEqual(self.tb._pm(self._p(2712.0, 2689.0, 2748.0)), "$2712^{+36}_{-23}$")
        self.assertEqual(self.tb._pm(self._p(0.17, 0.12, 0.22), 2, signo=True), "$+0.17^{+0.05}_{-0.05}$")
        self.assertEqual(self.tb._pm(self._p(-0.27, -0.31, -0.24), 2, signo=True), "$-0.27^{+0.03}_{-0.04}$")
        # `f` transforma los percentiles antes de restar: el error de modelo en %, desde ln f
        en_pct = lambda x: 100 * math.exp(x)
        self.assertEqual(self.tb._pm(self._p(math.log(0.15), math.log(0.13), math.log(0.17)), 1, en_pct),
                         "$15.0^{+2.0}_{-2.0}$")

    def test_orden_av_fija_sin_intervalo_y_lnz_relativo(self):
        corridas = [self._corrida("d-prior", "drift", None, 90.0), self._corrida("bt-3", "btsettl", 3.0, 100.0),
                    self._corrida("bt-prior", "btsettl", None, 97.5), self._corrida("bt-1.8", "btsettl", 1.8, 80.0),
                    # no entran: una plantilla, el modo evolutivo y la A_V libre, aunque tengan mejor ln Z
                    self._corrida("pl", "btsettl", None, 120.0, verdad={"plantilla": "TWA26"}),
                    self._corrida("evo", "btsettl", None, 130.0, modo="evo"),
                    self._corrida("libre", "btsettl", None, 140.0, av="libre")]
        deltas = {d["etiqueta"]: self._p(0.1, 0.05, 0.15) for d in corridas}
        filas = self.tb.filas_tabla(corridas, deltas, {"X": "Obj~X"})
        self.assertEqual(filas[0], r"\multicolumn{9}{l}{Obj~X} \\")
        celdas = [f.split(" & ") for f in filas[1:]]
        self.assertEqual([len(c) for c in celdas], [9] * 4)
        self.assertEqual([c[0] for c in celdas], [r"\quad BT-Settl"] * 3 + [r"\quad DRIFT-PHOENIX"])
        self.assertEqual([c[4] for c in celdas], ["$1.8$", "$3.0$", "$2.50^{+0.10}_{-0.20}$", "$2.50^{+0.10}_{-0.20}$"])
        self.assertEqual([c[7] for c in celdas], ["$+0.10^{+0.05}_{-0.05}$"] * 4)
        self.assertEqual([c[-1] for c in celdas], [r"$-20.0$ \\", r"$0.0$ \\", r"$-2.5$ \\", r"$-10.0$ \\"])

    def test_un_companero_sin_corridas_no_deja_cabecera(self):
        self.assertEqual(self.tb.filas_tabla([], {}, {"X": "Obj~X"}), [])


@unittest.skipUnless(_hay_runs(), "sin los QC de los runs del paper")
class UmbralDelPaperTests(unittest.TestCase):
    """El umbral cromosférico y la pendiente del margen, desde los runs del paper."""

    @classmethod
    def setUpClass(cls):
        cls.tb = _carga("teff_bayes")

    def test_umbral_de_12b_2460_k(self):
        T, ratio, tipo = self.tb.umbral("ROXs12b")
        self.assertEqual(tipo, "detection")
        self.assertAlmostEqual(ratio, -3.618, delta=0.001)
        self.assertAlmostEqual(T, 2460.0, delta=2.0)

    def test_umbral_de_42bb_1806_k(self):
        T, ratio, tipo = self.tb.umbral("ROXs42Bb")
        self.assertEqual(tipo, "upper_limit")
        self.assertAlmostEqual(ratio, -4.446, delta=0.001)
        self.assertAlmostEqual(T, 1806.0, delta=2.0)

    def test_pendiente_del_margen_0370_dex_por_magnitud(self):
        for slug in ("ROXs12b", "ROXs42Bb"):
            e = self.tb.entrada_margen(slug)
            self.assertAlmostEqual(0.4 * e["a"] * e["ahav"], 0.370, delta=0.001, msg=slug)


def _hay_datos():
    tp = _carga("tipo_ir_plantillas")
    ti = _carga("tipo_ir_indices")
    rej = _carga("teff_bayes").REJILLAS
    return (_hay_runs() and ti.DEFAULT_CDS.exists() and tp.DEFAULT_NIR.exists()
            and all(r["dir"].exists() for r in rej.values()))


@pytest.mark.slow
@unittest.skipUnless(_hay_datos(), "sin los runs del paper, el IR de /mnt/2TB o las rejillas de atmósferas")
class MaquinariaTests(unittest.TestCase):
    """Sobre los datos reales: el nodo que falta en DRIFT-PHOENIX y la inyección de un modelo conocido."""

    @classmethod
    def setUpClass(cls):
        cls.tb = _carga("teff_bayes")

    def test_drift_rellena_solo_el_nodo_2500_30(self):
        m = self.tb.Modelo("ROXs42Bb", rejilla="drift", av_fija=3.0)
        self.assertEqual(m.rellenados, [(2500.0, 3.0)])
        self.assertEqual((m.nT, m.nG), (11, 7))

    def test_la_verdad_inyectada_gana_a_200_k_de_distancia(self):
        m = self.tb.Modelo("ROXs42Bb", inyecta_bt=(2250.0, 3.9, 1.9, 2.1))
        P = m.pos
        th = np.empty(m.ndim)
        for k, v in dict(teff=2250.0, logg=3.9, r_rj=2.1, d_pc=m.d_pc, av=1.9, rk=0.0).items():
            th[P[k]] = v
        for s in m.segs:
            th[P[f"s_{s['nombre']}"]] = 1.0
        for b in m.bandas:
            th[P[f"lnf_{b}"]] = math.log(1e-3)
        verdad = m.loglike(th)
        for dt in (-200.0, 200.0):
            otro = th.copy()
            otro[P["teff"]] += dt
            self.assertGreater(verdad, m.loglike(otro), dt)


if __name__ == "__main__":
    unittest.main()
