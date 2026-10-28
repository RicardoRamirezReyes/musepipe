"""Paβ y Brγ en los espectros J/K publicados: `scripts/pab_brg.py`.

Las clases puras no tocan datos: el filtro adaptado sobre una línea sintética de amplitud conocida (recuperación
exacta con el perfil del ajuste, ~1 con la línea en una fase dada, cero sin línea), el ruido de los controles sobre
ruido blanco frente al error analítico del ajuste, la inyección, la conversión EW ↔ flujo ↔ L_acc con un caso hecho
a mano y su inversa, el límite y la dependencia con A_V de la predicción.

`MedidaDeLaSesionTests` (`external_data`) mide dos espectros del catálogo de Hayoz et al. (2026) con el BT-Settl de
`/mnt/2TB` y fija las cifras de la sesión del 2026-10-21 (`~/muse_sesion_20261021/pab_brg/salida/medidas.json`),
que el port reproduce. `CifrasDelApendiceTests` (`external_data`, ~5 s cada uno) corre el script entero sin la fotosfera:
con las entradas de la sesión fija sus L_acc y sus cocientes (`lacc_perfil.json`) y con las de hoy las cifras del
párrafo propuesto para el apéndice, tal como las imprime. Se saltan limpios sin los datos o sin `runs/`.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
#: raíz que contiene `runs/` (desde un worktree, `MUSE_PROJECT_ROOT`; los tests de las cifras se saltan sin ella)
PROYECTO = Path(os.environ.get("MUSE_PROJECT_ROOT", ROOT))
CDS = Path("/mnt/2TB/IR_Data/publicados_Hayoz2026_JAA708A312")
BTSETTL = Path("/mnt/2TB/IR_Data/btsettl_cifist/npz")
CADENAS = Path.home() / "muse_noche_20261024_c4" / "cadenas"


def _carga():
    nombre = "pab_brg"
    if nombre in sys.modules:
        return sys.modules[nombre]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _ccm():
    from musepipe.models.extinction import CCMExtinction
    return CCMExtinction(3.1, citation="Cardelli, Clayton & Mathis 1989")


def _rejilla(lo=11300.0, hi=13500.0, r_pix=500.0):
    """Rejilla log-uniforme con λ/Δλ = 500 por píxel, como el catálogo."""
    n = int(math.log(hi / lo) / math.log(1 + 1 / r_pix))
    return lo * (1 + 1 / r_pix) ** np.arange(n)


class FiltroAdaptadoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga()
        cls.w = _rejilla()
        cls.lam0 = cls.m.LINEAS["PaB"]["vacio_um"] * 1e4
        cls.c0 = int(np.argmin(np.abs(cls.w - cls.lam0)))
        cls.cont = 1e-16 * (1 + 0.1 * (cls.w - 12800.0) / 1000.0)
        cls.R = 5000.0

    def test_el_perfil_suma_uno_y_cae_en_el_pixel(self):
        for dfase in (0.0, 0.25):
            P = self.m.perfil_linea(self.w, self.lam0, self.R, dfase=dfase)
            self.assertAlmostEqual(P.sum(), 1.0, places=12)
            self.assertEqual(int(np.argmax(P)), self.c0)
            self.assertGreater(P[self.c0 - 1:self.c0 + 2].sum(), 0.99)

    def test_recupera_la_amplitud_exacta_con_el_perfil_del_ajuste(self):
        A = 3e-16
        P = self.m.perfil_linea(self.w, self.lam0, self.R, dfase=self.m.DFASE)
        f = self.cont + A * P / np.gradient(self.w)
        r = self.m.mide_perfil(self.w, f, self.c0, P)
        self.assertAlmostEqual(r["flujo"] / A, 1.0, places=9)
        # EW = −flujo / continuo en el píxel central (emisión: EW < 0)
        self.assertAlmostEqual(r["ew"] / (-A / self.cont[self.c0]), 1.0, places=9)
        self.assertLess(r["ew"], 0.0)

    def test_linea_en_una_fase_dada_y_sin_linea(self):
        """Con la línea supuesta en el centro del píxel, una línea real a ±0.25 píxeles se recupera a ±10 %, y
        la media sobre las fases es 1 (el perfil del ajuste es ese promedio). Sin línea, cero."""
        A = 3e-16
        dw = np.median(np.diff(self.w))
        lam_c = self.w[self.c0]
        Pfit = self.m.perfil_linea(self.w, lam_c, self.R, dfase=self.m.DFASE)
        rec = []
        for fase in np.linspace(-0.25, 0.25, 9):
            P = self.m.perfil_linea(self.w, lam_c + fase * dw, self.R)
            r = self.m.mide_perfil(self.w, self.cont + A * P / np.gradient(self.w), self.c0, Pfit)
            self.assertAlmostEqual(r["flujo"] / A, 1.0, delta=0.10, msg=f"fase {fase}")
            rec.append(r["flujo"] / A)
        self.assertAlmostEqual(float(np.mean(rec)), 1.0, delta=0.01)
        r0 = self.m.mide_perfil(self.w, self.cont, self.c0, Pfit)
        self.assertLess(abs(r0["flujo"]), 1e-12 * A)

    def test_la_inyeccion_recupera_uno(self):
        malos = self.m.mascara(self.w)
        lam_c = self.w[self.c0] + 0.1 * np.median(np.diff(self.w))
        eta = self.m.inyeccion("perfil", self.w, self.cont, lam_c, self.c0, self.R, malos)
        self.assertAlmostEqual(eta["media"], 1.0, delta=0.01)
        eta500 = self.m.inyeccion("perfil", self.w, self.cont, lam_c, self.c0, self.R, malos, extra500=True)
        self.assertLess(eta500["media"], 0.9)    # el perfil nominal pierde flujo si la línea es más ancha

    def test_el_ruido_de_los_controles_es_el_del_ajuste(self):
        """Sobre ruido blanco, la σ empírica de los controles cae cerca del error analítico del ajuste lineal."""
        rng = np.random.default_rng(20261028)
        s_pix = 2e-18
        f = self.cont + rng.normal(0.0, s_pix, self.w.size)
        malos = self.m.mascara(self.w)
        banda = tuple(x * 1e4 for x in self.m.LINEAS["PaB"]["banda_controles_um"])
        lin, ctl = self.m.corre("perfil", self.w, f, self.lam0, self.c0, self.R, malos, banda)
        r = self.m.resume(lin, ctl)
        self.assertGreaterEqual(r["n"], 15)
        # error analítico de A en el píxel de la línea
        h = self.m.NW // 2
        idx = np.r_[np.arange(self.c0 - h - self.m.NS, self.c0 - h), np.arange(self.c0 - h, self.c0 - h + self.m.NW),
                    np.arange(self.c0 - h + self.m.NW, self.c0 - h + self.m.NW + self.m.NS)]
        P = self.m.perfil_linea(self.w, self.lam0, self.R, dfase=self.m.DFASE)
        dw = np.gradient(self.w)
        X = np.vstack([np.ones(idx.size), self.w[idx] - self.w[self.c0], P[idx] / dw[idx]]).T
        s_an = s_pix * math.sqrt(np.linalg.inv(X.T @ X)[2, 2])
        self.assertAlmostEqual(r["sigma_flujo"] / s_an, 1.0, delta=0.35)

    def test_n_disjuntos(self):
        self.assertEqual(self.m.n_disjuntos([0, 1, 2, 3, 4, 5, 6]), 3)
        self.assertEqual(self.m.n_disjuntos([10, 0, 20]), 3)

    def test_lineas_cubiertas(self):
        self.assertEqual(self.m.lineas_cubiertas(self.w), ["PaB"])
        self.assertEqual(self.m.lineas_cubiertas(_rejilla(19500.0, 24000.0)), ["BrG"])
        self.assertEqual(self.m.lineas_cubiertas(_rejilla(22019.0, 24500.0)), [])   # el ERIS empieza en 2.20 µm
        self.assertEqual(self.m.lineas_cubiertas(_rejilla(14800.0, 18000.0)), [])

    def test_factor_de_escala(self):
        ir = {"hydrogen_lines": {"flux_rescale_mag": {"J": {"catalogue": 16.64, "adopted": 15.90}}},
              "colour_excess": {"k_rescale_mag": {"catalogue": 14.32, "adopted": 14.14}}}
        self.assertAlmostEqual(self.m.factor_escala(ir, "J")[0], 10 ** (0.4 * 0.74), places=12)
        self.assertAlmostEqual(self.m.factor_escala(ir, "K")[0], 10 ** (0.4 * 0.18), places=12)
        self.assertEqual(self.m.factor_escala(ir, "H")[0], 1.0)
        self.assertEqual(self.m.factor_escala({"hydrogen_lines": {}}, "K")[0], 1.0)


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga()
        cls.ext = _ccm()

    def test_caso_a_mano(self):
        # F = 1e-16 erg/s/cm2 a 100 pc sin extinción: 4 pi (3.0856775814913673e20 cm)^2 = 1.196495e42 cm2,
        # L = 1.196495e26 erg/s = 3.125640e-8 L_sol (L_sol = 3.828e33), log = -7.5050610;
        # Paβ (Alcalá 2017): log L_acc = 1.06 (-7.5050610) + 2.76 = -5.1953646 -> 6.377278e-6 L_sol;
        # Brγ: log L_acc = 1.19 (-7.5050610) + 4.02 = -4.9110226 -> 1.227375e-5 L_sol.
        c = self.m.lacc_de_flujo(1e-16, "PaB", av=0.0, distancia_pc=100.0, ext=self.ext)
        self.assertAlmostEqual(c["l_line_lsun"] / 3.125640e-8, 1.0, delta=2e-6)
        self.assertAlmostEqual(c["l_acc_lsun"] / 6.377278e-6, 1.0, delta=2e-6)
        c = self.m.lacc_de_flujo(1e-16, "BrG", av=0.0, distancia_pc=100.0, ext=self.ext)
        self.assertAlmostEqual(c["l_acc_lsun"] / 1.227375e-5, 1.0, delta=2e-6)
        # EW <-> flujo: EW = -F / continuo; 1e-16 sobre un continuo de 5e-17 /Å es una EW de -2 Å en emisión
        lim = self.m.limites(dict(flujo=1e-16, sigma_flujo=0.0), 5e-17, 0.0, 1.0, 1.0)
        self.assertAlmostEqual(lim["EW_lim3"], -2.0, places=12)

    def test_ew_flujo_y_limite_a_mano(self):
        # una absorción medida (F < 0) no baja el límite: F_lim = (0 + 3 σ + a_fot cont) / η
        cont = 1e-16
        lim = self.m.limites(dict(flujo=-1e-17, sigma_flujo=2e-17), cont, 0.4, 1.0, 0.5)
        self.assertAlmostEqual(lim["F_lim3"] / 1e-16, 1.0, places=12)            # 6e-17 + 4e-17
        self.assertAlmostEqual(lim["EW_lim3"], -1.0, places=12)                  # EW = −F / cont
        self.assertAlmostEqual(lim["F_umbral5"] / 1.4e-16, 1.0, places=12)       # 1e-16 + 4e-17
        self.assertAlmostEqual(lim["F_lim3_suavizado500"] / 2e-16, 1.0, places=12)
        # una emisión medida sí lo sube
        lim = self.m.limites(dict(flujo=1e-17, sigma_flujo=2e-17), cont, 0.0, 0.8, 0.8)
        self.assertAlmostEqual(lim["F_lim3"] / (7e-17 / 0.8), 1.0, places=12)

    def test_la_prediccion_es_la_inversa(self):
        for lin in ("PaB", "BrG", "Halpha"):
            for av in (0.0, 1.8, 3.7):
                c = self.m.lacc_de_flujo(2.5e-16, lin, av=av, distancia_pc=138.6, ext=self.ext)
                f = self.m.flujo_predicho(c["l_acc_lsun"], lin, av=av, distancia_pc=138.6, ext=self.ext)
                self.assertAlmostEqual(f / 2.5e-16, 1.0, places=12, msg=f"{lin} {av}")

    def test_la_extincion_cuesta_menos_en_el_infrarrojo(self):
        k = {lin: self.m.k_lambda(self.ext, self.m.reposo_A(lin)) for lin in ("Halpha", "PaB", "BrG")}
        self.assertGreater(k["Halpha"], k["PaB"])
        self.assertGreater(k["PaB"], k["BrG"])
        self.assertAlmostEqual(k["Halpha"], 0.8177774968937198, places=9)   # la del run (CCM89, R_V 3.1)

    def test_la_pendiente_con_av(self):
        """Con la Hα fija, log F_predicho es lineal en A_V con la pendiente que da `pendiente_av`."""
        for lin in ("PaB", "BrG"):
            def logf(av):
                la = self.m.lacc_de_flujo(2.38e-17, "Halpha", av=av, distancia_pc=138.6, ext=self.ext)["l_acc_lsun"]
                return math.log10(self.m.flujo_predicho(la, lin, av=av, distancia_pc=138.6, ext=self.ext))
            p = self.m.pendiente_av(lin, self.ext)
            self.assertAlmostEqual((logf(3.7) - logf(1.8)) / 1.9, p, places=10)
            self.assertAlmostEqual((logf(11.0) - logf(3.7)) / 7.3, p, places=10)
        # las pendientes de la sesión: 0.24 (Paβ) y 0.26 (Brγ)
        self.assertEqual(round(self.m.pendiente_av("PaB", self.ext), 2), 0.24)
        self.assertEqual(round(self.m.pendiente_av("BrG", self.ext), 2), 0.26)

    def test_las_relaciones(self):
        self.assertEqual((self.m.relacion("Halpha")["a"], self.m.relacion("Halpha")["b"]), (1.13, 1.74))
        self.assertEqual((self.m.relacion("PaB")["a"], self.m.relacion("PaB")["b"]), (1.06, 2.76))
        self.assertEqual((self.m.relacion("BrG")["a"], self.m.relacion("BrG")["b"]), (1.19, 4.02))
        self.assertAlmostEqual(self.m.dispersion_prediccion("PaB"), math.hypot(0.41, 0.45) / 1.06, places=12)

    def test_los_perfiles_de_targets(self):
        """Cada objeto del paper declara lo que el script necesita (y nada de eso está en el código)."""
        for f in sorted((ROOT / "targets").glob("*.json")):
            ficha = json.loads(f.read_text())
            if "paper" not in ficha:
                continue
            hl = ficha["ir_spectra"]["hydrogen_lines"]
            self.assertEqual(set(hl["photospheric_absorption_A"]), {"PaB", "BrG"}, msg=f.stem)
            self.assertIn("teff_k", hl["btsettl_reference"])
            self.assertTrue(hl["btsettl_photosphere"])
            for nombre, meta in hl["spectra"].items():
                self.assertIn(nombre, ficha["ir_spectra"]["spectra"], msg=f"{f.stem} {nombre}")
                self.assertGreater(meta["r_native"], 500)


def _hay_catalogo():
    return CDS.is_dir() and BTSETTL.is_dir()


def _hay_runs():
    """Los runs del paper (`targets/<slug>.json` -> `paper.run`) con sus QC de E3 y G3."""
    runs = [json.loads(f.read_text())["paper"]["run"] for f in sorted((ROOT / "targets").glob("*.json"))
            if "paper" in json.loads(f.read_text())]
    hace_falta = ("stage_h03_qc.json", "stage_g3_qc.json")
    return bool(runs) and all((PROYECTO / "runs" / r / "stages" / f).exists() for r in runs for f in hace_falta)


@pytest.mark.external_data
@unittest.skipUnless(_hay_catalogo(), "sin el catálogo de Hayoz et al. 2026 o sin BT-Settl en /mnt/2TB")
class MedidaDeLaSesionTests(unittest.TestCase):
    """El port reproduce la medida de la sesión del 2026-10-21 (estimador adoptado, sin fotosfera)."""

    @classmethod
    def setUpClass(cls):
        cls.m = _carga()

    def _mide(self, slug, nombre):
        p = self.m.perfil_objeto(slug, root=ROOT)
        ir = p["ir_spectra"]
        spec = ir["spectra"][nombre]
        hl = ir["hydrogen_lines"]
        fac, _ = self.m.factor_escala(ir, spec["band"])
        w, f, s = self.m.carga(CDS, spec["file"], fac)
        (lin,) = self.m.lineas_cubiertas(w)
        ref = hl["btsettl_reference"]
        modelo = self.m.btsettl(BTSETTL, ref["teff_k"], ref["logg"])
        return self.m.mide_espectro(w, f, s, lin, float(hl["spectra"][nombre]["r_native"]), modelo,
                                    estimadores=("perfil",), variantes=False)

    def _compara(self, res, esperado):
        r = res["estimadores"]["perfil"]
        self.assertAlmostEqual(res["desplazamiento"]["delta_A"], esperado["dlam"], places=9)
        self.assertAlmostEqual(res["fase_px"], esperado["fase"], places=9)
        for k in ("ew", "sigma_ew", "z", "flujo", "sigma_flujo"):
            self.assertAlmostEqual(r[k] / esperado[k], 1.0, places=7, msg=k)
        self.assertEqual((r["n"], r["n_disjuntos"]), esperado["n"])
        self.assertAlmostEqual(r["inyeccion"]["media"], esperado["eta"], places=7)

    def test_paschen_beta_nifs_del_detectado(self):
        self._compara(self._mide("ROXs12b", "NIFS_J"), dict(
            dlam=1.8799234146657897, fase=-0.06068331470883929, ew=-0.12660488967633912, sigma_ew=0.5285999196485983,
            z=0.2904993393189814, flujo=1.8943510078577337e-17, sigma_flujo=6.521016578897106e-17, n=(23, 9),
            eta=1.0000000019047481))

    def test_el_exceso_de_brackett_gamma_de_osiris(self):
        res = self._mide("ROXs42Bb", "OSIRIS_K")
        self._compara(res, dict(
            dlam=4.344684266815243, fase=0.49595593280948, ew=-2.6210390836192774, sigma_ew=0.8461863530640553,
            z=3.2887365327551428, flujo=1.1777715727144457e-16, sigma_flujo=3.5812281129366305e-17, n=(21, 8),
            eta=1.0000075893489417))
        # ningún control llega, y el exceso cae a +34 Å (≈ +470 km/s) de Brγ: no en los dos píxeles que
        # compartiría una línea en λ0 (fase +0.50)
        self.assertEqual(res["estimadores"]["perfil"]["fraccion_controles_mas_emision"], 0.0)
        c = self.m.centroide_exceso(res)
        self.assertEqual(round(c["desde_lam0_A"]), 34)
        self.assertEqual(round(c["desde_lam0_kms"], -1), 470)


@pytest.mark.external_data
@unittest.skipUnless(_hay_catalogo() and _hay_runs(), "sin el catálogo, BT-Settl o los runs del paper (MUSE_PROJECT_ROOT)")
class CifrasDelApendiceTests(unittest.TestCase):
    """El script entero (sin la fotosfera, que no entra en los límites): las cifras de la sesión con sus entradas
    y las del párrafo con las de hoy."""

    @classmethod
    def setUpClass(cls):
        cls.m = _carga()

    def _corre(self, *extra):
        import scripts.paper_figures as pf
        antes = pf.PROYECTO
        try:
            with tempfile.TemporaryDirectory() as tmp:
                j = Path(tmp) / "pab_brg.json"
                args = ["--project-root", str(PROYECTO), "--sin-fotosfera", "--json", str(j), *extra]
                self.assertEqual(self.m.main(args), 0)
                return json.loads(j.read_text())
        finally:
            pf.PROYECTO = antes

    def test_reproduce_la_sesion(self):
        d = self._corre("--f-halpha", "ROXs12b=2.54e-17", "--f-halpha", "ROXs42Bb=5.13e-18",
                        "--av", "ROXs12b=1.8,3.0,3.7", "--av", "ROXs42Bb=1.9,3.0", "--av-hipotesis", "1.9,3.0,5.5")
        # lacc_perfil.json de la sesión: F_lim3, EW_lim3, L_acc(3σ) y umbral5/predicción por A_V. La L_acc difiere
        # en 1.5e-5 relativo porque la sesión usaba pc = 3.0857e18 y aquí es el de musepipe (IAU 2015).
        sesion = {
            "ROXs12b": {"NIFS_J:PaB": (2.744248077157221e-16, -1.8340593882117977,
                                       {"1.8": (5.978023358754497e-05, 168.80009366831638),
                                        "3.7": (9.878404952944266e-05, 58.96589491892448)}),
                        "OSIRIS_J:PaB": (3.055303616975475e-16, -2.0790265963507144,
                                         {"1.8": (6.698637457865246e-05, 205.59959780560493)}),
                        "OSIRIS_K:BrG": (2.4665507878292764e-16, -2.3529867279407917,
                                         {"1.8": (9.833212705325871e-05, 129.80521422567858),
                                          "3.7": (0.00012530333621634996, 40.88629492669059)})},
            "ROXs42Bb": {"NIFS_J:PaB": (1.3788246006749248e-16, -2.5648768763838707,
                                        {"1.9": (3.325779524960759e-05, 468.3849931832806)}),
                         "OSIRIS_K:BrG": (2.2521229144487773e-16, -5.011924482334212,
                                          {"1.9": (0.00010189579873305761, 339.15513125389407)}),
                         "SINFONI_K:BrG": (2.264695476676835e-16, -4.830703490402071,
                                           {"1.9": (0.0001025730734460265, 658.2944577270031),
                                            "3.0": (0.00011802595888774051, 337.251568367518)})}}
        for slug, espectros in sesion.items():
            for clave, (flim, ewlim, por_av) in espectros.items():
                x = d["objetos"][slug]["espectros"][clave]
                self.assertAlmostEqual(x["F_lim3"] / flim, 1.0, places=7, msg=clave)
                self.assertAlmostEqual(x["EW_lim3"] / ewlim, 1.0, places=7, msg=clave)
                for av, (lacc, ratio) in por_av.items():
                    g = x["por_av"][av]
                    self.assertAlmostEqual(g["L_acc_lim3"] / lacc, 1.0, delta=3e-5, msg=f"{clave} {av}")
                    self.assertAlmostEqual(g["umbral5_sobre_pred"] / ratio, 1.0, delta=3e-5, msg=f"{clave} {av}")
        # si el exceso de OSIRIS fuera una línea (sesión §3.3): Hα ×297 el límite y Paβ a 3.8σ con A_V 1.9
        h = d["objetos"]["ROXs42Bb"]["excesos"]["OSIRIS_K:BrG"]["hipotesis_linea"]["1.9"]
        self.assertEqual(round(h["F_ha_pred_sobre_medida"]), 297)
        self.assertEqual(round(h["otras"]["NIFS_J:PaB"]["tension_sigma"], 1), 3.8)
        self.assertEqual(round(h["otras"]["NIFS_J:PaB"]["EW_pred"], 2), -3.56)

    @unittest.skipUnless(CADENAS.is_dir(), "sin las cadenas de T_eff del 2026-10-24")
    def test_las_cifras_del_parrafo(self):
        d = self._corre("--cadenas", str(CADENAS))
        c12, c42 = d["cifras"]["ROXs12b"], d["cifras"]["ROXs42Bb"]
        # entradas: la integración directa de G2 y el límite de E3; A_V 1.8–3.7 (la 3.7, la del ajuste con prior)
        self.assertEqual(f"{c12['f_halpha']:.2e}", "2.38e-17")
        self.assertEqual(f"{c42['f_halpha']:.2e}", "4.46e-18")
        self.assertEqual(c12["avs"], [1.8, 3.0, 3.7])
        # límites de EW a 3σ (Å) y L_acc a la A_V adoptada
        ew = [round(v["ew_lim3"], 1) for c in (c12, c42) for v in c["espectros"].values()]
        self.assertEqual(sorted(ew), [1.8, 2.1, 2.4, 2.6, 4.8, 5.0])
        self.assertEqual(f"{c12['espectros']['NIFS_J:PaB']['lacc_lim3_av_adoptada']:.1e}", "6.0e-05")
        self.assertEqual(f"{c12['espectros']['OSIRIS_K:BrG']['lacc_lim3_av_adoptada']:.1e}", "9.8e-05")
        self.assertEqual(f"{c42['espectros']['NIFS_J:PaB']['lacc_lim3_av_adoptada']:.1e}", "3.3e-05")
        # la lectura de acreción de 12 b: EW 0.014–0.065 Å, 43–220 veces bajo el umbral a 5σ (4–16 con +2σ),
        # A_V ≈ 11 (Paβ) y ≈ 10 (Brγ) para alcanzarlo
        self.assertEqual([round(x, 3) for x in c12["ew_pred"]], [0.014, 0.065])
        self.assertEqual([round(x) for x in c12["umbral5_sobre_pred"]], [43, 220])
        self.assertEqual([round(x) for x in c12["umbral5_sobre_pred_mas2sigma"]], [4, 16])
        self.assertEqual(round(c12["PaB"]["av_umbral5"][0]), 11)
        self.assertEqual(round(c12["BrG"]["av_umbral5"][0]), 10)
        # el exceso de OSIRIS en 42B b no cambia (los espectros son los mismos); su Hα, sí
        ex = d["objetos"]["ROXs42Bb"]["excesos"]["OSIRIS_K:BrG"]
        self.assertEqual(round(ex["z"], 1), 3.3)
        self.assertEqual(round(ex["hipotesis_linea"]["1.9"]["F_ha_pred_sobre_medida"]), 342)


if __name__ == "__main__":
    unittest.main()
