"""Cifras por noche que cita el paper (`scripts/cifras_por_noche_d2.py`).

Las primeras clases no leen nada de disco: el ancho a una fracción del pico, la resta de la LSF en cuadratura y la
EW con la ventana ancha, sobre gaussianas sintéticas donde la respuesta es analítica. `CifrasDelPaperTests` lee los
espectros calibrados de las dos noches de ROXs 12 b y fija las cifras del paper (la razón de la primaria, sus EW y
los anchos a 10 % de White & Basri 2003, añadidos el 2026-10-16); si faltan los runs se salta, no falla.
"""
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C_KMS = 299792.458


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _gauss(w, centro, sigma_A, area):
    return area / (sigma_A * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((w - centro) / sigma_A) ** 2)


class AnchosYEwTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cn = _carga("cifras_por_noche_d2")
        cls.w = np.arange(6400.0, 6700.0, 0.05)

    def test_ancho_a_10_y_a_50_por_ciento_de_una_gaussiana(self):
        sigma = 2.0
        f = 1.0 + _gauss(self.w, self.cn.HA_A, sigma, 10.0)
        a10 = self.cn.ancho_kms(self.w, f, fraccion=0.1)
        self.assertAlmostEqual(a10, C_KMS * 2 * math.sqrt(2 * math.log(10)) * sigma / self.cn.HA_A, delta=1.0)
        self.assertAlmostEqual(self.cn.fwhm_kms(self.w, f), C_KMS * 2 * math.sqrt(2 * math.log(2)) * sigma / self.cn.HA_A,
                               delta=1.0)
        self.assertAlmostEqual(a10 / self.cn.fwhm_kms(self.w, f), self.cn.ANCHO10_GAUSS, places=3)

    def test_la_lsf_se_resta_en_cuadratura(self):
        # gaussiana intrínseca convolucionada con una LSF gaussiana: el ancho observado es la suma en cuadratura
        sig_int, lsf_fwhm = 2.5, 2.3
        sig_lsf = lsf_fwhm / (2 * math.sqrt(2 * math.log(2)))
        f = 1.0 + _gauss(self.w, self.cn.HA_A, math.hypot(sig_int, sig_lsf), 10.0)
        intrinseco = C_KMS * 2 * math.sqrt(2 * math.log(10)) * sig_int / self.cn.HA_A
        a10 = self.cn.ancho_kms(self.w, f, fraccion=0.1)
        self.assertAlmostEqual(self.cn.sin_lsf_kms(a10, lsf_fwhm), intrinseco, delta=1.5)
        self.assertTrue(math.isnan(self.cn.sin_lsf_kms(10.0, lsf_fwhm)))  # más estrecho que la LSF

    def test_sin_cruce_dentro_de_la_ventana_da_nan(self):
        f = 1.0 + _gauss(self.w, self.cn.HA_A, 8.0, 50.0)  # a 10 % del pico la línea se sale de ±10 Å
        self.assertTrue(math.isnan(self.cn.ancho_kms(self.w, f, fraccion=0.1)))

    def test_ew_ancha_cuenta_las_alas_que_la_ventana_estrecha_deja_fuera(self):
        cont = 2.0 + 0.001 * (self.w - 6500.0)  # continuo con pendiente: el ajuste lineal lo sigue
        area = 15.0
        f = cont + _gauss(self.w, self.cn.HA_A, 4.0, area)
        c_ha = 2.0 + 0.001 * (self.cn.HA_A - 6500.0)
        self.assertAlmostEqual(self.cn.ew_ancha(self.w, f), -area / c_ha, delta=0.01)
        fl, c0, _, _ = self.cn.integrado(self.w, f)
        self.assertLess(abs(fl / c0), abs(self.cn.ew_ancha(self.w, f)))  # ±6 Å pierde las alas de σ = 4 Å


def _noches():
    """slug -> (run de la noche 1, run de la noche 2) de los objetos con segunda época declarada."""
    out = {}
    for p in sorted((ROOT / "targets").glob("*.json")):
        paper = json.loads(p.read_text()).get("paper", {})
        if paper.get("previous_run") and paper.get("second_epoch_run"):
            out[p.stem] = (paper["previous_run"], paper["second_epoch_run"])
    return out


def _hay_noches():
    return any(all((ROOT / "runs" / r / "stages" / f).exists() for r in runs
                   for f in ("spec_calibrated_psffit_star.fits", "spec_calibrated_psffit_object.fits"))
               for runs in _noches().values())


#: las cifras del paper por objeto (Sect. 4.1, Fig. de variabilidad y Sect. 5.1)
# 2026-10-25, evaluador de la PSF centrado (docs/2026-10-21_psf_evaluador_centrado.md, noches del 10-23) y C4 v2
# decoupled_c0 (docs/2026-10-24_c4_planos_desacoplados_decision.md): eran 6.05, (-1.55, -8.26), (-2.59, -11.37),
# (315, 526) y (251, 490)
DEL_PAPER = {"ROXs12b": {"razon": 5.87, "ew": (-1.54, -8.17), "ew_ancha": (-2.63, -11.265), "ancho10": (315, 521),
                         "sin_lsf": (250, 484)}}


@unittest.skipUnless(_hay_noches(), "sin los espectros calibrados de las dos noches")
class CifrasDelPaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cn = _carga("cifras_por_noche_d2")

    def test_las_cifras_por_noche_del_paper(self):
        for slug, (r1, r2) in _noches().items():
            if slug not in DEL_PAPER:
                continue
            esperado = DEL_PAPER[slug]
            n = [self.cn.noche(ROOT / "runs" / r / "stages", self.cn.lsf_del_run(r)) for r in (r1, r2)]
            with self.subTest(slug=slug):
                self.assertAlmostEqual(n[1]["F_prim"] / n[0]["F_prim"], esperado["razon"], delta=0.005)
                for i in range(2):
                    self.assertAlmostEqual(n[i]["EW_prim_A"], esperado["ew"][i], delta=0.005)
                    self.assertAlmostEqual(n[i]["EW_prim_ancha_A"], esperado["ew_ancha"][i], delta=0.005)
                    self.assertAlmostEqual(n[i]["ancho10_prim_kms"], esperado["ancho10"][i], delta=0.5)
                    self.assertAlmostEqual(n[i]["ancho10_prim_sin_lsf_kms"], esperado["sin_lsf"][i], delta=0.5)


if __name__ == "__main__":
    unittest.main()
