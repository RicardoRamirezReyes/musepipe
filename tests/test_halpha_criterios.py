"""Criterios de Hα de los compañeros que cita el paper (`scripts/halpha_criterios.py`).

Las primeras clases no leen productos: el estimador de W10 y el modelo de línea de la inyección sobre gaussianas de
anchura conocida, el intervalo de Neyman sobre bandas analíticas, la fracción P_pob con una población sintética, y
la dispersión alrededor de la Ec. 2 de Manara et al. (2013) sobre una tabla pequeña construida con residuos
conocidos. `DatosPublicadosTests` lee solo las tablas de VizieR del repo (`scripts/data/`) y fija las cifras del
apéndice B. `CifrasDelPaperTests` (`external_data`) lee los espectros de D2 del run del paper; sin ellos se salta.
Desde un worktree, que no lleva runs/, `MUSE_PROJECT_ROOT` apunta al clon que sí.
"""
import importlib.util
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
C_KMS = 299792.458
G10 = math.sqrt(math.log(10) / math.log(2))


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _gauss(x, centro, fwhm, area=1.0):
    s = fwhm / (2 * math.sqrt(2 * math.log(2)))
    return area / (s * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - centro) / s) ** 2)


class AnchoYModeloDeLineaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hc = _carga("halpha_criterios")

    def test_ancho_cruce_de_una_gaussiana_fina(self):
        x = np.arange(-20, 20, 0.01)
        y = _gauss(x, 0.3, 2.5)
        self.assertAlmostEqual(self.hc.ancho_cruce_A(x, y, 0.5), 2.5, places=3)
        self.assertAlmostEqual(self.hc.ancho_cruce_A(x, y, 0.1), G10 * 2.5, places=3)

    def test_sin_cruce_dentro_de_la_ventana_da_nan(self):
        x = np.arange(-2, 2, 0.1)
        self.assertTrue(math.isnan(self.hc.ancho_cruce_A(x, _gauss(x, 0.0, 10.0), 0.1)))

    def test_w10_de_gaussiana_intrinseca_por_kernel_gaussiano(self):
        """Gaussiana intrínseca de FWHM v ⊗ kernel gaussiano de FWHM L, muestreada fino: la W10 del estimador del paper
        es la de una gaussiana de FWHM sqrt(v² + L²), y el modelo integrado en canales conserva el flujo."""
        hc = self.hc
        w = np.arange(6400.0, 6720.0, 0.05)
        lsf = 2.285
        kx, kk = hc.kernel("gauss", lsf)
        v = 150.0
        linea = hc.linea_modelo(w, 1000.0, hc.HA_A, v, kx, kk)
        self.assertAlmostEqual(float(np.sum(linea * 0.05)), 1000.0, delta=0.5)
        fw_A = math.hypot(v / C_KMS * hc.HA_A, lsf)
        esperado = G10 * fw_A / hc.HA_A * C_KMS
        self.assertAlmostEqual(hc.w10_kms(w, linea + 50.0, hc.HA_A), esperado, delta=0.01 * esperado)

    def test_una_linea_sin_resolver_en_canales_de_125_a_es_mas_ancha_que_la_lsf(self):
        """El muestreo a 1.25 Å y la interpolación del cruce ensanchan la W10 de una línea sin resolver por encima de la
        de la gaussiana de la LSF (190 km/s a 2.285 Å): es lo que el paper llama la W10 instrumental del estimador."""
        hc = self.hc
        w = np.arange(4750.0, 9350.0, 1.25)
        kx, kk = hc.kernel("gauss", 2.285)
        inst, _ = hc.w10_instrumental(w, np.ones_like(w), np.zeros(w.size, int), 90.0, kx, kk)
        gauss = G10 * 2.285 / hc.HA_A * C_KMS
        self.assertGreater(inst, gauss + 10.0)
        self.assertLess(inst, gauss + 50.0)


class NeymanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hc = _carga("halpha_criterios")

    def test_bandas_lineales_dan_el_intervalo_analitico(self):
        """Estimador = v + u con u uniforme en [−10, 10]: el intervalo central al 50 % es v ± 5, así que con obs = 100
        el intervalo de Neyman de v es [95, 105]."""
        vs = np.arange(0.0, 201.0, 10.0)
        u = np.linspace(-10, 10, 20001)
        iv, lo, hi = self.hc.neyman(vs, [v + u for v in vs], 100.0, 50)
        self.assertAlmostEqual(iv[0], 95.0, delta=0.15)
        self.assertAlmostEqual(iv[1], 105.0, delta=0.15)

    def test_obs_fuera_de_todas_las_bandas_da_nan(self):
        vs = np.array([0.0, 10.0])
        iv, _, _ = self.hc.neyman(vs, [np.array([0.0, 1.0]), np.array([10.0, 11.0])], 500.0, 68)
        self.assertTrue(all(math.isnan(x) for x in iv))


class FraccionPoblacionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hc = _carga("halpha_criterios")

    def test_en_la_media_de_la_poblacion_es_un_medio(self):
        self.assertAlmostEqual(self.hc.fraccion_poblacion(np.full(100, 0.03), 0.03, 0.25), 0.5, places=12)

    def test_es_la_fraccion_de_una_poblacion_sintetica_que_alcanza_la_linea(self):
        rng = np.random.default_rng(1)
        mu, sd = 0.03, 0.25
        pob = rng.normal(mu, sd, 400000)
        for delta in (-0.3, 0.0, 0.13, 0.23, 0.5):
            self.assertAlmostEqual(self.hc.fraccion_poblacion(np.array([delta]), mu, sd),
                                   float(np.mean(pob >= delta)), delta=0.003)

    def test_promedia_sobre_las_muestras_y_suma_los_coeficientes_en_cuadratura(self):
        hc = self.hc
        d = np.array([0.0, 0.5])
        from scipy.stats import norm
        esperado = 0.5 * (norm.cdf(0.0) + norm.cdf(-0.5 / math.hypot(0.25, 0.1)))
        self.assertAlmostEqual(hc.fraccion_poblacion(d, 0.0, 0.25, np.array([0.1, 0.1])), esperado, places=12)
        self.assertAlmostEqual(hc.fraccion_poblacion(d, 0.0, 0.25, 0.25), hc.fraccion_poblacion(d, 0.0, 0.25 * math.sqrt(2)),
                               places=12)


class DispersionManaraTests(unittest.TestCase):
    """Una tabla pequeña con residuos elegidos: se fabrica F(Hα) para que a·log L_Hα + b − log L★ − Ec.2(T) dé el
    residuo pedido, y se comprueba que la dispersión sale la de esos residuos."""

    @classmethod
    def setUpClass(cls):
        cls.hc = _carga("halpha_criterios")

    def _estrella(self, nombre, teff, logL, residuo, a=1.13, b=1.74, dist=50.0):
        hc = self.hc
        lha = (residuo + logL + float(hc.suelo(teff)) - b) / a
        f = 10 ** lha * hc.LSUN_ERG_S / (4 * math.pi * (dist * hc.PC_CM) ** 2)
        return dict(nombre=nombre, spt="M5", teff=teff, logL=logL, lacc_media=logL - 3.0, dist_pc=dist, f_ha=f)

    def test_residuos_y_dispersion(self):
        hc = self.hc
        res = [0.1, -0.2, 0.3, 0.0]
        tabla = [self._estrella(f"S{i}", t, -1.0 - 0.1 * i, r) for i, (t, r) in enumerate(zip((3500, 3200, 2900, 2600), res))]
        filas = hc.residuos_manara(tabla, 1.13, 1.74)
        np.testing.assert_allclose([f["d_ha"] for f in filas], res, atol=1e-12)
        md = hc.media_y_dispersion([f["d_ha"] for f in filas])
        self.assertAlmostEqual(md["media"], np.mean(res), places=12)
        self.assertAlmostEqual(md["sd"], np.std(res, ddof=1), places=12)
        self.assertEqual(md["n"], 4)

    def test_poblacion_excluye_las_del_ajuste_y_separa_las_tardias(self):
        hc = self.hc
        tabla = [self._estrella("TWA9A", 4350, -0.6, 5.0), self._estrella("A", 3500, -1.0, 0.1),
                 self._estrella("B", 3200, -1.2, -0.1), self._estrella("C", 2900, -2.0, -0.3),
                 self._estrella("D", 2600, -2.5, -0.1), self._estrella("Sz122", 3600, -0.9, -5.0)]
        pob, _ = hc.poblacion_manara(1.13, 1.74, np.random.default_rng(0), estrellas=tabla, n_boot=50)
        self.assertEqual(pob["ha"]["n"], 4)
        self.assertAlmostEqual(pob["ha"]["media"], np.mean([0.1, -0.1, -0.3, -0.1]), places=12)
        self.assertEqual(pob["ha_tardias"]["n"], 2)
        self.assertAlmostEqual(pob["ha_tardias"]["media"], -0.2, places=12)
        self.assertEqual(len(pob["sigma_coef_T"]["sigma"]), len(pob["sigma_coef_T"]["teff"]))


class DatosPublicadosTests(unittest.TestCase):
    """Las tablas de VizieR del repo: las cifras del apéndice B (dispersión de Manara) y de la sesión (West)."""

    @classmethod
    def setUpClass(cls):
        cls.hc = _carga("halpha_criterios")

    def test_manara2013_tiene_las_24_class_iii(self):
        e = self.hc.lee_manara2013()
        self.assertEqual(len(e), 24)
        twa29 = next(x for x in e if x["nombre"] == "TWA29")
        self.assertEqual((twa29["spt"], twa29["teff"]), ("M9.5", 2330))

    def test_dispersion_del_apendice_b(self):
        pob, _ = self.hc.poblacion_manara(1.13, 1.74, np.random.default_rng(2013))
        self.assertEqual(pob["ha"]["n"], 22)
        self.assertAlmostEqual(pob["ha"]["media"], 0.0310, delta=5e-4)   # «0.03 dex above»
        self.assertAlmostEqual(pob["ha"]["sd"], 0.2501, delta=5e-4)      # «a scatter of 0.25 dex»
        self.assertEqual(pob["ha_tardias"]["n"], 3)
        self.assertAlmostEqual(pob["ha_tardias"]["media"], -0.2337, delta=5e-4)  # «0.23 dex below»
        s = [float(self.hc.sigma_coef(pob, t)) for t in (2400, 2800)]
        self.assertAlmostEqual(s[0], 0.106, delta=0.002)                 # «0.07–0.11 dex at 2400–2800 K»
        self.assertAlmostEqual(s[1], 0.066, delta=0.002)

    def test_west2011_m8_activas(self):
        ew, act = self.hc.lee_west2011("M8")
        f = self.hc.fracciones_west(ew, act)
        self.assertEqual((f["n_activas"], f["n_clasificadas"]), (334, 415))
        self.assertAlmostEqual(f["frac_activas_ge"]["26.2"], 7 / 334, places=12)

    def test_fracciones_west_sobre_una_tabla_pequena(self):
        ew = np.array([1.0, 5.0, 30.0, 50.0, 2.0])
        act = np.array(["1", "1", "1", "0", ""])
        f = self.hc.fracciones_west(ew, act, umbrales=(20,))
        self.assertEqual((f["n_activas"], f["n_clasificadas"]), (3, 4))
        self.assertAlmostEqual(f["frac_activas_ge"]["20"], 1 / 3, places=12)
        self.assertAlmostEqual(f["frac_todas_ge"]["20"], 1 / 4, places=12)


def _raiz_con_runs():
    """La raíz con los productos de D2 y G3 de los runs del paper (los de `targets/<slug>.json` -> `paper.run`)."""
    import json
    raiz = Path(os.environ.get("MUSE_PROJECT_ROOT", ROOT))
    runs = [json.loads(f.read_text())["paper"]["run"] for f in sorted((ROOT / "targets").glob("*.json"))
            if "paper" in json.loads(f.read_text())]
    hace_falta = ("spec_calibrated_psffit_object.fits", "spec_calibrated_psffit_controls.npz", "stage_g3_qc.json")
    ok = bool(runs) and all((raiz / "runs" / r / "stages" / f).exists() for r in runs for f in hace_falta)
    return raiz if ok else None


@pytest.mark.external_data
@unittest.skipUnless(_raiz_con_runs(), "sin los espectros de D2 del run del paper (MUSE_PROJECT_ROOT)")
class CifrasDelPaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hc = _carga("halpha_criterios")
        cls.ctx = cls.hc.Contexto(_raiz_con_runs())

    def test_un_detectado_y_uno_con_cota_salen_de_g3(self):
        ctx = self.ctx
        self.assertEqual(sorted(ctx.papel(s) for s in ctx.slugs), sorted((self.hc.DETECCION, self.hc.LIMITE)))
        self.assertNotEqual(ctx.detectado, ctx.limite)

    def test_ew_de_g2_y_su_error_en_los_controles(self):
        hc, ctx = self.hc, self.ctx
        det = ctx.detectado
        sp, ctr = ctx.espectro(det, "psffit"), ctx.controles(det, "psffit")
        r = hc.ew_con_controles(sp, ctr, ctx.catalogo(det), "G2", np.random.default_rng(0), n_mc=50000)
        g2 = ctx.g2_halpha(det)
        self.assertAlmostEqual(r["EW"], g2["ew_A"], places=9)      # la de G2: −26.2
        self.assertAlmostEqual(r["EW_err_ctrl"], 4.28, delta=0.1)  # «26 ± 4, el error de los 33 controles»

    def test_w10_del_companero(self):
        hc, ctx = self.hc, self.ctx
        sp = ctx.espectro(ctx.detectado, "psffit")
        self.assertAlmostEqual(hc.w10_kms(sp["wave"], sp["flux"], hc.HA_A), 256.0, delta=0.5)  # «256 km/s»


if __name__ == "__main__":
    unittest.main()
