"""Máscaras telúricas del espectro MUSE en los ajustes de T_eff (pruebas exploratorias del 2026-10-28).

Dos piezas: `--mascara-muse` de `scripts/teff_bayes.py` (con `tipo_ir_plantillas.sin_o2`, que quita las bandas de O2
de G3 en una copia del config) y el residuo telúrico fino en O2 de `scripts/g3_residuo_telurico_fino.py`. Las
primeras clases no leen nada de disco. `DatosRealesTests` (marca `external_data`) lee los runs del paper de
`MUSE_PROJECT_ROOT` y las cadenas de `MUSE_TEFF_CADENAS`; si falta algo, se salta.
"""
import argparse
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


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _args(**kw):
    base = dict(slug="X", rejilla="btsettl", modo="libre", tracks="ATMO2020", av="primaria", plantilla=None,
                av_fija=None, inyecta_bt=None, datos="todo", r250=False, degrada="cuadratura", semilla=1,
                etiqueta=None, mascara_muse="paper")
    base.update(kw)
    return argparse.Namespace(**base)


class OpcionMascaraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = _carga("teff_bayes")
        cls.tp = _carga("tipo_ir_plantillas")

    def test_las_cuatro_variantes(self):
        M = self.tb.MASCARAS_MUSE
        self.assertEqual(list(M), ["paper", "g3", "sin_o2", "sin_o2b", "sin_telurico"])
        self.assertEqual(M["paper"], dict(h2o=True, quita_o2=()))     # lo de las cadenas del paper
        self.assertEqual(M["g3"], dict(h2o=False, quita_o2=()))
        self.assertEqual(M["sin_o2"], dict(h2o=True, quita_o2=("O₂ B", "O₂ A")))
        self.assertEqual(M["sin_o2b"], dict(h2o=True, quita_o2=("O₂ B",)))
        self.assertEqual(M["sin_telurico"], dict(h2o=False, quita_o2=("O₂ B", "O₂ A")))
        self.assertEqual(set(M["sin_o2"]["quita_o2"]), set(self.tp.O2_MUSE))

    def test_una_variante_que_no_existe_no_se_acepta(self):
        with self.assertRaises(SystemExit):
            self.tb.main(["ajusta", "X", "--salida", "/nonexistent", "--mascara-muse", "otra"])

    def test_sin_la_opcion_vale_paper(self):
        a = _args()
        del a.mascara_muse   # un Namespace de antes de la opción
        self.assertEqual(self.tb.etiqueta_de(a), "X_libre_avprimaria_s1")

    def test_la_etiqueta_del_paper_no_cambia(self):
        self.assertEqual(self.tb.etiqueta_de(_args()), "X_libre_avprimaria_s1")
        self.assertEqual(self.tb.etiqueta_de(_args(av_fija=1.8, rejilla="drift")), "X_drift_libre_avprimaria_avfija1.8_s1")
        self.assertEqual(self.tb.etiqueta_de(_args(rejilla="atmo_ceq1500")), "X_atmo_ceq1500_libre_avprimaria_s1")

    def test_la_etiqueta_lleva_la_variante(self):
        for m in ("g3", "sin_o2", "sin_o2b", "sin_telurico"):
            et = self.tb.etiqueta_de(_args(av_fija=1.8, mascara_muse=m))
            self.assertEqual(et, f"X_libre_avprimaria_avfija1.8_mascara-{m}_s1")
        # una etiqueta explícita manda, como antes
        self.assertEqual(self.tb.etiqueta_de(_args(etiqueta="fija", mascara_muse="g3")), "fija")

    def test_una_variante_desconocida_falla_antes_de_leer_nada(self):
        R = self.tb.REJILLAS["btsettl"]
        with self.assertRaises(ValueError):
            self.tb._prepara("no-existe", R, mascara_muse="otra")

    def test_las_variantes_no_entran_en_las_tablas_del_paper(self):
        d = dict(etiqueta="e", slug="X", modo="libre", av="primaria", rejilla="btsettl", datos="todo", r_analisis=500)
        self.assertTrue(self.tb._de_la_tabla(d))
        self.assertTrue(self.tb._de_rejillas(d))
        for m in ("g3", "sin_o2", "sin_o2b", "sin_telurico"):
            self.assertFalse(self.tb._de_la_tabla(dict(d, mascara_muse=m)), m)
            self.assertFalse(self.tb._de_rejillas(dict(d, mascara_muse=m)), m)
        self.assertTrue(self.tb._de_la_tabla(dict(d, mascara_muse="paper")))

    def test_sin_o2_quita_solo_el_o2_y_no_toca_el_original(self):
        cfg = {"g3_fit_masks": {"line_window_kms": 300.0,
                                "telluric_bands_A": [[6860.0, 6960.0], [7130.0, 7360.0], [7590.0, 7700.0]]},
               "otra": [1, 2]}
        antes = json.dumps(cfg, sort_keys=True)
        out = self.tp.sin_o2(cfg)
        self.assertEqual(out["g3_fit_masks"]["telluric_bands_A"], [[7130.0, 7360.0]])   # H2O se queda
        self.assertEqual(out["g3_fit_masks"]["line_window_kms"], 300.0)
        self.assertEqual(out["otra"], [1, 2])
        self.assertEqual(json.dumps(cfg, sort_keys=True), antes)   # copia, no en el sitio
        # sin la clave valen las de D9 por defecto (O2 B y O2 A): no queda ninguna
        self.assertEqual(self.tp.sin_o2({})["g3_fit_masks"]["telluric_bands_A"], [])
        # una sola banda: O2 B fuera, O2 A dentro
        self.assertEqual(self.tp.sin_o2(cfg, ("O₂ B",))["g3_fit_masks"]["telluric_bands_A"],
                         [[7130.0, 7360.0], [7590.0, 7700.0]])
        with self.assertRaises(ValueError):
            self.tp.sin_o2(cfg, ("O2 Z",))


def _banda_o2_sintetica(w):
    """Transmisión con estructura de bosque (líneas estrechas) dentro de 7590–7700 Å."""
    T = np.ones_like(w)
    rng = np.random.default_rng(3)
    for c in rng.uniform(7592.0, 7698.0, 60):
        T *= 1 - 0.5 * np.exp(-0.5 * ((w - c) / 1.2) ** 2)
    return T


class ResiduoTeluricoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = _carga("g3_residuo_telurico_fino")
        cls.w = np.arange(7300.0, 8000.0, 1.25)
        cls.T = _banda_o2_sintetica(cls.w)
        cls.f0 = 1e-16 * (1 + 0.2 * (cls.w - 7300.0) / 700.0)

    def test_recupera_la_amplitud_inyectada(self):
        for a in (0.0, 0.1, 0.5, 1.0):
            r = self.g.mide(self.w, self.f0 * self.T ** a, self.T, "o2_a")
            self.assertAlmostEqual(r["alpha"], a, delta=0.03, msg=a)

    def test_el_residuo_en_flujo_es_uno_menos_t_a_la_alfa(self):
        a = 0.2
        r = self.g.residuo_flujo(self.w, self.T, "o2_a", a)
        m = (self.w >= 7590.0) & (self.w <= 7700.0)
        self.assertAlmostEqual(r["media"], 100 * np.mean(1 - self.T[m] ** a), places=10)
        self.assertGreater(r["max_canal"], r["max_bin15"])   # promediar en 15 Å lo diluye
        self.assertEqual(self.g.residuo_flujo(self.w, self.T, "o2_a", 0.0)["media"], 0.0)

    def test_la_nula_y_la_cota_a_dos_sigma(self):
        rng = np.random.default_rng(0)
        f = self.f0 * (1 + 0.01 * rng.normal(size=self.w.size))
        r = self.g.mide(self.w, f, self.T, "o2_a")
        self.assertEqual(r["n_nula"], 18)
        self.assertGreater(r["sigma_nula"], 0)
        cota = self.g.residuo_flujo(self.w, self.T, "o2_a", abs(r["alpha"]) + 2 * r["sigma_nula"])
        self.assertEqual(r["residuo_2sigma_pct"], cota)
        # los rasgos de K I salen de la medida y se dan aparte
        self.assertEqual([x["nombre"] for x in r["rasgos"]], ["K I 7665", "K I 7699"])
        self.assertIn("alpha_con_rasgos", r)
        self.assertLess(r["n_canales"], int(((self.w > 7590.0) & (self.w < 7700.0)).sum()))

    def test_ganancia_uno_sin_ruido(self):
        for v in self.g.ganancia(self.w, self.f0, self.T, "o2_a").values():
            self.assertAlmostEqual(v, 1.0, delta=0.05)

    def test_carga_patron_de_fits_y_de_npy(self):
        from astropy.io import fits
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cols = [fits.Column(name="wave_A", format="D", array=self.w),
                    fits.Column(name="transmission", format="D", array=self.T)]
            fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols, name="TELLURIC_TRANS")]).writeto(
                d / "TELLURIC_TRANS.fits")
            w, T = self.g.carga_patron(d / "TELLURIC_TRANS.fits")
            np.testing.assert_array_equal(T, self.T)
            np.save(d / "wave_A.npy", self.w)
            np.save(d / "transmission_molecfit_combined.npy", self.T)
            w2, T2 = self.g.carga_patron(d / "transmission_molecfit_combined.npy")
            np.testing.assert_array_equal(w2, self.w)
            np.testing.assert_array_equal(T2, self.T)

    def test_tramos_junta_los_huecos_cortos(self):
        w = np.arange(0.0, 100.0, 1.0)
        x = np.zeros_like(w)
        x[10:20] = x[23:30] = x[60:70] = 1.0
        self.assertEqual(self.g.tramos(w, x, 0.5, hueco=5.0), [[10.0, 29.0], [60.0, 69.0]])
        self.assertEqual(self.g.tramos(w, x, 0.5, hueco=2.0), [[10.0, 19.0], [23.0, 29.0], [60.0, 69.0]])
        self.assertEqual(self.g.tramos(w, np.zeros_like(w), 0.5), [])

    def test_bordes_cuenta_donde_la_curva_vale_uno(self):
        bd = self.g.bordes(self.w, self.T, regiones=((7500.0, 7800.0),), umbrales=(0.01,),
                           ventanas={"cola": (7710.0, 7750.0)})
        self.assertEqual(bd["canales"], self.w.size)
        self.assertGreater(bd["canales_T_igual_1"], 0)
        lo, hi = bd["tramos_T_menor_que_1"][0]
        self.assertGreaterEqual(lo, 7300.0)
        (t,) = bd["tramos"]
        self.assertTrue(t["nativa"] and all(7550.0 < a <= b < 7750.0 for a, b in t["nativa"]))
        self.assertLess(bd["ventanas"]["cola"]["media_pct"], 0.5)

    def test_en_rejilla_vale_uno_fuera_de_la_curva(self):
        w = np.array([7000.0, 7650.0, 9000.0])
        T = self.g.en_rejilla(w, self.w, self.T)
        self.assertEqual((T[0], T[2]), (1.0, 1.0))
        self.assertAlmostEqual(T[1], float(np.interp(7650.0, self.w, self.T)), places=12)


# ------------------------------------------------------------------ datos reales
PROYECTO = Path(os.environ.get("MUSE_PROJECT_ROOT", ROOT))
CADENAS = Path(os.environ.get("MUSE_TEFF_CADENAS", "~/muse_noche_20261024_c4/cadenas")).expanduser()


def _runs_del_paper():
    fichas = [json.loads(f.read_text()) for f in sorted((ROOT / "targets").glob("*.json"))]
    return {d["slug"]: d["paper"]["run"] for d in fichas if "paper" in d}


@pytest.mark.external_data
class DatosRealesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = _runs_del_paper()
        if not cls.runs or not all((PROYECTO / "runs" / r / "stages" / "spec_final_object.fits").exists()
                                   for r in cls.runs.values()):
            raise unittest.SkipTest("sin los runs del paper (MUSE_PROJECT_ROOT)")
        cls.tb = _carga("teff_bayes")
        cls.raiz_antes = cls.tb.ROOT
        cls.tb.fija_raiz(PROYECTO)
        from scripts import tipo_ir_plantillas as tp
        cls.tp = tp

    @classmethod
    def tearDownClass(cls):
        cls.tb.fija_raiz(cls.raiz_antes)

    def test_bins_por_variante(self):
        """84 bins con las máscaras del paper (apéndice B); cada variante añade los de sus bandas y nada más."""
        for slug, run in self.runs.items():
            n = {m: int(self.tp.segmento_muse(run, h2o=v["h2o"], quita_o2=v["quita_o2"])["bueno"].sum())
                 for m, v in self.tb.MASCARAS_MUSE.items()}
            self.assertEqual(n["paper"], 84, slug)
            self.assertEqual(n["g3"] - n["paper"], n["sin_telurico"] - n["sin_o2"], slug)   # los de H2O
            self.assertEqual(n["sin_o2"] - n["paper"], n["sin_telurico"] - n["g3"], slug)   # los de O2
            self.assertGreater(n["sin_o2"], n["sin_o2b"])
            self.assertGreater(n["sin_o2b"], n["paper"])
            self.assertGreater(n["g3"], n["paper"])
            # `teff_bayes.py bins` cuenta lo mismo y reparte los bins por banda telúrica
            b = self.tb.bins_muse(slug)
            self.assertEqual({m: v["n"] for m, v in b.items()}, n)
            for m in ("paper", "g3"):
                self.assertEqual((b[m]["por_banda"]["O₂ B"], b[m]["por_banda"]["O₂ A"]), (0, 0), (slug, m))
            for m in ("sin_o2", "sin_telurico"):
                self.assertGreater(b[m]["por_banda"]["O₂ A"], 0, (slug, m))
            self.assertEqual(b["sin_o2"]["por_banda"]["H₂O 8200"], 0, slug)
            self.assertEqual((b["sin_o2b"]["por_banda"]["O₂ B"] > 0, b["sin_o2b"]["por_banda"]["O₂ A"]), (True, 0), slug)

    def test_sin_o2_destapa_las_dos_bandas_de_o2(self):
        run = next(iter(self.runs.values()))
        w_paper = self.tp.segmento_muse(run)["w"]
        w = self.tp.segmento_muse(run, quita_o2=self.tp.O2_MUSE)["w"]
        for lo, hi in ((6860.0, 6960.0), (7590.0, 7700.0)):
            self.assertFalse(np.any((w_paper > lo) & (w_paper < hi)))
            self.assertTrue(np.any((w > lo) & (w < hi)))

    def test_la_variante_paper_da_el_logl_max_de_las_cadenas_bit_a_bit(self):
        d = CADENAS / "A1" / "res"
        if not d.is_dir():
            self.skipTest("sin las cadenas del paper (MUSE_TEFF_CADENAS)")
        try:
            filas = self.tb.comprueba([d])
        except (OSError, SystemExit, RuntimeError) as exc:   # sin las rejillas de /mnt/2TB
            self.skipTest(f"no se puede reconstruir el modelo: {exc}")
        self.assertTrue(filas)
        for f in filas:
            self.assertTrue(f["bit_a_bit"], (f["etiqueta"], f["loglike"], f["logl_max"]))
            self.assertTrue(f["n_puntos_iguales"], f["etiqueta"])


@pytest.mark.external_data
class ResiduoReproduceLaSesionTests(unittest.TestCase):
    """La α de H2O 8100–8400 Å del 2026-09-24 (docs/2026-09-24_residuo_telurico_h2o_y_mascara_d9.md, §2) con las
    entradas de entonces: los espectros respaldados en los runs y la curva de la estándar."""

    CASOS = {  # run del paper -> (respaldo con los espectros de entonces, primaria, compañero)
        "ROXs12b_invvar": ("_antes_d2_a3_20260925T141314", "spec_calibrated_psffit_star.fits",
                           "spec_calibrated_psffit_object.fits", (-0.01, 0.18), (+0.11, 0.54)),
        "ROXs42Bb_realigned": ("_antes_psf_centrado_20261021T224346", "spec_calibrated_psffit_star.fits",
                               "spec_final_object.fits", (-0.13, 0.16), (+0.01, 0.63)),
    }
    CURVA = PROYECTO / "runs" / "ROXs12b_raw" / "raw_reduction" / "TELLURIC_TRANS.fits"

    def test_reproduce_las_cuatro_alfas(self):
        g = _carga("g3_residuo_telurico_fino")
        if not self.CURVA.exists():
            self.skipTest("sin la curva de la estándar (MUSE_PROJECT_ROOT)")
        wT, T = g.carga_patron(self.CURVA)
        for run, (resp, prim, comp, a_prim, a_comp) in self.CASOS.items():
            base = PROYECTO / "runs" / run / "stages" / resp / "stages"
            for fichero, (a, s) in ((prim, a_prim), (comp, a_comp)):
                if not (base / fichero).exists():
                    self.skipTest(f"sin el respaldo {base / fichero}")
                w, f, _ = g.spec(base / fichero)
                r = g.mide(w, f, g.en_rejilla(w, wT, T), "h2o_8200")
                self.assertEqual(round(r["alpha"], 2) + 0.0, a, (run, fichero))
                self.assertEqual(round(r["sigma_nula"], 2), s, (run, fichero))
                self.assertFalse(math.isnan(r["z"]))

    def test_bin_a_bin_en_o2_con_la_primaria(self):
        g = _carga("g3_residuo_telurico_fino")
        if not self.CURVA.exists():
            self.skipTest("sin la curva de la estándar (MUSE_PROJECT_ROOT)")
        from scripts import teff_bayes as stb   # el módulo que `compara_por_bin` re-enraíza
        antes = stb.ROOT
        try:
            for slug in _runs_del_paper():
                filas = g.compara_por_bin(slug, PROYECTO, g.carga_patron(self.CURVA))
                self.assertEqual({f["banda"] for f in filas}, {"o2_b", "o2_a"}, slug)
                for f in filas:
                    self.assertGreaterEqual(f["residuo_2sigma_pct"], abs(f["residuo_central_pct"]) - 1e-12)
                    self.assertGreater(f["err_rel_companero_pct"], 0)
                    self.assertAlmostEqual(f["cota_sobre_error"], f["residuo_2sigma_pct"] / f["err_rel_companero_pct"])
        finally:
            stb.fija_raiz(antes)


if __name__ == "__main__":
    unittest.main()
