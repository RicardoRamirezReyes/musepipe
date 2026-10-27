"""ATMO 2020 como rejilla de `scripts/teff_bayes.py`: el adaptador (`scripts/convierte_atmo.py`) y el degradado de
una rejilla muestreada en número de onda (`teff_bayes.degrada_nativa`).

La prueba de ATMO 2020 del apéndice B del paper (agente `atmo`, 2026-10-26). Las primeras clases no leen nada de
disco: un fichero ATMO sintético (dos columnas, µm y W m⁻² µm⁻¹, un cuerpo negro en un muestreo uniforme de 10 cm⁻¹,
como los de los autores) para la lectura, la conversión de unidades, el criterio ∫F dλ = σT⁴ y el control
«BT-Settl con el muestreo de ATMO»; y el degradado con la resolución del muestreo restada en cuadratura.
`RejillaRealTests` (marca `external_data`) convierte nodos de la rejilla real de /mnt/2TB si está; si no, se salta.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ATMO_TXT = Path("/mnt/2TB/IR_Data/atmo2020/atmosphere_models/CEQ_spectra")


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


H, C, KB = 6.62607015e-27, 2.99792458e10, 1.380649e-16   # cgs


def _cuerpo_negro_atmo(path, teff, factor=1.0, dnu_cm=10.0):
    """Un espectro con el formato de ATMO 2020: cabecera `#`, λ (µm) descendente en ν uniforme (Δν = 10 cm⁻¹) y
    F_λ = π B_λ(T) en W m⁻² µm⁻¹ (flujo de superficie: ∫ = σT⁴). Devuelve (λ µm, F en W m⁻² µm⁻¹)."""
    nu = np.arange(50000.0, 5.0, -dnu_cm)                  # cm⁻¹: 0.2–2000 µm, como los de los autores
    lam_cm = 1.0 / nu
    b = 2 * H * C ** 2 / lam_cm ** 5 / np.expm1(H * C / (lam_cm * KB * teff))   # erg s⁻¹ cm⁻² cm⁻¹ sr⁻¹
    f_cgs_A = np.pi * b * 1e-8 * factor                     # erg s⁻¹ cm⁻² Å⁻¹
    lam_um, f_si = lam_cm * 1e4, f_cgs_A / 0.1              # W m⁻² µm⁻¹ = 0.1 erg s⁻¹ cm⁻² Å⁻¹
    with open(path, "w") as fh:
        fh.write("# ATMO emission spectrum\n# Wavelength [microns]  |  Flux [W/m2/micron]\n")
        for x, y in zip(lam_um, f_si):
            fh.write(f"{x:.8f}       {y:.8E}  \n")
    return lam_um, f_si


class LecturaYUnidadesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ca = _carga("convierte_atmo")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.txt = cls.dir / "spec_T2400_lg4.0_CEQ.txt"
        cls.lam_um, cls.f_si = _cuerpo_negro_atmo(cls.txt, 2400.0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_micras_a_angstrom_y_si_a_cgs(self):
        w, f = self.ca.lee_atmo(self.txt)
        d = np.loadtxt(self.txt, comments="#")
        np.testing.assert_array_equal(w, d[:, 0] * 1e4)
        np.testing.assert_array_equal(f, d[:, 1] * 0.1)
        # 1 W m⁻² µm⁻¹ = 10⁷ erg s⁻¹ · 10⁻⁴ cm⁻² · 10⁻⁴ Å⁻¹ = 0.1 erg s⁻¹ cm⁻² Å⁻¹
        self.assertAlmostEqual(1e7 * 1e-4 * 1e-4, 0.1, places=15)
        np.testing.assert_allclose(f, self.f_si * 0.1, rtol=1e-8)   # el fichero guarda 9 cifras

    def test_el_paso_es_10_cm_y_el_nombre_da_el_nodo(self):
        w, _ = self.ca.lee_atmo(self.txt)
        self.assertAlmostEqual(self.ca.paso_nu_cm(w), 10.0, places=4)
        self.assertEqual(self.ca.nodo_de("spec_T2400_lg4.0_CEQ.txt"), (2400.0, 4.0))
        self.assertEqual(self.ca.nodo_de("spec_T1500_lg5.5_CEQ"), (1500.0, 5.5))
        self.assertIsNone(self.ca.nodo_de("spec_T2400_lg4.0_NEQ_weak.txt"))
        self.assertIsNone(self.ca.nodo_de("README"))

    def test_la_resolucion_del_muestreo_a_22_micras_es_455(self):
        # λ/Δλ = 1/(λ Δν): 454.5 a 2.2 µm, la cifra del paper («≈ 450»)
        tb = _carga("teff_bayes")
        self.assertAlmostEqual(1.0 / (2.2e-4 * tb.REJILLAS["atmo_ceq"]["nu_bin_cm"]), 454.545, places=2)


class ConversionTests(unittest.TestCase):
    """`atmo`: los nodos con flujo de superficie se escriben recortados; los que no, no; fuera de T, se ignoran."""

    @classmethod
    def setUpClass(cls):
        cls.ca = _carga("convierte_atmo")
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.txt, cls.npz = base / "txt", base / "npz"
        cls.txt.mkdir()
        _cuerpo_negro_atmo(cls.txt / "spec_T2400_lg4.0_CEQ.txt", 2400.0)
        _cuerpo_negro_atmo(cls.txt / "spec_T2500_lg4.0_CEQ.txt", 2500.0, factor=2.0)   # el doble: no es σT⁴
        _cuerpo_negro_atmo(cls.txt / "spec_T1900_lg4.0_CEQ.txt", 1900.0)               # bajo tmin
        _cuerpo_negro_atmo(cls.txt / "spec_T2400_lg4.0_NEQ_weak.txt", 2400.0)          # otra química
        cls.filas = cls.ca.atmo(cls.txt, cls.npz, tmin=2000.0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_solo_el_nodo_con_flujo_de_superficie_se_escribe(self):
        por_nodo = {(f["teff"], f["logg"]): f for f in self.filas}
        self.assertEqual(set(por_nodo), {(2400.0, 4.0), (2500.0, 4.0)})        # 1900 K y NEQ fuera
        self.assertTrue(por_nodo[(2400.0, 4.0)]["escrito"])
        self.assertAlmostEqual(por_nodo[(2400.0, 4.0)]["flujo_sobre_sigmaT4"], 1.0, delta=0.01)
        self.assertFalse(por_nodo[(2500.0, 4.0)]["escrito"])
        self.assertAlmostEqual(por_nodo[(2500.0, 4.0)]["flujo_sobre_sigmaT4"], 2.0, delta=0.02)
        self.assertEqual(sorted(p.name for p in self.npz.glob("*.npz")), ["teff2400_logg4.0.npz"])
        self.assertEqual(json.loads((self.npz / "conversion.json").read_text()), self.filas)

    def test_el_npz_esta_recortado_y_en_cgs(self):
        from musepipe.models.cache import load_spectrum_npz
        s = load_spectrum_npz(self.npz / "teff2400_logg4.0.npz")
        w, f = self.ca.lee_atmo(self.txt / "spec_T2400_lg4.0_CEQ.txt")
        sel = (w >= self.ca.LO_A) & (w <= self.ca.HI_A)
        orden = np.argsort(w[sel])
        np.testing.assert_array_equal(s.wave_A, w[sel][orden])
        np.testing.assert_array_equal(s.flux, f[sel][orden])
        self.assertGreaterEqual(s.wave_A[0], 4000.0)
        self.assertLessEqual(s.wave_A[-1], 25500.0)
        self.assertTrue(np.all(np.diff(s.wave_A) > 0))
        self.assertEqual((s.meta["teff_k"], s.meta["logg"]), (2400.0, 4.0))
        self.assertEqual(s.meta["flux_unit"], "erg/cm2/s/A")

    def test_lo_lee_el_lector_de_rejillas_de_teff_bayes(self):
        aj = _carga("ajuste_conjunto")
        (m,) = aj.modelos_btsettl(self.npz, teff=(2000.0, 3000.0), logg=(2.5, 5.5))
        self.assertEqual((m["teff"], m["logg"]), (2400.0, 4.0))
        self.assertLess(m["vis"][0][-1], 10500.0)
        self.assertGreater(m["nir"][0][0], 9500.0)

    def test_sin_nodos_en_el_rango_falla(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
            self.ca.atmo(self.txt, Path(d) / "npz", tmin=2600.0, tmax=3000.0)


class ControlDelMuestreoTests(unittest.TestCase):
    """`btbin`: un modelo cualquiera promediado en las bandas de 10 cm⁻¹ de ATMO, en sus centros."""

    def test_una_recta_promediada_en_cada_banda_da_su_valor_en_el_centro_de_la_banda(self):
        ca = _carga("convierte_atmo")
        from musepipe.models.cache import load_spectrum_npz, write_spectrum_npz
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "bt").mkdir()
            # a 0.1 Å, la media por intervalos de la recta es exacta a < 1e-7 (los extremos fraccionarios de
            # `media_en_intervalos` son de segundo orden en el paso): una banda corrida 0.1 Å se notaría
            w = np.arange(3800.0, 26000.0, 0.1)
            write_spectrum_npz(d / "bt" / "teff2400_logg4.0.npz", w, 1.0 + 1e-4 * w,
                               {"teff_k": 2400.0, "logg": 4.0, "grid": "BT-Settl"}, wave_lo=3800.0, wave_hi=26000.0)
            txt = d / "spec_T2400_lg4.0_CEQ.txt"
            _cuerpo_negro_atmo(txt, 2400.0)
            self.assertEqual(ca.btbin(d / "bt", d / "bin", txt), 1)
            s = load_spectrum_npz(d / "bin" / "teff2400_logg4.0.npz")
            wc, lo, hi = ca.bandas_atmo(txt)
            np.testing.assert_allclose(s.wave_A, np.sort(wc))
            orden = np.argsort(wc)
            np.testing.assert_allclose(s.flux, (1.0 + 1e-4 * 0.5 * (lo + hi))[orden], rtol=1e-7)
            np.testing.assert_allclose(hi - lo, wc ** 2 * 10.0 * 1e-8, rtol=1e-3)   # Δλ = λ² Δν
            self.assertIn("muestreo de ATMO", s.meta["grid"])


class DegradadoNativoTests(unittest.TestCase):
    """`degrada_nativa`: la gaussiana de R = 500 menos, en cuadratura, la resolución que trae el muestreo."""

    @classmethod
    def setUpClass(cls):
        cls.tb = _carga("teff_bayes")
        cls.tp = _carga("tipo_ir_plantillas")
        cls.w = np.arange(6000.0, 9500.0, 1.0)
        rng = np.random.default_rng(3)
        cls.f = 1.0 + 0.2 * np.sin(cls.w / 7.0) + 0.05 * rng.normal(size=cls.w.size)

    def test_sin_muestreo_es_el_degradado_de_siempre(self):
        np.testing.assert_allclose(self.tb.degrada_nativa(self.w, self.f, 10.0, c_var=0.0),
                                   self.tp.degrada(self.w, self.f), rtol=1e-12, atol=0)

    def test_si_el_muestreo_ya_es_mas_basto_no_convoluciona(self):
        # con Δν = 1000 cm⁻¹, c·W² supera σ_500² en todo el rango: el modelo queda tal cual
        np.testing.assert_array_equal(self.tb.degrada_nativa(self.w, self.f, 1000.0), self.f)

    def test_resta_en_cuadratura(self):
        # una delta degradada con σ² = σ_500² − c W² tiene esa varianza (lejos de los bordes)
        w = np.arange(7000.0, 8000.0, 0.25)
        f = np.zeros(w.size)
        i0 = w.size // 2
        f[i0] = 1.0
        out = self.tb.degrada_nativa(w, f, 10.0)
        x0 = w[i0]
        s500 = x0 / self.tp.R_OBJ / 2.3548200450309493
        W = x0 ** 2 * 10.0 * 1e-8
        var = np.sum(out * (w - x0) ** 2) / np.sum(out)
        self.assertAlmostEqual(var, s500 ** 2 - self.tb.C_VAR_MUESTREO * W ** 2, delta=0.02 * s500 ** 2)
        self.assertLess(var, s500 ** 2)

    def test_los_huecos_vuelven_a_nan(self):
        f = self.f.copy()
        f[1000:1100] = np.nan
        out = self.tb.degrada_nativa(self.w, f, 10.0)
        self.assertTrue(np.all(np.isnan(out[1020:1080])))
        self.assertTrue(np.all(np.isfinite(out[:900])))

    def test_las_rejillas_atmo_y_las_demas(self):
        R = self.tb.REJILLAS
        self.assertEqual((R["atmo_ceq"]["teff"], R["atmo_ceq"]["logg"]), ((2000.0, 100.0, 11), (2.5, 0.5, 7)))
        self.assertEqual(R["atmo_ceq1500"]["teff"], (1500.0, 100.0, 16))
        self.assertEqual(R["btsettl_bin"]["teff"], R["btsettl"]["teff"])
        self.assertEqual(R["btsettl_bin"]["logg"], R["btsettl"]["logg"])
        for k in ("atmo_ceq", "atmo_ceq1500", "btsettl_bin"):
            self.assertEqual(R[k]["nu_bin_cm"], 10.0)
        # el degradado nuevo solo se activa con `nu_bin_cm`: ninguna de las rejillas de antes lo lleva
        antes = [k for k in R if k not in ("atmo_ceq", "atmo_ceq1500", "btsettl_bin")]
        self.assertIn("btsettl", antes)
        self.assertTrue(all("nu_bin_cm" not in R[k] for k in antes))
        self.assertEqual(self.tb.DEGRADA, ("cuadratura", "ingenua"))


@pytest.mark.external_data
@unittest.skipUnless(ATMO_TXT.is_dir(), "sin la rejilla ATMO 2020 de /mnt/2TB")
class RejillaRealTests(unittest.TestCase):
    """Los nodos de 2400 K de la rejilla real: flujo de superficie, 10 cm⁻¹ y el recorte de las demás rejillas."""

    def test_los_siete_nodos_de_2400_k(self):
        ca = _carga("convierte_atmo")
        with tempfile.TemporaryDirectory() as d:
            filas = ca.atmo(ATMO_TXT, Path(d), tmin=2400.0, tmax=2400.0)
            self.assertEqual([f["logg"] for f in filas], [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
            for f in filas:
                self.assertTrue(f["escrito"], f)
                self.assertAlmostEqual(f["flujo_sobre_sigmaT4"], 1.0, delta=0.02)
                self.assertAlmostEqual(f["dnu_cm"], 10.0, places=3)
                self.assertEqual(f["n"], 5000)
            aj = _carga("ajuste_conjunto")
            pls = aj.modelos_btsettl(d, teff=(2400.0, 2400.0), logg=(2.5, 5.5))
            self.assertEqual(len(pls), 7)
            w, fl = ca.lee_atmo(ATMO_TXT / "spec_T2400_lg4.0_CEQ.txt")
            d0 = np.loadtxt(ATMO_TXT / "spec_T2400_lg4.0_CEQ.txt", comments="#")
            np.testing.assert_array_equal(w, d0[:, 0] * 1e4)
            np.testing.assert_array_equal(fl, d0[:, 1] * 0.1)


if __name__ == "__main__":
    unittest.main()
