"""Verificaciones de la spec E4b (`docs/spec_E4b_codex_perexp_injection.md`)."""
import unittest

import numpy as np

from musepipe.stage_registry import STAGES, by_id
from musepipe.stages import stage_h04b_perexp_injection as H4B


class TestRegistroE4b(unittest.TestCase):
    def test_publicada_y_opcional(self):
        s = by_id("E4b")
        self.assertIsNotNone(s, "E4b no esta en el registry")
        self.assertTrue(s.qc_optional, "E4b tiene que ser opcional: no la tienen todos los runs")
        self.assertEqual(s.qc, "stages/stage_h04b_qc.json")
        self.assertTrue(s.parameterized, "E4b debe aceptar --run-id")

    def test_no_toca_a_e4(self):
        """E3 lee a E4: si E4b se colara en su QC, moveria el limite en silencio."""
        e4 = by_id("E4")
        self.assertEqual(e4.qc, "stages/stage_h04_qc.json")
        self.assertFalse(e4.qc_optional)
        self.assertNotEqual(by_id("E4b").qc, e4.qc)


class TestContratoE4b(unittest.TestCase):
    def test_metodos_declarados(self):
        self.assertEqual(set(H4B.METODOS), {"aperture", "psffit"})

    def test_metodo_desconocido_falla(self):
        with self.assertRaises(H4B.PerExpInjectionError):
            H4B._extrae("no_existe", None, None, None, (0, 0), (0, 0), None, None, None, {})

    def test_sigma_exige_dos_controles(self):
        """V3: sin dos controles no hay sigma empirica, y STAT no vale."""
        wave = np.linspace(6500.0, 6620.0, 60)
        uno = np.zeros((1, wave.size))
        with self.assertRaises(H4B.PerExpInjectionError):
            H4B._sigma_de_la_exposicion(uno, wave, 6562.8, 2.4, np.ones(wave.size), 80.0)

    def test_sigma_es_la_dispersion_de_los_controles(self):
        rng = np.random.default_rng(20260827)
        wave = np.linspace(6500.0, 6620.0, 121)
        err = np.ones(wave.size)
        ctrl = rng.normal(0.0, 1.0, size=(12, wave.size))
        sigma, medidas = H4B._sigma_de_la_exposicion(ctrl, wave, 6562.8, 2.4, err, 80.0)
        self.assertEqual(len(medidas), 12)
        self.assertGreater(sigma, 0.0)
        self.assertTrue(np.isfinite(sigma))

    def test_qc_separa_posiciones_de_filas(self):
        """Confundir posiciones con filas costo dos rondas: van como campos distintos."""
        import inspect
        src = inspect.getsource(H4B.compute_stage_h04b_products)
        for campo in ("positions_per_point", "injections_per_point", "rows_per_point"):
            self.assertIn(campo, src)


if __name__ == "__main__":
    unittest.main()


class TestVentanaEspectral(unittest.TestCase):
    """La ventana no es una optimizacion cosmetica: sin ella la etapa no corre.

    Medido el 2026-08-27: `psffit` cuesta 24 s por ajuste sobre los 3681 canales
    y 1.6 s sobre la banda, o sea 37 h contra 2.5 h para la rejilla real.
    """

    def test_el_default_cubre_la_linea_y_su_continuo(self):
        cfg = H4B.stage_h04b_config_from_run("ROXs12b_realigned")
        lo, hi = cfg["x06b_band_A"]
        linea = float(cfg.get("h04_line_center_A", cfg.get("h01_line_center_A", 6562.8)))
        ventana = float(cfg["x06b_continuum_window_A"])
        self.assertLess(lo, linea - ventana, "la banda no cubre el continuo por el azul")
        self.assertGreater(hi, linea + ventana, "la banda no cubre el continuo por el rojo")

    def test_se_puede_pedir_el_cubo_entero(self):
        cfg = H4B.stage_h04b_config_from_run("ROXs12b_realigned",
                                             overrides={"x06b_band_A": None})
        self.assertIsNone(cfg["x06b_band_A"])

    def test_la_banda_viaja_al_qc(self):
        import inspect
        self.assertIn('"band_A"', inspect.getsource(H4B.compute_stage_h04b_products))
