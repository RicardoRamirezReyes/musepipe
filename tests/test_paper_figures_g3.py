"""Ayudantes de las figuras de G3 en `scripts/paper_figures.py` (sin datos reales)."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _modulo():
    spec = importlib.util.spec_from_file_location("paper_figures", ROOT / "scripts" / "paper_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAyudantesG3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pf = _modulo()

    def test_con_huecos_corta_la_linea_en_lo_excluido(self):
        x = np.array([1.0, 2.0, 3.0, 10.0, 11.0])
        xo, yo = self.pf._con_huecos(x, x * 2)
        self.assertEqual(xo.size, 6)
        self.assertTrue(np.isnan(xo[3]) and np.isnan(yo[3]))
        np.testing.assert_array_equal(xo[~np.isnan(xo)], x)

    def test_tramos_contiguos(self):
        w = np.arange(10.0)
        m = np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 1], bool)
        self.assertEqual(self.pf._tramos(w, m), [(0.5, 2.5), (5.5, 6.5), (8.5, 9.5)])
        self.assertEqual(self.pf._tramos(w, np.zeros(10, bool)), [])

    def test_un_color_y_un_marcador_por_biblioteca(self):
        self.assertEqual(set(self.pf.COLOR_BIBLIOTECA), set(self.pf.MARCA_BIBLIOTECA))
        self.assertEqual(len(set(self.pf.COLOR_BIBLIOTECA.values())), len(self.pf.COLOR_BIBLIOTECA))
        self.assertEqual(len(set(self.pf.MARCA_BIBLIOTECA.values())), len(self.pf.MARCA_BIBLIOTECA))

    def test_figuras_registradas(self):
        self.assertIn("template_comparison", self.pf.FIGURAS)
        self.assertIn("companion_type", self.pf.FIGURAS)


if __name__ == "__main__":
    unittest.main()


class _ObjetoFalso:
    """Lo que `_mdot_de_este_trabajo` lee de un objeto del paper, sin runs/."""

    def __init__(self, veredicto, g3_qc, fila_mdot):
        self.nombre = "falso"
        self.config = {"h03_companion_mass_msun": 0.0167}
        self._qc = {"stage_g3_qc.json": g3_qc, "stage_h01_qc.json": {"verdict": {"verdict": veredicto}},
                    "stage_x11_qc.json": {"canonical_method": "psffit"}}
        self._filas = {"g3_physical_properties.csv": [fila_mdot] if fila_mdot else [],
                       "halpha_upper_limits.csv": [{"row_kind": "method", "method": "psffit",
                                                    "mdot_msun_yr": "3.3e-14"}]}

    def qc(self, nombre):
        return self._qc[nombre]

    def filas(self, nombre):
        return self._filas[nombre]


class TestMdotDeLaFigura(unittest.TestCase):
    """La figura Mdot-masa no puede cambiar de deteccion a limite sin avisar (2026-09-26)."""

    @classmethod
    def setUpClass(cls):
        cls.pf = _modulo()

    def test_deteccion_con_mdot_de_g3_se_dibuja_como_medida(self):
        fila = {"property": "mdot", "label": "empirical_inference", "err_stat_lo": "4e-14",
                "err_stat_hi": "3e-13"}
        r = self.pf._mdot_de_este_trabajo(_ObjetoFalso(
            "detection", {"mdot_p50_msun_yr": 2.2e-13, "mdot_label": "empirical_inference"}, fila))
        self.assertFalse(r["limite"])
        self.assertAlmostEqual(r["mdot"], 2.2e-13)

    def test_mdot_de_una_cota_se_dibuja_con_el_limite_de_e3(self):
        fila = {"property": "mdot", "label": "upper_limit", "err_stat_lo": "", "err_stat_hi": ""}
        r = self.pf._mdot_de_este_trabajo(_ObjetoFalso(
            "non_detection", {"mdot_p50_msun_yr": 1.9e-13, "mdot_label": "upper_limit"}, fila))
        self.assertTrue(r["limite"])
        self.assertAlmostEqual(r["mdot"], 3.3e-14)

    def test_deteccion_sin_mdot_de_g3_no_cae_en_silencio_al_limite(self):
        with self.assertRaises(SystemExit):
            self.pf._mdot_de_este_trabajo(_ObjetoFalso("detection", {}, None))
