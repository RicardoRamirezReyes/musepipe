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
