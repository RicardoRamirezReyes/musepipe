"""El corner plot de las cadenas de la T_eff (`teff_corner` de scripts/paper_figures.py), con muestras sintéticas.

Las cadenas de verdad viven fuera de runs/ (`teff_bayes.py ajusta`), así que aquí se prueba lo que la figura añade
a `corner`: un rango común a las dos rejillas y unos histogramas que no se recortan entre sí.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _modulo():
    spec = importlib.util.spec_from_file_location("paper_figures", ROOT / "scripts" / "paper_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(importlib.util.find_spec("corner"), "corner no está instalado (environment.yml lo fija)")
class CornerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pf = _modulo()
        rng = np.random.default_rng(0)
        # BT-Settl más estrecha y con más muestras que DRIFT, como en ROXs 42B b: su pico es el más alto
        cls.series = {
            "btsettl": rng.normal([2700, 4.3, 2.1, 3.4, 0.5, 0.17], [20, 0.08, 0.05, 0.2, 0.08, 0.05], (4000, 6)),
            "drift": rng.normal([2850, 5.2, 1.7, 3.8, 0.9, 0.25], [40, 0.15, 0.08, 0.3, 0.1, 0.08], (2000, 6)),
        }

    def test_el_rango_cubre_las_dos_rejillas(self):
        rangos = self.pf._rangos_corner(self.series)
        self.assertEqual(len(rangos), len(self.pf.PARAMETROS_CORNER))
        for j, (lo, hi) in enumerate(rangos):
            for x in self.series.values():
                self.assertLessEqual(lo, np.quantile(x[:, j], 0.01))
                self.assertGreaterEqual(hi, np.quantile(x[:, j], 0.99))

    def test_ningun_histograma_queda_recortado(self):
        """`corner` fija el alto de cada histograma con la última serie: la primera, más alta, se cortaba."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        self.pf._dibuja_corner(fig, self.series, "X")
        n = len(self.pf.PARAMETROS_CORNER)
        self.assertEqual(len(fig.axes), n * n)
        for i in range(n):
            ax = fig.axes[i * (n + 1)]
            self.assertEqual(len(ax.patches), len(self.series))
            alto = max(float(pt.get_path().vertices[:, 1].max()) for pt in ax.patches)
            self.assertGreaterEqual(ax.get_ylim()[1], alto)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
