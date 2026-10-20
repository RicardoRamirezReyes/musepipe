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

    def test_corner_de_los_datos_sin_r_k_y_con_su_leyenda(self):
        """Fig. teff_corner_data: tres conjuntos de datos, cinco parámetros (sin r_K) y su propia leyenda."""
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(1)
        series = {k: rng.normal([2700 + 50 * i, 4.3, 2.1, 3.4, 0.17], [20, 0.08, 0.05, 0.2, 0.05], (3000, 5))
                  for i, k in enumerate(self.pf.ESTILO_DATOS)}
        self.assertNotIn("rk", [p for p, _ in self.pf.PARAMETROS_CORNER_DATOS])
        fig = plt.figure()
        self.pf._dibuja_corner(fig, series, "X", parametros=self.pf.PARAMETROS_CORNER_DATOS,
                               estilos=self.pf.ESTILO_DATOS, nombres=self.pf.NOMBRE_DATOS)
        n = len(self.pf.PARAMETROS_CORNER_DATOS)
        self.assertEqual(len(fig.axes), n * n)
        self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()],
                         ["MUSE + near-IR", "Near-IR only", "MUSE only"])
        plt.close(fig)


class SeriesDeLosDatosTests(unittest.TestCase):
    """`_series_corner_datos` sobre cadenas sintéticas en disco: qué cadenas elige y en qué orden."""

    @classmethod
    def setUpClass(cls):
        cls.pf = _modulo()

    def _cadena(self, d, nombre, datos, **extra):
        import json

        nombres = ["teff", "logg", "r_rj", "d_pc", "av"] + ([] if datos == "muse" else ["rk"])
        n = 50
        muestras = np.column_stack([np.full(n, 2600.0 + len(datos)), np.full(n, 4.0), np.full(n, 1.8),
                                    np.full(n, 140.0), np.full(n, 3.0)] + ([] if datos == "muse" else [np.zeros(n)]))
        np.savez(d / f"{nombre}.npz", muestras=muestras, derivadas=np.zeros((n, 5)), nombres=np.array(nombres))
        meta = dict(etiqueta=nombre, slug="X", modo="libre", av="primaria", av_fija=None, rejilla="btsettl",
                    datos=datos, r_analisis=500, verdad=None)
        meta.update(extra)
        (d / f"{nombre}.json").write_text(json.dumps(meta))

    def test_elige_las_tres_con_prior_de_bt_settl_en_orden(self):
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self._cadena(d, "a_muse", "muse")
            self._cadena(d, "b_todo", "todo")
            self._cadena(d, "c_ir", "ir")
            # no entran: A_V fija, otra rejilla, una plantilla y R = 250
            self._cadena(d, "d_fija", "ir", av_fija=1.8)
            self._cadena(d, "e_drift", "todo", rejilla="drift")
            self._cadena(d, "f_pl", "todo", verdad={"plantilla": "TWA26"})
            self._cadena(d, "g_r250", "todo", r_analisis=250)
            with mock.patch("scripts.teff_bayes.entrada_margen", return_value={}), \
                    mock.patch("scripts.teff_bayes.margen_de_corrida", side_effect=lambda c, e: np.full(50, 0.1)):
                series = self.pf._series_corner_datos([d], "X")
                self.assertEqual(list(series), ["todo", "ir", "muse"])
                for x in series.values():
                    self.assertEqual(x.shape, (50, len(self.pf.PARAMETROS_CORNER_DATOS)))
                self.assertEqual([x[0, 0] for x in series.values()], [2604.0, 2602.0, 2604.0])
                # si falta un conjunto, falla en vez de dibujar dos
                (d / "a_muse.json").unlink()
                with self.assertRaises(SystemExit):
                    self.pf._series_corner_datos([d], "X")


if __name__ == "__main__":
    unittest.main()
