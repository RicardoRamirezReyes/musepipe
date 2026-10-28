"""La fotometría de caja con anillo de `scripts/banda_de_filas.py` (el azul de ROXs 42B b es una banda de filas).

Las clases sintéticas no tocan `runs/`: un cubo con una banda a lo largo de unas filas y una fuente
puntual solo en el rojo, que es el caso que la herramienta tiene que distinguir. `CifrasDelPaperTests`
lee el cubo de C1b de los dos runs del paper (solo lectura, ~2 s) y comprueba las cifras de la decisión A
(cabecera de procedencia de `paper/main.tex`) bit a bit contra las de la sesión del 2026-10-25. Busca
`runs/` en este clon o en `MUSE_PROJECT_ROOT` (un worktree no lleva `runs/`); si faltan, se salta.
"""
import importlib.util
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _raiz_con_runs(*runs):
    for raiz in (ROOT, os.environ.get("MUSE_PROJECT_ROOT")):
        if raiz and all((Path(raiz) / "runs" / r / "config" / "config.json").exists() for r in runs):
            return Path(raiz)
    return None


SHAPE = (100, 120)
COMP_YX = (50.3, 70.6)


def _cubo_sintetico():
    """Banda de +2 por canal en las filas 48-52 en el azul; fuente puntual en el rojo; NaN y basura en el hueco."""
    wave = np.arange(4800.0, 9300.0, 25.0)
    nz = wave.size
    cubo = np.zeros((nz,) + SHAPE, dtype=np.float32)
    azul = wave < 5500.0
    cubo[np.ix_(azul, np.arange(48, 53), np.arange(SHAPE[1]))] = 2.0
    yy, xx = np.indices(SHAPE, dtype=np.float64)
    psf = np.exp(-0.5 * ((yy - COMP_YX[0]) ** 2 + (xx - COMP_YX[1]) ** 2) / 1.2 ** 2)
    rojo = wave >= 8000.0
    cubo[rojo] += (10.0 * psf).astype(np.float32)
    cubo[(wave > 5780.0) & (wave < 6050.0)] = 1.0e6  # el hueco del láser: no debe entrar en nada
    cubo[3, 10, 10] = np.nan
    return cubo, wave


class FotometriaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("banda_de_filas")

    def test_el_colapso_quita_el_hueco_del_laser_y_promedia_los_finitos(self):
        cubo, wave = _cubo_sintetico()
        idx = self.m.canales_de_banda(wave, 5700.0, 6200.0)
        self.assertFalse(np.any((wave[idx] > 5780.0) & (wave[idx] < 6050.0)))
        img = self.m.colapsa_banda(cubo, wave, 5700.0, 6200.0, bloque=3)
        self.assertLess(float(np.max(np.abs(img))), 1.0)  # ningún canal del hueco
        azul = self.m.colapsa_banda(cubo, wave, 4800.0, 5500.0, bloque=4)
        self.assertEqual(azul[50, 5], 2.0)
        self.assertEqual(azul[10, 10], 0.0)  # el NaN no cuenta: media de los finitos
        self.assertEqual(azul[20, 5], 0.0)

    def test_la_caja_menos_el_anillo(self):
        img = np.zeros(SHAPE)
        img[48:53, :] = 2.0  # una banda de 5 filas
        # en la banda: la caja entera vale 2 y el anillo de 8-14 px casi no la toca (mediana 0)
        self.assertEqual(self.m.fotometria_caja_anillo(img, 50.0, 60.0), 50.0)
        # un fondo constante se va entero
        self.assertAlmostEqual(self.m.fotometria_caja_anillo(img + 7.5, 50.0, 60.0), 50.0, places=10)
        # la caja se centra en el spáxel más cercano; el anillo, en la posición
        self.assertEqual(self.m.fotometria_caja_anillo(img, 49.6, 60.4), 50.0)
        with self.assertRaises(ValueError):
            self.m.fotometria_caja_anillo(img, 50.0, 60.0, caja=4)
        with self.assertRaises(ValueError):
            self.m.fotometria_caja_anillo(img, 1.0, 60.0)

    def test_la_exclusion_de_la_primaria_solo_quita_pixeles_del_anillo(self):
        rng = np.random.default_rng(0)
        img = rng.normal(size=SHAPE)
        sin = self.m.fotometria_caja_anillo(img, 50.0, 60.0)
        lejos = self.m.fotometria_caja_anillo(img, 50.0, 60.0, excluye_yx=(50.0, 0.0), excluye_px=30.0)
        self.assertEqual(sin, lejos)  # la primaria a 60 px: el anillo (<= 14 px) no se entera
        cerca = self.m.fotometria_caja_anillo(img, 50.0, 60.0, excluye_yx=(50.0, 40.0), excluye_px=30.0)
        self.assertNotEqual(sin, cerca)

    def test_las_posiciones_de_la_sesion(self):
        misma = self.m.posiciones_misma_fila(COMP_YX, SHAPE)
        otras = self.m.posiciones_otras_filas(COMP_YX, SHAPE)
        self.assertEqual([x - COMP_YX[1] for _y, x in misma], [-18, -14, -10, 10, 14, 18])
        self.assertTrue(all(y == COMP_YX[0] for y, _x in misma))
        self.assertEqual(len(otras), 12)
        self.assertEqual(sorted({round(y - COMP_YX[0]) for y, _x in otras}), [-10, -7, 7, 10])
        # el margen de 16 px descarta las que no caben
        cerca_del_borde = self.m.posiciones_misma_fila((50.0, 30.0), SHAPE)
        self.assertEqual([x - 30.0 for _y, x in cerca_del_borde], [-14, -10, 10, 14, 18])

    def test_una_banda_de_filas_da_el_mismo_flujo_en_su_fila_con_o_sin_companero(self):
        cubo, wave = _cubo_sintetico()
        azul = self.m.mide_banda(self.m.colapsa_banda(cubo, wave, 4900.0, 5400.0), COMP_YX)
        self.assertAlmostEqual(azul["companero"], 50.0, places=9)
        self.assertAlmostEqual(azul["misma_fila"]["media_pct"], 100.0, places=9)
        self.assertAlmostEqual(azul["misma_fila"]["sd"], 0.0, places=9)
        self.assertEqual(azul["otras_filas"]["n"], 12)
        self.assertAlmostEqual(azul["otras_filas"]["media"], 0.0, places=9)

        rojo = self.m.mide_banda(self.m.colapsa_banda(cubo, wave, 8000.0, 9300.0), COMP_YX)
        self.assertGreater(rojo["companero"], 50.0)  # la fuente puntual: 10·2π·1.2² ≈ 90 en total
        self.assertLess(abs(rojo["misma_fila"]["media_pct"]), 1.0)
        self.assertLess(abs(rojo["otras_filas"]["media_pct"]), 1.0)

    def test_la_dispersion_es_entre_posiciones_con_ddof_1(self):
        st = self.m._estadistica([1.0, 2.0, 3.0, 4.0], 10.0)
        self.assertEqual(st["media"], 2.5)
        self.assertAlmostEqual(st["sd"], math.sqrt(5.0 / 3.0), places=12)
        self.assertAlmostEqual(st["media_pct"], 25.0, places=12)


RAIZ = _raiz_con_runs("ROXs42Bb_realigned", "ROXs12b_invvar")


@unittest.skipUnless(RAIZ is not None, "sin los runs del paper (ni aquí ni en MUSE_PROJECT_ROOT)")
class CifrasDelPaperTests(unittest.TestCase):
    """Decisión A: las cifras del pie de la Fig. full, la Sect. 4.2 y la Sect. 5.3."""

    @classmethod
    def setUpClass(cls):
        m = _carga("banda_de_filas")
        cubo = Path(RAIZ) / "runs" / "ROXs42Bb_realigned" / "stages" / "cube_psfsub_perobs.fits"
        if not cubo.exists():
            raise unittest.SkipTest("sin el cubo de C1b de ROXs42Bb_realigned")
        cls.b = m.lee_run("ROXs42Bb_realigned", RAIZ)["bandas"]
        cls.b12 = m.lee_run("ROXs12b_invvar", RAIZ)["bandas"]

    def test_azul_la_misma_fila_da_el_flujo_del_companero(self):
        b = self.b["B1 4900-5400"]
        # la sesión (apertura_en_fila.py), a precisión completa
        np.testing.assert_allclose([b["companero"], b["misma_fila"]["media"], b["misma_fila"]["sd"],
                                    b["otras_filas"]["media"], b["otras_filas"]["sd"]],
                                   [29.82629426364744, 29.875439697107538, 6.307990050051971,
                                    -1.3824285556587899, 12.633605129195262], rtol=1e-12)
        self.assertEqual((b["misma_fila"]["n"], b["otras_filas"]["n"]), (6, 12))
        # lo que imprime el paper: 100 ± 21 % y −5 ± 42 %
        self.assertEqual((round(b["misma_fila"]["media_pct"]), round(b["misma_fila"]["sd_pct"])), (100, 21))
        self.assertEqual((round(b["otras_filas"]["media_pct"]), round(b["otras_filas"]["sd_pct"])), (-5, 42))

    def test_rojo_la_misma_fila_no_tiene_flujo(self):
        b = self.b["rojo 8000-9300"]
        np.testing.assert_allclose([b["companero"], b["misma_fila"]["media"], b["misma_fila"]["sd"]],
                                   [422.10203978167135, 3.095244726862589, 28.561614264655894], rtol=1e-12)
        self.assertEqual((round(b["misma_fila"]["media_pct"]), round(b["misma_fila"]["sd_pct"])), (1, 7))

    def test_en_12b_no_hay_banda(self):
        b = self.b12["B1 4900-5400"]
        self.assertEqual((round(b["misma_fila"]["media"], 1), round(b["misma_fila"]["sd"], 1)), (-0.6, 2.5))


if __name__ == "__main__":
    unittest.main()
