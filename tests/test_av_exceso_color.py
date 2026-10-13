"""A_V propia por exceso de color (`scripts/av_exceso_color.py`).

Da las cifras de la extinción de la sección de sistemáticos y del apéndice B del paper
(docs/2026-10-13_av_exceso_color.md). Las primeras clases no leen nada de disco: la media en ventanas, que
la A_V se recupera exactamente de un color fabricado, y que solo entran las bandas que declara
`targets/<slug>.json` -> `ir_spectra.colour_excess`, con K reescalada. `CifrasDelPaperTests` lee los runs,
las bibliotecas de G3 y el IR de /mnt/2TB (marca `slow`); si falta algo se salta, no falla.
"""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _carga(nombre):
    spec = importlib.util.spec_from_file_location(nombre, ROOT / "scripts" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


class MediaYAvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("av_exceso_color")
        cls.w = np.linspace(8000.0, 24000.0, 16001)
        cls.f = (cls.w / 1e4) ** -2.0

    def test_una_constante_da_la_constante(self):
        self.assertAlmostEqual(self.m.media(self.w, np.full(self.w.size, 2.5), self.m.VENTANAS["J"]), 2.5, places=12)

    def test_enrojecer_baja_el_azul_frente_al_rojo(self):
        v = self.m.VENTANAS
        r0 = self.m.media(self.w, self.f, v["J"]) / self.m.media(self.w, self.f, v["K"])
        r2 = self.m.media(self.w, self.f, v["J"], 2.0) / self.m.media(self.w, self.f, v["K"], 2.0)
        # E(J−K) ≈ 0.17 A_V con CCM89: el cociente cae ~10^(-0.4·0.34)
        self.assertAlmostEqual(-2.5 * np.log10(r2 / r0) / 2.0, 0.17, delta=0.01)

    def test_recupera_la_av_inyectada(self):
        v = self.m.VENTANAS
        for a, b in (("J", "K"), ("H", "K"), ("MUSE", "K")):
            r = self.m.media(self.w, self.f, v[a], 2.3) / self.m.media(self.w, self.f, v[b], 2.3)
            self.assertAlmostEqual(self.m.av_que_iguala(self.w, self.f, v[a], self.w, self.f, v[b], r), 2.3, places=6)

    def test_sin_raiz_es_nan(self):
        v = self.m.VENTANAS
        self.assertTrue(np.isnan(self.m.av_que_iguala(self.w, self.f, v["J"], self.w, self.f, v["K"], 1e6)))


class BandasDeclaradasTests(unittest.TestCase):
    def test_solo_las_bandas_declaradas_y_k_reescalada(self):
        m = _carga("av_exceso_color")
        w_um = np.linspace(1.1, 2.45, 2000)
        perfil = {"ir_spectra": {
            "spectra": {"A_J": {"file": "j", "band": "J"}, "A_K": {"file": "k", "band": "K"}},
            "colour_excess": {"bands": ["K"], "k_rescale_mag": {"catalogue": 14.32, "adopted": 14.14}}}}
        with mock.patch.object(m.tir, "perfil", return_value=perfil), \
                mock.patch.object(m.tir, "load", return_value=(w_um, np.ones(w_um.size), np.ones(w_um.size))):
            out = m.bandas_del_companero("x")
        self.assertEqual(list(out), ["K"])
        self.assertAlmostEqual(out["K"][0][1], 0.1 * 10 ** (-0.4 * (14.14 - 14.32)), places=12)

    def test_r_k_divide_el_flujo_en_k(self):
        m = _carga("av_exceso_color")
        w_um = np.linspace(1.1, 2.45, 2000)
        perfil = {"ir_spectra": {"spectra": {"A_K": {"file": "k", "band": "K"}, "A_H": {"file": "h", "band": "H"}},
                                 "colour_excess": {"bands": ["H", "K"]}}}
        with mock.patch.object(m.tir, "perfil", return_value=perfil), \
                mock.patch.object(m.tir, "load", return_value=(w_um, np.ones(w_um.size), np.ones(w_um.size))):
            out = m.bandas_del_companero("x", r_k=0.25)
        self.assertAlmostEqual(out["K"][0][1], 0.1 / 1.25, places=12)
        self.assertAlmostEqual(out["H"][0][1], 0.1, places=12)

    def test_sin_declaracion_para(self):
        m = _carga("av_exceso_color")
        with mock.patch.object(m.tir, "perfil", return_value={"ir_spectra": {"spectra": {}}}):
            with self.assertRaises(SystemExit):
                m.bandas_del_companero("x")


def _hay_datos():
    return (_carga("tipo_ir_indices").DEFAULT_CDS.is_dir() and _carga("tipo_ir_plantillas").DEFAULT_NIR.is_dir()
            and (ROOT / "runs" / "ROXs12b_invvar" / "stages").is_dir()
            and (ROOT / "runs" / "ROXs42Bb_realigned" / "stages").is_dir())


@pytest.mark.slow
@unittest.skipUnless(_hay_datos(), "sin catálogo CDS, sin el NIR de las plantillas o sin los runs del paper")
class CifrasDelPaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("av_exceso_color")

    def test_42bb_con_el_tipo_de_los_indices(self):
        e = self.m.estimacion("ROXs42Bb")
        r = e["resumen"]
        self.assertEqual(e["tipo"], "M9.9")
        self.assertAlmostEqual(r["J-K"]["mediana"], 3.81, delta=0.01)
        self.assertAlmostEqual(r["J-H"]["mediana"], 3.42, delta=0.01)
        self.assertAlmostEqual(r["H-K"]["mediana"], 4.63, delta=0.01)
        self.assertAlmostEqual(r["MUSE-K psffit"]["mediana"], 3.33, delta=0.01)
        self.assertAlmostEqual(e["cota_superior"], 5.17, delta=0.01)

    def test_con_k_corregida_por_veiling(self):
        e = self.m.estimacion("ROXs42Bb", r_k=0.19)
        self.assertAlmostEqual(e["resumen"]["J-K"]["mediana"], 2.68, delta=0.01)
        self.assertAlmostEqual(e["resumen"]["H-K"]["mediana"], 1.84, delta=0.01)
        e12 = self.m.estimacion("ROXs12b", r_k=0.44)
        self.assertAlmostEqual(e12["resumen"]["MUSE-K psffit"]["mediana"], 4.20, delta=0.01)

    def test_12b_con_el_tipo_de_los_indices_y_con_m9(self):
        e = self.m.estimacion("ROXs12b")
        self.assertEqual(e["tipo"], "M7.5")
        self.assertEqual(sorted(e["bandas"]), ["K"])
        self.assertAlmostEqual(e["resumen"]["MUSE-K psffit"]["mediana"], 5.20, delta=0.01)
        self.assertAlmostEqual(e["cota_superior"], 5.39, delta=0.01)
        m9 = self.m.estimacion("ROXs12b", tipo="M9")
        self.assertAlmostEqual(m9["resumen"]["MUSE-K psffit"]["mediana"], 2.64, delta=0.01)


if __name__ == "__main__":
    unittest.main()
