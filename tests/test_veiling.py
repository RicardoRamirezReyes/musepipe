"""Veiling óptico y en K (`scripts/veiling.py`).

Da las cifras del veiling del apéndice B y de la sección de sistemáticos del paper (docs/2026-10-13_veiling.md).
Las primeras clases no leen nada de disco: la forma del exceso, que el ajuste recupera un veiling inyectado, que
la profundidad del CO se diluye como 1/(1 + r_K) y que el veiling esperado escala con L_acc/L_bol.
`CifrasDelPaperTests` lee los runs, las bibliotecas de G3 y el IR de /mnt/2TB (marca `slow`); si falta algo se
salta, no falla.
"""
import importlib.util
import sys
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


class FormaTests(unittest.TestCase):
    def test_las_formas_valen_uno_en_lambda_ref(self):
        v = _carga("veiling")
        segs = [dict(brazo="vis", w=np.array([7000.0, 7500.0, 8000.0])), dict(brazo="nir", w=np.array([2.0e4, 2.2e4]))]
        for par, f in v.formas(segs, "vis").items():
            self.assertAlmostEqual(f[0][1], 1.0, places=12)
            self.assertTrue(np.all(f[1] == 0))
        for par, f in v.formas(segs, "nir").items():
            self.assertAlmostEqual(f[1][1], 1.0, places=12)


class AjusteVeilingTests(unittest.TestCase):
    def test_recupera_el_veiling_inyectado(self):
        v = _carga("veiling")
        tp = _carga("tipo_ir_plantillas")
        w = np.linspace(6300.0, 9300.0, 200)
        m = (w / 7500) ** -3 * (1 - 0.3 * np.exp(-0.5 * ((w - 7100) / 80) ** 2)) * (1 - 0.2 * np.exp(-0.5 * ((w - 8500) / 80) ** 2))
        b = tp.bordes(w)
        e = 10 ** (-0.4 * tp.EXT.a_lambda_over_av(w) * 1.0)      # el dato, detrás de A_V = 1 como el modelo
        seg = dict(nombre="V", brazo="vis", w=w, lo=b[:-1], hi=b[1:], f=m * e, e=0.001 * m * e, bueno=np.ones(w.size, bool))
        datos = v.con_veiling([seg], [m], 1.0, "vis", 0.3, ("alfa", 0.0))
        datos[0]["f"] *= 2.5
        res, _, _ = v.ajusta_veiling(datos, {0: [m]}, [1.0], "vis")
        self.assertAlmostEqual(res[0]["r"], 0.3, delta=0.02)
        self.assertEqual(res[0]["par"], ["alfa", 0.0])


class CoTests(unittest.TestCase):
    def test_la_profundidad_se_diluye_con_el_veiling(self):
        v = _carga("veiling")
        w = np.linspace(22000.0, 23600.0, 3000)
        f = np.where(w > 22935.0, 0.88, 1.0)
        d0 = v.profundidad_co(w, f)
        d1 = v.profundidad_co(w, f + 0.4)                  # exceso plano de 0.4 (r_K = 0.4)
        self.assertAlmostEqual(d0 / d1 - 1, 0.4, places=6)
        self.assertAlmostEqual(d0, 0.12, delta=0.005)


class EsperadoTests(unittest.TestCase):
    def test_escala_con_lacc_sobre_lbol(self):
        v = _carga("veiling")
        w = np.linspace(5300.0, 24800.0, 20000)
        t = dict(vis=(w[w < 10000], np.ones((w < 10000).sum())), nir=(w[w >= 10000], np.ones((w >= 10000).sum())))
        r1 = v.r_esperado(1e-7, 1e-3, t, 7500.0, 8000.0)
        self.assertAlmostEqual(v.r_esperado(2e-7, 1e-3, t, 7500.0, 8000.0) / r1, 2.0, places=10)
        self.assertAlmostEqual(v.r_esperado(1e-7, 2e-3, t, 7500.0, 8000.0) / r1, 0.5, places=10)


def _hay_datos():
    return (_carga("tipo_ir_indices").DEFAULT_CDS.is_dir() and _carga("tipo_ir_plantillas").DEFAULT_NIR.is_dir()
            and (ROOT / "runs" / "ROXs12b_invvar" / "stages").is_dir()
            and (ROOT / "runs" / "ROXs42Bb_realigned" / "stages").is_dir())


@pytest.mark.slow
@unittest.skipUnless(_hay_datos(), "sin catálogo CDS, sin el NIR de las plantillas o sin los runs del paper")
class CifrasDelPaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = _carga("veiling")

    def test_co(self):
        r12 = self.v.r_k_co("ROXs12b")["espectros"]["OSIRIS_K"]
        self.assertAlmostEqual(r12["D_obs"], 0.070, delta=0.001)
        self.assertAlmostEqual(r12["r_K"], 0.44, delta=0.01)
        e42 = self.v.r_k_co("ROXs42Bb")["espectros"]
        self.assertAlmostEqual(e42["OSIRIS_K"]["r_K"], 0.19, delta=0.01)
        self.assertAlmostEqual(e42["SINFONI_K"]["r_K"], 0.37, delta=0.01)

    def test_optico_sin_veiling_y_su_suelo(self):
        o = self.v.veiling_ajuste("ROXs12b", "muse", av_fija=True, calibrar=True)
        self.assertEqual(o["mejor"]["r"], 0.0)
        self.assertAlmostEqual(o["calibracion"]["p95"], 0.085, delta=0.003)
