"""Tipo espectral por ajuste de plantillas (`scripts/tipo_ir_plantillas.py`).

Da las cifras del apéndice B del paper (tipo por plantillas solo con el IR y con MUSE + IR;
docs/2026-10-13_tipo_ir_plantillas.md). Las primeras clases no leen nada de disco: fijan la media por
intervalos, quién comparte escala, que el ajuste recupera la plantilla y la A_V con que se fabricó un dato,
la limpieza de QUAL de los ADP de X-shooter y la S/N de los datos falsos de la calibración.
`CifrasDelPaperTests` lee los runs, las bibliotecas de G3 y el NIR de /mnt/2TB (varios minutos: marca
`slow`); si falta algo se salta, no falla.
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


def _seg(nombre, brazo, w, f, e):
    tp = _carga("tipo_ir_plantillas")
    b = tp.bordes(w)
    return dict(nombre=nombre, brazo=brazo, w=w, lo=b[:-1], hi=b[1:], f=f, e=e, bueno=np.ones(w.size, bool))


class MediaEnIntervalosTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tp = _carga("tipo_ir_plantillas")
        cls.w = np.linspace(10000.0, 12000.0, 2001)

    def test_una_densidad_constante_da_la_constante(self):
        out = self.tp.media_en_intervalos(self.w, np.full(self.w.size, 3.0), np.array([10100.0]), np.array([10150.0]))
        self.assertAlmostEqual(out[0], 3.0, places=12)

    def test_una_recta_da_su_valor_en_el_centro(self):
        # extremos en nodos: exacto; extremos fraccionarios: error de segundo orden en el paso (1 Å aquí)
        f = 2.0 + 1e-3 * self.w
        exacto = self.tp.media_en_intervalos(self.w, f, np.array([10100.0]), np.array([10170.0]))
        self.assertAlmostEqual(exacto[0], 2.0 + 1e-3 * 10135.0, places=10)
        frac = self.tp.media_en_intervalos(self.w, f, np.array([10100.3]), np.array([10170.9]))
        self.assertAlmostEqual(frac[0] / (2.0 + 1e-3 * 0.5 * (10100.3 + 10170.9)), 1.0, delta=1e-6)

    def test_fuera_de_cobertura_o_con_un_hueco_es_nan(self):
        f = np.ones(self.w.size)
        f[500] = np.nan                                   # 10500 Å
        out = self.tp.media_en_intervalos(self.w, f, np.array([9990.0, 10490.0, 10600.0]),
                                          np.array([10010.0, 10510.0, 10620.0]))
        self.assertTrue(np.isnan(out[0]) and np.isnan(out[1]))
        self.assertAlmostEqual(out[2], 1.0, places=12)


class AjusteTests(unittest.TestCase):
    """Dos segmentos IR y uno «MUSE» con plantillas sintéticas de pendientes distintas."""

    @classmethod
    def setUpClass(cls):
        cls.tp = _carga("tipo_ir_plantillas")
        cls.wj = np.linspace(11800.0, 13200.0, 80)
        cls.wk = np.linspace(20900.0, 23500.0, 90)
        cls.wv = np.linspace(7000.0, 9000.0, 100)
        cls.base = {i: [(w / 1e4) ** (-a) for w in (cls.wj, cls.wk, cls.wv)] for i, a in enumerate((1.0, 2.5, 4.0))}

    def _datos(self, i, av, escalas):
        segs = []
        for (nombre, brazo, w), m, s in zip((("J", "nir", self.wj), ("K", "nir", self.wk), ("V", "vis", self.wv)),
                                            self.base[i], escalas):
            f = s * m * 10 ** (-0.4 * self.tp.EXT.a_lambda_over_av(w) * av)
            segs.append(_seg(nombre, brazo, w, f, 0.01 * f))
        return segs

    def test_recupera_la_plantilla_y_la_av(self):
        segs = self._datos(1, 1.2, (1.0, 3.0, 0.5))
        res, n = self.tp.ajusta(segs, self.base, np.round(np.arange(0, 3.01, 0.1), 2))
        mejor = min(res, key=lambda k: res[k]["chi2"])
        self.assertEqual(mejor, 1)
        self.assertAlmostEqual(res[1]["av"], 1.2, places=6)
        self.assertLess(res[1]["chi2"], 1e-12)
        self.assertEqual(n, [80, 90, 100])

    def test_la_escala_comun_ve_los_colores(self):
        segs = self._datos(1, 1.2, (1.0, 3.0, 0.5))           # K tres veces más brillante que J
        libre, _ = self.tp.ajusta(segs, self.base, [1.2])
        comun, _ = self.tp.ajusta(segs, self.base, [1.2], escala_comun_ir=True)
        self.assertLess(libre[1]["chi2"], 1e-12)
        self.assertGreater(comun[1]["chi2"], 1e3)

    def test_quien_comparte_escala(self):
        segs = self._datos(0, 0.0, (1, 1, 1))
        self.assertEqual(self.tp.grupos(segs), [[0], [1], [2]])
        self.assertEqual(self.tp.grupos(segs, escala_comun_ir=True), [[0, 1], [2]])

    def test_una_escala_negativa_no_vale(self):
        segs = self._datos(0, 0.0, (1, 1, 1))
        base = {**self.base, 9: [-m for m in self.base[0]]}
        res, _ = self.tp.ajusta(segs, base, [0.0])
        self.assertEqual(res[9]["chi2"], np.inf)

    def test_el_equilibrio_iguala_el_chi2nu_de_cada_segmento(self):
        rng = np.random.default_rng(1)
        segs = self._datos(1, 0.5, (1, 1, 1))
        for s, k in zip(segs, (1.0, 4.0, 0.5)):                # σ declarada k veces la real
            s["f"] = s["f"] + rng.normal(0, s["e"])
            s["e"] = s["e"] * k
        fac = self.tp.factores_equilibrio(segs, self.base, [0.5])
        np.testing.assert_allclose(fac, (1.0, 0.25, 2.0), rtol=0.25)


class EstructuraTests(unittest.TestCase):
    """El ajuste solo de estructuras: plantilla × polinomio de Legendre por segmento."""

    @classmethod
    def setUpClass(cls):
        cls.tp = _carga("tipo_ir_plantillas")
        cls.w = np.linspace(6300.0, 9300.0, 200)
        banda = lambda c, a: 1 - a * np.exp(-0.5 * ((cls.w - c) / 60.0) ** 2)
        cls.base = {0: [banda(7100, 0.3) * banda(8400, 0.1)], 1: [banda(7100, 0.1) * banda(8400, 0.3)],
                    2: [banda(7600, 0.3) * banda(8400, 0.2)]}

    def _seg(self, f):
        return [_seg("V", "vis", self.w, f, 0.01 * f)]

    def test_un_polinomio_multiplicativo_no_cambia_nada(self):
        poli = 2.0 + 0.3 * (self.w / 7800 - 1) - 0.5 * (self.w / 7800 - 1) ** 2
        res, _ = self.tp.ajusta_estructura(self._seg(self.base[1][0] * poli), self.base, {"vis": 2, "nir": 1})
        self.assertEqual(min(res, key=lambda k: res[k]["chi2"]), 1)
        self.assertLess(res[1]["chi2"], 1e-10)

    def test_el_enrojecimiento_no_cambia_el_tipo(self):
        e = 10 ** (-0.4 * self.tp.EXT.a_lambda_over_av(self.w) * 4.0)
        res, _ = self.tp.ajusta_estructura(self._seg(self.base[2][0] * e), self.base, {"vis": 2, "nir": 1})
        self.assertEqual(min(res, key=lambda k: res[k]["chi2"]), 2)
        # el orden 2 no absorbe del todo CCM a A_V = 4 (residuo ~1 %, χ²_ν 1.6 con σ del 1 %), pero queda muy por
        # debajo de cualquier plantilla equivocada
        self.assertLess(res[2]["chi2"], 0.05 * min(res[0]["chi2"], res[1]["chi2"]))   # medido: 0.025


class CurvasPorBibliotecaTests(unittest.TestCase):
    """Lo que dibujan `companion_type_nir` y `companion_type_combined` (la Fig. 5 con el IR)."""

    def test_cada_biblioteca_respecto_de_su_minimo(self):
        tp = _carga("tipo_ir_plantillas")
        pls = [dict(lib="a", spt=s) for s in ("M7", "M8", "M8", "M9")] + [dict(lib="b", spt=s) for s in ("L0", "L1")]
        chi2 = [30.0, 12.0, 10.0, 10.5, 50.0, 70.0]
        res = {i: dict(chi2=c) for i, c in enumerate(chi2)}
        out = tp.curvas_por_biblioteca(pls, res, escala=2.0)
        self.assertEqual(out["a"]["dchi2_by_spt"], {"M7": 10.0, "M8": 0.0, "M9": 0.25})
        self.assertEqual((out["a"]["spt_best"], out["a"]["spt_interval"], out["a"]["edge"]), ("M8", [8.0, 9.0], False))
        self.assertEqual(out["b"]["dchi2_by_spt"], {"L0": 0.0, "L1": 10.0})
        self.assertTrue(out["b"]["edge"])

    def test_figuras_registradas(self):
        pf = _carga("paper_figures")
        self.assertIs(pf.FIGURAS["companion_type_nir"], pf.companion_type_nir)
        self.assertIs(pf.FIGURAS["companion_type_combined"], pf.companion_type_combined)


class QualTests(unittest.TestCase):
    def test_el_bit_benigno_solo_quita_atipicos(self):
        tp = _carga("tipo_ir_plantillas")
        rng = np.random.default_rng(0)
        w = np.linspace(15000.0, 17000.0, 2000)
        f = 1.0 + 0.01 * rng.normal(size=w.size)
        q = np.zeros(w.size, np.int64)
        q[::4] = tp.BIT_BENIGNO                            # 25 % marcado, flujo normal
        f[400], q[400] = 3.0, tp.BIT_BENIGNO               # marcado y atípico
        q[401] = 2097152                                    # otro bit
        out = tp.limpia_qual(w, f, q)
        self.assertTrue(np.isnan(out[400]) and np.isnan(out[401]))
        self.assertEqual(int(np.isnan(out).sum()), 2)


class FalsosTests(unittest.TestCase):
    def test_conserva_la_sn_por_pixel(self):
        tp = _carga("tipo_ir_plantillas")
        w = np.linspace(20000.0, 23000.0, 50)
        datos = [_seg("K", "nir", w, np.full(50, 2.0), np.linspace(0.02, 0.2, 50))]
        out = tp.falsos(datos, [np.full(50, 5.0)], 0.0, np.random.default_rng(0))
        np.testing.assert_allclose(out[0]["e"] / 5.0, datos[0]["e"] / 2.0)


def _hay_datos():
    tir = _carga("tipo_ir_indices")
    return (tir.DEFAULT_CDS.is_dir() and _carga("tipo_ir_plantillas").DEFAULT_NIR.is_dir()
            and (ROOT / "runs" / "ROXs12b_invvar" / "stages").is_dir()
            and (ROOT / "runs" / "ROXs42Bb_realigned" / "stages").is_dir())


@pytest.mark.slow
@unittest.skipUnless(_hay_datos(), "sin catálogo CDS, sin el NIR de las plantillas o sin los runs del paper")
class CifrasDelPaperTests(unittest.TestCase):
    """Las cifras del apéndice B. El control (MUSE solo, A_V libre) es el tipo de G3."""

    @classmethod
    def setUpClass(cls):
        cls.tp = _carga("tipo_ir_plantillas")

    def _tipo(self, slug, modo, **kw):
        return self.tp.ajuste(slug, modo, **kw)["tipo"]

    def test_el_control_reproduce_g3(self):
        t12, t42 = self._tipo("ROXs12b", "muse"), self._tipo("ROXs42Bb", "muse")
        self.assertEqual((t12["spt"], t42["spt"]), ("M7.5", "M9.5"))
        self.assertAlmostEqual(t12["av"], 3.7, delta=0.15)

    def test_ir_y_combinado_con_la_av_de_la_tabla_1(self):
        self.assertEqual(self._tipo("ROXs12b", "ir", av_fija=True)["spt"], "M9")
        self.assertEqual(self._tipo("ROXs42Bb", "ir", av_fija=True)["spt"], "L0")
        self.assertEqual(self._tipo("ROXs12b", "combinado", av_fija=True)["spt"], "M9")
        # 2026-10-22, evaluador de la PSF centrado (docs/2026-10-21_psf_evaluador_centrado.md): era M9.5
        self.assertEqual(self._tipo("ROXs42Bb", "combinado", av_fija=True)["spt"], "L0")
        self.assertEqual(self._tipo("ROXs12b", "muse", av_fija=True)["spt"], "M9")

    def test_solo_estructuras_en_muse(self):
        pol = {"vis": 2, "nir": 1}
        # 2026-10-22, evaluador de la PSF centrado (docs/2026-10-21_psf_evaluador_centrado.md): eran M8, M9.5 y 0.64
        self.assertEqual(self._tipo("ROXs12b", "muse", av_fija=True, polinomio=pol)["spt"], "M7")
        self.assertEqual(self._tipo("ROXs42Bb", "muse", av_fija=True, polinomio=pol)["spt"], "L0")
        c = self.tp.calibra("ROXs12b", "muse", av_fija=True, polinomio=pol)
        # 2026-10-25, C4 v2 decoupled_c0 (docs/2026-10-24_c4_planos_desacoplados_decision.md): era 0.72
        self.assertAlmostEqual(c["rms"], 0.70, delta=0.01)

    def test_el_error_dejando_una_fuera(self):
        c = self.tp.calibra("ROXs42Bb", "combinado", av_fija=True)
        # 2026-10-22, evaluador de la PSF centrado (docs/2026-10-21_psf_evaluador_centrado.md): eran 1.03 y 0.0
        self.assertAlmostEqual(c["rms"], 1.06, delta=0.01)
        self.assertEqual(c["mediana"], -0.25)


if __name__ == "__main__":
    unittest.main()
