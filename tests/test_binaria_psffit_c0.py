"""El sesgo de la binaria de ROXs 42B sobre psffit con los dos fondos de C4 (`scripts/binaria_psffit_c0.py`).

Las clases sintéticas no tocan `runs/`: fijan la definición del porcentaje (cambio integrado por banda y
neto = objeto − control) y el mecanismo de la decisión B con una escena sin ruido —una primaria binaria
no modelada sesga el compañero un 12 % con el plano compartido y menos del 0.01 % con decoupled_c0—.
`ValidacionContraC4Tests` lee un run del paper (solo lectura, ~20 s) y comprueba que la variante `base`
de la sonda, por el factor de flujo de C4, es el producto de C4 bit a bit en unos canales. Busca `runs/`
en este clon o en `MUSE_PROJECT_ROOT`; si faltan, se salta.
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from musepipe.extraction.psffit import _fit_one_channel, fit_region_mask, psf_pair_design
from musepipe.psf import evaluate_psf_model
from tests.test_optimal_analytic import constant_model_doc

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


class PorcentajeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _carga("binaria_psffit_c0")

    def test_el_cambio_por_banda_es_integrado_y_no_canal_a_canal(self):
        w = np.array([7000.0, 7250.0, 7499.0, 7500.0, 8100.0, 8900.0])
        base = np.array([100.0, 300.0, 100.0, 1.0, 50.0, 10.0])
        var = np.array([90.0, 300.0, 100.0, 999.0, 49.0, np.nan])
        c = self.m.cambio_por_banda(w, base, var)
        self.assertAlmostEqual(c["7000-7500"], 100.0 * (490.0 - 500.0) / 500.0, places=12)  # -2 %, no -3.3 %
        self.assertAlmostEqual(c["8000-8500"], -2.0, places=12)
        self.assertTrue(np.isnan(c["8800-9300"]))  # el único canal no es finito en la variante

    def test_el_neto_es_objeto_menos_control(self):
        neto = self.m.cambio_neto({"a": -12.8, "b": -4.1}, {"a": -6.5, "b": -2.7})
        self.assertAlmostEqual(neto["a"], -6.3, places=12)
        self.assertAlmostEqual(neto["b"], -1.4, places=12)

    def test_libre_en_el_rojo(self):
        w = np.array([7900.0, 8000.0, 8100.0, 8200.0, 8300.0])
        base = np.array([1.0, 1.0, 100.0, 100.0, 200.0])
        libre = np.array([0.0, 0.0, 99.0, 98.0, 200.0])
        mediana, total = self.m.libre_en_el_rojo(w, base, libre)
        self.assertAlmostEqual(mediana, -1.0, places=12)  # λ > 8000: -1, -2, 0
        self.assertAlmostEqual(total, 100.0 * (397.0 - 400.0) / 400.0, places=12)

    def test_los_canales_de_la_sesion_en_la_rejilla_de_muse(self):
        wave = 4750.0 + 1.25 * np.arange(3681)
        ch = self.m.canales_de_ventanas(wave)
        self.assertEqual(ch.size, 744)  # 144 de Hα + 3 × 200
        self.assertTrue(np.all(np.diff(ch) > 0))
        ha = ch[(wave[ch] >= 6480.0) & (wave[ch] < 6660.0)]
        self.assertTrue(np.all(np.diff(ha) == 1))
        rojo = ch[(wave[ch] >= 8000.0) & (wave[ch] < 8500.0)]
        self.assertTrue(np.all(np.diff(rojo) == 2))
        self.assertEqual(self.m.canales_de_ventanas(wave, step=2).size, 72 + 3 * 100)

    def test_halpha_sin_cambio_da_cero_sigmas(self):
        w = 6480.0 + 1.25 * np.arange(144)
        flujo = 50.0 + 400.0 * np.exp(-0.5 * ((w - 6562.8) / 1.0) ** 2)
        err = np.full(w.size, 5.0)
        h = self.m.desplazamiento_halpha(w, err, {"base": flujo, "ligada": flujo.copy(), "libre": flujo + 0.0},
                                         lsf_fwhm_A=2.3)
        self.assertGreater(h["base_sigma"], 10.0)
        self.assertEqual(h["ligada_shift_sigma"], 0.0)
        self.assertEqual(h["libre_shift_sigma"], 0.0)
        mas = self.m.desplazamiento_halpha(w, err, {"base": flujo, "ligada": flujo + 20.0 * np.exp(
            -0.5 * ((w - 6562.8) / 1.0) ** 2)}, lsf_fwhm_A=2.3)
        self.assertGreater(mas["ligada_shift_sigma"], 0.0)


SHAPE = (120, 120)
STAR_YX = (60.0, 60.0)
COMP_YX = (60.0, 105.0)
OFFSET_YX = (-1.7095, -1.0682)  # la binaria de ROXs 42B en la rejilla de NFM (51 mas, PA 148°)
WAVE = np.array([7100.0, 7300.0, 8100.0, 8300.0, 8900.0, 9100.0])


class MecanismoSinteticoTests(unittest.TestCase):
    """Una primaria binaria no modelada, sin ruido: el porcentaje que publica la herramienta."""

    @classmethod
    def setUpClass(cls):
        cls.m = _carga("binaria_psffit_c0")
        cls.v1 = _carga("binaria_psffit_probe")
        cls.model = constant_model_doc()
        yy, xx = np.indices(SHAPE, dtype=np.float64)
        cls.mask = fit_region_mask(SHAPE, STAR_YX, COMP_YX, star_radius_px=20.0, comp_radius_px=12.0)
        f = 0.13
        cls.res = {}
        for fondo in ("shared", "decoupled_c0"):
            sal = {(lab, v): np.zeros(WAVE.size) for lab in ("objeto", "control") for v in ("base", "ligada")}
            for i, w in enumerate(WAVE):
                p2 = evaluate_psf_model(cls.model, float(w), yy - (STAR_YX[0] + OFFSET_YX[0]),
                                        xx - (STAR_YX[1] + OFFSET_YX[1]))
                d = psf_pair_design(SHAPE, w, STAR_YX, COMP_YX, cls.model, background=fondo)
                escenas = {"objeto": 1.0e5 * (d[..., 0] + f * p2) + 100.0 * d[..., 1],  # binaria de verdad
                           "control": 1.0e5 * d[..., 0] + 100.0 * d[..., 1]}           # estrella sola
                base, ligada, _libre = cls.v1.design_variants(d, p2, f)
                for lab, escena in escenas.items():
                    for v, diseno in (("base", base), ("ligada", ligada)):
                        sal[(lab, v)][i] = _fit_one_channel(escena, np.ones(SHAPE), diseno, cls.mask)[0][1]
            obj = cls.m.cambio_por_banda(WAVE, sal[("objeto", "base")], sal[("objeto", "ligada")])
            ctl = cls.m.cambio_por_banda(WAVE, sal[("control", "base")], sal[("control", "ligada")])
            cls.res[fondo] = {"sal": sal, "objeto": obj, "control": ctl, "neto": cls.m.cambio_neto(obj, ctl)}

    def test_la_ligada_recupera_el_compañero_de_la_escena_binaria(self):
        for fondo in ("shared", "decoupled_c0"):
            np.testing.assert_allclose(self.res[fondo]["sal"][("objeto", "ligada")], 100.0, rtol=1e-9)
            np.testing.assert_allclose(self.res[fondo]["sal"][("control", "base")], 100.0, rtol=1e-9)

    def test_con_el_plano_compartido_la_binaria_sesga_el_compañero(self):
        # la base sobreestima el compañero un 13 %: la ligada lo baja un 11.6 %, en todas las bandas
        for banda, valor in self.res["shared"]["objeto"].items():
            self.assertAlmostEqual(valor, -11.6416, places=3, msg=banda)
        # imponer la binaria a una estrella sola también mueve el compañero con el plano compartido: es el
        # suelo del método que el neto resta (el signo y el tamaño del neto dependen de la escena)
        for banda in self.res["shared"]["neto"]:
            self.assertGreater(abs(self.res["shared"]["control"][banda]), 5.0, msg=banda)
            self.assertGreater(abs(self.res["shared"]["neto"][banda]), 1.0, msg=banda)

    def test_con_decoupled_c0_el_sesgo_desaparece(self):
        for banda in self.res["decoupled_c0"]["neto"]:
            self.assertLess(abs(self.res["decoupled_c0"]["objeto"][banda]), 0.01, msg=banda)
            self.assertLess(abs(self.res["decoupled_c0"]["control"][banda]), 0.1, msg=banda)
            self.assertLess(abs(self.res["decoupled_c0"]["neto"][banda]), 0.1, msg=banda)


RAIZ = _raiz_con_runs("ROXs42Bb_realigned")


@unittest.skipUnless(RAIZ is not None, "sin el run del paper (ni aquí ni en MUSE_PROJECT_ROOT)")
class ValidacionContraC4Tests(unittest.TestCase):
    def test_la_base_por_el_factor_de_c4_es_el_producto_bit_a_bit(self):
        m = _carga("binaria_psffit_c0")
        meta, out = m.sonda("ROXs42Bb_realigned", RAIZ, fondos=("decoupled_c0",),
                            ventanas=((8000.0, 8010.0, 1), (6560.0, 6565.0, 1)), verbose=False)
        self.assertEqual(meta["x03_background_del_run"], "decoupled_c0")
        self.assertEqual(meta["n_canales"], 12)
        stage = RAIZ / "runs" / "ROXs42Bb_realigned" / "stages"
        val = m.valida_contra_c4(meta, out, RAIZ, productos={"decoupled_c0": stage})["decoupled_c0"]
        self.assertEqual(val["BKGMODE"], "psffit_plane_decoupled_c0")
        self.assertTrue(val["igual_bit_a_bit"], val)


if __name__ == "__main__":
    unittest.main()
