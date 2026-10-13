"""E4 extrae `aperture` y `optimal_ls` con la llamada EXACTA de C2 y C3.

La spec E4 §1.4 lo exige («los extractores se llaman con su configuración de
producción EXACTA»), pero E4 copiaba las dos llamadas a mano y la copia se quedó
atrás: seguía extrayendo del residuo de 04b (más el anillo, en `optimal_ls`)
cuando C2 y C3 ya extraían del cubo crudo con anillo. El throughput de esos dos
métodos salía un 1.3-3.8 % bajo en el umbral de E3, y nada lo declaraba
(`docs/2026-10-12_e4_extractores_de_produccion.md`).

Ahora E4 toma los argumentos de las mismas funciones que usan C2 y C3. Aquí se
fija que, sobre el cubo sin inyectar y en la posición del objeto, E4 da el mismo
espectro que el producto, que para si inyecta en otro cubo, y que el QC dice de
dónde extrae cada método.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.extraction.aperture import extract_aperture_products
from musepipe.extraction.optimal import make_optimal_product
from musepipe.stages import stage_x01_aperture as x01
from musepipe.stages import stage_x02_optimal as x02
from musepipe.stages.stage_h04_extractors import (
    production_aperture_call,
    production_optimal_ls_call,
)
from musepipe.stages.stage_h04_injection import extractor_substrates
from tests.test_optimal_ls_wings_intact import COMPANION_YX, _build_run


def _config(run_id, root, **extra):
    cfg = {
        "run_id": run_id,
        "project_root": str(root),
        "x01_aperture_correction": "psf_growth_curve",
        "x01_annulus_bkg_px": [8.0, 12.0, 20.0],
        "x02_window_radius_px": 6.0,
        "x02_aperture_correction": "psf_growth_curve",
        "x02_primary_fit_radius_px": 18.0,
        "x02_local_bkg_annulus_px": [8.0, 12.0, 20.0],
    }
    cfg.update(extra)
    return cfg


def _base_cube(paths):
    with fits.open(paths["stage02_cube_fits"]) as hdul:
        return np.asarray(hdul["CUBES"].data[0], dtype=np.float64), np.asarray(
            hdul["WAVELENGTH"].data, dtype=np.float64)


class ExtraeComoProduccionTests(unittest.TestCase):
    def test_aperture_da_el_espectro_de_c2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, run_id = _build_run(root)
            cfg = _config(run_id, root)
            base, wave = _base_cube(paths)
            ap_wave, kwargs, raw = production_aperture_call(
                cfg, x01.stage_x01_paths(run_id, root), base_cube=base, wave_A=wave)
            self.assertTrue(raw, "con la apcorr activa, C2 extrae del cubo crudo")
            self.assertEqual(kwargs["annulus_bkg_px"], [8.0, 12.0, 20.0])
            self.assertEqual(kwargs["n_controls"], 0)
            self.assertIsNone(kwargs["growth_curve"])
            e4 = extract_aperture_products(base, ap_wave, COMPANION_YX, **kwargs)["box3"].product
            c2 = x01.compute_stage_x01_products(cfg).extractions["box3"].product
            np.testing.assert_array_equal(np.asarray(e4.flux), np.asarray(c2.flux))

    def test_optimal_ls_da_el_espectro_de_c3(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, run_id = _build_run(root)
            cfg = _config(run_id, root)
            base, wave = _base_cube(paths)
            ls_wave, psf, kwargs, raw = production_optimal_ls_call(
                cfg, x02.stage_x02_paths(run_id, root), base_cube=base, wave_A=wave)
            self.assertTrue(raw, "C3 extrae ls del cubo crudo con anillo, no del residuo de 04b")
            self.assertEqual(kwargs["local_bkg_annulus_px"], [8.0, 12.0, 20.0])
            self.assertEqual(kwargs["n_controls"], 0)
            self.assertIsNone(kwargs["growth_curve"])
            e4 = make_optimal_product(base, ls_wave, COMPANION_YX, psf, variant="ls",
                                      input_cube_path="h04:inmemory", **kwargs).product
            c3 = x02.compute_stage_x02_products(cfg, paths).extractions["ls"].product
            np.testing.assert_array_equal(np.asarray(e4.flux), np.asarray(c3.flux))

    def test_sin_alas_intactas_sigue_al_residuo_de_04b(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, run_id = _build_run(root)
            cfg = _config(run_id, root, x01_wings_intact_apcorr=False, x02_wings_intact_ls=False)
            base, wave = _base_cube(paths)
            _w, _k, raw_ap = production_aperture_call(
                cfg, x01.stage_x01_paths(run_id, root), base_cube=base, wave_A=wave)
            _w, _p, _k, raw_ls = production_optimal_ls_call(
                cfg, x02.stage_x02_paths(run_id, root), base_cube=base, wave_A=wave)
            self.assertFalse(raw_ap)
            self.assertFalse(raw_ls)

    def test_si_inyecta_en_otro_cubo_para(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, run_id = _build_run(root)
            cfg = _config(run_id, root)
            base, wave = _base_cube(paths)
            otro = base + 1.0
            with self.assertRaisesRegex(RuntimeError, "different cube"):
                production_aperture_call(cfg, x01.stage_x01_paths(run_id, root),
                                         base_cube=otro, wave_A=wave)
            with self.assertRaisesRegex(RuntimeError, "different cube"):
                production_optimal_ls_call(cfg, x02.stage_x02_paths(run_id, root),
                                           base_cube=otro, wave_A=wave)


class DeclaraElSustratoTests(unittest.TestCase):
    def test_el_qc_dice_de_donde_extrae_cada_metodo(self):
        def declarado(*_args):
            return None

        def mudo(*_args):
            return None

        declarado.substrate = {"cube": "stage02 (raw, wings intact) + annulus background"}
        salida = extractor_substrates({"aperture": declarado, "psffit": mudo}, ["aperture", "psffit"])
        self.assertEqual(salida["aperture"]["cube"], "stage02 (raw, wings intact) + annulus background")
        self.assertIsNone(salida["psffit"], "lo que no se declara no se adivina")


if __name__ == "__main__":
    unittest.main()
