"""V4 de C3 comparaba contra nada: era el literal `True`.

Sus tres hermanos (`v1_snr_gain_ok`, `v2_error_ratio_ok`,
`v3_continuum_bias_ok`) deducen el veredicto de su medida; el cuarto no es que
no disparara con estos datos, es que no podia disparar con ningunos. Y la medida
que ignoraba se calculaba y se guardaba al lado.

Lo que el spec (C3 §6) pide es «sin concentracion en la posicion del compañero
(< 2x la tasa media en su ventana)», asi que la cifra que juzga es la del NUCLEO
contra el resto de la ventana -la auditada en
`docs/2026-09-08_d1_tension_continuo_medida.md` §8: 186.7 canales de media en el
nucleo contra 3.7 fuera, o sea 50x-, no el pico/media, cuyo pico puede caer en
cualquier pixel.
"""
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from musepipe.stages import stage_x02_optimal as x02
from musepipe.stages.stage_x02_optimal import (
    DEFAULT_CLIP_CONCENTRATION_MAX,
    compute_stage_x02_products,
    stage_x02_paths,
)
from tests.test_optimal_analytic import constant_model_doc
from tests.test_optimal_stage import psf_cube


def rejection_stub(rejection_map):
    return SimpleNamespace(rejection_map=np.asarray(rejection_map, dtype=np.float64))


class ClipConcentrationMeasureTests(unittest.TestCase):
    def test_recorte_uniforme_da_uno(self):
        ext = rejection_stub(np.full((40, 40), 3.7))
        out = x02._clip_concentration(ext, (20.0, 20.0), 8.0)
        self.assertAlmostEqual(out["core_over_outer"], 1.0, places=6)
        self.assertAlmostEqual(out["companion_window_over_mean"], 1.0, places=6)

    def test_el_recorte_cebado_en_el_nucleo_se_mide(self):
        """Reproduce la anomalia del run del paper: 186.7 contra 3.7."""
        cmap = np.full((40, 40), 3.7)
        yy, xx = np.indices(cmap.shape, dtype=np.float64)
        core = np.hypot(yy - 20.0, xx - 20.0) <= 1.5
        cmap[core] = 186.7
        out = x02._clip_concentration(ext := rejection_stub(cmap), (20.0, 20.0), 8.0)
        del ext
        self.assertAlmostEqual(out["core_over_outer"], 186.7 / 3.7, places=6)
        self.assertLessEqual(out["peak_offset_px"], 1.5)
        self.assertGreater(out["core_over_outer"], DEFAULT_CLIP_CONCENTRATION_MAX)

    def test_el_umbral_es_el_del_spec(self):
        self.assertEqual(DEFAULT_CLIP_CONCENTRATION_MAX, 2.0)

    def test_la_puerta_no_es_un_literal(self):
        src = inspect.getsource(x02._qc_payload)
        self.assertNotIn('"v4_clip_concentration_ok": True', src)
        self.assertIn("v4_measure", src)


def build_synthetic_run(root, run_id, *, hot_pixel=None):
    """Cubo sintetico de C3; `hot_pixel` mete un outlier en el nucleo."""

    model = constant_model_doc()
    wave = np.linspace(6500.0, 6540.0, 5)
    primary_yx = (30.0, 30.0)
    companion_yx = (30.0, 42.0)
    companion_cube = psf_cube(model, wave, companion_yx, 80.0)
    primary_cube = psf_cube(model, wave, primary_yx, 1000.0)
    stage02_cube = primary_cube + companion_cube
    if hot_pixel is not None:
        y, x = hot_pixel
        companion_cube[:, y, x] += 500.0
        stage02_cube[:, y, x] += 500.0
    stat = np.ones((1,) + stage02_cube.shape, dtype=np.float32)

    paths = stage_x02_paths(run_id, project_root=root)
    paths["paths"].ensure_base_dirs()
    hdr = fits.Header()
    hdr["WMIN"] = float(wave[0])
    hdr["DW"] = float(np.nanmedian(np.diff(wave)))
    hdr["BUNIT"] = "native"
    fits.PrimaryHDU(companion_cube, header=hdr).writeto(paths["cube_residual_object"])
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(stage02_cube[None], name="CUBES"),
            fits.ImageHDU(wave, name="WAVELENGTH"),
            fits.ImageHDU(stat, name="STAT"),
        ]
    ).writeto(paths["stage02_cube_fits"])
    paths["stage01c_qc_json"].write_text(
        json.dumps({"stage": "01c", "run_id": run_id,
                    "primary": {"pos_yx": list(primary_yx)},
                    "companion": {"pos_yx": list(companion_yx)}}),
        encoding="utf-8",
    )
    paths["psf_model_json"].write_text(json.dumps(model), encoding="utf-8")
    cfg = {
        "run_id": run_id,
        "project_root": str(root),
        "x02_window_radius_px": 6.0,
        "x02_aperture_correction": "psf_growth_curve",
        "x02_bad_windows_A": [],
        "x02_primary_fit_radius_px": 18.0,
    }
    return cfg, paths


class ClipConcentrationGateTests(unittest.TestCase):
    def test_la_puerta_dispara_con_el_recorte_cebado_en_el_compañero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, paths = build_synthetic_run(root, "synthetic_x02_v4", hot_pixel=(30, 42))
            qc = compute_stage_x02_products(cfg, paths).qc
            self.assertIs(qc["checks"]["v4_clip_concentration_ok"], False)
            self.assertTrue(
                any("V4" in str(issue.get("issue", "") if isinstance(issue, dict) else issue)
                    for issue in qc["open_issues"]),
                "una puerta que dispara y no se declara no la lee nadie",
            )

    def test_sin_recorte_no_inventa_un_fallo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, paths = build_synthetic_run(root, "synthetic_x02_v4_limpio")
            qc = compute_stage_x02_products(cfg, paths).qc
            self.assertIsNot(qc["checks"]["v4_clip_concentration_ok"], False)
            self.assertEqual(qc["clip_concentration"]["max_ratio"], DEFAULT_CLIP_CONCENTRATION_MAX)


if __name__ == "__main__":
    unittest.main()
