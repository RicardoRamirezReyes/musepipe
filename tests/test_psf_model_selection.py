"""Spec C1 §3.4: Moffat-vs-Psfao per-bin comparison and form selection.

Covers the acceptance tests from the WP-5 plan:
  (a) the QC contract carries a ``model_comparison`` block with both forms;
  (b) a scene where a plain elliptical Moffat suffices -> auto picks moffat;
  (c) a scene generated from the physical AO (Psfao) model -> auto picks psfao.

The hybrid-safety requirement (§3.5) is covered by test_psf_hybrid_safety.py.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.psf import evaluate_psf_model, moffat_image
from musepipe.stages.stage_e01_psf import (
    compute_stage_e01_products,
    stage_e01_paths,
    write_stage_e01_products,
)
from tests.test_stage01c_synthetic import gaussian2d

try:
    import maoppy  # noqa: F401

    HAS_MAOPPY = True
except Exception:  # pragma: no cover - environment dependent
    HAS_MAOPPY = False

NY = NX = 84
PRIMARY_YX = (NY // 2, NX // 2)          # Psfao fit assumes the star at the grid centre
COMPANION_YX = (NY // 2, NX // 2 + 22)   # 22 px separation, outside the mask
WAVES = np.linspace(6500.0, 6800.0, 21)  # 3 clean 100 A bins, off the 5780-6050 window


def _moffat_cube():
    cube = []
    for wave in WAVES:
        fmaj = 4.0 + 0.0005 * (wave - 6500.0)
        fmin = 3.5 + 0.0003 * (wave - 6500.0)
        star = moffat_image((NY, NX), PRIMARY_YX[0], PRIMARY_YX[1], fmaj, fmin, 15.0, 2.8,
                            amplitude=1000.0, background=1.0)
        companion = gaussian2d(NY, NX, COMPANION_YX[0], COMPANION_YX[1], 40.0, 1.4)
        cube.append(star + companion)
    return np.asarray(cube, dtype=np.float32)


def _psfao_cube():
    from maoppy.instrument import muse_nfm
    from maoppy.psfmodel import Psfao
    from musepipe.stages.stage_e01_psfao import DEFAULT_X0

    rng = np.random.default_rng(0)
    cube = []
    for wave in WAVES:
        samp = float(muse_nfm.samp(float(wave) * 1e-10))
        model = Psfao((NY, NX), system=muse_nfm, samp=samp)
        base = np.asarray(model(DEFAULT_X0), dtype=np.float64)
        base = base / base.max() * 1000.0  # AO core + halo, peak ~1000 at grid centre
        companion = gaussian2d(NY, NX, COMPANION_YX[0], COMPANION_YX[1], 40.0, 1.4)
        cube.append(base + companion + rng.normal(0.0, 0.5, (NY, NX)))
    return np.asarray(cube, dtype=np.float32)


def _write_inputs(root, cube):
    stage_dir = root / "runs" / "sel" / "stages"
    stage_dir.mkdir(parents=True)
    cube_path = stage_dir / "stage02_xcorr_cube_stack.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(cube[None].astype(np.float32), name="CUBES"),
        fits.ImageHDU(WAVES.astype(np.float64), name="WAVELENGTH"),
        fits.ImageHDU(np.ones_like(cube[None], dtype=np.float32), name="STAT"),
    ]).writeto(cube_path)
    qc = {
        "stage": "01c_target_localization",
        "run_id": "sel",
        "primary": {"pos_yx": list(map(float, PRIMARY_YX))},
        "companion": {"pos_yx": list(map(float, COMPANION_YX))},
        "field_source": None,
        "chromatic": {"chromatic_centroid_needed": False},
        "psf": {"fwhm_px": 4.0},
    }
    qc_path = stage_dir / "stage01c_qc.json"
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    return cube_path, qc_path


def _cfg(root, cube_path, qc_path):
    return {
        "run_id": "sel",
        "project_root": str(root),
        "stage_e01_input_cube_fits": str(cube_path),
        "stage_e01_positions_qc": str(qc_path),
        "e01_psf_form": "auto",
        "psf_bin_A": 100.0,
        "psf_min_channels_per_bin": 3,
        "psf_fit_radius_px": 38.0,
        "psf_norm_radius_px": 20.0,
        "psf_companion_mask_radius_px": 12.0,
        "psf_companion_ring_width_px": 3.0,
        "psf_hybrid_threshold_pct": 1e6,  # keep the comparison isolated from the hybrid step
    }


@unittest.skipUnless(HAS_MAOPPY, "maoppy required for the Psfao comparison")
class ModelSelectionTests(unittest.TestCase):
    def _run(self, cube):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cube_path, qc_path = _write_inputs(root, cube)
            cfg = _cfg(root, cube_path, qc_path)
            product = compute_stage_e01_products(cfg)
            paths = stage_e01_paths("sel", project_root=root)
            written = write_stage_e01_products(product, cfg, paths)
            return product, written

    def _assert_comparison_contract(self, qc):
        self.assertIn("model_comparison", qc)
        mc = qc["model_comparison"]
        self.assertEqual(mc["selection_mode"], "auto")
        self.assertIn(mc["form_chosen"], ("moffat", "psfao"))
        self.assertEqual(qc["fit"]["form_chosen"], mc["form_chosen"])
        for form in ("moffat", "psfao"):
            self.assertIn(form, mc)
        self.assertEqual(mc["psfao"]["status"], "ok")
        self.assertIsInstance(mc["moffat"]["ring_residual_pct_median"], float)
        self.assertIsInstance(mc["psfao"]["ring_residual_pct_median"], float)

    def test_moffat_scene_selects_moffat(self):
        product, written = self._run(_moffat_cube())
        self._assert_comparison_contract(product.qc)
        self.assertEqual(product.psf_form, "moffat")
        mc = product.qc["model_comparison"]
        self.assertLessEqual(mc["moffat"]["ring_residual_pct_median"],
                             mc["psfao"]["ring_residual_pct_median"])
        self.assertEqual(product.psf_model["form"], "moffat")

    def test_psfao_scene_selects_psfao(self):
        product, written = self._run(_psfao_cube())
        self._assert_comparison_contract(product.qc)
        self.assertEqual(product.psf_form, "psfao")
        mc = product.qc["model_comparison"]
        self.assertLess(mc["psfao"]["ring_residual_pct_median"],
                        mc["moffat"]["ring_residual_pct_median"])
        # The winning document must be consumable by the downstream evaluator.
        self.assertEqual(product.psf_model["form"], "psfao")
        yy, xx = np.mgrid[-5:6, -5:6].astype(float)
        vals = evaluate_psf_model(product.psf_model, float(WAVES[len(WAVES) // 2]), yy, xx)
        self.assertTrue(np.all(np.isfinite(vals)))
        self.assertLess(product.qc["normalization"]["roundtrip_error"], 0.05)


if __name__ == "__main__":
    unittest.main()


class FormIsFrozenPerObjectTests(unittest.TestCase):
    """Declarar la forma decide el ganador, y NO apaga la comparacion.

    El peso del ajuste (`psf_fit_weighting`) cambia el residuo de anillo de una
    forma y no de la otra —en ROXs 12 b mejora psfao x2.3 y en ROXs 42B b lo
    empeora x1.4—, asi que dejar la seleccion en `auto` es dejar que un knob de
    ajuste pueda voltear la familia de modelo de un objeto sin que nadie mire.
    Se congela por run; pero la §3.4 de la spec pide las dos formas medidas, y
    forzar no puede llevarse esa medida por delante.
    """

    def test_forcing_moffat_still_measures_psfao(self):
        import inspect

        from musepipe.stages import stage_e01_psf as m

        fuente = inspect.getsource(m.compute_stage_e01_products)
        self.assertIn('e01_psf_compare_forms', fuente)
        # La condicion tiene que dejar entrar a psfao tambien con moffat forzado.
        self.assertIn('form_cfg in ("auto", "psfao") or comparar', fuente)

    def test_the_runs_on_disk_declare_the_form_they_already_have(self):
        """Congelar no puede cambiar nada: la forma declarada es la que hay."""
        import json

        raiz = Path(__file__).resolve().parents[1] / "runs"
        vistos = 0
        for cfg_path in sorted(raiz.glob("*/config/config.json")):
            modelo = cfg_path.parent.parent / "stages" / "psf_model.json"
            if not modelo.exists():
                continue
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            declarada = cfg.get("config", {}).get("e01_psf_form")
            if declarada is None:
                continue
            vistos += 1
            with self.subTest(run=cfg_path.parent.parent.name):
                self.assertEqual(
                    declarada,
                    json.loads(modelo.read_text(encoding="utf-8"))["form"],
                    "la forma declarada en el config no es la del modelo en disco")
        if vistos == 0:
            self.skipTest("ningun run con forma declarada en este clon")
