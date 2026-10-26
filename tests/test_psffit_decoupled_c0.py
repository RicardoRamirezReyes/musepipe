"""C4's background term: `shared` (v1) and `decoupled_c0` (spec C4 v2 §1).

`shared` fits one constant and two gradients over the union of the star's and
the companion's discs. Whatever the PSF model leaves in the star's core, the
gradients absorb inside its disc and carry to the companion with a long lever
arm, and the sign depends on the companion's azimuth: a dipole in the controls
(`docs/2026-10-24_c4_planos_desacoplados_decision.md`). `decoupled_c0` keeps the
plane on the star's disc and gives the companion only its own constant.

These tests guard three things: the default is still v1 bit for bit (legacy
runs must reproduce), `decoupled_c0` builds the columns the decision describes,
and the mechanism it exists for.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.extraction.psffit import (
    BACKGROUND_MODELS,
    fit_psffit_cube,
    make_psffit_products,
    n_design_columns,
    psf_pair_design,
)
from musepipe.psf import evaluate_psf_model
from musepipe.stages.stage_x03_psffit import compute_stage_x03_products, stage_x03_paths
from tests.test_optimal_analytic import constant_model_doc

SHAPE = (120, 120)
STAR_YX = (60.0, 60.0)
SEPARATION_PX = 45.0  # like the companions of the paper runs (47 and 71 px)


def companion_at(angle_deg):
    a = math.radians(angle_deg)
    return (STAR_YX[0] + SEPARATION_PX * math.sin(a), STAR_YX[1] + SEPARATION_PX * math.cos(a))


class DesignColumnsTests(unittest.TestCase):
    def setUp(self):
        self.model = constant_model_doc()
        self.comp_yx = companion_at(30.0)

    def test_the_default_is_the_v1_design_bit_for_bit(self):
        yy, xx = np.indices(SHAPE, dtype=np.float64)
        p_star = evaluate_psf_model(self.model, 6520.0, yy - STAR_YX[0], xx - STAR_YX[1])
        p_comp = evaluate_psf_model(self.model, 6520.0, yy - self.comp_yx[0], xx - self.comp_yx[1])
        y0 = 0.5 * (STAR_YX[0] + self.comp_yx[0])
        x0 = 0.5 * (STAR_YX[1] + self.comp_yx[1])
        scale = max(math.hypot(self.comp_yx[0] - STAR_YX[0], self.comp_yx[1] - STAR_YX[1]), 1.0)
        v1 = np.stack([p_star, p_comp, np.ones(SHAPE), (yy - y0) / scale, (xx - x0) / scale], axis=-1)

        default = psf_pair_design(SHAPE, 6520.0, STAR_YX, self.comp_yx, self.model)
        shared = psf_pair_design(SHAPE, 6520.0, STAR_YX, self.comp_yx, self.model, background="shared")

        self.assertTrue(np.array_equal(default, v1))
        self.assertTrue(np.array_equal(shared, v1))
        self.assertEqual(n_design_columns(), 5)

    def test_decoupled_c0_keeps_the_plane_on_the_star_and_a_constant_on_the_companion(self):
        shared = psf_pair_design(SHAPE, 6520.0, STAR_YX, self.comp_yx, self.model)
        c0 = psf_pair_design(SHAPE, 6520.0, STAR_YX, self.comp_yx, self.model, background="decoupled_c0",
                             star_radius_px=20.0, comp_radius_px=12.0)
        yy, xx = np.indices(SHAPE, dtype=np.float64)
        star_disc = (yy - STAR_YX[0]) ** 2 + (xx - STAR_YX[1]) ** 2 <= 20.0 ** 2
        comp_disc = (yy - self.comp_yx[0]) ** 2 + (xx - self.comp_yx[1]) ** 2 <= 12.0 ** 2

        self.assertEqual(c0.shape[-1], 6)
        self.assertEqual(n_design_columns("decoupled_c0"), 6)
        np.testing.assert_array_equal(c0[..., :2], shared[..., :2])
        np.testing.assert_array_equal(c0[..., 2], star_disc.astype(float))
        np.testing.assert_array_equal(c0[..., 3], np.where(star_disc, shared[..., 3], 0.0))
        np.testing.assert_array_equal(c0[..., 4], np.where(star_disc, shared[..., 4], 0.0))
        np.testing.assert_array_equal(c0[..., 5], comp_disc.astype(float))

    def test_an_unknown_background_is_refused_everywhere(self):
        cube = np.zeros((1,) + SHAPE)
        with self.assertRaises(ValueError):
            psf_pair_design(SHAPE, 6520.0, STAR_YX, self.comp_yx, self.model, background="decoupled")
        with self.assertRaises(ValueError):
            fit_psffit_cube(cube, np.ones_like(cube), [6520.0], STAR_YX, self.comp_yx, self.model,
                            background="plane")
        self.assertEqual(BACKGROUND_MODELS, ("shared", "decoupled_c0"))


class DecoupledC0FitTests(unittest.TestCase):
    def setUp(self):
        self.model = constant_model_doc()
        self.yy, self.xx = np.indices(SHAPE, dtype=np.float64)

    def _psf(self, yx):
        return evaluate_psf_model(self.model, 6520.0, self.yy - yx[0], self.xx - yx[1])

    def test_it_recovers_its_own_scene_exactly(self):
        comp_yx = companion_at(120.0)
        truth = np.array([9000.0, 80.0, 3.0, 0.4, -0.3, -2.0])
        design = psf_pair_design(SHAPE, 6520.0, STAR_YX, comp_yx, self.model, background="decoupled_c0")
        cube = np.tensordot(design, truth, axes=([-1], [0]))[None]

        fit = fit_psffit_cube(cube, np.full_like(cube, 2.0), [6520.0], STAR_YX, comp_yx, self.model,
                              background="decoupled_c0")

        self.assertEqual(fit.coeffs.shape, (1, 6))
        self.assertEqual(fit.covariance.shape, (1, 6, 6))
        np.testing.assert_allclose(fit.coeffs[0], truth, rtol=1e-9, atol=1e-9)
        self.assertTrue(np.isfinite(fit.rho_bc[0]))

    def test_a_star_core_residual_reaches_the_companion_only_through_the_shared_plane(self):
        # The star sits 0.3 px off where the fit puts it, as C4's fixed
        # position does against the measured chromatic drift. Nothing else is
        # in the scene: no background, no noise.
        offset = (0.3, 0.3)
        bias = {}
        for angle in (0.0, 180.0):
            comp_yx = companion_at(angle)
            scene = 1.0e5 * self._psf((STAR_YX[0] + offset[0], STAR_YX[1] + offset[1])) + 100.0 * self._psf(comp_yx)
            cube = scene[None]
            for background in BACKGROUND_MODELS:
                fit = fit_psffit_cube(cube, np.ones_like(cube), [6520.0], STAR_YX, comp_yx, self.model,
                                      background=background)
                bias[(background, angle)] = float(fit.coeffs[0, 1]) - 100.0

        # Shared: a dipole, tens of percent of the companion, opposite signs on
        # opposite sides of the star.
        self.assertGreater(abs(bias[("shared", 0.0)]), 10.0)
        self.assertGreater(abs(bias[("shared", 180.0)]), 10.0)
        self.assertLess(bias[("shared", 0.0)] * bias[("shared", 180.0)], 0.0)
        # Decoupled: below 1 % on both sides.
        self.assertLess(abs(bias[("decoupled_c0", 0.0)]), 1.0)
        self.assertLess(abs(bias[("decoupled_c0", 180.0)]), 1.0)


class StageAndProductTests(unittest.TestCase):
    def _scene(self, model, wave, star_yx, comp_yx, shape=(80, 80)):
        yy, xx = np.indices(shape, dtype=np.float64)
        planes = [800.0 * evaluate_psf_model(model, float(w), yy - star_yx[0], xx - star_yx[1])
                  + 45.0 * evaluate_psf_model(model, float(w), yy - comp_yx[0], xx - comp_yx[1]) + 1.0
                  for w in wave]
        return np.asarray(planes, dtype=np.float64)

    def test_the_product_header_names_the_background(self):
        model = constant_model_doc()
        wave = np.linspace(6500.0, 6540.0, 3)
        star_yx, comp_yx = (40.0, 20.0), (40.0, 62.0)
        cube = self._scene(model, wave, star_yx, comp_yx)
        for background, bkgmode in (("shared", "psffit_plane"), ("decoupled_c0", "psffit_plane_decoupled_c0")):
            with self.subTest(background=background):
                prods = make_psffit_products(cube, wave, star_yx, comp_yx, model, run_id="synthetic",
                                             input_cube_path="cube.fits", variance_zyx=np.ones_like(cube),
                                             n_controls=0, background=background)
                self.assertEqual(prods.companion.header["BKGMODE"], bkgmode)
                np.testing.assert_allclose(prods.companion.flux, 45.0, rtol=1e-6)

    def test_c4_passes_the_knob_and_declares_it_in_its_qc(self):
        model = constant_model_doc()
        wave = np.linspace(6500.0, 6540.0, 3)
        star_yx, comp_yx = (40.0, 20.0), (40.0, 62.0)
        cube = self._scene(model, wave, star_yx, comp_yx).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_x03_c0"
            paths = stage_x03_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(cube[None], name="CUBES"),
                fits.ImageHDU(wave, name="WAVELENGTH"),
                fits.ImageHDU(np.ones((1,) + cube.shape, dtype=np.float32), name="STAT"),
            ]).writeto(paths["stage02_cube_fits"])
            paths["stage01c_qc_json"].write_text(json.dumps({
                "stage": "01c", "run_id": run_id,
                "primary": {"pos_yx": list(star_yx)}, "companion": {"pos_yx": list(comp_yx)},
            }), encoding="utf-8")
            paths["psf_model_json"].write_text(json.dumps(model), encoding="utf-8")
            base = {"run_id": run_id, "project_root": str(root), "x03_star_radius_px": 20.0,
                    "x03_comp_radius_px": 12.0, "x03_bad_windows_A": []}

            for background, n_par in (("shared", 5), ("decoupled_c0", 6)):
                with self.subTest(background=background):
                    product = compute_stage_x03_products({**base, "x03_background": background}, paths)
                    self.assertEqual(product.qc["fit_region"]["background"], background)
                    self.assertEqual(product.qc["fit_region"]["n_parameters"], n_par)
                    self.assertEqual(product.products.result.coeffs.shape[1], n_par)

            product = compute_stage_x03_products(base, paths)
            self.assertEqual(product.qc["fit_region"]["background"], "shared")


if __name__ == "__main__":
    unittest.main()
