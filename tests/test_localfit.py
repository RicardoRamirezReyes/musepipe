import unittest

import numpy as np

from musepipe.localfit import (
    fit_fast_surface_coefficients,
    fit_local_surface_2d,
    local_surface_spectra_fast,
    subtract_local_surface_cube,
)


def make_plane(ny=11, nx=11, level=10.0, x_slope=0.5, y_slope=-0.25, center=(5, 5)):
    yy, xx = np.mgrid[:ny, :nx]
    cy, cx = center
    return level + x_slope * (xx - cx) + y_slope * (yy - cy)


class LocalFitTests(unittest.TestCase):
    def test_fit_local_surface_2d_recovers_synthetic_plane(self):
        image = make_plane()
        image[5, 5] += 100.0

        model, n_good = fit_local_surface_2d(
            image,
            yc=5,
            xc=5,
            fit_radius_px=5.0,
            mask_radius_px=1.5,
            model_kind="plane",
            sigma_clip=3.0,
            max_iter=3,
            min_fit_pixels=20,
        )

        expected = make_plane().astype(np.float32)
        local = np.isfinite(model)
        self.assertGreaterEqual(n_good, 20)
        np.testing.assert_allclose(model[local], expected[local], atol=1e-5)
        self.assertAlmostEqual(image[5, 5] - model[5, 5], 100.0, places=5)

    def test_fit_local_surface_2d_constant_model(self):
        image = np.full((9, 9), 7.0, dtype=float)
        image[4, 4] += 20.0

        model, n_good = fit_local_surface_2d(
            image,
            yc=4,
            xc=4,
            fit_radius_px=4.0,
            mask_radius_px=1.0,
            model_kind="constant",
            min_fit_pixels=10,
        )

        self.assertGreaterEqual(n_good, 10)
        self.assertAlmostEqual(float(model[4, 4]), 7.0)

    def test_fit_local_surface_2d_returns_nan_model_when_too_few_pixels(self):
        image = np.ones((5, 5), dtype=float)
        model, n_good = fit_local_surface_2d(
            image,
            yc=2,
            xc=2,
            fit_radius_px=1.0,
            mask_radius_px=0.5,
            min_fit_pixels=50,
        )
        self.assertLess(n_good, 50)
        self.assertTrue(np.all(np.isnan(model)))

    def test_subtract_local_surface_cube_preserves_source_residual(self):
        plane = make_plane().astype(np.float32)
        cube = np.stack([plane.copy(), 2.0 * plane], axis=0)
        cube[0, 5, 5] += 12.0
        cube[1, 5, 5] += 30.0

        residual, model, nfit = subtract_local_surface_cube(
            cube,
            target_yx=(5, 5),
            fit_radius_px=5.0,
            mask_radius_px=1.5,
            model_kind="plane",
            min_fit_pixels=20,
        )

        self.assertEqual(residual.shape, cube.shape)
        self.assertEqual(model.shape, cube.shape)
        self.assertTrue(np.all(nfit >= 20))
        self.assertAlmostEqual(float(residual[0, 5, 5]), 12.0, places=5)
        self.assertAlmostEqual(float(residual[1, 5, 5]), 30.0, places=5)

    def test_fit_fast_surface_coefficients(self):
        a_fit = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        y_fit = np.array([[2.0, 4.0, 6.0], [5.0, np.nan, 9.0]])

        coeffs = fit_fast_surface_coefficients(y_fit, a_fit, model_kind="plane", min_pixels=2)
        np.testing.assert_allclose(coeffs[0], [2.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(coeffs[1], [5.0, 2.0], atol=1e-12)

        const = fit_fast_surface_coefficients(y_fit, a_fit, model_kind="constant", min_pixels=2)
        np.testing.assert_allclose(const[:, 0], [4.0, 7.0])

    def test_local_surface_spectra_fast_recovers_point_source_flux(self):
        plane = make_plane(ny=13, nx=13, center=(6, 6)).astype(float)
        cube = np.stack([plane.copy(), plane.copy(), plane.copy()], axis=0)
        cube[0, 6, 6] += 5.0
        cube[2, 6, 6] += 7.0

        apertures = [
            {"name": "pixel", "kind": "pixel"},
            {"name": "box3_sum", "kind": "box", "size": 3},
        ]
        spectra, meta, fit_meta = local_surface_spectra_fast(
            cube,
            center_yx=(6, 6),
            apertures=apertures,
            wave_indices=[0, 2],
            fit_radius_px=5.0,
            mask_radius_px=1.5,
            model_kind="plane",
            min_pixels=20,
        )

        np.testing.assert_allclose(spectra["pixel"][[0, 2]], [5.0, 7.0], atol=1e-10)
        self.assertTrue(np.isnan(spectra["pixel"][1]))
        np.testing.assert_allclose(spectra["box3_sum"][[0, 2]], [5.0, 7.0], atol=1e-10)
        self.assertEqual(meta["box3_sum"]["n_pix"], 9)
        self.assertGreaterEqual(fit_meta["n_fit_pixels"], 20)
        self.assertEqual(fit_meta["n_good_wavelengths_fit"], 2)


if __name__ == "__main__":
    unittest.main()
