import unittest

import numpy as np

from musepipe.extraction.psffit import fit_psffit_cube, make_psffit_products, psf_pair_design
from musepipe.growth_curve import factor_at_wavelengths
from tests.test_optimal_analytic import constant_model_doc


def synthetic_linear_scene(model, wave, star_yx, comp_yx, coeffs, shape=(80, 80)):
    planes = []
    for z, w in enumerate(wave):
        design = psf_pair_design(shape, float(w), star_yx, comp_yx, model)
        planes.append(np.tensordot(design, coeffs[z], axes=([-1], [0])))
    return np.asarray(planes, dtype=np.float64)


class PsfFitSyntheticTests(unittest.TestCase):
    def test_two_sources_plane_and_covariance_are_recovered(self):
        model = constant_model_doc()
        wave = np.linspace(6500.0, 6540.0, 5)
        star_yx = (36.0, 34.0)
        comp_yx = (36.0, 52.0)
        coeffs = np.column_stack(
            [
                np.linspace(1000.0, 1100.0, wave.size),
                np.linspace(50.0, 70.0, wave.size),
                np.full(wave.size, 4.0),
                np.full(wave.size, 0.3),
                np.full(wave.size, -0.2),
            ]
        )
        cube = synthetic_linear_scene(model, wave, star_yx, comp_yx, coeffs)
        variance = np.full_like(cube, 2.0)

        fit = fit_psffit_cube(
            cube,
            variance,
            wave,
            star_yx,
            comp_yx,
            model,
            star_radius_px=20.0,
            comp_radius_px=12.0,
        )
        np.testing.assert_allclose(fit.coeffs, coeffs, rtol=1e-10, atol=1e-10)
        self.assertLess(float(np.nanmax(fit.chi2r)), 1e-20)
        self.assertTrue(np.all(np.isfinite(fit.covariance[:, 1, 1])))

        products = make_psffit_products(
            cube,
            wave,
            star_yx,
            comp_yx,
            model,
            run_id="synthetic",
            input_cube_path="cube.fits",
            variance_zyx=variance,
            n_controls=0,
        )
        np.testing.assert_allclose(products.companion.flux, coeffs[:, 1], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(products.star.flux, coeffs[:, 0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(products.companion.flux_err, np.sqrt(fit.covariance[:, 1, 1]))

    def test_empirical_total_factor_scales_flux_errors_and_controls_once(self):
        model = constant_model_doc()
        wave = np.linspace(6500.0, 6540.0, 5)
        star_yx = (36.0, 34.0)
        comp_yx = (36.0, 52.0)
        coeffs = np.column_stack(
            [
                np.linspace(1000.0, 1100.0, wave.size),
                np.linspace(50.0, 70.0, wave.size),
                np.full(wave.size, 4.0),
                np.full(wave.size, 0.3),
                np.full(wave.size, -0.2),
            ]
        )
        cube = synthetic_linear_scene(model, wave, star_yx, comp_yx, coeffs)
        variance = np.full_like(cube, 2.0)
        growth_curve = {
            "bands": [
                {"wave_A": 6000.0, "ratio_total_over_normrad": 2.4},
                {"wave_A": 7000.0, "ratio_total_over_normrad": 2.1},
                {"wave_A": 8000.0, "ratio_total_over_normrad": 1.9},
            ]
        }
        common = dict(
            run_id="synthetic",
            input_cube_path="cube.fits",
            variance_zyx=variance,
            n_controls=4,
        )
        base = make_psffit_products(cube, wave, star_yx, comp_yx, model, **common)
        total = make_psffit_products(
            cube, wave, star_yx, comp_yx, model, growth_curve=growth_curve, **common
        )
        factor = factor_at_wavelengths(growth_curve, wave)

        self.assertEqual(total.companion.header["SCALEREF"], "empirical_total_flux")
        np.testing.assert_allclose(total.companion.apcorr, factor)
        np.testing.assert_allclose(total.companion.flux, base.companion.flux * factor)
        np.testing.assert_allclose(total.star.flux, base.star.flux * factor)
        np.testing.assert_allclose(total.companion.flux_err, base.companion.flux_err * factor)
        np.testing.assert_allclose(total.companion.flux_err_emp, base.companion.flux_err_emp * factor)
        np.testing.assert_allclose(
            total.control_comp_spectra, base.control_comp_spectra * factor[None, :]
        )


if __name__ == "__main__":
    unittest.main()
