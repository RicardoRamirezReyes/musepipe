import unittest

import numpy as np

from musepipe.extraction.psffit import fit_psffit_cube
from tests.test_optimal_analytic import constant_model_doc
from tests.test_psffit_synthetic import synthetic_linear_scene


class PsfFitBackgroundTests(unittest.TestCase):
    def test_strong_background_gradient_does_not_bias_companion_flux(self):
        model = constant_model_doc()
        wave = np.linspace(6500.0, 6540.0, 5)
        star_yx = (36.0, 34.0)
        comp_yx = (36.0, 52.0)
        coeffs = np.column_stack(
            [
                np.full(wave.size, 900.0),
                np.full(wave.size, 60.0),
                np.full(wave.size, 20.0),
                np.full(wave.size, 15.0),
                np.full(wave.size, -12.0),
            ]
        )
        cube = synthetic_linear_scene(model, wave, star_yx, comp_yx, coeffs)
        variance = np.ones_like(cube)
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
        np.testing.assert_allclose(fit.coeffs[:, 1], coeffs[:, 1], rtol=0.01, atol=0.01)
        self.assertTrue(np.all(np.isfinite(fit.rho_bc)))


if __name__ == "__main__":
    unittest.main()
