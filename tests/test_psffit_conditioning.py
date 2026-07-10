import unittest

import numpy as np

from musepipe.extraction.psffit import fit_psffit_cube
from tests.test_optimal_analytic import constant_model_doc
from tests.test_psffit_synthetic import synthetic_linear_scene


class PsfFitConditioningTests(unittest.TestCase):
    def test_rho_ab_grows_as_sources_approach(self):
        model = constant_model_doc()
        wave = np.linspace(6500.0, 6520.0, 3)
        star_yx = (36.0, 36.0)
        coeffs = np.tile(np.array([1000.0, 50.0, 0.0, 0.0, 0.0]), (wave.size, 1))
        variance = None
        medians = []
        for separation in (22.0, 6.0):
            comp_yx = (36.0, 36.0 + separation)
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
            medians.append(float(np.nanmedian(np.abs(fit.rho_ab))))

        self.assertLess(medians[0], 0.1)
        self.assertGreater(medians[1], medians[0] * 5.0)


if __name__ == "__main__":
    unittest.main()
