import unittest

import numpy as np

from musepipe.extraction.psffit import crosstalk_metric, make_psffit_products
from tests.test_optimal_analytic import constant_model_doc
from tests.test_psffit_synthetic import synthetic_linear_scene


class PsfFitCrosstalkTests(unittest.TestCase):
    def test_star_only_line_with_wrong_psf_is_detected_as_crosstalk(self):
        true_model = constant_model_doc(fwhm=4.0)
        wide_model = constant_model_doc(fwhm=4.4)
        wave = np.linspace(6500.0, 6600.0, 21)
        star_line = 1000.0 + 700.0 * np.exp(-0.5 * ((wave - 6550.0) / 6.0) ** 2)
        coeffs = np.column_stack(
            [
                star_line,
                np.zeros(wave.size),
                np.zeros(wave.size),
                np.zeros(wave.size),
                np.zeros(wave.size),
            ]
        )
        star_yx = (36.0, 34.0)
        comp_yx = (36.0, 46.0)
        cube = synthetic_linear_scene(true_model, wave, star_yx, comp_yx, coeffs)
        variance = np.ones_like(cube)
        products = make_psffit_products(
            cube,
            wave,
            star_yx,
            comp_yx,
            wide_model,
            run_id="synthetic",
            input_cube_path="cube.fits",
            variance_zyx=variance,
            star_radius_px=20.0,
            comp_radius_px=12.0,
            n_controls=0,
        )
        metric = crosstalk_metric(wave, products.star.flux, products.companion.flux, [[6525.0, 6575.0]])
        self.assertIsNotNone(metric["metric_corr_b_vs_a_lines"])
        self.assertGreater(abs(metric["metric_corr_b_vs_a_lines"]), 0.8)
        self.assertGreater(float(np.nanmax(np.abs(products.companion.flux))), 1.0)


if __name__ == "__main__":
    unittest.main()
