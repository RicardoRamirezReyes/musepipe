import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.extraction.optimal import make_optimal_product
from tests.test_optimal_analytic import constant_model_doc, synthetic_psf_cube


class OptimalPsfBiasTests(unittest.TestCase):
    def test_fwhm_mismatch_bias_is_monotonic(self):
        true_model = constant_model_doc(fwhm=4.0)
        narrow_model = constant_model_doc(fwhm=3.6)
        wide_model = constant_model_doc(fwhm=4.4)
        wave = np.linspace(6500.0, 6900.0, 5)
        total_flux = np.full(wave.size, 100.0)
        center = (30.0, 30.0)
        cube = synthetic_psf_cube(true_model, wave, total_flux, center_yx=center)
        variance = np.ones_like(cube)

        with tempfile.TemporaryDirectory() as tmp:
            cube_path = Path(tmp) / "cube.fits"
            cube_path.write_bytes(b"synthetic")
            correct = make_optimal_product(
                cube,
                wave,
                center,
                true_model,
                run_id="synthetic",
                input_cube_path=cube_path,
                variance_zyx=variance,
                aperture_correction="psf_growth_curve",
                window_radius_px=4.0,
                clip_sigma=None,
            )
            narrow = make_optimal_product(
                cube,
                wave,
                center,
                narrow_model,
                run_id="synthetic",
                input_cube_path=cube_path,
                variance_zyx=variance,
                aperture_correction="psf_growth_curve",
                window_radius_px=4.0,
                clip_sigma=None,
            )
            wide = make_optimal_product(
                cube,
                wave,
                center,
                wide_model,
                run_id="synthetic",
                input_cube_path=cube_path,
                variance_zyx=variance,
                aperture_correction="psf_growth_curve",
                window_radius_px=4.0,
                clip_sigma=None,
            )

        correct_bias = float(np.nanmedian(correct.product.flux / total_flux - 1.0))
        narrow_bias = float(np.nanmedian(narrow.product.flux / total_flux - 1.0))
        wide_bias = float(np.nanmedian(wide.product.flux / total_flux - 1.0))
        self.assertAlmostEqual(correct_bias, 0.0, delta=1e-10)
        self.assertLess(narrow_bias, correct_bias)
        self.assertGreater(wide_bias, correct_bias)
        self.assertGreater(abs(narrow_bias), 0.005)
        self.assertGreater(abs(wide_bias), 0.005)


if __name__ == "__main__":
    unittest.main()
