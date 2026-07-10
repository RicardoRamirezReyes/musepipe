import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.extraction.optimal import make_optimal_product
from tests.test_optimal_analytic import constant_model_doc, synthetic_psf_cube


class OptimalNormalizationTests(unittest.TestCase):
    def test_window_renormalization_plus_apcorr_recovers_total_flux(self):
        model = constant_model_doc()
        wave = np.array([6600.0, 6700.0, 6800.0], dtype=np.float64)
        total_flux = np.full(wave.size, 75.0, dtype=np.float64)
        center = (30.0, 30.0)
        cube = synthetic_psf_cube(model, wave, total_flux, center_yx=center)
        variance = np.ones_like(cube)

        with tempfile.TemporaryDirectory() as tmp:
            cube_path = Path(tmp) / "cube.fits"
            cube_path.write_bytes(b"synthetic")
            extraction = make_optimal_product(
                cube,
                wave,
                center,
                model,
                run_id="synthetic",
                input_cube_path=cube_path,
                variance_zyx=variance,
                aperture_correction="psf_growth_curve",
                window_radius_px=4.0,
                clip_sigma=None,
            )

        np.testing.assert_allclose(extraction.product.flux, total_flux, rtol=2e-12, atol=2e-12)
        self.assertEqual(extraction.apcorr_mode, "psf_growth_curve")
        self.assertTrue(np.all(extraction.product.apcorr > 1.05))
        self.assertGreater(float(np.nanmax(np.abs(extraction.raw_flux - total_flux))), 1.0)


if __name__ == "__main__":
    unittest.main()
