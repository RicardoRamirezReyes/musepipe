import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.extraction.optimal import make_optimal_product
from tests.test_optimal_analytic import constant_model_doc, synthetic_psf_cube


class OptimalNoLineClipTests(unittest.TestCase):
    def test_spatial_clipping_does_not_clip_real_emission_line_channel(self):
        model = constant_model_doc()
        wave = np.array([6500.0, 6510.0, 6520.0, 6530.0, 6540.0], dtype=np.float64)
        total_flux = np.array([20.0, 20.0, 1200.0, 20.0, 20.0], dtype=np.float64)
        center = (30.0, 30.0)
        cube = synthetic_psf_cube(model, wave, total_flux, center_yx=center)
        variance = np.full_like(cube, 0.01)

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
                window_radius_px=5.0,
                clip_sigma=2.0,
                clip_max_iter=2,
            )

        self.assertGreater(extraction.product.flux[2] / total_flux[2], 0.99)
        self.assertEqual(float(extraction.clip_fraction[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
