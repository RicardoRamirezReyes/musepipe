import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.stages.stage01c_localize import compute_stage01c_products
from tests.test_stage01c_synthetic import synthetic_cube, write_stack


class Stage01cYXTests(unittest.TestCase):
    def test_asymmetric_source_reports_yx_not_xy(self):
        companion_yx = (30.0, 90.0)
        cube, waves, _, _, field_yx = synthetic_cube(companion_yx=companion_yx)
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "stage02.fits"
            write_stack(input_path, cube, waves)
            cfg = {
                "run_id": "synthetic_yx",
                "project_root": tmp,
                "stage01c_input_cube_fits": str(input_path),
                "pixel_scale_arcsec": 0.05,
                "stage01c_companion_approx_yx": companion_yx,
                "stage01c_field_source_approx_yx": field_yx,
                "stage01c_psf_fwhm_px": 4.0,
                "stage01c_search_radius_px": 5.0,
                "stage01c_validate_astrometry": False,
                "stage01c_validate_legacy": False,
            }
            product = compute_stage01c_products(cfg, input_path=input_path)

        measured = product.qc["companion"]["pos_yx"]
        self.assertLess(abs(measured[0] - 30.0), 0.2)
        self.assertLess(abs(measured[1] - 90.0), 0.2)
        self.assertGreater(abs(measured[0] - 90.0), 20.0)
        self.assertEqual(product.qc["companion"]["pos_xy"][0], measured[1])


if __name__ == "__main__":
    unittest.main()
