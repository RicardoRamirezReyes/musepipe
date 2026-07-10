import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.extraction.aperture import make_aperture_product
from musepipe.psf import evaluate_psf_model


def constant_model_doc():
    coeffs = {}
    for key, value in {
        "y0": 40.0,
        "x0": 40.0,
        "fwhm_maj": 4.0,
        "fwhm_min": 3.6,
        "theta_deg": 10.0,
        "beta": 2.7,
    }.items():
        coeffs[key] = {
            "model": "polynomial",
            "degree": 0,
            "coefficients": [value],
            "wave_ref_A": 7000.0,
            "wave_scale_A": 1000.0,
        }
    return {"form": "moffat", "norm_radius_px": 18.0, "coefficients": coeffs, "hybrid": False}


class ApertureCorrectionTests(unittest.TestCase):
    def test_psf_growth_curve_makes_box3_and_box5_recover_total_flux(self):
        model = constant_model_doc()
        wave = np.linspace(6500.0, 6900.0, 6)
        total_flux = 123.4
        object_yx = (40.0, 40.0)
        yy, xx = np.indices((82, 82), dtype=np.float64)
        cube = []
        for w in wave:
            psf = evaluate_psf_model(model, float(w), yy - object_yx[0], xx - object_yx[1])
            cube.append(total_flux * psf)
        cube = np.asarray(cube, dtype=np.float64)

        with tempfile.TemporaryDirectory() as tmp:
            cube_path = Path(tmp) / "cube.fits"
            cube_path.write_bytes(b"synthetic")
            box3 = make_aperture_product(
                cube,
                wave,
                object_yx,
                {"name": "box3", "kind": "box", "size": 3},
                run_id="synthetic",
                input_cube_path=cube_path,
                psf_model=model,
                aperture_correction="psf_growth_curve",
            )
            box5 = make_aperture_product(
                cube,
                wave,
                object_yx,
                {"name": "box5", "kind": "box", "size": 5},
                run_id="synthetic",
                input_cube_path=cube_path,
                psf_model=model,
                aperture_correction="psf_growth_curve",
            )

        np.testing.assert_allclose(box3.product.flux, total_flux, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(box5.product.flux, total_flux, rtol=1e-10, atol=1e-10)
        self.assertTrue(np.all(box3.product.apcorr > box5.product.apcorr))
        self.assertEqual(box3.apcorr_mode, "psf_growth_curve")


if __name__ == "__main__":
    unittest.main()
