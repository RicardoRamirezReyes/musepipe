import unittest

import numpy as np

from musepipe.psf import evaluate_psf_model, fixed_radius_grid, psf_roundtrip_error


def constant_model_doc():
    coeffs = {}
    for key, value in {
        "y0": 45.0,
        "x0": 45.0,
        "fwhm_maj": 4.0,
        "fwhm_min": 3.5,
        "theta_deg": 15.0,
        "beta": 2.5,
    }.items():
        coeffs[key] = {
            "model": "polynomial",
            "degree": 0,
            "coefficients": [value],
            "wave_ref_A": 7000.0,
            "wave_scale_A": 1000.0,
        }
    return {"form": "moffat", "norm_radius_px": 18.0, "coefficients": coeffs, "hybrid": False}


class PSFNormalizationTests(unittest.TestCase):
    def test_psf_model_integrates_to_one_inside_norm_radius(self):
        model = constant_model_doc()
        yy, xx, mask = fixed_radius_grid(model["norm_radius_px"])
        for wave in np.linspace(6200.0, 8800.0, 10):
            psf = evaluate_psf_model(model, wave, yy, xx)
            self.assertAlmostEqual(float(np.sum(psf[mask])), 1.0, delta=5e-3)

    def test_roundtrip_error_reports_small_error(self):
        model = constant_model_doc()
        err = psf_roundtrip_error(model, np.linspace(6200.0, 8800.0, 10))
        self.assertLess(err, 5e-3)


if __name__ == "__main__":
    unittest.main()
