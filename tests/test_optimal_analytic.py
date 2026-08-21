import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.extraction.optimal import (
    circular_window_indices,
    make_optimal_product,
    normalized_psf_window,
)
from musepipe.psf import evaluate_psf_model


def constant_model_doc(fwhm=4.0):
    coeffs = {}
    for key, value in {
        "y0": 30.0,
        "x0": 30.0,
        "fwhm_maj": fwhm,
        "fwhm_min": 0.9 * fwhm,
        "theta_deg": 12.0,
        "beta": 2.8,
    }.items():
        coeffs[key] = {
            "model": "polynomial",
            "degree": 0,
            "coefficients": [value],
            "wave_ref_A": 6700.0,
            "wave_scale_A": 1000.0,
        }
    return {"form": "moffat", "norm_radius_px": 18.0, "coefficients": coeffs, "hybrid": False}


def synthetic_psf_cube(model, wave, total_flux, center_yx=(30.0, 30.0), shape=(64, 64)):
    yy, xx = np.indices(shape, dtype=np.float64)
    planes = []
    for w, flux in zip(wave, np.broadcast_to(total_flux, wave.shape)):
        psf = evaluate_psf_model(model, float(w), yy - center_yx[0], xx - center_yx[1])
        planes.append(float(flux) * psf)
    return np.asarray(planes, dtype=np.float64)


class OptimalAnalyticTests(unittest.TestCase):
    def test_flux_and_variance_match_horne_formula(self):
        model = constant_model_doc()
        wave = np.linspace(6500.0, 6900.0, 5)
        total_flux = np.linspace(100.0, 140.0, wave.size)
        center = (30.0, 30.0)
        window = 7.0
        cube = synthetic_psf_cube(model, wave, total_flux, center_yx=center)
        variance = np.full_like(cube, 4.0)

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
                window_radius_px=window,
                clip_sigma=None,
            )

        np.testing.assert_allclose(extraction.product.flux, total_flux, rtol=2e-12, atol=2e-12)

        ypix, xpix = circular_window_indices(cube.shape[1:], center, window)
        p = normalized_psf_window(model, wave[0], center, ypix, xpix)
        raw_var_expected = 1.0 / np.sum((p**2) / 4.0)
        err_expected = np.sqrt(raw_var_expected) * extraction.product.apcorr
        np.testing.assert_allclose(extraction.product.flux_err, err_expected, rtol=2e-12, atol=2e-12)
        self.assertEqual(extraction.error_mode, "stat")


    def test_cubesrc_is_stamped_only_when_the_caller_declares_it(self):
        # La procedencia del cubo viaja con el dato o no existe: D1 la lee del
        # header (`CUBESRC`) para aceptar un `INCUBESH` distinto. Un producto
        # que no tenga nada especial que decir no cambia de cabecera.
        model = constant_model_doc()
        wave = np.linspace(6500.0, 6900.0, 5)
        center = (30.0, 30.0)
        cube = synthetic_psf_cube(model, wave, np.full(wave.size, 100.0), center_yx=center)

        with tempfile.TemporaryDirectory() as tmp:
            cube_path = Path(tmp) / "cube.fits"
            cube_path.write_bytes(b"synthetic")
            common = dict(run_id="synthetic", input_cube_path=cube_path,
                          variance_zyx=np.full_like(cube, 4.0),
                          aperture_correction="psf_growth_curve",
                          window_radius_px=7.0, clip_sigma=None)
            plain = make_optimal_product(cube, wave, center, model, **common)
            declared = make_optimal_product(cube, wave, center, model,
                                            input_cube_source="C1b_perobs_subtract", **common)

        self.assertNotIn("CUBESRC", plain.product.header)
        self.assertEqual(declared.product.header["CUBESRC"], "C1b_perobs_subtract")


if __name__ == "__main__":
    unittest.main()
