import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.extraction.aperture import make_aperture_product
from musepipe.extraction.optimal import make_optimal_product
from tests.test_optimal_analytic import constant_model_doc


class ApertureControlsLikeObjectTests(unittest.TestCase):
    def test_constant_background_is_removed_from_calibrated_controls(self):
        wave = np.array([6600.0, 6700.0, 6800.0], dtype=np.float64)
        ny = nx = 61
        level = 5.0
        cube = np.full((wave.size, ny, nx), level, dtype=np.float64)
        object_yx = (30.0, 45.0)
        star_yx = (30.0, 30.0)

        with tempfile.TemporaryDirectory() as tmp:
            cube_path = Path(tmp) / "cube.fits"
            cube_path.write_bytes(b"synthetic")
            extraction = make_aperture_product(
                cube,
                wave,
                object_yx,
                {"name": "box3", "kind": "box", "size": 3},
                run_id="synthetic",
                input_cube_path=cube_path,
                star_yx=star_yx,
                aperture_correction="none",
                n_controls=6,
                annulus_bkg_px=[4.0, 7.0, 3.0],
            )

        self.assertGreaterEqual(extraction.control_spectra.shape[0], 2)
        # Raw control sums keep the pedestal (box3 x level = 45)...
        np.testing.assert_allclose(extraction.control_spectra, 9.0 * level, rtol=1e-9)
        # ...calibrated controls are processed like the object: pedestal gone.
        np.testing.assert_allclose(extraction.control_spectra_cal, 0.0, atol=1e-9)
        self.assertEqual(extraction.bkg_mode, "annulus_4_7")
        self.assertEqual(extraction.product.header["BKGMODE"], "annulus_4_7")
        self.assertEqual(extraction.product.header["SCALEREF"], "normrad_total_flux")
        # The object was processed identically.
        np.testing.assert_allclose(extraction.product.flux, 0.0, atol=1e-9)

    def test_optimal_local_background_removes_pedestal_for_object_and_controls(self):
        model = constant_model_doc()
        wave = np.array([6600.0, 6700.0, 6800.0], dtype=np.float64)
        ny = nx = 61
        level = 5.0
        cube = np.full((wave.size, ny, nx), level, dtype=np.float64)
        object_yx = (30.0, 45.0)
        star_yx = (30.0, 30.0)
        variance = np.ones_like(cube)

        common = dict(
            run_id="synthetic",
            input_cube_path="synthetic",
            star_yx=star_yx,
            variance_zyx=variance,
            aperture_correction="none",
            window_radius_px=4.0,
            clip_sigma=None,
            n_controls=6,
        )
        without = make_optimal_product(cube, wave, object_yx, model, **common)
        with_bkg = make_optimal_product(
            cube, wave, object_yx, model, local_bkg_annulus_px=[4.0, 7.0, 3.0], **common
        )

        # Without the local annulus the pedestal leaks into object and controls...
        self.assertGreater(float(np.nanmin(np.abs(without.product.flux))), 1.0)
        self.assertGreater(float(np.nanmin(np.abs(without.control_spectra))), 1.0)
        # ...with it, both are referenced to the local background (~0).
        np.testing.assert_allclose(with_bkg.product.flux, 0.0, atol=1e-9)
        np.testing.assert_allclose(with_bkg.control_spectra, 0.0, atol=1e-9)
        np.testing.assert_allclose(with_bkg.control_spectra_cal, 0.0, atol=1e-9)
        self.assertEqual(with_bkg.product.header["BKGMODE"], "annulus_4_7")
        self.assertEqual(without.product.header["BKGMODE"], "none")


if __name__ == "__main__":
    unittest.main()
