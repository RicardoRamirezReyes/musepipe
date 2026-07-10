import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.extraction.aperture import FLAG_BAD_WINDOW, make_aperture_product
from musepipe.extraction.product import SpectrumProduct
from musepipe.stages.stage_x01_aperture import (
    compute_stage_x01_products,
    stage_x01_paths,
    write_stage_x01_products,
)


class ApertureExtractorTests(unittest.TestCase):
    def test_box3_flux_and_stat_error_match_analytic_sum(self):
        wave = 5000.0 + np.arange(5, dtype=np.float64)
        cube = np.zeros((wave.size, 30, 30), dtype=np.float64)
        object_yx = (10.0, 16.0)
        star_yx = (10.0, 10.0)
        expected_flux = np.array([9.0, 18.0, 27.0, 36.0, 45.0], dtype=np.float64)
        for i, value in enumerate(expected_flux):
            cube[i, 9:12, 15:18] = value / 9.0
        stat = np.full_like(cube, 4.0)

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
                stat_zyx=stat,
                stat_factor=1.5,
                covariance_factor=2.0,
                aperture_correction="none",
                bad_windows_A=[[5001.0, 5001.1]],
            )

        np.testing.assert_allclose(extraction.raw_flux, expected_flux)
        np.testing.assert_allclose(extraction.product.flux, expected_flux)
        np.testing.assert_allclose(extraction.product.flux_err, np.sqrt(9.0 * 4.0 * 1.5 * 2.0))
        self.assertEqual(extraction.error_mode, "stat")
        self.assertGreaterEqual(len(extraction.controls_yx), 4)
        self.assertTrue(extraction.product.flags[1] & FLAG_BAD_WINDOW)
        self.assertFalse(extraction.product.flags[0] & FLAG_BAD_WINDOW)

    def test_stage_x01_writes_standard_products_and_qc(self):
        wave = 6000.0 + np.arange(4, dtype=np.float64)
        cube = np.zeros((wave.size, 24, 24), dtype=np.float32)
        object_yx = (12.0, 17.0)
        star_yx = (12.0, 12.0)
        expected_flux = np.array([9.0, 18.0, 27.0, 36.0], dtype=np.float64)
        for i, value in enumerate(expected_flux):
            cube[i, 11:14, 16:19] = value / 9.0
        stat_stack = np.full((1,) + cube.shape, 1.0, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_x01"
            paths = stage_x01_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            hdr = fits.Header()
            hdr["WMIN"] = float(wave[0])
            hdr["DW"] = 1.0
            hdr["BUNIT"] = "native"
            fits.PrimaryHDU(cube, header=hdr).writeto(paths["cube_residual_object"])
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(np.zeros_like(stat_stack), name="CUBES"),
                    fits.ImageHDU(wave, name="WAVELENGTH"),
                    fits.ImageHDU(stat_stack, name="STAT"),
                ]
            ).writeto(paths["stage02_cube_fits"])
            paths["stage01c_qc_json"].write_text(
                (
                    '{"stage":"01c","run_id":"synthetic_x01",'
                    f'"primary":{{"pos_yx":[{star_yx[0]},{star_yx[1]}]}},'
                    f'"companion":{{"pos_yx":[{object_yx[0]},{object_yx[1]}]}}}}'
                ),
                encoding="utf-8",
            )
            cfg = {
                "run_id": run_id,
                "project_root": str(root),
                "x01_reuse_stage04b": True,
                "x01_aperture_correction": "none",
                "x01_bad_windows_A": [],
            }
            product = compute_stage_x01_products(cfg, paths)
            written = write_stage_x01_products(product, cfg, paths)

            self.assertTrue(paths["spec_aperture_object"].exists())
            self.assertTrue(paths["spec_aperture_box5"].exists())
            self.assertTrue(written["qc_json"].exists())
            loaded = SpectrumProduct.read(paths["spec_aperture_object"])
            np.testing.assert_allclose(loaded.flux, expected_flux)
            self.assertEqual(loaded.header["ERRMODE"], "stat")
            self.assertEqual(written["qc"]["checks"]["v3_roundtrip_ok"], True)


if __name__ == "__main__":
    unittest.main()
