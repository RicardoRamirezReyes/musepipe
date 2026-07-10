import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.extraction.product import SpectrumProduct
from musepipe.stages.stage_x03_psffit import (
    compute_stage_x03_products,
    stage_x03_paths,
    write_stage_x03_products,
)
from tests.test_optimal_analytic import constant_model_doc
from tests.test_psffit_synthetic import synthetic_linear_scene


class PsfFitStageTests(unittest.TestCase):
    def test_stage_x03_writes_object_star_residual_and_qc(self):
        model = constant_model_doc()
        wave = np.linspace(6500.0, 6540.0, 5)
        star_yx = (36.0, 34.0)
        comp_yx = (36.0, 52.0)
        coeffs = np.column_stack(
            [
                np.full(wave.size, 800.0),
                np.full(wave.size, 45.0),
                np.full(wave.size, 1.0),
                np.zeros(wave.size),
                np.zeros(wave.size),
            ]
        )
        cube = synthetic_linear_scene(model, wave, star_yx, comp_yx, coeffs).astype(np.float32)
        stat = np.ones((1,) + cube.shape, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_x03"
            paths = stage_x03_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(cube[None], name="CUBES"),
                    fits.ImageHDU(wave, name="WAVELENGTH"),
                    fits.ImageHDU(stat, name="STAT"),
                ]
            ).writeto(paths["stage02_cube_fits"])
            paths["stage01c_qc_json"].write_text(
                json.dumps(
                    {
                        "stage": "01c",
                        "run_id": run_id,
                        "primary": {"pos_yx": list(star_yx)},
                        "companion": {"pos_yx": list(comp_yx)},
                    }
                ),
                encoding="utf-8",
            )
            paths["psf_model_json"].write_text(json.dumps(model), encoding="utf-8")
            cfg = {
                "run_id": run_id,
                "project_root": str(root),
                "x03_star_radius_px": 20.0,
                "x03_comp_radius_px": 12.0,
                "x03_bad_windows_A": [],
            }
            product = compute_stage_x03_products(cfg, paths)
            written = write_stage_x03_products(product, cfg, paths)

            self.assertTrue(paths["spec_psffit_object"].exists())
            self.assertTrue(paths["spec_psffit_star"].exists())
            self.assertTrue(paths["cube_psffit_residual"].exists())
            self.assertTrue(written["qc_json"].exists())
            obj = SpectrumProduct.read(paths["spec_psffit_object"])
            star = SpectrumProduct.read(paths["spec_psffit_star"])
            np.testing.assert_allclose(obj.flux, coeffs[:, 1], rtol=1e-5, atol=1e-4)
            np.testing.assert_allclose(star.flux, coeffs[:, 0], rtol=1e-5, atol=1e-4)
            self.assertEqual(written["qc"]["stage"], "x03_psffit")


if __name__ == "__main__":
    unittest.main()
