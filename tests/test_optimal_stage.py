import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.extraction.product import SpectrumProduct
from musepipe.psf import evaluate_psf_model
from musepipe.stages.stage_x02_optimal import (
    compute_stage_x02_products,
    stage_x02_paths,
    write_stage_x02_products,
)
from tests.test_optimal_analytic import constant_model_doc


def psf_cube(model, wave, center_yx, flux, shape=(64, 64)):
    yy, xx = np.indices(shape, dtype=np.float64)
    planes = []
    for w in wave:
        planes.append(float(flux) * evaluate_psf_model(model, float(w), yy - center_yx[0], xx - center_yx[1]))
    return np.asarray(planes, dtype=np.float32)


class OptimalStageTests(unittest.TestCase):
    def test_stage_x02_writes_ls_and_psfsub_products(self):
        model = constant_model_doc()
        wave = np.linspace(6500.0, 6540.0, 5)
        primary_yx = (30.0, 30.0)
        companion_yx = (30.0, 42.0)
        companion_flux = 80.0
        companion_cube = psf_cube(model, wave, companion_yx, companion_flux)
        primary_cube = psf_cube(model, wave, primary_yx, 1000.0)
        stage02_cube = primary_cube + companion_cube
        stat = np.ones((1,) + stage02_cube.shape, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_x02"
            paths = stage_x02_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            hdr = fits.Header()
            hdr["WMIN"] = float(wave[0])
            hdr["DW"] = float(np.nanmedian(np.diff(wave)))
            hdr["BUNIT"] = "native"
            fits.PrimaryHDU(companion_cube, header=hdr).writeto(paths["cube_residual_object"])
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(stage02_cube[None], name="CUBES"),
                    fits.ImageHDU(wave, name="WAVELENGTH"),
                    fits.ImageHDU(stat, name="STAT"),
                ]
            ).writeto(paths["stage02_cube_fits"])
            paths["stage01c_qc_json"].write_text(
                json.dumps(
                    {
                        "stage": "01c",
                        "run_id": run_id,
                        "primary": {"pos_yx": list(primary_yx)},
                        "companion": {"pos_yx": list(companion_yx)},
                    }
                ),
                encoding="utf-8",
            )
            paths["psf_model_json"].write_text(json.dumps(model), encoding="utf-8")
            cfg = {
                "run_id": run_id,
                "project_root": str(root),
                "x02_window_radius_px": 6.0,
                "x02_aperture_correction": "psf_growth_curve",
                "x02_bad_windows_A": [],
                "x02_primary_fit_radius_px": 18.0,
            }
            product = compute_stage_x02_products(cfg, paths)
            written = write_stage_x02_products(product, cfg, paths)

            self.assertTrue(paths["spec_optimal_object"].exists())
            self.assertTrue(paths["spec_optimal_psfsub_object"].exists())
            self.assertTrue(written["qc_json"].exists())
            ls = SpectrumProduct.read(paths["spec_optimal_object"])
            psfsub = SpectrumProduct.read(paths["spec_optimal_psfsub_object"])
            np.testing.assert_allclose(ls.flux, companion_flux, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(psfsub.flux, companion_flux, rtol=5e-2, atol=5e-2)
            self.assertEqual(written["qc"]["variants"], ["ls", "psfsub"])


if __name__ == "__main__":
    unittest.main()
