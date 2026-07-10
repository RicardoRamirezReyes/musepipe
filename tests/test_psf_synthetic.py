import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.psf import moffat_image
from musepipe.stages.stage_e01_psf import (
    compute_stage_e01_products,
    stage_e01_paths,
    write_stage_e01_products,
)


def synthetic_psf_cube():
    waves = np.linspace(6500.0, 6900.0, 24)
    ny, nx = 88, 88
    y0, x0 = 44.0, 43.5
    cube = []
    for wave in waves:
        fmaj = 4.0 + 0.0005 * (wave - 6500.0)
        fmin = 3.4 + 0.0003 * (wave - 6500.0)
        image = moffat_image(
            (ny, nx),
            y0,
            x0,
            fmaj,
            fmin,
            20.0,
            2.8,
            amplitude=1200.0,
            background=3.0,
        )
        cube.append(image)
    return np.asarray(cube, dtype=np.float32), waves


def write_inputs(root, cube, waves):
    stage_dir = root / "runs" / "synthetic_psf" / "stages"
    stage_dir.mkdir(parents=True)
    cube_path = stage_dir / "stage02_xcorr_cube_stack.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(cube[None].astype(np.float32), name="CUBES"),
            fits.ImageHDU(waves.astype(np.float64), name="WAVELENGTH"),
        ]
    ).writeto(cube_path)
    qc = {
        "stage": "01c_target_localization",
        "run_id": "synthetic_psf",
        "primary": {"pos_yx": [44.0, 43.5]},
        "companion": {"pos_yx": [44.0, 62.0]},
        "field_source": None,
        "chromatic": {"chromatic_centroid_needed": False},
        "psf": {"fwhm_px": 4.0},
    }
    qc_path = stage_dir / "stage01c_qc.json"
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    return cube_path, qc_path


class PSFSyntheticTests(unittest.TestCase):
    def test_stage_e01_recovers_synthetic_moffat_parameters(self):
        cube, waves = synthetic_psf_cube()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cube_path, qc_path = write_inputs(root, cube, waves)
            cfg = {
                "run_id": "synthetic_psf",
                "project_root": str(root),
                "stage_e01_input_cube_fits": str(cube_path),
                "stage_e01_positions_qc": str(qc_path),
                # Moffat parameter-recovery test: force the Moffat form so the run
                # stays fast/deterministic (auto would also run the Psfao fit).
                "e01_psf_form": "moffat",
                "psf_bin_A": 100.0,
                "psf_fit_radius_px": 25.0,
                "psf_norm_radius_px": 18.0,
                "psf_mask_radius_factor": 2.0,
                "psf_hybrid_threshold_pct": 1e6,
            }
            product = compute_stage_e01_products(cfg)
            paths = stage_e01_paths("synthetic_psf", project_root=root)
            written = write_stage_e01_products(product, cfg, paths)

            self.assertTrue(written["model_json"].exists())
            self.assertTrue(written["params_csv"].exists())
            self.assertLess(product.qc["normalization"]["roundtrip_error"], 5e-3)
            med_fmaj = float(np.nanmedian([row["fwhm_maj"] for row in product.fit_rows]))
            med_fmin = float(np.nanmedian([row["fwhm_min"] for row in product.fit_rows]))
            self.assertAlmostEqual(med_fmaj, 4.1, delta=0.25)
            self.assertAlmostEqual(med_fmin, 3.46, delta=0.25)
            self.assertEqual(product.qc["fit"]["form_chosen"], "moffat")


if __name__ == "__main__":
    unittest.main()
