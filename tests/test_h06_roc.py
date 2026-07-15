"""WP-H6 · E6 ROC curves (spec E6 §4)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.stages.stage_h06_roc import (
    compute_stage_h06_products,
    roc_curve,
    stage_h06_paths,
    write_stage_h06_products,
)
from tests.test_h05_contrast import LSF_FWHM_A, NZ, STAR_YX, WAVE, scene


class RocCoreTests(unittest.TestCase):
    def test_roc_curve_separated_and_overlapping_samples(self):
        rng = np.random.default_rng(2)
        null = rng.normal(0.0, 1.0, 4000)
        strong = rng.normal(8.0, 1.0, 200)
        weak = rng.normal(0.0, 1.0, 200)
        _t, dp, fap, auc_strong = roc_curve(strong, null)
        self.assertGreater(auc_strong, 0.99)
        # DP and FAP are non-increasing as the threshold rises (rows are
        # ordered by decreasing threshold -> non-decreasing arrays).
        self.assertTrue(np.all(np.diff(dp) >= -1e-12))
        self.assertTrue(np.all(np.diff(fap) >= -1e-12))
        _t, _dp, _fap, auc_weak = roc_curve(weak, null)
        self.assertAlmostEqual(auc_weak, 0.5, delta=0.1)


class RocStageTests(unittest.TestCase):
    def test_stage_end_to_end(self):
        rng = np.random.default_rng(9)
        pre, residual, model = scene(rng)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_h06"
            paths = stage_h06_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(pre[None].astype(np.float32), name="CUBES"),
                    fits.ImageHDU(WAVE, name="WAVELENGTH"),
                ]
            ).writeto(paths["stage02_cube_fits"])
            for key in ("sgf_residual_cube", "lpm_residual_cube"):
                fits.HDUList(
                    [
                        fits.PrimaryHDU(),
                        fits.ImageHDU(residual.astype(np.float32), name="RESIDUAL"),
                        fits.ImageHDU(WAVE, name="WAVELENGTH"),
                    ]
                ).writeto(paths[key])
            paths["psf_model_json"].write_text(json.dumps(model), encoding="utf-8")
            paths["stage01c_qc_json"].write_text(
                json.dumps({"primary": {"pos_yx": list(STAR_YX)}}), encoding="utf-8"
            )
            cfg = {
                "run_id": run_id,
                "project_root": str(root),
                "lsf_fwhm_A": LSF_FWHM_A,
                "h06_separations_px": [10.0, 18.0],
                "h06_n_angles": 12,
                # Strong contrast: the detector must be near-perfect.
                "h06_contrast": 5e-2,
                "h06_null_step_channels": 10,
            }
            product = compute_stage_h06_products(cfg, paths)
            written = write_stage_h06_products(product, cfg, paths)

            qc = written["qc"]
            self.assertEqual(qc["spec_version"], "E6_v1")
            self.assertEqual(set(qc["methods"]), {"sgf", "lpm"})
            self.assertTrue(qc["checks"]["v1_curves_written"])
            self.assertTrue(qc["checks"]["v2_auc_above_random"])
            self.assertTrue(paths["roc_csv"].exists())
            self.assertTrue(paths["summary_plot"].exists())
            for method, mq in qc["methods"].items():
                for scen in mq["scenarios"]:
                    self.assertGreater(scen["auc"], 0.9, f"{method}@{scen['separation_px']}")
                    self.assertGreaterEqual(scen["n_injections"], 8)
                    self.assertGreater(scen["n_null"], 100)


if __name__ == "__main__":
    unittest.main()
