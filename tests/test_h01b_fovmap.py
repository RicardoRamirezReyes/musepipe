"""WP-H4 · E1b blind spatio-spectral matched-filter maps (spec E1b §4)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.stages.stage_h01b_fovmap import (
    compute_stage_h01b_products,
    crosscorr_map,
    find_candidates,
    psf_kernel_from_model,
    ring_noise_stats,
    spectral_map,
    spectral_template,
    stage_h01b_paths,
    write_stage_h01b_products,
)
from tests.test_optimal_analytic import constant_model_doc

NZ = 201
WAVE = np.linspace(6400.0, 6700.0, NZ)
LINE_A = 6562.8
LSF_FWHM_A = 2.6
STAR_YX = (32.0, 32.0)
SOURCE_YX = (32, 46)


def noise_cube(rng, sigma=1.0, shape=(64, 64)):
    return rng.normal(0.0, sigma, size=(NZ,) + shape)


def add_point_line(cube, model, y, x, peak):
    sigma_A = LSF_FWHM_A / 2.35482
    line = peak * np.exp(-0.5 * ((WAVE - LINE_A) / sigma_A) ** 2)
    yy, xx = np.indices(cube.shape[1:], dtype=np.float64)
    from musepipe.psf import evaluate_psf_model

    psf = evaluate_psf_model(model, LINE_A, yy - y, xx - x)
    return cube + line[:, None, None] * psf[None, :, :]


class FovMapCoreTests(unittest.TestCase):
    def test_injected_source_recovered_and_pure_noise_clean(self):
        rng = np.random.default_rng(11)
        model = constant_model_doc()
        template = spectral_template(WAVE, LINE_A, LSF_FWHM_A)
        kernel = psf_kernel_from_model(model, LINE_A, 7)

        pure = noise_cube(rng)
        injected = add_point_line(noise_cube(rng), model, SOURCE_YX[0], SOURCE_YX[1], peak=80.0)

        for cube, expect_hit in ((injected, True), (pure, False)):
            m = spectral_map(cube, template)
            corr = crosscorr_map(m, kernel)
            zmap, rings = ring_noise_stats(corr, STAR_YX)
            cands = find_candidates(zmap, STAR_YX, threshold_sigma=5.0, r_min_px=3.0)
            if expect_hit:
                self.assertTrue(cands, "expected a candidate for the injected source")
                best = cands[0]
                self.assertLessEqual(np.hypot(best["y"] - SOURCE_YX[0], best["x"] - SOURCE_YX[1]), 1.5)
                self.assertGreaterEqual(best["z"], 5.0)
            else:
                self.assertEqual(cands, [])
            self.assertTrue(rings)

    def test_ring_normalization_tracks_radial_noise(self):
        rng = np.random.default_rng(7)
        ny = nx = 64
        yy, xx = np.indices((ny, nx), dtype=np.float64)
        rr = np.hypot(yy - STAR_YX[0], xx - STAR_YX[1])
        sigma_map = 1.0 + 4.0 * np.exp(-rr / 10.0)  # louder near the core
        corr = rng.normal(0.0, 1.0, size=(ny, nx)) * sigma_map
        zmap, rings = ring_noise_stats(corr, STAR_YX, ring_width_px=2)
        sigmas = [r["sigma"] for r in rings[:5]]
        self.assertGreater(sigmas[0], sigmas[-1])  # measured radial trend
        finite = zmap[np.isfinite(zmap)]
        # After per-ring normalization the z is calibrated: |z|>3 ~ 0.27%.
        self.assertLess(float(np.mean(np.abs(finite) > 3.0)), 0.02)


class FovMapStageTests(unittest.TestCase):
    def test_stage_writes_maps_qc_and_reports_known_source(self):
        rng = np.random.default_rng(3)
        model = constant_model_doc()
        residual = add_point_line(noise_cube(rng), model, SOURCE_YX[0], SOURCE_YX[1], peak=40.0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_h01b"
            paths = stage_h01b_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
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
                json.dumps(
                    {
                        "primary": {"pos_yx": list(STAR_YX)},
                        # Known companion elsewhere: the injected source must
                        # still surface as a NEW candidate.
                        "companion": {"pos_yx": [20.0, 20.0]},
                    }
                ),
                encoding="utf-8",
            )
            cfg = {
                "run_id": run_id,
                "project_root": str(root),
                "h01b_include_psfsub": False,
                "lsf_fwhm_A": LSF_FWHM_A,
            }
            product = compute_stage_h01b_products(cfg, paths)
            written = write_stage_h01b_products(product, cfg, paths)

            qc = written["qc"]
            self.assertEqual(qc["spec_version"], "E1b_v1")
            self.assertEqual(set(qc["methods"]), {"sgf", "lpm"})
            for method, mq in qc["methods"].items():
                self.assertTrue(Path(mq["map_fits"]).exists())
                self.assertTrue(mq["candidates"], f"no candidate for {method}")
                best = mq["candidates"][0]
                self.assertLessEqual(np.hypot(best["y"] - SOURCE_YX[0], best["x"] - SOURCE_YX[1]), 1.5)
                self.assertIsNotNone(mq["known_source"])
                self.assertTrue(mq["ring_noise"])
            self.assertTrue(qc["checks"]["v1_maps_written"])
            self.assertTrue(qc["checks"]["v2_known_source_reported"])
            self.assertTrue(qc["checks"]["v3_ring_noise_written"])
            self.assertTrue(qc["checks"]["v4_no_candidate_at_star_core"])
            self.assertTrue(paths["summary_plot"].exists())


if __name__ == "__main__":
    unittest.main()
