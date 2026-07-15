"""WP-H5 · E5 contrast curves (spec E5 §4)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.halosub import lpm_subtract, reference_spectrum, select_reference_spaxels
from musepipe.stages.stage_h05_contrast import (
    compute_stage_h05_products,
    contrast_curve_from_rows,
    stage_h05_paths,
    stellar_line_flux,
    write_stage_h05_products,
)
from tests.test_optimal_analytic import constant_model_doc

NZ = 201
WAVE = np.linspace(6400.0, 6700.0, NZ)
LINE_A = 6562.8
LSF_FWHM_A = 2.6
STAR_YX = (32.0, 32.0)


def scene(rng, noise_sigma=1.0):
    """Mother cube: bright star PSF x flat spectrum + noise; residual = noise."""

    from musepipe.psf import evaluate_psf_model

    model = constant_model_doc()
    yy, xx = np.indices((64, 64), dtype=np.float64)
    psf = evaluate_psf_model(model, LINE_A, yy - STAR_YX[0], xx - STAR_YX[1])
    star_spec = np.full(NZ, 5000.0)
    pre = star_spec[:, None, None] * psf[None, :, :] + 100.0 * np.exp(
        -np.hypot(yy - STAR_YX[0], xx - STAR_YX[1])[None, :, :] / 8.0
    )
    residual = rng.normal(0.0, noise_sigma, size=(NZ, 64, 64))
    return pre, residual, model


class ContrastCurveTests(unittest.TestCase):
    def test_curve_reflects_grid_and_fractions(self):
        rows = []
        for sep in (4.0, 8.0):
            for angle in range(8):
                for c in (1e-5, 1e-4, 1e-3):
                    detected = c >= (1e-4 if sep == 8.0 else 1e-3)
                    rows.append({"method": "lpm", "separation_px": sep, "contrast": c, "detected": detected})
        curves = contrast_curve_from_rows(rows, (1e-5, 1e-4, 1e-3))
        by_sep = {e["separation_px"]: e for e in curves["lpm"]}
        self.assertEqual(by_sep[8.0]["contrast_50"], 1e-4)
        self.assertEqual(by_sep[4.0]["contrast_50"], 1e-3)

    def test_stage_end_to_end_and_delta_matches_bruteforce(self):
        rng = np.random.default_rng(5)
        pre, residual, model = scene(rng)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_h05"
            paths = stage_h05_paths(run_id, project_root=root)
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
                "h05_contrasts": [1e-4, 1e-3, 1e-2, 1e-1],
                "h05_n_angles": 4,
                "h05_r_min_px": 6.0,
                "h05_r_step_px": 8.0,
            }
            product = compute_stage_h05_products(cfg, paths)
            written = write_stage_h05_products(product, cfg, paths)

            qc = written["qc"]
            self.assertEqual(qc["spec_version"], "E5_v1")
            self.assertEqual(set(qc["methods"]), {"sgf", "lpm"})
            self.assertTrue(qc["checks"]["v1_curves_written"])
            self.assertTrue(paths["contrast_curve_csv"].exists())
            self.assertTrue(paths["contrast_injections_csv"].exists())
            self.assertTrue(paths["summary_plot"].exists())

            # Flat noise: every separation detects at SOME grid contrast, and
            # detectability improves (or stays) with contrast.
            for method, mq in qc["methods"].items():
                self.assertTrue(mq["curve"], method)
                for entry in mq["curve"]:
                    self.assertIsNotNone(entry["contrast_50"], f"{method}@{entry['separation_px']}")

            # Delta-path exactness: rebuild one injected case brute force and
            # compare the detection z within numerical tolerance.
            rows = [r for r in product.rows if r["method"] == "lpm" and r["detected"]]
            self.assertTrue(rows)
            row = rows[0]
            keep, _ = select_reference_spaxels(pre, exclude_yx=None)
            s_hat = reference_spectrum(pre, keep)
            from musepipe.injection import InjectionSource, inject
            from musepipe.stages.stage_h01b_fovmap import (
                crosscorr_map,
                psf_kernel_from_model,
                ring_noise_stats,
                spectral_map,
                spectral_template,
            )

            source = InjectionSource(
                y=row["y"], x=row["x"], total_line_flux=row["injected_line_flux"],
                line_center_A=LINE_A, line_fwhm_A=LSF_FWHM_A, label="bf",
            )
            injected_delta = inject(np.zeros_like(residual), [source], wavelengths_A=WAVE, psf_model=model)
            # The residual file IS the already-subtracted cube; injecting into
            # the data adds (I - L)(delta) to it (linearity, same s_hat).
            brute = residual + lpm_subtract(injected_delta, WAVE, s_hat, degree=4).residual_cube
            template = spectral_template(WAVE, LINE_A, LSF_FWHM_A)
            kernel = psf_kernel_from_model(model, LINE_A, 7)
            corr_b = crosscorr_map(spectral_map(brute, template), kernel)
            corr_0 = crosscorr_map(spectral_map(residual, template), kernel)
            z_b, rings_b = ring_noise_stats(corr_0, STAR_YX)
            y0, x0 = int(round(row["y"])), int(round(row["x"]))
            r = float(np.hypot(y0 - STAR_YX[0], x0 - STAR_YX[1]))
            ring = next(rg for rg in rings_b if rg["r_lo"] <= r < rg["r_hi"])
            z_bruteforce = (corr_b[y0, x0] - ring["mu"]) / ring["sigma"]
            self.assertAlmostEqual(z_bruteforce, row["z"], delta=0.15)


if __name__ == "__main__":
    unittest.main()
