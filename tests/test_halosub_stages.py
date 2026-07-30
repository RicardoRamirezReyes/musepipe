"""WP-H1 · Synthetic end-to-end tests for stages X04/C5 (SGF) and X05/C6 (LPM).

Scene: chromatic stellar halo (stellar spectrum x smooth spatial pattern x
mild chromatic modulation) + companion with a WIDE emission line (so the SGF
self-subtraction is unambiguous: R ~ 0.2). Checks per specs C5/C6 §7:

* both stages remove the halo (control residuals compatible with 0);
* SGF visibly self-subtracts the companion line, LPM preserves it within 5%;
* frozen-product contract (headers, controls, QC checks, files).
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.extraction.product import SpectrumProduct
from musepipe.psf import evaluate_psf_model
from musepipe.stages.stage_x04_sgf import (
    compute_stage_x04_products,
    stage_x04_paths,
    write_stage_x04_products,
)
from musepipe.stages.stage_x05_lpm import (
    compute_stage_x05_products,
    stage_x05_paths,
    write_stage_x05_products,
)
from musepipe.stages.halosub_stage import expected_scaleref
from tests.test_optimal_analytic import constant_model_doc

NZ = 161
WAVE = np.linspace(6400.0, 6700.0, NZ)
LINE_CENTER_A = 6563.0
LINE_FWHM_A = 40.0  # deliberately wide: R ~ (40/1.875)/101 ~ 0.21
PRIMARY_YX = (30.0, 30.0)
COMPANION_YX = (30.0, 44.0)
LINE_PEAK = 60.0


def stellar_spectrum():
    cont = 100.0 + 10.0 * (WAVE - WAVE[0]) / (WAVE[-1] - WAVE[0])
    sigma = 4.0 / 2.35482
    line = 25.0 * np.exp(-0.5 * ((WAVE - 6520.0) / sigma) ** 2)
    return cont + line


def companion_line_spectrum():
    sigma = LINE_FWHM_A / 2.35482
    return LINE_PEAK * np.exp(-0.5 * ((WAVE - LINE_CENTER_A) / sigma) ** 2)


def synthetic_scene(model, shape=(64, 64)):
    yy, xx = np.indices(shape, dtype=np.float64)
    halo = np.exp(-np.hypot(yy - PRIMARY_YX[0], xx - PRIMARY_YX[1]) / 8.0)
    x = 2.0 * (WAVE - WAVE[0]) / (WAVE[-1] - WAVE[0]) - 1.0
    chroma = 1.0 + 0.04 * x + 0.02 * x**2  # smooth: inside both model spaces
    star = stellar_spectrum()
    cube = (star * chroma)[:, None, None] * halo[None, :, :]
    comp = companion_line_spectrum()
    for i, w in enumerate(WAVE):
        psf = evaluate_psf_model(model, float(w), yy - COMPANION_YX[0], xx - COMPANION_YX[1])
        cube[i] += comp[i] * psf
    return cube.astype(np.float64)


def write_run_inputs(paths, cube, run_id):
    paths["paths"].ensure_base_dirs()
    stat = np.ones((1,) + cube.shape, dtype=np.float32)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(cube[None].astype(np.float32), name="CUBES"),
            fits.ImageHDU(WAVE, name="WAVELENGTH"),
            fits.ImageHDU(stat, name="STAT"),
        ]
    ).writeto(paths["stage02_cube_fits"])
    paths["stage01c_qc_json"].write_text(
        json.dumps(
            {
                "stage": "01c",
                "run_id": run_id,
                "primary": {"pos_yx": list(PRIMARY_YX)},
                "companion": {"pos_yx": list(COMPANION_YX)},
            }
        ),
        encoding="utf-8",
    )
    paths["psf_model_json"].write_text(json.dumps(constant_model_doc()), encoding="utf-8")


def line_flux(product):
    wave = np.asarray(product.wave_A)
    flux = np.asarray(product.flux)
    in_line = np.abs(wave - LINE_CENTER_A) <= 1.5 * LINE_FWHM_A
    return float(np.nansum(np.where(np.isfinite(flux), flux, 0.0)[in_line]))


def injected_line_flux():
    comp = companion_line_spectrum()
    in_line = np.abs(WAVE - LINE_CENTER_A) <= 1.5 * LINE_FWHM_A
    return float(np.sum(comp[in_line]))


BASE_CFG = {
    "halosub_exclude_radius_px": 4.0,
    "lsf_fwhm_A": 2.6,
}


class FluxConventionTests(unittest.TestCase):
    def test_expected_scaleref_tracks_global_and_stage_override(self):
        self.assertEqual(
            expected_scaleref({"flux_convention": "total"}, knob="x04_flux_convention"),
            "empirical_total_flux",
        )
        self.assertEqual(
            expected_scaleref(
                {"flux_convention": "total", "x04_flux_convention": "normrad"},
                knob="x04_flux_convention",
            ),
            "normrad_total_flux",
        )


class SgfStageTests(unittest.TestCase):
    def test_stage_x04_contract_and_self_subtraction(self):
        model = constant_model_doc()
        cube = synthetic_scene(model)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_x04"
            paths = stage_x04_paths(run_id, project_root=root)
            write_run_inputs(paths, cube, run_id)
            cfg = {
                "run_id": run_id,
                "project_root": str(root),
                "x04_aperture_correction": "psf_growth_curve",
                "x04_bad_windows_A": [],
                **BASE_CFG,
            }
            product = compute_stage_x04_products(cfg, paths)
            written = write_stage_x04_products(product, cfg, paths)

            self.assertTrue(paths["product_fits"].exists())
            self.assertTrue(paths["controls_npz"].exists())
            self.assertTrue(paths["residual_cube_fits"].exists())
            self.assertTrue(written["qc_json"].exists())

            prod = SpectrumProduct.read(paths["product_fits"])
            self.assertEqual(prod.header["METHOD"], "sgf")
            self.assertEqual(prod.header["SCALEREF"], "normrad_total_flux")
            self.assertTrue(str(prod.header["BKGMODE"]).startswith("sgf_residual"))

            qc = written["qc"]
            self.assertEqual(qc["spec_version"], "C5_v1")
            self.assertTrue(qc["checks"]["v1_reference_ok"])
            self.assertTrue(qc["checks"]["v3_predictor_written"])
            self.assertTrue(qc["checks"]["v4_scale_convention_ok"])
            self.assertFalse(qc["pca_applied"])

            # Halo removed: control residuals compatible with 0.
            controls = np.load(paths["controls_npz"])["control_spectra"]
            self.assertLess(abs(float(np.nanmedian(controls))), 0.5)

            # Self-subtraction: the smoothing conserves the ratio-bump area,
            # so a line this wide vs the window loses most of its integrated
            # flux inside the measurement window (paper Sect. 2.2) — visible
            # but not total obliteration.
            recovery = line_flux(prod) / injected_line_flux()
            self.assertLess(recovery, 0.9)
            self.assertGreater(recovery, 0.2)

            # Negative continuum around the line, as Eq. 1 predicts.
            wave = np.asarray(prod.wave_A)
            side = (np.abs(wave - LINE_CENTER_A) > 1.6 * LINE_FWHM_A) & (
                np.abs(wave - LINE_CENTER_A) <= 2.4 * LINE_FWHM_A
            )
            self.assertLess(float(np.nanmedian(np.asarray(prod.flux)[side])), 0.0)


class LpmStageTests(unittest.TestCase):
    def test_stage_x05_contract_and_line_preservation(self):
        model = constant_model_doc()
        cube = synthetic_scene(model)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_x05"
            paths = stage_x05_paths(run_id, project_root=root)
            write_run_inputs(paths, cube, run_id)
            cfg = {
                "run_id": run_id,
                "project_root": str(root),
                "x05_aperture_correction": "psf_growth_curve",
                "x05_bad_windows_A": [],
                "lpm_degree": 4,
                # Mask must cover the wide synthetic line (spec C6 §4).
                "lpm_masked_lines_A": [[LINE_CENTER_A, 2.0 * LINE_FWHM_A]],
                **BASE_CFG,
            }
            product = compute_stage_x05_products(cfg, paths)
            written = write_stage_x05_products(product, cfg, paths)

            self.assertTrue(paths["product_fits"].exists())
            self.assertTrue(paths["coeff_maps_fits"].exists())
            prod = SpectrumProduct.read(paths["product_fits"])
            self.assertEqual(prod.header["METHOD"], "lpm")
            self.assertTrue(str(prod.header["BKGMODE"]).startswith("lpm_residual"))

            qc = written["qc"]
            self.assertEqual(qc["spec_version"], "C6_v1.1")
            self.assertTrue(qc["checks"]["v1_reference_ok"])
            self.assertTrue(qc["checks"]["v3_condition_ok"])
            self.assertTrue(qc["checks"]["v4_slow_path_ok"])
            self.assertTrue(qc["checks"]["v5_scale_convention_ok"])
            self.assertTrue(qc["checks"]["v6_degree_diagnostics_written"])

            # Line preserved within 5% (spec C6 §7).
            recovery = line_flux(prod) / injected_line_flux()
            self.assertGreater(recovery, 0.95)
            self.assertLess(recovery, 1.05)

            # Degree diagnostics: star-underfit monotonically decreasing.
            curve = qc["degree_diagnostics"]["mse_curve"]
            unders = [row["star_underfit"] for row in curve]
            self.assertTrue(all(a >= b - 1e-9 for a, b in zip(unders, unders[1:])))
            self.assertEqual(len(qc["degree_diagnostics"]["energy_share"]), 9)

            # Internal smoke: masked-line injection at a control recovers >90%.
            self.assertTrue(qc["checks"]["v2_line_preservation_ok"])

            # Coefficient maps persisted with both fits.
            with fits.open(paths["coeff_maps_fits"]) as hdul:
                self.assertIn("COEFFS", hdul)
                self.assertIn("COEFFS_DEG9", hdul)
                self.assertEqual(hdul["COEFFS"].data.shape[0], 5)
                self.assertEqual(hdul["COEFFS_DEG9"].data.shape[0], 10)

    def test_stage_x05_beats_sgf_on_line_flux(self):
        model = constant_model_doc()
        cube = synthetic_scene(model)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_cfg = {"project_root": str(tmp), **BASE_CFG}
            paths4 = stage_x04_paths("syn_cmp", project_root=root)
            write_run_inputs(paths4, cube, "syn_cmp")
            paths5 = stage_x05_paths("syn_cmp", project_root=root)
            sgf = compute_stage_x04_products(
                {"run_id": "syn_cmp", "x04_aperture_correction": "psf_growth_curve",
                 "x04_bad_windows_A": [], **base_cfg},
                paths4,
            )
            lpm = compute_stage_x05_products(
                {"run_id": "syn_cmp", "x05_aperture_correction": "psf_growth_curve",
                 "x05_bad_windows_A": [],
                 "lpm_masked_lines_A": [[LINE_CENTER_A, 2.0 * LINE_FWHM_A]], **base_cfg},
                paths5,
            )
            self.assertGreater(line_flux(lpm.extraction.product), line_flux(sgf.extraction.product))


if __name__ == "__main__":
    unittest.main()
