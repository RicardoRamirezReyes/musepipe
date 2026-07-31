import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.stages.stage_h01_detect import HALPHA_REST_A, gaussian_flux_template, matched_filter_point
from musepipe.stages.stage_h04_injection import (
    compute_stage_h04_products,
    historic_regression_check,
    stage_h04_paths,
    write_stage_h04_products,
)
from tests.test_optimal_analytic import constant_model_doc


def synthetic_h04_config(run_id, root, matched_sigma):
    methods = ["aperture", "optimal_ls", "optimal_psfsub", "psffit"]
    null = np.linspace(-1.0, 1.0, 8).tolist()
    return {
        "run_id": run_id,
        "project_root": str(root),
        "h04_positions_yx": [
            {"label": "real", "y": 30.0, "x": 30.0},
            {"label": "control1", "y": 30.0, "x": 38.0},
            {"label": "control2", "y": 38.0, "x": 30.0},
            {"label": "control3", "y": 30.0, "x": 22.0},
        ],
        "h04_methods": methods,
        "h04_lsf_fwhm_A": 2.5,
        "h04_injection_flux_sigma": matched_sigma,
        "h04_continuum_flux_density": 1.0,
        "h04_empirical_null_reference": {
            method: {"n_controls": len(null), "by_factor": {"1": null, "2": null}}
            for method in methods
        },
        "h04_expected_seconds_per_case_method": 0.01,
        "h04_historic_expected_snr": 8.97,
        "h04_historic_recovered_snr": 8.80,
        "h04_historic_tolerance_snr": 0.25,
    }


def matched_sigma_for_wave(wave, fwhm):
    err = np.full(wave.size, 0.2, dtype=np.float64)
    _flux, sigma, _z = matched_filter_point(
        wave,
        np.zeros(wave.size, dtype=np.float64),
        err,
        HALPHA_REST_A,
        fwhm,
        np.ones(wave.size, dtype=bool),
    )
    return sigma


def synthetic_extractor(cube, wave_A, case, method, config):
    if cube.ndim == 4:
        cube = np.nanmean(cube, axis=0)
    flux = np.nansum(cube, axis=(1, 2))
    return {"wave_A": wave_A, "flux": flux, "flux_err": np.full(wave_A.size, 0.2, dtype=np.float64)}


class InjectionRegressionTests(unittest.TestCase):
    def test_historic_regression_passes_nominal_stage06_numbers(self):
        result = historic_regression_check(
            {
                "h04_historic_expected_snr": 8.97,
                "h04_historic_recovered_snr": 8.80,
                "h04_historic_tolerance_snr": 0.25,
            }
        )

        self.assertEqual(result["verdict"], "pass")
        self.assertAlmostEqual(result["expected_snr"], 8.97)
        self.assertAlmostEqual(result["recovered"], 8.80)

    def test_stage_h04_synthetic_grid_writes_e3_throughput_contract(self):
        wave = np.arange(6525.0, 6601.0, 1.0, dtype=np.float64)
        fwhm = 2.5
        matched_sigma = matched_sigma_for_wave(wave, fwhm)
        cube = np.zeros((wave.size, 64, 64), dtype=np.float64)
        model = constant_model_doc(fwhm=4.0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_h04"
            paths = stage_h04_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            cfg = synthetic_h04_config(run_id, root, matched_sigma)

            product = compute_stage_h04_products(
                cfg,
                paths,
                extractors={method: synthetic_extractor for method in cfg["h04_methods"]},
                base_cube=cube,
                wavelengths_A=wave,
                psf_model=model,
            )
            written = write_stage_h04_products(product, cfg, paths)

            self.assertTrue(paths["throughput_csv"].exists())
            self.assertTrue(paths["stage_h04_qc_json"].exists())
            self.assertEqual(written["qc"]["regression_historic"]["verdict"], "pass")
            self.assertEqual(written["qc"]["grid"]["n_injections"], 112)
            self.assertEqual(written["qc"]["throughput"]["psf_perturbation_pct"], 0.0)
            self.assertIn("psffit", written["qc"]["throughput"]["per_method_at_snr5"])

            with paths["throughput_csv"].open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            psffit_snr5 = [
                row
                for row in rows
                if row["method"] == "psffit"
                and row["variant"] == "nominal"
                and row["continuum_mode"] == "none"
                and math.isclose(float(row["input_snr"]), 5.0)
                and math.isclose(float(row["template_factor"]), 1.0)
            ]
            self.assertTrue(psffit_snr5)
            throughputs = np.asarray([float(row["throughput"]) for row in psffit_snr5], dtype=np.float64)
            np.testing.assert_allclose(throughputs, np.ones_like(throughputs), rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
