"""The E4 fork ProcessPool backend must reproduce the serial grid exactly.

``compute_stage_h04_products`` runs each injection case as an independent pure
function of ``(case, shared read-only ctx)``. Distributing the cases over a fork
ProcessPool only changes which process runs which case; it must not change any
recovered row. This mirrors ``tests/test_parallel_equivalence.py`` for the
channel-loop threads, one level up at the case-grid granularity.
"""

import multiprocessing as mp
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.stages.stage_h01_detect import HALPHA_REST_A, matched_filter_point
from musepipe.stages.stage_h04_injection import (
    _resolve_h04_process_pool,
    compute_stage_h04_products,
    stage_h04_paths,
)
from tests.test_optimal_analytic import constant_model_doc


def _matched_sigma(wave, fwhm):
    err = np.full(wave.size, 0.2, dtype=np.float64)
    _flux, sigma, _z = matched_filter_point(
        wave, np.zeros(wave.size, dtype=np.float64), err, HALPHA_REST_A, fwhm,
        np.ones(wave.size, dtype=bool),
    )
    return sigma


def _synthetic_extractor(cube, wave_A, case, method, config):
    if cube.ndim == 4:
        cube = np.nanmean(cube, axis=0)
    flux = np.nansum(cube, axis=(1, 2))
    return {"wave_A": wave_A, "flux": flux, "flux_err": np.full(wave_A.size, 0.2, dtype=np.float64)}


def _synthetic_config(run_id, root, matched_sigma, **overrides):
    cfg = {
        "run_id": run_id,
        "project_root": str(root),
        "h04_positions_yx": [
            {"label": "real", "y": 30.0, "x": 30.0},
            {"label": "control1", "y": 30.0, "x": 38.0},
            {"label": "control2", "y": 38.0, "x": 30.0},
            {"label": "control3", "y": 30.0, "x": 22.0},
        ],
        "h04_methods": ["aperture", "psffit"],
        "h04_lsf_fwhm_A": 2.5,
        "h04_injection_flux_sigma": matched_sigma,
        "h04_continuum_flux_density": 0.0,
        "h04_expected_seconds_per_case_method": 0.01,
        "h04_require_historic_regression": False,
        "h04_historic_expected_snr": 8.97,
        "h04_historic_recovered_snr": 8.80,
        "h04_historic_tolerance_snr": 0.25,
    }
    cfg.update(overrides)
    return cfg


class ProcessPoolResolutionTests(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertEqual(_resolve_h04_process_pool({}, 100), 0)
        self.assertEqual(_resolve_h04_process_pool({"h04_process_pool": False}, 100), 0)

    def test_single_worker_is_disabled(self):
        self.assertEqual(_resolve_h04_process_pool({"h04_process_pool": 1}, 100), 0)

    def test_explicit_worker_count_capped_by_cases(self):
        self.assertEqual(_resolve_h04_process_pool({"h04_process_pool": 6}, 3), 3)
        self.assertEqual(_resolve_h04_process_pool({"h04_process_pool": 4}, 100), 4)


@unittest.skipUnless("fork" in mp.get_all_start_methods(), "fork start method unavailable")
class ProcessPoolDeterminismTests(unittest.TestCase):
    def _grid(self, process_pool):
        wave = np.arange(6525.0, 6601.0, 1.0, dtype=np.float64)
        fwhm = 2.5
        matched_sigma = _matched_sigma(wave, fwhm)
        cube = np.zeros((wave.size, 64, 64), dtype=np.float64)
        # A faint position-dependent gradient so recovered flux/SNR actually vary.
        yy, xx = np.mgrid[0:64, 0:64]
        cube += (0.001 * (yy + xx)).astype(np.float64)[None, :, :]
        model = constant_model_doc(fwhm=4.0)
        overrides = {} if process_pool is None else {"h04_process_pool": process_pool}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_h04_pool"
            paths = stage_h04_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            cfg = _synthetic_config(run_id, root, matched_sigma, **overrides)
            product = compute_stage_h04_products(
                cfg, paths,
                extractors={method: _synthetic_extractor for method in cfg["h04_methods"]},
                base_cube=cube, wavelengths_A=wave, psf_model=model,
            )
        return product.rows

    def test_process_pool_rows_identical_to_serial(self):
        serial = self._grid(None)
        pooled = self._grid(3)
        self.assertEqual(len(serial), len(pooled))
        self.assertTrue(serial)
        # _json_ready has already mapped non-finite values to None, so exact
        # dict equality is well defined (no NaN != NaN traps).
        self.assertEqual(serial, pooled)


if __name__ == "__main__":
    unittest.main()
