import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from musepipe.stages.stage_e01_psf import (
    StageE01Product,
    _b3_chromatic_track,
    _centroid_vs_b3,
    _ring_qc_summary,
    _write_psfao_csv,
    _write_summary_plot,
)
from musepipe.stages.stage_e01_psfao import DEFAULT_X0, fit_psfao_bins


class PsfaoConvergenceTests(unittest.TestCase):
    def test_initial_vector_stall_is_retained_but_excluded(self):
        optimizer = {
            "success": True,
            "status": 1,
            "message": "converged without moving",
            "nfev": 1,
            "cost": 12.0,
            "stalled_at_initial": True,
        }
        fit_result = (DEFAULT_X0, 1.0, 0.0, (0.0, 0.0), 8.0, np.ones((8, 8)), optimizer)
        bins = [(6500.0, 6600.0, 6550.0, np.ones(4, dtype=bool))]
        cube = np.ones((4, 8, 8), dtype=float)

        with mock.patch("musepipe.stages.stage_e01_psfao.fit_bin", return_value=fit_result):
            rows, recons = fit_psfao_bins(
                cube, cube, np.arange(4), bins, object(), (4.0, 6.0), 2.0, 3.0
            )

        self.assertEqual(rows[0]["status"], "fit_stalled:initial_vector")
        self.assertTrue(rows[0]["optimizer_stalled_at_initial"])
        self.assertEqual(rows[0]["optimizer_nfev"], 1)
        self.assertEqual(recons, {})

    def test_optimizer_failure_is_excluded(self):
        optimizer = {
            "success": False,
            "status": 0,
            "message": "maximum evaluations",
            "nfev": 400,
            "cost": 20.0,
            "stalled_at_initial": False,
        }
        fit_result = (DEFAULT_X0, 1.0, 0.0, (0.1, 0.0), 8.0, np.ones((8, 8)), optimizer)
        bins = [(6500.0, 6600.0, 6550.0, np.ones(4, dtype=bool))]
        cube = np.ones((4, 8, 8), dtype=float)

        with mock.patch("musepipe.stages.stage_e01_psfao.fit_bin", return_value=fit_result):
            rows, recons = fit_psfao_bins(
                cube, cube, np.arange(4), bins, object(), (4.0, 6.0), 2.0, 3.0
            )

        self.assertEqual(rows[0]["status"], "fit_failed:optimizer_status_0")
        self.assertFalse(rows[0]["optimizer_success"])
        self.assertEqual(recons, {})


class RingQCTests(unittest.TestCase):
    def test_post_hybrid_arrays_drive_all_qc_fields(self):
        summary = _ring_qc_summary(
            [1.0, 2.0, 8.0, 9.0],
            [2.0, 3.0, 10.0, 20.0],
            {"psf_hybrid_threshold_pct": 5.0, "psf_hybrid_bin_fraction": 0.6},
        )
        self.assertEqual(summary["median"], 5.0)
        self.assertAlmostEqual(summary["p90"], 17.0)
        self.assertEqual(summary["bins_above"], 2)
        self.assertFalse(summary["issue"])  # 50% is not greater than the configured 60%.

        summary = _ring_qc_summary(
            [1.0, 8.0, 9.0], [2.0, 10.0, 20.0],
            {"psf_hybrid_threshold_pct": 5.0, "psf_hybrid_bin_fraction": 0.2},
        )
        self.assertTrue(summary["issue"])


class B3CentroidTests(unittest.TestCase):
    def test_absolute_psfao_centroids_are_compared_to_interpolated_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "track.csv"
            path.write_text(
                "wave_min_A,wave_max_A,primary_y,primary_x\n"
                "6400,6500,10.0,10.0\n"
                "6600,6700,10.0,10.4\n",
                encoding="utf-8",
            )
            track, status = _b3_chromatic_track(path)
            rows = [
                {"lambda_A": 6450.0, "dy": 0.0, "dx": 0.0, "status": "ok"},
                {"lambda_A": 6650.0, "dy": 0.0, "dx": 0.4, "status": "ok"},
            ]

            diff = _centroid_vs_b3(rows, "psfao", track, image_shape=(20, 20))

        self.assertEqual(status, "used")
        self.assertAlmostEqual(diff, 0.0)

    def test_missing_track_is_explicit(self):
        track, status = _b3_chromatic_track(Path("/definitely/missing/track.csv"))
        self.assertIsNone(track)
        self.assertEqual(status, "unavailable:missing_track")
        self.assertIsNone(_centroid_vs_b3([], "moffat", track))


class SelectedOutputTests(unittest.TestCase):
    def _psfao_row(self):
        return {
            "lambda_A": 6500.0,
            "samp": 0.7,
            "amp": 1.0,
            "bck": 0.0,
            "dy": 0.1,
            "dx": 0.2,
            "ring_residual_pct": 99.0,
            "ring_residual_pct_canonical": 8.0,
            "ring_residual_p90_pct_canonical": 12.0,
            "ring_residual_pct_after_hybrid_canonical": 4.0,
            "ring_residual_p90_pct_after_hybrid_canonical": 6.0,
            "r0": 0.1,
            "C": 0.01,
            "A": 1.0,
            "alpha": 0.1,
            "ratio": 1.0,
            "theta": 0.0,
            "beta": 1.5,
            "optimizer_success": True,
            "optimizer_status": 1,
            "optimizer_message": "ok",
            "optimizer_nfev": 3,
            "optimizer_cost": 1.0,
            "optimizer_stalled_at_initial": False,
            "status": "ok",
        }

    def test_psfao_csv_contains_canonical_before_and_after_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "params.csv"
            _write_psfao_csv(path, [self._psfao_row()])
            row = next(csv.DictReader(path.open(encoding="utf-8")))

        self.assertEqual(float(row["ring_residual_pct_canonical"]), 8.0)
        self.assertEqual(float(row["ring_residual_pct_after_hybrid_canonical"]), 4.0)
        self.assertEqual(row["optimizer_success"], "True")

    def test_summary_plot_accepts_selected_psfao_rows_without_moffat_fields(self):
        product = StageE01Product(
            fit_rows=[{"wave_center_A": 6500.0}],
            psf_model={},
            qc={"hybrid": {"applied": True}},
            hybrid_profiles=None,
            hybrid_radii=None,
            psf_form="psfao",
            psfao_rows=[self._psfao_row()],
        )
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.png"
            _write_summary_plot(product, {"summary_plot": summary})
            self.assertTrue(summary.exists())


if __name__ == "__main__":
    unittest.main()
