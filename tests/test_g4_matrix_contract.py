"""G4 §7 V1: evidence-matrix contract — every cell has verdict+value+source,
all hypotheses present, QC keys well-formed."""

import json
import tempfile
import unittest
from pathlib import Path

from musepipe.classify import HYPOTHESES, VERDICTS
from musepipe.stages.stage_g4_classify import build_matrix, compute_stage_g4, stage_g4_paths


def _fake_run(root, run_id):
    stages = root / "runs" / run_id / "stages"
    tables = root / "runs" / run_id / "tables"
    stages.mkdir(parents=True); tables.mkdir(parents=True)
    (stages / "stage01c_qc.json").write_text(json.dumps({
        "companion": {"snr_detection": 12.3, "pos_yx": [155.6, 75.8]},
        "astrometry": {"sep_arcsec": 1.80, "sep_err": 0.003, "expected_sep_arcsec": 1.81,
                       "pa_deg": 240.2, "expected_pa_deg": 240.0, "sep_deviation_sigma": 3.3},
    }))
    (stages / "stage_h02_qc.json").write_text(json.dumps({"overall": "survives"}))
    (stages / "stage_g3_qc.json").write_text(json.dumps({"open_issues": [{"issue": "pending_libraries"}]}))
    (tables / "g2_line_measurements.csv").write_text("name,status\nHalpha,upper_limit\n")
    return {"run_id": run_id, "project_root": str(root),
            "g4_cpm_confirmed": True, "g4_background_density_per_arcsec2": 1e-3,
            "g4_search_radius_arcsec": 0.5}


class G4MatrixContractTests(unittest.TestCase):
    def test_every_cell_has_verdict_value_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = _fake_run(root, "TESTRUN")
            paths = stage_g4_paths("TESTRUN", project_root=str(root))
            matrix, _pbg = build_matrix(cfg, paths)
            for test, per_h in matrix.items():
                self.assertEqual(set(per_h.keys()), set(HYPOTHESES), msg=test)
                for h, cell in per_h.items():
                    self.assertIn(cell["verdict"], VERDICTS)
                    self.assertIn("source", cell)

    def test_qc_has_required_keys_and_ranking(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = _fake_run(root, "TESTRUN")
            paths = stage_g4_paths("TESTRUN", project_root=str(root))
            _matrix, qc = compute_stage_g4(cfg, paths)
            for key in ("hypotheses", "combined_ranking", "final_class",
                        "frozen_thresholds_hash", "background_probability"):
                self.assertIn(key, qc)
            self.assertEqual(len(qc["combined_ranking"]), len(HYPOTHESES))
            self.assertIn(qc["final_class"]["robustness"], ("secure", "probable", "ambiguous"))

    def test_deferred_g3_marks_t3_t4_t6_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = _fake_run(root, "TESTRUN")
            paths = stage_g4_paths("TESTRUN", project_root=str(root))
            _m, qc = compute_stage_g4(cfg, paths)
            for t in ("T3", "T4", "T6"):
                self.assertIn(t, qc["tests_unavailable"])


if __name__ == "__main__":
    unittest.main()
