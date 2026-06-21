import csv
import json
import tempfile
import unittest
from pathlib import Path

from musepipe.stages.stage06_pca_c2_summary import (
    load_pca_c2_cases,
    summarize_pca_c2_cases,
)


def synthetic_case(case_id, line_label, pa_deg, matched_snr, aperture_snr):
    targets = [0.0, 5.0, 10.0]
    rows = []
    for index, target in enumerate(targets):
        rows.append(
            {
                "target_snr_input": target,
                "realistic_matched_snr": matched_snr[index],
                "realistic_aperture_snr": aperture_snr[index],
                "delta_matched_signal_transfer": float("nan") if target == 0 else 0.99,
                "delta_aperture_signal_transfer": float("nan") if target == 0 else 0.60,
            }
        )
    return {
        "case_id": case_id,
        "line_label": line_label,
        "line_center_A": 6500.0,
        "pa_deg": pa_deg,
        "injection_y": 10,
        "injection_x": 20,
        "rows": rows,
    }


class Stage06PcaC2SummaryTests(unittest.TestCase):
    def test_summary_builds_case_grid_and_completeness_rows(self):
        cases = [
            synthetic_case("line_a_pa045", "line_a", 45.0, [-1.0, 4.0, 9.0], [-2.0, 3.0, 8.0]),
            synthetic_case("line_a_pa135", "line_a", 135.0, [1.0, 6.0, 11.0], [0.0, 5.0, 10.0]),
        ]
        products = summarize_pca_c2_cases(cases, detection_threshold_snr=5.0)
        self.assertEqual(len(products["grid_rows"]), 6)
        self.assertEqual(len(products["case_rows"]), 2)
        self.assertEqual(len(products["completeness_rows"]), 6)

        all_rows = [
            row for row in products["completeness_rows"] if row["line_group"] == "ALL"
        ]
        self.assertEqual([row["matched_completeness"] for row in all_rows], [0.0, 0.5, 1.0])
        self.assertEqual([row["aperture_completeness"] for row in all_rows], [0.0, 0.5, 1.0])
        self.assertAlmostEqual(
            products["case_rows"][0]["delta_matched_transfer_median"], 0.99
        )
        self.assertAlmostEqual(
            products["case_rows"][0]["input_snr_at_matched_detection"], 6.0
        )

    def test_loader_requires_common_grid_and_realistic_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case_id, pa_deg in (("line_pa045", 45.0), ("line_pa135", 135.0)):
                case_dir = root / case_id
                case_dir.mkdir()
                qc = {
                    "case_id": case_id,
                    "run_id": "synthetic",
                    "line_label": "line",
                    "line_center_A": 6500.0,
                    "actual_injection_pa_deg": pa_deg,
                    "injection_yx": [10, 20],
                    "recovery_mode": "realistic",
                }
                (case_dir / f"{case_id}_qc.json").write_text(json.dumps(qc))
                with open(case_dir / f"{case_id}_recovery.csv", "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "target_snr_input",
                            "realistic_matched_snr",
                            "realistic_aperture_snr",
                            "delta_matched_signal_transfer",
                            "delta_aperture_signal_transfer",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "target_snr_input": 0.0,
                            "realistic_matched_snr": 0.0,
                            "realistic_aperture_snr": 0.0,
                            "delta_matched_signal_transfer": "nan",
                            "delta_aperture_signal_transfer": "nan",
                        }
                    )
            cases = load_pca_c2_cases(root, expected_n_cases=2)
            self.assertEqual(len(cases), 2)
            self.assertIsInstance(cases[0]["rows"][0]["target_snr_input"], float)

            qc_path = root / "line_pa135" / "line_pa135_qc.json"
            qc = json.loads(qc_path.read_text())
            qc["recovery_mode"] = "deterministic"
            qc_path.write_text(json.dumps(qc))
            with self.assertRaisesRegex(ValueError, "not a realistic recovery run"):
                load_pca_c2_cases(root, expected_n_cases=2)


if __name__ == "__main__":
    unittest.main()
