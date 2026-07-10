import json
import tempfile
import unittest
from pathlib import Path

from musepipe.report import build_report, report_paths, report_tree_hash


def make_report_run(root, run_id="synthetic_report"):
    paths = report_paths(run_id, root)
    paths.run_paths.ensure_base_dirs()
    payload = {"config": {"run_id": run_id}, "meta": {}}
    paths.run_paths.config_json.write_text(json.dumps(payload), encoding="utf-8")
    paths.run_paths.stage_dir.joinpath("stage_h01_qc.json").write_text(
        json.dumps({"stage": "h01", "verdict": {"verdict": "non_detection"}, "checks": {}}),
        encoding="utf-8",
    )
    return paths


class ReportDeterminismTests(unittest.TestCase):
    def test_double_execution_produces_identical_report_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = make_report_run(root)

            build_report(paths.run_paths.run_id, project_root=root)
            first = report_tree_hash(paths.report_dir)
            build_report(paths.run_paths.run_id, project_root=root)
            second = report_tree_hash(paths.report_dir)

            self.assertEqual(first, second)
            self.assertTrue(paths.run_summary_json.exists())
            self.assertTrue(paths.report_md.exists())


if __name__ == "__main__":
    unittest.main()
