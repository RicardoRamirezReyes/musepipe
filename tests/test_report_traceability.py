import json
import tempfile
import unittest
from pathlib import Path

from musepipe.report import build_report, report_paths, template_contains_hardcoded_numbers
from tests.test_report_determinism import make_report_run


class ReportTraceabilityTests(unittest.TestCase):
    def test_template_has_no_hardcoded_numeric_results(self):
        self.assertFalse(template_contains_hardcoded_numbers())

    def test_declared_report_artifacts_exist_and_are_nonempty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = make_report_run(root, "synthetic_trace_report")

            result = build_report(paths.run_paths.run_id, project_root=root)
            summary = result["summary"]

            for table_path in summary["tables"].values():
                path = Path(table_path)
                self.assertTrue(path.exists(), table_path)
                self.assertGreater(path.stat().st_size, 0, table_path)
            for meta in summary["figures"].values():
                path = Path(meta["path"])
                self.assertTrue(path.exists(), meta["path"])
                self.assertGreater(path.stat().st_size, 0, meta["path"])
            self.assertTrue(paths.report_md.exists())
            self.assertIn(summary["overall_status"], paths.report_md.read_text(encoding="utf-8"))

            loaded = json.loads(paths.run_summary_json.read_text(encoding="utf-8"))
            self.assertEqual(loaded["traceability"]["report_source"], "run_summary.json")


if __name__ == "__main__":
    unittest.main()
