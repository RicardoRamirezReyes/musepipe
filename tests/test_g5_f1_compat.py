"""G5 §7 V6: the G5 extension does not modify or break the F1 report machinery."""

import tempfile
import unittest
from pathlib import Path

import musepipe.report as report
from musepipe.characterization import build_characterization
from tests._g5_fixture import make_run


class G5F1CompatTests(unittest.TestCase):
    def test_f1_public_functions_intact(self):
        for name in ("make_run_summary", "render_report_markdown", "write_json_deterministic",
                     "write_csv_deterministic", "report_tree_hash", "check_declared_hashes"):
            self.assertTrue(hasattr(report, name), msg=f"F1 lost {name}")

    def test_characterization_writes_only_under_its_subdir(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rid = make_run(root)
            report_dir = root / "runs" / rid / "report"
            # a pre-existing F1 file must survive untouched
            report_dir.mkdir(parents=True, exist_ok=True)
            f1 = report_dir / "run_summary.json"
            f1.write_text('{"f1": "sacred"}')
            build_characterization(rid, project_root=str(root))
            self.assertEqual(f1.read_text(), '{"f1": "sacred"}')  # F1 file untouched
            self.assertTrue((report_dir / "characterization" / "characterization.md").exists())


if __name__ == "__main__":
    unittest.main()
