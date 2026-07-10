import json
import tempfile
import unittest
from pathlib import Path

from musepipe.extraction.aperture import sha256_file
from musepipe.report import build_report, report_paths
from tests.test_report_determinism import make_report_run


class ReportHashChainTests(unittest.TestCase):
    def test_manipulated_declared_hash_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = make_report_run(root, "synthetic_hash_report")
            input_file = paths.run_paths.stage_dir / "input_cube.fits"
            input_file.write_bytes(b"original cube")
            wrong_hash = "0" * 64
            self.assertNotEqual(sha256_file(input_file), wrong_hash)
            paths.run_paths.stage_dir.joinpath("stage01c_qc.json").write_text(
                json.dumps({"input_cube": {"file": str(input_file), "sha256": wrong_hash}}),
                encoding="utf-8",
            )

            result = build_report(paths.run_paths.run_id, project_root=root)

            self.assertEqual(result["summary"]["hash_chain"]["status"], "fail")
            failures = [row for row in result["summary"]["hash_chain"]["checks"] if row["status"] == "fail"]
            self.assertEqual(len(failures), 1)
            self.assertIn("Hash mismatch", "\n".join(issue["issue"] for issue in result["summary"]["open_issues"]))


if __name__ == "__main__":
    unittest.main()
