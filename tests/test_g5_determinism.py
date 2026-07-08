"""G5 §7 V4: two builds → identical package (determinism hash), F1 pattern."""

import tempfile
import unittest
from pathlib import Path

from musepipe.characterization import build_characterization
from tests._g5_fixture import make_run


class G5DeterminismTests(unittest.TestCase):
    def test_two_builds_have_identical_determinism_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rid = make_run(root)
            a = build_characterization(rid, project_root=str(root))
            b = build_characterization(rid, project_root=str(root))
            self.assertEqual(a["determinism_hash"], b["determinism_hash"])

    def test_table_bytes_are_stable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rid = make_run(root)
            build_characterization(rid, project_root=str(root))
            out = root / "runs" / rid / "report" / "characterization"
            first = (out / "adopted_parameters.csv").read_bytes()
            build_characterization(rid, project_root=str(root))
            self.assertEqual(first, (out / "adopted_parameters.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
