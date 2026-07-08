"""G5 §7 V2: consistency across phases (Halpha category) — coherent vs incoherent."""

import tempfile
import unittest
from pathlib import Path

from musepipe.characterization import characterization_paths, consistency_checks
from tests._g5_fixture import make_run


class G5ConsistencyTests(unittest.TestCase):
    def test_coherent_run_has_no_issue(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rid = make_run(root, consistent=True)
            cp = characterization_paths(rid, project_root=str(root))
            out = consistency_checks(cp)
            self.assertEqual(out["issues"], [])
            ha = next(c for c in out["checks"] if c["check"] == "halpha_category_g2_vs_h01")
            self.assertTrue(ha["consistent"])

    def test_incoherent_run_raises_blocking_issue(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rid = make_run(root, consistent=False)  # G2 'detected' vs H01 'non_detection'
            cp = characterization_paths(rid, project_root=str(root))
            out = consistency_checks(cp)
            self.assertTrue(any(i["priority"] == "blocking" for i in out["issues"]))


if __name__ == "__main__":
    unittest.main()
