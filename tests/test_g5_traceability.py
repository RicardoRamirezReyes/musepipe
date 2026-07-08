"""G5 §7 V1: traceability chain complete; a missing citation/QC is detected."""

import json
import tempfile
import unittest
from pathlib import Path

from musepipe.characterization import characterization_paths, check_traceability
from tests._g5_fixture import make_run


class G5TraceabilityTests(unittest.TestCase):
    def test_complete_chain_passes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rid = make_run(root)
            cp = characterization_paths(rid, project_root=str(root))
            trace = check_traceability(cp)
            self.assertTrue(trace["complete"])
            self.assertEqual(trace["broken_chains"], [])

    def test_missing_citation_breaks_chain(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rid = make_run(root)
            cfgp = root / "runs" / rid / "config" / "config.json"
            cfg = json.loads(cfgp.read_text())
            del cfg["config"]["h03_distance_source"]  # drop a required citation
            cfgp.write_text(json.dumps(cfg))
            cp = characterization_paths(rid, project_root=str(root))
            trace = check_traceability(cp)
            self.assertFalse(trace["complete"])
            self.assertTrue(any(b["link"] == "distance" for b in trace["broken_chains"]))

    def test_missing_phase_qc_breaks_chain(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rid = make_run(root)
            (root / "runs" / rid / "stages" / "stage_g3_qc.json").unlink()
            cp = characterization_paths(rid, project_root=str(root))
            trace = check_traceability(cp)
            self.assertFalse(trace["complete"])
            self.assertTrue(any(b["link"] == "qc_g3" for b in trace["broken_chains"]))


if __name__ == "__main__":
    unittest.main()
