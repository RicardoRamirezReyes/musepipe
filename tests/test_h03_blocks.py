import json
import tempfile
import unittest
from pathlib import Path

from musepipe.stages.stage_h03_limits import compute_stage_h03_products, stage_h03_paths
from tests.test_h03_chain import h03_physical_config


class H03BlockTests(unittest.TestCase):
    def test_stage_refuses_without_valid_e4(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "synthetic_h03_block"
            paths = stage_h03_paths(run_id, project_root=root)
            paths["paths"].ensure_base_dirs()
            paths["stage_h01_qc_json"].write_text(
                json.dumps({"verdict": {"verdict": "non_detection"}}),
                encoding="utf-8",
            )
            paths["stage_h02_qc_json"].write_text(json.dumps({"overall": "survives"}), encoding="utf-8")
            cfg = h03_physical_config(run_id, root)

            with self.assertRaisesRegex(RuntimeError, "E4 throughput is mandatory"):
                compute_stage_h03_products(cfg, paths)

            paths["stage_h04_qc_json"].write_text(
                json.dumps({"regression_historic": {"verdict": "fail"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "E4 throughput is mandatory"):
                compute_stage_h03_products(cfg, paths)


if __name__ == "__main__":
    unittest.main()
