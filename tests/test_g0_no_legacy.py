"""G0 --no-legacy: targets without an archival ADP (e.g. ROXs 42B b) record the
legacy provenance comparison as unavailable instead of a meaningless cross-target
comparison, and never touch a legacy run directory."""

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_g0", ROOT / "scripts" / "run_g0.py")
assert SPEC is not None and SPEC.loader is not None
RUN_G0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_G0)


def _make_minimal_run(root: Path, run_id: str) -> None:
    run_dir = root / "runs" / run_id
    (run_dir / "report").mkdir(parents=True)
    (run_dir / "stages").mkdir(parents=True)
    (run_dir / "report" / "run_summary.json").write_text(json.dumps({
        "stages": [{"stage": "x11", "status": "green", "issue_count": 0}],
        "hash_chain": {"status": "pass", "checks": []},
        "overall_status": "green",
    }))
    (run_dir / "stages" / "stage00q_qc.json").write_text(json.dumps({
        "wavelength_frame": "unknown",
        "m5_stat": {"status": "red", "factor_spaxel_median": 4.0, "note": "STAT underestimates noise."},
    }))
    cube = np.ones((1, 2, 2, 2), dtype=np.float32)
    hdu = fits.ImageHDU(data=cube, name="CUBES")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(
        run_dir / "stages" / "stage02_xcorr_cube_stack.fits"
    )


class G0NoLegacyTests(unittest.TestCase):
    def test_no_legacy_records_unavailable_without_legacy_run(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "ROXs42Bb_realigned"
            _make_minimal_run(root, run_id)
            # No legacy run dir exists at all -> would crash without no_legacy.
            qc = RUN_G0.build_g0(run_id, project_root=root, no_legacy=True)
            leg = qc["legacy_comparison"]
            self.assertEqual(leg["status"], "unavailable")
            self.assertIsNone(leg["n_flagged"])
            self.assertIsNone(leg["legacy_run"])
            # No comparison CSV is written when legacy is unavailable.
            self.assertFalse((root / "runs" / run_id / "tables" / "g0_legacy_comparison.csv").exists())
            # The unavailable note reaches the open issues.
            self.assertTrue(any(
                "unavailable" in i["issue"].lower() and "legacy" in i["issue"].lower()
                for i in qc["open_issues"]
            ))


if __name__ == "__main__":
    unittest.main()
