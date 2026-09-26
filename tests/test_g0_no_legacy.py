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


class G0IssuesDerivadosTests(unittest.TestCase):
    """G0 dice el estado de A4/M5 y de F1, no el de julio (2026-09-26)."""

    def _qc(self, *, f1, m5):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_minimal_run(root, "R")
            rd = root / "runs" / "R"
            rs = json.loads((rd / "report" / "run_summary.json").read_text())
            rs.update(f1)
            (rd / "report" / "run_summary.json").write_text(json.dumps(rs))
            (rd / "stages" / "stage00q_qc.json").write_text(json.dumps({"m5_stat": m5}))
            return RUN_G0.build_g0("R", project_root=root, no_legacy=True)

    def test_f1_verde_y_stat_verde_no_dejan_bloqueante(self):
        qc = self._qc(f1={"overall_status": "green"}, m5={"status": "green"})
        textos = " ".join(i["issue"] for i in qc["open_issues"])
        self.assertFalse(any(i["priority"] == "blocking" for i in qc["open_issues"]))
        self.assertNotIn("A-block", textos)
        self.assertNotIn("STAT", textos)

    def test_f1_rojo_nombra_sus_bloqueantes_y_se_marca_como_repeticion(self):
        qc = self._qc(f1={"overall_status": "red", "open_issues": [
            {"issue": "a", "priority": "blocking", "stage": "C2_aperture"},
            {"issue": "b", "priority": "major", "stage": "C1_psf"}]}, m5={"status": "red"})
        f1 = [i for i in qc["open_issues"] if i.get("restates") == "F1"]
        self.assertEqual(len(f1), 1)
        self.assertEqual(f1[0]["priority"], "blocking")
        self.assertIn("1 blocking issue(s)", f1[0]["issue"])
        self.assertIn("C2_aperture", f1[0]["issue"])
        self.assertNotIn("A-block", f1[0]["issue"])
        self.assertTrue(any("STAT red" in i["issue"] for i in qc["open_issues"]))

if __name__ == "__main__":
    unittest.main()
