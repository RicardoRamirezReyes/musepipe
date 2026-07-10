import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.reduction.telluric import (
    TelluricError,
    check_molecfit_environment,
    decide_telluric,
    measure_telluric_depths,
    resolve_input_cube,
)


def _write_data_stat_cube(path, data):
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.asarray(data, dtype=np.float32), name="DATA"),
            fits.ImageHDU(data=np.ones_like(data, dtype=np.float32), name="STAT"),
        ]
    ).writeto(path)


class TelluricDecisionTests(unittest.TestCase):
    def test_depth_pct_detects_synthetic_o2_band(self):
        wave = np.linspace(6800, 7000, 201)
        spec = np.ones_like(wave)
        spec[(wave >= 6864) & (wave <= 6960)] *= 0.92
        depths = measure_telluric_depths(wave, spec)
        self.assertGreater(depths["O2_B"], 7.0)

    def test_decision_skips_when_science_does_not_need_red_continuum(self):
        decision = decide_telluric({"O2_B": 10.0}, science_needs_red_continuum=False)
        self.assertFalse(decision.telluric_applied)
        self.assertEqual(decision.decision, "not_needed_science")

    def test_decision_needs_checkpoint_for_deep_band(self):
        decision = decide_telluric({"O2_B": 4.0}, science_needs_red_continuum=True)
        self.assertTrue(decision.telluric_applied)
        self.assertTrue(decision.checkpoint_required)

    def test_resolve_adp_input_accepts_data_stat_without_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "adp.fits"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            info = resolve_input_cube(cube, upstream="ADP", checksum=False)
            self.assertEqual(info.upstream, "ADP")
            self.assertTrue(info.has_stat)

    def test_resolve_a2_input_requires_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "cube.fits"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            with self.assertRaises(TelluricError):
                resolve_input_cube(cube, upstream="A2", checksum=False)

    def test_check_molecfit_environment_requires_all_recipes(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "esorex"
            exe.write_text("#!/bin/sh\n", encoding="utf-8")
            exe.chmod(0o755)

            def runner(args):
                return subprocess.CompletedProcess(args, 0, stdout="molecfit_model : recipe", stderr="")

            with self.assertRaises(TelluricError):
                check_molecfit_environment(esorex=str(exe), runner=runner)

    def test_check_molecfit_environment_accepts_required_recipes(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "esorex"
            exe.write_text("#!/bin/sh\n", encoding="utf-8")
            exe.chmod(0o755)

            def runner(args):
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout=(
                        "molecfit_model : recipe\n"
                        "molecfit_calctrans : recipe\n"
                        "molecfit_correct : recipe\n"
                    ),
                    stderr="",
                )

            env = check_molecfit_environment(esorex=str(exe), runner=runner)
            self.assertIn("molecfit_correct", env["molecfit_recipes"])


if __name__ == "__main__":
    unittest.main()
