import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.telluric_lines import TELLURIC_BANDS as CATALOGO_BANDS
from musepipe.reduction.telluric import (
    DEFAULT_FIT_REGIONS,
    TELLURIC_BANDS,
    TelluricError,
    check_molecfit_environment,
    decide_telluric,
    measure_telluric_depths,
    resolve_input_cube,
    stage00t_qc_skeleton,
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

    def test_o2_a_is_measured_and_can_drive_the_verdict(self):
        # La banda mas profunda del rango de MUSE, que la etapa no medía.
        wave = np.linspace(7450, 7850, 401)
        spec = np.ones_like(wave)
        spec[(wave >= 7590) & (wave <= 7700)] *= 0.90
        depths = measure_telluric_depths(wave, spec)
        self.assertIn("O2_A", depths)
        self.assertGreater(depths["O2_A"], 9.0)
        self.assertEqual(
            decide_telluric(depths, science_needs_red_continuum=True).decision, "needed"
        )

    def test_o2_a_is_one_of_the_bands_the_stage_declares(self):
        self.assertEqual(TELLURIC_BANDS["O2_A"], (7590.0, 7700.0))
        # Los mismos bordes que el catálogo de `telluric_lines`, G3 y el QC de
        # ruido: una sola definición del intervalo, no cuatro.
        catalogo = next(b for b in CATALOGO_BANDS if b["species"] == "O2" and b["lo_A"] == 7590.0)
        self.assertEqual((catalogo["lo_A"], catalogo["hi_A"]), TELLURIC_BANDS["O2_A"])
        self.assertEqual(len(DEFAULT_FIT_REGIONS), len(TELLURIC_BANDS))

    def test_decision_skips_when_science_does_not_need_red_continuum(self):
        decision = decide_telluric({"O2_B": 10.0}, science_needs_red_continuum=False)
        self.assertFalse(decision.telluric_applied)
        self.assertEqual(decision.decision, "not_needed_science")

    def test_decision_needs_checkpoint_for_deep_band(self):
        decision = decide_telluric({"O2_B": 4.0}, science_needs_red_continuum=True)
        self.assertTrue(decision.telluric_applied)
        self.assertTrue(decision.checkpoint_required)

    def test_the_qc_declares_the_aperture_it_measured_with(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "adp.fits"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            info = resolve_input_cube(cube, upstream="ADP", checksum=False)
            qc = stage00t_qc_skeleton(info, run_id="R", primary_yx=(100, 100),
                                      aperture_radius_px=8.0)
            self.assertEqual(qc["input"]["primary_yx"], [100.0, 100.0])
            self.assertEqual(qc["input"]["aperture_radius_px"], 8.0)
            self.assertEqual(qc["fit"]["regions_A"],
                             [list(r) for r in DEFAULT_FIT_REGIONS])

    def test_the_qc_cannot_be_built_without_the_aperture(self):
        # Sin default: el esquema multi-noche perdió estos dos campos justamente
        # porque nada obligaba a declararlos.
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "adp.fits"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            info = resolve_input_cube(cube, upstream="ADP", checksum=False)
            with self.assertRaises(TypeError):
                stage00t_qc_skeleton(info, run_id="R")

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
