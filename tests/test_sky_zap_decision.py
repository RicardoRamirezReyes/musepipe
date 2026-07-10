import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from musepipe.reduction.sky_zap import (
    SkyZapError,
    aperture_pixel_mask,
    classify_zap_decision,
    compute_sky_residual_metrics,
    resolve_input_cube,
)


def _write_data_stat_cube(path, data):
    hdus = [
        fits.PrimaryHDU(),
        fits.ImageHDU(data=np.asarray(data, dtype=np.float32), name="DATA"),
        fits.ImageHDU(data=np.ones_like(data, dtype=np.float32), name="STAT"),
    ]
    fits.HDUList(hdus).writeto(path)


class SkyZapDecisionTests(unittest.TestCase):
    def test_resolve_adp_input_does_not_require_a1_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adp.fits"
            _write_data_stat_cube(path, np.ones((4, 5, 5)))
            info = resolve_input_cube(path, provenance="adp", checksum=False)
            self.assertEqual(info.provenance, "adp")
            self.assertTrue(info.a1_not_applicable)
            self.assertFalse(info.a1_gates_ok)
            self.assertTrue(info.has_stat)

    def test_resolve_raw_reduction_requires_green_a1_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "cube.fits"
            qc = Path(tmp) / "stage00r_qc.json"
            _write_data_stat_cube(cube, np.ones((4, 5, 5)))
            qc.write_text(json.dumps({"gates_passed": [], "verification": {}, "open_issues": []}))
            with self.assertRaises(SkyZapError):
                resolve_input_cube(cube, provenance="raw_reduction", a1_qc_path=qc, checksum=False)

    def test_compute_sky_residual_metrics_sees_skyline_excess(self):
        rng = np.random.default_rng(123)
        wave = np.array([5200.0, 5400.0, 5577.3, 6300.3, 6650.0, 6750.0])
        cube = rng.normal(0.0, 1.0, size=(wave.size, 20, 20))
        cube[2] = rng.normal(0.0, 4.0, size=(20, 20))
        cube[3] = rng.normal(0.0, 3.5, size=(20, 20))
        sky_mask = aperture_pixel_mask((20, 20), [(5, 5), (14, 14)], radius_px=3)
        metrics = compute_sky_residual_metrics(cube, wave, sky_mask)
        self.assertGreater(metrics["R_skyline_over_continuum"], 2.0)

    def test_classify_zap_decision_thresholds(self):
        self.assertEqual(classify_zap_decision(1.2, 0.8).decision, "not_needed")
        needed = classify_zap_decision(2.2, 0.8)
        self.assertEqual(needed.decision, "needed")
        self.assertTrue(needed.zap_applied)
        gray = classify_zap_decision(1.7, 0.8)
        self.assertEqual(gray.decision, "gray_zone_checkpoint")
        self.assertTrue(gray.checkpoint_required)
        low_sky = classify_zap_decision(2.5, 0.1)
        self.assertEqual(low_sky.decision, "insufficient_sky_checkpoint")
        self.assertFalse(low_sky.zap_applied)


if __name__ == "__main__":
    unittest.main()
