import unittest

import numpy as np

from musepipe.qc.frame_qc import (
    GAUSS_FWHM,
    apply_discard_criteria,
    frame_metrics,
    locate_primary,
    moment_fwhm,
    ring_background,
)


def gaussian_image(ny, nx, y0, x0, sigma, amp=1000.0, background=5.0):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return background + amp * np.exp(-(((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * sigma ** 2)))


class TestPrimitives(unittest.TestCase):
    def test_locate_primary_finds_peak(self):
        img = gaussian_image(120, 130, 70, 40, 3.0)
        iy, ix = locate_primary(img)
        self.assertLessEqual(abs(iy - 70), 1)
        self.assertLessEqual(abs(ix - 40), 1)

    def test_ring_background_recovers_level(self):
        img = gaussian_image(120, 130, 60, 60, 3.0, background=7.0)
        bg = ring_background(img, 60, 60, 40, 55)
        self.assertAlmostEqual(bg, 7.0, delta=0.2)

    def test_moment_fwhm_recovers_sigma(self):
        sigma = 3.0
        img = gaussian_image(120, 130, 60, 60, sigma, background=5.0)
        fwhm = moment_fwhm(img, 60, 60, box_half=12, background=5.0)
        self.assertAlmostEqual(fwhm, GAUSS_FWHM * sigma, delta=1.0)


class TestFrameMetricsAndCriteria(unittest.TestCase):
    def test_frame_metrics_on_synthetic_cube(self):
        waves = np.linspace(7900.0, 9100.0, 40)
        cube = np.stack([gaussian_image(120, 130, 55, 45, 3.0) for _ in waves])
        m = frame_metrics(cube, waves, ring_inner_px=40, ring_outer_px=55)
        self.assertLessEqual(abs(m["primary_yx"][0] - 55), 1)
        self.assertAlmostEqual(m["fwhm_px"], GAUSS_FWHM * 3.0, delta=1.5)

    def test_discard_criteria_flags_outlier(self):
        rows = [
            {"fwhm_px": 4.0, "background_median": 5.0},
            {"fwhm_px": 4.1, "background_median": 5.1},
            {"fwhm_px": 4.0, "background_median": 4.9},
            {"fwhm_px": 9.0, "background_median": 5.0},   # FWHM > 1.5x median -> flagged
        ]
        summary = apply_discard_criteria(rows)
        self.assertEqual(summary["n_flagged"], 1)
        self.assertTrue(rows[3]["flag_discardable"])
        self.assertFalse(rows[0]["flag_discardable"])

    def test_discard_criteria_all_clean(self):
        rows = [{"fwhm_px": 4.0 + 0.1 * i, "background_median": 5.0 + 0.05 * i} for i in range(7)]
        summary = apply_discard_criteria(rows)
        self.assertEqual(summary["n_flagged"], 0)


if __name__ == "__main__":
    unittest.main()
