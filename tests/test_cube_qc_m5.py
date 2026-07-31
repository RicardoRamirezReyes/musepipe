import unittest

import numpy as np

from musepipe.qc.cube_qc import empty_aperture_centers, measure_stat_factors


class CubeQcM5Tests(unittest.TestCase):
    def test_empty_apertures_avoid_sources_and_invalid_pixels(self):
        sources = np.zeros((40, 40), dtype=bool)
        sources[10:20, 10:20] = True
        valid = np.ones_like(sources)
        valid[:5] = False
        centers = empty_aperture_centers(
            sources, valid, radius_px=2, spacing_px=5, max_apertures=20
        )
        self.assertGreaterEqual(len(centers), 10)
        for y, x in centers:
            self.assertFalse(sources[y - 2 : y + 3, x - 2 : x + 3].any())
            self.assertTrue(valid[y - 2 : y + 3, x - 2 : x + 3].all())

    def test_stat_factor_13_is_recovered(self):
        rng = np.random.default_rng(11)
        n_wave, ny, nx = 40, 30, 30
        stat = np.ones((n_wave, ny, nx), dtype=float)
        cube = rng.normal(0.0, np.sqrt(1.3), size=stat.shape)
        sky_mask = np.ones((ny, nx), dtype=bool)
        result = measure_stat_factors(cube, stat, sky_mask)
        self.assertAlmostEqual(result["factor_spaxel_median"], 1.3, delta=0.12)
        self.assertEqual(result["status"], "green")


if __name__ == "__main__":
    unittest.main()
