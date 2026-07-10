import unittest

import numpy as np

from musepipe.qc.cube_qc import measure_stat_factors


class CubeQcM5Tests(unittest.TestCase):
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
