import unittest

import numpy as np

from musepipe.stages.stage01c_localize import AmbiguousDetectionError, detect_restricted_source
from tests.test_stage01c_synthetic import gaussian2d


class Stage01cConfusionTests(unittest.TestCase):
    def test_two_significant_sources_inside_tolerance_raise(self):
        ny, nx = 60, 60
        image = np.zeros((ny, nx), dtype=np.float64)
        image += gaussian2d(ny, nx, 28.0, 28.0, 25.0, 1.2)
        image += gaussian2d(ny, nx, 32.0, 34.0, 23.0, 1.2)
        image += np.random.default_rng(4).normal(0.0, 0.02, size=image.shape)

        with self.assertRaises(AmbiguousDetectionError):
            detect_restricted_source(
                image,
                predicted_yx=(30.0, 31.0),
                search_radius_px=8.0,
                fwhm_px=3.0,
                snr_min=5.0,
                primary_yx=(30.0, 30.0),
            )


if __name__ == "__main__":
    unittest.main()
