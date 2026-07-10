import unittest

import numpy as np

from musepipe.stages.stage_h02_artifacts import t2_spatial_coherence


def gaussian_stamp(shape=(21, 21), center=(10.0, 10.0), sigma_y=2.0, sigma_x=2.0):
    yy, xx = np.indices(shape, dtype=np.float64)
    return np.exp(
        -0.5
        * (
            ((yy - float(center[0])) / float(sigma_y)) ** 2
            + ((xx - float(center[1])) / float(sigma_x)) ** 2
        )
    )


class H02T2ShapeTests(unittest.TestCase):
    def test_psf_shaped_source_passes(self):
        psf = gaussian_stamp()
        yy, xx = np.indices(psf.shape, dtype=np.float64)
        plane = 2.0 + 0.01 * yy - 0.02 * xx
        stamp = plane + 15.0 * psf

        result = t2_spatial_coherence(stamp, psf, expected_center_yx=(10.0, 10.0))

        self.assertEqual(result["status"], "pass")
        self.assertLess(result["centroid_offset_px"], 0.1)
        self.assertLess(result["elongation_vs_psf"], 1.2)

    def test_elongated_blob_fails(self):
        psf = gaussian_stamp()
        yy, xx = np.indices(psf.shape, dtype=np.float64)
        plane = 2.0 + 0.01 * yy - 0.02 * xx
        stamp = plane + 15.0 * gaussian_stamp(sigma_y=2.0, sigma_x=5.0)

        result = t2_spatial_coherence(stamp, psf, expected_center_yx=(10.0, 10.0))

        self.assertEqual(result["status"], "fail")
        self.assertGreater(result["elongation_vs_psf"], 1.5)


if __name__ == "__main__":
    unittest.main()
