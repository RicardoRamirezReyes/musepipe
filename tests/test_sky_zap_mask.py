import unittest

import numpy as np

from musepipe.reduction.sky_zap import (
    SourceRegion,
    binary_dilate,
    build_source_mask,
    estimate_halo_radius,
    sky_fraction_from_source_mask,
)


class SkyZapMaskTests(unittest.TestCase):
    def test_binary_dilate_expands_one_pixel(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        out = binary_dilate(mask, 1)
        self.assertEqual(np.count_nonzero(out), 9)

    def test_forced_companion_region_is_masked_even_if_faint(self):
        image = np.zeros((30, 30), dtype=float)
        image[10, 10] = 100.0
        regions = [
            SourceRegion("primary", (10, 10), radius_px=4, auto_halo=False),
            SourceRegion("companion", (20, 20), radius_px=3, auto_halo=False),
        ]
        mask, radii = build_source_mask(image, regions, threshold_sigma=5, dilation_px=0)
        self.assertTrue(mask[20, 20])
        self.assertEqual(radii["companion"], 3.0)
        self.assertLess(sky_fraction_from_source_mask(mask), 1.0)

    def test_auto_halo_radius_grows_to_include_wings(self):
        yy, xx = np.indices((60, 60), dtype=float)
        rr = np.hypot(yy - 30, xx - 30)
        image = np.exp(-0.5 * (rr / 6.0) ** 2)
        radius = estimate_halo_radius(image, (30, 30), min_radius_px=2, margin_px=2)
        self.assertGreater(radius, 5.0)


if __name__ == "__main__":
    unittest.main()
