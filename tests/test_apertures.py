import math
import unittest

import numpy as np

from musepipe.apertures import (
    angular_separation_deg,
    aperture_weights,
    box_spectrum_mean,
    box_spectrum_sum,
    same_radius_control_positions,
)


class ApertureTests(unittest.TestCase):
    def test_angular_separation_wraps_across_zero(self):
        sep = angular_separation_deg(math.radians(10.0), math.radians(350.0))
        self.assertAlmostEqual(sep, 20.0)

    def test_same_radius_control_positions_excludes_object_angle(self):
        controls = same_radius_control_positions(
            object_yx=(2, 4),
            star_yx=(2, 2),
            ny=7,
            nx=7,
            n_positions=4,
            exclude_angle_deg=10.0,
            margin_px=0,
        )
        self.assertEqual(controls, [(4, 2), (2, 0), (0, 2)])

    def test_aperture_weights_pixel_box_circle_and_gaussian(self):
        pixel = aperture_weights(5, 5, (2, 2), {"kind": "pixel"})
        self.assertEqual(pixel.sum(), 1.0)
        self.assertEqual(pixel[2, 2], 1.0)

        box = aperture_weights(5, 5, (2, 2), {"kind": "box", "size": 3})
        self.assertEqual(box.sum(), 9.0)

        clipped_box = aperture_weights(5, 5, (0, 0), {"kind": "box", "size": 3})
        self.assertEqual(clipped_box.sum(), 4.0)

        circle = aperture_weights(5, 5, (2, 2), {"kind": "circle", "radius_px": 1.0})
        self.assertEqual(circle.sum(), 5.0)

        gaussian = aperture_weights(5, 5, (2, 2), {"kind": "gaussian", "sigma_px": 1.0, "radius_px": 1.0})
        self.assertEqual(gaussian[2, 2], 1.0)
        self.assertGreater(gaussian[2, 1], 0.0)
        self.assertEqual(gaussian[0, 0], 0.0)

    def test_box_spectrum_sum_and_mean_clip_at_edges(self):
        cube = np.arange(2 * 5 * 5, dtype=float).reshape(2, 5, 5)

        np.testing.assert_allclose(box_spectrum_sum(cube, 2, 2, box_size=3), [108.0, 333.0])
        np.testing.assert_allclose(box_spectrum_mean(cube, 2, 2, box_size=3), [12.0, 37.0])

        np.testing.assert_allclose(box_spectrum_sum(cube, 0, 0, box_size=3), [12.0, 112.0])
        np.testing.assert_allclose(box_spectrum_mean(cube, 0, 0, box_size=3), [3.0, 28.0])

    def test_unknown_aperture_kind_errors(self):
        with self.assertRaises(ValueError):
            aperture_weights(5, 5, (2, 2), {"kind": "triangle"})


if __name__ == "__main__":
    unittest.main()
