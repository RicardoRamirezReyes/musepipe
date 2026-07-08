"""G2 §7: continuum windows exclude neighbouring catalog lines and bad_ranges."""

import unittest

import numpy as np

from musepipe.lines import build_windows


class LinesWindowTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.linspace(6500.0, 6650.0, 300)  # 0.5 A/ch
        self.catalog = [
            {"name": "Ha", "wave_A": 6562.8},
            {"name": "neighbor", "wave_A": 6548.0},  # falls in the blue continuum window
        ]

    def test_neighbor_line_excluded_from_continuum(self):
        _line, blue, red = build_windows(self.wave, self.catalog[0], catalog=self.catalog,
                                         neighbor_exclude_A=6.0, cont_gap_A=8.0, cont_width_A=40.0)
        # channels within 6 A of the neighbour must not be in the blue continuum mask
        near_neighbor = np.abs(self.wave - 6548.0) <= 6.0
        self.assertFalse(np.any(blue & near_neighbor))
        self.assertTrue(np.any(blue))  # but some blue continuum survives

    def test_bad_range_excluded_from_continuum(self):
        _line, blue, red = build_windows(self.wave, self.catalog[0], catalog=[self.catalog[0]],
                                         bad_ranges=[(6600.0, 6620.0)], cont_gap_A=8.0, cont_width_A=40.0)
        in_bad = (self.wave >= 6600.0) & (self.wave <= 6620.0)
        self.assertFalse(np.any(red & in_bad))

    def test_line_mask_centered_on_rest(self):
        line, _b, _r = build_windows(self.wave, self.catalog[0], catalog=[self.catalog[0]], line_half_A=5.0)
        self.assertTrue(np.all(np.abs(self.wave[line] - 6562.8) <= 5.0 + 1e-9))


if __name__ == "__main__":
    unittest.main()
