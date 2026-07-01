"""Tests for musepipe.plotting.smoothing_label (extracted in O2)."""

import unittest

from musepipe.plotting import smoothing_label


class SmoothingLabel(unittest.TestCase):
    def test_positive_sigma(self):
        self.assertEqual(smoothing_label(2.0), "smoothed: Gaussian sigma=2 px")
        self.assertEqual(smoothing_label(1.5), "smoothed: Gaussian sigma=1.5 px")

    def test_none_or_zero_is_raw(self):
        self.assertEqual(smoothing_label(None), "raw pixels (no smoothing)")
        self.assertEqual(smoothing_label(0), "raw pixels (no smoothing)")
        self.assertEqual(smoothing_label(-1), "raw pixels (no smoothing)")


if __name__ == "__main__":
    unittest.main()
