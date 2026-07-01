"""Synthetic tests for musepipe.stripes (refactor task O2a, first increment).

Covers the pure primitives extracted from 02_xcorr_stripes.ipynb: orientation
parsing, stripe-range construction/validation, mask building (axis-aligned and
projected), and the 1D spectral-shift primitives (integer, NaN-safe, and the
cross-correlation lag estimator).
"""

import unittest
import warnings

import numpy as np

from musepipe.stripes import (
    _axis_aligned_masks,
    _axis_from_stripe_angle,
    _build_projected_stripe_geometry,
    _build_stripe_geometry,
    _build_stripe_ranges,
    _build_stripe_ranges_flexible,
    _normalize_stripe_angle_deg,
    _normalize_stripe_orientation,
    _optional_float,
    _shift_spectral_integer,
    _shift_spectral_nan_safe,
    _stripe_group_key,
    _validate_manual_range_list_for_stage02,
    _validate_manual_stripe_ranges,
    _xcorr_shift_pixels,
    safe_float,
)


class ScalarHelpers(unittest.TestCase):
    def test_safe_float_filters_nonfinite_and_garbage(self):
        self.assertEqual(safe_float("3.5"), 3.5)
        self.assertIsNone(safe_float("abc"))
        self.assertIsNone(safe_float(np.inf))
        self.assertIsNone(safe_float(np.nan))

    def test_optional_float_passthrough_and_default(self):
        self.assertEqual(_optional_float(None, default=7.0), 7.0)
        self.assertEqual(_optional_float("2"), 2.0)

    def test_group_key_folds_modulo_180(self):
        self.assertEqual(_stripe_group_key(10.0), _stripe_group_key(190.0))


class OrientationParsing(unittest.TestCase):
    def test_orientation_aliases(self):
        self.assertEqual(_normalize_stripe_orientation("H"), "horizontal")
        self.assertEqual(_normalize_stripe_orientation("x"), "vertical")
        self.assertEqual(_normalize_stripe_orientation("oblique"), "angled")

    def test_invalid_orientation_raises(self):
        with self.assertRaises(ValueError):
            _normalize_stripe_orientation("sideways")

    def test_angle_normalization_folds_and_falls_back(self):
        self.assertAlmostEqual(_normalize_stripe_angle_deg(200.0), 20.0)
        # non-finite falls back to orientation default
        self.assertEqual(_normalize_stripe_angle_deg(np.nan, "vertical"), 0.0)
        self.assertEqual(_normalize_stripe_angle_deg(np.nan, "horizontal"), 90.0)

    def test_axis_from_angle(self):
        self.assertEqual(_axis_from_stripe_angle(0.0)[0], "x")
        self.assertEqual(_axis_from_stripe_angle(90.0)[0], "y")
        self.assertEqual(_axis_from_stripe_angle(32.0)[0], "angle")


class StripeRanges(unittest.TestCase):
    def test_build_ranges_are_contiguous_and_cover_axis(self):
        ranges, axis = _build_stripe_ranges(ny=20, nx=30, nstripes=6, orientation="vertical")
        self.assertEqual(axis, "x")
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 30)
        for (a1, a2), (b1, b2) in zip(ranges, ranges[1:]):
            self.assertEqual(a2, b1)  # no gaps, no overlaps

    def test_manual_ranges_reject_out_of_bounds(self):
        with self.assertRaises(ValueError):
            _validate_manual_stripe_ranges([(0, 40)], ny=20, nx=30, orientation="vertical")

    def test_flexible_manual_path_matches_validation(self):
        ranges, axis = _build_stripe_ranges_flexible(
            ny=20, nx=30, nstripes=3, orientation="horizontal",
            manual_stripe_ranges=[(0, 10), (10, 20)],
        )
        self.assertEqual(axis, "y")
        self.assertEqual(ranges, [(0, 10), (10, 20)])

    def test_stage02_range_list_rejects_overlap_and_wrong_count(self):
        with self.assertRaises(ValueError):
            _validate_manual_range_list_for_stage02([[0, 10], [5, 15]], nstripes=2, label="X")
        with self.assertRaises(ValueError):
            _validate_manual_range_list_for_stage02([[0, 10]], nstripes=2, label="X")

    def test_stage02_range_list_warns_on_gap(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = _validate_manual_range_list_for_stage02([[0, 10], [12, 20]], nstripes=2, label="X")
        self.assertEqual(out, [[0, 10], [12, 20]])
        self.assertTrue(any("gap" in str(x.message) for x in w))


class MaskBuilding(unittest.TestCase):
    def test_axis_aligned_masks_partition_frame(self):
        ny, nx = 20, 30
        ranges, axis = _build_stripe_ranges(ny, nx, 5, "vertical")
        masks = _axis_aligned_masks(ny, nx, ranges, axis)
        stacked = np.sum(masks, axis=0)
        # every pixel assigned exactly once
        np.testing.assert_array_equal(stacked, np.ones((ny, nx), dtype=int))

    def test_full_geometry_vertical_matches_axis_aligned(self):
        ny, nx = 16, 24
        ranges, axis, angle, masks = _build_stripe_geometry(ny, nx, 4, "vertical")
        self.assertEqual(axis, "x")
        self.assertAlmostEqual(angle, 0.0)
        stacked = np.sum(masks, axis=0)
        np.testing.assert_array_equal(stacked, np.ones((ny, nx), dtype=int))

    def test_projected_geometry_covers_all_pixels_once(self):
        ny, nx = 18, 26
        ranges, masks = _build_projected_stripe_geometry(ny, nx, nstripes=5, stripe_angle_deg=32.0)
        stacked = np.sum(masks, axis=0)
        # angled masks are contiguous half-open bins plus a closed last bin:
        # every pixel belongs to exactly one stripe
        np.testing.assert_array_equal(stacked, np.ones((ny, nx), dtype=int))


class SpectralShifts(unittest.TestCase):
    def test_integer_shift_roundtrip(self):
        block = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
        shifted = _shift_spectral_integer(block, 3)
        # first 3 channels become NaN, rest are the shifted originals
        self.assertTrue(np.all(np.isnan(shifted[:3, 0, 0])))
        np.testing.assert_array_equal(shifted[3:, 0, 0], block[:7, 0, 0])

    def test_nan_safe_shift_recovers_integer_shift(self):
        rng = np.random.default_rng(0)
        spec = rng.normal(size=(64, 2, 2)).astype(np.float32)
        shifted = _shift_spectral_nan_safe(spec, 4.0)
        # interior channels should match the original shifted by 4, up to interp error
        inner = shifted[8:-8]
        ref = spec[4:-12]
        self.assertLess(np.nanmax(np.abs(inner - ref)), 0.05)

    def test_xcorr_recovers_known_lag(self):
        rng = np.random.default_rng(1)
        base = rng.normal(size=400)
        # smooth so the correlation peak is well defined
        kernel = np.exp(-0.5 * (np.arange(-5, 6) / 2.0) ** 2)
        kernel /= kernel.sum()
        base = np.convolve(base, kernel, mode="same")
        lag = 3
        ref = base[:-lag]
        tgt = base[lag:]
        est = _xcorr_shift_pixels(ref, tgt, max_lag=6)
        self.assertAlmostEqual(est, -lag, delta=0.5)

    def test_xcorr_returns_nan_on_flat_input(self):
        flat = np.ones(100)
        self.assertTrue(np.isnan(_xcorr_shift_pixels(flat, flat)))


if __name__ == "__main__":
    unittest.main()
