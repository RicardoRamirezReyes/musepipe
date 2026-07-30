import unittest
from unittest.mock import patch

import numpy as np

from musepipe.stages.stage01c_localize import (
    AmbiguousDetectionError,
    DetectionResult,
    _merge_sub_fwhm_peaks,
    detect_restricted_source,
    locate_companion,
)
from tests.test_stage01c_synthetic import gaussian2d


class Stage01cConfusionTests(unittest.TestCase):
    def test_ambiguous_red_band_uses_configured_fallback(self):
        image = np.zeros((20, 20), dtype=np.float32)
        detection = DetectionResult(
            pos_yx=(12.0, 8.0),
            err_px=0.1,
            snr=20.0,
            peak_yx=(12, 8),
            band_used_A=(7500.0, 9000.0),
            snr_map=image,
            image=image,
        )
        cfg = {
            "expected_sep_arcsec": 1.0,
            "expected_pa_deg": 0.0,
            "stage01c_red_band_A": [8800.0, 9350.0],
            "stage01c_fallback_bands_A": [[7500.0, 9000.0]],
        }
        with patch(
            "musepipe.stages.stage01c_localize.detect_restricted_source",
            side_effect=[AmbiguousDetectionError("ambiguous"), detection],
        ) as detect:
            with patch("musepipe.stages.stage01c_localize.collapse_band", return_value=image):
                with patch(
                    "musepipe.stages.stage01c_localize.subtract_azimuthal_median",
                    return_value=(image, image),
                ):
                    result, _ = locate_companion(
                        image[None],
                        np.array([8000.0]),
                        {"pos_yx": (10.0, 10.0)},
                        cfg,
                        pixel_scale=0.1,
                        north_angle=0.0,
                        fwhm_px=3.0,
                    )
        self.assertIs(result, detection)
        self.assertEqual(detect.call_count, 2)

    def test_two_significant_sources_inside_tolerance_raise(self):
        ny, nx = 60, 60
        image = np.zeros((ny, nx), dtype=np.float64)
        image += gaussian2d(ny, nx, 28.0, 28.0, 25.0, 1.2)
        image += gaussian2d(ny, nx, 32.0, 34.0, 23.0, 1.2)
        image += np.random.default_rng(4).normal(0.0, 0.02, size=image.shape)

        # Sources 7.2 px apart with fwhm 3 px stay resolved -> still ambiguous.
        with self.assertRaises(AmbiguousDetectionError):
            detect_restricted_source(
                image,
                predicted_yx=(30.0, 31.0),
                search_radius_px=8.0,
                fwhm_px=3.0,
                snr_min=5.0,
                primary_yx=(30.0, 30.0),
            )

    def test_merge_collapses_sub_fwhm_peaks_to_strongest(self):
        snr = np.zeros((60, 60), dtype=np.float64)
        snr[30, 30] = 9.0
        snr[31, 32] = 8.0
        snr[29, 31] = 7.0
        coords = [(31, 32), (29, 31), (30, 30)]
        merged = _merge_sub_fwhm_peaks(coords, snr, merge_radius_px=5.0)
        self.assertEqual(merged, [(30, 30)])

    def test_merge_keeps_peaks_beyond_one_fwhm(self):
        snr = np.zeros((60, 60), dtype=np.float64)
        snr[30, 30] = 9.0
        snr[30, 40] = 8.0
        merged = _merge_sub_fwhm_peaks([(30, 30), (30, 40)], snr, merge_radius_px=5.0)
        self.assertEqual(merged, [(30, 30), (30, 40)])

    def test_merge_is_greedy_from_the_strongest(self):
        # A chain of peaks 3 px apart: the strongest anchors, the 3-px neighbour
        # merges into it, and the 6-px one survives as a second source.
        snr = np.zeros((60, 60), dtype=np.float64)
        snr[30, 30] = 9.0
        snr[30, 33] = 8.0
        snr[30, 36] = 7.0
        merged = _merge_sub_fwhm_peaks([(30, 30), (30, 33), (30, 36)], snr, merge_radius_px=4.0)
        self.assertEqual(merged, [(30, 30), (30, 36)])


if __name__ == "__main__":
    unittest.main()
