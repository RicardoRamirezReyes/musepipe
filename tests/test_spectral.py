import unittest

import numpy as np

from musepipe.spectral import (
    continuum_running_median,
    make_wavelength_mask,
    nearest_channel_index,
    nearest_channel_indices,
)


class SpectralTests(unittest.TestCase):
    def test_nearest_channel_ignores_nonfinite_wavelengths(self):
        waves = np.array([5000.0, 5001.0, np.nan, 5003.0])
        self.assertEqual(nearest_channel_index(waves, 5002.6), 3)
        np.testing.assert_array_equal(nearest_channel_indices(waves, [5000.2, 5002.6]), [0, 3])

    def test_nearest_channel_rejects_bad_inputs(self):
        with self.assertRaises(RuntimeError):
            nearest_channel_index([np.nan], 5000.0)
        with self.assertRaises(RuntimeError):
            nearest_channel_index([5000.0], np.nan)

    def test_continuum_running_median_respects_window_and_mask(self):
        waves = np.arange(5.0)
        spec = np.arange(5.0)
        good = np.ones(5, dtype=bool)
        continuum = continuum_running_median(waves, spec, good, window_A=2.0, min_pixels=1)
        np.testing.assert_allclose(continuum, [0.5, 1.0, 2.0, 3.0, 3.5])

        good[2] = False
        continuum = continuum_running_median(waves, spec, good, window_A=10.0, min_pixels=1)
        np.testing.assert_allclose(continuum, np.full(5, 2.0))

    def test_make_wavelength_mask_applies_bounds_and_drop_range(self):
        waves = np.array([5700.0, 5800.0, 6060.0, 7000.0, np.nan])
        cfg = {"drop_wave_min_A": 5780.0, "drop_wave_max_A": 6050.0}
        mask = make_wavelength_mask(waves, cfg, wave_min_A=5600.0, wave_max_A=7100.0, min_channels=2)
        np.testing.assert_array_equal(mask, [True, False, True, True, False])

    def test_make_wavelength_mask_errors_when_too_few_channels_remain(self):
        with self.assertRaises(RuntimeError):
            make_wavelength_mask([5800.0, 5900.0], {"drop_wave_min_A": 5700.0, "drop_wave_max_A": 6000.0})


if __name__ == "__main__":
    unittest.main()
