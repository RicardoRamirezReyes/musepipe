import unittest
import warnings

import numpy as np

from stage08_full_spectrum_for_modeling import make_products_for_aperture


class Stage08FullSpectrumTests(unittest.TestCase):
    def test_masked_channels_are_nan_without_runtime_warnings(self):
        waves = np.arange(40, dtype=float)
        good = np.zeros(40, dtype=bool)
        good[:30] = True
        object_spec = np.linspace(1.0, 2.0, 40)
        controls = [
            np.linspace(0.8, 1.8, 40),
            np.linspace(1.0, 2.0, 40),
            np.linspace(1.2, 2.2, 40),
        ]
        object_spec[~good] = np.nan
        for control in controls:
            control[~good] = np.nan

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            products = make_products_for_aperture(
                waves,
                good,
                object_spec,
                controls,
                continuum_window_A=30.0,
            )

        runtime_warnings = [item for item in caught if issubclass(item.category, RuntimeWarning)]
        self.assertEqual(runtime_warnings, [])
        for key in (
            "object_local_residual",
            "control_median_local_residual",
            "control_sigma",
            "flux_native",
            "flux_continuum_sub",
            "snr_like",
        ):
            self.assertTrue(np.all(np.isfinite(products[key][good])), key)
            self.assertTrue(np.all(np.isnan(products[key][~good])), key)


if __name__ == "__main__":
    unittest.main()
