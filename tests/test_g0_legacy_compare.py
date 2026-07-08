"""G0 §5.3/§7: identical spectra → zero discrepancies; a scaled one → flagged."""

import unittest

import numpy as np

from musepipe.g0 import legacy_comparison


class G0LegacyCompareTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.linspace(6000.0, 7000.0, 200)
        self.flux = 100.0 + np.sin((self.wave - 6000.0) / 50.0)
        self.bands = [(6100.0, 6300.0), (6500.0, 6600.0), (6700.0, 6900.0)]

    def test_identical_spectra_have_no_flag(self):
        rows = legacy_comparison(self.wave, self.flux, self.wave, self.flux, self.bands)
        self.assertTrue(all(not r["flagged"] for r in rows))
        for r in rows:
            self.assertAlmostEqual(r["ratio_new_over_legacy"], 1.0, places=12)

    def test_scaled_spectrum_is_flagged(self):
        rows = legacy_comparison(self.wave, 5.0 * self.flux, self.wave, self.flux, self.bands, sigma_threshold=2.0)
        self.assertTrue(all(r["flagged"] for r in rows))
        for r in rows:
            self.assertAlmostEqual(r["ratio_new_over_legacy"], 5.0, places=10)

    def test_band_outside_coverage_is_nan_not_flagged(self):
        rows = legacy_comparison(self.wave, self.flux, self.wave, self.flux, [(4000.0, 4100.0)])
        self.assertTrue(np.isnan(rows[0]["ratio_new_over_legacy"]))
        self.assertFalse(rows[0]["flagged"])


if __name__ == "__main__":
    unittest.main()
