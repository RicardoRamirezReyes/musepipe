import unittest

import numpy as np

from musepipe.stages.stage_x10_compare import METHOD_ORDER, compare_methods
from tests.test_compare_verdicts import clean_controls, make_product, make_wave, synthetic_g1


def attenuated_products(true_level=100.0, throughput=None):
    """Products whose measured flux is T_method * true flux."""

    wave = make_wave()
    throughput = throughput or {}
    products = {}
    for method in METHOD_ORDER:
        t_val = float(throughput.get(method, 1.0))
        flux = np.full(wave.size, true_level * t_val, dtype=np.float64)
        products[method] = make_product(method, flux, wave=wave)
    return products


class ThroughputRoundtripTests(unittest.TestCase):
    def test_throughput_correction_is_line_only_and_products_stay_unchanged(self):
        tmap = {"psffit": 0.70, "optimal_psfsub": 0.68}
        g1 = synthetic_g1(throughput=tmap)
        products = attenuated_products(throughput=tmap)
        flux_before = {m: p.flux.copy() for m, p in products.items()}

        rows, _controls, qc = compare_methods(products, clean_controls(), g1_inputs=g1)

        primary = [r for r in rows if r["pair"] == "psffit_vs_optimal_psfsub"]
        self.assertTrue(primary)
        continuum = [row for row in primary if row["band_kind"] == "continuum" and row["n_chan_used"] > 0]
        lines = [row for row in primary if row["band_kind"] == "line" and row["n_chan_used"] > 0]
        self.assertTrue(continuum)
        self.assertTrue(lines)
        for row in continuum:
            self.assertEqual(row["flux_i_corr"], row["flux_i"])
            self.assertFalse(row["throughput_applied_i"])
        for row in lines:
            self.assertTrue(row["throughput_applied_i"])
            self.assertEqual(row["throughput_source_i"], "g1_bias_budget.throughput_loss")
        self.assertEqual(qc["throughput_correction"]["applied_in_memory"], "line_bands_only")

        # Products are NEVER modified: E3/G2 correct the raw products
        # downstream, so an in-place change here would double count.
        for method, before in flux_before.items():
            np.testing.assert_array_equal(products[method].flux, before)

    def test_rejected_methods_are_not_corrected(self):
        tmap = {"psffit": 0.70, "optimal_psfsub": 0.68, "aperture": 0.10}
        g1 = synthetic_g1(throughput=tmap)  # aperture stays rejected in verdicts
        products = attenuated_products(throughput={"psffit": 0.70, "optimal_psfsub": 0.68})

        rows, _controls, qc = compare_methods(products, clean_controls(), g1_inputs=g1)

        self.assertEqual(
            qc["throughput_correction"]["by_method"]["aperture"]["source"],
            "not_applied_rejected_method",
        )
        secondary = [r for r in rows if r["pair"] == "psffit_vs_aperture" and r["n_chan_used"] > 0]
        for row in secondary:
            self.assertFalse(row["throughput_applied_j"])


if __name__ == "__main__":
    unittest.main()
