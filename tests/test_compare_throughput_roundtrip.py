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
    def test_throughput_correction_recovers_common_scale_in_memory_only(self):
        tmap = {"psffit": 0.70, "optimal_psfsub": 0.68}
        g1 = synthetic_g1(throughput=tmap)
        products = attenuated_products(throughput=tmap)
        flux_before = {m: p.flux.copy() for m, p in products.items()}

        rows, _controls, qc = compare_methods(products, clean_controls(), g1_inputs=g1)

        primary = [r for r in rows if r["pair"] == "psffit_vs_optimal_psfsub"]
        self.assertTrue(primary)
        for row in primary:
            if row["n_chan_used"] == 0:
                continue
            # Raw fluxes stay attenuated; corrected fluxes recover the diff ~ 0.
            self.assertAlmostEqual(row["flux_i_corr"], row["flux_i"] / 0.70, places=6)
            width_total = row["flux_i_corr"] * 0.0 + abs(row["flux_i_corr"])
            self.assertLess(abs(row["diff_corr"]), 1e-6 * max(width_total, 1.0))
            self.assertEqual(row["throughput_source_i"], "g1_bias_budget.throughput_loss")
        self.assertEqual(qc["verdict"], "consistent")
        self.assertTrue(qc["throughput_correction"]["applied_in_memory"])

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
            self.assertAlmostEqual(row["flux_j_corr"], row["flux_j"], places=9)


if __name__ == "__main__":
    unittest.main()
