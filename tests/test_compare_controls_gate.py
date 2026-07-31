import unittest

import numpy as np

from musepipe.stages.stage_x10_compare import compare_methods
from tests.test_compare_verdicts import clean_controls, make_products, make_wave, synthetic_g1


def rough_controls(wave, n_controls=8, sigma=1.0, seed=11):
    """Noisy (non-degenerate) controls shared by all methods plus own noise."""

    rng = np.random.default_rng(seed)
    out = {}
    for method in ("aperture", "optimal_ls", "optimal_psfsub", "psffit"):
        out[method] = rng.normal(0.0, sigma, size=(n_controls, wave.size))
    return out


class CompareControlsGateTests(unittest.TestCase):
    def test_single_correlated_outlier_does_not_create_binomial_excess_per_observable(self):
        wave = make_wave()
        controls = rough_controls(wave)
        # One control sits on an artifact: a strong offset in every band makes
        # its centred t an outlier in each band -> binomial excess -> dirty.
        controls["psffit"] = controls["psffit"].copy()
        controls["psffit"][0] += 50.0
        products = make_products()

        _rows, _controls, qc = compare_methods(products, controls)

        self.assertEqual(qc["verdict"], "consistent")
        self.assertEqual(qc["pairs_degraded"], {})

    def test_uniform_additive_bias_is_absorbed_by_centring(self):
        # A constant offset common to ALL controls of a method (comparable to
        # the control scatter, like the real psffit halo level) is that
        # method's additive bias at the same-radius ring: it affects object
        # and controls alike, and spec v2 §3.4 absorbs it with the mu
        # centring, so the pair stays clean AND consistent.
        wave = make_wave()
        controls = rough_controls(wave)
        controls["psffit"] = controls["psffit"] + 1.0
        products = make_products({"psffit": [(float(wave[0]), float(wave[-1]), 1.0)]})

        _rows, _controls, qc = compare_methods(products, controls)

        self.assertEqual(qc["verdict"], "consistent")
        self.assertEqual(qc["pairs_degraded"], {})
        self.assertTrue(qc["scale_check"]["psffit_vs_optimal_psfsub"]["ok"])

    def test_secondary_pair_dirty_does_not_degrade_verdict(self):
        wave = make_wave()
        controls = rough_controls(wave)
        controls["aperture"] = controls["aperture"].copy()
        controls["aperture"][0] += 50.0
        products = make_products()

        _rows, _controls, qc = compare_methods(products, controls)

        self.assertEqual(qc["verdict"], "consistent")
        self.assertEqual(qc["pairs_degraded"], {})
        self.assertEqual(qc["verdict_by_pair"]["psffit_vs_aperture"]["role"], "secondary")

    def test_one_dirty_primary_of_three_keeps_verdict_from_clean_pairs(self):
        g1 = synthetic_g1(
            verdicts={
                "psffit": "validated_with_bias",
                "optimal_psfsub": "validated_with_bias",
                "aperture": "validated",
                "optimal_ls": "rejected",
            }
        )
        wave = make_wave()
        controls = rough_controls(wave)
        controls["aperture"] = controls["aperture"].copy()
        controls["aperture"][0] += 50.0
        products = make_products()

        _rows, _controls, qc = compare_methods(products, controls, g1_inputs=g1)

        self.assertIn("psffit_vs_aperture", qc["primary_pairs"])
        self.assertNotIn("psffit_vs_aperture", qc["pairs_degraded"])
        self.assertNotIn("optimal_psfsub_vs_aperture", qc["pairs_degraded"])
        self.assertEqual(qc["verdict"], "consistent")
        self.assertNotIn("psffit_vs_optimal_psfsub", qc["pairs_degraded"])


if __name__ == "__main__":
    unittest.main()
