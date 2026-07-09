import unittest

import numpy as np

from musepipe.stages.stage_x10_compare import compare_methods
from tests.test_compare_verdicts import clean_controls, make_products, make_wave, synthetic_g1


class CompareControlsGateTests(unittest.TestCase):
    def test_biased_controls_on_only_primary_pair_make_verdict_uninterpretable(self):
        wave = make_wave()
        controls = clean_controls(wave)
        controls["psffit"] = controls["psffit"].copy()
        controls["psffit"][:, (wave >= 4900.0) & (wave <= 5400.0)] += 10.0
        products = make_products({"psffit": [(4900.0, 5400.0, 5.0), (6100.0, 6400.0, 5.0)]})

        _rows, _controls, qc = compare_methods(products, controls)

        self.assertEqual(qc["verdict"], "uninterpretable")
        self.assertEqual(qc["reason"], "all_primary_pairs_degraded")
        self.assertIn("psffit_vs_optimal_psfsub", qc["pairs_degraded"])

    def test_secondary_pair_dirty_does_not_degrade_verdict(self):
        wave = make_wave()
        controls = clean_controls(wave)
        controls["aperture"] = controls["aperture"].copy()
        controls["aperture"][:, (wave >= 4900.0) & (wave <= 5400.0)] += 10.0
        products = make_products()

        _rows, _controls, qc = compare_methods(products, controls)

        self.assertEqual(qc["verdict"], "consistent")
        self.assertEqual(qc["pairs_degraded"], {})
        self.assertFalse(qc["scale_check"]["psffit_vs_aperture"]["ok"])
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
        controls = clean_controls(wave)
        controls["aperture"] = controls["aperture"].copy()
        controls["aperture"][:, (wave >= 4900.0) & (wave <= 5400.0)] += 10.0
        products = make_products()

        _rows, _controls, qc = compare_methods(products, controls, g1_inputs=g1)

        self.assertIn("psffit_vs_aperture", qc["primary_pairs"])
        self.assertIn("psffit_vs_aperture", qc["pairs_degraded"])
        self.assertIn("optimal_psfsub_vs_aperture", qc["pairs_degraded"])
        self.assertEqual(qc["verdict"], "consistent")
        self.assertNotIn("psffit_vs_optimal_psfsub", qc["pairs_degraded"])


if __name__ == "__main__":
    unittest.main()
