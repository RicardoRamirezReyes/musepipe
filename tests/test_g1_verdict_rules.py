"""G1 §7: per-method verdict rules, including the 5% boundary and rejected cases."""

import unittest

from musepipe.covariance import method_verdict


class G1VerdictRulesTests(unittest.TestCase):
    def test_small_bias_low_sensitivity_is_validated(self):
        self.assertEqual(method_verdict(0.03, 0.005, 0.01), "validated")

    def test_bias_above_threshold_but_stable_is_validated_with_bias(self):
        self.assertEqual(method_verdict(0.08, 0.02, 0.03, bias_stable=True), "validated_with_bias")

    def test_sensitivity_above_stat_downgrades_from_validated(self):
        # bias below 5% but sensitivity exceeds statistical sigma -> not clean validated
        self.assertEqual(method_verdict(0.02, 0.05, 0.01, bias_stable=True), "validated_with_bias")

    def test_unbounded_bias_is_rejected(self):
        self.assertEqual(method_verdict(0.02, 0.005, 0.01, bias_bounded=False), "rejected")

    def test_unstable_bias_above_threshold_is_rejected(self):
        self.assertEqual(method_verdict(0.20, 0.10, 0.02, bias_stable=False), "rejected")

    def test_boundary_just_below_five_percent(self):
        self.assertEqual(method_verdict(0.0499, 0.005, 0.01), "validated")
        self.assertEqual(method_verdict(0.0501, 0.005, 0.01, bias_stable=True), "validated_with_bias")


if __name__ == "__main__":
    unittest.main()
