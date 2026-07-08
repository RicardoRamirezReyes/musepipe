"""G1 §7: bias-budget quadrature and propagation on analytic cases."""

import math
import unittest

from musepipe.covariance import combine_budget_quadrature


class G1BudgetMathTests(unittest.TestCase):
    def test_signed_values_add_errors_in_quadrature(self):
        terms = [
            {"value_frac": 0.03, "err_frac": 0.04},
            {"value_frac": -0.01, "err_frac": 0.03},
        ]
        out = combine_budget_quadrature(terms)
        self.assertAlmostEqual(out["total_value_frac"], 0.02, places=12)
        self.assertAlmostEqual(out["total_err_frac"], math.hypot(0.04, 0.03), places=12)

    def test_empty_budget_is_zero(self):
        out = combine_budget_quadrature([])
        self.assertEqual(out["total_value_frac"], 0.0)
        self.assertEqual(out["total_err_frac"], 0.0)

    def test_missing_fields_default_to_zero(self):
        out = combine_budget_quadrature([{"value_frac": 0.05}, {"err_frac": 0.02}])
        self.assertAlmostEqual(out["total_value_frac"], 0.05, places=12)
        self.assertAlmostEqual(out["total_err_frac"], 0.02, places=12)


if __name__ == "__main__":
    unittest.main()
