"""WP-G3R-11: consistency σ + frozen verdicts."""

import unittest

import numpy as np

from musepipe.models.consistency import (
    consistency_row, not_computed_row, sigma_discrepancy, verdict,
)


class ConsistencyTests(unittest.TestCase):
    def test_sigma_and_verdicts(self):
        self.assertAlmostEqual(sigma_discrepancy(10.0, 1.0, 11.0, 0.0), 1.0)
        self.assertEqual(verdict(1.5), "consistent")
        self.assertEqual(verdict(2.5), "tension")
        self.assertEqual(verdict(3.5), "discrepant")
        self.assertEqual(verdict(float("nan")), "not_computed")

    def test_row_thresholds(self):
        r = consistency_row("a", 10.0, 1.0, "b", 14.0, 1.0)  # 4/sqrt(2)=2.83 sigma
        self.assertEqual(r["verdict"], "tension")
        r2 = consistency_row("a", 10.0, 0.5, "b", 20.0, 0.5)  # far -> discrepant
        self.assertEqual(r2["verdict"], "discrepant")
        r3 = consistency_row("a", 10.0, 2.0, "b", 11.0, 2.0)  # close -> consistent
        self.assertEqual(r3["verdict"], "consistent")

    def test_nan_inputs(self):
        r = consistency_row("a", float("nan"), 1.0, "b", 5.0, 1.0)
        self.assertEqual(r["verdict"], "not_computed")
        self.assertEqual(not_computed_row("a", "b", "pending")["verdict"], "not_computed")

    def test_zero_error_identical(self):
        self.assertEqual(sigma_discrepancy(5.0, 0.0, 5.0, 0.0), 0.0)
        self.assertEqual(sigma_discrepancy(5.0, 0.0, 6.0, 0.0), float("inf"))


if __name__ == "__main__":
    unittest.main()
