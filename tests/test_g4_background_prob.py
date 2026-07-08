"""G4 §7: background contamination probability (Poisson, analytic)."""

import math
import unittest

from musepipe.classify import background_probability


class G4BackgroundProbTests(unittest.TestCase):
    def test_poisson_formula(self):
        # density 1e-4 /arcsec^2 over a 3.14 arcsec^2 area -> mu ~ 3.14e-4
        p = background_probability(1e-4, math.pi * 1.0 ** 2)
        self.assertAlmostEqual(p, 1.0 - math.exp(-1e-4 * math.pi), places=12)

    def test_zero_area_zero_prob(self):
        self.assertEqual(background_probability(1e-3, 0.0), 0.0)

    def test_monotone_in_density_and_area(self):
        self.assertLess(background_probability(1e-4, 1.0), background_probability(1e-3, 1.0))
        self.assertLess(background_probability(1e-4, 1.0), background_probability(1e-4, 10.0))

    def test_high_density_saturates_toward_one(self):
        self.assertGreater(background_probability(10.0, 5.0), 0.99)


if __name__ == "__main__":
    unittest.main()
