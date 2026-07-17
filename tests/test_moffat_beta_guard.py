"""Regression: moffat_alpha_from_fwhm must not overflow on unphysical beta.

Smoke finding (LkCa 15, 2026-07-15): a degree-N beta(lambda) polynomial from
C1 can extrapolate to beta <= 0 at band edges outside its fit range (382/3681
channels on the LkCa 15 Moffat fit), which sent 2**(1/beta) to an
OverflowError and crashed every downstream aperture correction. The floor at
the Moffat normalizability limit (beta > 1) keeps apcorr finite and is a
strict no-op for any healthy PSF.
"""

import math
import unittest

from musepipe.psf import MOFFAT_BETA_FLOOR, moffat_alpha_from_fwhm


class MoffatBetaGuardTests(unittest.TestCase):
    def test_no_overflow_on_tiny_or_negative_beta(self):
        for beta in (1e-4, 1e-6, 0.0, -2.96, float("nan")):
            alpha = moffat_alpha_from_fwhm(4.0, beta)
            self.assertTrue(math.isfinite(alpha) and alpha > 0, f"beta={beta}")

    def test_no_op_for_healthy_beta(self):
        # Well above the floor: value must be exactly the unclamped formula.
        for beta in (1.5, 2.8, 4.0, 6.98):
            denom = 2.0 * math.sqrt(max(2.0 ** (1.0 / beta) - 1.0, 1e-12))
            self.assertAlmostEqual(moffat_alpha_from_fwhm(4.0, beta), 4.0 / denom, places=12)

    def test_floor_is_continuous(self):
        # Just below the floor clamps to exactly the floor value.
        at_floor = moffat_alpha_from_fwhm(4.0, MOFFAT_BETA_FLOOR)
        self.assertAlmostEqual(moffat_alpha_from_fwhm(4.0, 0.5), at_floor, places=12)


if __name__ == "__main__":
    unittest.main()
