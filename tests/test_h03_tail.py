import unittest

import numpy as np

from musepipe.stages.stage_h03_limits import ONE_SIDED_5SIGMA_FAP, fit_gumbel_moments, gumbel_isf, tail_limit_summary


class H03TailTests(unittest.TestCase):
    def test_gumbel_tail_recovers_synthetic_quantiles(self):
        rng = np.random.default_rng(12)
        loc = 1.7
        beta = 0.42
        samples = rng.gumbel(loc=loc, scale=beta, size=80000)

        fit_loc, fit_beta = fit_gumbel_moments(samples)
        q99_fit = gumbel_isf(0.01, fit_loc, fit_beta)
        q99_expected = gumbel_isf(0.01, loc, beta)

        self.assertAlmostEqual(fit_loc, loc, delta=0.02)
        self.assertAlmostEqual(fit_beta, beta, delta=0.02)
        self.assertAlmostEqual(q99_fit, q99_expected, delta=0.08)

    def test_five_sigma_tail_is_marked_as_extrapolated(self):
        values = np.linspace(-0.5, 2.5, 64)
        summary = tail_limit_summary(values, fap_99=0.01, fap_5sigma=ONE_SIDED_5SIGMA_FAP)

        self.assertTrue(summary["tail_extrapolated_5sigma"])
        self.assertLess(ONE_SIDED_5SIGMA_FAP, summary["minimum_resolvable_fap"])
        self.assertGreater(summary["z_5sigma_extrap"], summary["z_99_empirical"])


if __name__ == "__main__":
    unittest.main()
