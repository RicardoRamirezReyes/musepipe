import unittest

import numpy as np
from scipy import stats

from musepipe.stages.stage08c_look_elsewhere import fit_gumbel_tail, parametric_fap
from musepipe.stages.stage_h03_limits import ONE_SIDED_5SIGMA_FAP, gumbel_isf, tail_limit_summary


class H03TailTests(unittest.TestCase):
    def test_gumbel_tail_recovers_synthetic_quantiles(self):
        rng = np.random.default_rng(12)
        loc = 1.7
        beta = 0.42
        samples = rng.gumbel(loc=loc, scale=beta, size=80000)

        fit_loc, fit_beta = fit_gumbel_tail(samples)
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


class H03SameTailAsE1Tests(unittest.TestCase):
    """El umbral del 99 % de E3 es el z al que la FAP de E1 vale 0.01.

    Hasta el 2026-10-20 E3 ajustaba la Gumbel por momentos y E1 por maxima
    verosimilitud: con 33 controles los umbrales diferian ~10 %, y en ROXs 42B b
    el flujo medido (z = 2.97) superaba el limite (z = 2.73) mientras E1 lo daba
    por no detectado (umbral 3.00). Estos tests fijan que es un solo ajuste.
    """

    def _null(self):
        # 33 maximos, como los controles de los dos runs del paper; la semilla da
        # una muestra en la que momentos y maxima verosimilitud difieren > 0.1 en z.
        rng = np.random.default_rng(7)
        values = rng.gumbel(loc=1.0, scale=0.44, size=33)
        b = np.std(values, ddof=1) * np.sqrt(6.0) / np.pi
        z_moments = np.mean(values) - np.euler_gamma * b - b * np.log(-np.log(0.99))
        z_mle = stats.gumbel_r.isf(0.01, *stats.gumbel_r.fit(values))
        self.assertGreater(abs(z_mle - z_moments), 0.1, "la muestra no distingue los dos ajustes")
        return values

    def test_threshold_is_the_e1_fit(self):
        values = self._null()
        summary = tail_limit_summary(values, fap_99=0.01)
        loc, scale = stats.gumbel_r.fit(values)

        self.assertEqual(summary["tail_model"], "gumbel_mle")
        self.assertAlmostEqual(summary["gumbel_loc"], loc, places=12)
        self.assertAlmostEqual(summary["gumbel_beta"], scale, places=12)
        self.assertAlmostEqual(parametric_fap(summary["z_99_gumbel"], values)["fap"], 0.01, places=9)

    def test_verdict_and_threshold_agree(self):
        values = self._null()
        z99 = tail_limit_summary(values, fap_99=0.01)["z_99_gumbel"]

        self.assertLess(parametric_fap(z99 + 0.01, values)["fap"], 0.01)
        self.assertGreater(parametric_fap(z99 - 0.01, values)["fap"], 0.01)


if __name__ == "__main__":
    unittest.main()
