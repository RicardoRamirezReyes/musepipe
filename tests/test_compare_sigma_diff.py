import unittest

import numpy as np

from musepipe.stages.stage_x10_compare import METHOD_ORDER, empirical_sigma_diff


class CompareSigmaDiffTests(unittest.TestCase):
    def test_empirical_sigma_diff_recovers_own_noise_not_shared_noise(self):
        rng = np.random.default_rng(123)
        n_controls = 500
        n_wave = 64
        shared_sigma = 10.0
        own_sigma = 2.0
        shared = rng.normal(0.0, shared_sigma, size=(n_controls, n_wave))
        method_a = shared + rng.normal(0.0, own_sigma, size=(n_controls, n_wave))
        method_b = shared + rng.normal(0.0, own_sigma, size=(n_controls, n_wave))

        sigma = empirical_sigma_diff(method_a, method_b, smooth_channels=1)
        recovered = float(np.nanmedian(sigma))
        analytic = np.sqrt(2.0) * own_sigma
        naive = float(
            np.nanmedian(
                np.sqrt(
                    np.nanstd(method_a, axis=0, ddof=1) ** 2
                    + np.nanstd(method_b, axis=0, ddof=1) ** 2
                )
            )
        )

        self.assertLess(abs(recovered - analytic) / analytic, 0.12)
        self.assertLess(recovered, 0.35 * naive)

    def test_control_centred_t_is_calibrated_on_known_noise(self):
        from musepipe.stages.stage_x10_compare import compare_methods
        from tests.test_compare_verdicts import make_product, make_wave

        rng = np.random.default_rng(321)
        wave = make_wave()
        base = np.full(wave.size, 100.0, dtype=np.float64)
        products = {
            m: make_product(m, base, wave=wave)
            for m in METHOD_ORDER
        }
        controls = {
            m: rng.normal(0.0, 2.0, size=(30, wave.size))
            for m in METHOD_ORDER
        }
        _rows, control_rows, qc = compare_methods(products, controls)
        primary = [r for r in control_rows if r["pair"] == "psffit_vs_optimal_psfsub"]
        pvals = np.array([np.nan if r["p_ctrl"] is None else r["p_ctrl"] for r in primary])
        pvals = pvals[np.isfinite(pvals)]
        self.assertGreaterEqual(pvals.size, 200)
        # Independent noise: the t on control diffs should flag ~5% of rows.
        bad_fraction = float(np.mean(pvals < 0.0455))
        self.assertLess(bad_fraction, 0.12)
        self.assertEqual(qc["verdict"], "consistent")
        for row in _rows:
            if row["pair"] == "psffit_vs_optimal_psfsub" and row["n_chan_used"] > 0:
                self.assertEqual(row["df"], 29)


if __name__ == "__main__":
    unittest.main()
