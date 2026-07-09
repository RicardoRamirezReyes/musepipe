import unittest

import numpy as np

from musepipe.stages.stage_x10_compare import compare_methods
from tests.test_compare_verdicts import make_product, make_wave, synthetic_g1


def ar1_controls(rng, n_controls, n_chan, phi=0.5, sigma=1.0):
    """AR(1) channel noise with corr_length = (1+phi)/(1-phi)."""

    out = np.empty((n_controls, n_chan), dtype=np.float64)
    innov_sigma = sigma * np.sqrt(1.0 - phi**2)
    for k in range(n_controls):
        eps = rng.normal(0.0, innov_sigma, size=n_chan)
        x = np.empty(n_chan)
        x[0] = rng.normal(0.0, sigma)
        for i in range(1, n_chan):
            x[i] = phi * x[i - 1] + eps[i]
        out[k] = x
    return out


class NeffCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.wave = make_wave()
        self.phi = 0.5
        self.corr_length = (1.0 + self.phi) / (1.0 - self.phi)  # = 3.0
        rng = np.random.default_rng(42)
        base = np.full(self.wave.size, 100.0, dtype=np.float64)
        self.products = {
            m: make_product(m, base, wave=self.wave)
            for m in ("aperture", "optimal_ls", "optimal_psfsub", "psffit")
        }
        self.controls = {
            m: ar1_controls(rng, 30, self.wave.size, phi=self.phi)
            for m in ("aperture", "optimal_ls", "optimal_psfsub", "psffit")
        }
        self.g1 = synthetic_g1(corr_length=self.corr_length)

    def test_naive_z_is_miscalibrated_and_t_is_not_on_correlated_noise(self):
        rows, control_rows, qc = compare_methods(
            self.products, self.controls, g1_inputs=self.g1
        )
        primary_ctrl = [r for r in control_rows if r["pair"] == "psffit_vs_optimal_psfsub"]
        self.assertGreaterEqual(len(primary_ctrl), 200)
        zvals = np.array([np.nan if r["z"] is None else r["z"] for r in primary_ctrl])
        pvals = np.array([np.nan if r["p_ctrl"] is None else r["p_ctrl"] for r in primary_ctrl])
        z_bad = np.mean(np.abs(zvals[np.isfinite(zvals)]) >= 2.0)
        t_bad = np.mean(pvals[np.isfinite(pvals)] < 0.0455)
        # Independent-channel integration under AR(1) inflates z by ~sqrt(3):
        # the naive z flags far more than the nominal ~4.6%; the control-centred
        # t stays calibrated.
        self.assertGreater(z_bad, 0.12)
        self.assertLess(t_bad, 0.12)
        self.assertGreater(z_bad, 2.0 * max(t_bad, 0.02))
        self.assertEqual(qc["verdict"], "consistent")

    def test_z_neff_shrinks_z_by_sqrt_corr_length(self):
        products = dict(self.products)
        wave = self.wave
        flux = np.full(wave.size, 100.0)
        band = (wave >= 4900.0) & (wave <= 5400.0)
        flux = flux.copy()
        flux[band] += 3.0
        products["psffit"] = make_product("psffit", flux, wave=wave)

        rows, _c, _qc = compare_methods(products, self.controls, g1_inputs=self.g1)
        row = next(
            r for r in rows if r["pair"] == "psffit_vs_optimal_psfsub" and r["band"] == "B1"
        )
        # T=1 for every method here, so diff_corr == diff and sigma_corr == sigma:
        # z_neff must equal z / sqrt(corr_length) exactly.
        self.assertAlmostEqual(row["z_neff"], row["z"] / np.sqrt(self.corr_length), places=9)


if __name__ == "__main__":
    unittest.main()
