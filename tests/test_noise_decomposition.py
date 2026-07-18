import unittest

import numpy as np

from musepipe.qc.noise_decomposition import (
    build_lambda_filters,
    compute_noise_decomposition,
    fit_noise_model,
)


STEP = 1.25
WAVES = np.arange(6520.0, 6620.0, STEP)


def primary_flux(waves):
    return 1000.0 * (1.0 + 0.5 * np.sin((waves - 6520.0) / 20.0))


def make_residual_cube(waves, fstar, *, halpha_boost=1.0, ny=80, nx=80, seed=4):
    """Cube whose per-pixel noise std ~ sqrt(F) (photon-like), optionally with
    extra noise injected in the Halpha window."""
    rng = np.random.default_rng(seed)
    cube = np.empty((waves.size, ny, nx))
    k0 = 0.02
    for k, (w, f) in enumerate(zip(waves, fstar)):
        s = k0 * np.sqrt(max(f, 1.0))
        if halpha_boost != 1.0 and 6540.0 <= w <= 6590.0:
            s *= halpha_boost
        cube[k] = rng.normal(0.0, s, (ny, nx))
    return cube


class TestFilters(unittest.TestCase):
    def test_halpha_flagged_and_excluded(self):
        filt = build_lambda_filters(WAVES, lo_A=6520, hi_A=6618, width_ch=3)
        halpha = [f for f in filt if f["is_halpha"]]
        self.assertTrue(halpha)
        self.assertTrue(all(6540 - 25 <= f["center_A"] <= 6590 + 25 for f in halpha))
        non = [f for f in filt if not f["is_halpha"]]
        self.assertGreaterEqual(len(non), 4)


class TestFit(unittest.TestCase):
    def test_recovers_photon_power_law(self):
        F = np.linspace(200.0, 2000.0, 40)
        sigma = 0.7 * F ** 0.5  # pure photon, negligible bg
        fit = fit_noise_model(F, sigma)
        self.assertAlmostEqual(fit["alpha"], 0.5, delta=0.05)
        self.assertAlmostEqual(fit["c"], 0.7, delta=0.1)


class TestDecomposition(unittest.TestCase):
    def _run(self, halpha_boost):
        fstar = primary_flux(WAVES)
        cube = make_residual_cube(WAVES, fstar, halpha_boost=halpha_boost)
        return compute_noise_decomposition(
            cube, WAVES, fstar, primary_yx=(40, 40), r_B_px=25.0,
            aperture_radius_px=3.0, n_angles=36, lo_A=6520, hi_A=6618)

    def test_pure_photon_factor_near_one(self):
        qc = self._run(1.0)
        self.assertAlmostEqual(qc["alpha"], 0.5, delta=0.2)
        self.assertGreater(qc["factor_over_photon_at_halpha_rB"], 0.6)
        self.assertLess(qc["factor_over_photon_at_halpha_rB"], 1.6)

    def test_injected_systematic_recovered(self):
        qc = self._run(3.0)
        self.assertGreater(qc["factor_over_photon_at_halpha_rB"], 1.8)

    def test_qc_shape(self):
        qc = self._run(1.0)
        for k in ("c", "alpha", "sigma_bg", "factor_over_photon_at_halpha_rB",
                  "per_lambda_table", "n_fit", "r_B_px"):
            self.assertIn(k, qc)
        self.assertTrue(any(t["is_halpha"] for t in qc["per_lambda_table"]))


if __name__ == "__main__":
    unittest.main()
