import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.qc.halpha_map import (
    compute_halpha_map,
    fit_halpha_spaxel,
    halpha_structure_metrics,
    main,
    phi_model,
    stripe_halpha_correlation,
)


STEP_A = 1.25
WAVES = np.arange(6480.0, 6650.0 + STEP_A, STEP_A)
MU_TRUE = 6562.8


def spaxel_spectrum(a, sigma, b=100.0, mu=MU_TRUE):
    return phi_model(WAVES, b, a, mu, sigma)


def make_cube(column_a, sigma_true=2.5, ny=12, noise_sigma=0.05, seed=3):
    """Cube (nlam, ny, nx) where column ix has line amplitude column_a[ix]."""
    rng = np.random.default_rng(seed)
    nx = len(column_a)
    cube = np.empty((WAVES.size, ny, nx))
    for ix, a in enumerate(column_a):
        base = spaxel_spectrum(a, sigma_true)
        for iy in range(ny):
            cube[:, iy, ix] = base + rng.normal(0.0, noise_sigma * 100.0, WAVES.size)
    return cube


class TestFit(unittest.TestCase):
    def test_recovers_known_a_and_sigma(self):
        res = fit_halpha_spaxel(WAVES, spaxel_spectrum(0.8, 3.0))
        self.assertEqual(res["success"], 1.0)
        self.assertAlmostEqual(res["a"], 0.8, delta=0.05 * 0.8)
        self.assertAlmostEqual(res["sigma_A"], 3.0, delta=0.05 * 3.0)
        self.assertAlmostEqual(res["mu_A"], MU_TRUE, delta=0.5)

    def test_lineless_spectrum_fails_clean(self):
        rng = np.random.default_rng(1)
        flat = 100.0 + rng.normal(0.0, 1.0, WAVES.size)
        res = fit_halpha_spaxel(WAVES, flat)
        self.assertEqual(res["success"], 0.0)
        self.assertTrue(np.isnan(res["a"]))
        self.assertTrue(np.isnan(res["sigma_A"]))

    def test_all_nan_spectrum_fails_clean(self):
        res = fit_halpha_spaxel(WAVES, np.full(WAVES.size, np.nan))
        self.assertEqual(res["success"], 0.0)
        self.assertTrue(np.isnan(res["P"]))


class TestMap(unittest.TestCase):
    def test_recovers_column_amplitudes(self):
        col_a = [0.4] * 5 + [0.9] * 5 + [0.6] * 5
        result = compute_halpha_map(make_cube(col_a), WAVES, brightness_percentile=0.0)
        a_map = result["a_map"]
        # column medians recover the injected amplitudes within 5%
        col_med = np.nanmedian(a_map, axis=0)
        np.testing.assert_allclose(col_med, np.asarray(col_a), rtol=0.05)
        self.assertFalse(result["fail_mask"].all())
        self.assertAlmostEqual(result["channel_step_A"], STEP_A, places=6)

    def test_structure_metrics_keys_and_correlation(self):
        # a and sigma anti-correlated by column, at (near) constant P
        col_a = np.linspace(0.4, 1.2, 12)
        # choose sigma so that a*sigma ~ const -> P roughly const (b fixed)
        col_sigma = 0.8 / col_a * 3.0
        rng = np.random.default_rng(5)
        cube = np.empty((WAVES.size, 8, 12))
        for ix in range(12):
            base = spaxel_spectrum(col_a[ix], col_sigma[ix])
            for iy in range(8):
                cube[:, iy, ix] = base + rng.normal(0.0, 3.0, WAVES.size)
        result = compute_halpha_map(cube, WAVES, brightness_percentile=0.0)
        m = halpha_structure_metrics(result, "vertical")
        for key in ("a_p95", "sigma_median_A", "sigma_structure_significance",
                    "a_structure_significance", "corr_a_sigma", "P_cov", "n_fit"):
            self.assertIn(key, m)
        # injected anti-correlation must show up as a strong negative corr
        self.assertLess(m["corr_a_sigma"], -0.5)


class TestStripeCorrelation(unittest.TestCase):
    def test_detects_column_aligned_correlation(self):
        # offset-map column scatter grows with column; Halpha sigma also grows
        # with column -> strong positive corr(stripe scatter, sigma).
        ny, nx = 30, 20
        rng = np.random.default_rng(2)
        offset = np.empty((ny, nx))
        sigma = np.empty((ny, nx))
        for ix in range(nx):
            offset[:, ix] = rng.normal(0.0, 0.02 + 0.1 * ix / nx, ny)
            sigma[:, ix] = 2.0 + 1.5 * ix / nx
        a_map = np.full((ny, nx), 0.5)
        mu_map = np.full((ny, nx), 6562.8)
        c = stripe_halpha_correlation(offset, a_map, sigma, mu_map, "vertical")
        self.assertGreater(c["corr_stripe_scatter_vs_halpha_sigma"], 0.8)
        self.assertGreaterEqual(c["n_columns"], nx - 2)

    def test_independent_maps_uncorrelated(self):
        rng = np.random.default_rng(9)
        offset = rng.normal(0.0, 0.05, (30, 20))
        sigma = rng.normal(2.4, 0.3, (30, 20))
        a_map = rng.normal(0.5, 0.1, (30, 20))
        mu_map = np.full((30, 20), 6562.8)
        c = stripe_halpha_correlation(offset, a_map, sigma, mu_map, "vertical")
        self.assertLess(abs(c["corr_stripe_scatter_vs_halpha_sigma"]), 0.6)


class TestCLI(unittest.TestCase):
    def test_cli_end_to_end(self):
        from astropy.io import fits

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cube = make_cube([0.5] * 6 + [1.0] * 6)
            hdu = fits.PrimaryHDU(data=cube.astype(np.float32))
            hdu.header["CRVAL3"] = float(WAVES[0])
            hdu.header["CD3_3"] = STEP_A
            hdu.header["CRPIX3"] = 1.0
            cube_fits = tmp / "cube.fits"
            hdu.writeto(cube_fits, overwrite=True)

            qc_out = tmp / "stageS1_qc.json"
            map_out = tmp / "stageS1_map.fits"
            plot_out = tmp / "stageS1.png"
            rc = main([
                "--cube", str(cube_fits), "--qc-output", str(qc_out),
                "--map-output", str(map_out), "--plot-output", str(plot_out),
                "--brightness-percentile", "0",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(qc_out.exists() and map_out.exists() and plot_out.exists())
            qc = json.loads(qc_out.read_text())
            for key in ("cube", "sha256", "binning", "windows", "channel_step_A",
                        "halpha_map", "runtime_s"):
                self.assertIn(key, qc)
            with fits.open(map_out) as hdul:
                names = [h.name for h in hdul]
            for ext in ("A", "SIGMA_A", "MU_A", "POWER", "SELECT"):
                self.assertIn(ext, names)


if __name__ == "__main__":
    unittest.main()
