"""WP-G3R-6: observed-spectrum preparation. Synthetic fixtures only — the real
spectrum is never rebinned here (anti-bias; that happens in WP-G3R-11)."""

import csv
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from musepipe.models.observed import (
    FitSpectrum,
    build_fit_masks,
    fit_spectrum,
    load_final_spectrum,
    plot_fit_spectrum,
    rebin_for_fit,
)


def _write_spectrum(stage, wave, flux, err, sys, *, drop=()):
    from astropy.io import fits
    cols = []
    for name, arr in (("wave_A", wave), ("flux", flux),
                      ("flux_err_total", err), ("sys_fluxcal", sys)):
        if name in drop:
            continue
        cols.append(fits.Column(name=name, format="E", array=np.asarray(arr, float)))
    hdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols), name="SPECTRUM")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(stage / "spec_final_object.fits")


def _make_run(tmp, *, n=100, wave0=6000.0, dl=1.0, flux_val=5.0, sigma=2.0,
              bad_idx=(), lines=(), block_bounds=None, neff=None, drop=()):
    stage = Path(tmp) / "stages"
    table = Path(tmp) / "tables"
    stage.mkdir(parents=True, exist_ok=True)
    table.mkdir(parents=True, exist_ok=True)
    wave = wave0 + dl * np.arange(n)
    flux = np.full(n, flux_val)
    err = np.full(n, sigma)
    _write_spectrum(stage, wave, flux, err, np.zeros(n), drop=drop)
    bad = np.zeros(n, bool)
    for i in bad_idx:
        bad[i] = True
    np.save(stage / "stage04b_bad_wavelength_mask.npy", bad)
    if block_bounds is None:
        block_bounds = np.array([[0, int((~bad).sum())]])
        neff = np.array([1.0])
    np.savez(stage / "g1_channel_covariance.npz",
             block_bounds=np.asarray(block_bounds),
             n_eff_over_n_by_block=np.asarray(neff, float))
    with (table / "g2_line_measurements.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "rest_A", "status"])
        for L in lines:
            w.writerow(["line", L, "upper_limit"])
    return types.SimpleNamespace(stage_dir=stage, table_dir=table,
                                 plot_dir=Path(tmp) / "plots")


def _cfg(**over):
    cfg = {
        "g3_fit_masks": {"use_stage04b_bad_mask": True, "line_window_kms": 300.0,
                         "telluric_bands_A": []},
        "g3_fit_bin_channels": 20, "g3_fit_bin_max_masked_frac": 0.5,
        "g3_fit_err_column": "flux_err_total",
        "g3_fit_wave_range_A": [6000.0, 6099.0],
    }
    cfg.update(over)
    return cfg


class LoadSchemaTests(unittest.TestCase):
    def test_load_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            rp = _make_run(tmp)
            d = load_final_spectrum(rp)
            self.assertEqual(d["n_channels"], 100)
            self.assertEqual(d["flux_err"].shape, (100,))

    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            rp = _make_run(tmp, drop=("sys_fluxcal",))
            with self.assertRaises(RuntimeError):
                load_final_spectrum(rp)


class MaskTests(unittest.TestCase):
    def test_line_window_masks_correct_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            rp = _make_run(tmp, n=200, wave0=6500.0, dl=0.5, lines=(6562.80,))
            wave = 6500.0 + 0.5 * np.arange(200)
            cfg = _cfg(g3_fit_wave_range_A=[6500.0, 6599.5])
            mask, prov = build_fit_masks(cfg, wave, rp)
            half = 6562.80 * 300.0 / 299792.458  # ~6.57 A
            inside = (wave >= 6562.80 - half) & (wave <= 6562.80 + half)
            self.assertTrue(np.all(mask[inside]))
            self.assertFalse(mask[np.argmin(np.abs(wave - 6520.0))])  # far channel free
            self.assertEqual(prov["line_windows"]["n_lines"], 1)

    def test_bad_mask_shape_mismatch_parada(self):
        with tempfile.TemporaryDirectory() as tmp:
            rp = _make_run(tmp, n=100)
            wave = np.linspace(6000.0, 6099.0, 80)  # wrong length
            with self.assertRaises(RuntimeError):
                build_fit_masks(_cfg(), wave, rp)


class RebinTests(unittest.TestCase):
    def _cov(self, block_bounds, neff):
        return {"block_bounds": np.asarray(block_bounds),
                "n_eff_over_n_by_block": np.asarray(neff, float)}

    def test_conserves_flux_constant(self):
        wave = 6000.0 + np.arange(100)
        flux = np.full(100, 5.0)
        err = np.full(100, 2.0)
        mask = np.zeros(100, bool)
        out = rebin_for_fit(wave, flux, err, mask, self._cov([[0, 100]], [1.0]),
                            n_channels=20, wave_range=[6000.0, 6099.0],
                            bad_mask=np.zeros(100, bool))
        self.assertEqual(out["n_bins"], 5)
        np.testing.assert_allclose(out["flux_bin"], 5.0)

    def test_variance_inflation_by_block(self):
        wave = 6000.0 + np.arange(100)
        flux = np.full(100, 5.0)
        err = np.full(100, 2.0)
        mask = np.zeros(100, bool)
        cov = self._cov([[0, 50], [50, 100]], [1.0, 0.5])
        out = rebin_for_fit(wave, flux, err, mask, cov, n_channels=20,
                            wave_range=[6000.0, 6099.0], bad_mask=np.zeros(100, bool))
        # block0 (n_eff/n=1): err = sigma/sqrt(20); block1 (0.5): x sqrt(2)
        self.assertAlmostEqual(out["err_bin"][0], 2.0 / np.sqrt(20.0), places=6)
        self.assertAlmostEqual(out["err_bin"][-1] / out["err_bin"][0],
                               np.sqrt(2.0), places=6)
        self.assertEqual(out["blocks_used"][0], 0)
        self.assertEqual(out["blocks_used"][-1], 1)

    def test_discards_heavily_masked_bin(self):
        wave = 6000.0 + np.arange(100)
        flux = np.full(100, 5.0)
        err = np.full(100, 2.0)
        mask = np.zeros(100, bool)
        mask[0:11] = True  # 11/20 masked in the first bin -> dropped
        out = rebin_for_fit(wave, flux, err, mask, self._cov([[0, 100]], [1.0]),
                            n_channels=20, wave_range=[6000.0, 6099.0],
                            bad_mask=np.zeros(100, bool))
        self.assertEqual(out["n_bins"], 4)

    def test_wave_range_excludes_out_of_range(self):
        wave = 6000.0 + np.arange(100)
        flux = np.full(100, 5.0)
        err = np.full(100, 2.0)
        mask = np.zeros(100, bool)
        out = rebin_for_fit(wave, flux, err, mask, self._cov([[0, 100]], [1.0]),
                            n_channels=20, wave_range=[6000.0, 6039.0],
                            bad_mask=np.zeros(100, bool))
        self.assertEqual(out["n_bins"], 2)  # only 40 channels in range

    def test_coverage_mismatch_parada(self):
        wave = 6000.0 + np.arange(100)
        flux = np.full(100, 5.0)
        err = np.full(100, 2.0)
        mask = np.zeros(100, bool)
        cov = self._cov([[0, 50], [50, 90]], [1.0, 0.5])  # covers 90 != 100 good
        with self.assertRaises(RuntimeError):
            rebin_for_fit(wave, flux, err, mask, cov, n_channels=20,
                          wave_range=[6000.0, 6099.0], bad_mask=np.zeros(100, bool))


class FitSpectrumTests(unittest.TestCase):
    def test_end_to_end_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            rp = _make_run(tmp, n=100, bad_idx=(3, 4))
            fs = fit_spectrum(_cfg(), rp)
            self.assertIsInstance(fs, FitSpectrum)
            self.assertGreater(fs.n_bins, 0)
            self.assertGreater(fs.n_eff, 0.0)
            self.assertIn("variance_rule", fs.mask_provenance)
            self.assertEqual(fs.mask_provenance["bad_mask"]["n_excluded"], 2)

    def test_plot_runs_on_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            rp = _make_run(tmp, n=100)
            fs = fit_spectrum(_cfg(), rp)
            wave = 6000.0 + np.arange(100)
            mask, _ = build_fit_masks(_cfg(), wave, rp)
            out = Path(tmp) / "g3_fit_spectrum.png"
            plot_fit_spectrum(out, wave=wave, flux=np.full(100, 5.0), mask=mask,
                              fit_spec=fs, wave_range=[6000.0, 6099.0])
            self.assertTrue(out.exists() and out.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
