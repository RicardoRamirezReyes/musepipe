import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.qc.wavesol_map import (
    brightness_selection_mask,
    build_reference_spectrum,
    compute_offset_map,
    evaluate_gate_g1,
    main,
    normalize_window,
    normalize_window_cube,
    parse_windows_arg,
    spaxel_brightness_map,
    stripe_profile,
    structure_metrics,
    window_slices,
)


STEP_A = 1.25
WAVES = np.arange(6100.0, 6900.0, STEP_A)
TEST_WINDOWS = ((6150.0, 6450.0), (6620.0, 6850.0))


def stellar_spectrum(waves):
    """Continuum with a sloped shape and a comb of absorption lines."""
    cont = 100.0 * (1.0 + 2e-4 * (waves - 6500.0))
    spec = cont.copy()
    rng = np.random.default_rng(7)
    centers = rng.uniform(6120.0, 6880.0, 24)
    for c in centers:
        spec -= 30.0 * np.exp(-0.5 * ((waves - c) / 2.0) ** 2)
    return spec


def make_cube(column_shifts_ch, ny=20, noise_sigma=0.3, seed=11):
    """Cube (nlam, ny, nx) where column ix is spectrally shifted by its entry."""
    base = stellar_spectrum(WAVES)
    rng = np.random.default_rng(seed)
    nx = len(column_shifts_ch)
    cube = np.empty((WAVES.size, ny, nx))
    channels = np.arange(WAVES.size, dtype=np.float64)
    for ix, dz in enumerate(column_shifts_ch):
        # positive dz = features appear at larger channel index
        shifted = np.interp(channels - dz, channels, base)
        for iy in range(ny):
            cube[:, iy, ix] = shifted + rng.normal(0.0, noise_sigma, WAVES.size)
    return cube


class TestHelpers(unittest.TestCase):
    def test_window_slices_drops_out_of_range(self):
        slices = window_slices(WAVES, ((6150, 6450), (9000, 9300), (6600, 6610)))
        self.assertEqual(len(slices), 1)
        lo, hi = WAVES[slices[0].start], WAVES[slices[0].stop - 1]
        self.assertGreaterEqual(lo, 6150.0)
        self.assertLessEqual(hi, 6450.0)

    def test_normalize_window_removes_slope_and_scale(self):
        sl = window_slices(WAVES, TEST_WINDOWS)[0]
        base = stellar_spectrum(WAVES)[sl]
        scaled = 0.05 * base * (1.0 + 3e-4 * (WAVES[sl] - WAVES[sl][0]))
        n1 = normalize_window(WAVES[sl], base)
        n2 = normalize_window(WAVES[sl], scaled)
        good = np.isfinite(n1) & np.isfinite(n2)
        self.assertGreater(good.sum(), 100)
        # same features after normalization despite x2000 amplitude + slope
        self.assertGreater(np.corrcoef(n1[good], n2[good])[0, 1], 0.98)

    def test_brightness_selection(self):
        cube = make_cube([0.0, 0.0], ny=4)
        cube[:, :, 1] *= 0.01
        bright = spaxel_brightness_map(cube)
        mask = brightness_selection_mask(bright, 50.0)
        self.assertTrue(mask[:, 0].all())
        self.assertFalse(mask[:, 1].any())
        ref = build_reference_spectrum(cube, mask)
        self.assertEqual(ref.size, WAVES.size)


class TestOffsetMap(unittest.TestCase):
    def _run(self, column_shifts, **kwargs):
        cube = make_cube(column_shifts)
        return compute_offset_map(
            cube, WAVES, windows_A=TEST_WINDOWS,
            brightness_percentile=0.0, **kwargs,
        )

    def test_recovers_injected_column_shifts(self):
        injected = [0.0] * 6 + [0.3] * 6 + [-0.2] * 6
        result = self._run(injected)
        prof = stripe_profile(result["offset_map_ch"], "vertical")["profile"]
        # profile is relative to the field reference; compare shape, not zero-point
        expected = np.asarray(injected, dtype=float)
        np.testing.assert_allclose(
            prof - np.median(prof), expected - np.median(expected), atol=0.05
        )
        self.assertAlmostEqual(result["channel_step_A"], STEP_A, places=6)

    def test_structure_metrics_and_gate_fire_on_stripes(self):
        injected = [0.0] * 6 + [0.35] * 6 + [-0.25] * 6
        result = self._run(injected)
        metrics = structure_metrics(result["offset_map_ch"], result["channel_step_A"])
        # 0.35 ch = 0.44 A: must exceed the 0.1 A gate threshold
        self.assertGreater(metrics["p95_abs_offset_A"], 0.1)
        self.assertGreater(metrics["structure_significance"], 3.0)
        gate = evaluate_gate_g1(metrics)
        self.assertEqual(gate["recommendation"], "fase2_justificada")
        self.assertEqual(gate["decision"], "pending_human")

    def test_flat_cube_recommends_descartable(self):
        result = self._run([0.0] * 18)
        metrics = structure_metrics(result["offset_map_ch"], result["channel_step_A"])
        self.assertLess(metrics["p95_abs_offset_A"], 0.1)
        gate = evaluate_gate_g1(metrics)
        self.assertEqual(gate["recommendation"], "fase2_descartable")

    def test_nan_block_inside_window_is_tolerated(self):
        cube = make_cube([0.0] * 8 + [0.3] * 8)
        # kill 20% of the first window in every spaxel (finite fraction 0.8 > 0.7)
        sl = window_slices(WAVES, TEST_WINDOWS)[0]
        width = sl.stop - sl.start
        cube[sl.start:sl.start + width // 5, :, :] = np.nan
        result = compute_offset_map(
            cube, WAVES, windows_A=TEST_WINDOWS, brightness_percentile=0.0
        )
        prof = stripe_profile(result["offset_map_ch"], "vertical")["profile"]
        self.assertTrue(np.isfinite(prof).all())
        self.assertGreater(prof[-1] - prof[0], 0.2)

    def test_fully_masked_window_uses_remaining_window(self):
        cube = make_cube([0.0] * 4 + [0.3] * 4)
        sl = window_slices(WAVES, TEST_WINDOWS)[0]
        cube[sl, :, :] = np.nan
        result = compute_offset_map(
            cube, WAVES, windows_A=TEST_WINDOWS, brightness_percentile=0.0
        )
        self.assertTrue((result["nwin_map"] <= 1).all())
        prof = stripe_profile(result["offset_map_ch"], "vertical")["profile"]
        self.assertGreater(prof[-1] - prof[0], 0.2)

    def test_no_window_in_range_raises(self):
        cube = make_cube([0.0, 0.0], ny=3)
        with self.assertRaises(ValueError):
            compute_offset_map(cube, WAVES, windows_A=((9000, 9300),))


class TestVectorization(unittest.TestCase):
    def test_normalize_window_cube_matches_scalar(self):
        # S0b equivalence gate: vectorized normalization == scalar, < 1e-9.
        cube = make_cube([0.0] * 6 + [0.3] * 6 + [-0.2] * 6, ny=8)
        # add a dead column (normalize_window would return None) to exercise
        # the all-NaN column path
        cube[:, :, 0] = np.nan
        sl = window_slices(WAVES, TEST_WINDOWS)[0]
        sub = cube[sl].reshape(sl.stop - sl.start, -1)  # (nwin, nspax)

        ref_cols = []
        for j in range(sub.shape[1]):
            nw = normalize_window(WAVES[sl], sub[:, j])
            ref_cols.append(np.full(sub.shape[0], np.nan) if nw is None else nw)
        ref = np.stack(ref_cols, axis=1)

        vec = normalize_window_cube(WAVES[sl], sub)
        # identical NaN pattern and < 1e-9 elsewhere
        self.assertTrue(np.array_equal(np.isfinite(ref), np.isfinite(vec)))
        both = np.isfinite(ref) & np.isfinite(vec)
        self.assertLess(float(np.max(np.abs(vec[both] - ref[both]))), 1e-9)

    def test_vectorized_offset_map_matches_scalar(self):
        cube = make_cube([0.0] * 6 + [0.35] * 6 + [-0.25] * 6)
        vec = compute_offset_map(cube, WAVES, windows_A=TEST_WINDOWS,
                                 brightness_percentile=0.0, vectorized=True)
        ref = compute_offset_map(cube, WAVES, windows_A=TEST_WINDOWS,
                                 brightness_percentile=0.0, vectorized=False)
        for key in ("offset_map_ch", "err_map_ch"):
            a, b = vec[key], ref[key]
            both = np.isfinite(a) & np.isfinite(b)
            self.assertTrue(np.array_equal(np.isfinite(a), np.isfinite(b)))
            self.assertLess(float(np.max(np.abs(a[both] - b[both]))), 1e-6)

    def test_err_map_and_low_err_cut(self):
        result = compute_offset_map(make_cube([0.0] * 6 + [0.35] * 6 + [-0.25] * 6),
                                    WAVES, windows_A=TEST_WINDOWS,
                                    brightness_percentile=0.0)
        err = result["err_map_ch"]
        measured = np.isfinite(result["offset_map_ch"])
        # every measured spaxel used >=2 windows here -> finite error
        self.assertTrue(np.isfinite(err[measured]).all())

        cut = float(np.nanpercentile(err[measured], 60.0))
        metrics = structure_metrics(result["offset_map_ch"], result["channel_step_A"],
                                    err_map_ch=err, max_err_ch=cut)
        expected_n = int((measured & (err < cut)).sum())
        self.assertEqual(metrics["n_selected_low_err"], expected_n)
        self.assertEqual(metrics["max_err_ch"], cut)

    def test_parse_windows_arg(self):
        self.assertEqual(
            parse_windows_arg("5100:5550, 6100:6500"),
            ((5100.0, 5550.0), (6100.0, 6500.0)),
        )


class TestCLI(unittest.TestCase):
    def _write_cube_fits(self, path, column_shifts):
        from astropy.io import fits

        cube = make_cube(column_shifts, ny=12)
        hdu = fits.PrimaryHDU(data=cube.astype(np.float32))
        hdu.header["CRVAL3"] = float(WAVES[0])
        hdu.header["CD3_3"] = STEP_A
        hdu.header["CRPIX3"] = 1.0
        hdu.writeto(path, overwrite=True)

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cube_fits = tmp / "cube.fits"
            self._write_cube_fits(cube_fits, [0.0] * 6 + [0.35] * 6 + [-0.25] * 6)
            qc_out = tmp / "stageS0_qc.json"
            map_out = tmp / "stageS0_offset_map.fits"
            plot_out = tmp / "stageS0.png"
            rc = main([
                "--cube", str(cube_fits),
                "--qc-output", str(qc_out),
                "--map-output", str(map_out),
                "--plot-output", str(plot_out),
                "--orientation", "vertical",
                "--windows", "6150:6450,6620:6850",
                "--brightness-percentile", "0",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(qc_out.exists() and map_out.exists() and plot_out.exists())
            qc = json.loads(qc_out.read_text())
            for key in ("cube", "sha256", "binning", "windows_used_A",
                        "channel_step_A", "metrics", "gate_g1", "runtime_s"):
                self.assertIn(key, qc)
            self.assertIn("n_selected_low_err", qc["metrics"])
            self.assertEqual(qc["gate_g1"]["decision"], "pending_human")

            from astropy.io import fits
            with fits.open(map_out) as hdul:
                names = [h.name for h in hdul]
            for ext in ("OFFSET_A", "ERR_CH", "NWIN", "SELECT"):
                self.assertIn(ext, names)


if __name__ == "__main__":
    unittest.main()
