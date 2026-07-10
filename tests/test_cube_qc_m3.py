import unittest

import numpy as np

from musepipe.qc.cube_qc import (
    compute_m3_flux,
    detect_primary_yx,
    load_passband_csv,
    synthetic_band_flux,
)


def _flat_passband(lo, hi, step=10.0):
    w = np.arange(lo, hi + step, step, dtype=np.float64)
    return w, np.ones_like(w)


def _make_config(ref_flambda_cgs, *, band="RP", overlap=1.0, pb_dir, flux_unit_cgs=1e-20):
    return {
        "m3_recommended_band": band,
        "m3_flux_unit_cgs": flux_unit_cgs,
        "m3_passband_dir": str(pb_dir),
        "m3_passbands": {
            band: {
                "file": f"test_{band}.csv",
                "mag": 12.0,
                "ref_flambda_cgs": ref_flambda_cgs,
                "muse_overlap_frac": overlap,
            }
        },
        "m3_passband_source": "synthetic test",
    }


class CubeQcM3Tests(unittest.TestCase):
    def setUp(self):
        # A flat-response band 6100-9000 A fully inside the "MUSE" cube 5000-9300.
        self.wave = np.arange(5000.0, 9300.0, 5.0, dtype=np.float64)
        self.ny = self.nx = 41
        self.primary = (20, 20)
        # A constant total-flux SED: per-channel star flux F_lambda (native 1e-20 units).
        self.star_flux_native = 1000.0  # in 1e-20 erg/s/cm2/A
        cube = np.zeros((self.wave.size, self.ny, self.nx), dtype=np.float64)
        cube[:, self.primary[0], self.primary[1]] = self.star_flux_native  # a point source
        self.cube = cube

    def _write_passband(self, tmp, band, lo, hi):
        from pathlib import Path

        w, r = _flat_passband(lo, hi)
        path = Path(tmp) / f"test_{band}.csv"
        with path.open("w") as f:
            f.write("# synthetic flat passband\n")
            f.write("wavelength_A,response_photon\n")
            for wi, ri in zip(w, r):
                f.write(f"{wi:.4f},{ri:.6e}\n")
        return path

    def test_factor_is_one_when_flux_matches_reference(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._write_passband(tmp, "RP", 6100.0, 9000.0)
            # Reference = the star's true F_lambda in cgs. The point source's total
            # flux equals star_flux_native (a single spaxel), so the aperture-summed
            # synthetic band flux (native) == star_flux_native; in cgs:
            ref_cgs = self.star_flux_native * 1e-20
            cfg = _make_config(ref_cgs, pb_dir=tmp)
            m3 = compute_m3_flux(self.cube, self.wave, cfg, primary_yx=self.primary, aperture_radius_px=5.0)
            self.assertEqual(m3["band"], "RP")
            self.assertAlmostEqual(m3["flux_factor"], 1.0, places=6)
            self.assertEqual(m3["status"], "green")

    def test_half_flux_gives_yellow_or_red_factor_half(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._write_passband(tmp, "RP", 6100.0, 9000.0)
            ref_cgs = 2.0 * self.star_flux_native * 1e-20  # cube is half the reference
            cfg = _make_config(ref_cgs, pb_dir=tmp)
            m3 = compute_m3_flux(self.cube, self.wave, cfg, primary_yx=self.primary, aperture_radius_px=5.0)
            self.assertAlmostEqual(m3["flux_factor"], 0.5, places=6)
            self.assertEqual(m3["status"], "red")

    def test_unavailable_when_no_passbands_in_config(self):
        m3 = compute_m3_flux(self.cube, self.wave, {}, primary_yx=self.primary)
        self.assertEqual(m3["status"], "unavailable")
        self.assertEqual(m3["reason"], "no_m3_passbands_in_config")

    def test_detect_primary_finds_brightest_spaxel(self):
        yx = detect_primary_yx(self.cube, self.wave)
        self.assertEqual((int(yx[0]), int(yx[1])), self.primary)

    def test_growth_curve_recovers_extended_source_total_flux(self):
        import tempfile

        # Extended source: bright core + a broad halo spread over many spaxels.
        wave = np.arange(5000.0, 9300.0, 5.0, dtype=np.float64)
        ny = nx = 81
        cy = cx = 40
        yy, xx = np.indices((ny, nx), dtype=np.float64)
        r = np.hypot(yy - cy, xx - cx)
        profile = np.exp(-0.5 * (r / 2.0) ** 2) + 0.02 * np.exp(-0.5 * (r / 15.0) ** 2)
        total_per_channel = 5000.0
        image = profile / profile.sum() * total_per_channel
        cube = np.repeat(image[None, :, :], wave.size, axis=0)

        with tempfile.TemporaryDirectory() as tmp:
            self._write_passband(tmp, "RP", 6100.0, 9000.0)
            ref_cgs = total_per_channel * 1e-20  # reference = the true TOTAL flux
            cfg = _make_config(ref_cgs, pb_dir=tmp)
            # A fixed small aperture misses the halo -> factor < 1.
            small = compute_m3_flux(cube, wave, cfg, primary_yx=(cy, cx), aperture_radius_px=5.0)
            self.assertLess(small["flux_factor"], 0.9)
            # Growth curve to the plateau recovers ~all the flux -> factor ~ 1.
            grown = compute_m3_flux(
                cube, wave, cfg, primary_yx=(cy, cx),
                aperture_correction="growth_curve", growth_radii_px=[5, 10, 20, 30, 38], growth_tol=0.005,
            )
            self.assertEqual(grown["aperture_correction"], "growth_curve")
            self.assertGreater(grown["flux_factor"], small["flux_factor"])
            self.assertAlmostEqual(grown["flux_factor"], 1.0, delta=0.03)
            self.assertEqual(grown["status"], "green")

    def test_truncation_correction_raises_factor_for_red_sed(self):
        import tempfile

        # Rising red SED and a passband extending beyond the cube's red cutoff.
        wave = np.arange(6000.0, 9300.0, 5.0, dtype=np.float64)
        ny = nx = 21
        prim = (10, 10)
        sed = (wave / 6000.0) ** 2.0  # F_lambda rising to the red
        cube = np.zeros((wave.size, ny, nx), dtype=np.float64)
        cube[:, prim[0], prim[1]] = sed

        with tempfile.TemporaryDirectory() as tmp:
            self._write_passband(tmp, "RP", 6100.0, 10500.0)  # extends past cube max 9295
            cfg = _make_config(1e-20, pb_dir=tmp)  # ref value irrelevant here
            base = compute_m3_flux(cube, wave, cfg, primary_yx=prim, aperture_radius_px=5.0)
            corr = compute_m3_flux(
                cube, wave, cfg, primary_yx=prim, aperture_radius_px=5.0, apply_truncation_correction=True
            )
            self.assertGreater(corr["truncation_correction"], 1.0)
            self.assertGreater(corr["flux_factor"], base["flux_factor"])

    def test_load_passband_csv_skips_header_and_comments(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_passband(tmp, "G", 6000.0, 6100.0)
            w, r = load_passband_csv(path)
            self.assertEqual(w.size, r.size)
            self.assertTrue(np.all(r == 1.0))
            self.assertAlmostEqual(float(w[0]), 6000.0)


if __name__ == "__main__":
    unittest.main()
