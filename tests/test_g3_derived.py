"""WP-G3R-9: derived chain (L_bol, R, mass/age) + end-to-end MC. Synthetic only."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.constants import (
    LSUN_ERG_S, PC_CM, RSUN_CM, SIGMA_SB_CGS,
)
from musepipe.models.cache import write_tracks_npz
from musepipe.models.derived import (
    lbol_from_scaled_model, mass_age_from_tracks, radius_from_omega, run_mc_chain,
)
from musepipe.models.tracks import TrackGrid

_AGES = np.array([0.001, 0.01, 0.1, 1.0])
_LS = np.array([1e-4, 1e-3, 1e-2, 1e-1])


def _mass(a, l):
    return 0.05 - 0.01 * np.log10(a) + 0.02 * np.log10(l)


def _make_tracks(path, offset=0.0):
    aa, ll = np.meshgrid(_AGES, _LS)
    age, lbol = aa.ravel(), ll.ravel()
    la, lg = np.log10(age), np.log10(lbol)
    write_tracks_npz(path, {
        "mass_msun": _mass(age, lbol) + offset, "age_gyr": age,
        "teff_k": 3000.0 + 100.0 * la, "l_bol_lsun": lbol,
        "radius_rsun": 0.5 + 0.1 * la, "logg": 4.0 + 0.2 * la}, {"family": "syn"})
    return TrackGrid(path, family="syn", citation="c", version="v")


class RadiusLbolTests(unittest.TestCase):
    def test_radius_analytic(self):
        r_rsun, d_pc = 0.135, 138.6
        r_over_d = (r_rsun * RSUN_CM) / (d_pc * PC_CM)
        omega = r_over_d ** 2 / 1e-20
        res = radius_from_omega(omega, 0.0, d_pc, 0.0, flux_unit_cgs=1e-20)
        self.assertAlmostEqual(res["r_rsun"], r_rsun, places=6)
        self.assertAlmostEqual(res["r_err_rsun"], 0.0, places=9)

    def test_radius_nonphysical_omega(self):
        res = radius_from_omega(-1.0, 0.0, 138.6, 0.0)
        self.assertTrue(np.isnan(res["r_rsun"]))

    def test_lbol_analytic(self):
        r_rsun, d_pc, teff = 0.135, 138.6, 3000.0
        r_over_d = (r_rsun * RSUN_CM) / (d_pc * PC_CM)
        omega = r_over_d ** 2 / 1e-20
        res = lbol_from_scaled_model(teff, omega, d_pc, flux_unit_cgs=1e-20)
        expected = 4 * np.pi * (r_rsun * RSUN_CM) ** 2 * SIGMA_SB_CGS * teff ** 4 / LSUN_ERG_S
        self.assertAlmostEqual(res["l_bol_lsun"] / expected, 1.0, places=6)


class TracksMassTests(unittest.TestCase):
    def test_mass_exact_at_constant_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            tg = _make_tracks(Path(tmp) / "t.npz")
            n = 100
            res = mass_age_from_tracks({"syn": tg}, np.full(n, 3e-3), np.full(n, 0.03))
            self.assertAlmostEqual(res["combined_mass_msun"][1],
                                   float(_mass(0.03, 3e-3)), places=6)

    def test_between_family_err_sys(self):
        with tempfile.TemporaryDirectory() as tmp:
            tg1 = _make_tracks(Path(tmp) / "t1.npz", offset=0.0)
            tg2 = _make_tracks(Path(tmp) / "t2.npz", offset=0.02)
            n = 100
            res = mass_age_from_tracks({"a": tg1, "b": tg2},
                                       np.full(n, 3e-3), np.full(n, 0.03))
            self.assertAlmostEqual(res["mass_err_sys"], 0.01, places=6)  # 0.5*0.02


class MCChainTests(unittest.TestCase):
    def _atmo(self, omega0):
        T = np.array([2800.0, 3000.0])
        G = np.array([4.0, 4.5])
        A = np.array([0.0, 1.0])
        dchi2 = np.full((2, 2, 2), np.inf)
        dchi2[0, 0, 0] = 0.0
        scales = np.zeros((2, 2, 2))
        scales[0, 0, 0] = omega0
        return {"teff_axis": T, "logg_axis": G, "av_axis": A, "dchi2_3d": dchi2,
                "scales_3d": scales, "interp_error": {"teff": 0.0, "logg": 0.0, "av": 0.0}}

    def test_zero_errors_degenerate_percentiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            tg = _make_tracks(Path(tmp) / "t.npz")
            teff, d = 2800.0, 138.6
            target_L = 3e-3
            omega_phys = target_L * LSUN_ERG_S / (4 * np.pi * (d * PC_CM) ** 2 * SIGMA_SB_CGS * teff ** 4)
            omega0 = omega_phys / 1e-20
            cfg = {"g3_n_mc": 200, "g3_seed": 0, "g3_sys_fluxcal_frac": 0.0,
                   "h03_flux_unit_cgs": 1e-20, "h03_distance_pc": d,
                   "h03_distance_err_pc": 0.0, "g3_age_myr": 6.0,
                   "g3_age_err_myr": [0.0, 0.0]}
            mc = run_mc_chain(cfg, self._atmo(omega0), {"syn": tg}, np.random.default_rng(0))
            lp = mc["percentiles"]["l_bol_lsun"]
            self.assertAlmostEqual(lp[0], lp[2], places=8)      # degenerate
            self.assertAlmostEqual(lp[1], target_L, places=4)
            self.assertEqual(mc["percentiles"]["teff_k"][0], mc["percentiles"]["teff_k"][2])

    def test_persist_and_reload_posterior(self):
        from musepipe.stages.stage_g3_derived import (
            compute_stage_g3_derived, stage_g3_derived_paths, write_stage_g3_derived)
        with tempfile.TemporaryDirectory() as tmp:
            tg = _make_tracks(Path(tmp) / "t.npz")
            teff, d = 2800.0, 138.6
            omega_phys = 3e-3 * LSUN_ERG_S / (4 * np.pi * (d * PC_CM) ** 2 * SIGMA_SB_CGS * teff ** 4)
            omega0 = omega_phys / 1e-20
            cfg = {"g3_n_mc": 300, "g3_seed": 0, "g3_sys_fluxcal_frac": 0.1,
                   "h03_flux_unit_cgs": 1e-20, "h03_distance_pc": d,
                   "h03_distance_err_pc": 0.3, "g3_age_myr": 6.0,
                   "g3_age_err_myr": [2.0, 4.0], "g3_age_citation": "Bowler+2017"}
            paths = stage_g3_derived_paths("syn", project_root=tmp)
            rows, mc = compute_stage_g3_derived(
                cfg, paths, atmo_result=self._atmo(omega0), track_grids={"syn": tg})
            props = {r["property"] for r in rows}
            self.assertEqual(props, {"radius", "l_bol", "mass", "age_used", "logg_evol"})
            written = write_stage_g3_derived(rows, mc, paths, track_grids={"syn": tg})
            self.assertTrue(Path(written["mass_posterior"]).exists())
            self.assertTrue(Path(written["hrd_plot"]).exists())
            with np.load(written["mass_posterior"]) as z:
                self.assertIn("combined", z.files)
                self.assertIn("family_syn", z.files)
                self.assertGreater(z["combined"].size, 0)


if __name__ == "__main__":
    unittest.main()
