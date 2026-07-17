"""WP-G3R-5: BHAC15/ATMO2020 track adapter (EvolutionaryModel)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from musepipe.models import EvolutionaryModel
from musepipe.models.cache import write_tracks_npz
from musepipe.models.tracks import TrackGrid

# Outputs affine in (log10 age, log10 L) -> barycentric interpolation is EXACT.
_AGES = np.array([0.001, 0.01, 0.1, 1.0])
_LS = np.array([1e-4, 1e-3, 1e-2, 1e-1])


def _mass(a, l):
    return 0.05 - 0.01 * np.log10(a) + 0.02 * np.log10(l)


def _teff(a, l):
    return 3000.0 + 100.0 * np.log10(a) + 50.0 * np.log10(l)


def _build(path):
    aa, ll = np.meshgrid(_AGES, _LS)
    age, lbol = aa.ravel(), ll.ravel()
    la, lg = np.log10(age), np.log10(lbol)
    write_tracks_npz(path, {
        "mass_msun": _mass(age, lbol), "age_gyr": age, "teff_k": _teff(age, lbol),
        "l_bol_lsun": lbol, "radius_rsun": 0.5 + 0.1 * la - 0.05 * lg,
        "logg": 4.0 + 0.2 * la + 0.03 * lg}, {"family": "synthetic"})


class TrackGridSyntheticTests(unittest.TestCase):
    def _grid(self, tmp):
        p = Path(tmp) / "t.npz"
        _build(p)
        return TrackGrid(p, family="synthetic", citation="c", version="v0")

    def test_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            tg = self._grid(tmp)
            self.assertIsInstance(tg, EvolutionaryModel)

    def test_exact_recovery_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tg = self._grid(tmp)
            r = tg.lookup(l_bol=3e-3, age=0.03)
            self.assertTrue(r["in_range"])
            self.assertAlmostEqual(r["mass_msun"], float(_mass(0.03, 3e-3)), places=9)
            self.assertAlmostEqual(r["teff_k"], float(_teff(0.03, 3e-3)), places=6)
            self.assertEqual(r["clamped"], [])

    def test_err_interp_coherent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tg = self._grid(tmp)
            r = tg.lookup(l_bol=3e-3, age=0.03)
            e = r["mass_msun_err_interp"]
            self.assertGreater(e, 0.0)
            self.assertLess(e, 1.0)

    def test_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            tg = self._grid(tmp)
            r = tg.lookup(l_bol=1.0, age=0.03)  # L above the grid
            self.assertFalse(r["in_range"])
            self.assertTrue(np.isnan(r["mass_msun"]))
            self.assertIn("l_bol", r["clamped"])
            r2 = tg.lookup(l_bol=3e-3, age=100.0)  # age above the grid
            self.assertIn("age", r2["clamped"])

    def test_teff_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            tg = self._grid(tmp)
            # at age 0.03 the fixture's teff (correlated with L) spans ~2650-2800
            r = tg.lookup(l_bol=None, age=0.03, teff=2700.0)
            self.assertEqual(r["mode"], "age_teff")
            self.assertTrue(r["in_range"])
            self.assertTrue(np.isfinite(r["mass_msun"]))

    def test_citation_mandatory(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.npz"
            _build(p)
            with self.assertRaises(RuntimeError):
                TrackGrid(p, family="synthetic", citation=None, version="v0")

    def test_sample_vectorized(self):
        with tempfile.TemporaryDirectory() as tmp:
            tg = self._grid(tmp)
            out = tg.sample([3e-3, 1.0], [0.03, 0.03])
            np.testing.assert_array_equal(out["in_range"], [True, False])
            self.assertAlmostEqual(out["mass_msun"][0], float(_mass(0.03, 3e-3)), places=9)
            self.assertTrue(np.isnan(out["mass_msun"][1]))

    def test_two_families_scatter(self):
        # a different synthetic family yields a different mass at the same query
        with tempfile.TemporaryDirectory() as tmp:
            tg1 = self._grid(tmp)
            p2 = Path(tmp) / "t2.npz"
            aa, ll = np.meshgrid(_AGES, _LS)
            age, lbol = aa.ravel(), ll.ravel()
            write_tracks_npz(p2, {
                "mass_msun": _mass(age, lbol) + 0.01, "age_gyr": age,
                "teff_k": _teff(age, lbol), "l_bol_lsun": lbol,
                "radius_rsun": np.full_like(age, 0.4),
                "logg": np.full_like(age, 4.2)}, {"family": "synthetic2"})
            tg2 = TrackGrid(p2, family="synthetic2", citation="c", version="v")
            m1 = tg1.lookup(l_bol=3e-3, age=0.03)["mass_msun"]
            m2 = tg2.lookup(l_bol=3e-3, age=0.03)["mass_msun"]
            self.assertAlmostEqual(m2 - m1, 0.01, places=6)


def _lib_root_or_skip():
    from musepipe.config import load_run_config
    from musepipe.models.manifest import library_root
    repo = Path(__file__).resolve().parents[1]
    try:
        rc = load_run_config("ROXs12b_B_adp", project_root=repo)
        return library_root(dict(rc.config), project_root=repo)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"external libraries unavailable: {exc}")


@pytest.mark.external_data
def test_tracks_real_families():
    root = _lib_root_or_skip()
    families = [("BHAC15", "tracks_bhac15/bhac15_tracks.npz"),
                ("ATMO2020", "tracks_atmo2020/atmo2020_ceq_tracks.npz")]
    finite = []
    for family, rel in families:
        npz = root / rel
        if not npz.exists():
            continue
        tg = TrackGrid(npz, family=family, citation="cite", version="v")
        r = tg.lookup(l_bol=1e-3, age=0.006)  # ~1e-3 Lsun at D12 age (6 Myr)
        finite.append(bool(r["in_range"]) and np.isfinite(r["mass_msun"]))
    if not finite:
        pytest.skip("no track families downloaded")
    assert any(finite)  # mass finite in at least one family


if __name__ == "__main__":
    unittest.main()
