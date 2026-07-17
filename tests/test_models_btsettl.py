"""WP-G3R-3: BT-Settl SpectralLibrary adapter.

Synthetic 3x3 grid built in-test (no external data); one external_data test that
runs only when the real family is present.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from musepipe.models import SpectralLibrary, TemplateSpectrum
from musepipe.models.btsettl import BTSettlLibrary
from musepipe.models.cache import write_spectrum_npz
from musepipe.models.manifest import write_manifest

_TEFFS = (2600.0, 2800.0, 3000.0)
_LOGGS = (4.0, 4.5, 5.0)
# log-flux linear in (Teff, logg) -> bilinear-in-log interpolation is EXACT.
_A_T, _B_G = 0.001, 0.5


def _logflux(teff, logg, wave):
    return _A_T * teff + _B_G * logg + 0.3 * np.sin(wave / 1000.0)


def _build_grid(family: Path, *, skip=()):
    wave = np.linspace(5000.0, 9000.0, 60)
    rels = []
    for teff in _TEFFS:
        for logg in _LOGGS:
            if (teff, logg) in skip:
                continue
            flux = np.exp(_logflux(teff, logg, wave))
            rel = f"teff{int(teff)}_logg{logg:.1f}.npz"
            write_spectrum_npz(family / rel, wave, flux, {"teff_k": teff, "logg": logg})
            rels.append(rel)
    write_manifest(family, rels)
    return wave


class BTSettlSyntheticTests(unittest.TestCase):
    def _lib(self, tmp, **kw):
        family = Path(tmp) / "grid"
        family.mkdir()
        wave = _build_grid(family, skip=kw.pop("skip", ()))
        return BTSettlLibrary(family, citation="synthetic", version="v0", **kw), wave

    def test_protocol_and_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib, _ = self._lib(tmp)
            self.assertIsInstance(lib, SpectralLibrary)
            g = lib.grid()
            np.testing.assert_allclose(g["teff"], _TEFFS)
            np.testing.assert_allclose(g["logg"], _LOGGS)

    def test_exact_node_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib, wave = self._lib(tmp)
            spec = lib.get(teff=2800.0, logg=4.5)
            self.assertIsInstance(spec, TemplateSpectrum)
            self.assertEqual(spec.meta["interp"], "exact")
            np.testing.assert_allclose(spec.flux, np.exp(_logflux(2800.0, 4.5, wave)))
            self.assertIn("interp_error_halfstep", spec.meta)

    def test_bilinear_exact_for_loglinear(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib, wave = self._lib(tmp)
            spec = lib.get(teff=2700.0, logg=4.25)
            self.assertEqual(spec.meta["interp"], "bilinear")
            np.testing.assert_allclose(
                spec.flux, np.exp(_logflux(2700.0, 4.25, wave)), rtol=1e-10)
            self.assertEqual(spec.meta["nodes"],
                             [[2600.0, 4.0], [2600.0, 4.5], [2800.0, 4.0], [2800.0, 4.5]])
            self.assertAlmostEqual(spec.meta["interp_error_halfstep"]["teff"], 100.0)
            self.assertAlmostEqual(spec.meta["interp_error_halfstep"]["logg"], 0.25)

    def test_edge_interpolation_along_one_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib, wave = self._lib(tmp)
            spec = lib.get(teff=3000.0, logg=4.25)  # Teff exact, logg interpolated
            np.testing.assert_allclose(
                spec.flux, np.exp(_logflux(3000.0, 4.25, wave)), rtol=1e-10)

    def test_citation_mandatory(self):
        with tempfile.TemporaryDirectory() as tmp:
            family = Path(tmp) / "grid"
            family.mkdir()
            _build_grid(family)
            with self.assertRaises(RuntimeError):
                BTSettlLibrary(family, citation=None, version="v0")

    def test_out_of_grid_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib, _ = self._lib(tmp)
            with self.assertRaises(RuntimeError):
                lib.get(teff=1000.0, logg=4.0)
            with self.assertRaises(RuntimeError):
                lib.get(teff=2700.0, logg=6.0)

    def test_missing_node_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib, _ = self._lib(tmp, skip=((2800.0, 4.5),))
            # a cell that needs the missing corner must raise
            with self.assertRaises(RuntimeError):
                lib.get(teff=2700.0, logg=4.25)


def _real_family_or_skip(subdir):
    from musepipe.config import load_run_config
    from musepipe.models.manifest import MANIFEST_NAME, library_root
    repo = Path(__file__).resolve().parents[1]
    try:
        rc = load_run_config("ROXs12b_B_adp", project_root=repo)
        root = library_root(dict(rc.config), project_root=repo)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"external libraries unavailable: {exc}")
    family = root / subdir
    if not (family / MANIFEST_NAME).exists():
        pytest.skip(f"{subdir} not downloaded")
    return family


@pytest.mark.external_data
def test_btsettl_real_grid_and_get():
    family = _real_family_or_skip("bt-settl-cifist")
    lib = BTSettlLibrary(family, citation="Allard et al. 2012", version="CIFIST2011")
    g = lib.grid()
    assert g["teff"].min() <= 2000.0 and g["teff"].max() >= 4500.0  # D4 covered
    assert set(g["logg"].tolist()) >= {3.5, 4.0, 4.5, 5.0, 5.5}
    spec = lib.get(teff=3000.0, logg=4.0)
    win = (spec.wave_A >= 4000.0) & (spec.wave_A <= 10000.0)
    assert win.sum() > 1000
    assert np.all(np.isfinite(spec.flux[win])) and np.all(spec.flux[win] > 0)


if __name__ == "__main__":
    unittest.main()
