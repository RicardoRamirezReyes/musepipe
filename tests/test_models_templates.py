"""WP-G3R-4: empirical template adapter + SpT encoding + resolution rule."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from musepipe.constants import spt_code, spt_label
from musepipe.models import SpectralLibrary, TemplateSpectrum
from musepipe.models.cache import write_spectrum_npz
from musepipe.models.manifest import MANIFEST_NAME, write_manifest
from musepipe.models.prep import prepare_template
from musepipe.models.templates import EmpiricalTemplateLibrary


class SptEncodingTests(unittest.TestCase):
    def test_code_anchors(self):
        self.assertEqual(spt_code("M0"), 0.0)
        self.assertEqual(spt_code("M9"), 9.0)
        self.assertEqual(spt_code("L0"), 10.0)
        self.assertEqual(spt_code("K5"), -5.0)
        self.assertEqual(spt_code("K7"), -3.0)
        self.assertEqual(spt_code("M4.5"), 4.5)
        self.assertEqual(spt_code("O5"), -55.0)

    def test_label_inverse_and_rounding(self):
        for label in ("O5", "G5", "K7", "M0", "M4.5", "M9", "L0", "L3"):
            self.assertEqual(spt_label(spt_code(label)), label)
        self.assertEqual(spt_label(0.26), "M0.5")  # nearest half subtype
        self.assertEqual(spt_label(0.24), "M0")

    def test_bad_label_raises(self):
        with self.assertRaises(ValueError):
            spt_code("Q3")


def _build(family, specs):
    """specs: list of (spt_label, extra_meta_dict). Distinct flux per template."""
    wave = np.linspace(5000.0, 9000.0, 60)
    rels = []
    for i, (spt, extra) in enumerate(specs):
        flux = np.full_like(wave, float(i + 1))
        rel = f"tmpl{i}_{spt}.npz"
        write_spectrum_npz(family / rel, wave, flux, {"spt": spt, **extra})
        rels.append(rel)
    write_manifest(family, rels)
    return wave


class EmpiricalTemplateTests(unittest.TestCase):
    def _lib(self, tmp, specs, **kw):
        family = Path(tmp) / "lib"
        family.mkdir()
        _build(family, specs)
        return EmpiricalTemplateLibrary(
            family, gravity_class=kw.pop("gravity_class", "young"),
            citation="synthetic", version="v0", **kw)

    def test_protocol_and_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = self._lib(tmp, [("M0", {}), ("M5", {}), ("M9", {})])
            self.assertIsInstance(lib, SpectralLibrary)
            np.testing.assert_allclose(lib.grid()["spt"], [0.0, 5.0, 9.0])

    def test_nearest_spt_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = self._lib(tmp, [("M0", {}), ("M4", {}), ("M8", {})])
            s = lib.get(spt=5.0)
            self.assertEqual(s.meta["spt_code"], 4.0)
            self.assertAlmostEqual(s.meta["spt_delta"], 1.0)
            self.assertEqual(lib.get(spt="M0").meta["spt"], "M0")

    def test_multiple_per_spt(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = self._lib(tmp, [("M5", {"object": "a"}), ("M5", {"object": "b"})])
            s = lib.get(spt="M5")
            self.assertEqual(s.meta["n_at_spt"], 2)
            self.assertEqual(len(lib.grid()["spt"]), 1)

    def test_resolution_surfaced_from_init_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = self._lib(tmp, [("M5", {}), ("M6", {"resolution_fwhm_A": 0.7})],
                            resolution_fwhm_A=4.0)
            self.assertEqual(lib.get(spt="M5").meta["resolution_fwhm_A"], 4.0)
            self.assertEqual(lib.get(spt="M6").meta["resolution_fwhm_A"], 0.7)

    def test_gravity_class_and_citation_mandatory(self):
        with tempfile.TemporaryDirectory() as tmp:
            family = Path(tmp) / "lib"
            family.mkdir()
            _build(family, [("M5", {})])
            with self.assertRaises(RuntimeError):
                EmpiricalTemplateLibrary(family, gravity_class="giant",
                                         citation="x", version="v")
            with self.assertRaises(RuntimeError):
                EmpiricalTemplateLibrary(family, gravity_class="young",
                                         citation=None, version="v")


class ResolutionRuleTests(unittest.TestCase):
    def test_prepare_template_resolution_both_directions(self):
        wave = np.linspace(6000.0, 7000.0, 400)
        flux = np.exp(-0.5 * ((wave - 6500.0) / 3.0) ** 2) + 0.1
        tmpl = TemplateSpectrum(wave, flux, {})
        wout = np.linspace(6100.0, 6900.0, 120)
        # finer than LSF -> degraded, no mismatch
        f_fine, mm_fine = prepare_template(
            tmpl, wout, lsf_fwhm_A=2.383, template_fwhm_A=1.0, return_flag=True)
        self.assertFalse(mm_fine)
        # coarser than LSF -> not degraded, mismatch flagged
        f_coarse, mm_coarse = prepare_template(
            tmpl, wout, lsf_fwhm_A=2.383, template_fwhm_A=4.0, return_flag=True)
        self.assertTrue(mm_coarse)

    def test_default_backcompat_returns_array(self):
        wave = np.linspace(6000.0, 7000.0, 200)
        tmpl = TemplateSpectrum(wave, np.ones_like(wave), {})
        out = prepare_template(tmpl, np.linspace(6100.0, 6900.0, 50), lsf_fwhm_A=2.5)
        self.assertIsInstance(out, np.ndarray)


def _real_family_or_skip(subdir):
    from musepipe.config import load_run_config
    from musepipe.models.manifest import library_root
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
def test_templates_real_both_classes():
    young = EmpiricalTemplateLibrary(
        _real_family_or_skip("templates_young"), gravity_class="young",
        citation="Manara et al. 2013/2017", version="v")
    field = EmpiricalTemplateLibrary(
        _real_family_or_skip("templates_field"), gravity_class="field",
        citation="Kesseli et al. 2017", version="v")
    yc = young.grid()["spt"]
    assert yc.min() <= spt_code("M0") and yc.max() >= spt_code("M9")  # M0-M9 spanned
    for lib in (young, field):
        s = lib.get(spt="M5")
        assert np.all(np.isfinite(s.flux)) and np.any(s.flux > 0)
        assert s.meta["gravity_class"] == lib.gravity_class


if __name__ == "__main__":
    unittest.main()
