"""WP-G3R-7: dual-track SpT (template chi2 + indices). Synthetic fixtures only."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.constants import spt_label
from musepipe.models.cache import write_spectrum_npz
from musepipe.models.extinction import CCMExtinction
from musepipe.models.indices import indices_to_spt, measure_indices
from musepipe.models.manifest import write_manifest
from musepipe.models.observed import FitSpectrum
from musepipe.models.prep import prepare_template
from musepipe.models.template_fit import classify_gravity, fit_powerlaw, fit_templates
from musepipe.models.templates import EmpiricalTemplateLibrary

_WAVE = np.linspace(6000.0, 9000.0, 700)
_BAND = 8180.0


def _template_flux(code):
    cont = (_WAVE / 6000.0) ** (-1.0)
    band = 1.0 - 0.05 * code * np.exp(-0.5 * ((_WAVE - _BAND) / 80.0) ** 2)
    return cont * band


def _make_library(tmp, name, codes, gravity="young"):
    family = Path(tmp) / name
    family.mkdir()
    rels = []
    for c in codes:
        spt = spt_label(float(c))
        rel = f"t_{spt}.npz"
        write_spectrum_npz(family / rel, _WAVE, _template_flux(c), {"spt": spt})
        rels.append(rel)
    write_manifest(family, rels)
    return EmpiricalTemplateLibrary(family, gravity_class=gravity,
                                    citation="synthetic", version="v")


def _observed(code, av, *, scale=2.0, veiling_a=0.0, veiling_alpha=0.0, seed=0):
    ext = CCMExtinction(rv=3.1, citation="c")
    lib_wave = _WAVE
    from musepipe.models import TemplateSpectrum
    tmpl = TemplateSpectrum(lib_wave, _template_flux(code), {})
    wave_bin = np.linspace(6300.0, 9000.0, 120)
    model = prepare_template(tmpl, wave_bin, lsf_fwhm_A=2.383, extinction=ext, av=av, scale=scale)
    model = model + veiling_a * (wave_bin / 7500.0) ** veiling_alpha
    rng = np.random.default_rng(seed)
    err = np.full_like(model, 0.02 * float(np.mean(model)))
    flux = model + rng.normal(0.0, err)
    return FitSpectrum(wave_bin, flux, err, wave_bin.size, float(wave_bin.size), {}), ext


class FitTemplatesTests(unittest.TestCase):
    def test_v1_recovers_spt_and_av(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = _make_library(tmp, "young", range(0, 9))
            fs, ext = _observed(5.0, 1.0, seed=1)
            res = fit_templates(fs, lib, ext, av_axis=np.arange(0.0, 5.01, 0.25),
                                lsf_fwhm_A=2.383)
            self.assertEqual(res["spt_best_code"], 5.0)
            self.assertLessEqual(abs(res["av_best"] - 1.0), 0.5)

    def test_ranking_degrades_with_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = _make_library(tmp, "young", range(0, 9))
            fs, ext = _observed(5.0, 1.0, seed=2)
            res = fit_templates(fs, lib, ext, av_axis=np.arange(0.0, 5.01, 0.5),
                                lsf_fwhm_A=2.383)
            by_code = {r["spt_code"]: r["chi2"] for r in res["ranking"]}
            self.assertLess(by_code[5.0], by_code[0.0])
            self.assertLess(by_code[5.0], by_code[8.0])

    def test_veiling_recovers_injected(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = _make_library(tmp, "young", range(0, 9))
            a0 = 0.3
            fs, ext = _observed(5.0, 0.0, veiling_a=a0, veiling_alpha=0.0, seed=3)
            res = fit_templates(fs, lib, ext, av_axis=np.arange(0.0, 3.01, 0.5),
                                lsf_fwhm_A=2.383, veiling=True,
                                veiling_alpha_axis=[-1.0, 0.0, 1.0])
            best = res["ranking"][0]
            self.assertGreater(best["veiling_a"], 0.0)
            self.assertLess(abs(best["veiling_a"] - a0) / a0, 0.4)


class ClassifyGravityTests(unittest.TestCase):
    def test_distinguishes_classes(self):
        g = classify_gravity({"young": {"chi2_min": 10.0},
                              "field": {"chi2_min": 25.0},
                              "nonstellar": {"chi2_min": 50.0}})
        self.assertEqual(g["best_class"], "young")
        self.assertAlmostEqual(g["dchi2_by_class"]["field"], 15.0)
        self.assertEqual(g["young"], 10.0)


class IndicesTests(unittest.TestCase):
    def _spectrum(self):
        wave = np.linspace(7000.0, 8500.0, 1500)
        flux = np.ones_like(wave)
        flux[(wave >= 8000) & (wave <= 8100)] = 2.0
        flux[(wave >= 7200) & (wave <= 7300)] = 4.0
        return wave, flux, 0.01 * flux

    def test_index_analytic(self):
        wave, flux, err = self._spectrum()
        defs = {"idx": {"numerator": [[8000, 8100]], "denominator": [[7200, 7300]],
                        "citation": "c"}}
        res = measure_indices(wave, flux, err, defs, seed=0, n_mc=100)
        self.assertAlmostEqual(res["idx"]["value"], 0.5, places=6)
        spt, err_spt = indices_to_spt(res, {"idx": {"spt_poly": [1.0, 4.0], "citation": "c"}})
        self.assertAlmostEqual(spt, 3.0, places=3)  # 1 + 4*0.5
        self.assertGreater(err_spt, 0.0)

    def test_window_out_of_coverage_raises(self):
        wave, flux, err = self._spectrum()
        with self.assertRaises(RuntimeError):
            measure_indices(wave, flux, err,
                            {"bad": {"numerator": [[9000, 9100]],
                                     "denominator": [[7200, 7300]]}})

    def test_centered_calibration_polynomial(self):
        # Riddick-style relation SpT = c0 + c1*(x-center)+... (value 0.5, center 0.5)
        wave, flux, err = self._spectrum()
        defs = {"idx": {"numerator": [[8000, 8100]], "denominator": [[7200, 7300]]}}
        res = measure_indices(wave, flux, err, defs, seed=0, n_mc=50)  # value = 0.5
        spt, _ = indices_to_spt(res, {"idx": {"spt_poly": [2.0, 10.0], "center": 0.5}})
        self.assertAlmostEqual(spt, 2.0, places=6)  # (x-center)=0 -> c0


class PowerlawAndStageTests(unittest.TestCase):
    def test_powerlaw_fit(self):
        wave = np.linspace(6300.0, 9000.0, 120)
        flux = 3.0 * (wave / 7500.0) ** (-1.5)
        err = 0.01 * flux
        fs = FitSpectrum(wave, flux, err, 120, 120.0, {})
        res = fit_powerlaw(fs, alpha_axis=np.arange(-3.0, 3.01, 0.5))
        self.assertAlmostEqual(res["alpha_best"], -1.5, places=6)

    def test_stage_assembly_injected(self):
        from musepipe.stages.stage_g3_template_fit import (
            compute_stage_g3_template_fit, stage_g3_template_fit_paths,
            write_stage_g3_template_fit)
        with tempfile.TemporaryDirectory() as tmp:
            young = _make_library(tmp, "young", range(0, 9), gravity="young")
            field = _make_library(tmp, "field", range(0, 9), gravity="field")
            fs, _ = _observed(5.0, 1.0, seed=5)
            cfg = {"h01_lsf_fwhm_A": 2.383, "g3_atmo_av_axis": [0.0, 3.0, 0.5],
                   "h03_extinction_law_citation": "c", "g3_seed": 0,
                   "g3_nonstellar_alpha_axis": [-3.0, 3.0, 1.0], "run_id": "syn"}
            paths = stage_g3_template_fit_paths("syn", project_root=tmp)
            rows, fj = compute_stage_g3_template_fit(
                cfg, paths, fit_spec=fs, libraries=(young, field))
            props = {r["property"] for r in rows}
            self.assertEqual(props, {"spectral_type", "spt_templates", "spt_indices"})
            self.assertIn("gravity_classes", fj)
            self.assertIn("best_class", fj["gravity_classes"])
            # spt_indices is not_constrained without config index definitions (D7)
            spt_idx = next(r for r in rows if r["property"] == "spt_indices")
            self.assertEqual(spt_idx["label"], "not_constrained")
            written = write_stage_g3_template_fit(rows, fj, paths)
            self.assertTrue(Path(written["fit_json"]).exists())


if __name__ == "__main__":
    unittest.main()
