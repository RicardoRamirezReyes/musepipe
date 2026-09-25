"""G3 a la resolución de MUSE, una prueba por biblioteca (plan 2026-09-23).

Decisión: ``docs/2026-09-23_decision_g3_resolucion_y_bibliotecas.md`` (Q1–Q4
aprobadas por el autor el 2026-09-23). Solo fixtures sintéticos: nada aquí lee
``runs/`` ni las bibliotecas externas.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musepipe.constants import fwhm_to_sigma, spt_label
from musepipe.models import TemplateSpectrum
from musepipe.models.airvac import (air_to_vacuum, convert_frame, normalize_frame,
                                    vacuum_to_air)
from musepipe.models.cache import write_spectrum_npz
from musepipe.models.calibration import calibrate_gof_threshold, correlated_noise
from musepipe.models.extinction import CCMExtinction
from musepipe.models.libraries import (DOCUMENTED_DEFAULTS, LibraryDeclaration,
                                       resolve_declaration, template_library_entries)
from musepipe.models.manifest import PROVENANCE_NAME, write_manifest
from musepipe.models.observed import (FitSpectrum, degrade_observed,
                                      fit_spectrum_from_inputs, rebin_for_fit)
from musepipe.models.prep import degrade_variable, prepare_template, resample_conserve_flux
from musepipe.models.template_fit import fit_templates
from musepipe.models.templates import EmpiricalTemplateLibrary


def _fwhm_of(wave, flux):
    """FWHM de una línea de emisión sobre continuo nulo, por momento de 2º orden."""
    f = np.clip(flux, 0, None)
    mu = np.sum(wave * f) / np.sum(f)
    var = np.sum(f * (wave - mu) ** 2) / np.sum(f)
    return 2.0 * math.sqrt(2.0 * math.log(2.0)) * math.sqrt(var), mu


# --------------------------------------------------------------------------- #
# aire / vacío
# --------------------------------------------------------------------------- #
class AirVacuumTests(unittest.TestCase):
    def test_halpha_known_value(self):
        # Hα: 6564.61 Å en vacío ↔ 6562.80 Å en aire (valor estándar)
        self.assertAlmostEqual(float(vacuum_to_air(6564.61)), 6562.80, delta=0.01)

    def test_difference_at_7500_is_about_one_lsf(self):
        d = 7500.0 - float(vacuum_to_air(7500.0))
        self.assertAlmostEqual(d, 2.07, delta=0.02)

    def test_roundtrip_exact(self):
        w = np.linspace(4000.0, 10000.0, 101)
        np.testing.assert_allclose(air_to_vacuum(vacuum_to_air(w)), w, rtol=0, atol=1e-8)

    def test_frames_are_declared_not_guessed(self):
        self.assertEqual(normalize_frame("vacuum (SDSS)"), "vacuum")
        self.assertEqual(normalize_frame("AIR"), "air")
        # «verify air/vacuum» dice las dos cosas: no está declarado
        self.assertIsNone(normalize_frame("X-shooter reduced VIS; verify air/vacuum"))
        self.assertIsNone(normalize_frame(None))
        with self.assertRaises(RuntimeError):
            convert_frame([7000.0], None, "air")


# --------------------------------------------------------------------------- #
# declaraciones
# --------------------------------------------------------------------------- #
def _family(tmp, name, prov=None):
    fam = Path(tmp) / name
    fam.mkdir(parents=True)
    if prov is not None:
        (fam / PROVENANCE_NAME).write_text(json.dumps(prov), encoding="utf-8")
    return fam


class DeclarationTests(unittest.TestCase):
    def test_new_library_reads_frame_and_R_from_its_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            fam = _family(tmp, "templates_young_lateM_xshooter",
                          {"citation": "Autor et al. 2099", "wave_frame": "air",
                           "resolution_R": 5400, "instrument": "X-shooter VIS"})
            d = resolve_declaration({"name": "lateM", "subdir": fam.name,
                                     "gravity_class": "young", "age": "~2 Myr"}, fam)
            self.assertEqual(d.wave_frame, "air")
            self.assertEqual(d.resolution_R, 5400.0)
            self.assertEqual(d.citation, "Autor et al. 2099")
            self.assertTrue(d.sources["resolution_R"].startswith(PROVENANCE_NAME))
            self.assertEqual(d.sources["gravity_class"], "config")

    def test_config_entry_overrides_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            fam = _family(tmp, "lib", {"citation": "c", "wave_frame": "vacuum", "R": "R~3000"})
            d = resolve_declaration({"name": "lib", "wave_frame": "air",
                                     "gravity_class": "young"}, fam)
            self.assertEqual(d.wave_frame, "air")
            self.assertEqual(d.resolution_R, 3000.0)  # parsed from text

    def test_missing_resolution_or_frame_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            fam = _family(tmp, "nores", {"citation": "c", "wave_frame": "air"})
            with self.assertRaises(RuntimeError):
                resolve_declaration({"name": "nores", "gravity_class": "young"}, fam)
            fam2 = _family(tmp, "noframe", {"citation": "c", "resolution_R": 5000,
                                            "wave_frame": "verify air/vacuum"})
            with self.assertRaises(RuntimeError):
                resolve_declaration({"name": "noframe", "gravity_class": "young"}, fam2)

    def test_documented_defaults_for_the_three_old_libraries(self):
        with tempfile.TemporaryDirectory() as tmp:
            # sus PROVENANCE.json reales no declaran ni R ni marco
            y = resolve_declaration({"name": "templates_young", "subdir": "templates_young",
                                     "gravity_class": "young"},
                                    _family(tmp, "templates_young", {"citation": "Manara"}))
            f = resolve_declaration({"name": "templates_field", "subdir": "templates_field",
                                     "gravity_class": "field"},
                                    _family(tmp, "templates_field", {"citation": "Kesseli"}))
            b = resolve_declaration({"name": "bt-settl-cifist", "subdir": "bt-settl-cifist"},
                                    _family(tmp, "bt-settl-cifist", {"citation": "Allard"}))
        self.assertEqual((y.wave_frame, f.wave_frame, b.wave_frame), ("air", "vacuum", "vacuum"))
        self.assertEqual(f.resolution_R, 2000.0)
        self.assertTrue(b.resolution_sharp)
        self.assertFalse(y.resolution_sharp)
        self.assertIn("resolution_source", f.notes)
        # por objeto (Manara+2013 Tabla 2): rendija 0.4″ → 17400; 1.5″ → 5400
        self.assertEqual(y.resolution_for({"object": "TWA7"}), (17400.0, None))
        self.assertEqual(y.resolution_for({"object": "TWA29"}), (5400.0, None))
        self.assertEqual(y.resolution_for({"object": "LM601"}), (8800.0, None))
        self.assertEqual(y.min_R(), 5400.0)
        self.assertEqual(b.resolution_for(None)[0], math.inf)
        for key in ("templates_young", "templates_field", "bt-settl-cifist"):
            self.assertIn("resolution_source", DOCUMENTED_DEFAULTS[key])

    def test_uncorrected_telluric_library_with_R_per_object(self):
        prov = {"citation": "c", "wave_frame": "air, topocentric (not barycentric-corrected), Å",
                "R_range": [5040.0, 8935.0], "telluric_corrected": False,
                "objects": [{"object": "TWA26", "R": 8935.0}, {"object": "USco1", "R": 5040.0}],
                "telluric_bands": [{"lo_A": 6270.0, "hi_A": 6330.0, "severity": "weak"},
                                   {"lo_A": 7590.0, "hi_A": 7700.0, "severity": "strong"},
                                   {"lo_A": 8100.0, "hi_A": 8400.0, "severity": "moderate"}]}
        with tempfile.TemporaryDirectory() as tmp:
            d = resolve_declaration({"name": "lateM", "gravity_class": "young"},
                                    _family(tmp, "lateM", prov))
        self.assertEqual(d.wave_frame, "air")
        self.assertEqual(d.velocity_frame, "topocentric")
        self.assertEqual(d.resolution_R, 5040.0)
        self.assertEqual(d.resolution_for({"object": "TWA26"}), (8935.0, None))
        self.assertEqual(d.resolution_for({"object": "x", "R": 6505.0}), (6505.0, None))
        self.assertEqual([b["lo_A"] for b in d.telluric_mask_bands], [7590.0, 8100.0])

    def test_library_entries_from_config(self):
        self.assertEqual([e["name"] for e in template_library_entries({})],
                         ["templates_young", "templates_field"])
        entries = template_library_entries({"g3_template_libraries": [
            {"name": "a", "gravity_class": "young"}, {"subdir": "b", "gravity_class": "field"}]})
        self.assertEqual([(e["name"], e["subdir"]) for e in entries], [("a", "a"), ("b", "b")])
        with self.assertRaises(RuntimeError):
            template_library_entries({"g3_template_libraries": [{"name": "a"}, {"name": "a"}]})


# --------------------------------------------------------------------------- #
# preparación de la plantilla
# --------------------------------------------------------------------------- #
class PrepareTemplateTests(unittest.TestCase):
    def _line(self, fwhm_A, center=7000.0, step=0.1):
        wave = np.arange(6800.0, 7200.0, step)
        s = fwhm_to_sigma(fwhm_A)
        return TemplateSpectrum(wave, np.exp(-0.5 * ((wave - center) / s) ** 2), {})

    def test_quadrature_with_declared_R_lands_on_the_lsf(self):
        lsf, R = 2.3, 7000.0 / 1.0  # plantilla con FWHM 1.0 Å a 7000 Å
        t = self._line(1.0)
        wout = np.arange(6900.0, 7100.0, 0.25)
        out = prepare_template(t, wout, lsf_fwhm_A=lsf, template_R=R)
        fw, _ = _fwhm_of(wout, out)
        self.assertAlmostEqual(fw, lsf, delta=0.05)
        # tratarla como nítida la sobre-suaviza: √(2.3² + 1²) ≈ 2.51
        out_sharp = prepare_template(t, wout, lsf_fwhm_A=lsf)
        self.assertGreater(_fwhm_of(wout, out_sharp)[0], lsf + 0.15)

    def test_frame_conversion_moves_the_line(self):
        t = self._line(0.5, center=7000.0)
        wout = np.arange(6950.0, 7050.0, 0.05)
        out = prepare_template(t, wout, lsf_fwhm_A=1.0, template_R=math.inf,
                               template_frame="vacuum", data_frame="air")
        _, mu = _fwhm_of(wout, out)
        self.assertAlmostEqual(mu, float(vacuum_to_air(7000.0)), delta=0.02)
        with self.assertRaises(RuntimeError):
            prepare_template(t, wout, lsf_fwhm_A=1.0, template_frame="vacuum")

    def test_coarser_template_is_flagged_and_target_can_be_its_resolution(self):
        t = self._line(3.5)
        wout = np.arange(6900.0, 7100.0, 0.25)
        _, mm = prepare_template(t, wout, lsf_fwhm_A=2.3, template_R=7000.0 / 3.5,
                                 return_flag=True)
        self.assertTrue(mm)
        out, mm2 = prepare_template(t, wout, lsf_fwhm_A=2.3, template_R=7000.0 / 3.5,
                                    target_fwhm_A=lambda w: np.maximum(2.3, w / 2000.0),
                                    return_flag=True)
        self.assertFalse(mm2)
        self.assertAlmostEqual(_fwhm_of(wout, out)[0], 3.5, delta=0.06)

    def test_variable_kernel_follows_lambda_over_R(self):
        wave = np.arange(6000.0, 9000.0, 0.2)
        flux = np.zeros_like(wave)
        for c in (6200.0, 8800.0):
            flux[np.argmin(np.abs(wave - c))] = 1.0
        out = degrade_variable(wave, flux, wave / 3000.0)
        for c in (6200.0, 8800.0):
            m = np.abs(wave - c) < 20
            self.assertAlmostEqual(_fwhm_of(wave[m], out[m])[0], c / 3000.0, delta=0.15)

    def test_vectorized_resample_matches_the_loop(self):
        rng = np.random.default_rng(3)
        win = np.sort(rng.uniform(5000.0, 6000.0, 700))
        fin = rng.normal(1.0, 0.3, win.size)
        wout = np.linspace(4990.0, 6010.0, 211)
        from musepipe.models.prep import _bin_edges
        ein, eout = _bin_edges(win), _bin_edges(wout)
        ref = np.zeros(wout.size)
        for j in range(wout.size):
            lo, hi = eout[j], eout[j + 1]
            acc = sum(fin[k] * max(0.0, min(hi, ein[k + 1]) - max(lo, ein[k]))
                      for k in range(win.size))
            ref[j] = acc / (hi - lo)
        np.testing.assert_allclose(resample_conserve_flux(win, fin, wout), ref,
                                   rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- #
# ajuste nativo y degradación del dato
# --------------------------------------------------------------------------- #
def _cov(n, ratio):
    return {"block_bounds": np.array([[0, n]]), "n_eff_over_n_by_block": np.array([ratio])}


class NativeFitTests(unittest.TestCase):
    def test_native_dof_are_effective(self):
        n = 400
        wave = 7000.0 + 1.25 * np.arange(n)
        flux = np.ones(n)
        err = np.full(n, 0.1)
        mask = np.zeros(n, bool)
        mask[50:60] = True
        inputs = {"wave": wave, "flux": flux, "err": err, "err_column": "e", "mask": mask,
                  "mask_prov": {}, "bad_mask": None, "cov": _cov(n, 0.69)}
        fs = fit_spectrum_from_inputs(inputs, n_channels=1, wave_range=[wave[0], wave[-1]])
        self.assertEqual(fs.n_bins, n - 10)
        self.assertAlmostEqual(fs.n_dof, 0.69 * (n - 10), places=6)
        np.testing.assert_allclose(fs.err_bin, 0.1 / math.sqrt(0.69))
        self.assertAlmostEqual(fs.dof_used(np.arange(fs.n_bins) < 100), 69.0, places=6)
        # la variante binada (Q4) sigue: dof = nº de bins
        fb = fit_spectrum_from_inputs(inputs, n_channels=20, wave_range=[wave[0], wave[-1]])
        self.assertEqual(fb.n_dof, float(fb.n_bins))

    def test_degrade_observed_errors_and_correlation(self):
        n = 4000
        wave = 7000.0 + 1.25 * np.arange(n)
        rng = np.random.default_rng(0)
        sig = 1.0
        flux = rng.normal(0.0, sig, n)  # ruido blanco (L = 1)
        err = np.full(n, sig)
        mask = np.zeros(n, bool)
        mask[2000] = True
        kern = 3.0  # Å
        f2, e2, m2, r2 = degrade_observed(wave, flux, err, mask, kern, 1.0)
        # varianza real de la salida ≈ la que se declara
        emp = np.std(f2[100:1900])
        self.assertAlmostEqual(emp / float(np.median(e2)), 1.0, delta=0.12)
        self.assertTrue(np.all(r2 < 1.0))  # la correlación crece → n_eff/n baja
        self.assertTrue(m2[1998] and m2[2002] and not m2[1990])  # máscara crecida ±FWHM

    def test_rebin_uses_per_channel_neff(self):
        n = 40
        wave = 7000.0 + np.arange(n)
        reb = rebin_for_fit(wave, np.ones(n), np.ones(n), np.zeros(n, bool), _cov(n, 0.5),
                            n_channels=1, wave_range=[wave[0], wave[-1]],
                            neff_over_n_channel=np.full(n, 0.25))
        self.assertAlmostEqual(reb["n_eff_total"], 10.0)
        np.testing.assert_allclose(reb["err_bin"], 2.0)


# --------------------------------------------------------------------------- #
# pruebas por biblioteca, gravedad a resolución común, G4
# --------------------------------------------------------------------------- #
_WAVE = np.arange(6000.0, 9400.0, 0.5)


def _tflux(code):
    cont = (_WAVE / 6000.0) ** (-1.0)
    bands = 1.0 - 0.05 * code * np.exp(-0.5 * ((_WAVE - 8180.0) / 60.0) ** 2)
    lines = 1.0 - 0.02 * code * np.exp(-0.5 * ((_WAVE - 7700.0) / 1.0) ** 2)
    return cont * bands * lines


def _lib(tmp, name, codes, gravity, **decl_kw):
    fam = Path(tmp) / name
    fam.mkdir()
    rels = []
    for k, c in enumerate(codes):
        rel = f"t{k}_{spt_label(float(c))}.npz"
        # dos estrellas del mismo subtipo difieren un poco (no son copias)
        write_spectrum_npz(fam / rel, _WAVE, _tflux(c) * (1.0 + 0.003 * k),
                           {"spt": spt_label(float(c)), "object": f"obj{k}_{c}"})
        rels.append(rel)
    write_manifest(fam, rels)
    decl = LibraryDeclaration(name=name, subdir=name, gravity_class=gravity,
                              citation=f"cite {name}", **{"wave_frame": "air", **decl_kw})
    return EmpiricalTemplateLibrary(fam, gravity_class=gravity, citation="x", version="v",
                                    declaration=decl)


def _inputs(code=5.0, n=2400, seed=0):
    wave = 6300.0 + 1.25 * np.arange(n)
    tmpl = TemplateSpectrum(_WAVE, _tflux(code), {})
    model = prepare_template(tmpl, wave, lsf_fwhm_A=2.3)
    err = np.full(n, 0.01 * float(np.nanmean(model)))
    flux = model + np.random.default_rng(seed).normal(0.0, err)
    return {"wave": wave, "flux": flux, "err": err, "err_column": "e",
            "mask": np.zeros(n, bool), "mask_prov": {}, "bad_mask": None,
            "cov": _cov(n, 0.69)}


class LibraryTestStageTests(unittest.TestCase):
    def _cfg(self, **kw):
        cfg = {"h01_lsf_fwhm_A": 2.3, "g3_atmo_av_axis": [0.0, 1.0, 0.5],
               "h03_extinction_law_citation": "c", "run_id": "syn",
               "g3_fit_wave_range_A": [6300.0, 9290.0], "g3_gof_bin_channels": 20,
               "g3_gof_calibration_n_draws": 1}
        cfg.update(kw)
        return cfg

    def _run(self, tmp, libs, **cfg_kw):
        from musepipe.stages.stage_g3_template_fit import (
            compute_stage_g3_template_fit, stage_g3_template_fit_paths)
        inp = _inputs(5.0)
        paths = stage_g3_template_fit_paths("syn", project_root=tmp)
        return compute_stage_g3_template_fit(
            self._cfg(**cfg_kw), paths, fit_inputs=inp,
            fit_spec=fit_spectrum_from_inputs(inp, n_channels=1,
                                              wave_range=[6300.0, 9290.0]),
            libraries=libs)

    def test_each_library_is_its_own_test_type_native_gof_binned(self):
        from musepipe.stages.stage_g4_classify import _t3_gravity
        with tempfile.TemporaryDirectory() as tmp:
            young = _lib(tmp, "templates_young", [3, 4, 5, 5, 6, 6, 7, 8, 9], "young",
                         resolution_R=9000.0)
            field = _lib(tmp, "templates_field", range(3, 10), "field", resolution_R=2000.0)
            # una tercera, añadida SOLO por declaración (p. ej. la L de X-SHYNE)
            extra = _lib(tmp, "templates_xshyne_L", range(8, 13), "intermediate",
                         resolution_R=5400.0)
            rows, fj = self._run(tmp, [young, field, extra])
        tests = fj["library_tests"]
        self.assertEqual(set(tests), {"templates_young", "templates_field", "templates_xshyne_L"})
        self.assertFalse(tests["templates_young"]["resolution"]["data_degraded"])
        self.assertTrue(tests["templates_field"]["resolution"]["data_degraded"])
        nt = tests["templates_young"]["native_type"]
        self.assertEqual(nt["spt_best"], "M5")
        self.assertEqual(nt["dchi2_by_spt"]["M5"], 0.0)
        self.assertIn("n_eff/n", nt["dchi2_correction"])
        # la biblioteca L no llega al M5: su mínimo cae en el borde de su eje
        self.assertTrue(tests["templates_xshyne_L"]["native_type"]["edge"])
        for blk in tests.values():
            gof = blk["binned_gof"]
            self.assertIsNotNone(gof["chi2_red_native_best"])
            self.assertIsNotNone(gof["threshold"])      # calibrado en el run (Q1 binado)
            self.assertIn(gof["pass"], (True, False))
            self.assertIn("binned_best_spt", gof)
            self.assertIn("n_spectra_by_spt", blk["provenance"])
        self.assertTrue(tests["templates_young"]["binned_gof"]["pass"])
        self.assertIn("calibrated in-run", fj["gof_threshold"]["source"])
        # regla por defecto: solo las plantillas con vecino del mismo subtipo (M5, M6)
        self.assertEqual(fj["gof_threshold"]["rule"], "same_subtype_neighbour")
        self.assertEqual(fj["gof_threshold"]["n_templates_used"], 4)
        self.assertEqual(fj["acceptance"]["threshold_n_templates"], 4)
        self.assertEqual(fj["acceptance"]["gate"], "binned_gof")
        self.assertTrue(fj["acceptance"]["pass"])
        g = fj["gravity_classes"]
        self.assertIn("common resolution", g["resolution"]["label"])
        self.assertTrue(g["resolution"]["data_degraded"])
        self.assertAlmostEqual(g["resolution"]["target_fwhm_A_at"]["9290"], 9290.0 / 2000.0)
        self.assertIn("templates_xshyne_L", g["other_libraries_vs_field"])
        self.assertEqual(g["best_class"], "young")
        verdicts = _t3_gravity(g, 9.0, 4.0)
        self.assertEqual(verdicts["substellar_companion"][0], "supports")
        self.assertEqual(fj["libraries"]["templates_young"], "cite templates_young")
        self.assertEqual(fj["spt_templates"]["library"], "templates_young")

    def test_declared_threshold_overrides_calibration_and_can_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            young = _lib(tmp, "templates_young", range(3, 10), "young", resolution_R=9000.0)
            field = _lib(tmp, "templates_field", range(3, 10), "field", resolution_R=2000.0)
            _, fj = self._run(tmp, [young, field], g3_gof_chi2red_threshold=1e-3)
        self.assertEqual(fj["gof_threshold"]["source"], "config g3_gof_chi2red_threshold")
        self.assertFalse(fj["acceptance"]["pass"])

    def test_telluric_bands_of_an_uncorrected_library_are_masked_in_the_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            young = _lib(tmp, "templates_young", range(3, 10), "young", resolution_R=9000.0)
            field = _lib(tmp, "templates_field", range(3, 10), "field", resolution_R=2000.0)
            late = _lib(tmp, "templates_young_lateM", range(3, 10), "young",
                        resolution_R=5040.0, telluric_corrected=False,
                        telluric_mask_bands=[{"lo_A": 8100.0, "hi_A": 8400.0, "name": "H2O"}])
            _, fj = self._run(tmp, [young, field, late])
        t_y = fj["library_tests"]["templates_young"]
        t_l = fj["library_tests"]["templates_young_lateM"]
        self.assertLess(t_l["native_type"]["n_bins"], t_y["native_type"]["n_bins"])
        self.assertEqual(t_l["resolution"]["telluric_bands_masked_in_data"][0]["lo_A"], 8100.0)
        self.assertNotIn("telluric_bands_masked_in_data", t_y["resolution"])

    def test_per_spectrum_fits_every_star_of_a_subtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            fam = Path(tmp) / "lib"
            fam.mkdir()
            rels = []
            for i, (spt, c) in enumerate((("M5", 5.0), ("M5", 5.0), ("M7", 7.0))):
                rel = f"t{i}.npz"
                write_spectrum_npz(fam / rel, _WAVE, _tflux(c) * (1 + 0.01 * i),
                                   {"spt": spt, "object": f"star{i}"})
                rels.append(rel)
            write_manifest(fam, rels)
            decl = LibraryDeclaration(name="lib", subdir="lib", gravity_class="young",
                                      citation="c", wave_frame="air", resolution_R=9000.0)
            lib = EmpiricalTemplateLibrary(fam, gravity_class="young", citation="c",
                                           version="v", declaration=decl)
            self.assertEqual(lib.n_by_spt(), {"M5": 2, "M7": 1})
            fs = fit_spectrum_from_inputs(_inputs(5.0), n_channels=1,
                                          wave_range=[6300.0, 9290.0])
            ext = CCMExtinction(3.1, citation="c")
            r1 = fit_templates(fs, lib, ext, av_axis=[0.0], lsf_fwhm_A=2.3,
                               data_frame="air")
            r2 = fit_templates(fs, lib, ext, av_axis=[0.0], lsf_fwhm_A=2.3,
                               data_frame="air", per_spectrum=True)
            self.assertEqual(len(r1["ranking"]), 2)
            self.assertEqual(len(r2["ranking"]), 3)
            r3 = fit_templates(fs, lib, ext, av_axis=[0.0], lsf_fwhm_A=2.3, data_frame="air",
                               per_spectrum=True, exclude=lambda i, c, o: o == "star0")
            self.assertNotIn("star0", [r["object"] for r in r3["ranking"]])


# --------------------------------------------------------------------------- #
# Q1: calibración del umbral
# --------------------------------------------------------------------------- #
class CalibrationTests(unittest.TestCase):
    def test_default_rule_counts_only_same_subtype_neighbours(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = _lib(tmp, "young", [3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0], "young",
                       resolution_R=9000.0)
            inp = _inputs(5.0, n=1200)
            kw = dict(wave_range=[6300.0, 7790.0], lsf_fwhm_A=2.3,
                      extinction=CCMExtinction(3.1, citation="c"), av_axis=[0.0],
                      data_frame="air", n_draws=2, seed=0, n_channels=20)
            res = calibrate_gof_threshold(lib, inp, **kw)
            allr = calibrate_gof_threshold(lib, inp, rule="nearest_available", **kw)
        self.assertEqual(res["n_templates_used"], 4)
        self.assertEqual(res["n_templates_library"], 7)
        self.assertEqual({t["spt"] for t in res["templates_used"]}, {"M4", "M6"})
        self.assertTrue(all(r["correct_how"] == "same_subtype" for r in res["rows"]))
        self.assertEqual(allr["n_templates_used"], 7)
        with self.assertRaises(RuntimeError):
            calibrate_gof_threshold(lib, inp, rule="otra", **kw)

    def test_correlated_noise_has_the_requested_length(self):
        rng = np.random.default_rng(1)
        y = correlated_noise(200000, 2.5, rng)
        self.assertAlmostEqual(float(np.var(y)), 1.0, delta=0.02)
        ac = [np.mean(y[:-k] * y[k:]) for k in range(1, 20)]
        self.assertAlmostEqual(1.0 + 2.0 * sum(ac), 2.5, delta=0.15)

    def test_leave_one_out_distribution_binned_and_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = _lib(tmp, "young", [3.0, 4.0, 5.0, 6.0, 7.0], "young", resolution_R=9000.0)
            inp = _inputs(5.0, n=1200)
            kw = dict(wave_range=[6300.0, 7790.0], lsf_fwhm_A=2.3,
                      extinction=CCMExtinction(3.1, citation="c"), av_axis=[0.0, 0.5],
                      data_frame="air", n_draws=3, seed=0)
            res = calibrate_gof_threshold(lib, inp, n_channels=20,
                                          rule="nearest_available", **kw)
            nat = calibrate_gof_threshold(lib, inp, n_channels=1,
                                          rule="nearest_available", **kw)
            none = calibrate_gof_threshold(lib, inp, n_channels=20, **kw)
        # regla por defecto: sin vecinos del mismo subtipo no hay calibración
        self.assertEqual(none["rule"], "same_subtype_neighbour")
        self.assertEqual(none["n_templates_used"], 0)
        self.assertIsNone(none["threshold_proposed"])
        self.assertEqual(len(res["rows"]), 15)
        # una estrella por subtipo: el «correcto» es siempre el más cercano disponible
        self.assertTrue(all(r["correct_how"] == "nearest_available" for r in res["rows"]))
        self.assertEqual(res["rows"][0]["correct_subtypes"], [4.0])  # M3 → M4
        self.assertAlmostEqual(res["threshold_proposed"], res["distribution"]["p95"])
        # en bins los dof son el nº de bins; en nativo, los efectivos
        self.assertLess(res["rows"][0]["ndof"], 70)
        self.assertGreater(nat["rows"][0]["ndof"], 500)
        self.assertIn("bins", res["representation"])
        self.assertIn("native", nat["representation"])


# --------------------------------------------------------------------------- #
# índices fuera de rango de calibración y máscara D9 por defecto (2026-09-24)
# --------------------------------------------------------------------------- #
_RIDDICK = {  # Riddick+2007 Tabla A3/A2, como en los configs de publicación
    "PC3": {"spt_poly": [2.0395, 24.61, -50.292, 39.489], "center": 0.956, "range": "M3-M8"},
    "VO2": {"spt_poly": [2.6102, -7.9389, -8.3231, -14.66], "center": 0.963, "range": "M3-M8"},
    "R1": {"spt_poly": [2.8078, 21.085, -53.025, 60.755], "center": 1.044, "range": "M2.5-M8"},
}


class IndexRangeTests(unittest.TestCase):
    def test_range_maps_through_the_polynomial(self):
        from musepipe.models.indices import calibration_index_range
        for name, cal in _RIDDICK.items():
            r = calibration_index_range(cal)
            lo, hi = r["index"]
            coeffs = np.asarray(cal["spt_poly"], float)[::-1]
            spts = sorted(np.polyval(coeffs, [lo - cal["center"], hi - cal["center"]]))
            np.testing.assert_allclose(spts, r["spt"], atol=1e-8, err_msg=name)
        self.assertEqual(calibration_index_range({"spt_poly": [0, 1], "index_range": [2, 1]})
                         ["index"], [1.0, 2.0])
        self.assertIsNone(calibration_index_range({"spt_poly": [0, 1]}))

    def test_out_of_range_indices_are_excluded_and_declared(self):
        from musepipe.models.indices import indices_to_spt, select_in_range
        # los valores medidos con la máscara D9 nueva (42B b, 2026-09-24)
        vals = {"PC3": {"value": 2.861, "err": 0.05}, "VO2": {"value": 0.219, "err": 0.02},
                "R1": {"value": 1.10, "err": 0.01}, "Na_a": {"value": 1.0, "err": 0.01}}
        used, excl = select_in_range(vals, _RIDDICK)
        self.assertEqual(sorted(used), ["R1"])
        self.assertEqual(sorted(excl), ["PC3", "VO2"])
        self.assertIn("outside calibration range", excl["PC3"]["reason"])
        self.assertEqual(excl["PC3"]["spt_range"], [3.0, 8.0])
        self.assertNotIn("Na_a", used)            # sin calibración: no entra en el tipo
        spt, _ = indices_to_spt(used, _RIDDICK)
        self.assertTrue(2.5 <= spt <= 8.0)
        none_used, _ = select_in_range({"PC3": vals["PC3"]}, _RIDDICK)
        self.assertEqual(none_used, {})

    def test_v3_uses_only_in_range_indices_and_says_how_many(self):
        from musepipe.stages.stage_g3_assemble import _v3
        tj = {"spt_templates": {"code": 7.5},
              "spt_indices": {"code": 7.0, "err": 0.3, "used": ["R1"],
                              "excluded_out_of_range": {"PC3": {}, "VO2": {}}}}
        v = _v3(tj)
        self.assertTrue(v["pass"])
        self.assertEqual(v["n_indices_used"], 1)
        self.assertEqual(v["indices_excluded_out_of_range"], ["PC3", "VO2"])
        tj["spt_indices"] = {"code": float("nan"), "err": float("nan"), "used": [],
                             "excluded_out_of_range": {"PC3": {}}}
        v = _v3(tj)
        self.assertEqual(v["status"], "not_checked")
        self.assertEqual(v["n_indices_used"], 0)
        self.assertNotIn("pass", v)

    def test_default_d9_telluric_mask_is_o2_only(self):
        from musepipe.models.observed import DEFAULT_TELLURIC_BANDS_A
        self.assertEqual([list(b) for b in DEFAULT_TELLURIC_BANDS_A],
                         [[6860.0, 6960.0], [7590.0, 7700.0]])


# --------------------------------------------------------------------------- #
# R por plantilla desde la rendija VIS de la cabecera original (bug 2026-09-25)
# --------------------------------------------------------------------------- #
class XshooterHeaderRTests(unittest.TestCase):
    def _family(self, tmp, specs, *, write_headers=True):
        """specs: [(npz, source_file, slit | None, extra_meta)]"""
        from astropy.io import fits
        fam = Path(tmp) / "templates_young"
        (fam / "_source" / "CAT").mkdir(parents=True)
        rels = []
        for npz, src, slit, extra in specs:
            write_spectrum_npz(fam / npz, _WAVE, _tflux(5.0),
                               {"spt": "M5", "object": npz[:-4], "source_file": src, **extra})
            rels.append(npz)
            if write_headers and slit is not None:
                h = fits.Header()
                h["HIERARCH ESO SEQ ARM"] = "VIS"
                h["HIERARCH ESO INS OPTI4 NAME"] = slit
                fits.PrimaryHDU(header=h).writeto(fam / "_source" / "CAT" / src)
        write_manifest(fam, rels)
        (fam / PROVENANCE_NAME).write_text(json.dumps({"citation": "Manara+13,17"}))
        return fam

    def test_map_and_header_parsing(self):
        from astropy.io import fits
        from musepipe.models.libraries import XSHOOTER_VIS_R_BY_SLIT, xshooter_vis_R_from_header
        self.assertEqual(XSHOOTER_VIS_R_BY_SLIT[0.4], 18400.0)
        self.assertEqual(XSHOOTER_VIS_R_BY_SLIT[0.9], 8900.0)
        h = fits.Header()
        h["HIERARCH ESO SEQ ARM"] = "VIS"
        h["HIERARCH ESO INS OPTI4 NAME"] = "0.4x11"
        r, w, src = xshooter_vis_R_from_header(h)
        self.assertEqual((r, w), (18400.0, 0.4))
        self.assertIn("OPTI4", src)
        h["HIERARCH ESO SEQ ARM"] = "NIR"
        self.assertIsNone(xshooter_vis_R_from_header(h))
        self.assertIsNone(xshooter_vis_R_from_header(fits.Header()))

    def test_per_template_R_comes_from_the_header_and_reaches_the_qc(self):
        import warnings as _w
        with tempfile.TemporaryDirectory() as tmp:
            fam = self._family(tmp, [
                ("a_M5.npz", "A_V.fit", "0.4x11", {}),      # Manara+2017 a 0.4″
                ("b_M5.npz", "B_V.fit", "0.9x11", {}),
                ("c_M5.npz", "C_V.fit", "1.5x11", {}),
                ("d_M5.npz", "D_V.fit", None, {}),          # sin cabecera
                ("e_M5.npz", "E_V.fit", "0.4x11", {"R": 6505.0, "R_source": "IDP SPEC_RES"}),
            ])
            decl = resolve_declaration({"name": "templates_young", "subdir": "templates_young",
                                        "gravity_class": "young"}, fam)
            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter("always")
                lib = EmpiricalTemplateLibrary(fam, gravity_class="young", citation="c",
                                               version="v", declaration=decl)
            self.assertTrue(any("sin cabecera" in str(c.message) for c in caught))
            tab = lib.resolution_table()
            self.assertEqual(tab["a_M5.npz"]["R"], 18400.0)
            self.assertEqual(tab["b_M5.npz"]["R"], 8900.0)
            self.assertEqual(tab["c_M5.npz"]["R"], 5000.0)
            self.assertIn("OPTI4", tab["a_M5.npz"]["source"])
            self.assertEqual(tab["d_M5.npz"]["R"], 8800.0)     # default declarado, con aviso
            self.assertIn("header missing", tab["d_M5.npz"]["source"])
            self.assertEqual(tab["e_M5.npz"]["R"], 6505.0)     # la meta del archivo manda
            self.assertEqual(tab["e_M5.npz"]["source"], "IDP SPEC_RES")
            items = {m["object"]: lib.load_entry(c, p, m) for c, p, m in lib.entries()}
            self.assertEqual(items["a_M5"].meta["declared_resolution_R"], 18400.0)
            self.assertEqual(items["d_M5"].meta["declared_resolution_R"], 8800.0)


# --------------------------------------------------------------------------- #
# etiquetas desde el disco
# --------------------------------------------------------------------------- #
class LabelTests(unittest.TestCase):
    def test_accretion_qc_reads_template_citations_from_provenance(self):
        from musepipe.stages.stage_g3_accretion import template_libraries_label
        with tempfile.TemporaryDirectory() as tmp:
            libs = Path(tmp) / "libs"
            _family(libs, "templates_young", {"citation": "Manara et al. 2013; 2017"})
            _family(libs, "templates_field", {"citation": "Kesseli et al. 2017"})
            cfg = {"g3_libraries_root": "libs", "project_root": tmp,
                   "g3_template_family": "Luhman/Bonnefoy young M-L [deferred]"}
            label = template_libraries_label(cfg)
        self.assertEqual(label, {"templates_young": "Manara et al. 2013; 2017",
                                 "templates_field": "Kesseli et al. 2017"})
        self.assertNotIn("Bonnefoy", json.dumps(label))
        # sin biblioteca en disco: lo declarado, marcado como no verificado
        self.assertIn("declared_in_config_not_verified",
                      template_libraries_label({"g3_template_family": "x"}))


if __name__ == "__main__":
    unittest.main()
