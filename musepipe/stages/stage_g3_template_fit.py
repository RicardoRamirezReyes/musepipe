"""Stage G3 (template-fit slice): dual-track SpT — templates + indices (§3.2).

Runs the empirical-template chi2 fit and the spectral-index estimator over the
fit-ready spectrum, plus the non-stellar power-law proxy, and records the
gravity-class Δχ² for G4. The two SpT vias are kept as SEPARATE rows; their
discrepancy feeds err_sys of ``spectral_type`` but is never averaged away.

Plan 2026-09-23 (decisión ``docs/2026-09-23_decision_g3_resolucion_y_bibliotecas.md``,
Q1–Q4 aprobadas): **cada biblioteca es una prueba independiente** con su bloque
en ``library_tests`` (procedencia leída de su PROVENANCE.json, resultado, borde
del eje de SpT y variante binada de sensibilidad, Q4). Una biblioteca más
gruesa que MUSE (Kesseli) se compara con el DATO degradado a su resolución
(Q2). El Δχ² de gravedad que lee G4 se calcula a la resolución común del par
joven/campo, la más gruesa (Q3), y se declara como tal.

Only WP-G3R-11 runs this on real data; ``compute_`` accepts injected inputs so
the machinery is validated on synthetic fixtures here (plan §0.5.7).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..models import validate_label
from ..models.extinction import CCMExtinction
from ..models.indices import indices_to_spt, measure_indices
from ..models.libraries import (data_frame, gravity_pair, resolve_declaration,
                                template_library_entries)
from ..models.manifest import library_root
from ..models.observed import (DEFAULT_BIN_CHANNELS, build_fit_masks, fit_spectrum,
                               fit_spectrum_from_inputs, load_final_spectrum,
                               load_fit_inputs)
from ..models.template_fit import classify_gravity, fit_powerlaw, fit_templates
from ..models.templates import EmpiricalTemplateLibrary
from ..paths import RunPaths
from .stage_g3_accretion import TABLE_FIELDS


def stage_g3_template_fit_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    p = RunPaths.from_project_root(run_id, root)
    return {"paths": p,
            "fit_json": p.stage_dir / "g3_template_fit.json",
            "rows_json": p.stage_dir / "g3_rows_template.json"}


def _axis(triple):
    lo, hi, step = (float(x) for x in triple)
    n = int(round((hi - lo) / step)) + 1
    return lo + step * np.arange(n)


def _row(prop, value, label, **kw):
    assert validate_label(prop, label), f"invalid label {label} for {prop}"
    row = {k: "" for k in TABLE_FIELDS}
    row.update(property=prop, value=value, label=label)
    row.update({k: v for k, v in kw.items() if k in TABLE_FIELDS})
    return row


def _build_libraries(cfg):
    """Una biblioteca por entrada de ``g3_template_libraries`` (o las dos del
    D1/D2), cada una con su declaración leída de su ``PROVENANCE.json``."""
    root = library_root(cfg, project_root=cfg["project_root"])
    out = []
    for entry in template_library_entries(cfg):
        family = root / entry["subdir"]
        decl = resolve_declaration(entry, family)
        if decl.gravity_class is None:
            raise RuntimeError(f"biblioteca {decl.name}: gravity_class no declarada")
        out.append(EmpiricalTemplateLibrary(
            family, gravity_class=decl.gravity_class, citation=decl.citation,
            version=entry.get("version"), declaration=decl))
    return out


def _coarsest_fwhm(lib):
    """FWHM(λ) de la R más gruesa que declara la biblioteca (0 si nítida)."""
    decl = lib.declaration
    r = decl.min_R() if decl is not None else None
    if r is None and decl is not None and decl.resolution_fwhm_A is not None:
        fw = float(decl.resolution_fwhm_A)
        return lambda w: np.full(np.shape(w), fw)
    if r is None or np.isinf(r):
        return lambda w: np.zeros(np.shape(w))
    return lambda w: np.asarray(w, float) / float(r)


def _target(lsf, libs):
    """Resolución común: la más gruesa entre el dato y las bibliotecas (λ)."""
    fns = [_coarsest_fwhm(lib) for lib in libs]
    return lambda w: np.maximum.reduce([np.full(np.shape(w), lsf)] + [f(w) for f in fns])


def _coarser_than_data(lsf, libs, wave_range):
    w = np.linspace(float(wave_range[0]), float(wave_range[1]), 64)
    return bool(np.any(_target(lsf, libs)(w) > lsf * (1 + 1e-9)))


def _resolution_block(lsf, libs, wave_range, *, degraded, label):
    w = np.array([float(wave_range[0]), 0.5 * (float(wave_range[0]) + float(wave_range[1])),
                  float(wave_range[1])])
    tgt = _target(lsf, libs)(w)
    return {"label": label, "data_lsf_fwhm_A": lsf,
            "target_fwhm_A_at": {f"{x:.0f}": float(t) for x, t in zip(w, tgt)},
            "data_degraded": bool(degraded),
            "libraries": [lib.name for lib in libs]}


def _telluric_mask(lib, wave):
    """Canales del dato a enmascarar para la prueba de ``lib`` (bandas telúricas
    que la biblioteca NO tiene corregidas)."""
    wave = np.asarray(wave, float)
    m = np.zeros(wave.shape, bool)
    decl = lib.declaration
    for band in (decl.telluric_mask_bands if decl is not None else []):
        m |= (wave >= float(band["lo_A"])) & (wave <= float(band["hi_A"]))
    return m


def _masked_inputs(inputs, libs):
    extra = np.zeros(np.shape(inputs["wave"]), bool)
    for lib in libs:
        extra |= _telluric_mask(lib, inputs["wave"])
    if not extra.any():
        return inputs
    return {**inputs, "mask": np.asarray(inputs["mask"], bool) | extra}


def _masked_fit_spec(fs, libs):
    """Sin datos por canal (tests): los bins en bandas enmascaradas → NaN."""
    extra = np.zeros(np.shape(fs.wave_bin), bool)
    for lib in libs:
        extra |= _telluric_mask(lib, fs.wave_bin)
    if not extra.any():
        return fs
    from dataclasses import replace
    return replace(fs, flux_bin=np.where(extra, np.nan, fs.flux_bin))


class _Spectra:
    """Espectros de ajuste de una prueba: nativo (tipo) y binado (bondad), con el
    dato degradado si alguna biblioteca de la prueba es más gruesa que MUSE y
    las bandas telúricas de sus bibliotecas enmascaradas."""

    def __init__(self, libs, *, lsf, inputs, fs_native, cfg, wave_range):
        self.libs = list(libs)
        self.coarser = _coarser_than_data(lsf, self.libs, wave_range)
        self.target = _target(lsf, self.libs) if self.coarser else None
        self.inputs = None if inputs is None else _masked_inputs(inputs, self.libs)
        self.degraded = self.coarser and inputs is not None
        self.fs_native = fs_native
        self.cfg, self.lsf, self.wave_range = cfg, lsf, wave_range
        self.masked = any(lib.declaration is not None and lib.declaration.telluric_mask_bands
                          for lib in self.libs)

    def get(self, n_channels):
        if self.inputs is None:
            # espectro inyectado sin bin_channels = nativo (1)
            if int(n_channels) != int(getattr(self.fs_native, "bin_channels", None) or 1):
                return None
            return _masked_fit_spec(self.fs_native, self.libs)
        lsf, target = self.lsf, self.target
        kern = ((lambda w: np.sqrt(np.clip(target(w) ** 2 - lsf ** 2, 0.0, None)))
                if self.degraded else None)
        return fit_spectrum_from_inputs(
            self.inputs, n_channels=int(n_channels), wave_range=self.wave_range,
            max_masked_frac=float(self.cfg.get("g3_fit_bin_max_masked_frac", 0.5)),
            data_kernel_fwhm_A=kern)

    @property
    def model_target(self):
        return self.target if self.degraded else None

    def resolution(self, label_native, label_degraded):
        blk = _resolution_block(self.lsf, self.libs, self.wave_range,
                                degraded=self.degraded,
                                label=label_degraded if self.degraded else label_native)
        if self.coarser and not self.degraded:
            blk["note"] = ("fit_spec inyectado sin datos por canal: el dato no se pudo "
                           "degradar; resolution_mismatch marcado")
        bands = [b for lib in self.libs if lib.declaration is not None
                 for b in lib.declaration.telluric_mask_bands]
        if bands:
            blk["telluric_bands_masked_in_data"] = bands
        return blk


_DCHI2_NOTE = ("native chi2 uses per-channel variance sigma^2 * (n/n_eff) of the G1 block, "
               "i.e. Delta-chi2 = Delta-chi2_raw * n_eff/n: intervals use effective dof")


def _native_type(res, spectra):
    return {
        "spt_best": res["spt_best"], "spt_best_code": res["spt_best_code"],
        "spt_best_object": res["spt_best_object"], "spt_interval": res["spt_interval"],
        "dchi2_confidence": res["dchi2_confidence"], "dchi2_by_spt": res["dchi2_by_spt"],
        "dchi2_correction": _DCHI2_NOTE, "edge": res["edge"], "spt_axis": res["spt_axis"],
        "av_best": res["av_best"],
        "chi2_red_informative": res["chi2_red_min"],
        "chi2_red_note": ("information only: at native sampling every channel is noise-"
                          "dominated and right and wrong templates all give chi2_red ~ 1; "
                          "no absolute acceptance gate here"),
        "ndof": res["ndof"], "n_bins": res["n_bins"], "n_eff": res["n_eff"],
        "n_templates_fitted": res["n_templates_fitted"],
    }


def _binned_gof(res_bin, native, threshold):
    """χ²_ν en 25 Å del MISMO espectro que ganó en el nativo (A_V y escala se
    reajustan), contra el umbral calibrado; y el mejor subtipo del binado."""
    if res_bin is None:
        return {"status": "not_computed", "reason": "no per-channel data to bin"}
    row = next((r for r in res_bin["ranking"]
                if r["spt_code"] == native["spt_best_code"]
                and r.get("object") == native["spt_best_object"]), None)
    chi2_red = float(row["chi2_red"]) if row is not None else None
    thr = threshold.get("value") if threshold else None
    ok = None if (chi2_red is None or thr is None) else bool(chi2_red <= float(thr))
    return {
        "native_best_spt": native["spt_best"], "native_best_object": native["spt_best_object"],
        "chi2_red_native_best": chi2_red,
        "av_native_best": (row["av_best"] if row is not None else None),
        "threshold": thr, "threshold_source": (threshold or {}).get("source"),
        "pass": ok,
        "binned_best_spt": res_bin["spt_best"], "binned_best_object": res_bin["spt_best_object"],
        "binned_spt_interval": res_bin["spt_interval"], "binned_edge": res_bin["edge"],
        "chi2_red_binned_best": res_bin["chi2_red_min"], "ndof": res_bin["ndof"],
        "fit_bin_channels": res_bin.get("bin_channels"),
        "note": "goodness of fit only; never averaged with native_type",
    }


def _library_test(lib, *, lsf, ext, av_axis, infl, dframe, fs_native, inputs, cfg,
                  n_type, n_gof, wave_range, dchi2_conf, threshold):
    """La prueba de UNA biblioteca: procedencia + ``native_type`` + ``binned_gof``."""
    sp = _Spectra([lib], lsf=lsf, inputs=inputs, fs_native=fs_native, cfg=cfg,
                  wave_range=wave_range)
    kw = dict(av_axis=av_axis, lsf_fwhm_A=lsf, chi2red_inflate_threshold=infl,
              data_frame=dframe, target_fwhm_A=sp.model_target, per_spectrum=True)
    fs = sp.get(n_type)
    res = fit_templates(fs, lib, ext, dchi2_confidence=dchi2_conf, **kw)
    fs_b = sp.get(n_gof) if int(n_gof) != int(n_type) else None
    res_b = fit_templates(fs_b, lib, ext, **kw) if fs_b is not None else None
    if res_b is not None:
        res_b["bin_channels"] = int(n_gof)
    native = _native_type(res, sp)
    block = {
        "provenance": (lib.declaration.to_qc(n_by_spt=lib.n_by_spt())
                       if lib.declaration is not None else {"name": lib.name}),
        "resolution": sp.resolution("MUSE LSF (template degraded)",
                                    "library resolution (data degraded, Q2)"),
        "native_type": native,
        "binned_gof": _binned_gof(res_b, native, threshold),
    }
    return block, res, fs, sp


def _result_summary(res):
    return {k: res[k] for k in ("spt_best", "spt_best_code", "spt_best_object",
                                "spt_interval", "av_best", "chi2_min", "chi2_red_min",
                                "ndof", "n_bins", "n_eff", "edge", "spt_axis",
                                "chi2_by_spt", "n_templates_fitted")}


def _gravity_common(pair_libs, *, lsf, ext, av_axis, infl, dframe, fs_native, inputs,
                    cfg, n_type, wave_range, alpha_axis, **_):
    """Δχ² entre clases a RESOLUCIÓN COMÚN (Q3), en el ajuste NATIVO (tipo)."""
    libs = list(pair_libs.values())
    sp = _Spectra(libs, lsf=lsf, inputs=inputs, fs_native=fs_native, cfg=cfg,
                  wave_range=wave_range)
    fs = sp.get(n_type)
    fits = {cls: fit_templates(fs, lib, ext, av_axis=av_axis, lsf_fwhm_A=lsf,
                               chi2red_inflate_threshold=infl, data_frame=dframe,
                               target_fwhm_A=sp.model_target, per_spectrum=True)
            for cls, lib in pair_libs.items()}
    pl = fit_powerlaw(fs, alpha_axis=alpha_axis)
    grav = classify_gravity({**fits, "nonstellar": pl})
    out = {cls: grav[cls] for cls in (*pair_libs, "nonstellar")}
    out.update({
        "best_class": grav["best_class"], "dchi2_by_class": grav["dchi2_by_class"],
        "dchi2_correction": _DCHI2_NOTE,
        "resolution": sp.resolution("MUSE LSF (common)",
                                    "common resolution (coarsest of the pair; Q3)"),
        "libraries": {cls: lib.name for cls, lib in pair_libs.items()},
        "spt_best_at_common_resolution": {cls: f["spt_best"] for cls, f in fits.items()},
        "fit_bin_channels": int(n_type),
        "representation": "native (type)",
    })
    return out, pl


def _gof_threshold(cfg, young_lib, inputs, *, lsf, ext, av_axis, dframe, wave_range, n_gof):
    """Umbral de la bondad binada: el del config si lo declara, si no calibrado
    en el run (Q1 sobre la representación binada)."""
    declared = cfg.get("g3_gof_chi2red_threshold")
    if declared not in (None, "", "auto"):
        return {"value": float(declared), "source": "config g3_gof_chi2red_threshold"}, None
    if inputs is None or young_lib is None:
        return {"value": None, "source": "not calibrated (no per-channel data)"}, None
    from ..models.calibration import calibrate_gof_threshold
    cal = calibrate_gof_threshold(
        young_lib, inputs, wave_range=wave_range, n_channels=int(n_gof), lsf_fwhm_A=lsf,
        extinction=ext, av_axis=av_axis, data_frame=dframe,
        n_draws=int(cfg.get("g3_gof_calibration_n_draws", 20)),
        seed=int(cfg.get("g3_seed", 0)),
        percentile=float(cfg.get("g3_gof_calibration_percentile", 95.0)),
        max_masked_frac=float(cfg.get("g3_fit_bin_max_masked_frac", 0.5)))
    summary = {k: v for k, v in cal.items() if k != "rows"}
    return {"value": cal["threshold_proposed"],
            "source": f"calibrated in-run on {young_lib.name} (Q1, binned, "
                      f"p{summary['percentile']:.0f} of {summary['distribution']['n']})",
            "calibration": summary}, cal


def compute_stage_g3_template_fit(cfg, paths, *, fit_spec=None, per_channel=None,
                                  libraries=None, fit_inputs=None):
    """Una prueba por biblioteca: TIPO en el ajuste nativo (Δχ² entre subtipos) y
    BONDAD en la variante binada de 25 Å contra un umbral calibrado (2026-09-24).

    ``libraries``: lista de :class:`EmpiricalTemplateLibrary` con declaración
    (por defecto, las de ``g3_template_libraries``). ``fit_spec``/``fit_inputs``
    permiten inyectar el dato (tests); sin ellos se lee el run.
    """
    lsf = float(cfg["h01_lsf_fwhm_A"])
    av_axis = _axis(cfg["g3_atmo_av_axis"])
    ext = CCMExtinction(rv=float(cfg.get("h03_rv_extinction", 3.1)),
                        citation=cfg.get("h03_extinction_law_citation", "Cardelli+1989"))
    infl = float(cfg.get("g3_chi2red_inflate_threshold", 1.5))
    dframe = data_frame(cfg)
    n_type = int(cfg.get("g3_type_bin_channels", DEFAULT_BIN_CHANNELS))
    n_gof = int(cfg.get("g3_gof_bin_channels", 20))
    dchi2_conf = float(cfg.get("g3_dchi2_1sigma_per_param", 1.0))
    alpha_axis = _axis(cfg.get("g3_nonstellar_alpha_axis", [-3.0, 3.0, 1.0]))
    inputs = fit_inputs
    if fit_spec is None:
        if inputs is None:
            inputs = load_fit_inputs(cfg, paths["paths"])
        fit_spec = fit_spectrum(cfg, paths["paths"], inputs=inputs, n_channels=n_type)
    wave_range = cfg.get("g3_fit_wave_range_A") or [float(np.min(fit_spec.wave_bin)),
                                                   float(np.max(fit_spec.wave_bin))]
    libs = list(libraries) if libraries is not None else _build_libraries(cfg)
    for lib in libs:
        if lib.declaration is None:
            raise RuntimeError(
                f"biblioteca {lib.family_dir} sin declaración (marco, R, cita): "
                "ninguna biblioteca se ajusta sin declarar su resolución")
    by_name = {lib.name: lib for lib in libs}
    pair = gravity_pair(cfg)
    pair_libs = {cls: by_name[name] for cls, name in pair.items() if name in by_name}

    threshold, _cal = _gof_threshold(cfg, pair_libs.get("young"), inputs, lsf=lsf, ext=ext,
                                     av_axis=av_axis, dframe=dframe, wave_range=wave_range,
                                     n_gof=n_gof)
    common = dict(lsf=lsf, ext=ext, av_axis=av_axis, infl=infl, dframe=dframe,
                  fs_native=fit_spec, inputs=inputs, cfg=cfg, n_type=n_type,
                  wave_range=wave_range)
    tests, fits, test_fs, test_sp = {}, {}, {}, {}
    for lib in libs:
        block, res, fs_lib, sp = _library_test(lib, n_gof=n_gof, dchi2_conf=dchi2_conf,
                                               threshold=threshold, **common)
        tests[lib.name], fits[lib.name], test_fs[lib.name], test_sp[lib.name] = \
            block, res, fs_lib, sp

    if set(pair_libs) == {"young", "field"}:
        gravity, pl_fit = _gravity_common(pair_libs, alpha_axis=alpha_axis, **common)
        others = {}
        for lib in libs:
            if lib.name in (pair_libs["young"].name, pair_libs["field"].name):
                continue
            # cada biblioteca extra contra la de campo, a su resolución común y con
            # sus telúricas enmascaradas en las dos; nunca juntas con la del par
            g, _ = _gravity_common({lib.gravity_class if lib.gravity_class != "field"
                                    else "other": lib, "field": pair_libs["field"]},
                                   alpha_axis=alpha_axis, **common)
            others[lib.name] = {k: g[k] for k in ("dchi2_by_class", "best_class",
                                                  "resolution", "libraries")}
        gravity["other_libraries_vs_field"] = others
        stellar_class = gravity["best_class"] if gravity["best_class"] in pair_libs else "young"
    else:
        missing = sorted({"young", "field"} - set(pair_libs))
        gravity = {"not_available": f"g3_gravity_pair sin biblioteca {missing} "
                                    f"(declaradas: {sorted(by_name)})",
                   "dchi2_by_class": None}
        pl_fit = fit_powerlaw(fit_spec, alpha_axis=alpha_axis)
        stellar_class = "young" if "young" in pair_libs else next(iter(pair_libs), None)
        if stellar_class is None:
            stellar_class = libs[0].gravity_class
            pair_libs[stellar_class] = libs[0]

    lib_best = pair_libs[stellar_class]
    stellar_best = fits[lib_best.name]
    veil_fit = fit_templates(test_fs[lib_best.name], lib_best, ext, av_axis=av_axis,
                             lsf_fwhm_A=lsf, veiling=True, data_frame=dframe,
                             veiling_alpha_axis=cfg.get("g3_veiling_alpha_axis"),
                             target_fwhm_A=test_sp[lib_best.name].model_target,
                             per_spectrum=True)
    veiling_spt_shift = abs(veil_fit["spt_best_code"] - stellar_best["spt_best_code"])
    young_fit = fits[pair_libs["young"].name] if "young" in pair_libs else None
    field_fit = fits[pair_libs["field"].name] if "field" in pair_libs else None
    gof_best = tests[lib_best.name]["binned_gof"]
    acceptance = {"gate": "binned_gof", "library": lib_best.name,
                  "chi2_red": gof_best.get("chi2_red_native_best"),
                  "threshold": gof_best.get("threshold"),
                  "threshold_source": gof_best.get("threshold_source"),
                  "pass": gof_best.get("pass"),
                  "note": ("acceptance of the template match is judged on the 25 A binned "
                           "fit; the native fit measures the type (Delta-chi2) and has no "
                           "absolute gate")}

    # indices (per-channel; only if config supplies definitions — D7)
    idx_defs = dict(cfg.get("g3_spt_indices", {}))
    idx_cal = dict(cfg.get("g3_spt_indices_calibration", {}))
    indices, idx_skipped = {}, {}
    spt_idx_code, spt_idx_err = float("nan"), float("nan")
    if idx_defs:
        if per_channel is None:
            spec = load_final_spectrum(paths["paths"],
                                       err_column=cfg.get("g3_fit_err_column", "flux_err_total"))
            pmask, _ = build_fit_masks(cfg, spec["wave_A"], paths["paths"])
            per_channel = (spec["wave_A"], spec["flux"], spec["flux_err"], pmask)
        pw, pf, pe, pmask = per_channel
        seed = int(cfg.get("g3_seed", 0))
        # measure per index so an unusable one (out of coverage / >50% masked,
        # D7) is skipped and recorded, not fatal to the stage.
        for iname, idef in idx_defs.items():
            try:
                indices.update(measure_indices(pw, pf, pe, {iname: idef}, mask=pmask, seed=seed))
            except RuntimeError as exc:
                idx_skipped[iname] = str(exc)
        if indices:
            spt_idx_code, spt_idx_err = indices_to_spt(indices, idx_cal)

    spt_disc = (abs(stellar_best["spt_best_code"] - spt_idx_code)
                if np.isfinite(spt_idx_code) else float("nan"))
    err_sys_parts = [x for x in (spt_disc, veiling_spt_shift) if np.isfinite(x)]
    err_sys = float(np.sqrt(np.sum(np.square(err_sys_parts)))) if err_sys_parts else ""

    native = n_type == 1
    data_used = ("fit_spectrum (native channels, n_eff dof)" if native
                 else f"fit_spectrum ({n_type}-channel bins)")
    lib_cite = lib_best.declaration.citation
    rows = [
        _row("spectral_type", stellar_best["spt_best"], "empirical_inference",
             unit="SpT_subtype", data_used=data_used,
             method=f"template chi2 fit (library={lib_best.name}, class={stellar_class})",
             calibrations_citations=lib_cite, err_sys=err_sys,
             limitations=("err_sys = |templates - indices| (+) veiling SpT shift"
                          + ("; best SpT on the EDGE of the library axis"
                             if stellar_best["edge"] else "")),
             depends_on="[empirical_templates]", mc_seed=int(cfg.get("g3_seed", 0))),
        _row("spt_templates", stellar_best["spt_best"], "empirical_inference",
             unit="SpT_subtype",
             method=f"template chi2 (library={lib_best.name}, class={stellar_class})",
             calibrations_citations=lib_cite,
             assumptions=f"interval {stellar_best['spt_interval']} by dchi2<=1"
                         + ("; edge of library axis" if stellar_best["edge"] else "")),
        _row("spt_indices",
             (float(spt_idx_code) if np.isfinite(spt_idx_code) else ""),
             "empirical_inference" if np.isfinite(spt_idx_code) else "not_constrained",
             unit="SpT_subtype", method="spectral indices",
             calibrations_citations="; ".join(cfg.get("g3_spt_indices_citations", [])),
             err_stat_lo=(spt_idx_err if np.isfinite(spt_idx_err) else ""),
             err_stat_hi=(spt_idx_err if np.isfinite(spt_idx_err) else ""),
             limitations=("no index definitions in config (D7 transcription pending)"
                          if not idx_defs else "")),
    ]

    fit_json = {
        "stage": "g3_template_fit", "run_id": str(cfg.get("run_id", "")),
        "provisional": True,
        "decision": "docs/2026-09-23_decision_g3_resolucion_y_bibliotecas.md",
        "type_bin_channels": n_type, "gof_bin_channels": n_gof, "data_wave_frame": dframe,
        "acceptance": acceptance, "gof_threshold": threshold,
        # etiquetas LEÍDAS de cada biblioteca (PROVENANCE.json), no del config
        "libraries": {lib.name: lib.declaration.citation for lib in libs},
        "libraries_declared_in_config": {
            "g3_template_citation": cfg.get("g3_template_citation"),
            "g3_template_family": cfg.get("g3_template_family"),
            "g3_template_field_citation": cfg.get("g3_template_field_citation"),
            "note": "config strings are NOT used as labels; each library's "
                    "PROVENANCE.json is (cf. 9dbb3d4)"},
        "library_tests": tests,
        "young": young_fit, "field": field_fit, "nonstellar_powerlaw": pl_fit,
        "gravity_classes": gravity,
        "veiling_variant": {"class": stellar_class, "library": lib_best.name,
                            "spt_best": veil_fit["spt_best"],
                            "spt_shift": veiling_spt_shift, "chi2_min": veil_fit["chi2_min"]},
        "indices": indices, "indices_skipped": idx_skipped,
        "spt_indices": {"code": spt_idx_code, "err": spt_idx_err},
        "spt_templates": {"class": stellar_class, "library": lib_best.name,
                          "code": stellar_best["spt_best_code"],
                          "interval": stellar_best["spt_interval"],
                          "edge": stellar_best["edge"],
                          "chi2_red_informative": stellar_best["chi2_red_min"],
                          "representation": "native (Delta-chi2)"},
        "spt_discrepancy_subtypes": spt_disc,
        "n_bins": int(fit_spec.n_bins), "n_eff": float(fit_spec.n_eff),
    }
    return rows, fit_json


def write_stage_g3_template_fit(rows, fit_json, paths):
    paths["paths"].ensure_base_dirs()
    paths["fit_json"].write_text(json.dumps(fit_json, indent=1, default=str))
    paths["rows_json"].write_text(json.dumps(rows, indent=1, default=str))
    return {"fit_json": paths["fit_json"], "rows_json": paths["rows_json"]}


def run_stage_g3_template_fit(run_id=None, *, project_root=None, overrides=None,
                              allow_run_id_mismatch=False):
    rc = load_run_config(run_id, project_root=project_root,
                         allow_run_id_mismatch=allow_run_id_mismatch)
    cfg = dict(rc.config)
    if overrides:
        cfg.update(overrides)
    cfg["run_id"] = rc.run_id
    cfg["project_root"] = str(rc.paths.project_root)
    paths = stage_g3_template_fit_paths(cfg["run_id"], project_root=cfg["project_root"])
    rows, fit_json = compute_stage_g3_template_fit(cfg, paths)
    written = write_stage_g3_template_fit(rows, fit_json, paths)
    return {"config": cfg, "paths": paths, "fit_json": fit_json, "written": written}


__all__ = ["compute_stage_g3_template_fit", "run_stage_g3_template_fit",
           "stage_g3_template_fit_paths", "write_stage_g3_template_fit"]
