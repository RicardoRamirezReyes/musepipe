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


def _library_test(lib, *, lsf, ext, av_axis, infl, dframe, fs_native, inputs, cfg,
                  n_primary, n_sens, wave_range):
    """La prueba de UNA biblioteca: procedencia + resultado + variante binada (Q4)."""
    coarser = _coarser_than_data(lsf, [lib], wave_range)
    target = _target(lsf, [lib]) if coarser else None
    data_kernel = ((lambda w: np.sqrt(np.clip(target(w) ** 2 - lsf ** 2, 0.0, None)))
                   if coarser else None)

    def spectrum(n_channels):
        if inputs is None:
            return None
        if not coarser and n_channels == n_primary:
            return fs_native
        return fit_spectrum_from_inputs(
            inputs, n_channels=n_channels, wave_range=wave_range,
            max_masked_frac=float(cfg.get("g3_fit_bin_max_masked_frac", 0.5)),
            data_kernel_fwhm_A=data_kernel)

    fs = spectrum(n_primary) if inputs is not None else fs_native
    degraded = coarser and inputs is not None
    res = fit_templates(fs, lib, ext, av_axis=av_axis, lsf_fwhm_A=lsf,
                        chi2red_inflate_threshold=infl, data_frame=dframe,
                        target_fwhm_A=(target if degraded else None), per_spectrum=True)
    block = {
        "provenance": (lib.declaration.to_qc(n_by_spt=lib.n_by_spt())
                       if lib.declaration is not None else {"name": lib.name}),
        "resolution": _resolution_block(
            lsf, [lib], wave_range, degraded=degraded,
            label=("library resolution (data degraded, Q2)" if degraded
                   else "MUSE LSF (template degraded)")),
        "fit_bin_channels": int(n_primary),
        "result": _result_summary(res),
        "fit": res,
    }
    if coarser and inputs is None:
        block["resolution"]["note"] = ("fit_spec inyectado sin datos por canal: el dato "
                                       "no se pudo degradar; resolution_mismatch marcado")
    if inputs is not None and n_sens and int(n_sens) != int(n_primary):
        fs_s = spectrum(int(n_sens))
        rs = fit_templates(fs_s, lib, ext, av_axis=av_axis, lsf_fwhm_A=lsf,
                           chi2red_inflate_threshold=infl, data_frame=dframe,
                           target_fwhm_A=(target if degraded else None), per_spectrum=True)
        block["sensitivity_binned"] = {
            "label": f"Q4: {int(n_sens)}-channel bins (~25 A); sensitivity only, never averaged",
            "fit_bin_channels": int(n_sens), "result": _result_summary(rs)}
    return block, res, fs


def _result_summary(res):
    return {k: res[k] for k in ("spt_best", "spt_best_code", "spt_best_object",
                                "spt_interval", "av_best", "chi2_min", "chi2_red_min",
                                "ndof", "n_bins", "n_eff", "edge", "spt_axis",
                                "chi2_by_spt", "n_templates_fitted")}


def _gravity_common(pair_libs, *, lsf, ext, av_axis, infl, dframe, fs_native, inputs,
                    cfg, n_primary, wave_range, alpha_axis):
    """Δχ² joven/campo/no estelar a RESOLUCIÓN COMÚN (Q3): la más gruesa del par."""
    libs = list(pair_libs.values())
    coarser = _coarser_than_data(lsf, libs, wave_range)
    target = _target(lsf, libs) if coarser else None
    degraded = coarser and inputs is not None
    if degraded:
        fs = fit_spectrum_from_inputs(
            inputs, n_channels=n_primary, wave_range=wave_range,
            max_masked_frac=float(cfg.get("g3_fit_bin_max_masked_frac", 0.5)),
            data_kernel_fwhm_A=lambda w: np.sqrt(np.clip(target(w) ** 2 - lsf ** 2, 0, None)))
    else:
        fs = fs_native
    fits = {cls: fit_templates(fs, lib, ext, av_axis=av_axis, lsf_fwhm_A=lsf,
                               chi2red_inflate_threshold=infl, data_frame=dframe,
                               target_fwhm_A=(target if degraded else None),
                               per_spectrum=True)
            for cls, lib in pair_libs.items()}
    pl = fit_powerlaw(fs, alpha_axis=alpha_axis)
    grav = classify_gravity({**fits, "nonstellar": pl})
    return {
        "young": grav["young"], "field": grav["field"], "nonstellar": grav["nonstellar"],
        "best_class": grav["best_class"], "dchi2_by_class": grav["dchi2_by_class"],
        "resolution": _resolution_block(
            lsf, libs, wave_range, degraded=degraded,
            label="common resolution (coarsest of the pair; Q3)"),
        "libraries": {cls: lib.name for cls, lib in pair_libs.items()},
        "spt_best_at_common_resolution": {cls: f["spt_best"] for cls, f in fits.items()},
        "fit_bin_channels": int(n_primary),
    }, pl


def compute_stage_g3_template_fit(cfg, paths, *, fit_spec=None, per_channel=None,
                                  libraries=None, fit_inputs=None):
    """Una prueba por biblioteca (plan 2026-09-23) + Δχ² de gravedad a resolución común.

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
    n_primary = int(cfg.get("g3_fit_bin_channels", DEFAULT_BIN_CHANNELS))
    n_sens = int(cfg.get("g3_fit_bin_channels_sensitivity", 20))
    alpha_axis = _axis(cfg.get("g3_nonstellar_alpha_axis", [-3.0, 3.0, 1.0]))
    inputs = fit_inputs
    if fit_spec is None:
        if inputs is None:
            inputs = load_fit_inputs(cfg, paths["paths"])
        fit_spec = fit_spectrum(cfg, paths["paths"], inputs=inputs, n_channels=n_primary)
    wave_range = cfg.get("g3_fit_wave_range_A") or [float(np.min(fit_spec.wave_bin)),
                                                   float(np.max(fit_spec.wave_bin))]
    libs = list(libraries) if libraries is not None else _build_libraries(cfg)
    for lib in libs:
        if lib.declaration is None:
            raise RuntimeError(
                f"biblioteca {lib.family_dir} sin declaración (marco, R, cita): "
                "ninguna biblioteca se ajusta sin declarar su resolución")
    by_name = {lib.name: lib for lib in libs}

    common = dict(lsf=lsf, ext=ext, av_axis=av_axis, infl=infl, dframe=dframe,
                  fs_native=fit_spec, inputs=inputs, cfg=cfg, n_primary=n_primary,
                  wave_range=wave_range)
    tests, fits, test_fs = {}, {}, {}
    for lib in libs:
        block, res, fs_lib = _library_test(lib, n_sens=n_sens, **common)
        tests[lib.name] = block
        fits[lib.name] = res
        test_fs[lib.name] = fs_lib

    pair = gravity_pair(cfg)
    pair_libs = {cls: by_name[name] for cls, name in pair.items() if name in by_name}
    if set(pair_libs) == {"young", "field"}:
        gravity, pl_fit = _gravity_common(pair_libs, alpha_axis=alpha_axis, **common)
        by_young = {}
        for lib in libs:
            if lib.gravity_class == "young" and lib.name != pair_libs["young"].name:
                g, _ = _gravity_common({"young": lib, "field": pair_libs["field"]},
                                       alpha_axis=alpha_axis, **common)
                by_young[lib.name] = {k: g[k] for k in ("dchi2_by_class", "best_class",
                                                        "resolution", "libraries")}
        gravity["other_young_libraries"] = by_young
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
                             target_fwhm_A=(_target(lsf, [lib_best])
                                            if tests[lib_best.name]["resolution"]["data_degraded"]
                                            else None),
                             per_spectrum=True)
    veiling_spt_shift = abs(veil_fit["spt_best_code"] - stellar_best["spt_best_code"])
    young_fit = fits[pair_libs["young"].name] if "young" in pair_libs else None
    field_fit = fits[pair_libs["field"].name] if "field" in pair_libs else None

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

    native = n_primary == 1
    data_used = ("fit_spectrum (native channels, n_eff dof)" if native
                 else f"fit_spectrum ({n_primary}-channel bins)")
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
        "fit_bin_channels": n_primary, "data_wave_frame": dframe,
        # etiquetas LEÍDAS de cada biblioteca (PROVENANCE.json), no del config
        "libraries": {lib.name: lib.declaration.citation for lib in libs},
        "libraries_declared_in_config": {
            "g3_template_citation": cfg.get("g3_template_citation"),
            "g3_template_family": cfg.get("g3_template_family"),
            "g3_template_field_citation": cfg.get("g3_template_field_citation"),
            "note": "config strings are NOT used as labels; each library's "
                    "PROVENANCE.json is (cf. 9dbb3d4)"},
        "library_tests": {name: {k: v for k, v in blk.items() if k != "fit"}
                          for name, blk in tests.items()},
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
                          "chi2_red_min": stellar_best["chi2_red_min"]},
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
