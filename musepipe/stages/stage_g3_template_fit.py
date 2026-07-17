"""Stage G3 (template-fit slice): dual-track SpT — templates + indices (§3.2).

Runs the empirical-template chi2 fit (young + field) and the spectral-index
estimator over the fit-ready spectrum, plus the non-stellar power-law proxy, and
records the gravity-class Δχ² for G4. The two SpT vias are kept as SEPARATE rows;
their discrepancy feeds err_sys of ``spectral_type`` but is never averaged away.

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
from ..models.manifest import library_root
from ..models.observed import build_fit_masks, fit_spectrum, load_final_spectrum
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
    root = library_root(cfg, project_root=cfg["project_root"])
    young = EmpiricalTemplateLibrary(
        root / "templates_young", gravity_class="young",
        citation=cfg["g3_template_citation"], version=cfg.get("g3_template_version"),
        resolution_fwhm_A=cfg.get("g3_template_young_fwhm_A"))
    field = EmpiricalTemplateLibrary(
        root / "templates_field", gravity_class="field",
        citation=cfg["g3_template_field_citation"],
        version=cfg.get("g3_template_field_version"),
        resolution_fwhm_A=cfg.get("g3_template_field_fwhm_A"))
    return young, field


def compute_stage_g3_template_fit(cfg, paths, *, fit_spec=None, per_channel=None,
                                  libraries=None):
    lsf = float(cfg["h01_lsf_fwhm_A"])
    av_axis = _axis(cfg["g3_atmo_av_axis"])
    ext = CCMExtinction(rv=float(cfg.get("h03_rv_extinction", 3.1)),
                        citation=cfg.get("h03_extinction_law_citation", "Cardelli+1989"))
    infl = float(cfg.get("g3_chi2red_inflate_threshold", 1.5))
    if fit_spec is None:
        fit_spec = fit_spectrum(cfg, paths["paths"])
    young, field = libraries if libraries is not None else _build_libraries(cfg)

    young_fit = fit_templates(fit_spec, young, ext, av_axis=av_axis, lsf_fwhm_A=lsf,
                              chi2red_inflate_threshold=infl)
    field_fit = fit_templates(fit_spec, field, ext, av_axis=av_axis, lsf_fwhm_A=lsf,
                              chi2red_inflate_threshold=infl)
    pl_fit = fit_powerlaw(fit_spec, alpha_axis=_axis(
        cfg.get("g3_nonstellar_alpha_axis", [-3.0, 3.0, 1.0])))
    gravity = classify_gravity({"young": young_fit, "field": field_fit,
                                "nonstellar": pl_fit})

    stellar_best, stellar_class = ((young_fit, "young")
                                   if young_fit["chi2_min"] <= field_fit["chi2_min"]
                                   else (field_fit, "field"))
    lib_best = young if stellar_class == "young" else field
    veil_fit = fit_templates(fit_spec, lib_best, ext, av_axis=av_axis, lsf_fwhm_A=lsf,
                             veiling=True,
                             veiling_alpha_axis=cfg.get("g3_veiling_alpha_axis"))
    veiling_spt_shift = abs(veil_fit["spt_best_code"] - stellar_best["spt_best_code"])

    # indices (per-channel; only if config supplies definitions — D7)
    idx_defs = dict(cfg.get("g3_spt_indices", {}))
    idx_cal = dict(cfg.get("g3_spt_indices_calibration", {}))
    indices, spt_idx_code, spt_idx_err = {}, float("nan"), float("nan")
    if idx_defs:
        if per_channel is None:
            spec = load_final_spectrum(paths["paths"],
                                       err_column=cfg.get("g3_fit_err_column", "flux_err_total"))
            pmask, _ = build_fit_masks(cfg, spec["wave_A"], paths["paths"])
            per_channel = (spec["wave_A"], spec["flux"], spec["flux_err"], pmask)
        pw, pf, pe, pmask = per_channel
        indices = measure_indices(pw, pf, pe, idx_defs, mask=pmask,
                                  seed=int(cfg.get("g3_seed", 0)))
        spt_idx_code, spt_idx_err = indices_to_spt(indices, idx_cal)

    spt_disc = (abs(stellar_best["spt_best_code"] - spt_idx_code)
                if np.isfinite(spt_idx_code) else float("nan"))
    err_sys_parts = [x for x in (spt_disc, veiling_spt_shift) if np.isfinite(x)]
    err_sys = float(np.sqrt(np.sum(np.square(err_sys_parts)))) if err_sys_parts else ""

    rows = [
        _row("spectral_type", stellar_best["spt_best"], "empirical_inference",
             unit="SpT_subtype", data_used="fit_spectrum (binned)",
             method=f"template chi2 fit (class={stellar_class})",
             calibrations_citations=lib_best.citation, err_sys=err_sys,
             limitations="err_sys = |templates - indices| (+) veiling SpT shift",
             depends_on="[empirical_templates]", mc_seed=int(cfg.get("g3_seed", 0))),
        _row("spt_templates", stellar_best["spt_best"], "empirical_inference",
             unit="SpT_subtype", method=f"template chi2 (class={stellar_class})",
             calibrations_citations=lib_best.citation,
             assumptions=f"interval {stellar_best['spt_interval']} by dchi2<=1"),
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
        "libraries": {"young": young.citation, "field": field.citation},
        "young": young_fit, "field": field_fit, "nonstellar_powerlaw": pl_fit,
        "gravity_classes": {"young": gravity["young"], "field": gravity["field"],
                            "nonstellar": gravity["nonstellar"],
                            "best_class": gravity["best_class"],
                            "dchi2_by_class": gravity["dchi2_by_class"]},
        "veiling_variant": {"class": stellar_class, "spt_best": veil_fit["spt_best"],
                            "spt_shift": veiling_spt_shift, "chi2_min": veil_fit["chi2_min"]},
        "indices": indices, "spt_indices": {"code": spt_idx_code, "err": spt_idx_err},
        "spt_templates": {"class": stellar_class, "code": stellar_best["spt_best_code"],
                          "interval": stellar_best["spt_interval"]},
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
