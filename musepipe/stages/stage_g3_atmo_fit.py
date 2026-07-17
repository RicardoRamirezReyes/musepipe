"""Stage G3 (atmosphere-fit slice): Teff, A_V, logg, Omega + degeneracy maps
(spec G3 §3.3, plan WP-G3R-8).

Runs the 3-D BT-Settl grid fit over the fit-ready spectrum and treats the
Δχ²(Teff,A_V) / Δχ²(Teff,logg) maps as first-class products (V2). The full-range
(D8) and veiling (D10) variants are reported as systematics. logg is expected to
come out ``not_constrained`` — that is a valid result, not a failure.

Only WP-G3R-11 runs this on real data; ``compute_`` accepts injected inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..models import validate_label
from ..models.btsettl import BTSettlLibrary
from ..models.extinction import CCMExtinction
from ..models.fit import fit_grid_3d
from ..models.manifest import library_root
from ..models.observed import fit_spectrum
from ..paths import RunPaths
from .stage_g3_accretion import TABLE_FIELDS

# Joint 2-parameter Δχ² levels for 1/2/3σ (Numerical Recipes / Avni 1976).
_SIGMA_LEVELS = (2.30, 6.17, 11.83)


def stage_g3_atmo_fit_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    p = RunPaths.from_project_root(run_id, root)
    return {"paths": p,
            "npz": p.stage_dir / "g3_atmo_fit.npz",
            "rows_json": p.stage_dir / "g3_rows_atmo.json",
            "plot_teff_av": p.plot_dir / "g3_dchi2_teff_av.png",
            "plot_teff_logg": p.plot_dir / "g3_dchi2_teff_logg.png"}


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


def _interval_row(prop, best, interval, base_label, *, unit, cite, depends,
                  err_sys="", extra_lim=""):
    if interval == "not_constrained" or interval is None:
        return _row(prop, "", "not_constrained", unit=unit,
                    calibrations_citations=cite, depends_on=depends,
                    limitations=extra_lim or "flat / unconstrained over the axis")
    lo, hi = interval
    return _row(prop, float(best), base_label, unit=unit,
                err_stat_lo=float(best) - float(lo), err_stat_hi=float(hi) - float(best),
                err_sys=err_sys, calibrations_citations=cite, depends_on=depends,
                limitations=extra_lim)


def compute_stage_g3_atmo_fit(cfg, paths, *, fit_spec=None, fit_spec_full=None,
                              library=None):
    lsf = float(cfg["h01_lsf_fwhm_A"])
    ext = CCMExtinction(rv=float(cfg.get("h03_rv_extinction", 3.1)),
                        citation=cfg.get("h03_extinction_law_citation", "Cardelli+1989"))
    teff_axis = _axis(cfg["g3_atmo_teff_axis_k"])
    logg_axis = _axis(cfg["g3_atmo_logg_axis"])
    av_axis = _axis(cfg["g3_atmo_av_axis"])
    infl = float(cfg.get("g3_chi2red_inflate_threshold", 1.5))
    sysfrac = float(cfg.get("g3_sys_fluxcal_frac", 0.10))
    if fit_spec is None:
        fit_spec = fit_spectrum(cfg, paths["paths"])
    if library is None:
        root = library_root(cfg, project_root=cfg["project_root"])
        library = BTSettlLibrary(root / "bt-settl-cifist",
                                 citation=cfg.get("g3_atmosphere_citation", "Allard et al. 2012"),
                                 version=cfg.get("g3_atmosphere_version"))

    def run(fs, veiling=False):
        return fit_grid_3d(fs, library, ext, teff_axis=teff_axis, logg_axis=logg_axis,
                           av_axis=av_axis, lsf_fwhm_A=lsf, chi2red_inflate_threshold=infl,
                           sys_fluxcal_frac=sysfrac, veiling=veiling,
                           veiling_alpha_axis=cfg.get("g3_veiling_alpha_axis",
                                                      (-2.0, -1.0, 0.0, 1.0, 2.0)))

    primary = run(fit_spec)
    veil = run(fit_spec, veiling=True) if cfg.get("g3_veiling_variant", True) else None
    full = run(fit_spec_full) if fit_spec_full is not None else None

    # systematics on Teff / A_V from the variants
    def sys_of(key):
        parts = [primary["interp_error"][key]]
        for var in (veil, full):
            if var is not None:
                parts.append(abs(var[f"{key}_best"] - primary[f"{key}_best"]))
        return float(np.sqrt(np.sum(np.square(parts))))

    cite = getattr(library, "citation", None) or cfg.get(
        "g3_atmosphere_citation", "Allard et al. 2012")
    rows = [
        _interval_row("teff", primary["teff_best"], primary["teff_interval"],
                      "atmospheric_model_dependent", unit="K", cite=cite,
                      depends="[atmospheric_model]", err_sys=sys_of("teff"),
                      extra_lim="Teff-A_V degeneracy; see dchi2 maps"),
        _interval_row("a_v_spectral", primary["av_best"], primary["av_interval"],
                      "atmospheric_model_dependent", unit="mag", cite=cite,
                      depends="[atmospheric_model, extinction_law]", err_sys=sys_of("av")),
        _interval_row("logg", primary["logg_best"], primary["logg_interval"],
                      "atmospheric_model_dependent", unit="dex", cite=cite,
                      depends="[atmospheric_model]",
                      extra_lim="expected not_constrained (optical low-g sensitivity)"),
        _row("omega_scale", float(primary["omega_best"]), "atmospheric_model_dependent",
             unit="(R/d)^2_arbitrary", err_stat_lo=primary["omega_err_stat"],
             err_stat_hi=primary["omega_err_stat"], err_sys=primary["omega_err_sys"],
             calibrations_citations=cite, depends_on="[atmospheric_model, flux_calibration]",
             limitations="abs. flux-cal systematic 10% (sys_fluxcal)"),
    ]

    qc = {
        "stage": "g3_atmo_fit", "run_id": str(cfg.get("run_id", "")), "provisional": True,
        "library": {"family": cfg.get("g3_atmosphere_family"), "citation": cite},
        "axes": {"teff": [float(teff_axis[0]), float(teff_axis[-1]), float(teff_axis.size)],
                 "logg": logg_axis.tolist(), "av_n": int(av_axis.size)},
        "best": {k: primary[k] for k in ("teff_best", "logg_best", "av_best",
                                          "omega_best", "chi2_min", "chi2_red")},
        "intervals": {"teff": primary["teff_interval"], "logg": primary["logg_interval"],
                      "av": primary["av_interval"]},
        "edge_touch": primary["edge_touch"], "interp_error": primary["interp_error"],
        "omega": {"stat": primary["omega_err_stat"], "sys": primary["omega_err_sys"],
                  "total": primary["omega_err_total"]},
        "inflate": primary["inflate"],
        "variants": {
            "veiling": None if veil is None else {"teff_best": veil["teff_best"],
                                                  "av_best": veil["av_best"],
                                                  "chi2_min": veil["chi2_min"]},
            "full_range": None if full is None else {"teff_best": full["teff_best"],
                                                     "av_best": full["av_best"],
                                                     "chi2_min": full["chi2_min"]},
        },
    }
    return rows, qc, primary


def _plot_map(out_path, x_axis, y_axis, dchi2, *, xlabel, ylabel, best):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(5, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    z = np.asarray(dchi2).T  # (ny, nx)
    z = np.where(np.isfinite(z), z, np.nanmax(z[np.isfinite(z)]) if np.isfinite(z).any() else 1e3)
    ax.contourf(x_axis, y_axis, z, levels=[0.0, *_SIGMA_LEVELS, np.inf],
                colors=["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7"])
    cs = ax.contour(x_axis, y_axis, z, levels=_SIGMA_LEVELS, colors="k", linewidths=0.7)
    ax.clabel(cs, fmt={l: f"{s}σ" for l, s in zip(_SIGMA_LEVELS, (1, 2, 3))}, fontsize=7)
    if best is not None:
        ax.plot(best[0], best[1], "r*", ms=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    return out_path


def write_stage_g3_atmo_fit(rows, qc, primary, paths, *, make_plots=True):
    paths["paths"].ensure_base_dirs()
    np.savez(paths["npz"], dchi2_3d=primary["dchi2_3d"],
             scales_3d=primary["scales_3d"],
             dchi2_teff_av=primary["dchi2_teff_av"],
             dchi2_teff_logg=primary["dchi2_teff_logg"],
             teff_axis=primary["teff_axis"], logg_axis=primary["logg_axis"],
             av_axis=primary["av_axis"])
    paths["rows_json"].write_text(json.dumps(rows, indent=1, default=str))
    (paths["paths"].stage_dir / "g3_atmo_fit_qc.json").write_text(
        json.dumps(qc, indent=1, default=str))
    out = {"npz": paths["npz"], "rows_json": paths["rows_json"]}
    if make_plots:
        out["plot_teff_av"] = _plot_map(
            paths["plot_teff_av"], primary["teff_axis"], primary["av_axis"],
            primary["dchi2_teff_av"], xlabel="Teff [K]", ylabel="A_V [mag]",
            best=(primary["teff_best"], primary["av_best"]))
        out["plot_teff_logg"] = _plot_map(
            paths["plot_teff_logg"], primary["teff_axis"], primary["logg_axis"],
            primary["dchi2_teff_logg"], xlabel="Teff [K]", ylabel="logg [dex]",
            best=(primary["teff_best"], primary["logg_best"]))
    return out


def run_stage_g3_atmo_fit(run_id=None, *, project_root=None, overrides=None,
                          allow_run_id_mismatch=False):
    rc = load_run_config(run_id, project_root=project_root,
                         allow_run_id_mismatch=allow_run_id_mismatch)
    cfg = dict(rc.config)
    if overrides:
        cfg.update(overrides)
    cfg["run_id"] = rc.run_id
    cfg["project_root"] = str(rc.paths.project_root)
    paths = stage_g3_atmo_fit_paths(cfg["run_id"], project_root=cfg["project_root"])
    fit_spec = fit_spectrum(cfg, paths["paths"])
    full_cfg = dict(cfg)
    full_cfg["g3_fit_wave_range_A"] = cfg.get("g3_fit_wave_range_full_A",
                                              cfg.get("g3_fit_wave_range_A"))
    fit_spec_full = fit_spectrum(full_cfg, paths["paths"])
    rows, qc, primary = compute_stage_g3_atmo_fit(
        cfg, paths, fit_spec=fit_spec, fit_spec_full=fit_spec_full)
    written = write_stage_g3_atmo_fit(rows, qc, primary, paths)
    return {"config": cfg, "paths": paths, "qc": qc, "written": written}


__all__ = ["compute_stage_g3_atmo_fit", "run_stage_g3_atmo_fit",
           "stage_g3_atmo_fit_paths", "write_stage_g3_atmo_fit"]
