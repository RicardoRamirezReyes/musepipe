"""Stage G3 orchestrator (plan WP-G3R-11): run template_fit → atmo_fit →
derived → accretion on the real spectrum, merge the rows into the final table,
run V1–V6, quantify consistency (§1.3), and evaluate the §8.2 STOP conditions.

This is the FIRST real execution; the code is frozen (committed) before it runs.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..constants import MSUN_OVER_MJUP, RJUP_CM, RSUN_CM
from ..models import validate_label
from ..models.btsettl import BTSettlLibrary
from ..models.consistency import (
    CONSISTENCY_FIELDS, consistency_row, not_computed_row,
)
from ..models.extinction import CCMExtinction
from ..models.fit import fit_grid_3d
from ..models.manifest import library_root, verify_manifest
from ..models.accretion import mdot_mc
from ..models.observed import FitSpectrum, build_fit_masks, fit_spectrum, load_final_spectrum, plot_fit_spectrum
from ..models.prep import prepare_template
from ..models.templates import EmpiricalTemplateLibrary
from ..paths import RunPaths
from .stage_g3_accretion import TABLE_FIELDS, run_stage_g3_accretion
from .stage_g3_atmo_fit import run_stage_g3_atmo_fit
from .stage_g3_derived import run_stage_g3_derived
from .stage_g3_template_fit import run_stage_g3_template_fit

REAL_PROPS = {"spectral_type", "spt_templates", "spt_indices", "teff", "a_v_spectral",
              "logg", "omega_scale", "radius", "l_bol", "mass", "age_used", "logg_evol"}

# Atmosphere-dependent quantities that become not_constrained when the run is
# accepted as systematics-limited (WP-11 §8.2). age_used (adopted prior) and the
# continuum-independent accretion rows are kept.
NC_ON_SYSTEMATICS = {"spectral_type", "spt_templates", "spt_indices", "teff",
                     "a_v_spectral", "logg", "omega_scale", "radius", "l_bol",
                     "mass", "logg_evol"}


def _relabel_systematics_limited(rows, *, reason, citation, mdot_lit):
    """Return rows with atmosphere-dependent quantities → not_constrained (the
    railed value preserved in limitations) and Mdot recomputed with literature
    M,R. Continuum-independent accretion rows and age_used are kept."""
    out = []
    for r in rows:
        p = r["property"]
        if p in NC_ON_SYSTEMATICS:
            flagged = str(r.get("value", "")).strip()
            row = {k: "" for k in TABLE_FIELDS}
            row.update(property=p, value="", label="not_constrained",
                       calibrations_citations=citation, depends_on=r.get("depends_on", ""),
                       limitations=f"systematics-limited: {reason} | railed-fit value "
                                   f"{p}={flagged} NOT trusted")
            out.append(row)
        elif p == "mdot" and mdot_lit is not None:
            row = dict(r)
            row.update(value=mdot_lit["p50"], err_stat_lo=mdot_lit["p16"],
                       err_stat_hi=mdot_lit["p84"], data_used="l_acc_combined + literature M,R",
                       assumptions="M,R from literature (Bowler+2017); G3-derived M,R "
                                   "systematics-limited", depends_on="[lacc_relation]")
            out.append(row)
        else:
            out.append(r)
    return out


def stage_g3_assemble_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    p = RunPaths.from_project_root(run_id, root)
    return {"paths": p,
            "template_json": p.stage_dir / "g3_template_fit.json",
            "atmo_qc": p.stage_dir / "g3_atmo_fit_qc.json",
            "atmo_npz": p.stage_dir / "g3_atmo_fit.npz",
            "template_rows": p.stage_dir / "g3_rows_template.json",
            "atmo_rows": p.stage_dir / "g3_rows_atmo.json",
            "derived_rows": p.stage_dir / "g3_rows_derived.json",
            "mass_posterior": p.stage_dir / "g3_mass_posterior.npz",
            "accretion_csv": p.table_dir / "g3_physical_properties.csv",
            "accretion_qc": p.stage_dir / "stage_g3_qc.json",
            "final_csv": p.table_dir / "g3_physical_properties.csv",
            "qc_json": p.stage_dir / "stage_g3_qc.json",
            "consistency_csv": p.table_dir / "g3_consistency.csv",
            "fit_spectrum_png": p.plot_dir / "g3_fit_spectrum.png",
            "best_fit_png": p.plot_dir / "g3_best_fit.png"}


def _frozen_commit(project_root):
    try:
        out = subprocess.run(
            ["git", "-C", str(project_root), "log", "--follow", "--format=%H",
             "docs/2026-07-16_g3_real_frozen_decisions.md"],
            capture_output=True, text=True, timeout=15)
        commits = [c for c in out.stdout.split() if c]
        return commits[-1] if commits else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _axes(cfg):
    def ax(triple):
        lo, hi, step = (float(x) for x in triple)
        n = int(round((hi - lo) / step)) + 1
        return lo + step * np.arange(n)
    return {"teff": ax(cfg["g3_atmo_teff_axis_k"]), "logg": ax(cfg["g3_atmo_logg_axis"]),
            "av": ax(cfg["g3_atmo_av_axis"])}


def _merge_rows(paths):
    rows = []
    with open(paths["accretion_csv"]) as fh:
        for r in csv.DictReader(fh):
            if r["property"].startswith("l_acc") or r["property"] == "mdot":
                rows.append({k: r.get(k, "") for k in TABLE_FIELDS})
    for key in ("template_rows", "atmo_rows", "derived_rows"):
        rows += json.loads(Path(paths[key]).read_text())
    return rows


def _num(row):
    if row is None or row.get("value") in ("", None):
        return None
    try:
        v = float(row["value"])
        e = 0.5 * (float(row.get("err_stat_lo") or 0.0) + float(row.get("err_stat_hi") or 0.0))
        es = float(row.get("err_sys") or 0.0)
        return v, float(np.hypot(e, es))
    except (ValueError, TypeError):
        return None


def _get(rows, prop):
    return _num(next((r for r in rows if r["property"] == prop), None))


# --------------------------------------------------------------------------- #
# Verifications
# --------------------------------------------------------------------------- #
def _v1_real_noise(cfg, run_paths, library, ext, lsf, axes, seed):
    fs = fit_spectrum(cfg, run_paths)
    teff_t, logg_t, av_t = 3000.0, 4.0, 1.0
    base = prepare_template(library.get(teff=teff_t, logg=logg_t), fs.wave_bin,
                            lsf_fwhm_A=lsf, extinction=ext, av=av_t, scale=1.0)
    med_obs = float(np.nanmedian(np.abs(fs.flux_bin)))
    med_mod = float(np.nanmedian(np.abs(base)))
    scale = med_obs / med_mod if med_mod > 0 else 1.0
    rng = np.random.default_rng(int(seed) + 101)
    injected = base * scale + rng.normal(0.0, np.abs(fs.err_bin))
    inj = FitSpectrum(fs.wave_bin, injected, fs.err_bin, fs.n_bins, fs.n_eff, {})
    res = fit_grid_3d(inj, library, ext, teff_axis=axes["teff"], logg_axis=axes["logg"],
                      av_axis=axes["av"], lsf_fwhm_A=lsf)

    def within(interval, truth):
        if interval in (None, "not_constrained"):
            return None
        return bool(interval[0] - 1e-6 <= truth <= interval[1] + 1e-6)
    return {"teff_truth": teff_t, "teff_best": res["teff_best"],
            "teff_interval": res["teff_interval"], "teff_recovered_1sigma": within(res["teff_interval"], teff_t),
            "av_truth": av_t, "av_best": res["av_best"], "av_interval": res["av_interval"],
            "av_recovered_1sigma": within(res["av_interval"], av_t),
            "note": "inject BT-Settl (3000K,logg4,A_V1) + real err_bin noise; refit"}


def _v2(atmo_qc):
    edge, intervals = atmo_qc["edge_touch"], atmo_qc["intervals"]
    undeclared = [ax for ax in ("teff", "logg", "av")
                  if edge.get(ax) and intervals.get(ax) not in ("not_constrained", None)]
    return {"edge_touch": edge, "undeclared_edges": undeclared, "pass": len(undeclared) == 0}


def _v3(template_json):
    st = template_json["spt_templates"]["code"]
    si = template_json["spt_indices"]["code"]
    if si is None or not np.isfinite(si):
        return {"status": "not_checked", "spt_templates": st,
                "reason": "spt_indices not constrained"}
    diff = abs(float(st) - float(si))
    return {"spt_templates": st, "spt_indices": si, "diff_subtypes": diff,
            "pass": diff <= 2.0}


def _v5(accretion_qc, h03_qc):
    v5 = dict(accretion_qc.get("halpha_h03_consistency_v5", {"status": "not_checked"}))
    g3 = v5.get("g3_l_acc_halpha_lsun")
    h03 = h03_qc.get("l_acc_lsun") if isinstance(h03_qc, dict) else None
    if g3 is not None and h03 not in (None, "", 0):
        ratio = float(g3) / float(h03)
        v5.update({"h03_l_acc_halpha_lsun": h03, "ratio_g3_over_h03": ratio,
                   "pass": abs(ratio - 1.0) < 0.02})
    else:
        v5.update({"pass": None, "note": v5.get("note", "") +
                   " | H03 l_acc not found for direct ratio; same-chain guaranteed by unit test"})
    return v5


def _v6(rows):
    invalid = [r["property"] for r in rows if not validate_label(r["property"], r["label"])]
    return {"pass": len(invalid) == 0, "invalid_labels": invalid}


# --------------------------------------------------------------------------- #
# Consistency (§1.3)
# --------------------------------------------------------------------------- #
def _consistency(cfg, rows, template_json, mass_post_path):
    out = []
    st = template_json["spt_templates"]["code"]
    si = template_json["spt_indices"]["code"]
    sie = template_json["spt_indices"]["err"]
    if si is not None and np.isfinite(si):
        out.append(consistency_row("SpT_templates", st, 0.5, "SpT_indices", si,
                                   sie if np.isfinite(sie) else 1.0, "subtype code"))
    else:
        out.append(not_computed_row("SpT_templates", "SpT_indices", "spt_indices not constrained"))

    out.append(not_computed_row(
        "Teff_atmo", "Teff_SpT_HH14",
        "HH14 SpT-Teff scale not transcribed (paper table not machine-accessible; D6 checkpoint pending)"))

    with np.load(mass_post_path) as z:
        fams = {k[len("family_"):]: np.asarray(z[k], float) for k in z.files if k.startswith("family_")}

    def mjup(v):
        return float(np.nanpercentile(v, 50)) * MSUN_OVER_MJUP if v.size else float("nan")

    def mjup_err(v):
        if v.size == 0:
            return float("nan")
        return 0.5 * (np.nanpercentile(v, 84) - np.nanpercentile(v, 16)) * MSUN_OVER_MJUP
    names = [f for f in fams if fams[f].size]
    if len(names) >= 2:
        a, b = names[0], names[1]
        out.append(consistency_row(f"mass_{a}", mjup(fams[a]), mjup_err(fams[a]),
                                   f"mass_{b}", mjup(fams[b]), mjup_err(fams[b]), "M_Jup"))
    else:
        out.append(not_computed_row("mass_family_a", "mass_family_b",
                                    f"only {names} family in-range"))

    mass = _get(rows, "mass")
    lit_m = float(cfg.get("h03_companion_mass_msun", 0.0167)) * MSUN_OVER_MJUP
    lit_me = float(cfg.get("h03_companion_mass_err_msun", 0.0014)) * MSUN_OVER_MJUP
    if mass:
        out.append(consistency_row("mass_G3", mass[0], mass[1], "mass_literature",
                                   lit_m, lit_me, "Bowler+2017 hot-start; M_Jup"))
    else:
        out.append(not_computed_row("mass_G3", "mass_literature", "mass not constrained"))

    av = _get(rows, "a_v_spectral")
    if av:
        out.append(consistency_row("A_V_spectral", av[0], av[1], "A_V_system",
                                   float(cfg["h03_av"]), float(cfg.get("h03_av_err", 0.5)), "mag"))
    else:
        out.append(not_computed_row("A_V_spectral", "A_V_system", "A_V not constrained"))

    rad = _get(rows, "radius")
    r2j = RSUN_CM / RJUP_CM
    lit_r = float(cfg.get("h03_companion_radius_rsun", 0.135)) * r2j
    lit_re = float(cfg.get("h03_companion_radius_err_rsun", 0.02)) * r2j
    if rad:
        out.append(consistency_row("radius_G3", rad[0], rad[1], "radius_literature",
                                   lit_r, lit_re, "Bowler+2017; R_Jup"))
    else:
        out.append(not_computed_row("radius_G3", "radius_literature", "radius not constrained"))
    return out


def _mass_coverage(cfg, mass_post_path):
    n_mc = int(cfg.get("g3_n_mc", 4000))
    with np.load(mass_post_path) as z:
        cov = {k[len("family_"):]: float(np.asarray(z[k]).size) / n_mc
               for k in z.files if k.startswith("family_")}
    return cov


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plot_best_fit(out_path, fs, atmo_model, tmpl_model):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 5))
    FigureCanvasAgg(fig)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.errorbar(fs.wave_bin, fs.flux_bin, yerr=fs.err_bin, fmt=".", ms=3, color="0.3",
                 lw=0.7, label="rebinned data")
    if atmo_model is not None:
        ax1.plot(fs.wave_bin, atmo_model, "-", color="C3", lw=1.0, label="BT-Settl best")
    if tmpl_model is not None:
        ax1.plot(fs.wave_bin, tmpl_model, "-", color="C0", lw=1.0, label="template best")
    ax1.set_ylabel("flux")
    ax1.legend(fontsize=8)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    if atmo_model is not None:
        ax2.plot(fs.wave_bin, (fs.flux_bin - atmo_model) / fs.err_bin, ".", ms=3, color="C3")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("(data-atmo)/err")
    ax2.set_xlabel("wavelength [Å]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    return out_path


def _build_models(cfg, run_paths, atmo_qc, template_json, ext, lsf):
    fs = fit_spectrum(cfg, run_paths)
    root = library_root(cfg, project_root=cfg["project_root"])
    atmo_model = None
    try:
        lib = BTSettlLibrary(root / "bt-settl-cifist",
                             citation=cfg.get("g3_atmosphere_citation", "Allard et al. 2012"),
                             version=cfg.get("g3_atmosphere_version"))
        best = atmo_qc["best"]
        atmo_model = prepare_template(lib.get(teff=best["teff_best"], logg=best["logg_best"]),
                                      fs.wave_bin, lsf_fwhm_A=lsf, extinction=ext,
                                      av=best["av_best"], scale=best["omega_best"])
    except Exception:  # noqa: BLE001
        pass
    tmpl_model = None
    try:
        cls = template_json["spt_templates"]["class"]
        rank0 = template_json[cls]["ranking"][0]
        sub = "templates_young" if cls == "young" else "templates_field"
        cite = cfg["g3_template_citation"] if cls == "young" else cfg["g3_template_field_citation"]
        fw = cfg.get("g3_template_young_fwhm_A") if cls == "young" else cfg.get("g3_template_field_fwhm_A")
        tlib = EmpiricalTemplateLibrary(root / sub, gravity_class=cls, citation=cite,
                                        version=None, resolution_fwhm_A=fw)
        tmpl_model = prepare_template(tlib.get(spt=rank0["spt_code"]), fs.wave_bin,
                                      lsf_fwhm_A=lsf, extinction=ext, av=rank0["av_best"],
                                      scale=rank0["scale_best"],
                                      template_fwhm_A=fw)
    except Exception:  # noqa: BLE001
        pass
    return fs, atmo_model, tmpl_model


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_stage_g3_all(run_id, *, project_root=None, make_figures=True):
    rc = load_run_config(run_id, project_root=project_root)
    cfg = dict(rc.config)
    cfg["run_id"] = rc.run_id
    cfg["project_root"] = str(rc.paths.project_root)
    paths = stage_g3_assemble_paths(run_id, project_root=cfg["project_root"])
    lsf = float(cfg["h01_lsf_fwhm_A"])
    ext = CCMExtinction(rv=float(cfg.get("h03_rv_extinction", 3.1)),
                        citation=cfg.get("h03_extinction_law_citation", "Cardelli+1989"))
    seed = int(cfg.get("g3_seed", 0))

    # 1. run the four stages in dependency order
    run_stage_g3_template_fit(run_id, project_root=project_root)
    run_stage_g3_atmo_fit(run_id, project_root=project_root)
    run_stage_g3_derived(run_id, project_root=project_root)
    run_stage_g3_accretion(run_id, project_root=project_root)

    template_json = json.loads(paths["template_json"].read_text())
    atmo_qc = json.loads(paths["atmo_qc"].read_text())
    accretion_qc = json.loads(paths["accretion_qc"].read_text())
    h03_path = paths["paths"].stage_dir / "stage_h03_qc.json"
    h03_qc = json.loads(h03_path.read_text()) if h03_path.exists() else {}

    rows = _merge_rows(paths)

    # 2. verifications
    root = library_root(cfg, project_root=cfg["project_root"])
    lib = BTSettlLibrary(root / "bt-settl-cifist",
                         citation=cfg.get("g3_atmosphere_citation", "Allard et al. 2012"),
                         version=cfg.get("g3_atmosphere_version"))
    axes = _axes(cfg)
    v1 = _v1_real_noise(cfg, paths["paths"], lib, ext, lsf, axes, seed)
    v2 = _v2(atmo_qc)
    v3 = _v3(template_json)
    v5 = _v5(accretion_qc, h03_qc)
    v6 = _v6(rows)

    consistency = _consistency(cfg, rows, template_json, paths["mass_posterior"])
    coverage = _mass_coverage(cfg, paths["mass_posterior"])

    # 3. STOP conditions (spec §8.2)
    stops = []
    chi2_red = float(atmo_qc["best"]["chi2_red"])
    if chi2_red > 3.0:
        stops.append(f"chi2_red={chi2_red:.2f} > 3 (atmo fit)")
    if not v2["pass"]:
        stops.append(f"V2 undeclared 3-sigma edge on axes {v2['undeclared_edges']}")
    if v3.get("pass") is False:
        stops.append(f"V3 |SpT_templates-SpT_indices|={v3['diff_subtypes']:.1f} > 2 subtypes")
    if v5.get("pass") is False:
        stops.append("V5 H03 consistency failed")
    zero_cov = [f for f, c in coverage.items() if c <= 0.01]
    if zero_cov:
        stops.append(f"mass out of track coverage for {zero_cov} (D5)")

    # 4. libraries provenance
    libs = {}
    for name, sub in (("bt-settl-cifist", "bt-settl-cifist"), ("templates_young", "templates_young"),
                      ("templates_field", "templates_field"), ("tracks_bhac15", "tracks_bhac15"),
                      ("tracks_atmo2020", "tracks_atmo2020")):
        info = verify_manifest(root / sub)
        libs[name] = {"sha256_of_manifest": info["sha256_of_manifest"], "n_files": info["n_files"]}

    qc = {
        "stage": "g3_assemble", "run_id": run_id, "provisional": True,
        "frozen_decisions_commit": _frozen_commit(cfg["project_root"]),
        "libraries": libs,
        "atmo": {k: atmo_qc["best"][k] for k in atmo_qc["best"]},
        "atmo_intervals": atmo_qc["intervals"], "atmo_edge_touch": atmo_qc["edge_touch"],
        "atmo_omega": atmo_qc["omega"], "atmo_inflate": atmo_qc["inflate"],
        "atmo_variants": atmo_qc["variants"],
        "spt": {"templates": template_json["spt_templates"], "indices": template_json["spt_indices"],
                "gravity_classes": template_json["gravity_classes"]},
        "mass_coverage_by_family": coverage,
        "verifications": {"V1_real_noise": v1, "V2_edge": v2, "V3_spt": v3,
                          "V5_h03": v5, "V6_labels": v6,
                          "V4_hrd": str(paths["paths"].plot_dir / "g3_hrd_tracks.png")},
        "consistency_pairs": consistency,
        "stops": stops, "open_issues": [],
    }
    if not stops:
        qc["open_issues"] = []  # pending_libraries removed: both fits produced results
    else:
        qc["open_issues"] = [{"issue": s, "priority": "blocking"} for s in stops]

    # 5. write outputs
    paths["paths"].ensure_base_dirs()
    with paths["final_csv"].open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TABLE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    paths["qc_json"].write_text(json.dumps(qc, indent=1, default=str))
    with paths["consistency_csv"].open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CONSISTENCY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(consistency)

    if make_figures:
        spec = load_final_spectrum(paths["paths"], err_column=cfg.get("g3_fit_err_column", "flux_err_total"))
        pmask, _ = build_fit_masks(cfg, spec["wave_A"], paths["paths"])
        fs, atmo_model, tmpl_model = _build_models(cfg, paths["paths"], atmo_qc, template_json, ext, lsf)
        plot_fit_spectrum(paths["fit_spectrum_png"], wave=spec["wave_A"], flux=spec["flux"],
                          mask=pmask, fit_spec=fs, wave_range=cfg["g3_fit_wave_range_A"])
        _plot_best_fit(paths["best_fit_png"], fs, atmo_model, tmpl_model)

    return {"rows": rows, "qc": qc, "consistency": consistency, "stops": stops,
            "paths": paths}


def finalize_systematics_limited(run_id, *, project_root=None):
    """Post-process an existing G3 run into the honest systematics-limited table
    + QC (no re-fitting), per the human decision (config
    ``g3_atmosphere_systematics_limited``). Atmosphere/derived quantities become
    not_constrained; Mdot uses literature M,R; the young gravity class and the
    accretion limits are retained as the robust results."""
    rc = load_run_config(run_id, project_root=project_root)
    cfg = dict(rc.config)
    cfg["run_id"] = rc.run_id
    cfg["project_root"] = str(rc.paths.project_root)
    paths = stage_g3_assemble_paths(run_id, project_root=cfg["project_root"])
    reason = cfg.get("g3_atmosphere_systematics_limited_reason", "systematics-limited")
    citation = cfg.get("g3_atmosphere_systematics_limited_citation", "")

    with open(paths["final_csv"]) as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    qc = json.loads(paths["qc_json"].read_text())
    accretion_qc = (json.loads(paths["accretion_qc"].read_text())
                    if paths["accretion_qc"].exists() else {})
    template_json = json.loads(paths["template_json"].read_text())

    lacc = next((float(r["value"]) for r in rows
                 if r["property"] == "l_acc_combined" and r.get("value") not in ("", None)), None)
    mdot_lit = None
    if lacc is not None:
        mdot_lit = mdot_mc(
            lacc, float(cfg["h03_companion_mass_msun"]),
            float(cfg.get("h03_companion_mass_err_msun", 0.0014)),
            float(cfg["h03_companion_radius_rsun"]),
            float(cfg.get("h03_companion_radius_err_rsun", 0.02)),
            float(cfg.get("h03_relation_scatter_dex", 0.30)),
            n_mc=int(cfg.get("g3_n_mc", 4000)), seed=int(cfg.get("g3_seed", 0)))

    final = _relabel_systematics_limited(rows, reason=reason, citation=citation,
                                         mdot_lit=mdot_lit)
    consistency = _consistency(cfg, final, template_json, paths["mass_posterior"])

    qc["systematics_limited"] = {
        "accepted": True, "reason": reason, "citation": citation,
        "stops_resolution": ("accepted as systematics-limited (human decision 2026-07-16); "
                             "atmo/derived rows -> not_constrained"),
        "robust_results": {
            "gravity_class": qc.get("spt", {}).get("gravity_classes", {}),
            "accretion": {"l_acc_combined_lsun": lacc,
                          "mdot_msun_yr_literature_MR": (mdot_lit["p50"] if mdot_lit else None)},
        },
        "flagged_atmo_fit_values": qc.get("atmo", {}),
    }
    qc["consistency_pairs"] = consistency
    qc["open_issues"] = [{"issue": "G3 atmospheric inference systematics-limited (C3 continuum "
                                   "systematic); Teff/A_V/R/mass not_constrained",
                          "priority": "accepted_limitation"}]

    with paths["final_csv"].open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TABLE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(final)
    paths["qc_json"].write_text(json.dumps(qc, indent=1, default=str))
    with paths["consistency_csv"].open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CONSISTENCY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(consistency)
    return {"rows": final, "qc": qc, "consistency": consistency, "mdot_literature": mdot_lit}


__all__ = ["finalize_systematics_limited", "run_stage_g3_all", "stage_g3_assemble_paths"]
