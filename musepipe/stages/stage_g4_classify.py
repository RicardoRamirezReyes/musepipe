"""Stage G4: assemble the hypothesis × test evidence matrix from G0–G3 QC.

Transparent classification (spec G4 §1.1): the product is the matrix. Each cell
records verdict + value + source (re-derivable). Frozen thresholds live in
config and their hash is stamped in the QC (§1.2 / V5 anti-bias). Deferred G3
fits leave T3/T4/T6 `not_available` (they do not score).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from ..classify import (
    HYPOTHESES,
    background_probability,
    classify,
    combine_evidence,
    dominant_tests,
    frozen_thresholds_hash,
)
from ..config import load_run_config
from ..constants import MSUN_OVER_MJUP
from ..paths import RunPaths

CORRELATED_GROUPS = [("T3", "T4", "T6")]  # share the spectrum + A_V fit (spec §4.1)

# Which gravity/mass class each hypothesis belongs to (D13).
_CLASS_OF_HYP = {"planet_forming": "young", "substellar_companion": "young",
                 "brown_dwarf": "young", "m_star_associated": "young",
                 "m_star_background": "field", "contaminant": "nonstellar",
                 "artifact": "nonstellar"}
_HYP_MASS_RANGE = {"planet_forming": "planet_lt13", "substellar_companion": "planet_lt13",
                   "brown_dwarf": "bd_13_75", "m_star_associated": "star_gt75",
                   "m_star_background": "star_gt75"}
_VERDICT_RANK = {"excludes": 0, "disfavors": 1, "neutral": 2, "supports": 3}
_BOUND_HYP = ("planet_forming", "substellar_companion", "brown_dwarf", "m_star_associated")


def stage_g4_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    p = RunPaths.from_project_root(run_id, root)
    return {
        "paths": p,
        "stage01c_qc_json": p.stage_dir / "stage01c_qc.json",
        "stage_h02_qc_json": p.stage_dir / "stage_h02_qc.json",
        "g2_table": p.table_dir / "g2_line_measurements.csv",
        "stage_g3_qc_json": p.stage_dir / "stage_g3_qc.json",
        "g3_mass_posterior": p.stage_dir / "g3_mass_posterior.npz",
        "g3_table": p.table_dir / "g3_physical_properties.csv",
        "matrix_csv": p.table_dir / "g4_evidence_matrix.csv",
        "qc_json": p.stage_dir / "stage_g4_classification.json",
    }


def _read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _cell(verdict, value, source):
    return {"verdict": verdict, "value": value, "source": source}


def _table_value(paths, prop):
    """(value, err) from the G3 table if the row is constrained; else None."""
    p = Path(paths["g3_table"])
    if not p.exists():
        return None
    for r in csv.DictReader(open(p)):
        if r["property"] == prop and r.get("label") != "not_constrained" and r.get("value") not in ("", None):
            try:
                v = float(r["value"])
                e = 0.5 * (float(r.get("err_stat_lo") or 0.0) + float(r.get("err_stat_hi") or 0.0))
                es = float(r.get("err_sys") or 0.0)
                return v, float(math.hypot(e, es))
            except (ValueError, TypeError):
                return None
    return None


def _t3_gravity(grav, excl, disfav):
    """Per-hypothesis gravity-class verdict from classify_gravity Δχ² (D13)."""
    dchi2 = (grav or {}).get("dchi2_by_class")
    if not dchi2:
        return None
    out = {}
    for h in HYPOTHESES:
        d = dchi2.get(_CLASS_OF_HYP[h])
        if d is None:
            out[h] = ("not_available", None)
            continue
        d = float(d)
        v = ("supports" if d <= 0.0 else "excludes" if d > excl
             else "disfavors" if d >= disfav else "neutral")
        out[h] = (v, round(d, 2))
    return out


def _mass_probabilities(mass_post_path, boundaries):
    """P(m<13), P(13-75), P(>75) [M_Jup] from the combined mass posterior."""
    p = Path(mass_post_path)
    if not p.exists():
        return None
    with np.load(p) as z:
        if "combined" not in z.files:
            return None
        m = np.asarray(z["combined"], float) * MSUN_OVER_MJUP
    m = m[np.isfinite(m)]
    if m.size == 0:
        return None
    lo, hi = float(boundaries[0]), float(boundaries[1])
    return {"planet_lt13": float(np.mean(m < lo)),
            "bd_13_75": float(np.mean((m >= lo) & (m < hi))),
            "star_gt75": float(np.mean(m >= hi)), "n_samples": int(m.size)}


def _mass_verdicts(probs, verdict_thr):
    if probs is None:
        return None
    sup = float(verdict_thr.get("supports", 0.6))
    neu = float(verdict_thr.get("neutral", 0.2))
    dis = float(verdict_thr.get("disfavors", 0.05))
    out = {}
    for h in HYPOTHESES:
        rng = _HYP_MASS_RANGE.get(h)
        if rng is None:
            out[h] = ("not_available", None)
            continue
        p = probs[rng]
        v = ("supports" if p > sup else "neutral" if p >= neu
             else "disfavors" if p >= dis else "excludes")
        out[h] = (v, round(p, 3))
    return out


def _combine_verdicts(*parts):
    """Most-unfavorable of the available components (D13); value keeps both."""
    avail = [x for x in parts if x is not None and x[0] != "not_available"]
    if not avail:
        return "not_available", None
    worst = min(avail, key=lambda x: _VERDICT_RANK[x[0]])
    return worst[0], "; ".join(f"{x[0]}({x[1]})" for x in avail)


def _ambiguity_quantification(cfg, paths, g3):
    """P(m<13 / 13-75 / >75 M_Jup) per family + combined vs the frozen mass
    boundaries — the quantitative resolution of the G4 ambiguity. Reports
    not_available (with the gravity Δχ²) when the mass is systematics-limited."""
    syslim = g3.get("systematics_limited", {})
    grav = (g3.get("spt", {}).get("gravity_classes")
            or syslim.get("robust_results", {}).get("gravity_class") or {})
    dchi2 = grav.get("dchi2_by_class")
    boundaries = cfg.get("g4_mass_boundaries_mjup", [13.0, 75.0])
    if syslim.get("accepted"):
        return {"status": "not_available",
                "reason": syslim.get("reason", "mass systematics-limited"),
                "dchi2_gravity_classes": dchi2}
    probs = _mass_probabilities(paths["g3_mass_posterior"], boundaries)
    if probs is None:
        return {"status": "not_available", "reason": "mass posterior unavailable",
                "dchi2_gravity_classes": dchi2}
    per_family = {}
    lo, hi = float(boundaries[0]), float(boundaries[1])
    with np.load(Path(paths["g3_mass_posterior"])) as z:
        for k in z.files:
            if not k.startswith("family_"):
                continue
            m = np.asarray(z[k], float) * MSUN_OVER_MJUP
            m = m[np.isfinite(m)]
            if m.size:
                per_family[k[len("family_"):]] = {
                    "planet_lt13": float(np.mean(m < lo)),
                    "bd_13_75": float(np.mean((m >= lo) & (m < hi))),
                    "star_gt75": float(np.mean(m >= hi))}
    return {"status": "computed", "mass_prob_combined": probs,
            "mass_prob_by_family": per_family, "dchi2_gravity_classes": dchi2}


def build_matrix(cfg, paths):
    b3 = _read(paths["stage01c_qc_json"]) or {}
    h02 = _read(paths["stage_h02_qc_json"]) or {}
    astro = b3.get("astrometry", {})
    comp = b3.get("companion", {})
    matrix = {}

    # T1 astrometry: does sep/PA match the bound-companion prediction?
    sep_dev = astro.get("sep_deviation_sigma")
    if sep_dev is None and astro.get("sep_arcsec") is not None and astro.get("expected_sep_arcsec"):
        sep_dev = abs(astro["sep_arcsec"] - astro["expected_sep_arcsec"]) / max(astro.get("sep_err", 1e-6), 1e-6)
    cpm = bool(cfg.get("g4_cpm_confirmed", False))
    src1 = "stage01c_qc.astrometry" + ("+config.g4_cpm_confirmed" if cpm else "")
    if astro.get("expected_sep_arcsec") is not None and sep_dev is not None and sep_dev < 5:
        # matches the predicted bound position (Bowler+ literature)
        v_bound = "supports"
        v_bg = "disfavors" if cpm else "neutral"
    else:
        v_bound, v_bg = "neutral", "neutral"
    matrix["T1"] = {h: _cell(v_bound if h in ("planet_forming", "substellar_companion", "brown_dwarf", "m_star_associated")
                             else (v_bg if h == "m_star_background" else "neutral"),
                             sep_dev, src1) for h in HYPOTHESES}

    # T2 morphology: point source (B3 clean detection) vs extended (galaxy) / artifact
    snr = comp.get("snr_detection")
    pointlike = snr is not None and float(snr) >= float(cfg.get("g4_pointsource_snr_min", 5.0))
    matrix["T2"] = {h: _cell(
        "supports" if (pointlike and h in ("planet_forming", "substellar_companion", "brown_dwarf", "m_star_associated", "m_star_background"))
        else ("disfavors" if (pointlike and h in ("contaminant", "artifact")) else "not_available"),
        snr, "stage01c_qc.companion.snr_detection") for h in HYPOTHESES}

    # T3 (gravity class + mass) / T4 (radius) / T6 (A_V) from the REAL G3 QC.
    # An input G3 reports not_constrained / systematics-limited -> that component
    # is not_available (spec: never punish for not measuring).
    g3 = _read(paths["stage_g3_qc_json"]) or {}
    g3_deferred = any("pending_libraries" in str(i.get("issue", "")) for i in (g3.get("open_issues") or []))
    g3_syslim = bool(g3.get("systematics_limited", {}).get("accepted"))
    excl = float(cfg.get("g4_t3_dchi2_excl", 9.0))
    disfav = float(cfg.get("g4_t3_dchi2_disfavor", 4.0))
    grav = (g3.get("spt", {}).get("gravity_classes")
            or g3.get("systematics_limited", {}).get("robust_results", {}).get("gravity_class") or {})
    t3_grav = _t3_gravity(grav, excl, disfav)
    # mass component only if trustworthy (not systematics-limited / deferred)
    mass_ok = grav and not g3_syslim and not g3_deferred
    mass_probs = _mass_probabilities(paths["g3_mass_posterior"], cfg.get("g4_mass_boundaries_mjup", [13.0, 75.0])) if mass_ok else None
    t3_mass = _mass_verdicts(mass_probs, cfg.get("g4_mass_prob_verdicts", {}))
    matrix["T3"] = {}
    for h in HYPOTHESES:
        gvh = t3_grav.get(h) if t3_grav else None
        mvh = t3_mass.get(h) if t3_mass else None
        v, val = _combine_verdicts(gvh, mvh)
        note = "" if mvh else ("; mass systematics-limited" if g3_syslim else "; mass n/a")
        matrix["T3"][h] = _cell(v, val, "g3.gravity_class(dchi2)+mass" + note)

    rad = None if (g3_syslim or g3_deferred) else _table_value(paths, "radius")
    if rad is None:
        note = " (systematics-limited)" if g3_syslim else " (deferred)" if g3_deferred else " (not_constrained)"
        matrix["T4"] = {h: _cell("not_available", None, "g3.radius(Omega,d)" + note) for h in HYPOTHESES}
    else:
        r, _ = rad
        lo, hi = cfg.get("g4_t4_radius_rjup_range", [0.5, 3.0])
        vb = "supports" if lo <= r <= hi else "disfavors"
        matrix["T4"] = {h: _cell(vb if h in _BOUND_HYP else "neutral", round(r, 3),
                                 "g3.radius(Omega,d)") for h in HYPOTHESES}

    av = None if (g3_syslim or g3_deferred) else _table_value(paths, "a_v_spectral")
    if av is None:
        note = " (systematics-limited)" if g3_syslim else " (deferred)" if g3_deferred else " (not_constrained)"
        matrix["T6"] = {h: _cell("not_available", None, "g3.a_v_spectral" + note) for h in HYPOTHESES}
    else:
        a, ae = av
        sys_av, sys_err = float(cfg["h03_av"]), float(cfg.get("h03_av_err", 0.5))
        denom = math.hypot(ae, sys_err)
        sig = abs(a - sys_av) / denom if denom > 0 else float("inf")
        sup_s = float(cfg.get("g4_t6_av_sigma_support", 2.0))
        bg_s = float(cfg.get("g4_t6_av_sigma_background", 3.0))
        v_assoc = "supports" if sig < sup_s else "neutral"
        v_bg = "supports" if sig > bg_s else "neutral"
        matrix["T6"] = {h: _cell(v_assoc if h in _BOUND_HYP else v_bg if h == "m_star_background"
                                 else "neutral", round(sig, 2), "g3.a_v_spectral vs system") for h in HYPOTHESES}

    # T5 RV: only meaningful if G2 detected lines (none here)
    g2_rows = list(csv.DictReader(open(paths["g2_table"]))) if Path(paths["g2_table"]).exists() else []
    n_detected = sum(1 for r in g2_rows if r.get("status") == "detected")
    matrix["T5"] = {h: _cell("not_available", n_detected,
                             "g2 (no detected lines -> RV power null)") for h in HYPOTHESES}

    # T7 artifact: import H02 verdict + B3 detection reality
    h02_overall = str(h02.get("overall", "unknown"))
    real_source = pointlike and h02_overall in ("survives", "mixed")
    matrix["T7"] = {h: _cell(
        ("excludes" if h == "artifact" else "supports") if real_source else "neutral",
        f"h02={h02_overall}, snr={snr}", "stage_h02_qc.overall + B3 snr") for h in HYPOTHESES}

    # T8 emission lines: absence is weakly informative (spec §3 asymmetry)
    matrix["T8"] = {h: _cell(
        "disfavors" if (n_detected == 0 and h == "planet_forming") else "neutral",
        n_detected, "g2 (accretion lines)") for h in HYPOTHESES}

    # T9 background probability (Besancon density from config; provisional)
    density = cfg.get("g4_background_density_per_arcsec2")
    search_r = float(cfg.get("g4_search_radius_arcsec", 0.5))
    src9 = str(cfg.get("g4_background_density_source", "config (Besancon, provisional)"))
    if density is not None:
        area = math.pi * search_r ** 2
        p_bg = background_probability(float(density), area)
        low_bg = p_bg < float(cfg.get("g4_thresholds", {}).get("probable_bg_prob_max", 0.10))
        matrix["T9"] = {h: _cell(
            ("disfavors" if h in ("m_star_background", "contaminant") else "neutral") if low_bg else
            ("supports" if h in ("m_star_background", "contaminant") else "neutral"),
            p_bg, src9) for h in HYPOTHESES}
    else:
        p_bg = None
        matrix["T9"] = {h: _cell("not_available", None, "no background density in config") for h in HYPOTHESES}
    return matrix, p_bg


def compute_stage_g4(cfg, paths):
    thresholds = dict(cfg.get("g4_thresholds", {
        "secure_bg_prob_max": 0.01, "probable_bg_prob_max": 0.10, "secure_min_supports": 3}))
    weights = cfg.get("g4_verdict_weights")  # None -> classify defaults
    matrix, p_bg = build_matrix(cfg, paths)
    _ll, ranking = combine_evidence(matrix, groups=CORRELATED_GROUPS, weights=weights)
    for row in ranking:
        row["dominant_tests"] = dominant_tests(matrix, row["hypothesis"], weights=weights)
    p_bg_val = p_bg if p_bg is not None else 1.0
    final = classify(matrix, background_prob=p_bg_val, thresholds=thresholds,
                     groups=CORRELATED_GROUPS, weights=weights)

    # D13 combined class + ambiguity quantification from the real G3 QC.
    g3 = _read(paths["stage_g3_qc_json"]) or {}
    ambiguity = _ambiguity_quantification(cfg, paths, g3)
    substellar = {"planet_forming", "substellar_companion", "brown_dwarf"}
    margin = float(cfg.get("g4_combined_class_margin", 0.5))
    tied_at_top = [r["hypothesis"] for r in ranking if abs(r["log_l_rel"]) < margin]
    combined_label = None
    if len(ranking) >= 2:
        if (abs(ranking[0]["log_l_rel"] - ranking[1]["log_l_rel"]) < margin
                and ranking[0]["hypothesis"] in substellar and ranking[1]["hypothesis"] in substellar):
            combined_label = "companion_substellar_or_planetary"

    available = [t for t, per in matrix.items() if any(c["verdict"] != "not_available" for c in per.values())]
    unavailable = [t for t in matrix if t not in available]
    open_issues = []
    if unavailable:
        why = ("G3 atmospheric inference systematics-limited (C3 continuum); T5 no detected lines"
               if g3.get("systematics_limited", {}).get("accepted")
               else "G3 real fits deferred / no detected lines")
        open_issues.append({"issue": f"Tests {unavailable} not_available ({why}). Robustness reflects the reduced evidence.", "priority": "major"})
    if ambiguity.get("status") != "computed":
        open_issues.append({"issue": f"ambiguity_quantification not_available: {ambiguity.get('reason')}. "
                                     "Mass-based planet/BD/star resolution requires a trustworthy mass.",
                            "priority": "major"})
    if p_bg is None:
        open_issues.append({"issue": "T9 background density absent from config; P(background) not computed.", "priority": "major"})
    else:
        open_issues.append({"issue": f"T9 background density is provisional (config Besancon placeholder); P(bg)={p_bg:.2g}.", "priority": "major"})

    qc = {
        "stage": "g4_classification", "run_id": str(cfg["run_id"]), "provisional": True,
        "hypotheses": list(HYPOTHESES),
        "tests_available": available, "tests_unavailable": unavailable,
        "correlated_groups": CORRELATED_GROUPS,
        "combined_ranking": ranking,
        "background_probability": {"raw": p_bg, "source": str(cfg.get("g4_background_density_source", "config (provisional)"))},
        "leave_one_out_stable": final["leave_one_out_stable"],
        "final_class": {"label": combined_label or final["label"], "leader": final["label"],
                        "combined_class": combined_label, "tied_at_top": tied_at_top,
                        "robustness": final["robustness"],
                        "n_independent_supports": final["n_independent_supports"],
                        "exclusion_against_leader": final["exclusion_against_leader"]},
        "ambiguity_quantification": ambiguity,
        "frozen_thresholds": thresholds,
        "frozen_thresholds_hash": frozen_thresholds_hash(thresholds),
        "open_issues": open_issues,
    }
    return matrix, qc


def write_stage_g4(matrix, qc, paths):
    paths["paths"].ensure_base_dirs()
    with paths["matrix_csv"].open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test"] + list(HYPOTHESES))
        for test in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"):
            if test not in matrix:
                continue
            cells = []
            for h in HYPOTHESES:
                c = matrix[test][h]
                cells.append(f"{c['verdict']}:{c['value']}@{c['source']}")
            w.writerow([test] + cells)
    paths["qc_json"].write_text(json.dumps(qc, indent=1, default=str))
    return {"matrix": paths["matrix_csv"], "qc_json": paths["qc_json"]}


def run_stage_g4(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    rc = load_run_config(run_id, project_root=project_root, allow_run_id_mismatch=allow_run_id_mismatch)
    cfg = dict(rc.config)
    if overrides:
        cfg.update(overrides)
    cfg["run_id"] = rc.run_id
    cfg["project_root"] = str(rc.paths.project_root)
    paths = stage_g4_paths(cfg["run_id"], project_root=cfg["project_root"])
    matrix, qc = compute_stage_g4(cfg, paths)
    written = write_stage_g4(matrix, qc, paths)
    return {"config": cfg, "paths": paths, "qc": qc, "written": written}


__all__ = ["build_matrix", "compute_stage_g4", "run_stage_g4", "stage_g4_paths"]
