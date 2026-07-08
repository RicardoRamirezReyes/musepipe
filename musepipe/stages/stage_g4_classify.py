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

from ..classify import (
    HYPOTHESES,
    background_probability,
    classify,
    combine_evidence,
    dominant_tests,
    frozen_thresholds_hash,
)
from ..config import load_run_config
from ..paths import RunPaths

CORRELATED_GROUPS = [("T3", "T4", "T6")]  # share the spectrum + A_V fit (spec §4.1)


def stage_g4_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    p = RunPaths.from_project_root(run_id, root)
    return {
        "paths": p,
        "stage01c_qc_json": p.stage_dir / "stage01c_qc.json",
        "stage_h02_qc_json": p.stage_dir / "stage_h02_qc.json",
        "g2_table": p.table_dir / "g2_line_measurements.csv",
        "stage_g3_qc_json": p.stage_dir / "stage_g3_qc.json",
        "matrix_csv": p.table_dir / "g4_evidence_matrix.csv",
        "qc_json": p.stage_dir / "stage_g4_classification.json",
    }


def _read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _cell(verdict, value, source):
    return {"verdict": verdict, "value": value, "source": source}


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

    # T3/T4/T6 deferred (G3 real fits pending libraries)
    g3 = _read(paths["stage_g3_qc_json"]) or {}
    g3_deferred = any("pending_libraries" in str(i.get("issue", "")) for i in (g3.get("open_issues") or []))
    for t, src in (("T3", "g3.spectral_type"), ("T4", "g3.radius/l_bol"), ("T6", "g3.a_v")):
        matrix[t] = {h: _cell("not_available", None, src + " (deferred)" if g3_deferred else src) for h in HYPOTHESES}

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

    available = [t for t, per in matrix.items() if any(c["verdict"] != "not_available" for c in per.values())]
    unavailable = [t for t in matrix if t not in available]
    open_issues = []
    if unavailable:
        open_issues.append({"issue": f"Tests {unavailable} not_available (G3 real fits deferred / no detected lines). Robustness reflects the reduced evidence.", "priority": "major"})
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
        "final_class": {"label": final["label"], "robustness": final["robustness"],
                        "n_independent_supports": final["n_independent_supports"],
                        "exclusion_against_leader": final["exclusion_against_leader"]},
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
