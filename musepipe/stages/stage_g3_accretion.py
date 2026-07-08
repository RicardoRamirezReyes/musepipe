"""Stage G3 (accretion slice): multiline L_acc + Mdot from G2 measurements.

Library-agnostic part of G3 (spec §3.5): consumes the G2 line table and the
config L_acc–L_line relations (with citation) + the config M/R; the template /
atmosphere / evolutionary-track fits (§3.2–§3.4) are DEFERRED (`pending_libraries`,
gated by the §8.1 human checkpoint) and appear as not_constrained rows. Every
row carries a dependency label (§1.1).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..models import validate_label
from ..models.accretion import combine_accretion, line_lacc, mdot_mc
from ..models.extinction import CCMExtinction
from ..paths import RunPaths

TABLE_FIELDS = [
    "property", "value", "err_stat_lo", "err_stat_hi", "err_sys", "unit", "label",
    "data_used", "method", "assumptions", "calibrations_citations", "validity_range",
    "limitations", "depends_on", "mc_seed",
]


def stage_g3_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    p = RunPaths.from_project_root(run_id, root)
    return {
        "paths": p,
        "g2_table": p.table_dir / "g2_line_measurements.csv",
        "stage_h03_qc_json": p.stage_dir / "stage_h03_qc.json",
        "table_csv": p.table_dir / "g3_physical_properties.csv",
        "qc_json": p.stage_dir / "stage_g3_qc.json",
    }


def _read_g2(path):
    rows = list(csv.DictReader(open(path)))
    def f(x):
        try: return float(x)
        except: return float("nan")
    return rows, f


def compute_stage_g3_accretion(cfg, paths):
    rows_g2, f = _read_g2(paths["g2_table"])
    distance = float(cfg["h03_distance_pc"])
    av = float(cfg["h03_av"]); av_err = float(cfg.get("h03_av_err", 0.0))
    rv = float(cfg.get("h03_rv_extinction", 3.1))
    ext = CCMExtinction(rv=rv, citation=cfg.get("h03_extinction_law_citation", "Cardelli+1989"))
    flux_unit = float(cfg.get("h03_flux_unit_cgs", 1.0))
    mass = float(cfg["h03_companion_mass_msun"]); mass_err = float(cfg.get("h03_companion_mass_err_msun", 0.1 * mass))
    radius = float(cfg["h03_companion_radius_rsun"]); radius_err = float(cfg.get("h03_companion_radius_err_rsun", 0.15 * radius))
    seed = int(cfg.get("g3_seed", 0))

    # relations per line: config dict {line_name: {a,b,scatter_dex,citation,validity_range}};
    # default provides Halpha (Alcala+2017) from the H03 config keys.
    relations = dict(cfg.get("g3_lacc_relations", {}))
    relations.setdefault("Halpha", {
        "a": float(cfg.get("h03_lacc_lha_a", 1.13)), "b": float(cfg.get("h03_lacc_lha_b", 1.74)),
        "scatter_dex": float(cfg.get("h03_relation_scatter_dex", 0.30)),
        "citation": cfg.get("h03_lacc_lha_citation", "Alcala+2017"),
        "validity_range": cfg.get("h03_lacc_validity_range", "L_line calibration range"),
    })

    per_line, open_issues = [], []
    for r in rows_g2:
        name = r["name"]
        rel = relations.get(name)
        if rel is None:
            continue
        status = r["status"]
        # use the 5-sigma upper-limit flux for limits, the direct flux for detections
        flux_val = f(r["flux_upper_limit_5sigma"]) if status in ("upper_limit", "marginal") else f(r["flux_direct"])
        if not np.isfinite(flux_val) or flux_val <= 0:
            continue
        a_lam = float(ext.a_lambda_over_av(f(r["rest_A"])))
        res = line_lacc(flux_val, distance_pc=distance, av=av, a_lambda_over_av=a_lam,
                        relation=rel, flux_unit_cgs=flux_unit)
        res.update({"name": name, "status": status, "l_acc_err_lsun": np.nan})
        per_line.append(res)

    combined = combine_accretion(per_line)
    rows = []

    def add(prop, value, label, *, unit="", data_used="", method="", assumptions="",
            cite="", validity="", limitations="", depends="", lo="", hi="", sys="", seed_v=""):
        assert validate_label(prop, label), f"invalid label {label} for {prop}"
        rows.append({"property": prop, "value": value, "err_stat_lo": lo, "err_stat_hi": hi,
                     "err_sys": sys, "unit": unit, "label": label, "data_used": data_used,
                     "method": method, "assumptions": assumptions, "calibrations_citations": cite,
                     "validity_range": validity, "limitations": limitations, "depends_on": depends,
                     "mc_seed": seed_v})

    for p in per_line:
        add(f"l_acc_{p['name']}", p["l_acc_lsun"], "upper_limit" if p["status"] in ("upper_limit", "marginal") else "empirical_inference",
            unit="Lsun", data_used=f"G2 {p['name']}", method="Lacc-Lline relation",
            cite=p["citation"], limitations=f"scatter {p['scatter_dex']} dex", depends="[lacc_relation, extinction, distance]")

    if combined.get("l_acc_lsun") is not None:
        add("l_acc_combined", combined["l_acc_lsun"],
            "upper_limit" if combined["kind"] == "upper_limit" else "empirical_inference",
            unit="Lsun", data_used="G2 multiline", method=combined.get("rule", "compatibility_combined"),
            cite="Alcala+2017", limitations=f"from {combined.get('from_line', 'lines')}",
            depends="[lacc_relation, extinction, distance]")
        md = mdot_mc(combined["l_acc_lsun"], mass, mass_err, radius, radius_err,
                     float(cfg.get("h03_relation_scatter_dex", 0.30)), n_mc=int(cfg.get("g3_n_mc", 2000)), seed=seed)
        add("mdot", md["p50"], "empirical_inference", unit="Msun/yr",
            data_used="l_acc_combined + config M,R", method="Mdot=1.25 Lacc R/GM (MC)",
            assumptions="M,R from config (Bowler+2017 hot-start; deferred G3 tracks)",
            cite="Alcala+2017; Bowler+2017", depends="[atmospheric_model(deferred), evolutionary_model(deferred), lacc_relation]",
            lo=md["p16"], hi=md["p84"], seed_v=seed)

    # deferred physical properties (pending external libraries, §8.1 checkpoint)
    for prop, label in (("spectral_type", "not_constrained"), ("teff", "not_constrained"),
                        ("a_v_spectral", "not_constrained"), ("radius", "not_constrained"),
                        ("l_bol", "not_constrained"), ("mass", "not_constrained"), ("age", "not_constrained")):
        add(prop, "", label, limitations="pending_libraries (BT-Settl / BHAC15+ATMO2020 / Luhman-Bonnefoy; §8.1 checkpoint)",
            method="deferred")
    open_issues.append({"issue": "Template/atmosphere/track fits deferred (pending_libraries; §8.1 checkpoint). Only the multiline accretion slice is computed.", "priority": "major"})

    # V5: Halpha consistency with H03
    h03 = json.loads(paths["stage_h03_qc_json"].read_text()) if paths["stage_h03_qc_json"].exists() else {}
    v5 = {"status": "not_checked"}
    ha = next((p for p in per_line if p["name"] == "Halpha"), None)
    if ha is not None:
        v5 = {"status": "computed", "g3_l_acc_halpha_lsun": ha["l_acc_lsun"],
              "note": "compare against stage_h03 l_acc_lsun (same Alcala relation + inputs)"}

    qc = {
        "stage": "g3_accretion", "run_id": str(cfg["run_id"]), "provisional": True,
        "libraries": {
            "atmosphere": cfg.get("g3_atmosphere_family", "BT-Settl (deferred)"),
            "tracks": cfg.get("g3_tracks_families", ["BHAC15", "ATMO2020"]),
            "templates": cfg.get("g3_template_family", "Luhman/Bonnefoy (deferred)"),
            "accretion_relation": relations.get("Halpha", {}).get("citation"),
        },
        "n_lines_with_relation": len(per_line),
        "combined_accretion": combined,
        "mdot_p50_msun_yr": next((r["value"] for r in rows if r["property"] == "mdot"), None),
        "halpha_h03_consistency_v5": v5,
        "mc": {"seed": seed, "n": int(cfg.get("g3_n_mc", 2000))},
        "open_issues": open_issues,
    }
    return rows, qc


def write_stage_g3(rows, qc, paths):
    paths["paths"].ensure_base_dirs()
    with paths["table_csv"].open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TABLE_FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    paths["qc_json"].write_text(json.dumps(qc, indent=1, default=str))
    return {"table": paths["table_csv"], "qc_json": paths["qc_json"]}


def run_stage_g3_accretion(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    rc = load_run_config(run_id, project_root=project_root, allow_run_id_mismatch=allow_run_id_mismatch)
    cfg = dict(rc.config)
    if overrides:
        cfg.update(overrides)
    cfg["run_id"] = rc.run_id
    cfg["project_root"] = str(rc.paths.project_root)
    paths = stage_g3_paths(cfg["run_id"], project_root=cfg["project_root"])
    rows, qc = compute_stage_g3_accretion(cfg, paths)
    written = write_stage_g3(rows, qc, paths)
    return {"config": cfg, "paths": paths, "qc": qc, "written": written}


__all__ = ["compute_stage_g3_accretion", "run_stage_g3_accretion", "stage_g3_paths", "TABLE_FIELDS"]
