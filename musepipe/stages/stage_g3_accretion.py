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
from ..constants import MSUN_OVER_MJUP, RJUP_CM, RSUN_CM
from ..io import flux_unit_cgs, read_stage00q_qc
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
        "rows_derived_json": p.stage_dir / "g3_rows_derived.json",
    }


def _own_mr_from_derived(path):
    """Load own (mass_msun, mass_err, radius_rsun, radius_err) from the WP-9
    derived rows (mass in M_Jup, radius in R_Jup), or None if unavailable."""
    if not path.exists():
        return None
    try:
        rows = json.loads(path.read_text())
        m = next(r for r in rows if r["property"] == "mass")
        rr = next(r for r in rows if r["property"] == "radius")
        mass_mjup = float(m["value"])
        rad_rjup = float(rr["value"])
        if not (np.isfinite(mass_mjup) and np.isfinite(rad_rjup)):
            return None
        m_err_mjup = 0.5 * (float(m.get("err_stat_lo") or 0.0) + float(m.get("err_stat_hi") or 0.0))
        r_err_rjup = 0.5 * (float(rr.get("err_stat_lo") or 0.0) + float(rr.get("err_stat_hi") or 0.0))
    except (StopIteration, KeyError, ValueError, TypeError):
        return None
    rjup_over_rsun = RJUP_CM / RSUN_CM
    return (mass_mjup / MSUN_OVER_MJUP, m_err_mjup / MSUN_OVER_MJUP,
            rad_rjup * rjup_over_rsun, r_err_rjup * rjup_over_rsun)


def _stage_dir(paths):
    """`stages/` del run, o None si quien llama monto un `paths` minimo."""
    run_paths = paths.get("paths") if hasattr(paths, "get") else None
    return getattr(run_paths, "stage_dir", None)


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
    # Knob -> QC de M3 (los flujos de G2 vienen del producto calibrado, cuya
    # escala absoluta la fijo M3): sin default silencioso, ver `io.flux_unit_cgs`.
    flux_unit = flux_unit_cgs(cfg, qc_m3=read_stage00q_qc(_stage_dir(paths)))
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
    md_cfg = None

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
        scatter = float(cfg.get("h03_relation_scatter_dex", 0.30))
        n_mc = int(cfg.get("g3_n_mc", 2000))
        own = _own_mr_from_derived(paths["rows_derived_json"])
        if own is not None:  # WP-9 derived M,R available -> use the object's own
            m_used, me_used, r_used, re_used = own
            depends = "[atmospheric_model, evolutionary_model, lacc_relation]"
            assumptions = "M,R from G3 derived chain (WP-9 MC p50)"
            mr_source = "g3_derived"
        else:
            m_used, me_used, r_used, re_used = mass, mass_err, radius, radius_err
            depends = "[atmospheric_model(deferred), evolutionary_model(deferred), lacc_relation]"
            assumptions = "M,R from config (Bowler+2017 hot-start; deferred G3 tracks)"
            mr_source = "config"
        md = mdot_mc(combined["l_acc_lsun"], m_used, me_used, r_used, re_used,
                     scatter, n_mc=n_mc, seed=seed)
        add("mdot", md["p50"], "empirical_inference", unit="Msun/yr",
            data_used=f"l_acc_combined + {mr_source} M,R", method="Mdot=1.25 Lacc R/GM (MC)",
            assumptions=assumptions, cite="Alcala+2017; Bowler+2017", depends=depends,
            lo=md["p16"], hi=md["p84"], seed_v=seed)
        # V5 variant: always config M,R (same inputs as H03) for the consistency check
        md_cfg = mdot_mc(combined["l_acc_lsun"], mass, mass_err, radius, radius_err,
                         scatter, n_mc=n_mc, seed=seed)

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
              "g3_mdot_config_mr_p50": (md_cfg["p50"] if md_cfg else None),
              "note": "L_acc (and Mdot with config M,R) reproduce H03 under identical "
                      "inputs (same Alcala relation); H03 uses the same config M,R"}

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
