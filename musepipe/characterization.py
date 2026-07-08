"""Phase G5 final synthesis: consolidate G0–G4 into report/characterization/.

Extends the F1 machinery (spec G5: reuses report.py's deterministic writers and
hash helpers; does NOT modify its interfaces). Read-only aggregator — the only
arithmetic allowed is aggregation/formatting and consistency verification
(spec §1.1). Nothing is filtered: not_constrained / upper_limit rows appear
as-is.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .paths import RunPaths
from .report import report_tree_hash, write_csv_deterministic, write_json_deterministic


def characterization_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    p = RunPaths.from_project_root(run_id, root)
    stages, tables = p.stage_dir, p.table_dir
    out = p.run_dir / "report" / "characterization"
    return {
        "paths": p, "out_dir": out, "extra_dir": out / "extra",
        "qc": {"g0": stages / "stage_g0_qc.json", "g1": stages / "stage_g1_qc.json",
               "g2": stages / "stage_g2_qc.json", "g3": stages / "stage_g3_qc.json",
               "g4": stages / "stage_g4_classification.json"},
        "g2_table": tables / "g2_line_measurements.csv",
        "g3_table": tables / "g3_physical_properties.csv",
        "g4_matrix": tables / "g4_evidence_matrix.csv",
        "h03_qc": stages / "stage_h03_qc.json", "h01_qc": stages / "stage_h01_qc.json",
        "config": p.run_dir / "config" / "config.json",
    }


def _read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _sha256(path):
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def check_traceability(cp):
    """Walk cube→config→...→value; a link is broken when a required field is absent.
    Fields are contract outputs of prior phases; missing → issue against that phase."""
    chains, broken = [], []
    cfg = (_read(cp["config"]) or {}).get("config", {})
    required = {
        "distance": cfg.get("h03_distance_source"),
        "a_v": cfg.get("h03_av_source"),
        "lacc_relation": cfg.get("h03_lacc_lha_citation"),
        "extinction_law": cfg.get("h03_extinction_law_citation"),
        "flux_unit": cfg.get("h03_flux_unit_source"),
    }
    for name, val in required.items():
        ok = bool(val)
        chains.append({"link": name, "citation_present": ok, "value": val})
        if not ok:
            broken.append({"link": name, "reason": "missing citation/source in config"})
    for phase, path in cp["qc"].items():
        present = Path(path).exists()
        chains.append({"link": f"qc_{phase}", "present": present, "sha256": _sha256(path)})
        if not present:
            broken.append({"link": f"qc_{phase}", "reason": "phase QC missing"})
    return {"complete": len(broken) == 0, "chains": chains, "broken_chains": broken}


def consistency_checks(cp):
    """V2: repeated numbers across products must agree (Halpha category)."""
    h01 = (_read(cp["h01_qc"]) or {}).get("verdict", {}).get("verdict")
    g2 = _read(cp["qc"]["g2"]) or {}
    g2_ha = (g2.get("halpha_reconciliation_v3") or {})
    issues = []
    checks = []
    if h01 and g2_ha.get("h01_verdict"):
        consistent = bool(g2_ha.get("consistent"))
        checks.append({"check": "halpha_category_g2_vs_h01", "consistent": consistent,
                       "g2_status": g2_ha.get("g2_halpha_status"), "h01": h01})
        if not consistent:
            issues.append({"issue": "Halpha category disagrees between G2 and H01/H03.", "priority": "blocking"})
    # G4 leader vs G3 (SpT deferred → not comparable yet)
    g4 = _read(cp["qc"]["g4"]) or {}
    checks.append({"check": "classification_robustness", "value": (g4.get("final_class") or {}).get("robustness")})
    return {"checks": checks, "issues": issues}


def _copy_table(src, dst):
    src = Path(src)
    if not src.exists():
        return 0
    rows = list(csv.DictReader(src.open()))
    if rows:
        write_csv_deterministic(dst, rows, list(rows[0].keys()))
    else:
        Path(dst).write_text(Path(src).read_text())
    return len(rows)


def _adopted_parameters(cfg):
    def row(param, value, err, cite, phase):
        return {"parameter": param, "value": value, "error": err, "citation": cite, "consumed_by": phase}
    return [
        row("distance_pc", cfg.get("h03_distance_pc"), cfg.get("h03_distance_err_pc"), cfg.get("h03_distance_source"), "E3/G3"),
        row("a_v", cfg.get("h03_av"), cfg.get("h03_av_err"), cfg.get("h03_av_source"), "E3/G3/G4"),
        row("rv_sys_kms", cfg.get("h03_rv_sys_kms"), None, cfg.get("h03_rv_source"), "E1/G2"),
        row("extinction_law", cfg.get("h03_extinction_law"), cfg.get("h03_rv_extinction"), cfg.get("h03_extinction_law_citation"), "E3/G3"),
        row("lacc_lha_relation", f"a={cfg.get('h03_lacc_lha_a')},b={cfg.get('h03_lacc_lha_b')}", cfg.get("h03_relation_scatter_dex"), cfg.get("h03_lacc_lha_citation"), "E3/G3"),
        row("companion_mass_msun", cfg.get("h03_companion_mass_msun"), None, cfg.get("h03_mass_source"), "E3/G3"),
        row("companion_radius_rsun", cfg.get("h03_companion_radius_rsun"), None, cfg.get("h03_radius_source"), "E3/G3"),
        row("atmosphere_library", cfg.get("g3_atmosphere_family"), None, cfg.get("g3_atmosphere_citation"), "G3(deferred)"),
        row("evolutionary_tracks", ",".join(cfg.get("g3_tracks_families", [])), None, ";".join(cfg.get("g3_tracks_citations", [])), "G3(deferred)"),
        row("background_density", cfg.get("g4_background_density_per_arcsec2"), None, cfg.get("g4_background_density_source"), "G4"),
    ]


def _uncertainty_budget(cp, cfg):
    h03 = _read(cp["h03_qc"]) or {}
    g1 = _read(cp["qc"]["g1"]) or {}
    rows = []
    lim = next((r for r in (h03.get("limits") or []) if r.get("row_kind") == "combined_final"), {})
    rows.append({"result": "Mdot_upper_limit", "stat_term": "control tail (7 controls)",
                 "sys_term": f"relation scatter {cfg.get('h03_relation_scatter_dex')} dex + A_V {cfg.get('h03_av_err')} + flux-cal (M3 unavailable)",
                 "dominant": "L_acc-L_Halpha relation scatter"})
    rows.append({"result": "throughput_psffit", "stat_term": f"{lim.get('throughput_err','?')}",
                 "sys_term": "PSF-perturbation no-op (unmeasured); edge-position", "dominant": "position/PSF"})
    rows.append({"result": "spectral_covariance", "stat_term": f"corr_len {(g1.get('covariance') or {}).get('corr_length_channels_median')} ch",
                 "sys_term": f"n_eff/n {(g1.get('covariance') or {}).get('n_eff_over_n_median')}", "dominant": "resampling (box3 ~6.5x)"})
    return rows


def build_characterization(run_id, project_root=None):
    cp = characterization_paths(run_id, project_root)
    cfg_full = _read(cp["config"]) or {}
    cfg = cfg_full.get("config", {})
    cp["out_dir"].mkdir(parents=True, exist_ok=True)
    cp["extra_dir"].mkdir(parents=True, exist_ok=True)

    trace = check_traceability(cp)
    consist = consistency_checks(cp)

    # tables §4
    n_lines = _copy_table(cp["g2_table"], cp["out_dir"] / "final_line_table.csv")
    n_props = _copy_table(cp["g3_table"], cp["out_dir"] / "final_physical_properties.csv")
    write_csv_deterministic(cp["out_dir"] / "adopted_parameters.csv", _adopted_parameters(cfg),
                            ["parameter", "value", "error", "citation", "consumed_by"])
    g4 = _read(cp["qc"]["g4"]) or {}
    class_rows = [{"rank": i + 1, "hypothesis": r["hypothesis"], "log_l_rel": r["log_l_rel"],
                   "dominant_tests": "|".join(r.get("dominant_tests", []))}
                  for i, r in enumerate(g4.get("combined_ranking", []))]
    write_csv_deterministic(cp["out_dir"] / "final_classification.csv", class_rows,
                            ["rank", "hypothesis", "log_l_rel", "dominant_tests"])
    write_csv_deterministic(cp["out_dir"] / "uncertainty_budget.csv", _uncertainty_budget(cp, cfg),
                            ["result", "stat_term", "sys_term", "dominant"])
    # spectra index
    spec_rows = []
    for name in ("spec_final_object.fits", "spec_calibrated_psffit_object.fits"):
        f = cp["paths"].stage_dir / name
        if f.exists():
            spec_rows.append({"spectrum": name, "sha256": _sha256(f), "method": "psffit",
                              "covariance": "g1_channel_covariance.npz"})
    write_csv_deterministic(cp["out_dir"] / "final_spectra_index.csv", spec_rows,
                            ["spectrum", "sha256", "method", "covariance"])

    tables = ["final_line_table.csv", "final_physical_properties.csv", "adopted_parameters.csv",
              "final_classification.csv", "uncertainty_budget.csv", "final_spectra_index.csv"]
    figures = _index_figures(cp["paths"])

    open_issues = list(consist["issues"])
    if trace["broken_chains"]:
        open_issues.append({"issue": f"Traceability: {len(trace['broken_chains'])} broken chain(s) — {[b['link'] for b in trace['broken_chains']]}.", "priority": "major"})
    # inherit blocking issues awareness from phases (provisional)
    open_issues.append({"issue": "PROVISIONAL package on the ADP cube; A-block open (alignment, M3 flux, M5 STAT). G3 real fits deferred → SpT/Teff/mass not constrained → G4 ambiguous.", "priority": "blocking"})

    summary = {
        "stage": "g5_final_synthesis", "run_id": str(run_id), "provisional": True,
        "inputs": {"phase_qc_hashes": {k: _sha256(v) for k, v in cp["qc"].items()}},
        "traceability": {"complete": trace["complete"], "broken_chains": trace["broken_chains"], "chains": trace["chains"]},
        "consistency": consist["checks"],
        "n_lines": n_lines, "n_physical_properties": n_props,
        "final_class": g4.get("final_class"),
        "tables_generated": tables, "figures_generated": [f["name"] for f in figures],
        "f1_compatibility": {"run_summary_extended": True, "f1_tests_green": None},
        "open_issues": open_issues,
    }
    write_json_deterministic(cp["out_dir"] / "characterization_summary.json", summary)
    _write_assumptions(cp["out_dir"] / "assumptions_and_limitations.md", cfg, trace, g4)
    _render_markdown(cp["out_dir"] / "characterization.md", summary, figures)
    # determinism hash over the package (excludes nothing volatile but timestamps)
    summary["determinism_hash"] = report_tree_hash(cp["out_dir"])
    write_json_deterministic(cp["out_dir"] / "characterization_summary.json", summary)
    return summary


def _index_figures(paths):
    candidates = [
        ("G5-1_final_spectrum", paths.plot_dir / ".." / "report" / "figures" / "fig04_final_spectrum.png"),
        ("G5-2_line_windows", paths.plot_stage_dir("stage_g2") / "g2_v4_line_windows.png"),
        ("G5-5_rv_by_line", paths.plot_stage_dir("stage_g2") / "g2_rv_by_line.png"),
        ("G5-6_throughput", paths.plot_stage_dir("stage_g1") / "g1_v2_recovered_vs_injected.png"),
        ("G5-6b_covariance", paths.plot_stage_dir("stage_g1") / "g1_v3_covariance.png"),
        ("G5-6c_bias_tornado", paths.plot_stage_dir("stage_g1") / "g1_v4_tornado_bias.png"),
    ]
    out = []
    for name, path in candidates:
        p = Path(path)
        out.append({"name": name, "path": str(p), "present": p.exists()})
    out.append({"name": "G5-3_template_fit", "path": "deferred(pending_libraries)", "present": False})
    out.append({"name": "G5-4_HRD_tracks", "path": "deferred(pending_libraries)", "present": False})
    out.append({"name": "G5-7_evidence_heatmap", "path": "deferred(figure)", "present": False})
    return out


def _write_assumptions(path, cfg, trace, g4):
    lines = ["# Assumptions and limitations (G5)\n",
             "Numbered active assumptions, the phase that introduced them, and estimated effect.\n"]
    items = [
        ("STAT unreliable (A4/M5 red ~4-6x resampling covariance) → empirical noise used", "A4/G1", "error budget"),
        ("Spectral channels mildly correlated (G1 n_eff/n~0.69) → per-channel FAPs slightly optimistic", "G1", "significance"),
        ("Absolute flux uncalibrated (D2 scale=1, M3 unavailable); flux unit = MUSE 1e-20 native", "D2/E3", "L/Mdot absolute scale"),
        ("Primary variability ~10% → absolute-calibration systematic", "X11", "all absolute fluxes"),
        ("Region age adopted with error (evolutionary mass depends on it)", "G3", "mass/age"),
        ("LSF from config estimate (A4 M2 unavailable)", "G2", "FWHM/EW"),
        ("G3 template/atmosphere/track fits deferred (pending external libraries)", "G3", "SpT/Teff/mass/radius not constrained"),
        ("Single astrometric epoch here (Bowler CPM adopted from config)", "G4", "bound-vs-background power"),
        ("Background density provisional (Besancon placeholder)", "G4", "P(contaminant)"),
        ("Whole B-F-G chain provisional on the ESO ADP; A-block open", "A/G0", "paper-validity"),
    ]
    for i, (txt, phase, effect) in enumerate(items, 1):
        lines.append(f"{i}. {txt}  \n   _phase:_ {phase} · _effect:_ {effect}")
    Path(path).write_text("\n".join(lines) + "\n")


def _render_markdown(path, summary, figures):
    fc = summary.get("final_class") or {}
    lines = [
        "# Characterization summary (G5)\n",
        f"- Run: {summary['run_id']}  \n- Provisional: {summary['provisional']}",
        f"- Lines measured (G2): {summary['n_lines']}  \n- Physical properties (G3): {summary['n_physical_properties']}",
        f"- Classification: **{fc.get('label')}** (robustness **{fc.get('robustness')}**)",
        f"- Traceability complete: {summary['traceability']['complete']}  \n- Open issues: {len(summary['open_issues'])}\n",
        "## Tables\n" + "\n".join(f"- `{t}`" for t in summary["tables_generated"]),
        "\n## Figures\n" + "\n".join(f"- {f['name']}: {'ok' if f['present'] else f['path']}" for f in figures),
        "\n## Open issues\n" + "\n".join(f"- [{i['priority']}] {i['issue']}" for i in summary["open_issues"]),
        "\nSee `assumptions_and_limitations.md` for the full assumption list.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


__all__ = ["build_characterization", "characterization_paths", "check_traceability", "consistency_checks"]
