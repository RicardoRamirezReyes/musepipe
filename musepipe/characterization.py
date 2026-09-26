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
        "a4_qc": stages / "stage00q_qc.json",
        "f1_summary": p.run_dir / "report" / "run_summary.json",
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
    g4 = _read(cp["qc"]["g4"]) or {}
    checks.append({"check": "classification_robustness", "value": (g4.get("final_class") or {}).get("robustness")})
    # V3: an Mdot derived from an upper-limit L_acc is an upper limit too (G3 labelled it
    # `empirical_inference` for ROXs 42B b until 2026-09-26).
    g3 = {r.get("property"): r for r in _read_rows(cp["g3_table"])}
    lacc, mdot = g3.get("l_acc_combined"), g3.get("mdot")
    if lacc and mdot:
        coherent = not (lacc.get("label") == "upper_limit" and mdot.get("label") != "upper_limit")
        checks.append({"check": "mdot_label_follows_lacc", "consistent": coherent,
                       "l_acc_combined": lacc.get("label"), "mdot": mdot.get("label")})
        if not coherent:
            issues.append({"issue": f"G3 labels Mdot `{mdot.get('label')}` although L_acc_combined is an "
                                    "upper limit; the Mdot is an upper limit (re-run G3).",
                           "priority": "blocking"})
    return {"checks": checks, "issues": issues}


def _read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _canonical_limit(h03):
    """E3 row of the canonical method (older QCs: the `combined_final` row)."""
    limits = h03.get("limits") or []
    canon = h03.get("canonical_method")
    return (next((r for r in limits if canon and r.get("method") == canon), None)
            or next((r for r in limits if r.get("row_kind") == "combined_final"), None) or {})


def inherited_blocking_issues(cp):
    """Blocking issues open upstream: F1's gate (A1→E3) and the G0–G4 phase QCs.

    G5 does not judge them; it carries them so that the package does not look closed while
    they are open (spec §8, §9). The package is provisional exactly when this list is not empty.
    """
    out = []
    f1 = _read(cp["f1_summary"]) or {}
    for i in f1.get("open_issues") or []:
        if i.get("priority") == "blocking":
            out.append({"source": "F1", "stage": i.get("stage"), "issue": i.get("issue")})
    for phase, path in cp["qc"].items():
        for i in (_read(path) or {}).get("open_issues") or []:
            if isinstance(i, dict) and (i.get("priority") or i.get("severity")) == "blocking":
                out.append({"source": phase.upper(), "stage": phase.upper(),
                            "issue": i.get("issue") or i.get("message")})
    return out


def headline(cp):
    """The main results, read from the QCs that own them (nothing computed here)."""
    h01 = _read(cp["h01_qc"]) or {}
    h03 = _read(cp["h03_qc"]) or {}
    g4 = _read(cp["qc"]["g4"]) or {}
    g3 = {r.get("property"): r for r in _read_rows(cp["g3_table"])}
    canon = h03.get("canonical_method")
    lim = _canonical_limit(h03)
    n_null = (((h01.get("parametric_fap") or {}).get("by_method") or {}).get(canon) or {}).get("n_null")
    mdot = g3.get("mdot") or {}
    spt = g3.get("spectral_type") or {}
    return {
        "halpha_e1_verdict": (h01.get("verdict") or {}).get("verdict"),
        "e3_canonical_method": canon,
        "e3_mdot_99_msun_yr": _num(lim.get("mdot")),
        "n_control_positions": n_null,
        "g3_mdot_msun_yr": _num(mdot.get("value")), "g3_mdot_label": mdot.get("label"),
        "g3_spectral_type": spt.get("value") or None, "g3_spectral_type_label": spt.get("label"),
        "g4_class": (g4.get("final_class") or {}).get("label"),
        "g4_robustness": (g4.get("final_class") or {}).get("robustness"),
    }


def _fmt(value, spec=".2e"):
    return "n/a" if value is None else format(value, spec)


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
        row("atmosphere_library", cfg.get("g3_atmosphere_family"), None, cfg.get("g3_atmosphere_citation"), "G3"),
        row("evolutionary_tracks", ",".join(cfg.get("g3_tracks_families", [])), None, ";".join(cfg.get("g3_tracks_citations", [])), "G3"),
        row("background_density", cfg.get("g4_background_density_per_arcsec2"), None, cfg.get("g4_background_density_source"), "G4"),
    ]


def _uncertainty_budget(cp, cfg):
    h03 = _read(cp["h03_qc"]) or {}
    g1 = _read(cp["qc"]["g1"]) or {}
    head = headline(cp)
    m3 = ((_read(cp["a4_qc"]) or {}).get("m3_flux") or {}).get("status", "absent")
    rows = []
    lim = _canonical_limit(h03)
    rows.append({"result": "Mdot_upper_limit",
                 "stat_term": f"control tail ({head['n_control_positions'] or '?'} controls)",
                 "sys_term": f"relation scatter {cfg.get('h03_relation_scatter_dex')} dex + A_V {cfg.get('h03_av_err')} + flux-cal (A4/M3 {m3})",
                 "dominant": "L_acc-L_Halpha relation scatter"})
    rows.append({"result": f"throughput_{head['e3_canonical_method'] or 'canonical'}",
                 "stat_term": f"{lim.get('throughput_err', '?')}",
                 "sys_term": "injection model and control positions (E4)", "dominant": "position/PSF"})
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
    inherited = inherited_blocking_issues(cp)
    if inherited:
        stages = sorted({str(i["stage"]) for i in inherited})
        open_issues.append({"issue": f"{len(inherited)} blocking issue(s) open upstream ({', '.join(stages)}); "
                                     "listed in `inherited_blocking_issues`.", "priority": "blocking"})

    summary = {
        "stage": "g5_final_synthesis", "run_id": str(run_id), "provisional": bool(inherited),
        "headline": headline(cp),
        "inherited_blocking_issues": inherited,
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
    _write_assumptions(cp["out_dir"] / "assumptions_and_limitations.md", cfg, cp)
    _render_markdown(cp["out_dir"] / "characterization.md", summary, figures)
    _write_readme(cp["out_dir"] / "README.md", summary, figures)
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
        # G3 writes its figures at the top of plots/, not in a stage subdirectory
        ("G5-3_template_fit", paths.plot_dir / "g3_fit_spectrum.png"),
        ("G5-4_HRD_tracks", paths.plot_dir / "g3_hrd_tracks.png"),
    ]
    out = []
    for name, path in candidates:
        p = Path(path)
        out.append({"name": name, "path": str(p), "present": p.exists()})
    out.append({"name": "G5-7_evidence_heatmap", "path": "not produced (G4 writes no evidence heatmap)",
                "present": False})
    return out


def _write_assumptions(path, cfg, cp):
    lines = ["# Assumptions and limitations (G5)\n",
             "Numbered active assumptions, the phase that introduced them, and estimated effect.\n"]
    a4 = _read(cp["a4_qc"]) or {}
    g1 = _read(cp["qc"]["g1"]) or {}
    templates = (_read(cp["h01_qc"]) or {}).get("templates") or {}
    n_eff = _num((g1.get("covariance") or {}).get("n_eff_over_n_median"))
    lsf = _num(templates.get("lsf_fwhm_A"))
    not_constrained = sorted(r.get("property") for r in _read_rows(cp["g3_table"])
                             if r.get("label") == "not_constrained")
    items = [
        (f"Noise from control positions processed like the object (docs/noise_model.md); "
         f"A4/M5 STAT status {(a4.get('m5_stat') or {}).get('status', 'absent')}", "A4/G1", "error budget"),
        (f"Spectral channels correlated (G1 n_eff/n = {_fmt(n_eff, '.2f')}) → per-channel FAPs slightly "
         "optimistic", "G1", "significance"),
        (f"Absolute flux calibration: A4/M3 status {(a4.get('m3_flux') or {}).get('status', 'absent')}; "
         f"flux unit source: {cfg.get('h03_flux_unit_source', 'not declared')}", "D2/E3", "L/Mdot absolute scale"),
        ("Primary variability → absolute-calibration systematic (D2 sys_fluxcal)", "D2", "all absolute fluxes"),
        ("Region age adopted with error (evolutionary mass depends on it)", "G3", "mass/age"),
        (f"LSF {_fmt(lsf, '.3f')} Å from {templates.get('lsf_source', 'not declared')}", "E1/G2", "FWHM/EW"),
        (f"G3 properties not constrained: {', '.join(not_constrained) or 'none'}", "G3", "SpT/Teff/mass/radius"),
        ("Single astrometric epoch here (Bowler CPM adopted from config)", "G4", "bound-vs-background power"),
        (f"Background density source: {cfg.get('g4_background_density_source', 'not declared')}", "G4", "P(contaminant)"),
    ]
    for i, (txt, phase, effect) in enumerate(items, 1):
        lines.append(f"{i}. {txt}  \n   _phase:_ {phase} · _effect:_ {effect}")
    Path(path).write_text("\n".join(lines) + "\n")


TABLE_DOCS = {
    "final_line_table.csv": "Every catalog line (G2): status, flux (direct + fit), EW, centroid, "
                            "FWHM (obs + intrinsic), RV, z, 5σ upper limit with throughput applied.",
    "final_physical_properties.csv": "G3 properties (full schema): value, err_stat, err_sys, unit, "
                                     "label, assumptions, citations, validity_range, limitations, depends_on. "
                                     "not_constrained rows kept.",
    "adopted_parameters.csv": "Input parameters adopted (distance, A_V, RV, extinction law, L_acc–L_line "
                              "relation, mass, radius, libraries) with value, error, citation and consuming phase.",
    "final_classification.csv": "G4 hypothesis ranking with log-likelihood and dominant tests.",
    "uncertainty_budget.csv": "Consolidated statistical vs systematic budget per main result "
                              "(Mdot, throughput, spectral covariance).",
    "final_spectra_index.csv": "Published spectra (final + per method) with sha256, method, covariance.",
}


def _status_line(summary):
    inherited = summary.get("inherited_blocking_issues") or []
    if not inherited:
        return "> No blocking issue is open upstream (F1 gate and G0–G4).\n"
    stages = sorted({str(i["stage"]) for i in inherited})
    return (f"> **Provisional:** {len(inherited)} blocking issue(s) are open upstream "
            f"({', '.join(stages)}). They are listed at the end of this file.\n")


def _write_readme(path, summary, figures):
    fc = summary.get("final_class") or {}
    hl = summary.get("headline") or {}
    lines = [
        f"# Characterization package — {summary['run_id']}\n",
        "Machine-readable consolidation of phases **G0–G5** (extends the F1 `report/`). "
        "Generated deterministically by `musepipe.characterization.build_characterization`; "
        "**do not edit by hand** (the determinism test would catch it).\n",
        _status_line(summary),
        "## Headline results\n",
        f"- **Hα (E1):** `{hl.get('halpha_e1_verdict')}` · 99 % detection threshold on Ṁ (E3, "
        f"`{hl.get('e3_canonical_method')}`): {_fmt(hl.get('e3_mdot_99_msun_yr'))} M☉/yr · Ṁ (G3): "
        f"{_fmt(hl.get('g3_mdot_msun_yr'))} M☉/yr, label `{hl.get('g3_mdot_label')}`.",
        f"- **Spectral type (G3):** {hl.get('g3_spectral_type') or 'not constrained'}"
        f" (`{hl.get('g3_spectral_type_label')}`).",
        f"- **Source (G4):** class **{fc.get('label')}**, robustness **{fc.get('robustness')}**.",
        f"- **Lines measured (G2):** {summary['n_lines']}  ·  **physical properties (G3):** "
        f"{summary['n_physical_properties']}.",
        f"- **Traceability:** {'complete' if summary['traceability']['complete'] else 'BROKEN'}  ·  "
        f"**open issues:** {len(summary['open_issues'])}.\n",
        "## Tables (`.csv`)\n",
    ]
    for t in summary["tables_generated"]:
        lines.append(f"- **`{t}`** — {TABLE_DOCS.get(t, '')}")
    lines += [
        "\n## Other files\n",
        "- **`characterization_summary.json`** — machine-readable summary (phase QC hashes, "
        "traceability chains, consistency checks, determinism hash).",
        "- **`characterization.md`** — human-readable summary.",
        "- **`assumptions_and_limitations.md`** — numbered active assumptions, the phase that "
        "introduced each, and its estimated effect.",
        "- **`extra/`** — non-standard extras (kept separate from the fixed set).\n",
        "## Figures\n",
    ]
    for f in figures:
        state = "available" if f["present"] else f["path"]
        lines.append(f"- **{f['name']}** — {state}")
    lines += [
        "\n## Reproduce\n",
        "```bash",
        "conda activate MUSE",
        f"python scripts/build_characterization.py --run-id {summary['run_id']}",
        "```",
        "Requires the phase products/QCs (G0–G4) and the F1 `report/` package to exist. "
        "Two runs produce byte-identical files (determinism hash in the summary).\n",
        "## Blocking issues open upstream\n",
    ]
    inherited = summary.get("inherited_blocking_issues") or []
    lines += [f"- **{i['stage']}** ({i['source']}): {i['issue']}" for i in inherited] or ["- None."]
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
