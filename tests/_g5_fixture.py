"""Shared fixture: fabricate a minimal run with G0–G4 QCs/tables for G5 tests."""

import json
from pathlib import Path


def make_run(root: Path, run_id="G5RUN", *, consistent=True):
    rd = root / "runs" / run_id
    stages = rd / "stages"; tables = rd / "tables"; config = rd / "config"
    for d in (stages, tables, config):
        d.mkdir(parents=True, exist_ok=True)
    for phase in ("g0", "g1", "g2", "g3"):
        (stages / f"stage_{phase}_qc.json").write_text(json.dumps({"stage": phase}))
    (stages / "stage_g4_classification.json").write_text(json.dumps({
        "final_class": {"label": "substellar_companion", "robustness": "ambiguous"},
        "combined_ranking": [{"hypothesis": "substellar_companion", "log_l_rel": 0.0, "dominant_tests": ["T1", "T2"]}],
    }))
    h01_verdict = "non_detection"
    (stages / "stage_h01_qc.json").write_text(json.dumps({"verdict": {"verdict": h01_verdict}}))
    (stages / "stage_h03_qc.json").write_text(json.dumps({"limits": [{"row_kind": "combined_final", "throughput_err": 0.2}]}))
    g2_ha = "upper_limit" if consistent else "detected"
    (stages / "stage_g2_qc.json").write_text(json.dumps({
        "halpha_reconciliation_v3": {"g2_halpha_status": g2_ha, "h01_verdict": h01_verdict, "consistent": consistent}}))
    (tables / "g2_line_measurements.csv").write_text("name,status,label\nHalpha,upper_limit,upper_limit\n")
    (tables / "g3_physical_properties.csv").write_text(
        "property,value,label\nl_acc_Halpha,4.5e-6,upper_limit\nmass,,not_constrained\n")
    (tables / "g4_evidence_matrix.csv").write_text("test,substellar_companion\nT1,supports:1@src\n")
    (config / "config.json").write_text(json.dumps({"config": {
        "h03_distance_pc": 138.6, "h03_distance_source": "Gaia", "h03_av": 1.8, "h03_av_err": 0.5,
        "h03_av_source": "Rizzuto+2015", "h03_lacc_lha_citation": "Alcala+2017",
        "h03_extinction_law_citation": "Cardelli+1989", "h03_flux_unit_source": "MUSE 1e-20",
        "h03_lacc_lha_a": 1.13, "h03_lacc_lha_b": 1.74, "h03_relation_scatter_dex": 0.30,
        "h03_companion_mass_msun": 0.0167, "h03_companion_radius_rsun": 0.135,
        "g4_background_density_per_arcsec2": 8e-4}}))
    return run_id
