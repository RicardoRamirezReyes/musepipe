"""Stage G3 (derived slice): L_bol, R, mass/age via both track families + the
end-to-end MC (spec G3 §3.4, plan WP-G3R-9).

Consumes the atmosphere fit (Teff/A_V/Ω + the Δχ² 3-D grid) and the two track
families; produces the derived-property rows, the persisted mass posterior
(``g3_mass_posterior.npz``, per family + combined) and the HRD figure (V4).
Only WP-G3R-11 runs this on real data; ``compute_`` accepts injected inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..constants import MSUN_OVER_MJUP
from ..models import validate_label
from ..models.derived import plot_hrd, run_mc_chain
from ..models.manifest import library_root
from ..models.tracks import TrackGrid
from ..paths import RunPaths
from .stage_g3_accretion import TABLE_FIELDS


def stage_g3_derived_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    p = RunPaths.from_project_root(run_id, root)
    return {"paths": p,
            "atmo_npz": p.stage_dir / "g3_atmo_fit.npz",
            "atmo_qc": p.stage_dir / "g3_atmo_fit_qc.json",
            "rows_json": p.stage_dir / "g3_rows_derived.json",
            "mass_posterior": p.stage_dir / "g3_mass_posterior.npz",
            "hrd_plot": p.plot_dir / "g3_hrd_tracks.png"}


def _row(prop, value, label, **kw):
    assert validate_label(prop, label), f"invalid label {label} for {prop}"
    row = {k: "" for k in TABLE_FIELDS}
    row.update(property=prop, value=value, label=label)
    row.update({k: v for k, v in kw.items() if k in TABLE_FIELDS})
    return row


def _load_atmo_result(paths):
    with np.load(paths["atmo_npz"]) as z:
        res = {k: z[k] for k in ("dchi2_3d", "scales_3d", "teff_axis",
                                 "logg_axis", "av_axis")}
    qc = json.loads(paths["atmo_qc"].read_text()) if paths["atmo_qc"].exists() else {}
    res["interp_error"] = qc.get("interp_error", {"teff": 0.0, "logg": 0.0, "av": 0.0})
    res["teff_best"] = qc.get("best", {}).get("teff_best")
    res["av_best"] = qc.get("best", {}).get("av_best")
    return res


def _build_track_grids(cfg):
    root = library_root(cfg, project_root=cfg["project_root"])
    fams = list(cfg.get("g3_tracks_families", ["BHAC15", "ATMO2020"]))
    cites = list(cfg.get("g3_tracks_citations", []))
    files = {"BHAC15": "tracks_bhac15/bhac15_tracks.npz",
             "ATMO2020": "tracks_atmo2020/atmo2020_ceq_tracks.npz"}
    grids = {}
    for i, fam in enumerate(fams):
        cite = cites[i] if i < len(cites) else fam
        grids[fam] = TrackGrid(root / files[fam], family=fam, citation=cite,
                               version=cfg.get("g3_tracks_version"))
    return grids


def compute_stage_g3_derived(cfg, paths, *, atmo_result=None, track_grids=None):
    if atmo_result is None:
        atmo_result = _load_atmo_result(paths)
    if track_grids is None:
        track_grids = _build_track_grids(cfg)
    rng = np.random.default_rng(int(cfg.get("g3_seed", 0)))
    mc = run_mc_chain(cfg, atmo_result, track_grids, rng)

    pc = mc["percentiles"]
    tr = mc["tracks"]
    seed = int(cfg.get("g3_seed", 0))

    def stat(pcts):
        return {"value": pcts[1], "lo": pcts[1] - pcts[0], "hi": pcts[2] - pcts[1]}

    r = stat(pc["radius_rjup"])
    lb = stat(pc["l_bol_lsun"])
    mmsun = stat(tr["combined_mass_msun"])
    lg = stat(tr["combined_logg"])
    age = stat(pc["age_gyr"])

    rows = [
        _row("radius", r["value"], "atmospheric_model_dependent", unit="R_Jup",
             err_stat_lo=r["lo"], err_stat_hi=r["hi"], method="R = sqrt(Omega)*d (MC)",
             depends_on="[atmospheric_model, distance]", mc_seed=seed,
             limitations="assumes SVO BT-Settl surface flux normalization"),
        _row("l_bol", lb["value"], "atmospheric_model_dependent", unit="Lsun",
             err_stat_lo=lb["lo"], err_stat_hi=lb["hi"],
             method="L_bol = 4*pi*d^2*Omega_phys*sigma*Teff^4 (MC)",
             depends_on="[atmospheric_model, distance]", mc_seed=seed),
        _row("mass", mmsun["value"] * MSUN_OVER_MJUP, "evolutionary_model_dependent",
             unit="M_Jup", err_stat_lo=mmsun["lo"] * MSUN_OVER_MJUP,
             err_stat_hi=mmsun["hi"] * MSUN_OVER_MJUP,
             err_sys=tr["mass_err_sys"] * MSUN_OVER_MJUP,
             method="tracks lookup(L_bol, age) MC, both families combined",
             assumptions=f"age prior {cfg.get('g3_age_myr')} Myr (D12)",
             calibrations_citations="; ".join(cfg.get("g3_tracks_citations", [])),
             depends_on="[atmospheric_model, evolutionary_model, age_prior]", mc_seed=seed,
             limitations=f"between-family err_sys; value_msun={mmsun['value']:.4g}"),
        _row("age_used", age["value"] * 1000.0, "evolutionary_model_dependent",
             unit="Myr", err_stat_lo=age["lo"] * 1000.0, err_stat_hi=age["hi"] * 1000.0,
             method="adopted prior (not a G3 measurement)",
             assumptions="D12: Bowler et al. 2017, 6 +4/-2 Myr",
             calibrations_citations=cfg.get("g3_age_citation", ""),
             depends_on="[age_prior]", mc_seed=seed),
        _row("logg_evol", lg["value"], "evolutionary_model_dependent", unit="dex",
             err_stat_lo=lg["lo"], err_stat_hi=lg["hi"],
             method="tracks lookup MC (both families)",
             depends_on="[atmospheric_model, evolutionary_model, age_prior]", mc_seed=seed),
    ]
    return rows, mc


def write_stage_g3_derived(rows, mc, paths, track_grids=None, *, make_plot=True):
    paths["paths"].ensure_base_dirs()
    paths["rows_json"].write_text(json.dumps(rows, indent=1, default=str))
    by_fam = mc["mass_posterior_by_family"]
    np.savez(paths["mass_posterior"],
             combined=np.asarray(mc["mass_posterior_combined"], float),
             **{f"family_{k}": np.asarray(v, float) for k, v in by_fam.items()})
    out = {"rows_json": paths["rows_json"], "mass_posterior": paths["mass_posterior"]}
    if make_plot and track_grids is not None:
        out["hrd_plot"] = plot_hrd(paths["hrd_plot"], track_grids, mc,
                                   age_gyr=mc["age_samples"].mean() if mc["age_samples"].size else 0.006,
                                   age_err_gyr=(0.002, 0.004))
    return out


def run_stage_g3_derived(run_id=None, *, project_root=None, overrides=None,
                         allow_run_id_mismatch=False):
    rc = load_run_config(run_id, project_root=project_root,
                         allow_run_id_mismatch=allow_run_id_mismatch)
    cfg = dict(rc.config)
    if overrides:
        cfg.update(overrides)
    cfg["run_id"] = rc.run_id
    cfg["project_root"] = str(rc.paths.project_root)
    paths = stage_g3_derived_paths(cfg["run_id"], project_root=cfg["project_root"])
    track_grids = _build_track_grids(cfg)
    rows, mc = compute_stage_g3_derived(cfg, paths, track_grids=track_grids)
    written = write_stage_g3_derived(rows, mc, paths, track_grids=track_grids)
    return {"config": cfg, "paths": paths, "written": written}


__all__ = ["compute_stage_g3_derived", "run_stage_g3_derived",
           "stage_g3_derived_paths", "write_stage_g3_derived"]
