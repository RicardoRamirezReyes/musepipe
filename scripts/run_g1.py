#!/usr/bin/env python
"""WP-6 / G1 runner — extraction validation QC for an arbitrary run.

Reproduces the structure of ``runs/ROXs12b_B_adp/stages/stage_g1_qc.json`` (the
contract) using ONLY the public helpers in ``musepipe.covariance`` plus the
run's E4 injection-throughput table. No new covariance logic is invented here:
covariance comes from ``spectral_covariance_blocks`` / ``bootstrap_corr_length_error``
/ ``spatial_inflation_by_box`` and the per-method verdict from ``method_verdict``.

Assembly rules made explicit (they were embedded in the ad-hoc ADP G1 pass):
 - A method's throughput loss is ``throughput_real - 1`` (corrected in E3 by
   dividing by throughput); the PSF-model perturbation term is a no-op (0) because
   the injection PSF is flux-normalised — same placeholder as the ADP budget.
 - ``bias_bounded`` is False when the real-position throughput is below
   ``BIAS_BOUNDED_MIN_THROUGHPUT`` (0.4): such methods are insensitive at the
   companion edge and cannot be reliably throughput-corrected → ``rejected``
   (the documented ADP rationale for aperture / optimal_ls).

Usage: python scripts/run_g1.py --run-id ROXs12b_realigned
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musepipe.covariance import (
    bootstrap_corr_length_error,
    combine_budget_quadrature,
    method_verdict,
    spatial_inflation_by_box,
    spectral_covariance_blocks,
)

BIAS_BOUNDED_MIN_THROUGHPUT = 0.4
BIAS_THRESHOLD = 0.05
# sgf/lpm added in WP-H2 (docs/plan_integracion_halosub_julo2025.md). Methods
# with no rows in the E4 table (historical runs) get verdict "absent": never
# validated, never mistaken for "rejected".
METHODS = ("aperture", "optimal_ls", "optimal_psfsub", "psffit", "sgf", "lpm")


def _thr(rows, method, pos, *, snr=5.0, fwhm_scale=1.0, variant="nominal"):
    vals = [
        float(r["throughput"])
        for r in rows
        if r["method"] == method
        and r["position_label"] == pos
        and r.get("variant", "nominal") == variant
        and float(r["snr"]) == snr
        and float(r.get("psf_fwhm_scale", 1.0)) == fwhm_scale
        and r["throughput"] not in ("", "nan")
    ]
    return float(np.median(vals)) if vals else float("nan")


def _stat_err_frac(rows, method):
    snrs = [
        float(r["recovered_snr"])
        for r in rows
        if r["method"] == method and r["position_label"] == "real"
        and float(r["snr"]) == 5.0 and r.get("recovered_snr") not in ("", "nan")
    ]
    snr = float(np.median(snrs)) if snrs else np.nan
    return float(1.0 / snr) if np.isfinite(snr) and snr > 0 else 0.2


def covariance_npz_payload(cov):
    """Arrays written to ``g1_channel_covariance.npz``.

    Emits BOTH the legacy per-block arrays (``block_z0``/``block_z1``/
    ``n_eff_over_n``/...) AND the schema the G3 consumer expects: ``block_bounds``
    (Nx2 [z0, z1] per block) and ``n_eff_over_n_by_block`` (read by
    ``musepipe.models.observed.rebin_for_fit``), plus ``rho_by_block`` /
    ``corr_length_by_block`` (validated by ``scripts/check_g3_real_inputs.py``).
    Without ``block_bounds`` the legacy-only file makes G3 fail with
    ``KeyError: 'block_bounds'``.
    """
    blocks = cov["blocks"]
    z0 = np.array([b["z0"] for b in blocks])
    z1 = np.array([b["z1"] for b in blocks])
    corr_len = np.array([b["corr_length_channels"] for b in blocks])
    neff = np.array([b["n_eff_over_n"] for b in blocks])
    rho = np.array([b["rho"] for b in blocks])
    return {
        # legacy schema (kept for backward compatibility)
        "block_z0": z0,
        "block_z1": z1,
        "corr_length_channels": corr_len,
        "n_eff_over_n": neff,
        "rho_stack": rho,
        # schema consumed by G3 (observed.rebin_for_fit) + check_g3_real_inputs
        "block_bounds": np.column_stack([z0, z1]),
        "n_eff_over_n_by_block": neff,
        "rho_by_block": rho,
        "corr_length_by_block": corr_len,
    }


def build_g1(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    run_dir = root / "runs" / run_id
    stage_dir = run_dir / "stages"
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    # --- spectral covariance from the psffit controls (continuum-agnostic rows) ---
    z = np.load(stage_dir / "spec_psffit_controls.npz", allow_pickle=True)
    ctrl = np.asarray(z["control_spectra"], dtype=np.float64)
    # G3 (observed.rebin_for_fit / _block_upper_waves, WP-G3R-6) requires the
    # covariance block_bounds to index the GOOD-channel subspace: the last bound
    # must equal n_good, not the full 3681. Restrict the controls to the stage04b
    # good channels before blocking so block_bounds cover exactly n_good.
    bad_mask_file = stage_dir / "stage04b_bad_wavelength_mask.npy"
    if bad_mask_file.exists():
        good = ~np.load(bad_mask_file).astype(bool)
        ctrl = ctrl[:, good]
    cov = spectral_covariance_blocks(ctrl, block_size=200, max_lag=15)
    neff_blocks = [b["n_eff_over_n"] for b in cov["blocks"] if np.isfinite(b["n_eff_over_n"])]
    n_eff_med = float(np.median(neff_blocks)) if neff_blocks else float("nan")
    boot_err = bootstrap_corr_length_error(ctrl, block_size=200, max_lag=15, n_boot=200, seed=0)

    # --- spatial inflation from a source-free corner of a continuum slice ---
    with fits.open(stage_dir / "stage02_xcorr_cube_stack.fits", memmap=True) as h:
        cube = np.asarray(h["CUBES"].data, dtype=np.float64)
        wave = np.asarray(h["WAVELENGTH"].data, dtype=np.float64)
    if cube.ndim == 4:
        cube = cube[0]
    band = (6600.0, 6800.0)
    sel = np.isfinite(wave) & (wave >= band[0]) & (wave <= band[1])
    cont = np.nanmedian(cube[sel], axis=0)
    corner = cont[0:60, 0:60]  # away from primary (~85,85) and companion (~155,76)
    spatial = spatial_inflation_by_box(corner, boxes=(1, 2, 3, 4, 5), n_samples=400, seed=0)

    np.savez(stage_dir / "g1_channel_covariance.npz", **covariance_npz_payload(cov))

    # --- per-method bias budget + verdict from the E4 throughput table ---
    rows = list(csv.DictReader(open(tables_dir / "injection_throughput_by_method.csv")))
    budget_rows = []
    sens_rows = []
    verdicts = {}
    methods_present = {str(r["method"]) for r in rows}
    for m in METHODS:
        if m not in methods_present:
            verdicts[m] = "absent"
            continue
        t_real = _thr(rows, m, "real")
        bounded = bool(np.isfinite(t_real) and t_real >= BIAS_BOUNDED_MIN_THROUGHPUT)
        # throughput_loss: corrected in E3; unreliable (nan) when not bias_bounded.
        loss = float(t_real - 1.0) if bounded else float("nan")
        terms = [
            {"value_frac": loss, "err_frac": 0.0},
            {"value_frac": 0.0, "err_frac": 0.0},  # psf_model perturbation no-op
        ]
        total = combine_budget_quadrature(terms) if bounded else {"total_value_frac": float("nan"), "total_err_frac": 0.0}
        pos_thr = [_thr(rows, m, p) for p in ("real", "control1", "control2", "control3")]
        pos_thr = [x for x in pos_thr if np.isfinite(x)]
        sens = float(max(pos_thr) - min(pos_thr)) if len(pos_thr) > 1 else 0.0
        stat = _stat_err_frac(rows, m)
        v = method_verdict(
            total["total_value_frac"] if bounded else 0.0,
            sens, stat,
            bias_threshold=BIAS_THRESHOLD, bias_stable=True, bias_bounded=bounded,
        )
        verdicts[m] = v
        budget_rows.append({"method": m, "term": "throughput_loss", "value_frac": loss, "err_frac": 0.0,
                            "source": "injection_real_position", "note": "corrected in E3 by dividing by throughput"})
        budget_rows.append({"method": m, "term": "psf_model", "value_frac": 0.0, "err_frac": 0.0,
                            "source": "injection_psf_perturbation", "note": "perturbation is a no-op (flux-normalized PSF); term ~0/unmeasured"})
        budget_rows.append({"method": m, "term": "TOTAL", "value_frac": total["total_value_frac"], "err_frac": total["total_err_frac"],
                            "source": "quadrature", "note": ""})
        if bounded:
            sens_rows.append({"method": m, "axis": "position(real+3controls)", "range_frac": sens,
                              "note": "throughput spread across same-radius positions"})
            sens_rows.append({"method": m, "axis": "psf_fwhm_pm10", "range_frac": 0.0,
                              "note": "no-op: flux-normalized injection PSF"})

    with open(tables_dir / "g1_bias_budget.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "term", "value_frac", "err_frac", "source", "note"])
        w.writeheader(); w.writerows(budget_rows)
    with open(tables_dir / "g1_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "axis", "range_frac", "note"])
        w.writeheader(); w.writerows(sens_rows)

    n_ctrl = int(ctrl.shape[0])
    qc = {
        "stage": "g1_extraction_validation",
        "run_id": run_id,
        "provisional": True,
        "inputs": {"g0_qc": "stage_g0_qc.json", "e4_throughput": "tables/injection_throughput_by_method.csv"},
        "method_verdicts": verdicts,
        "bias_budget_table": "tables/g1_bias_budget.csv",
        "covariance": {
            "spectral_file": "stages/g1_channel_covariance.npz",
            "corr_length_channels_median": cov["corr_length_channels_median"],
            "corr_length_bootstrap_err": boot_err,
            "n_eff_over_n_median": n_eff_med,
            "rho_1_median": cov["rho_1_median"],
            "spatial_inflation_by_box": {str(k): v for k, v in spatial.items()},
            "domain_note": "control covariance does NOT capture the companion-position halo systematic (that is the injection bias budget).",
        },
        "sensitivity_table": "tables/g1_sensitivity.csv",
        "impact_on_x10_chi2": {
            "status": "not_computed",
            "reason": "provisional; X10 chi2 recompute deferred (D1 verdict divergent_continuum).",
        },
        "open_issues": [
            {"issue": f"{n_ctrl} controls used; covariance/per-position bias still noisy; bootstrap error reported.", "priority": "major"},
            {"issue": "PSF-perturbation bias term is unmeasured (injection PSF is flux-normalized -> no-op); the PSF-model systematic of the budget is a placeholder 0.", "priority": "major"},
            {"issue": (
                ", ".join(sorted(m for m, v in verdicts.items() if v == "rejected")) or "none"
            ) + " throughput below bias-bounded threshold at the companion edge -> rejected as extraction methods for this source.", "priority": "major"},
        ],
    }
    (stage_dir / "stage_g1_qc.json").write_text(json.dumps(qc, indent=2))
    return qc


def main(argv=None):
    ap = argparse.ArgumentParser(description="G1 extraction-validation QC runner.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args(argv)
    qc = build_g1(args.run_id, project_root=args.project_root)
    print(json.dumps(qc["method_verdicts"]))
    print("corr_length_channels_median:", qc["covariance"]["corr_length_channels_median"])


if __name__ == "__main__":
    main()
