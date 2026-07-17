"""Stage H06/E6: ROC curves of the matched-filter detector.

Spec: docs/spec_E6_codex_roc_curves.md (Julo et al. 2025 §3.3.3, Fig. 11).
Reuses the E5 method states (base residual map + ring stats + linear delta
injections); the null distribution is empirical: base-map ring pixels plus
z maps rebuilt at line-free wavelengths.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..io import write_json
from ..paths import RunPaths
from ..spectral import standard_line_free_mask
from .stage_h01_detect import expected_line_center_A
from .stage_h01b_fovmap import crosscorr_map, ring_noise_stats, spectral_map, spectral_template
from .stage_h01b_fovmap import _load_residual, _lsf_fwhm, _positions, _read_optional_json, _rv_sys_kms
from .stage_h05_contrast import (
    build_method_states,
    ring_positions,
    stage_h05_paths,
    stellar_line_flux,
)
from .stage_h01b_fovmap import _load_stage02_mean

SPEC_VERSION = "E6_v1"
DEFAULT_N_ANGLES = 16
DEFAULT_NULL_STEP_CHANNELS = 25
DEFAULT_CONTRAST = 1.7e-3  # paper's illustrative detection-limit contrast
MIN_NULL_SAMPLES = 500


@dataclass(frozen=True)
class StageH06Product:
    rows: list
    qc: dict


def stage_h06_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage_h06")
    base = stage_h05_paths(run_id, root)
    base.update(
        {
            "roc_csv": paths.table_dir / "roc_curves.csv",
            "stage_h06_qc_json": paths.stage_dir / "stage_h06_qc.json",
            "plot_dir": plot_dir,
            "summary_plot": plot_dir / "stage_h06_roc.png",
        }
    )
    return base


def stage_h06_config_from_run(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    run_config = load_run_config(
        run_id,
        project_root=project_root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    cfg = dict(run_config.config)
    if overrides:
        cfg.update(overrides)
    cfg["run_id"] = run_config.run_id
    cfg["project_root"] = str(run_config.paths.project_root)
    cfg.setdefault("h01b_line_rest_A", 6562.8)
    cfg.setdefault("h06_n_angles", DEFAULT_N_ANGLES)
    cfg.setdefault("h06_null_step_channels", DEFAULT_NULL_STEP_CHANNELS)
    cfg.setdefault("h06_contrast", DEFAULT_CONTRAST)
    return cfg


def roc_curve(injection_z, null_z):
    """DP vs FAP over all unique thresholds + trapezoidal AUC."""

    inj = np.asarray([z for z in injection_z if np.isfinite(z)], dtype=np.float64)
    null = np.asarray([z for z in null_z if np.isfinite(z)], dtype=np.float64)
    if inj.size == 0 or null.size == 0:
        raise ValueError("ROC needs non-empty injection and null samples.")
    thresholds = np.unique(np.concatenate([inj, null]))[::-1]
    dp = np.array([float(np.mean(inj >= t)) for t in thresholds])
    fap = np.array([float(np.mean(null >= t)) for t in thresholds])
    order = np.argsort(fap, kind="stable")
    auc = float(np.trapz(np.concatenate([[0.0], dp[order], [1.0]]),
                         np.concatenate([[0.0], fap[order], [1.0]])))
    return thresholds, dp, fap, auc


def null_z_samples(state, *, separation_px, band_px=2.0, null_centers_A=(), lsf_fwhm_A=2.6):
    """Empirical null: ring-band pixels of the base z map + line-free maps."""

    ny, nx = state.zmap.shape
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    rr = np.hypot(yy - float(state.star_yx[0]), xx - float(state.star_yx[1]))
    band = (rr >= separation_px - band_px) & (rr <= separation_px + band_px)
    samples = [state.zmap[band & np.isfinite(state.zmap)]]
    for center in null_centers_A:
        template = spectral_template(state.wave, float(center), lsf_fwhm_A)
        # Rebuild the map on the BASE residual with a line-free template: an
        # independent-ish noise realization of the same detector (paper §3.3.3).
        base_resid = state.base_residual
        m = spectral_map(base_resid, template)
        corr = crosscorr_map(m, state.kernel)
        zmap, _rings = ring_noise_stats(corr, state.star_yx)
        samples.append(zmap[band & np.isfinite(zmap)])
    return np.concatenate(samples) if samples else np.array([])


def null_wavelengths(wave, line_center_A, *, lsf_fwhm_A, step_channels=DEFAULT_NULL_STEP_CHANNELS):
    """Line-free template centers: standard-line-free channels, away from the target line."""

    wave = np.asarray(wave, dtype=np.float64)
    mask = standard_line_free_mask(wave)
    mask &= np.abs(wave - float(line_center_A)) > 4.0 * float(lsf_fwhm_A)
    # Stay away from the edges so the template support is complete.
    margin = 4.0 * float(lsf_fwhm_A)
    finite = wave[np.isfinite(wave)]
    mask &= (wave > finite.min() + margin) & (wave < finite.max() - margin)
    idx = np.where(mask)[0][:: max(int(step_channels), 1)]
    return [float(wave[i]) for i in idx]


def _contrast_for(method, separation_px, h05_qc, fallback):
    methods = (h05_qc or {}).get("methods") or {}
    curve = (methods.get(method) or {}).get("curve") or []
    best = None
    best_gap = None
    for entry in curve:
        if entry.get("contrast_50") is None:
            continue
        gap = abs(float(entry["separation_px"]) - float(separation_px))
        if best_gap is None or gap < best_gap:
            best, best_gap = float(entry["contrast_50"]), gap
    return best if best is not None else float(fallback)


def compute_stage_h06_products(config, paths=None):
    cfg = dict(config)
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_h06_paths(cfg["run_id"], root) if paths is None else paths
    open_issues = []

    star_yx, companion_yx = _positions(paths, cfg)
    lsf_fwhm_A = _lsf_fwhm(paths, cfg)
    line_center = expected_line_center_A(float(cfg.get("h01b_line_rest_A", 6562.8)), _rv_sys_kms(cfg))

    states, pre_cube, pre_wave, _psf_model = build_method_states(
        paths, cfg, star_yx=star_yx, companion_yx=companion_yx,
        line_center_A=line_center, lsf_fwhm_A=lsf_fwhm_A, open_issues=open_issues,
    )
    # Attach base residual cubes for the null maps (kept by build for reuse).
    for method, state in states.items():
        key = "sgf_residual_cube" if method == "sgf" else "lpm_residual_cube"
        state.base_residual, _w = _load_residual(paths[key])

    f_star = stellar_line_flux(
        pre_cube, pre_wave, star_yx,
        line_center_A=line_center, lsf_fwhm_A=lsf_fwhm_A,
        norm_radius_px=float(cfg.get("psf_norm_radius_px", 25.0)),
    )

    h05_qc = _read_optional_json(paths["stage_h05_qc_json"])
    if not h05_qc:
        open_issues.append("E5 QC unavailable; using h06_contrast for every scenario.")

    shape = pre_cube.shape[1:]
    separations = cfg.get("h06_separations_px")
    if separations is None:
        r_lo = float(cfg.get("h05_r_min_px", 4.0))
        halfsize = int(cfg.get("h01b_kernel_halfsize_px", 7))
        r_hi = min(shape) / 2.0 - halfsize - 1
        separations = [float(np.quantile(np.linspace(r_lo, r_hi, 64), q)) for q in (0.25, 0.5, 0.75)]
    separations = [float(s) for s in separations]
    n_angles = int(cfg.get("h06_n_angles", DEFAULT_N_ANGLES))
    guard = 2.0 * max(lsf_fwhm_A, 4.0)
    null_centers = null_wavelengths(
        pre_wave, line_center, lsf_fwhm_A=lsf_fwhm_A,
        step_channels=int(cfg.get("h06_null_step_channels", DEFAULT_NULL_STEP_CHANNELS)),
    )
    threshold = float(cfg.get("h05_threshold_sigma", 5.0))

    rows = []
    methods_qc = {}
    contrasts_used = {}
    for method, state in states.items():
        scenarios = []
        for sep in separations:
            contrast = _contrast_for(method, sep, h05_qc, cfg.get("h06_contrast", DEFAULT_CONTRAST))
            contrasts_used.setdefault(method, {})[str(sep)] = contrast
            inj_z = []
            for (y, x, _angle) in ring_positions(star_yx, sep, n_angles, shape, angle_offset_deg=11.25):
                if companion_yx is not None and np.hypot(y - companion_yx[0], x - companion_yx[1]) < guard:
                    continue
                result = state.detect((y, x), contrast, f_star, threshold)
                if result is not None:
                    inj_z.append(result["z"])
            null_z = null_z_samples(
                state, separation_px=sep, null_centers_A=null_centers, lsf_fwhm_A=lsf_fwhm_A
            )
            thresholds, dp, fap, auc = roc_curve(inj_z, null_z)
            for t, d, f in zip(thresholds, dp, fap):
                rows.append(
                    {
                        "method": method,
                        "separation_px": float(sep),
                        "contrast": float(contrast),
                        "threshold": float(t),
                        "dp": float(d),
                        "fap": float(f),
                    }
                )
            scenarios.append(
                {
                    "separation_px": float(sep),
                    "contrast": float(contrast),
                    "auc": float(auc),
                    "n_injections": int(len(inj_z)),
                    "n_null": int(null_z.size),
                }
            )
        methods_qc[method] = {"scenarios": scenarios}

    all_scenarios = [s for mq in methods_qc.values() for s in mq["scenarios"]]
    checks = {
        "v1_curves_written": bool(all_scenarios),
        "v2_auc_above_random": bool(
            all(s["auc"] >= 0.5 - 2.0 / np.sqrt(max(s["n_injections"], 1)) for s in all_scenarios)
        ),
        "v3_null_sample_ok": bool(all(s["n_null"] >= MIN_NULL_SAMPLES for s in all_scenarios)),
    }
    if not checks["v3_null_sample_ok"]:
        open_issues.append(f"Null sample below {MIN_NULL_SAMPLES} in some scenario; lower h06_null_step_channels.")

    qc = {
        "stage": "h06_roc",
        "spec_version": SPEC_VERSION,
        "run_id": str(cfg["run_id"]),
        "params": {
            "separations_px": separations,
            "contrasts_used": contrasts_used,
            "n_angles": n_angles,
            "null_step_channels": int(cfg.get("h06_null_step_channels", DEFAULT_NULL_STEP_CHANNELS)),
            "n_null_wavelengths": len(null_centers),
            "line_center_A": float(line_center),
        },
        "methods": methods_qc,
        "checks": checks,
        "open_issues": open_issues,
    }
    return StageH06Product(rows=rows, qc=qc)


def write_stage_h06_products(product: StageH06Product, config, paths):
    paths["paths"].ensure_base_dirs()
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    with open(paths["roc_csv"], "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["method", "separation_px", "contrast", "threshold", "dp", "fap"]
        )
        writer.writeheader()
        writer.writerows(product.rows)
    _write_stage_h06_plot(product, paths)
    write_json(paths["stage_h06_qc_json"], product.qc)
    return {"qc_json": paths["stage_h06_qc_json"], "qc": product.qc}


def _write_stage_h06_plot(product: StageH06Product, paths):
    import os

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for method in sorted({row["method"] for row in product.rows}):
        for sep in sorted({row["separation_px"] for row in product.rows if row["method"] == method}):
            pts = [(row["fap"], row["dp"]) for row in product.rows
                   if row["method"] == method and row["separation_px"] == sep]
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.2, label=f"{method} r={sep:g}px")
    ax.plot([0, 1], [0, 1], color="k", ls=":", lw=0.8, label="random")
    ax.set_xlabel("False Alarm Probability")
    ax.set_ylabel("Detection Probability")
    ax.legend(fontsize=7)
    ax.set_title("E6: ROC (Julo+25 Fig. 11)")
    fig.tight_layout()
    fig.savefig(paths["summary_plot"], dpi=120)
    plt.close(fig)


def run_stage_h06(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_h06_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_h06_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_h06_products(cfg, paths)
    written = write_stage_h06_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage H06/E6 ROC curves.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage_h06(
        args.run_id,
        project_root=args.project_root,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage_h06_qc_json"])


__all__ = [
    "SPEC_VERSION",
    "StageH06Product",
    "compute_stage_h06_products",
    "null_wavelengths",
    "null_z_samples",
    "roc_curve",
    "run_stage_h06",
    "stage_h06_config_from_run",
    "stage_h06_paths",
    "write_stage_h06_products",
]


if __name__ == "__main__":
    main()
