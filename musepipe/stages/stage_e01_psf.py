"""Stage E01/C1: chromatic Moffat PSF model for extraction stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from ..io import read_json, write_json
from ..paths import RunPaths
from ..psf import (
    build_psf_model_document,
    companion_ring_metric,
    corner_background,
    evaluate_moffat_fit,
    evaluate_radial_profile,
    fit_moffat_image,
    psf_roundtrip_error,
    radial_hybrid_profile,
    source_mask,
)


@dataclass(frozen=True)
class StageE01Product:
    fit_rows: list[dict]
    psf_model: dict
    qc: dict
    hybrid_profiles: np.ndarray | None
    hybrid_radii: np.ndarray | None


def stage_e01_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage_e01")
    return {
        "paths": paths,
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "stage_e01_params_csv": paths.stage_dir / "stage_e01_psf_params.csv",
        "psf_model_json": paths.stage_dir / "psf_model.json",
        "stage_e01_qc_json": paths.stage_dir / "stage_e01_qc.json",
        "psf_hybrid_residual_fits": paths.stage_dir / "psf_hybrid_residual.fits",
        "plot_dir": plot_dir,
        "summary_plot": plot_dir / "stage_e01_psf_summary.png",
    }


def stage_e01_config_from_run(
    run_id=None,
    *,
    project_root=None,
    overrides=None,
    allow_run_id_mismatch=False,
):
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
    cfg.setdefault("stage_e01_input_cube_fits", str(run_config.paths.stage_dir / "stage02_xcorr_cube_stack.fits"))
    cfg.setdefault("stage_e01_positions_qc", str(run_config.paths.stage_dir / "stage01c_qc.json"))
    cfg.setdefault("psf_bin_A", 100.0)
    cfg.setdefault("psf_min_channels_per_bin", 3)
    cfg.setdefault("psf_fit_radius_px", 28.0)
    cfg.setdefault("psf_norm_radius_px", 25.0)
    cfg.setdefault("psf_companion_ring_width_px", 3.0)
    cfg.setdefault("psf_mask_radius_factor", 3.0)
    cfg.setdefault("psf_sigma_clip", 3.0)
    cfg.setdefault("psf_max_clip_iter", 3)
    cfg.setdefault("psf_hybrid_threshold_pct", 5.0)
    cfg.setdefault("psf_hybrid_bin_fraction", 0.2)
    cfg.setdefault("psf_hybrid_smoothing_scale_factor", 2.0)
    cfg.setdefault("psf_save_plots", bool(cfg.get("save_intermediate_plots", False)))
    cfg.setdefault("stage_e01_bad_windows_A", _bad_windows_from_config(cfg))
    return cfg


def _bad_windows_from_config(cfg):
    windows = cfg.get("stage_e01_bad_windows_A") or cfg.get("bad_wavelength_ranges_A")
    if windows is not None:
        return windows
    if cfg.get("drop_wave_min_A") is not None and cfg.get("drop_wave_max_A") is not None:
        return [[float(cfg["drop_wave_min_A"]), float(cfg["drop_wave_max_A"])]]
    return []


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maoppy_status():
    try:
        import maoppy  # noqa: F401

        return "available_not_used"
    except Exception as exc:
        return f"unavailable:{exc.__class__.__name__}"


def _load_cube(path):
    with fits.open(path, memmap=True) as hdul:
        cubes = hdul["CUBES"].data.astype(np.float32)
        wavelengths = hdul["WAVELENGTH"].data.astype(np.float64)
        stat = hdul["STAT"].data.astype(np.float32) if "STAT" in hdul else None
    if cubes.ndim != 4:
        raise RuntimeError(f"Expected CUBES shape (N,nz,ny,nx), got {cubes.shape}.")
    return cubes, wavelengths, stat


def _good_wave_mask(wavelengths, bad_windows_A):
    wave = np.asarray(wavelengths, dtype=np.float64)
    good = np.isfinite(wave)
    for lo, hi in bad_windows_A or ():
        good &= ~((wave >= float(lo)) & (wave <= float(hi)))
    return good


def make_psf_bins(wavelengths, *, bin_A=100.0, bad_windows_A=(), min_channels=3):
    wave = np.asarray(wavelengths, dtype=np.float64)
    good = _good_wave_mask(wave, bad_windows_A)
    if np.count_nonzero(good) < int(min_channels):
        raise RuntimeError("Too few good wavelength channels for PSF bins.")
    wmin = float(np.nanmin(wave[good]))
    wmax = float(np.nanmax(wave[good]))
    edges = np.arange(wmin, wmax + float(bin_A), float(bin_A))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = good & (wave >= lo) & (wave < hi)
        if np.count_nonzero(mask) >= int(min_channels):
            bins.append(
                {
                    "wave_min_A": float(lo),
                    "wave_max_A": float(hi),
                    "wave_center_A": float(np.nanmedian(wave[mask])),
                    "indices": np.where(mask)[0],
                }
            )
    if not bins:
        raise RuntimeError("No valid PSF bins after wavelength masking.")
    return bins


def _median_image(cubes, indices):
    with np.errstate(all="ignore"):
        return np.nanmedian(cubes[:, indices, :, :], axis=(0, 1)).astype(np.float64)


def _positions_from_qc(qc):
    primary = tuple(map(float, qc["primary"]["pos_yx"]))
    companion = tuple(map(float, qc["companion"]["pos_yx"]))
    field = None
    if qc.get("field_source") is not None:
        field = tuple(map(float, qc["field_source"]["pos_yx"]))
    return primary, companion, field


def _fit_source_mask(shape, primary_yx, companion_yx, field_yx, mask_radius_px):
    centers = [companion_yx]
    if field_yx is not None:
        centers.append(field_yx)
    return source_mask(shape, centers, float(mask_radius_px))


def _detect_core_mask(image, primary_yx, cfg):
    threshold = cfg.get("psf_saturation_threshold")
    if threshold is None:
        return 0.0, False
    y, x = map(int, map(round, primary_yx))
    peak = float(image[y, x])
    if peak > float(threshold):
        return float(cfg.get("psf_core_mask_px", 2.0)), True
    return 0.0, False


def _row_from_fit(bin_index, bin_info, fit, metric, metric_hybrid=None):
    p = fit.params
    row = {
        "bin_index": int(bin_index),
        "wave_min_A": float(bin_info["wave_min_A"]),
        "wave_max_A": float(bin_info["wave_max_A"]),
        "wave_center_A": float(bin_info["wave_center_A"]),
        "n_channels": int(len(bin_info["indices"])),
        "success": bool(fit.success),
        "background": float(fit.background),
        "amplitude": float(p["amplitude"]),
        "y0": float(p["y0"]),
        "x0": float(p["x0"]),
        "fwhm_maj": float(p["fwhm_maj"]),
        "fwhm_min": float(p["fwhm_min"]),
        "theta_deg": float(p["theta_deg"]),
        "beta": float(p["beta"]),
        "chi2r": float(fit.chi2r),
        "clip_frac": float(fit.clip_frac),
        "n_fit": int(fit.n_fit),
        "ring_residual_pct": float(metric["median_pct"]),
        "ring_residual_p90_pct": float(metric["p90_pct"]),
    }
    for key, value in fit.errors.items():
        row[f"{key}_err"] = None if not np.isfinite(value) else float(value)
    if metric_hybrid is not None:
        row["ring_residual_pct_after_hybrid"] = float(metric_hybrid["median_pct"])
        row["ring_residual_p90_pct_after_hybrid"] = float(metric_hybrid["p90_pct"])
    return row


def _moffat_fit_rows(cubes, wavelengths, bins, positions_qc, cfg):
    primary_yx, companion_yx, field_yx = _positions_from_qc(positions_qc)
    fwhm_prelim = float(positions_qc.get("psf", {}).get("fwhm_px", cfg.get("psf_prelim_fwhm_px", 4.0)))
    mask_radius = float(cfg.get("psf_companion_mask_radius_px", cfg.get("psf_mask_radius_factor", 3.0) * fwhm_prelim))
    rows = []
    images = []
    models = []
    masks = []
    core_masks = []
    for i, bin_info in enumerate(bins):
        image = _median_image(cubes, bin_info["indices"])
        mask = _fit_source_mask(image.shape, primary_yx, companion_yx, field_yx, mask_radius)
        core_mask_px, saturation = _detect_core_mask(image, primary_yx, cfg)
        background = corner_background(image)
        fit = fit_moffat_image(
            image,
            center_yx=primary_yx,
            fit_radius_px=float(cfg.get("psf_fit_radius_px", 28.0)),
            mask=mask,
            background=background,
            core_mask_px=core_mask_px,
            sigma_clip=cfg.get("psf_sigma_clip", 3.0),
            max_iter=int(cfg.get("psf_max_clip_iter", 3)),
        )
        model = evaluate_moffat_fit(image.shape, fit)
        metric = companion_ring_metric(
            image,
            model,
            primary_yx,
            companion_yx,
            width_px=float(cfg.get("psf_companion_ring_width_px", 3.0)),
            source_exclusion_radius_px=mask_radius,
        )
        row = _row_from_fit(i, bin_info, fit, metric)
        row["core_mask_px"] = float(core_mask_px)
        row["saturation_detected"] = bool(saturation)
        rows.append(row)
        images.append(image)
        models.append(model)
        masks.append(mask)
        core_masks.append(core_mask_px)
    return rows, images, models, masks, {
        "mask_radius_px": mask_radius,
        "core_mask_px_max": float(np.nanmax(core_masks)) if core_masks else 0.0,
        "saturation_detected": bool(any(row["saturation_detected"] for row in rows)),
    }


def _apply_hybrid_if_needed(rows, images, models, masks, positions_qc, cfg):
    threshold = float(cfg.get("psf_hybrid_threshold_pct", 5.0))
    frac_limit = float(cfg.get("psf_hybrid_bin_fraction", 0.2))
    values = np.asarray([row["ring_residual_pct"] for row in rows], dtype=np.float64)
    fail_frac = float(np.count_nonzero(values > threshold) / max(values.size, 1))
    if fail_frac <= frac_limit:
        return rows, models, None, None, False

    primary_yx, companion_yx, _ = _positions_from_qc(positions_qc)
    fwhm_med = float(np.nanmedian([row["fwhm_maj"] for row in rows]))
    smooth = max(2.0 * fwhm_med, float(cfg.get("psf_hybrid_smoothing_scale_factor", 2.0)) * fwhm_med)
    profiles = []
    radii_ref = None
    models_hybrid = []
    for row, image, model, mask in zip(rows, images, models, masks):
        radii, profile = radial_hybrid_profile(
            image - model,
            primary_yx,
            mask=mask,
            smoothing_scale_px=smooth,
        )
        hybrid = evaluate_radial_profile(image.shape, primary_yx, radii, profile)
        model_h = model + hybrid
        metric_h = companion_ring_metric(
            image,
            model_h,
            primary_yx,
            companion_yx,
            width_px=float(cfg.get("psf_companion_ring_width_px", 3.0)),
            source_exclusion_radius_px=float(cfg.get("psf_companion_mask_radius_px", cfg.get("psf_mask_radius_factor", 3.0) * fwhm_med)),
        )
        row["ring_residual_pct_after_hybrid"] = float(metric_h["median_pct"])
        row["ring_residual_p90_pct_after_hybrid"] = float(metric_h["p90_pct"])
        if radii_ref is None:
            radii_ref = radii
        if radii.size != radii_ref.size:
            profile = np.interp(radii_ref, radii, profile)
        profiles.append(profile)
        models_hybrid.append(model_h)
    return rows, models_hybrid, np.asarray(profiles, dtype=np.float32), radii_ref.astype(np.float32), True


def compute_stage_e01_products(config) -> StageE01Product:
    cfg = dict(config)
    paths = stage_e01_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    cube_path = Path(cfg.get("stage_e01_input_cube_fits", paths["stage02_cube_fits"]))
    positions_path = Path(cfg.get("stage_e01_positions_qc", paths["stage01c_qc_json"]))
    if not positions_path.exists():
        raise FileNotFoundError(f"Stage01c QC is required for C1: {positions_path}")
    positions_qc = read_json(positions_path)
    cubes, wavelengths, _ = _load_cube(cube_path)
    bins = make_psf_bins(
        wavelengths,
        bin_A=float(cfg.get("psf_bin_A", 100.0)),
        bad_windows_A=cfg.get("stage_e01_bad_windows_A", []),
        min_channels=int(cfg.get("psf_min_channels_per_bin", 3)),
    )
    rows, images, models, masks, mask_meta = _moffat_fit_rows(cubes, wavelengths, bins, positions_qc, cfg)
    rows, final_models, hybrid_profiles, hybrid_radii, hybrid_applied = _apply_hybrid_if_needed(
        rows,
        images,
        models,
        masks,
        positions_qc,
        cfg,
    )
    model_doc = build_psf_model_document(
        [row["wave_center_A"] for row in rows],
        rows,
        form="moffat",
        norm_radius_px=float(cfg.get("psf_norm_radius_px", 25.0)),
        hybrid=hybrid_applied,
    )
    roundtrip = psf_roundtrip_error(
        model_doc,
        np.linspace(rows[0]["wave_center_A"], rows[-1]["wave_center_A"], min(10, len(rows))),
    )
    metric_key = "ring_residual_pct_after_hybrid" if hybrid_applied else "ring_residual_pct"
    metric_values = np.asarray([row.get(metric_key, row["ring_residual_pct"]) for row in rows], dtype=np.float64)
    p90_key = "ring_residual_p90_pct_after_hybrid" if hybrid_applied else "ring_residual_p90_pct"
    p90_values = np.asarray([row.get(p90_key, row["ring_residual_p90_pct"]) for row in rows], dtype=np.float64)
    primary_yx, companion_yx, _ = _positions_from_qc(positions_qc)
    sep_px = float(np.hypot(companion_yx[0] - primary_yx[0], companion_yx[1] - primary_yx[1]))
    centroid_diff = max(
        float(abs(row["y0"] - primary_yx[0])) for row in rows
    )
    centroid_diff = max(centroid_diff, max(float(abs(row["x0"] - primary_yx[1])) for row in rows))
    open_issues = []
    maoppy_status = _maoppy_status()
    if maoppy_status.startswith("unavailable"):
        open_issues.append("maoppy contrast fit not run because maoppy is unavailable in this environment.")
    if np.nanpercentile(metric_values, 90) > float(cfg.get("psf_hybrid_threshold_pct", 5.0)):
        open_issues.append("Companion-ring residual remains above 5 pct after the fixed C1 sequence.")
    if centroid_diff > 0.3 and positions_qc.get("chromatic", {}).get("chromatic_centroid_needed"):
        open_issues.append("PSF centroid differs from B3 chromatic centroid story by >0.3 px.")
    bins_interpolated = []
    for lo, hi in cfg.get("stage_e01_bad_windows_A", []):
        if float(hi) >= rows[0]["wave_center_A"] and float(lo) <= rows[-1]["wave_center_A"]:
            bins_interpolated.append([float(lo), float(hi)])
    qc = {
        "stage": "e01_chromatic_psf",
        "run_id": cfg["run_id"],
        "input": {"cube": str(cube_path), "sha256": _sha256(cube_path), "positions_from": str(positions_path)},
        "binning": {
            "bin_A": float(cfg.get("psf_bin_A", 100.0)),
            "n_bins": int(len(rows)),
            "excluded_windows_A": cfg.get("stage_e01_bad_windows_A", []),
            "bins_interpolated": bins_interpolated,
        },
        "masks": {
            "companion_radius_px": float(mask_meta["mask_radius_px"]),
            "chromatic_tracking": bool(positions_qc.get("chromatic", {}).get("chromatic_centroid_needed", False)),
            "core_mask_px": float(mask_meta["core_mask_px_max"]),
            "saturation_detected": bool(mask_meta["saturation_detected"]),
        },
        "fit": {
            "form_chosen": "moffat",
            "maoppy_status": maoppy_status,
            "background_mode": "fixed_external",
            "clip_frac_max": float(np.nanmax([row["clip_frac"] for row in rows])),
            "chi2r_median": float(np.nanmedian([row["chi2r"] for row in rows])),
        },
        "smoothing": {
            "per_param_model": {key: model_doc["coefficients"][key]["model"] for key in model_doc["coefficients"]},
            "outlier_bins": [],
            "centroid_vs_b3_max_diff_px": float(centroid_diff),
        },
        "companion_ring_metric": {
            "radius_px": sep_px,
            "width_px": float(cfg.get("psf_companion_ring_width_px", 3.0)),
            "residual_pct_median": float(np.nanmedian(metric_values)),
            "residual_pct_p90": float(np.nanpercentile(p90_values, 90)),
            "bins_above_5pct": int(np.count_nonzero(metric_values > 5.0)),
            "after_hybrid": bool(hybrid_applied),
        },
        "hybrid": {
            "applied": bool(hybrid_applied),
            "smoothing_scale_px": None
            if hybrid_profiles is None
            else float(max(2.0 * np.nanmedian([row["fwhm_maj"] for row in rows]), 0.0)),
        },
        "normalization": {
            "norm_radius_px": float(cfg.get("psf_norm_radius_px", 25.0)),
            "roundtrip_error": float(roundtrip),
        },
        "open_issues": open_issues,
    }
    return StageE01Product(
        fit_rows=rows,
        psf_model=model_doc,
        qc=qc,
        hybrid_profiles=hybrid_profiles,
        hybrid_radii=hybrid_radii,
    )


def write_stage_e01_products(product: StageE01Product, config, paths):
    paths["paths"].ensure_base_dirs()
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    _write_csv(paths["stage_e01_params_csv"], product.fit_rows)
    write_json(paths["psf_model_json"], product.psf_model)
    if product.hybrid_profiles is not None:
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(product.hybrid_profiles.astype(np.float32), name="PROFILE"),
                fits.ImageHDU(product.hybrid_radii.astype(np.float32), name="RADIUS_PX"),
            ]
        ).writeto(paths["psf_hybrid_residual_fits"], overwrite=True)
    if bool(config.get("psf_save_plots", config.get("save_intermediate_plots", False))):
        _write_summary_plot(product, paths)
        product.qc["figures"] = {"summary": str(Path("plots") / "stage_e01" / paths["summary_plot"].name)}
    write_json(paths["stage_e01_qc_json"], product.qc)
    return {"params_csv": paths["stage_e01_params_csv"], "model_json": paths["psf_model_json"], "qc_json": paths["stage_e01_qc_json"], "qc": product.qc}


def _write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_plot(product, paths):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = product.fit_rows
    wave = np.asarray([row["wave_center_A"] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(wave, [row["fwhm_maj"] for row in rows], label="maj")
    axes[0, 0].plot(wave, [row["fwhm_min"] for row in rows], label="min")
    axes[0, 0].set_ylabel("FWHM [px]")
    axes[0, 0].legend()
    axes[0, 1].plot(wave, [row["beta"] for row in rows])
    axes[0, 1].set_ylabel("beta")
    axes[1, 0].plot(wave, [row["ring_residual_pct"] for row in rows], label="Moffat")
    if product.qc["hybrid"]["applied"]:
        axes[1, 0].plot(wave, [row["ring_residual_pct_after_hybrid"] for row in rows], label="hybrid")
    axes[1, 0].axhline(5.0, color="0.4", ls="--")
    axes[1, 0].set_ylabel("Ring residual [%]")
    axes[1, 0].legend()
    axes[1, 1].plot(wave, [row["chi2r"] for row in rows])
    axes[1, 1].set_ylabel("chi2r")
    for ax in axes.ravel():
        ax.set_xlabel("Wavelength [A]")
    fig.savefig(paths["summary_plot"], dpi=150)
    plt.close(fig)


def run_stage_e01(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_e01_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_e01_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_e01_products(cfg)
    written = write_stage_e01_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": product.qc, "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage E01/C1 chromatic PSF model.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args(argv)
    overrides = {"psf_save_plots": True} if args.save_plots else None
    result = run_stage_e01(
        args.run_id,
        project_root=args.project_root,
        overrides=overrides,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage_e01_qc_json"])


__all__ = [
    "StageE01Product",
    "compute_stage_e01_products",
    "make_psf_bins",
    "run_stage_e01",
    "stage_e01_config_from_run",
    "stage_e01_paths",
    "write_stage_e01_products",
]


if __name__ == "__main__":
    main()
