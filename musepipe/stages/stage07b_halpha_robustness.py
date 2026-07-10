"""Stage 07b: Halpha robustness checks for local-surface extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from ..apertures import aperture_weights, same_radius_control_positions
from ..config import load_run_config
from ..io import get_cube_data, read_json, read_wavelengths_and_masks, write_csv, write_json
from ..paths import RunPaths
from ..stats import empirical_z, median_finite, robust_sigma


HALPHA_A = 6562.80
LINE_WINDOW_HALF_WIDTH_A = 18.0
LINE_INTEGRATION_HALF_WIDTH_A = 2.5
LINE_CONTINUUM_INNER_A = 6.0
LINE_CONTINUUM_OUTER_A = 16.0


def default_apertures():
    return [
        {"name": "pixel", "kind": "pixel"},
        {"name": "box3_sum", "kind": "box", "size": 3},
        {"name": "circle_r1p5_sum", "kind": "circle", "radius_px": 1.5},
        {"name": "circle_r2p0_sum", "kind": "circle", "radius_px": 2.0},
        {"name": "gauss_sig1p0_r3_sum", "kind": "gaussian", "sigma_px": 1.0, "radius_px": 3.0},
    ]


STAGE07B_DEFAULTS = {
    "halpha_A": HALPHA_A,
    "line_window_half_width_A": LINE_WINDOW_HALF_WIDTH_A,
    "line_integration_half_width_A": LINE_INTEGRATION_HALF_WIDTH_A,
    "line_continuum_inner_A": LINE_CONTINUUM_INNER_A,
    "line_continuum_outer_A": LINE_CONTINUUM_OUTER_A,
    "local_model_kind": "plane",
    "fit_radius_px": 12.0,
    "mask_radius_px": 3.0,
    "local_fit_sigma_clip": 3.0,
    "local_fit_max_iter": 3,
    "local_fit_min_pixels": 30,
    "control_apertures": 8,
    "control_exclude_angle_deg": 25.0,
    "control_margin_px": 4,
}


def _as_yx(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if "y" in value and "x" in value:
            return (int(value["y"]), int(value["x"]))
        return None
    if len(value) != 2:
        return None
    return (int(value[0]), int(value[1]))


def _as_xy(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return (int(value["x"]), int(value["y"]))
        return None
    if len(value) != 2:
        return None
    return (int(value[0]), int(value[1]))


def _spatial_shape_from_shape(shape):
    if shape is None:
        return None
    vals = list(shape)
    if vals and isinstance(vals[0], (list, tuple)):
        return _spatial_shape_from_shape(vals[0])
    if len(vals) < 2:
        return None
    return (int(vals[-2]), int(vals[-1]))


def resolve_spatial_shape(cfg, qc04b):
    for key in ("output_shape", "input_shape", "cube_shape", "spatial_shape"):
        shape = _spatial_shape_from_shape(qc04b.get(key))
        if shape is not None:
            return shape
    crop_npix = cfg.get("crop_npix")
    if crop_npix is not None:
        npix = int(crop_npix)
        return (npix, npix)
    return None


def resolve_object_yx(cfg, qc04b, fallback_xy=None):
    for source in (cfg, qc04b):
        for key in ("stage07b_object_yx", "stage08_object_yx", "object_yx", "target_yx", "stage04b_target_yx"):
            yx = _as_yx(source.get(key))
            if yx is not None:
                return yx
        for key in (
            "stage07b_object_xy",
            "stage08_object_xy",
            "object_xy",
            "target_xy",
            "stage04b_target_xy",
            "science_object_xy",
        ):
            xy = _as_xy(source.get(key))
            if xy is not None:
                return (xy[1], xy[0])

    target_object = qc04b.get("target_object")
    detected_peaks = qc04b.get("detected_peaks")
    if target_object is not None and isinstance(detected_peaks, dict):
        yx = _as_yx(detected_peaks.get(str(target_object)))
        if yx is not None:
            return yx

    xy = _as_xy(fallback_xy)
    if xy is not None:
        return (xy[1], xy[0])
    raise RuntimeError("Could not resolve Stage 07b object position from config or stage04b_qc.json.")


def resolve_star_yx(cfg, qc04b):
    for source in (cfg, qc04b):
        for key in ("stage07b_star_yx", "stage08_star_yx", "star_yx", "reference_geometry_center_yx"):
            yx = _as_yx(source.get(key))
            if yx is not None:
                return yx
        for key in ("stage07b_star_xy", "stage08_star_xy", "star_xy", "reference_geometry_center_xy"):
            xy = _as_xy(source.get(key))
            if xy is not None:
                return (xy[1], xy[0])

    shape = resolve_spatial_shape(cfg, qc04b)
    if shape is None:
        raise RuntimeError("Could not resolve Stage 07b star center; set stage07b_star_yx in config.")
    ny, nx = shape
    return (int(ny // 2), int(nx // 2))


def stage07b_config_from_run(
    run_id=None,
    *,
    project_root=None,
    overrides=None,
    allow_run_id_mismatch=False,
):
    """Build Stage 07b config from the active run config and Stage04b QC."""

    run_config = load_run_config(
        run_id,
        project_root=project_root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    cfg = run_config.config
    paths = run_config.paths
    qc_path = paths.stage_dir / "stage04b_qc.json"
    qc04b = read_json(qc_path) if qc_path.exists() else {}

    object_yx = resolve_object_yx(cfg, qc04b)
    star_yx = resolve_star_yx(cfg, qc04b)

    config = {
        "run_id": run_config.run_id,
        "project_root": str(paths.project_root),
        "object_xy": (int(object_yx[1]), int(object_yx[0])),
        "star_yx": (int(star_yx[0]), int(star_yx[1])),
        "apertures": default_apertures(),
        **STAGE07B_DEFAULTS,
    }

    for key, default_value in STAGE07B_DEFAULTS.items():
        config[key] = cfg.get(f"stage07b_{key}", cfg.get(key, default_value))
    if "stage07b_apertures" in cfg:
        config["apertures"] = cfg["stage07b_apertures"]
    if overrides:
        config.update(overrides)
    return config


def fit_surface_model(patch, a_flat, fit_mask_flat, model_kind, sigma_clip, max_iter, min_pixels):
    """Fit one local patch with robust clipping, matching the original notebook."""

    z = np.asarray(patch, dtype=np.float64).ravel()
    finite = np.isfinite(z)
    base = np.asarray(fit_mask_flat, dtype=bool) & finite
    if int(np.sum(base)) < int(min_pixels):
        level = np.nanmedian(z[finite]) if np.any(finite) else 0.0
        return np.full(patch.shape, level, dtype=np.float64), int(np.sum(base))

    valid = base.copy()
    if model_kind == "constant":
        level = np.nanmedian(z[valid])
        for _ in range(int(max_iter)):
            resid = z - level
            sig = robust_sigma(resid[valid])
            if not np.isfinite(sig) or sig <= 0:
                break
            med = np.nanmedian(resid[valid])
            new_valid = base & (np.abs(resid - med) <= float(sigma_clip) * sig)
            if int(np.sum(new_valid)) < int(min_pixels) or np.array_equal(new_valid, valid):
                break
            valid = new_valid
            level = np.nanmedian(z[valid])
        return np.full(patch.shape, level, dtype=np.float64), int(np.sum(valid))

    coeff = None
    for _ in range(int(max_iter)):
        if int(np.sum(valid)) < int(min_pixels):
            break
        coeff, *_ = np.linalg.lstsq(a_flat[valid], z[valid], rcond=None)
        model_flat = a_flat @ coeff
        resid = z - model_flat
        sig = robust_sigma(resid[valid])
        if not np.isfinite(sig) or sig <= 0:
            break
        med = np.nanmedian(resid[valid])
        new_valid = base & (np.abs(resid - med) <= float(sigma_clip) * sig)
        if int(np.sum(new_valid)) < int(min_pixels) or np.array_equal(new_valid, valid):
            break
        valid = new_valid

    if coeff is None:
        coeff, *_ = np.linalg.lstsq(a_flat[valid], z[valid], rcond=None)
    return (a_flat @ coeff).reshape(patch.shape), int(np.sum(valid))


def local_surface_spectra_for_apertures(
    cube,
    center_yx,
    apertures,
    wave_indices,
    fit_radius_px,
    mask_radius_px,
    model_kind,
    sigma_clip,
    max_iter,
    min_pixels,
):
    """Extract robust local-surface residual spectra around one position."""

    nz, ny, nx = cube.shape
    cy, cx = map(float, center_yx)
    y1 = max(0, int(np.floor(cy - fit_radius_px)))
    y2 = min(ny, int(np.ceil(cy + fit_radius_px)) + 1)
    x1 = max(0, int(np.floor(cx - fit_radius_px)))
    x2 = min(nx, int(np.ceil(cx + fit_radius_px)) + 1)

    yy, xx = np.mgrid[y1:y2, x1:x2]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    fit_mask = (rr <= float(fit_radius_px)) & (rr > float(mask_radius_px))
    a_flat = np.column_stack(
        [
            np.ones(yy.size, dtype=np.float64),
            yy.ravel().astype(np.float64) - cy,
            xx.ravel().astype(np.float64) - cx,
        ]
    )
    fit_mask_flat = fit_mask.ravel()

    out = {}
    aperture_meta = {}
    local_weights = {}
    for aperture in apertures:
        weights = aperture_weights(ny, nx, center_yx, aperture)[y1:y2, x1:x2]
        ys, xs = np.where(weights > 0)
        w = weights[ys, xs].astype(np.float64)
        local_weights[aperture["name"]] = (ys, xs, w)
        out[aperture["name"]] = np.full(nz, np.nan, dtype=np.float64)
        aperture_meta[aperture["name"]] = {
            "n_pix": int(w.size),
            "weight_sum": float(np.sum(w)),
            "effective_n_pix": float((np.sum(w) ** 2) / np.sum(w**2)) if np.sum(w**2) > 0 else np.nan,
        }

    nfit_values = []
    for k in wave_indices:
        patch = cube[int(k), y1:y2, x1:x2]
        model, nfit = fit_surface_model(patch, a_flat, fit_mask_flat, model_kind, sigma_clip, max_iter, min_pixels)
        residual = patch.astype(np.float64) - model
        nfit_values.append(nfit)
        for aperture in apertures:
            name = aperture["name"]
            ys, xs, w = local_weights[name]
            out[name][int(k)] = float(np.nansum(residual[ys, xs] * w))

    return out, aperture_meta, {
        "fit_y1": int(y1),
        "fit_y2": int(y2),
        "fit_x1": int(x1),
        "fit_x2": int(x2),
        "nfit_median": float(np.nanmedian(nfit_values)) if nfit_values else np.nan,
        "nfit_min": int(np.nanmin(nfit_values)) if nfit_values else 0,
    }


def halpha_masks(
    waves,
    good_mask,
    *,
    halpha_A=HALPHA_A,
    line_integration_half_width_A=LINE_INTEGRATION_HALF_WIDTH_A,
    line_continuum_inner_A=LINE_CONTINUUM_INNER_A,
    line_continuum_outer_A=LINE_CONTINUUM_OUTER_A,
    line_window_half_width_A=LINE_WINDOW_HALF_WIDTH_A,
):
    dw = np.abs(np.asarray(waves, dtype=np.float64) - float(halpha_A))
    good = np.asarray(good_mask, dtype=bool) & np.isfinite(waves)
    line = good & (dw <= float(line_integration_half_width_A))
    cont = good & (dw >= float(line_continuum_inner_A)) & (dw <= float(line_continuum_outer_A))
    plot = good & (dw <= float(line_window_half_width_A))
    return line, cont, plot


def median_stack_spectrum(specs, wave_indices):
    if not specs:
        return None
    out = np.full_like(specs[0], np.nan, dtype=np.float64)
    stack = np.vstack([s[wave_indices] for s in specs])
    out[wave_indices] = np.nanmedian(stack, axis=0)
    return out


def leave_one_out_control_sub(control_specs, wave_indices):
    out = []
    for i, spec in enumerate(control_specs):
        others = [s for j, s in enumerate(control_specs) if j != i]
        sub = np.full_like(spec, np.nan, dtype=np.float64)
        if others:
            med = np.nanmedian(np.vstack([s[wave_indices] for s in others]), axis=0)
            sub[wave_indices] = spec[wave_indices] - med
        out.append(sub)
    return out


def line_metrics(
    waves,
    spec,
    good_mask,
    *,
    halpha_A=HALPHA_A,
    line_integration_half_width_A=LINE_INTEGRATION_HALF_WIDTH_A,
    line_continuum_inner_A=LINE_CONTINUUM_INNER_A,
    line_continuum_outer_A=LINE_CONTINUUM_OUTER_A,
    line_window_half_width_A=LINE_WINDOW_HALF_WIDTH_A,
):
    spec = np.asarray(spec, dtype=np.float64)
    line_mask, cont_mask, _ = halpha_masks(
        waves,
        good_mask,
        halpha_A=halpha_A,
        line_integration_half_width_A=line_integration_half_width_A,
        line_continuum_inner_A=line_continuum_inner_A,
        line_continuum_outer_A=line_continuum_outer_A,
        line_window_half_width_A=line_window_half_width_A,
    )
    cont_values = spec[cont_mask]
    line_values = spec[line_mask]
    continuum = float(np.nanmedian(cont_values)) if np.any(np.isfinite(cont_values)) else np.nan
    sigma = robust_sigma(cont_values - continuum)
    line_minus_cont = line_values - continuum

    if np.any(np.isfinite(line_minus_cont)):
        peak_idx_local = int(np.nanargmax(line_minus_cont))
        line_waves = np.asarray(waves, dtype=np.float64)[line_mask]
        peak_wave = float(line_waves[peak_idx_local])
        peak = float(line_minus_cont[peak_idx_local])
        mean = float(np.nanmean(line_minus_cont))
        flux = float(np.nansum(line_minus_cont))
    else:
        peak_wave = np.nan
        peak = np.nan
        mean = np.nan
        flux = np.nan

    return {
        "continuum_median": continuum,
        "continuum_sigma": float(sigma),
        "line_flux_native": flux,
        "line_peak_above_continuum": peak,
        "line_mean_above_continuum": mean,
        "line_peak_snr": float(peak / sigma) if np.isfinite(sigma) and sigma > 0 else np.nan,
        "line_mean_snr": float(mean / sigma) if np.isfinite(sigma) and sigma > 0 else np.nan,
        "line_peak_wave_A": peak_wave,
        "n_line_channels": int(np.sum(line_mask)),
        "n_cont_channels": int(np.sum(cont_mask)),
    }


def empirical_percentile(value, reference_values):
    ref = np.asarray(reference_values, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0 or not np.isfinite(value):
        return np.nan
    return float(100.0 * np.mean(ref <= value))


def _stage_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    return {
        "paths": paths,
        "plot_dir": paths.plot_stage_dir("stage07b_halpha_robustness"),
        "summary_csv": paths.table_dir / "stage07b_halpha_robustness_summary.csv",
        "control_csv": paths.table_dir / "stage07b_halpha_control_metrics.csv",
        "qc_json": paths.stage_dir / "stage07b_halpha_robustness_qc.json",
        "stage04b_stack_fits": paths.stage_dir / "stage04b_local_surface_cube_stack.fits",
        "input_cube_fits": paths.stage_dir / "cube_input_local_object.fits",
        "residual_cube_fits": paths.stage_dir / "cube_residual_local_object.fits",
        "stage04b_qc_json": paths.stage_dir / "stage04b_qc.json",
        "good_wave_mask_npy": paths.stage_dir / "stage04b_good_wavelength_mask.npy",
        "bad_wave_mask_npy": paths.stage_dir / "stage04b_bad_wavelength_mask.npy",
    }


def compute_stage07b_products(cube, waves, good_wave_mask, config, stage04b_qc=None):
    """Compute Stage 07b spectra, metrics, and QC-ready payload fragments."""

    stage04b_qc = {} if stage04b_qc is None else stage04b_qc
    nz, ny, nx = cube.shape
    if nz != len(waves):
        raise RuntimeError(f"Cube spectral axis {nz} != wavelength axis {len(waves)}.")

    object_yx = resolve_object_yx(config, stage04b_qc, fallback_xy=config.get("object_xy"))
    star_yx = resolve_star_yx(config, stage04b_qc)
    line_mask, cont_mask, plot_mask = halpha_masks(
        waves,
        good_wave_mask,
        halpha_A=config["halpha_A"],
        line_integration_half_width_A=config["line_integration_half_width_A"],
        line_continuum_inner_A=config["line_continuum_inner_A"],
        line_continuum_outer_A=config["line_continuum_outer_A"],
        line_window_half_width_A=config["line_window_half_width_A"],
    )
    analysis_wave_indices = np.where(plot_mask)[0]

    control_yx = same_radius_control_positions(
        object_yx,
        star_yx,
        ny,
        nx,
        n_positions=config["control_apertures"],
        exclude_angle_deg=config["control_exclude_angle_deg"],
        margin_px=config["control_margin_px"],
    )
    if not control_yx:
        raise RuntimeError("Stage 07b could not place any same-radius control apertures.")

    object_specs, object_aperture_meta, object_fit_meta = local_surface_spectra_for_apertures(
        cube,
        object_yx,
        config["apertures"],
        analysis_wave_indices,
        config["fit_radius_px"],
        config["mask_radius_px"],
        config["local_model_kind"],
        config["local_fit_sigma_clip"],
        config["local_fit_max_iter"],
        config["local_fit_min_pixels"],
    )

    control_specs_by_aperture = {ap["name"]: [] for ap in config["apertures"]}
    control_fit_meta = []
    for pos in control_yx:
        specs_i, _, fit_i = local_surface_spectra_for_apertures(
            cube,
            pos,
            config["apertures"],
            analysis_wave_indices,
            config["fit_radius_px"],
            config["mask_radius_px"],
            config["local_model_kind"],
            config["local_fit_sigma_clip"],
            config["local_fit_max_iter"],
            config["local_fit_min_pixels"],
        )
        control_fit_meta.append(fit_i)
        for aperture in config["apertures"]:
            control_specs_by_aperture[aperture["name"]].append(specs_i[aperture["name"]])

    summary_rows = []
    control_rows = []
    aperture_products = {}
    for aperture in config["apertures"]:
        name = aperture["name"]
        object_spec = object_specs[name]
        control_specs = control_specs_by_aperture[name]
        median_control_spec = median_stack_spectrum(control_specs, analysis_wave_indices)
        object_ctrlsub_spec = object_spec - median_control_spec

        control_ctrlsub_specs = leave_one_out_control_sub(control_specs, analysis_wave_indices)
        control_metrics = [
            line_metrics(
                waves,
                spec,
                good_wave_mask,
                halpha_A=config["halpha_A"],
                line_integration_half_width_A=config["line_integration_half_width_A"],
                line_continuum_inner_A=config["line_continuum_inner_A"],
                line_continuum_outer_A=config["line_continuum_outer_A"],
                line_window_half_width_A=config["line_window_half_width_A"],
            )
            for spec in control_ctrlsub_specs
        ]
        object_metrics = line_metrics(
            waves,
            object_ctrlsub_spec,
            good_wave_mask,
            halpha_A=config["halpha_A"],
            line_integration_half_width_A=config["line_integration_half_width_A"],
            line_continuum_inner_A=config["line_continuum_inner_A"],
            line_continuum_outer_A=config["line_continuum_outer_A"],
            line_window_half_width_A=config["line_window_half_width_A"],
        )

        control_fluxes = [m["line_flux_native"] for m in control_metrics]
        control_peaks = [m["line_peak_above_continuum"] for m in control_metrics]
        control_peak_snrs = [m["line_peak_snr"] for m in control_metrics]
        aperture_meta = object_aperture_meta[name]

        row = {
            "run_id": config["run_id"],
            "aperture": name,
            "object_y": int(object_yx[0]),
            "object_x": int(object_yx[1]),
            "n_controls": int(len(control_yx)),
            "n_pix": aperture_meta["n_pix"],
            "weight_sum": aperture_meta["weight_sum"],
            "effective_n_pix": aperture_meta["effective_n_pix"],
            **{f"object_ctrlsub_{k}": v for k, v in object_metrics.items()},
            "control_flux_median": median_finite(control_fluxes),
            "control_flux_sigma": robust_sigma(control_fluxes),
            "control_peak_median": median_finite(control_peaks),
            "control_peak_sigma": robust_sigma(control_peaks),
            "control_peak_snr_median": median_finite(control_peak_snrs),
            "control_peak_snr_sigma": robust_sigma(control_peak_snrs),
            "object_empirical_z_flux": empirical_z(object_metrics["line_flux_native"], control_fluxes),
            "object_empirical_z_peak": empirical_z(object_metrics["line_peak_above_continuum"], control_peaks),
            "object_empirical_z_peak_snr": empirical_z(object_metrics["line_peak_snr"], control_peak_snrs),
            "object_flux_percentile_vs_controls": empirical_percentile(
                object_metrics["line_flux_native"],
                control_fluxes,
            ),
            "object_peak_percentile_vs_controls": empirical_percentile(
                object_metrics["line_peak_above_continuum"],
                control_peaks,
            ),
        }
        summary_rows.append(row)

        for idx, (pos, metrics_i) in enumerate(zip(control_yx, control_metrics), start=1):
            control_rows.append(
                {
                    "run_id": config["run_id"],
                    "aperture": name,
                    "control_id": idx,
                    "control_y": int(pos[0]),
                    "control_x": int(pos[1]),
                    **metrics_i,
                }
            )

        aperture_products[name] = {
            "aperture": aperture,
            "object_spec": object_spec,
            "control_specs": control_specs,
            "median_control_spec": median_control_spec,
            "object_ctrlsub_spec": object_ctrlsub_spec,
            "control_ctrlsub_specs": control_ctrlsub_specs,
            "summary": row,
            "control_metrics": control_metrics,
        }

    return {
        "object_yx": object_yx,
        "star_yx": star_yx,
        "control_yx": control_yx,
        "line_mask": line_mask,
        "cont_mask": cont_mask,
        "plot_mask": plot_mask,
        "analysis_wave_indices": analysis_wave_indices,
        "object_fit_meta": object_fit_meta,
        "control_fit_meta": control_fit_meta,
        "summary_rows": summary_rows,
        "control_rows": control_rows,
        "aperture_products": aperture_products,
    }


def save_stage07b_plots(plot_dir, run_id, waves, cube, residual_cube, products, config, show_plots=False):
    """Save Stage 07b diagnostic plots."""

    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    plot_mask = products["plot_mask"]
    line_mask = products["line_mask"]
    cont_mask = products["cont_mask"]
    aperture_products = products["aperture_products"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, aperture_product in aperture_products.items():
        spec = aperture_product["object_ctrlsub_spec"]
        row = aperture_product["summary"]
        ax.plot(
            waves[plot_mask],
            spec[plot_mask],
            lw=1.4,
            label=(
                f"{name}: peakSNR={row['object_ctrlsub_line_peak_snr']:.2f}, "
                f"z_flux={row['object_empirical_z_flux']:.2f}"
            ),
        )
    ax.axvline(config["halpha_A"], color="k", ls="--", lw=1.0, alpha=0.8)
    ax.axvspan(
        config["halpha_A"] - config["line_integration_half_width_A"],
        config["halpha_A"] + config["line_integration_half_width_A"],
        color="tab:red",
        alpha=0.08,
        label="line integration",
    )
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("Object - median local controls [native units]")
    ax.set_title(f"{run_id}: Halpha robustness across aperture choices")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = plot_dir / "stage07b_halpha_aperture_spectra.png"
    fig.savefig(path, bbox_inches="tight")
    plot_paths.append(path)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    apertures = config["apertures"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ypos = np.arange(len(apertures))
    for i, aperture in enumerate(apertures):
        name = aperture["name"]
        aperture_product = aperture_products[name]
        row = aperture_product["summary"]
        control_fluxes = np.array([m["line_flux_native"] for m in aperture_product["control_metrics"]], dtype=float)
        control_peaks = np.array(
            [m["line_peak_above_continuum"] for m in aperture_product["control_metrics"]],
            dtype=float,
        )
        axes[0].scatter(control_fluxes, np.full_like(control_fluxes, i, dtype=float), color="0.45", s=28, alpha=0.8)
        axes[0].scatter(row["object_ctrlsub_line_flux_native"], i, color="tab:red", s=58, zorder=3)
        axes[1].scatter(control_peaks, np.full_like(control_peaks, i, dtype=float), color="0.45", s=28, alpha=0.8)
        axes[1].scatter(row["object_ctrlsub_line_peak_above_continuum"], i, color="tab:red", s=58, zorder=3)

    axes[0].set_xlabel("Halpha line flux, leave-one-out controls")
    axes[1].set_xlabel("Halpha peak above continuum")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([a["name"] for a in apertures])
    axes[0].set_ylabel("Aperture")
    for ax in axes:
        ax.axvline(0, color="k", lw=0.8, alpha=0.5)
        ax.invert_yaxis()
    fig.suptitle(f"{run_id}: object (red) against same-radius controls (gray)")
    fig.tight_layout()
    path = plot_dir / "stage07b_halpha_control_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plot_paths.append(path)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    map_cube = residual_cube if residual_cube is not None else cube
    continuum_map = np.nanmedian(map_cube[cont_mask, :, :].astype(np.float64), axis=0)
    line_map = np.nansum(map_cube[line_mask, :, :].astype(np.float64) - continuum_map[None, :, :], axis=0)
    finite = np.isfinite(line_map)
    vmin, vmax = np.nanpercentile(line_map[finite], [2, 98]) if np.any(finite) else (-1, 1)
    lim = max(abs(vmin), abs(vmax))
    _, ny, nx = map_cube.shape
    object_yx = products["object_yx"]
    star_yx = products["star_yx"]
    control_yx = products["control_yx"]

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    im = ax.imshow(line_map, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.scatter([star_yx[1]], [star_yx[0]], marker="+", s=90, color="yellow", linewidths=1.8, label="star")
    ax.scatter([object_yx[1]], [object_yx[0]], marker="o", s=75, facecolors="none", edgecolors="lime", linewidths=1.8, label="object")
    if control_yx:
        ax.scatter([p[1] for p in control_yx], [p[0] for p in control_yx], marker="x", s=45, color="white", linewidths=1.4, label="controls")
    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ny - 1)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_title(f"{run_id}: residual Halpha map")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="line - continuum [native units]")
    fig.tight_layout()
    path = plot_dir / "stage07b_halpha_residual_map.png"
    fig.savefig(path, bbox_inches="tight")
    plot_paths.append(path)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    return plot_paths


def run_stage07b(
    config=None,
    *,
    show_plots=False,
    save_plots=True,
    project_root=None,
    allow_run_id_mismatch=False,
):
    """Run Stage 07b and write the same products as the original notebook."""

    if config is None:
        config = stage07b_config_from_run(
            project_root=project_root,
            allow_run_id_mismatch=allow_run_id_mismatch,
        )
    else:
        config = dict(config)

    run_id = config["run_id"]
    root = Path(project_root or config.get("project_root") or Path.cwd()).resolve()
    paths = _stage_paths(run_id, root)
    stage_paths = paths["paths"]
    stage_paths.table_dir.mkdir(parents=True, exist_ok=True)
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)

    for needed in [paths["stage04b_stack_fits"], paths["stage04b_qc_json"]]:
        if not needed.exists():
            raise FileNotFoundError(needed)

    stage04b_qc = read_json(paths["stage04b_qc_json"])
    input_cube_fits = paths["input_cube_fits"]
    if not input_cube_fits.exists():
        input_cube_fits = Path(stage04b_qc.get("input_cube_fits", input_cube_fits))
    if not input_cube_fits.exists():
        raise FileNotFoundError(input_cube_fits)

    for key in (
        "local_model_kind",
        "fit_radius_px",
        "mask_radius_px",
        "local_fit_sigma_clip",
        "local_fit_max_iter",
        "local_fit_min_pixels",
    ):
        if key in stage04b_qc:
            config[key] = stage04b_qc[key]

    config["fit_radius_px"] = float(config["fit_radius_px"])
    config["mask_radius_px"] = float(config["mask_radius_px"])
    config["local_fit_sigma_clip"] = float(config["local_fit_sigma_clip"])
    config["local_fit_max_iter"] = int(config["local_fit_max_iter"])
    config["local_fit_min_pixels"] = int(config["local_fit_min_pixels"])

    waves, good_wave_mask, bad_wave_mask = read_wavelengths_and_masks(
        paths["stage04b_stack_fits"],
        good_mask_path=paths["good_wave_mask_npy"],
        bad_mask_path=paths["bad_wave_mask_npy"],
    )

    with fits.open(input_cube_fits, memmap=True) as input_hdul:
        cube = get_cube_data(input_hdul)
        products = compute_stage07b_products(cube, waves, good_wave_mask, config, stage04b_qc)

        residual_cube = None
        if paths["residual_cube_fits"].exists() and save_plots:
            with fits.open(paths["residual_cube_fits"], memmap=True) as residual_hdul:
                residual_cube = get_cube_data(residual_hdul)
                plot_paths = save_stage07b_plots(
                    paths["plot_dir"],
                    run_id,
                    waves,
                    cube,
                    residual_cube,
                    products,
                    config,
                    show_plots=show_plots,
                )
        elif save_plots:
            plot_paths = save_stage07b_plots(
                paths["plot_dir"],
                run_id,
                waves,
                cube,
                None,
                products,
                config,
                show_plots=show_plots,
            )
        else:
            plot_paths = []

    write_csv(paths["summary_csv"], products["summary_rows"], list(products["summary_rows"][0].keys()))
    write_csv(paths["control_csv"], products["control_rows"], list(products["control_rows"][0].keys()))

    qc = {
        "run_id": run_id,
        "input_cube_fits": str(input_cube_fits),
        "residual_cube_fits_for_map": str(paths["residual_cube_fits"]),
        "stage04b_stack_fits": str(paths["stage04b_stack_fits"]),
        "object_yx": list(map(int, products["object_yx"])),
        "object_xy": [int(products["object_yx"][1]), int(products["object_yx"][0])],
        "star_yx": list(map(int, products["star_yx"])),
        "control_yx": [list(map(int, p)) for p in products["control_yx"]],
        "halpha_A": float(config["halpha_A"]),
        "line_integration_half_width_A": float(config["line_integration_half_width_A"]),
        "line_continuum_inner_A": float(config["line_continuum_inner_A"]),
        "line_continuum_outer_A": float(config["line_continuum_outer_A"]),
        "line_window_half_width_A": float(config["line_window_half_width_A"]),
        "local_model_kind": config["local_model_kind"],
        "fit_radius_px": float(config["fit_radius_px"]),
        "mask_radius_px": float(config["mask_radius_px"]),
        "local_fit_sigma_clip": float(config["local_fit_sigma_clip"]),
        "local_fit_max_iter": int(config["local_fit_max_iter"]),
        "local_fit_min_pixels": int(config["local_fit_min_pixels"]),
        "object_fit_meta": products["object_fit_meta"],
        "control_fit_meta": products["control_fit_meta"],
        "n_good_wave_channels": int(np.sum(good_wave_mask)),
        "n_bad_wave_channels": int(np.sum(bad_wave_mask)),
        "n_halpha_plot_channels": int(np.sum(products["plot_mask"])),
        "n_halpha_line_channels": int(np.sum(products["line_mask"])),
        "n_halpha_continuum_channels": int(np.sum(products["cont_mask"])),
        "apertures": config["apertures"],
        "summary_csv": str(paths["summary_csv"]),
        "control_csv": str(paths["control_csv"]),
        "plots": [str(path) for path in plot_paths],
    }
    write_json(paths["qc_json"], qc)

    return {
        "summary_csv": paths["summary_csv"],
        "control_csv": paths["control_csv"],
        "qc_json": paths["qc_json"],
        "plots": plot_paths,
        "qc": qc,
        "summary_rows": products["summary_rows"],
        "control_rows": products["control_rows"],
    }


__all__ = [
    "STAGE07B_DEFAULTS",
    "compute_stage07b_products",
    "default_apertures",
    "empirical_percentile",
    "fit_surface_model",
    "halpha_masks",
    "leave_one_out_control_sub",
    "line_metrics",
    "local_surface_spectra_for_apertures",
    "median_stack_spectrum",
    "resolve_object_yx",
    "resolve_spatial_shape",
    "resolve_star_yx",
    "run_stage07b",
    "save_stage07b_plots",
    "stage07b_config_from_run",
]
