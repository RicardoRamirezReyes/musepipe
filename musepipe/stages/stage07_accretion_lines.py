"""Stage 07: spectra and metrics for accretion-line candidates."""

from __future__ import annotations

from contextlib import ExitStack
import math
from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter

from ..apertures import box_spectrum_sum, same_radius_control_positions
from ..config import load_run_config
from ..io import get_cube_data, read_json, read_wavelength_axis, write_csv, write_json
from ..localfit import fit_local_surface_2d
from ..paths import RunPaths
from ..stats import robust_sigma
from .stage07b_halpha_robustness import resolve_object_yx, resolve_star_yx


def default_accretion_lines():
    return [
        {"name": "Hbeta", "wave_A": 4861.33, "family": "Balmer", "kind": "accretion"},
        {"name": "He I 5016", "wave_A": 5015.68, "family": "He I", "kind": "accretion/chromosphere"},
        {"name": "He I 5876", "wave_A": 5875.62, "family": "He I", "kind": "accretion"},
        {"name": "[O I] 6300", "wave_A": 6300.30, "family": "forbidden", "kind": "outflow/accretion"},
        {"name": "[O I] 6364", "wave_A": 6363.78, "family": "forbidden", "kind": "outflow/accretion"},
        {"name": "Halpha", "wave_A": 6562.80, "family": "Balmer", "kind": "accretion"},
        {"name": "He I 6678", "wave_A": 6678.15, "family": "He I", "kind": "accretion"},
        {"name": "[S II] 6716", "wave_A": 6716.44, "family": "forbidden", "kind": "outflow/accretion"},
        {"name": "[S II] 6731", "wave_A": 6730.82, "family": "forbidden", "kind": "outflow/accretion"},
        {"name": "He I 7065", "wave_A": 7065.19, "family": "He I", "kind": "accretion"},
        {"name": "Pa 18", "wave_A": 8437.96, "family": "Paschen", "kind": "accretion"},
        {"name": "O I 8446", "wave_A": 8446.36, "family": "O I", "kind": "accretion/fluorescence"},
        {"name": "Pa 17", "wave_A": 8467.25, "family": "Paschen", "kind": "accretion"},
        {"name": "Ca II 8498", "wave_A": 8498.02, "family": "Ca II IRT", "kind": "accretion"},
        {"name": "Pa 16", "wave_A": 8502.49, "family": "Paschen", "kind": "accretion"},
        {"name": "Ca II 8542", "wave_A": 8542.09, "family": "Ca II IRT", "kind": "accretion"},
        {"name": "Pa 15", "wave_A": 8545.38, "family": "Paschen", "kind": "accretion"},
        {"name": "Pa 14", "wave_A": 8598.39, "family": "Paschen", "kind": "accretion"},
        {"name": "Ca II 8662", "wave_A": 8662.14, "family": "Ca II IRT", "kind": "accretion"},
        {"name": "Pa 13", "wave_A": 8665.02, "family": "Paschen", "kind": "accretion"},
        {"name": "Pa 12", "wave_A": 8750.47, "family": "Paschen", "kind": "accretion"},
        {"name": "Pa 11", "wave_A": 8862.78, "family": "Paschen", "kind": "accretion"},
        {"name": "Pa 10", "wave_A": 9014.91, "family": "Paschen", "kind": "accretion"},
        {"name": "Pa 9", "wave_A": 9229.01, "family": "Paschen", "kind": "accretion"},
    ]


LINE_COLORS = {
    "Balmer": "tab:red",
    "He I": "tab:purple",
    "forbidden": "tab:green",
    "O I": "tab:olive",
    "Ca II IRT": "tab:orange",
    "Paschen": "tab:blue",
}

STAGE07_DEFAULTS = {
    "integrated_box_size": 3,
    "use_local_control_apertures": True,
    "control_apertures": 8,
    "control_exclude_angle_deg": 25.0,
    "control_margin_px": 1,
    "full_continuum_filter_A": 80.0,
    "line_window_half_width_A": 18.0,
    "line_integration_half_width_A": 2.5,
    "line_continuum_inner_A": 6.0,
    "line_continuum_outer_A": 16.0,
    "min_continuum_pixels": 4,
    "exclude_bad_line_candidates": True,
    "local_model_kind": "plane",
    "fit_radius_px": 12.0,
    "mask_radius_px": 3.0,
    "local_fit_sigma_clip": 3.0,
    "local_fit_max_iter": 3,
    "local_fit_min_pixels": 30,
}


def _stage_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage07_accretion_lines")
    return {
        "paths": paths,
        "plot_dir": plot_dir,
        "line_plot_dir": plot_dir / "line_windows",
        "metrics_csv": paths.table_dir / "stage07_accretion_line_metrics.csv",
        "qc_json": paths.stage_dir / "stage07_accretion_line_qc.json",
        "star_stage02_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "star_stage01_fits": paths.stage_dir / "stage01_cropped_cube_stack.fits",
        "object_residual_fits": paths.stage_dir / "cube_residual_local_object.fits",
        "object_residual_fallback_fits": paths.stage_dir / "cube_residual_best4.fits",
        "control_input_fits": paths.stage_dir / "cube_input_local_object.fits",
        "stage04b_qc_json": paths.stage_dir / "stage04b_qc.json",
    }


def stage07_config_from_run(
    run_id=None,
    *,
    project_root=None,
    overrides=None,
    allow_run_id_mismatch=False,
):
    """Build Stage 07 configuration from the active run and Stage 04b QC."""

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
        "target_name": cfg.get("target_name"),
        "object_yx": tuple(map(int, object_yx)),
        "star_yx": tuple(map(int, star_yx)),
        "star_spectrum_yx": tuple(map(int, star_yx)),
        "star_cube_index": 0,
        "object_name": cfg.get("stage07_object_name", f"{cfg.get('target_name', 'science')} object"),
        "object_source_label": "04b local object residual",
        "bad_wavelength_ranges_A": qc04b.get("bad_wavelength_ranges_A", []),
        "accretion_lines": default_accretion_lines(),
        **STAGE07_DEFAULTS,
    }
    for key, default in STAGE07_DEFAULTS.items():
        config[key] = cfg.get(f"stage07_{key}", cfg.get(key, default))
    for key in (
        "local_model_kind",
        "fit_radius_px",
        "mask_radius_px",
        "local_fit_sigma_clip",
        "local_fit_max_iter",
        "local_fit_min_pixels",
    ):
        if key in qc04b:
            config[key] = qc04b[key]
    if cfg.get("stage07_star_spectrum_yx") is not None:
        config["star_spectrum_yx"] = tuple(map(int, cfg["stage07_star_spectrum_yx"]))
    if cfg.get("stage07_accretion_lines") is not None:
        config["accretion_lines"] = [dict(line) for line in cfg["stage07_accretion_lines"]]
    if overrides:
        config.update(overrides)

    box_size = int(config["integrated_box_size"])
    if box_size < 1 or box_size % 2 == 0:
        raise ValueError("stage07_integrated_box_size must be an odd positive integer.")
    return config


def _validate_center(yx, shape, label, half=0):
    y, x = map(int, yx)
    ny, nx = map(int, shape[-2:])
    if not (half <= y < ny - half and half <= x < nx - half):
        raise ValueError(f"{label} center {(y, x)} is outside spatial shape {(ny, nx)}.")


def pixel_spectrum(cube, yx):
    _validate_center(yx, cube.shape, "pixel")
    y, x = map(int, yx)
    return np.asarray(cube[:, y, x], dtype=np.float64)


def box_spectrum(cube, yx, box_size):
    half = int(box_size) // 2
    _validate_center(yx, cube.shape, "box", half=half)
    return box_spectrum_sum(cube, int(yx[0]), int(yx[1]), box_size=box_size)


def extract_local_surface_box_spectrum(cube, yx, config, extra_exclusion_yx=None):
    """Extract a robust local-surface-subtracted box spectrum."""

    box_size = int(config["integrated_box_size"])
    half = box_size // 2
    _validate_center(yx, cube.shape, "local box", half=half)
    y, x = map(int, yx)
    nz = int(cube.shape[0])
    spec = np.full(nz, np.nan, dtype=np.float64)
    nfit = np.zeros(nz, dtype=int)
    for k in range(nz):
        model, n_good = fit_local_surface_2d(
            cube[k],
            y,
            x,
            fit_radius_px=config["fit_radius_px"],
            mask_radius_px=config["mask_radius_px"],
            model_kind=config["local_model_kind"],
            extra_exclusion_yx=extra_exclusion_yx,
            sigma_clip=config["local_fit_sigma_clip"],
            max_iter=config["local_fit_max_iter"],
            min_fit_pixels=config["local_fit_min_pixels"],
        )
        y1, y2 = y - half, y + half + 1
        x1, x2 = x - half, x + half + 1
        residual = np.asarray(cube[k, y1:y2, x1:x2], dtype=np.float64) - model[y1:y2, x1:x2]
        if np.any(np.isfinite(residual)):
            spec[k] = float(np.nansum(residual))
        nfit[k] = int(n_good)
    return spec, {
        "y": y,
        "x": x,
        "box_size": box_size,
        "n_pix": box_size**2,
        "nfit_median": float(np.nanmedian(nfit)),
        "nfit_min": int(np.min(nfit)),
    }


def continuum_model(spec, waves, width_A=80.0):
    spec = np.asarray(spec, dtype=np.float64)
    waves = np.asarray(waves, dtype=np.float64)
    dw = float(np.nanmedian(np.diff(waves)))
    if not np.isfinite(dw) or dw == 0:
        raise ValueError("Wavelength axis must have a finite non-zero spacing.")
    size = max(5, int(round(float(width_A) / abs(dw))))
    if size % 2 == 0:
        size += 1
    finite = np.isfinite(spec)
    fill = float(np.nanmedian(spec[finite])) if np.any(finite) else 0.0
    values = spec.copy()
    values[~finite] = fill
    return median_filter(values, size=size, mode="nearest").astype(np.float64)


def continuum_normalized(spec, waves, width_A=80.0):
    cont = continuum_model(spec, waves, width_A=width_A)
    scale = float(np.nanmedian(np.abs(cont[np.isfinite(cont)])))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    safe = np.where(np.abs(cont) > 1e-8 * scale, cont, np.nan)
    normalized = np.asarray(spec, dtype=np.float64) / safe - 1.0
    if np.all(~np.isfinite(normalized)):
        normalized = np.asarray(spec, dtype=np.float64) - np.nanmedian(spec)
    return normalized, cont


def residual_snr_like(spec, waves, width_A=80.0):
    cont = continuum_model(spec, waves, width_A=width_A)
    residual = np.asarray(spec, dtype=np.float64) - cont
    sigma = robust_sigma(residual)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return residual / sigma, cont, float(sigma)


def filter_lines_in_range(lines, waves, edge_A=3.0):
    wmin = float(np.nanmin(waves))
    wmax = float(np.nanmax(waves))
    return [dict(line) for line in lines if wmin + edge_A <= float(line["wave_A"]) <= wmax - edge_A]


def wavelength_in_bad_ranges(wave_A, bad_ranges_A):
    return any(float(start) <= float(wave_A) <= float(end) for start, end in bad_ranges_A)


def line_metrics(
    waves,
    spec,
    center_A,
    *,
    line_half_A=2.5,
    cont_inner_A=6.0,
    cont_outer_A=16.0,
    min_continuum_pixels=4,
):
    waves = np.asarray(waves, dtype=np.float64)
    spec = np.asarray(spec, dtype=np.float64)
    distance = np.abs(waves - float(center_A))
    line_mask = (distance <= float(line_half_A)) & np.isfinite(spec)
    cont_mask = (
        (distance >= float(cont_inner_A))
        & (distance <= float(cont_outer_A))
        & np.isfinite(spec)
    )
    finite = np.isfinite(spec)
    sample = spec[cont_mask] if int(np.sum(cont_mask)) >= int(min_continuum_pixels) else spec[finite]
    continuum = float(np.nanmedian(sample)) if sample.size else np.nan
    sigma = robust_sigma(sample - continuum) if sample.size else np.nan
    signal = spec[line_mask] - continuum
    dw = float(np.nanmedian(np.diff(waves)))
    peak = float(np.nanmax(signal)) if signal.size else np.nan
    mean = float(np.nanmean(signal)) if signal.size else np.nan
    return {
        "continuum_median": continuum,
        "continuum_sigma": float(sigma) if np.isfinite(sigma) else np.nan,
        "line_flux_native": float(np.nansum(signal) * dw) if signal.size else np.nan,
        "line_peak_above_continuum": peak,
        "line_mean_above_continuum": mean,
        "line_peak_snr": float(peak / sigma) if np.isfinite(sigma) and sigma > 0 else np.nan,
        "line_mean_snr": float(mean / sigma) if np.isfinite(sigma) and sigma > 0 else np.nan,
        "n_line_channels": int(np.sum(line_mask)),
        "n_cont_channels": int(np.sum(cont_mask)),
    }


def safe_name(text):
    return "".join(char if char.isalnum() else "_" for char in text).strip("_").lower()


def _axes_match(first, second, tolerance_A=0.01):
    first = np.asarray(first)
    second = np.asarray(second)
    return first.shape == second.shape and float(np.nanmax(np.abs(first - second))) <= tolerance_A


def compute_stage07_products(
    star_cube,
    star_waves,
    object_cube,
    object_waves,
    config,
    *,
    control_input_cube=None,
    control_waves=None,
):
    """Compute spectra, control subtraction, and per-line metric rows."""

    star_cube = np.asarray(star_cube)
    object_cube = np.asarray(object_cube)
    star_waves = np.asarray(star_waves, dtype=np.float64)
    object_waves = np.asarray(object_waves, dtype=np.float64)
    if star_cube.ndim != 3 or object_cube.ndim != 3:
        raise ValueError("Stage 07 expects 3D star and object cubes.")
    if star_cube.shape[0] != star_waves.size or object_cube.shape[0] != object_waves.size:
        raise ValueError("Cube spectral dimensions must match their wavelength axes.")
    if not _axes_match(star_waves, object_waves):
        warnings.warn("Star and object wavelength axes differ; each spectrum keeps its native axis.")

    box_size = int(config["integrated_box_size"])
    star_yx = tuple(map(int, config["star_yx"]))
    star_spectrum_yx = tuple(map(int, config.get("star_spectrum_yx", star_yx)))
    object_yx = tuple(map(int, config["object_yx"]))
    _validate_center(star_yx, object_cube.shape, "geometry star")

    spectra = {
        "star_center": (star_waves, pixel_spectrum(star_cube, star_spectrum_yx)),
        "star_box": (star_waves, box_spectrum(star_cube, star_spectrum_yx, box_size)),
        "object_center": (object_waves, pixel_spectrum(object_cube, object_yx)),
        "object_box": (object_waves, box_spectrum(object_cube, object_yx, box_size)),
    }
    control_positions = []
    control_specs = []
    control_meta = []
    control_median = None
    object_ctrlsub = None

    if bool(config["use_local_control_apertures"]) and control_input_cube is not None:
        control_input_cube = np.asarray(control_input_cube)
        control_waves = object_waves if control_waves is None else np.asarray(control_waves, dtype=np.float64)
        if control_input_cube.shape[0] != control_waves.size or not _axes_match(control_waves, object_waves):
            raise RuntimeError("Control-input and object wavelength axes differ.")
        _, ny, nx = control_input_cube.shape
        margin = max(box_size // 2, int(config["control_margin_px"]))
        control_positions = same_radius_control_positions(
            object_yx,
            star_yx,
            ny,
            nx,
            n_positions=config["control_apertures"],
            exclude_angle_deg=config["control_exclude_angle_deg"],
            margin_px=margin,
        )
        for position in control_positions:
            spec, meta = extract_local_surface_box_spectrum(
                control_input_cube,
                position,
                config,
                extra_exclusion_yx=[object_yx, star_yx],
            )
            control_specs.append(spec)
            control_meta.append(meta)
        if control_specs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                control_median = np.nanmedian(np.vstack(control_specs), axis=0)
            object_ctrlsub = spectra["object_box"][1] - control_median
            spectra["object_box_ctrlsub"] = (object_waves, object_ctrlsub)

    lines_raw = filter_lines_in_range(config["accretion_lines"], star_waves)
    bad_ranges = config.get("bad_wavelength_ranges_A", [])
    if config["exclude_bad_line_candidates"]:
        lines_excluded = [line for line in lines_raw if wavelength_in_bad_ranges(line["wave_A"], bad_ranges)]
        lines = [line for line in lines_raw if not wavelength_in_bad_ranges(line["wave_A"], bad_ranges)]
    else:
        lines_excluded = []
        lines = lines_raw

    rows = []
    for line in lines:
        row = {
            "line_name": line["name"],
            "wave_A": float(line["wave_A"]),
            "family": line["family"],
            "kind": line["kind"],
            "star_y": int(star_spectrum_yx[0]),
            "star_x": int(star_spectrum_yx[1]),
            "star_box_size_px": box_size,
            "object_y": int(object_yx[0]),
            "object_x": int(object_yx[1]),
            "object_box_size_px": box_size,
        }
        for prefix, (waves_i, spec_i) in spectra.items():
            metrics = line_metrics(
                waves_i,
                spec_i,
                line["wave_A"],
                line_half_A=config["line_integration_half_width_A"],
                cont_inner_A=config["line_continuum_inner_A"],
                cont_outer_A=config["line_continuum_outer_A"],
                min_continuum_pixels=config["min_continuum_pixels"],
            )
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        rows.append(row)

    normalized = {}
    for name in ("star_center", "star_box"):
        waves_i, spec_i = spectra[name]
        normalized[name], _ = continuum_normalized(spec_i, waves_i, config["full_continuum_filter_A"])
    for name in ("object_center", "object_box", "object_box_ctrlsub"):
        if name in spectra:
            waves_i, spec_i = spectra[name]
            normalized[name], _, _ = residual_snr_like(spec_i, waves_i, config["full_continuum_filter_A"])

    return {
        "star_yx": star_yx,
        "star_spectrum_yx": star_spectrum_yx,
        "object_yx": object_yx,
        "spectra": spectra,
        "normalized": normalized,
        "control_positions_yx": control_positions,
        "control_specs": control_specs,
        "control_median_spec": control_median,
        "control_meta": control_meta,
        "lines_in_range": lines,
        "lines_excluded_bad": lines_excluded,
        "metrics_rows": rows,
    }


def _mark_lines(ax, lines):
    ymin, ymax = ax.get_ylim()
    height = ymax - ymin
    for line in lines:
        wave = float(line["wave_A"])
        color = LINE_COLORS.get(line["family"], "tab:gray")
        ax.axvline(wave, color=color, lw=0.9, alpha=0.7)
        ax.text(wave, ymax - 0.04 * height, line["name"], rotation=90, va="top", ha="right", fontsize=7, color=color)


def _clear_managed_line_plots(line_dir):
    """Remove obsolete Stage 07 line-window plots before regenerating them."""

    line_dir = Path(line_dir)
    line_dir.mkdir(parents=True, exist_ok=True)
    for path in line_dir.glob("stage07_line_*.png"):
        path.unlink()


def save_stage07_plots(plot_dir, run_id, products, config, show_plots=False):
    """Save the four legacy summaries and one window plot per retained line."""

    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    line_dir = plot_dir / "line_windows"
    _clear_managed_line_plots(line_dir)
    spectra = products["spectra"]
    normalized = products["normalized"]
    lines = products["lines_in_range"]
    object_yx = products["object_yx"]
    box_size = int(config["integrated_box_size"])
    saved = {}

    def finish(fig, name):
        path = plot_dir / name
        fig.savefig(path, dpi=180)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        return path

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, constrained_layout=True)
    for name, color, label in (("star_center", "0.55", "central pixel"), ("star_box", "black", f"integrated {box_size}x{box_size}")):
        axes[0].plot(spectra[name][0], normalized[name], color=color, lw=0.8, label=label)
    for name, color, label in (("object_center", "0.55", "central pixel"), ("object_box", "black", f"integrated {box_size}x{box_size}"), ("object_box_ctrlsub", "tab:red", "box - median local controls")):
        if name in spectra:
            axes[1].plot(spectra[name][0], normalized[name], color=color, lw=0.8, label=label)
    axes[0].set_ylabel("Star: continuum normalized")
    axes[1].set_ylabel("Object: residual sigma-like")
    axes[1].set_xlabel("Wavelength [Angstrom]")
    axes[0].set_title(f"{run_id}: central star accretion-line candidates")
    axes[1].set_title(f"{run_id}: local residual at (y,x)={object_yx}")
    for ax in axes:
        ax.axhline(0, color="0.6", lw=0.8)
        _mark_lines(ax, lines)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    saved["full_spectrum_figure"] = finish(fig, "stage07_full_accretion_line_spectra.png")

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, constrained_layout=True)
    axes[0].plot(spectra["star_box"][0], normalized["star_box"], color="black", lw=0.9, label=f"integrated {box_size}x{box_size}")
    axes[1].plot(spectra["object_box"][0], normalized["object_box"], color="black", lw=0.9, label=f"integrated {box_size}x{box_size}")
    if "object_box_ctrlsub" in spectra:
        axes[1].plot(spectra["object_box_ctrlsub"][0], normalized["object_box_ctrlsub"], color="tab:red", lw=0.8, label="box - median local controls")
    axes[0].set_ylabel("Star: continuum normalized")
    axes[1].set_ylabel("Object: residual sigma-like")
    axes[1].set_xlabel("Wavelength [Angstrom]")
    for ax in axes:
        ax.axhline(0, color="0.6", lw=0.8)
        _mark_lines(ax, lines)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    saved["full_integrated_box_figure"] = finish(fig, "stage07_full_integrated_box_spectra.png")

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, constrained_layout=True)
    for name, color in (("star_center", "0.55"), ("star_box", "black")):
        axes[0].plot(*spectra[name], color=color, lw=0.8, label=name)
    for name, color in (("object_center", "0.55"), ("object_box", "black"), ("object_box_ctrlsub", "tab:red")):
        if name in spectra:
            axes[1].plot(*spectra[name], color=color, lw=0.8, label=name)
    axes[0].set_ylabel("Star native units")
    axes[1].set_ylabel("Object residual native units")
    axes[1].set_xlabel("Wavelength [Angstrom]")
    for ax in axes:
        _mark_lines(ax, lines)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    saved["full_native_units_figure"] = finish(fig, "stage07_full_native_units_spectra.png")

    line_paths = []
    for line in lines:
        center = float(line["wave_A"])
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
        for name, color in (("star_center", "0.55"), ("star_box", "black")):
            waves, _ = spectra[name]
            window = np.abs(waves - center) <= config["line_window_half_width_A"]
            axes[0].plot(waves[window], normalized[name][window], color=color, lw=1.1, label=name)
        for name, color in (("object_center", "0.55"), ("object_box", "black"), ("object_box_ctrlsub", "tab:red")):
            if name in spectra:
                waves, _ = spectra[name]
                window = np.abs(waves - center) <= config["line_window_half_width_A"]
                axes[1].plot(waves[window], normalized[name][window], color=color, lw=1.1, label=name)
        for ax in axes:
            ax.axvline(center, color=LINE_COLORS.get(line["family"], "tab:gray"), lw=1.4)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="best")
        axes[0].set_ylabel("Star norm.")
        axes[1].set_ylabel("Object sigma-like")
        axes[1].set_xlabel("Wavelength [Angstrom]")
        axes[0].set_title(f"{line['name']} | {line['kind']}")
        path = line_dir / f"stage07_line_{safe_name(line['name'])}_{center:.1f}A.png"
        fig.savefig(path, dpi=180)
        line_paths.append(path)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    saved["individual_line_figures"] = line_paths

    ncols = 3
    nrows = max(1, int(math.ceil(len(lines) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(3.0 * nrows, 5)), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, line in zip(axes, lines):
        center = float(line["wave_A"])
        for name, color in (("object_center", "0.55"), ("object_box", "black"), ("object_box_ctrlsub", "tab:red")):
            if name in spectra:
                waves, _ = spectra[name]
                window = np.abs(waves - center) <= config["line_window_half_width_A"]
                ax.plot(waves[window], normalized[name][window], color=color, lw=0.9, label=name)
        ax.axvline(center, color=LINE_COLORS.get(line["family"], "tab:gray"), lw=1.2)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(f"{line['name']} ({center:.1f} A)", fontsize=9)
        ax.grid(alpha=0.22)
    for ax in axes[len(lines):]:
        ax.axis("off")
    fig.suptitle(f"{run_id}: candidate accretion-line windows", fontsize=14)
    saved["line_grid_figure"] = finish(fig, "stage07_accretion_line_windows_grid.png")
    return saved


def write_stage07_products(config, products, *, project_root=None, save_plots=True, show_plots=False):
    """Write Stage 07 metrics, QC, and optional figures."""

    root = Path(project_root or config.get("project_root") or Path.cwd()).resolve()
    outputs = _stage_paths(config["run_id"], root)
    outputs["paths"].table_dir.mkdir(parents=True, exist_ok=True)
    outputs["paths"].stage_dir.mkdir(parents=True, exist_ok=True)
    rows = products["metrics_rows"]
    fieldnames = list(rows[0]) if rows else ["line_name", "wave_A", "family", "kind"]
    write_csv(outputs["metrics_csv"], rows, fieldnames)

    plot_paths = {}
    if save_plots:
        outputs["plot_dir"].mkdir(parents=True, exist_ok=True)
        plot_paths = save_stage07_plots(outputs["plot_dir"], config["run_id"], products, config, show_plots=show_plots)

    qc = {
        "run_id": config["run_id"],
        "target_name": config.get("target_name"),
        "stage": "stage07_accretion_line_spectra",
        "star_cube_fits": str(config.get("star_cube_fits", "")),
        "object_cube_fits": str(config.get("object_cube_fits", "")),
        "object_source_label": config["object_source_label"],
        "integrated_box_size_px": int(config["integrated_box_size"]),
        "star_yx": list(map(int, products["star_yx"])),
        "star_spectrum_yx": list(map(int, products["star_spectrum_yx"])),
        "object_yx": list(map(int, products["object_yx"])),
        "use_local_control_apertures": bool(config["use_local_control_apertures"]),
        "local_control_input_fits": str(config.get("control_input_fits", "")),
        "local_control_positions_yx": [list(map(int, pos)) for pos in products["control_positions_yx"]],
        "local_control_box_meta": products["control_meta"],
        "line_window_half_width_A": float(config["line_window_half_width_A"]),
        "line_integration_half_width_A": float(config["line_integration_half_width_A"]),
        "bad_wavelength_ranges_A": config.get("bad_wavelength_ranges_A", []),
        "exclude_bad_line_candidates": bool(config["exclude_bad_line_candidates"]),
        "lines_excluded_bad": products["lines_excluded_bad"],
        "lines_in_range": products["lines_in_range"],
        "metrics_csv": str(outputs["metrics_csv"]),
        "metrics_note": "Per-line metrics use star_center, star_box, object_center, object_box, and object_box_ctrlsub prefixes when controls are available.",
    }
    for key, value in plot_paths.items():
        if isinstance(value, list):
            qc[key] = [str(path) for path in value]
        else:
            qc[key] = str(value)
    write_json(outputs["qc_json"], qc)
    return {
        "metrics_csv": outputs["metrics_csv"],
        "qc_json": outputs["qc_json"],
        "plot_paths": plot_paths,
        "metrics_rows": rows,
        "qc": qc,
    }


def _waves_or_fallback(hdul, cube, fallback):
    try:
        return read_wavelength_axis(hdul, data_shape=cube.shape)
    except RuntimeError:
        if fallback is None or len(fallback) != cube.shape[0]:
            raise
        return np.asarray(fallback, dtype=np.float64)


def run_stage07(
    config=None,
    *,
    show_plots=False,
    save_plots=True,
    project_root=None,
    allow_run_id_mismatch=False,
):
    """Run Stage 07 using the active run and write its legacy products."""

    if config is None:
        config = stage07_config_from_run(
            project_root=project_root,
            allow_run_id_mismatch=allow_run_id_mismatch,
        )
    else:
        config = dict(config)
    root = Path(project_root or config.get("project_root") or Path.cwd()).resolve()
    outputs = _stage_paths(config["run_id"], root)
    if not outputs["stage04b_qc_json"].exists():
        raise FileNotFoundError(outputs["stage04b_qc_json"])

    star_path = outputs["star_stage02_fits"] if outputs["star_stage02_fits"].exists() else outputs["star_stage01_fits"]
    object_path = outputs["object_residual_fits"] if outputs["object_residual_fits"].exists() else outputs["object_residual_fallback_fits"]
    for path in (star_path, object_path):
        if not path.exists():
            raise FileNotFoundError(path)
    control_path = outputs["control_input_fits"]

    config["star_cube_fits"] = str(star_path)
    config["object_cube_fits"] = str(object_path)
    config["control_input_fits"] = str(control_path)

    with ExitStack() as stack:
        star_hdul = stack.enter_context(fits.open(star_path, memmap=True))
        object_hdul = stack.enter_context(fits.open(object_path, memmap=True))
        star_cube = get_cube_data(star_hdul, cube_index=config.get("star_cube_index", 0))
        object_cube = get_cube_data(object_hdul)
        star_waves = read_wavelength_axis(star_hdul, data_shape=star_cube.shape)
        object_waves = _waves_or_fallback(object_hdul, object_cube, star_waves)

        control_cube = None
        control_waves = None
        if bool(config["use_local_control_apertures"]) and control_path.exists():
            control_hdul = stack.enter_context(fits.open(control_path, memmap=True))
            control_cube = get_cube_data(control_hdul)
            control_waves = _waves_or_fallback(control_hdul, control_cube, object_waves)

        products = compute_stage07_products(
            star_cube,
            star_waves,
            object_cube,
            object_waves,
            config,
            control_input_cube=control_cube,
            control_waves=control_waves,
        )
    return write_stage07_products(
        config,
        products,
        project_root=root,
        save_plots=save_plots,
        show_plots=show_plots,
    )


__all__ = [
    "STAGE07_DEFAULTS",
    "compute_stage07_products",
    "default_accretion_lines",
    "filter_lines_in_range",
    "line_metrics",
    "run_stage07",
    "save_stage07_plots",
    "stage07_config_from_run",
    "wavelength_in_bad_ranges",
    "write_stage07_products",
]
