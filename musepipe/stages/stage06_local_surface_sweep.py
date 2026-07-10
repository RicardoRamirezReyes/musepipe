"""C2 multi-wavelength and multi-position local-surface injection sweep."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from ..config import load_run_config
from ..io import read_json, write_csv, write_json
from ..paths import RunPaths
from .stage04b_local_surface import load_stage04b_input, stage04b_config_from_run, stage04b_paths
from .stage06_local_surface_injection import (
    _selected_cube_indices,
    compute_stage06_local_products,
    stage06_local_config_from_run,
    stage06_local_paths,
)


STAGE06_LOCAL_SWEEP_DEFAULTS = {
    "sweep_lines": [
        {"label": "Hbeta", "center_A": 4861.33},
        {"label": "Halpha", "center_A": 6562.80},
        {"label": "OI_8446", "center_A": 8446.36},
    ],
    "sweep_pa_offsets_deg": [90.0, 180.0, 270.0],
    "sweep_n_line_channels": 3,
    "sweep_snr_grid": [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    "sweep_continuum_inner_A": 10.0,
    "sweep_continuum_outer_A": 60.0,
    "sweep_detection_threshold_snr": 5.0,
    "sweep_isolate_lines": True,
}


def nearest_line_channel_targets(wavelengths, center_A, n_channels=3):
    """Return wavelengths of the closest distinct channels to one line center."""

    waves = np.asarray(wavelengths, dtype=np.float64)
    finite_indices = np.where(np.isfinite(waves))[0]
    n_channels = int(n_channels)
    if n_channels <= 0 or finite_indices.size < n_channels:
        raise ValueError("Not enough finite wavelength channels for the requested line aperture.")
    order = np.argsort(np.abs(waves[finite_indices] - float(center_A)), kind="stable")[:n_channels]
    indices = np.sort(finite_indices[order])
    return [float(value) for value in waves[indices]]


def symmetric_continuum_windows(center_A, inner_A=10.0, outer_A=60.0):
    """Return two continuum sidebands around a line center."""

    center = float(center_A)
    inner = float(inner_A)
    outer = float(outer_A)
    if not 0 < inner < outer:
        raise ValueError("Continuum widths must satisfy 0 < inner_A < outer_A.")
    return [(center - outer, center - inner), (center + inner, center + outer)]


def _line_overlaps_bad_range(center_A, bad_ranges):
    center = float(center_A)
    return any(float(lo) <= center <= float(hi) for lo, hi in bad_ranges)


def _line_case_prefix(label, center_A):
    safe_label = "".join(char.lower() if char.isalnum() else "_" for char in str(label)).strip("_")
    safe_center = f"{float(center_A):.2f}".replace(".", "p")
    return f"{safe_label or 'line'}_{safe_center}"


def resolve_sweep_cases(config, wavelengths):
    """Expand configured line and PA grids into concrete Stage 06 cases."""

    waves = np.asarray(wavelengths, dtype=np.float64)
    wave_min = float(np.nanmin(waves))
    wave_max = float(np.nanmax(waves))
    bad_ranges = config.get("bad_wavelength_ranges_A", [])
    cases = []
    for line_index, line in enumerate(config["sweep_lines"], start=1):
        label = str(line["label"])
        center_A = float(line["center_A"])
        case_prefix = _line_case_prefix(label, center_A)
        if not wave_min <= center_A <= wave_max:
            raise ValueError(f"Sweep line {label} at {center_A} A is outside the wavelength axis.")
        if _line_overlaps_bad_range(center_A, bad_ranges):
            raise ValueError(f"Sweep line {label} at {center_A} A lies inside a bad wavelength range.")
        line_channels_A = line.get("line_channels_A")
        if line_channels_A is None:
            line_channels_A = nearest_line_channel_targets(
                waves,
                center_A,
                n_channels=line.get("n_line_channels", config["sweep_n_line_channels"]),
            )
        continuum_windows_A = line.get("continuum_windows_A")
        if continuum_windows_A is None:
            continuum_windows_A = symmetric_continuum_windows(
                center_A,
                line.get("continuum_inner_A", config["sweep_continuum_inner_A"]),
                line.get("continuum_outer_A", config["sweep_continuum_outer_A"]),
            )
        for position_index, pa_offset_deg in enumerate(config["sweep_pa_offsets_deg"], start=1):
            case_config = dict(config)
            case_config.update(
                {
                    "line_label": label,
                    "line_center_A": center_A,
                    "line_channels_A": list(map(float, line_channels_A)),
                    "continuum_windows_A": [list(map(float, value)) for value in continuum_windows_A],
                    "line_fwhm_A": line.get("line_fwhm_A"),
                    "spectral_resolution": line.get(
                        "spectral_resolution", config.get("spectral_resolution")
                    ),
                    "injection_yx": None,
                    "match_reference_separation": True,
                    "injection_pa_offset_deg": float(pa_offset_deg),
                    "injection_snr_grid": config.get(
                        "sweep_snr_grid", config["injection_snr_grid"]
                    ),
                }
            )
            cases.append(
                {
                    "case_id": f"{case_prefix}_pa{position_index:02d}",
                    "line_index": line_index,
                    "position_index": position_index,
                    "line_label": label,
                    "line_center_A": center_A,
                    "pa_offset_deg": float(pa_offset_deg),
                    "config": case_config,
                }
            )
    if not cases:
        raise ValueError("The Stage 06 local C2 sweep contains no cases.")
    return cases


def _finite_percentiles(values):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(float(value) for value in np.nanpercentile(vals, [16, 50, 84]))


def completeness_rows_from_grid(grid_rows, detection_threshold_snr):
    """Summarize detection fractions and recovered S/N at every input level."""

    completeness_rows = []
    target_values = sorted({float(row["target_snr_input"]) for row in grid_rows})
    for target_snr in target_values:
        rows = [row for row in grid_rows if np.isclose(row["target_snr_input"], target_snr)]
        matched = np.asarray([row["local_surface_matched_snr"] for row in rows], dtype=float)
        aperture = np.asarray([row["local_surface_aperture_snr"] for row in rows], dtype=float)
        matched_p16, matched_p50, matched_p84 = _finite_percentiles(matched)
        aperture_p16, aperture_p50, aperture_p84 = _finite_percentiles(aperture)
        completeness_rows.append(
            {
                "target_snr_input": target_snr,
                "n_trials": len(rows),
                "detection_threshold_snr": float(detection_threshold_snr),
                "matched_completeness": float(np.mean(matched >= float(detection_threshold_snr))),
                "aperture_completeness": float(np.mean(aperture >= float(detection_threshold_snr))),
                "matched_recovered_snr_p16": matched_p16,
                "matched_recovered_snr_median": matched_p50,
                "matched_recovered_snr_p84": matched_p84,
                "aperture_recovered_snr_p16": aperture_p16,
                "aperture_recovered_snr_median": aperture_p50,
                "aperture_recovered_snr_p84": aperture_p84,
            }
        )
    return completeness_rows


def input_snr_at_completeness(completeness_rows, metric, target_fraction):
    """Interpolate the input S/N where a monotonic completeness envelope reaches a target."""

    x = np.asarray([row["target_snr_input"] for row in completeness_rows], dtype=np.float64)
    y = np.asarray([row[metric] for row in completeness_rows], dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = np.maximum.accumulate(y[order])
    target = float(target_fraction)
    if x.size == 0 or not np.any(np.isfinite(y)) or np.nanmax(y) < target:
        return np.nan
    index = int(np.where(y >= target)[0][0])
    if index == 0 or y[index] == y[index - 1]:
        return float(x[index])
    fraction = (target - y[index - 1]) / (y[index] - y[index - 1])
    return float(x[index - 1] + fraction * (x[index] - x[index - 1]))


def summarize_sweep(case_results, detection_threshold_snr):
    """Build flat grid, per-case, and completeness rows from sweep products."""

    grid_rows = []
    case_rows = []
    for case, products in case_results:
        geometry = products["geometry"]
        positive_rows = [row for row in products["rows"] if row["target_snr_input"] > 0]
        matched_transfer = [row["delta_matched_signal_transfer"] for row in positive_rows]
        aperture_transfer = [row["delta_aperture_signal_transfer"] for row in positive_rows]
        matched_p16, matched_p50, matched_p84 = _finite_percentiles(matched_transfer)
        aperture_p16, aperture_p50, aperture_p84 = _finite_percentiles(aperture_transfer)
        for row in products["rows"]:
            grid_rows.append(
                {
                    "case_id": case["case_id"],
                    "line_label": case["line_label"],
                    "line_center_A": case["line_center_A"],
                    "position_index": case["position_index"],
                    "pa_offset_deg": case["pa_offset_deg"],
                    "injection_y": int(geometry["injection_yx"][0]),
                    "injection_x": int(geometry["injection_yx"][1]),
                    "actual_injection_pa_deg": float(geometry["actual_injection_pa_deg"]),
                    **row,
                }
            )
        case_rows.append(
            {
                "case_id": case["case_id"],
                "line_label": case["line_label"],
                "line_center_A": case["line_center_A"],
                "position_index": case["position_index"],
                "pa_offset_deg": case["pa_offset_deg"],
                "injection_y": int(geometry["injection_yx"][0]),
                "injection_x": int(geometry["injection_yx"][1]),
                "actual_injection_pa_deg": float(geometry["actual_injection_pa_deg"]),
                "n_controls": int(len(products["control_positions_yx"])),
                "injection_flux_sigma_native": float(products["injection_flux_sigma_native"]),
                "delta_matched_transfer_p16": matched_p16,
                "delta_matched_transfer_median": matched_p50,
                "delta_matched_transfer_p84": matched_p84,
                "delta_aperture_transfer_p16": aperture_p16,
                "delta_aperture_transfer_median": aperture_p50,
                "delta_aperture_transfer_p84": aperture_p84,
            }
        )

    completeness_rows = completeness_rows_from_grid(grid_rows, detection_threshold_snr)
    return grid_rows, case_rows, completeness_rows


def compute_stage06_local_sweep_products(
    cubes_norm,
    wavelengths,
    med_pix_stack,
    config,
    *,
    selected_indices=None,
    selection_source="explicit",
):
    """Run all configured C2 line-position injection cases."""

    cases = resolve_sweep_cases(config, wavelengths)
    case_results = []
    for case in cases:
        products = compute_stage06_local_products(
            cubes_norm,
            wavelengths,
            med_pix_stack,
            case["config"],
            selected_indices=selected_indices,
            selection_source=selection_source,
        )
        case_results.append(
            (
                case,
                {
                    "geometry": products["geometry"],
                    "rows": products["rows"],
                    "control_positions_yx": products["control_positions_yx"],
                    "injection_flux_sigma_native": products["injection_flux_sigma_native"],
                },
            )
        )
        del products
        gc.collect()
    grid_rows, case_rows, completeness_rows = summarize_sweep(
        case_results,
        config["sweep_detection_threshold_snr"],
    )
    return {
        "cases": cases,
        "case_results": case_results,
        "grid_rows": grid_rows,
        "case_rows": case_rows,
        "completeness_rows": completeness_rows,
        "selection_source": selection_source,
        "selected_indices": list(map(int, selected_indices or range(np.asarray(cubes_norm).shape[0]))),
    }


def stage06_local_sweep_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage06_local_surface_c2")
    return {
        "paths": paths,
        "grid_csv": paths.table_dir / "stage06_local_surface_c2_grid.csv",
        "case_summary_csv": paths.table_dir / "stage06_local_surface_c2_case_summary.csv",
        "completeness_csv": paths.table_dir / "stage06_local_surface_c2_completeness.csv",
        "qc_json": paths.stage_dir / "stage06_local_surface_c2_qc.json",
        "plot_dir": plot_dir,
        "summary_plot": plot_dir / "stage06_local_surface_c2_summary.png",
    }


def save_stage06_local_sweep_plot(path, products, *, show_plots=False):
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    completeness = products["completeness_rows"]
    cases = products["case_rows"]
    grid = products["grid_rows"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    target = np.asarray([row["target_snr_input"] for row in completeness])
    threshold = float(completeness[0]["detection_threshold_snr"])
    axes[0, 0].plot(target, [row["matched_completeness"] for row in completeness], "o-", label="matched")
    axes[0, 0].plot(target, [row["aperture_completeness"] for row in completeness], "s-", label="3x3")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].set_xlabel("Injected S/N")
    axes[0, 0].set_ylabel(f"Fraction with recovered S/N >= {threshold:g}")
    axes[0, 0].set_title("Empirical completeness across line-position cases")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    labels = sorted({row["line_label"] for row in cases})
    for label in labels:
        rows = [row for row in cases if row["line_label"] == label]
        axes[0, 1].scatter(
            [row["line_center_A"] for row in rows],
            [row["delta_matched_transfer_median"] for row in rows],
            label=label,
        )
    axes[0, 1].axhline(1.0, color="0.3", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("Wavelength [A]")
    axes[0, 1].set_ylabel("Median delta matched transfer")
    axes[0, 1].set_title("Throughput by wavelength and PA")
    axes[0, 1].grid(alpha=0.25)

    for label in labels:
        rows = [row for row in cases if row["line_label"] == label]
        axes[1, 0].plot(
            [row["pa_offset_deg"] for row in rows],
            [row["delta_matched_transfer_median"] for row in rows],
            "o-",
            label=label,
        )
    axes[1, 0].axhline(1.0, color="0.3", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("PA offset from reference [deg]")
    axes[1, 0].set_ylabel("Median delta matched transfer")
    axes[1, 0].set_title("Spatial throughput variation")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    for case_id in sorted({row["case_id"] for row in grid}):
        rows = [row for row in grid if row["case_id"] == case_id]
        axes[1, 1].plot(
            [row["target_snr_input"] for row in rows],
            [row["local_surface_matched_snr"] for row in rows],
            color="0.55",
            alpha=0.55,
            linewidth=0.9,
        )
    lo = min(row["target_snr_input"] for row in grid)
    hi = max(row["target_snr_input"] for row in grid)
    axes[1, 1].plot([lo, hi], [lo, hi], "k--", linewidth=1, label="1:1")
    axes[1, 1].set_xlabel("Injected S/N")
    axes[1, 1].set_ylabel("Recovered matched S/N")
    axes[1, 1].set_title("Recovery curves for all cases")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    return path


def _isolated_sweep_plot_worker(products_path, plot_path):
    products = read_json(products_path)
    save_stage06_local_sweep_plot(plot_path, products, show_plots=False)


def write_stage06_local_sweep_products(
    products,
    config,
    paths,
    *,
    save_plots=True,
    show_plots=False,
):
    run_paths = paths["paths"]
    run_paths.ensure_base_dirs()
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    write_csv(paths["grid_csv"], products["grid_rows"], list(products["grid_rows"][0].keys()))
    write_csv(
        paths["case_summary_csv"],
        products["case_rows"],
        list(products["case_rows"][0].keys()),
    )
    write_csv(
        paths["completeness_csv"],
        products["completeness_rows"],
        list(products["completeness_rows"][0].keys()),
    )
    plot_paths = []
    if save_plots:
        if show_plots:
            plot_paths.append(
                save_stage06_local_sweep_plot(paths["summary_plot"], products, show_plots=True)
            )
        else:
            worker_code = (
                "from musepipe.stages.stage06_local_surface_sweep import "
                "_isolated_sweep_plot_worker; import sys; "
                "_isolated_sweep_plot_worker(sys.argv[1], sys.argv[2])"
            )
            with tempfile.TemporaryDirectory(prefix="musepipe-c2-plot-") as tmp:
                products_path = Path(tmp) / "plot_products.json"
                write_json(
                    products_path,
                    {
                        "completeness_rows": products["completeness_rows"],
                        "case_rows": products["case_rows"],
                        "grid_rows": products["grid_rows"],
                    },
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        worker_code,
                        str(products_path),
                        str(paths["summary_plot"]),
                    ],
                    cwd=run_paths.project_root,
                    env={**os.environ, "MPLBACKEND": "Agg"},
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if completed.returncode != 0:
                    detail = completed.stderr.strip() or completed.stdout.strip()
                    raise RuntimeError(f"C2 summary plot worker failed: {detail}")
            plot_paths.append(paths["summary_plot"])
    matched_transfer = [row["delta_matched_transfer_median"] for row in products["case_rows"]]
    aperture_transfer = [row["delta_aperture_transfer_median"] for row in products["case_rows"]]
    qc = {
        "run_id": config["run_id"],
        "stage": "stage06_local_surface_c2_sweep",
        "input_mode": config["input_mode"],
        "input_cube_fits": str(config["input_cube_fits"]),
        "input_shape": list(map(int, config["input_shape"])),
        "recovery_mode": config["recovery_mode"],
        "injection_snr_reference": config["injection_snr_reference"],
        "sweep_lines": config["sweep_lines"],
        "sweep_pa_offsets_deg": list(map(float, config["sweep_pa_offsets_deg"])),
        "detection_threshold_snr": float(config["sweep_detection_threshold_snr"]),
        "n_cases": len(products["case_rows"]),
        "selected_indices": products["selected_indices"],
        "selection_source": products["selection_source"],
        "delta_matched_transfer_percentiles": list(_finite_percentiles(matched_transfer)),
        "delta_aperture_transfer_percentiles": list(_finite_percentiles(aperture_transfer)),
        "completeness": products["completeness_rows"],
        "input_snr_at_completeness": {
            "matched_50pct": input_snr_at_completeness(
                products["completeness_rows"], "matched_completeness", 0.5
            ),
            "matched_90pct": input_snr_at_completeness(
                products["completeness_rows"], "matched_completeness", 0.9
            ),
            "aperture_50pct": input_snr_at_completeness(
                products["completeness_rows"], "aperture_completeness", 0.5
            ),
            "aperture_90pct": input_snr_at_completeness(
                products["completeness_rows"], "aperture_completeness", 0.9
            ),
        },
        "outputs": {
            "grid_csv": str(paths["grid_csv"]),
            "case_summary_csv": str(paths["case_summary_csv"]),
            "completeness_csv": str(paths["completeness_csv"]),
            "plots": [str(value) for value in plot_paths],
        },
        "scope_note": (
            "Completeness is the fraction of configured wavelength-position cases above the "
            "detection threshold at each injected S/N; it is not a Monte Carlo probability."
        ),
    }
    write_json(paths["qc_json"], qc)
    return {
        "grid_csv": paths["grid_csv"],
        "case_summary_csv": paths["case_summary_csv"],
        "completeness_csv": paths["completeness_csv"],
        "qc_json": paths["qc_json"],
        "plots": plot_paths,
        "qc": qc,
    }


def stage06_local_sweep_config_from_run(
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
    config = stage06_local_config_from_run(
        run_config.run_id,
        project_root=run_config.paths.project_root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    cfg = run_config.config
    for key, default_value in STAGE06_LOCAL_SWEEP_DEFAULTS.items():
        config[key] = cfg.get(f"stage06_local_{key}", default_value)
    if overrides:
        config.update(overrides)
    return config


def _compute_sweep_from_run(config, root, allow_run_id_mismatch=False):
    single_paths = stage06_local_paths(config["run_id"], root)
    stage04_paths = stage04b_paths(config["run_id"], root)
    stage04_cfg = stage04b_config_from_run(
        config["run_id"],
        project_root=root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    stage04_cfg.update(
        {
            "input_mode": config["input_mode"],
            "bad_wavelength_ranges_A": config.get("bad_wavelength_ranges_A", []),
        }
    )
    input_payload = load_stage04b_input(stage04_cfg, stage04_paths)
    selected_indices, selection_source = _selected_cube_indices(
        config,
        input_payload["cubes"].shape[0],
        paths=single_paths,
    )
    config["input_cube_fits"] = str(input_payload["input_cube_fits"])
    config["input_shape"] = list(map(int, input_payload["cubes"].shape))
    return compute_stage06_local_sweep_products(
        input_payload["cubes"],
        input_payload["wavelengths"],
        input_payload["med_pix_stack"],
        config,
        selected_indices=selected_indices,
        selection_source=selection_source,
    )


def _isolated_line_worker(config_path, result_path):
    """Process one wavelength in a short-lived interpreter."""

    config = read_json(config_path)
    root = Path(config.get("project_root") or Path.cwd()).resolve()
    allow_mismatch = bool(config.pop("_allow_run_id_mismatch", False))
    products = _compute_sweep_from_run(config, root, allow_mismatch)
    write_json(
        result_path,
        {
            "grid_rows": products["grid_rows"],
            "case_rows": products["case_rows"],
            "selected_indices": products["selected_indices"],
            "selection_source": products["selection_source"],
            "input_cube_fits": config["input_cube_fits"],
            "input_shape": config["input_shape"],
        },
    )


def _compute_isolated_line_sweeps(config, root, allow_run_id_mismatch=False):
    """Run each wavelength in a fresh process and merge scalar products."""

    grid_rows = []
    case_rows = []
    selected_indices = None
    selection_source = None
    input_cube_fits = None
    input_shape = None
    worker_code = (
        "from musepipe.stages.stage06_local_surface_sweep import _isolated_line_worker; "
        "import sys; _isolated_line_worker(sys.argv[1], sys.argv[2])"
    )
    with tempfile.TemporaryDirectory(prefix="musepipe-c2-") as tmp:
        tmp_dir = Path(tmp)
        for index, line in enumerate(config["sweep_lines"]):
            worker_config = dict(config)
            worker_config["sweep_lines"] = [line]
            worker_config["sweep_isolate_lines"] = False
            worker_config["_allow_run_id_mismatch"] = bool(allow_run_id_mismatch)
            config_path = tmp_dir / f"config_{index:02d}.json"
            result_path = tmp_dir / f"result_{index:02d}.json"
            write_json(config_path, worker_config)
            completed = subprocess.run(
                [sys.executable, "-c", worker_code, str(config_path), str(result_path)],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    f"C2 worker failed for line {line.get('label', index)!r}: {detail}"
                )
            payload = read_json(result_path)
            grid_rows.extend(payload["grid_rows"])
            case_rows.extend(payload["case_rows"])
            selected_indices = payload["selected_indices"]
            selection_source = payload["selection_source"]
            input_cube_fits = payload["input_cube_fits"]
            input_shape = payload["input_shape"]
    config["input_cube_fits"] = input_cube_fits
    config["input_shape"] = input_shape
    completeness_rows = completeness_rows_from_grid(
        grid_rows,
        config["sweep_detection_threshold_snr"],
    )
    return {
        "cases": [],
        "case_results": [],
        "grid_rows": grid_rows,
        "case_rows": case_rows,
        "completeness_rows": completeness_rows,
        "selection_source": selection_source,
        "selected_indices": selected_indices,
    }


def run_stage06_local_sweep(
    config=None,
    *,
    show_plots=False,
    save_plots=True,
    project_root=None,
    allow_run_id_mismatch=False,
):
    """Run the C2 local-surface sweep without replacing single-line products."""

    if config is None:
        config = stage06_local_sweep_config_from_run(
            project_root=project_root,
            allow_run_id_mismatch=allow_run_id_mismatch,
        )
    else:
        config = dict(config)
    root = Path(project_root or config.get("project_root") or Path.cwd()).resolve()
    paths = stage06_local_sweep_paths(config["run_id"], root)
    if bool(config.get("sweep_isolate_lines", True)) and len(config["sweep_lines"]) > 1:
        products = _compute_isolated_line_sweeps(config, root, allow_run_id_mismatch)
    else:
        products = _compute_sweep_from_run(config, root, allow_run_id_mismatch)
    written = write_stage06_local_sweep_products(
        products,
        config,
        paths,
        save_plots=save_plots,
        show_plots=show_plots,
    )
    return {**written, "products": products}


__all__ = [
    "STAGE06_LOCAL_SWEEP_DEFAULTS",
    "compute_stage06_local_sweep_products",
    "completeness_rows_from_grid",
    "input_snr_at_completeness",
    "nearest_line_channel_targets",
    "resolve_sweep_cases",
    "run_stage06_local_sweep",
    "save_stage06_local_sweep_plot",
    "stage06_local_sweep_config_from_run",
    "stage06_local_sweep_paths",
    "summarize_sweep",
    "symmetric_continuum_windows",
    "write_stage06_local_sweep_products",
]
