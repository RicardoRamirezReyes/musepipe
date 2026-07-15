"""Stage H02/E2: fixed artifact tests for Halpha results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from ..io import get_cube_data, read_json, read_wavelength_axis, write_json
from ..paths import RunPaths
from ..psf import evaluate_psf_model
from .stage_h01_detect import (
    DEFAULT_TEMPLATE_WIDTH_FACTORS,
    HALPHA_REST_A,
    analyze_halpha_method,
    load_h01_controls,
    load_h01_products,
)
from .stage_h01_detect import (
    _control_paths_from_config as _h01_control_paths_from_config,
)
from .stage_h01_detect import (
    _lsf_fwhm_from_qc_or_config as _h01_lsf_fwhm_from_qc_or_config,
)
from .stage_h01_detect import (
    _product_paths_from_config as _h01_product_paths_from_config,
)
from .stage_h01_detect import _rv_err_from_config as _h01_rv_err_from_config
from .stage_h01_detect import _rv_from_config as _h01_rv_from_config


PLACEBO_CENTERS_A = (6200.0, 6400.0, 6700.0, 7100.0)
ARTIFACT_THRESHOLD_CHANNELS = 2


@dataclass(frozen=True)
class StageH02Product:
    qc: dict
    figures: dict


def _finite_or_none(value):
    if value is None:
        return None
    val = float(value)
    if not np.isfinite(val):
        return None
    return val


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_or_none(value)
    return value


def stage_h02_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage_h02")
    return {
        "paths": paths,
        "stage_h01_qc_json": paths.stage_dir / "stage_h01_qc.json",
        "halpha_detection_csv": paths.table_dir / "halpha_detection_by_method.csv",
        "stage02_qc_json": paths.stage_dir / "stage02_xcorr_qc.json",
        "stage00q_qc_json": paths.stage_dir / "stage00q_qc.json",
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "psf_model_json": paths.stage_dir / "psf_model.json",
        "spec_calibrated_aperture_object": paths.stage_dir / "spec_calibrated_aperture_object.fits",
        "spec_calibrated_optimal_ls_object": paths.stage_dir / "spec_calibrated_optimal_ls_object.fits",
        "spec_calibrated_optimal_psfsub_object": paths.stage_dir / "spec_calibrated_optimal_psfsub_object.fits",
        "spec_calibrated_psffit_object": paths.stage_dir / "spec_calibrated_psffit_object.fits",
        "spec_calibrated_sgf_object": paths.stage_dir / "spec_calibrated_sgf_object.fits",
        "spec_calibrated_lpm_object": paths.stage_dir / "spec_calibrated_lpm_object.fits",
        "controls_calibrated_aperture_npz": paths.stage_dir / "spec_calibrated_aperture_controls.npz",
        "controls_calibrated_optimal_ls_npz": paths.stage_dir / "spec_calibrated_optimal_ls_controls.npz",
        "controls_calibrated_optimal_psfsub_npz": paths.stage_dir / "spec_calibrated_optimal_psfsub_controls.npz",
        "controls_calibrated_psffit_npz": paths.stage_dir / "spec_calibrated_psffit_controls.npz",
        "controls_calibrated_sgf_npz": paths.stage_dir / "spec_calibrated_sgf_controls.npz",
        "controls_calibrated_lpm_npz": paths.stage_dir / "spec_calibrated_lpm_controls.npz",
        "cube_psffit_residual": paths.stage_dir / "cube_psffit_residual.fits",
        "cube_residual_object": paths.stage_dir / "cube_residual_local_object.fits",
        "stage_h02_qc_json": paths.stage_dir / "stage_h02_qc.json",
        "plot_dir": plot_dir,
        "t1_plot": plot_dir / "stage_h02_t1_artifact_channels.png",
        "t2_plot": plot_dir / "stage_h02_t2_stamp_psf.png",
        "t4_plot": plot_dir / "stage_h02_t4_tornado.png",
        "t5_plot": plot_dir / "stage_h02_t5_placebos.png",
    }


def stage_h02_config_from_run(
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
    cfg.setdefault("h02_artifact_threshold_channels", ARTIFACT_THRESHOLD_CHANNELS)
    cfg.setdefault("h02_placebo_centers_A", list(PLACEBO_CENTERS_A))
    cfg.setdefault("h02_detection_fap", cfg.get("h01_detection_fap", 0.01))
    cfg.setdefault("h02_n_exposures", cfg.get("n_exposures", 1))
    cfg.setdefault("h02_stamp_half_size_px", 5)
    cfg.setdefault("h02_template_width_factors", cfg.get("h01_template_width_factors", list(DEFAULT_TEMPLATE_WIDTH_FACTORS)))
    return cfg


def _read_optional_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return read_json(path)


def _nearest_index(wave_A, value_A):
    wave = np.asarray(wave_A, dtype=np.float64)
    good = np.isfinite(wave)
    if not np.any(good) or value_A is None or not np.isfinite(float(value_A)):
        return None
    indices = np.where(good)[0]
    return int(indices[np.argmin(np.abs(wave[good] - float(value_A)))])


def _nearest_abs_delta(index, candidate_indices):
    if index is None or not candidate_indices:
        return None
    vals = [abs(int(index) - int(candidate)) for candidate in candidate_indices]
    return int(min(vals)) if vals else None


def t1_instrumental_coincidence(
    wave_A,
    signal_wave_A,
    *,
    stripe_channels=None,
    skyline_waves_A=None,
    gap_edges_A=None,
    threshold_channels=ARTIFACT_THRESHOLD_CHANNELS,
):
    signal_index = _nearest_index(wave_A, signal_wave_A)
    skyline_indices = [
        _nearest_index(wave_A, wave)
        for wave in (skyline_waves_A or [])
        if wave is not None and np.isfinite(float(wave))
    ]
    gap_indices = [
        _nearest_index(wave_A, wave)
        for wave in (gap_edges_A or [])
        if wave is not None and np.isfinite(float(wave))
    ]
    stripe_channels = [int(ch) for ch in (stripe_channels or [])]
    distances = {
        "stripe": _nearest_abs_delta(signal_index, stripe_channels),
        "skyline": _nearest_abs_delta(signal_index, [idx for idx in skyline_indices if idx is not None]),
        "laser_gap": _nearest_abs_delta(signal_index, [idx for idx in gap_indices if idx is not None]),
    }
    verdicts = {}
    for name, value in distances.items():
        if value is None:
            verdicts[name] = "unavailable"
        elif int(value) <= int(threshold_channels):
            verdicts[name] = "fail"
        else:
            verdicts[name] = "pass"
    failed = [name for name, status in verdicts.items() if status == "fail"]
    available = [status for status in verdicts.values() if status != "unavailable"]
    status = "unavailable" if not available else ("fail" if failed else "pass")
    return {
        "signal_channel": signal_index,
        "nearest_stripe_dch": distances["stripe"],
        "nearest_skyline_dch": distances["skyline"],
        "nearest_laser_gap_dch": distances["laser_gap"],
        "coincidence": bool(failed),
        "by_list": verdicts,
        "status": status,
    }


def _plane_design(shape):
    yy, xx = np.indices(shape, dtype=np.float64)
    y0 = 0.5 * (shape[0] - 1)
    x0 = 0.5 * (shape[1] - 1)
    scale = max(shape)
    return np.column_stack(
        [
            np.ones(shape[0] * shape[1], dtype=np.float64),
            ((yy - y0) / scale).ravel(),
            ((xx - x0) / scale).ravel(),
        ]
    )


def _linear_chi2(data, design):
    vals = np.asarray(data, dtype=np.float64).ravel()
    design = np.asarray(design, dtype=np.float64)
    valid = np.isfinite(vals) & np.all(np.isfinite(design), axis=1)
    if int(np.count_nonzero(valid)) <= design.shape[1]:
        return np.nan, np.full_like(vals, np.nan)
    coeff, *_ = np.linalg.lstsq(design[valid], vals[valid], rcond=None)
    model = design @ coeff
    resid = vals[valid] - model[valid]
    dof = max(1, int(np.count_nonzero(valid)) - design.shape[1])
    return float(np.nansum(resid**2) / dof), model.reshape(np.asarray(data).shape)


def _positive_centroid_and_axis_ratio(image):
    img = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(img)
    if not np.any(finite):
        return (np.nan, np.nan), np.nan
    bg = float(np.nanmedian(img[finite]))
    weights = img - bg
    weights[~finite] = 0.0
    weights = np.clip(weights, 0.0, None)
    if float(np.nansum(weights)) <= 0:
        weights = np.clip(img, 0.0, None)
    total = float(np.nansum(weights))
    if total <= 0:
        return (np.nan, np.nan), np.nan
    yy, xx = np.indices(img.shape, dtype=np.float64)
    cy = float(np.nansum(yy * weights) / total)
    cx = float(np.nansum(xx * weights) / total)
    dy = yy - cy
    dx = xx - cx
    cov = np.array(
        [
            [np.nansum(weights * dy * dy) / total, np.nansum(weights * dy * dx) / total],
            [np.nansum(weights * dy * dx) / total, np.nansum(weights * dx * dx) / total],
        ],
        dtype=np.float64,
    )
    eig = np.linalg.eigvalsh(cov)
    if eig[0] <= 0 or not np.all(np.isfinite(eig)):
        ratio = np.nan
    else:
        ratio = float(np.sqrt(eig[1] / eig[0]))
    return (cy, cx), ratio


def t2_spatial_coherence(
    stamp,
    psf_stamp,
    *,
    expected_center_yx=None,
    centroid_threshold_px=1.0,
    elongation_threshold=1.5,
    chi2_ratio_threshold=0.8,
):
    data = np.asarray(stamp, dtype=np.float64)
    psf = np.asarray(psf_stamp, dtype=np.float64)
    if data.shape != psf.shape:
        raise ValueError("stamp and psf_stamp must have matching shapes.")
    psf = psf - np.nanmedian(psf[np.isfinite(psf)])
    norm = float(np.sqrt(np.nansum(psf**2)))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("psf_stamp has no finite structure.")
    psf = psf / norm
    plane = _plane_design(data.shape)
    chi2_plane, plane_model = _linear_chi2(data, plane)
    design_psf = np.column_stack([psf.ravel(), plane])
    chi2_psf, psf_model = _linear_chi2(data, design_psf)
    ratio = np.nan
    if np.isfinite(chi2_plane) and chi2_plane > 0:
        ratio = float(chi2_psf / chi2_plane)
    centroid, axis_ratio = _positive_centroid_and_axis_ratio(data - plane_model)
    psf_centroid, psf_axis_ratio = _positive_centroid_and_axis_ratio(psf)
    if expected_center_yx is None:
        expected_center_yx = psf_centroid
    centroid_offset = float(np.hypot(centroid[0] - expected_center_yx[0], centroid[1] - expected_center_yx[1]))
    elongation_vs_psf = np.nan
    if np.isfinite(axis_ratio) and np.isfinite(psf_axis_ratio) and psf_axis_ratio > 0:
        elongation_vs_psf = float(axis_ratio / psf_axis_ratio)
    status = "pass"
    if not np.isfinite(ratio) or not np.isfinite(centroid_offset) or not np.isfinite(elongation_vs_psf):
        status = "unavailable"
    elif (
        ratio >= float(chi2_ratio_threshold)
        or centroid_offset >= float(centroid_threshold_px)
        or elongation_vs_psf >= float(elongation_threshold)
    ):
        status = "fail"
    return {
        "chi2_ratio_psf_vs_plane": _finite_or_none(ratio),
        "centroid_offset_px": _finite_or_none(centroid_offset),
        "elongation_vs_psf": _finite_or_none(elongation_vs_psf),
        "centroid_y": _finite_or_none(centroid[0]),
        "centroid_x": _finite_or_none(centroid[1]),
        "status": status,
    }


def t3_temporal_stability(total_z, half_z_values, *, n_exposures):
    if int(n_exposures) < 2:
        return {"status": "unavailable", "reason": "single_exposure"}
    vals = np.asarray(half_z_values, dtype=np.float64)
    if vals.size < 2 or np.any(~np.isfinite(vals)):
        return {"status": "unavailable", "reason": "substack_measurements_missing"}
    threshold = float(total_z) / np.sqrt(2.0)
    passed = bool(np.all(vals[:2] > threshold))
    return {"status": "pass" if passed else "fail", "threshold_z": threshold, "half_z": vals[:2].tolist()}


def t4_parameter_stability(variant_rows, *, threshold_dz=1.0):
    rows = list(variant_rows or [])
    finite = [row for row in rows if row.get("z") is not None and np.isfinite(float(row["z"]))]
    if not finite:
        return {"status": "unavailable", "reason": "validation_variant_measurements_missing", "z_range": None, "worst_knob": None, "variants": rows}
    values = np.asarray([float(row["z"]) for row in finite], dtype=np.float64)
    z_range = float(np.nanmax(values) - np.nanmin(values))
    worst = finite[int(np.nanargmax(np.abs(values - np.nanmedian(values))))]
    return {
        "status": "pass" if z_range < float(threshold_dz) else "fail",
        "z_range": z_range,
        "worst_knob": str(worst.get("knob", worst.get("name", "unknown"))),
        "variants": rows,
    }


def t5_placebo_calibration(placebo_rows, *, detection_fap=0.01):
    rows = list(placebo_rows or [])
    ran = [row for row in rows if row.get("status", "ran") == "ran" and row.get("global_empirical_fap") is not None]
    if not ran:
        return {"status": "unavailable", "reason": "no_placebo_in_spectral_range", "placebo_max_fap_global": None, "any_above_threshold": None, "rows": rows}
    faps = np.asarray([float(row["global_empirical_fap"]) for row in ran], dtype=np.float64)
    any_hit = bool(np.any(faps < float(detection_fap)))
    return {
        "status": "fail" if any_hit else "pass",
        "placebo_max_fap_global": float(np.nanmin(faps)),
        "any_above_threshold": any_hit,
        "rows": rows,
    }


def _read_detection_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_cell(row, key, default=np.nan):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _selected_h01_row(rows, preferred_method=None):
    if not rows:
        return None
    if preferred_method:
        for row in rows:
            if row.get("method") == preferred_method:
                return row
    finite = [row for row in rows if np.isfinite(_float_cell(row, "global_empirical_fap"))]
    if finite:
        return min(finite, key=lambda row: _float_cell(row, "global_empirical_fap"))
    return rows[0]


def _extract_stripe_channels(qc):
    if not qc:
        return None
    return qc.get("stripe_metric", {}).get("dirty_channels")


def _extract_skyline_waves(qc):
    if not qc:
        return None
    m1 = qc.get("m1_wavelength", {})
    if isinstance(m1.get("skyline_waves_A"), list):
        return m1["skyline_waves_A"]
    measurements = m1.get("measurements") or m1.get("lines") or []
    out = []
    for row in measurements:
        if isinstance(row, dict):
            for key in ("expected_wave_A", "lab_wave_A", "wave_A"):
                if row.get(key) is not None:
                    out.append(float(row[key]))
                    break
    return out or None


def _extract_gap_edges(qc):
    if not qc:
        return None
    windows = qc.get("excluded_windows_A") or qc.get("bad_wavelength_ranges_A") or []
    edges = []
    for window in windows:
        if window is not None and len(window) == 2:
            edges.extend([float(window[0]), float(window[1])])
    return edges or None


def _load_signal_cube_and_wave(paths):
    candidates = [paths["cube_psffit_residual"], paths["cube_residual_object"]]
    for path in candidates:
        if not Path(path).exists():
            continue
        with fits.open(path, memmap=True) as hdul:
            cube = get_cube_data(hdul).astype(np.float64)
            wave = read_wavelength_axis(hdul, data_shape=cube.shape)
        return cube, wave, Path(path)
    raise FileNotFoundError("No residual cube available for T2.")


def _companion_yx_from_qc(qc):
    if not qc:
        raise KeyError("stage01c QC missing")
    for key in ("companion", "target", "object"):
        if key in qc and isinstance(qc[key], dict) and qc[key].get("pos_yx") is not None:
            return tuple(map(float, qc[key]["pos_yx"]))
    if qc.get("target_object") and qc.get("detected_peaks"):
        peak = qc["detected_peaks"][qc["target_object"]]
        return float(peak["y"]), float(peak["x"])
    raise KeyError("Could not recover companion position from stage01c QC.")


def _stamp(image, center_yx, half):
    y, x = map(float, center_yx)
    y0 = int(round(y))
    x0 = int(round(x))
    half = int(half)
    y1 = max(0, y0 - half)
    y2 = min(image.shape[0], y0 + half + 1)
    x1 = max(0, x0 - half)
    x2 = min(image.shape[1], x0 + half + 1)
    return image[y1:y2, x1:x2], (y - y1, x - x1)


def _psf_stamp(psf_model, wave_A, shape, center_yx):
    yy, xx = np.indices(shape, dtype=np.float64)
    return evaluate_psf_model(
        psf_model,
        float(wave_A),
        yy - float(center_yx[0]),
        xx - float(center_yx[1]),
    )


def _compute_t2_from_files(paths, cfg, signal_wave_A):
    try:
        cube, wave, _cube_path = _load_signal_cube_and_wave(paths)
        qc01c = read_json(paths["stage01c_qc_json"])
        psf_model = read_json(paths["psf_model_json"])
        yx = _companion_yx_from_qc(qc01c)
        channel = _nearest_index(wave, signal_wave_A)
        if channel is None:
            raise ValueError("Signal channel unavailable.")
        stamp, local_center = _stamp(cube[int(channel)], yx, int(cfg.get("h02_stamp_half_size_px", 5)))
        psf = _psf_stamp(psf_model, wave[int(channel)], stamp.shape, local_center)
        return t2_spatial_coherence(stamp, psf, expected_center_yx=local_center)
    except Exception as exc:
        return {
            "chi2_ratio_psf_vs_plane": None,
            "centroid_offset_px": None,
            "elongation_vs_psf": None,
            "status": "unavailable",
            "reason": str(exc),
        }


def _compute_t5_placebos(paths, cfg):
    try:
        qc00 = read_json(paths["stage00q_qc_json"])
        products = load_h01_products(_h01_product_paths_from_config(cfg, paths))
        controls = load_h01_controls(
            _h01_control_paths_from_config(cfg, paths),
            products,
            min_controls=int(cfg.get("h01_min_controls", 3)),
        )
        rv_sys, _rv_source = _h01_rv_from_config(cfg)
        rv_err = _h01_rv_err_from_config(cfg)
        lsf_fwhm, _lsf_source = _h01_lsf_fwhm_from_qc_or_config(qc00, cfg)
    except Exception as exc:
        return t5_placebo_calibration(
            [{"status": "unavailable", "reason": f"could_not_run_placebos:{exc}"}],
            detection_fap=float(cfg.get("h02_detection_fap", 0.01)),
        )
    rows = []
    for center_A in cfg.get("h02_placebo_centers_A", PLACEBO_CENTERS_A):
        for method in products:
            try:
                result = analyze_halpha_method(
                    method,
                    products[method],
                    controls[method],
                    rv_sys_kms=rv_sys,
                    rv_sys_err_kms=rv_err,
                    lsf_fwhm_A=lsf_fwhm,
                    rest_A=float(center_A),
                    search_half_width_kms=float(cfg.get("h01_search_half_width_kms", 500.0)),
                    width_factors=cfg.get("h02_template_width_factors", DEFAULT_TEMPLATE_WIDTH_FACTORS),
                    continuum_window_A=float(cfg.get("h01_continuum_window_A", 80.0)),
                )
                rows.append(
                    {
                        "center_A": float(center_A),
                        "method": method,
                        "status": "ran",
                        "global_empirical_fap": result.row["global_empirical_fap"],
                        "matched_z": result.row["matched_z"],
                    }
                )
            except ValueError:
                rows.append({"center_A": float(center_A), "method": method, "status": "skipped_outside_range"})
    return t5_placebo_calibration(rows, detection_fap=float(cfg.get("h02_detection_fap", 0.01)))


def _overall_status(t1, t2, t3, t4, t5):
    if t1["status"] == "fail" or t2["status"] == "fail" or t5["status"] == "fail":
        return "fails"
    applicable = [test for test in (t1, t2, t3, t4, t5) if test["status"] != "unavailable"]
    if applicable and all(test["status"] == "pass" for test in applicable):
        return "survives"
    return "mixed"


def compute_stage_h02_products(config, paths=None) -> StageH02Product:
    cfg = dict(config)
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_h02_paths(cfg["run_id"], root) if paths is None else paths
    h01_qc = _read_optional_json(paths["stage_h01_qc_json"]) or {}
    detection_rows = _read_detection_rows(paths["halpha_detection_csv"])
    selected = _selected_h01_row(detection_rows, cfg.get("h02_method"))
    signal_wave = _float_cell(selected or {}, "peak_wave_A", float(cfg.get("h02_signal_wave_A", HALPHA_REST_A)))
    input_verdict = h01_qc.get("verdict", {}).get("verdict", "unknown")
    wave_for_t1 = None
    try:
        if selected and selected.get("method"):
            products = load_h01_products(_h01_product_paths_from_config(cfg, paths))
            wave_for_t1 = products[selected["method"]].wave_A
    except Exception:
        wave_for_t1 = None
    if wave_for_t1 is None:
        wave_for_t1 = np.arange(0, 10000, dtype=np.float64)
    qc02 = _read_optional_json(paths["stage02_qc_json"])
    qc00 = _read_optional_json(paths["stage00q_qc_json"])
    t1 = t1_instrumental_coincidence(
        wave_for_t1,
        signal_wave,
        stripe_channels=_extract_stripe_channels(qc02),
        skyline_waves_A=cfg.get("h02_skyline_waves_A", _extract_skyline_waves(qc00)),
        gap_edges_A=cfg.get("h02_gap_edges_A", _extract_gap_edges(qc00)),
        threshold_channels=int(cfg.get("h02_artifact_threshold_channels", ARTIFACT_THRESHOLD_CHANNELS)),
    )
    t2 = _compute_t2_from_files(paths, cfg, signal_wave)
    total_z = _float_cell(selected or {}, "matched_z", np.nan)
    t3 = t3_temporal_stability(
        total_z,
        cfg.get("h02_substack_z", []),
        n_exposures=int(cfg.get("h02_n_exposures", 1)),
    )
    t4 = t4_parameter_stability(cfg.get("h02_parameter_variants", []))
    t5 = _compute_t5_placebos(paths, cfg)
    qc = {
        "stage": "h02_artifact_tests",
        "run_id": str(cfg["run_id"]),
        "input_verdict_e1": input_verdict,
        "selected_method": None if selected is None else selected.get("method"),
        "selected_peak_wave_A": _finite_or_none(signal_wave),
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "t4": t4,
        "t5": t5,
        "overall": _overall_status(t1, t2, t3, t4, t5),
        "open_issues": [],
    }
    # Spec E2 erratum v1.1 (2026-07-15; F1 audit note 2026-07-10): in a
    # NON-DETECTION, T2 characterizes the dominant noise maximum (spec sec.2),
    # so "not PSF-shaped" SUPPORTS the non-detection instead of failing the
    # battery. overall_raw keeps the unbranched value.
    qc["overall_raw"] = qc["overall"]
    if str(input_verdict) == "non_detection" and t2.get("status") == "fail":
        qc["overall"] = _overall_status(t1, {**t2, "status": "pass"}, t3, t4, t5)
        qc["open_issues"].append(
            "T2 not-PSF-shaped on the global maximum SUPPORTS the non-detection "
            "(spec sec.2 repurposing); overall reinterpreted, overall_raw retained."
        )
    if t5["status"] == "fail":
        qc["open_issues"].append("T5 placebo failure: E1 FAP calibration is suspect and blocks E1/E3.")
    if qc["overall"] == "mixed":
        qc["open_issues"].append("At least one applicable artifact axis is unavailable or mixed; user checkpoint required.")
    figures = {}
    return StageH02Product(qc=_json_ready(qc), figures=figures)


def _plot_status_bar(qc, path):
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    names = ["t1", "t2", "t3", "t4", "t5"]
    color = {"pass": "tab:green", "fail": "tab:red", "unavailable": "0.6"}
    statuses = [qc[name]["status"] for name in names]
    fig, ax = plt.subplots(figsize=(7, 2.5), constrained_layout=True)
    ax.bar(names, np.ones(len(names)), color=[color.get(status, "tab:orange") for status in statuses])
    for i, status in enumerate(statuses):
        ax.text(i, 0.5, status, ha="center", va="center", color="white", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(f"H02 artifact tests: {qc['overall']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_t2_placeholder(qc, path):
    _plot_status_bar(qc, path)


def _plot_t4_tornado(qc, path):
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = [row for row in qc["t4"].get("variants", []) if row.get("z") is not None]
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    if rows:
        labels = [str(row.get("knob", row.get("name", i))) for i, row in enumerate(rows)]
        zvals = [float(row["z"]) for row in rows]
        ax.barh(labels, zvals, color="tab:blue")
        ax.set_xlabel("H01 z")
    else:
        ax.text(0.5, 0.5, qc["t4"].get("reason", qc["t4"]["status"]), ha="center", va="center")
        ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_t5_placebos(qc, path):
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = [row for row in qc["t5"].get("rows", []) if row.get("global_empirical_fap") is not None]
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    if rows:
        labels = [f"{row['method']}@{row['center_A']:.0f}" for row in rows]
        faps = [float(row["global_empirical_fap"]) for row in rows]
        ax.bar(np.arange(len(rows)), faps, color="tab:purple")
        ax.axhline(0.01, color="tab:red", ls="--", lw=1)
        ax.set_xticks(np.arange(len(rows)), labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("global FAP")
    else:
        ax.text(0.5, 0.5, qc["t5"].get("reason", qc["t5"]["status"]), ha="center", va="center")
        ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_stage_h02_products(product: StageH02Product, config, paths):
    paths["paths"].ensure_base_dirs()
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    qc = dict(product.qc)
    _plot_status_bar(qc, paths["t1_plot"])
    _plot_t2_placeholder(qc, paths["t2_plot"])
    _plot_t4_tornado(qc, paths["t4_plot"])
    _plot_t5_placebos(qc, paths["t5_plot"])
    qc["figures"] = {
        "t1": str(paths["t1_plot"]),
        "t2": str(paths["t2_plot"]),
        "t4": str(paths["t4_plot"]),
        "t5": str(paths["t5_plot"]),
    }
    write_json(paths["stage_h02_qc_json"], _json_ready(qc))
    return {"qc_json": paths["stage_h02_qc_json"], "qc": qc, "figures": qc["figures"]}


def run_stage_h02(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_h02_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_h02_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_h02_products(cfg, paths)
    written = write_stage_h02_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="stage_h02_artifacts.py",
        description="Run Stage H02/E2 fixed artifact tests.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage_h02(
        args.run_id,
        project_root=args.project_root,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage_h02_qc_json"])


__all__ = [
    "ARTIFACT_THRESHOLD_CHANNELS",
    "PLACEBO_CENTERS_A",
    "StageH02Product",
    "compute_stage_h02_products",
    "run_stage_h02",
    "stage_h02_config_from_run",
    "stage_h02_paths",
    "t1_instrumental_coincidence",
    "t2_spatial_coherence",
    "t3_temporal_stability",
    "t4_parameter_stability",
    "t5_placebo_calibration",
    "write_stage_h02_products",
]


if __name__ == "__main__":
    main()
