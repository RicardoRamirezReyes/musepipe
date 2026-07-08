"""Stage H04/E4: Halpha injection-recovery throughput calibration."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from ..extraction.product import SpectrumProduct
from ..injection import InjectionSource, create_run_clone, inject, tree_sha256
from ..io import read_json, read_wavelength_axis, write_csv, write_json
from ..paths import RunPaths
from ..spectral import continuum_running_median as _CONTINUUM_RUNMED
from .stage_h01_detect import HALPHA_REST_A, matched_filter_point
from .stage_x10_compare import METHOD_ORDER


C_KMS = 299792.458
H01_MATCHED_FILTER_POINT = matched_filter_point
DEFAULT_SNR_GRID = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
DEFAULT_TEMPLATE_FACTORS = (1.0, 2.0)
DEFAULT_METHODS = ("aperture", "optimal_ls", "optimal_psfsub", "psffit")


TABLE_FIELDS = [
    "injection_id",
    "variant",
    "method",
    "position_label",
    "position_y",
    "position_x",
    "template_width",
    "template_factor",
    "continuum_mode",
    "psf_fwhm_scale",
    "snr",
    "input_snr",
    "injected_flux",
    "recovered_flux",
    "recovered_sigma",
    "recovered_snr",
    "throughput",
    "throughput_err",
    "complete",
    "bias_flux_pct",
    "estimator",
]


@dataclass(frozen=True)
class H04Case:
    injection_id: str
    variant: str
    position_label: str
    position_y: float
    position_x: float
    template_factor: float
    input_snr: float
    continuum_mode: str
    psf_fwhm_scale: float = 1.0


@dataclass(frozen=True)
class StageH04Product:
    rows: list[dict]
    qc: dict


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_or_none(value)
    return value


def stage_h04_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage_h04")
    return {
        "paths": paths,
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "stage00q_qc_json": paths.stage_dir / "stage00q_qc.json",
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "psf_model_json": paths.stage_dir / "psf_model.json",
        "stage06_truth_json": paths.table_dir / "stage06_local_surface_halpha_injection_truth_summary.json",
        "throughput_csv": paths.table_dir / "injection_throughput_by_method.csv",
        "stage_h04_qc_json": paths.stage_dir / "stage_h04_qc.json",
        "clone_manifest_json": paths.stage_dir / "stage_h04_clone_manifest.json",
        "plot_dir": plot_dir,
        "throughput_plot": plot_dir / "stage_h04_throughput.png",
        "recovery_plot": plot_dir / "stage_h04_recovered_vs_injected.png",
        "completeness_plot": plot_dir / "stage_h04_completeness.png",
    }


def stage_h04_config_from_run(
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
    cfg.setdefault("h04_methods", list(DEFAULT_METHODS))
    cfg.setdefault("h04_snr_grid", list(DEFAULT_SNR_GRID))
    cfg.setdefault("h04_template_width_factors", list(DEFAULT_TEMPLATE_FACTORS))
    cfg.setdefault("h04_continuum_modes", ["none", "flat"])
    cfg.setdefault("h04_psf_perturb_snr_grid", [3.0, 5.0])
    cfg.setdefault("h04_psf_perturb_scales", [0.9, 1.1])
    cfg.setdefault("h04_max_runtime_hours", 4.0)
    cfg.setdefault("h04_expected_seconds_per_case_method", 60.0)
    cfg.setdefault("h04_allow_long_run", False)
    cfg.setdefault("h04_detection_threshold_snr", 5.0)
    cfg.setdefault("h04_historic_expected_snr", 8.97)
    cfg.setdefault("h04_historic_tolerance_snr", 0.25)
    cfg.setdefault("h04_require_historic_regression", True)
    cfg.setdefault("h04_injection_flux_sigma", cfg.get("h01_matched_sigma"))
    return cfg


def template_width_label(factor):
    val = float(factor)
    if np.isclose(val, 1.0):
        return "lsf"
    if val.is_integer():
        return f"{int(val)}x_lsf"
    return f"{val:g}x_lsf"


def _read_optional_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return read_json(path)


def _source_pos_yx(qc, *keys):
    for key in keys:
        item = (qc or {}).get(key)
        if isinstance(item, dict) and item.get("pos_yx") is not None:
            yx = item["pos_yx"]
            return float(yx[0]), float(yx[1])
    raise KeyError(f"Could not find any position in {keys}.")


def _same_radius_control_positions(star_yx, target_yx, *, n_controls=3):
    sy, sx = map(float, star_yx)
    ty, tx = map(float, target_yx)
    radius = math.hypot(ty - sy, tx - sx)
    start = math.atan2(ty - sy, tx - sx)
    controls = []
    for index in range(int(n_controls)):
        theta = start + 2.0 * math.pi * (index + 1) / float(n_controls + 1)
        controls.append((sy + radius * math.sin(theta), sx + radius * math.cos(theta)))
    return controls


def resolve_h04_positions(config, paths=None):
    raw_positions = config.get("h04_positions_yx")
    if raw_positions:
        out = []
        for index, item in enumerate(raw_positions):
            if isinstance(item, dict):
                label = str(item.get("label", f"pos{index}"))
                y = float(item["y"])
                x = float(item["x"])
            else:
                label = "real" if index == 0 else f"control{index}"
                y = float(item[0])
                x = float(item[1])
            out.append({"label": label, "y": y, "x": x})
        if len(out) != 4:
            raise RuntimeError("H04 grid requires exactly 4 positions: real + 3 controls.")
        return out

    real = config.get("h04_real_position_yx")
    controls = config.get("h04_control_positions_yx")
    if real is not None and controls is not None:
        out = [{"label": "real", "y": float(real[0]), "x": float(real[1])}]
        out.extend(
            {"label": f"control{index + 1}", "y": float(y), "x": float(x)}
            for index, (y, x) in enumerate(controls)
        )
        if len(out) != 4:
            raise RuntimeError("H04 requires exactly 3 control positions.")
        return out

    if paths is None:
        raise RuntimeError("H04 requires h04_positions_yx or stage01c_qc.json.")
    qc = _read_optional_json(paths["stage01c_qc_json"])
    target = _source_pos_yx(qc, "companion", "target", "object")
    star = _source_pos_yx(qc, "primary", "star")
    out = [{"label": "real", "y": float(target[0]), "x": float(target[1])}]
    out.extend(
        {"label": f"control{index + 1}", "y": float(y), "x": float(x)}
        for index, (y, x) in enumerate(_same_radius_control_positions(star, target, n_controls=3))
    )
    return out


def build_h04_cases(config, positions):
    cases = []
    snr_grid = [float(value) for value in config.get("h04_snr_grid", DEFAULT_SNR_GRID)]
    factors = [float(value) for value in config.get("h04_template_width_factors", DEFAULT_TEMPLATE_FACTORS)]
    continuum_modes = [str(value) for value in config.get("h04_continuum_modes", ["none", "flat"])]
    for pos in positions:
        for factor in factors:
            for continuum in continuum_modes:
                for snr in snr_grid:
                    cases.append(
                        H04Case(
                            injection_id=f"inj{len(cases):04d}",
                            variant="nominal",
                            position_label=str(pos["label"]),
                            position_y=float(pos["y"]),
                            position_x=float(pos["x"]),
                            template_factor=factor,
                            input_snr=snr,
                            continuum_mode=continuum,
                            psf_fwhm_scale=1.0,
                        )
                    )
    real = positions[0]
    for snr in [float(value) for value in config.get("h04_psf_perturb_snr_grid", [3.0, 5.0])]:
        for scale in [float(value) for value in config.get("h04_psf_perturb_scales", [0.9, 1.1])]:
            label = "psf_minus10" if scale < 1.0 else "psf_plus10"
            cases.append(
                H04Case(
                    injection_id=f"inj{len(cases):04d}",
                    variant=label,
                    position_label=str(real["label"]),
                    position_y=float(real["y"]),
                    position_x=float(real["x"]),
                    template_factor=1.0,
                    input_snr=snr,
                    continuum_mode="none",
                    psf_fwhm_scale=scale,
                )
            )
    return cases


def _resolve_h04_n_jobs(config, n_cases):
    """Worker count for the E4 case grid (ThreadPool over independent cases)."""
    from ..parallel import resolve_n_jobs

    requested = config.get("h04_n_jobs")
    if requested is None:
        return 1  # default: serial, to keep the frozen behaviour unless opted in
    return max(1, min(resolve_n_jobs(requested), int(n_cases)))


def estimate_runtime_budget(config, n_cases, n_methods):
    seconds = float(n_cases) * float(n_methods) * float(config.get("h04_expected_seconds_per_case_method", 60.0))
    hours = seconds / 3600.0
    max_hours = float(config.get("h04_max_runtime_hours", 4.0))
    return {
        "n_cases": int(n_cases),
        "n_methods": int(n_methods),
        "estimated_seconds": seconds,
        "estimated_hours": hours,
        "max_hours_without_checkpoint": max_hours,
        "requires_checkpoint": bool(hours > max_hours and not bool(config.get("h04_allow_long_run", False))),
    }


def _lsf_fwhm_from_config_or_qc(config, paths=None):
    for key in ("h04_lsf_fwhm_A", "h01_lsf_fwhm_A", "lsf_fwhm_A"):
        if config.get(key) is not None:
            return float(config[key]), f"config.{key}"
    qc = _read_optional_json(paths["stage00q_qc_json"]) if paths is not None else None
    m2 = (qc or {}).get("m2_lsf", {})
    for key in ("fwhm_at_halpha_A", "halpha_fwhm_A", "lsf_fwhm_A"):
        if m2.get(key) is not None:
            return float(m2[key]), f"stage00q_qc.m2_lsf.{key}"
    raise RuntimeError("H04 requires LSF FWHM at Halpha from config or stage00q QC.")


def _line_center_from_config(config):
    if config.get("h04_line_center_A") is not None:
        return float(config["h04_line_center_A"])
    rv = float(config.get("h04_rv_sys_kms", config.get("rv_sys_kms", config.get("systemic_rv_kms", 0.0))))
    return float(HALPHA_REST_A * (1.0 + rv / C_KMS))


def _injection_sigma(config):
    value = config.get("h04_injection_flux_sigma")
    if value is None:
        raise RuntimeError("H04 requires h04_injection_flux_sigma to convert input S/N into line flux.")
    val = float(value)
    if not np.isfinite(val) or val <= 0:
        raise RuntimeError("h04_injection_flux_sigma must be positive and finite.")
    return val


def _load_stage02_cube(paths, config):
    path = Path(config.get("h04_stage02_cube_fits", paths["stage02_cube_fits"]))
    if not path.exists():
        raise FileNotFoundError(path)
    with fits.open(path, memmap=False) as hdul:
        if "CUBES" in hdul:
            cube = hdul["CUBES"].data.astype(np.float64)
            wave = hdul["WAVELENGTH"].data.astype(np.float64) if "WAVELENGTH" in hdul else read_wavelength_axis(hdul)
        elif hdul[0].data is not None:
            cube = hdul[0].data.astype(np.float64)
            wave = read_wavelength_axis(hdul, data_shape=cube.shape[-3:])
        else:
            raise RuntimeError(f"No cube data found in {path}.")
    return cube, wave, path


def _load_psf_model(paths, config):
    if config.get("h04_psf_model") is not None:
        return config["h04_psf_model"], "config.h04_psf_model"
    path = Path(config.get("h04_psf_model_json", paths["psf_model_json"]))
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path), str(path)


def _coerce_spectrum_product(result):
    if isinstance(result, SpectrumProduct):
        extra = result.extra_columns or {}
        return {
            "wave_A": np.asarray(result.wave_A, dtype=np.float64),
            "flux": np.asarray(result.flux, dtype=np.float64),
            "flux_err": np.asarray(extra.get("flux_err_total", result.flux_err), dtype=np.float64),
            "flags": np.asarray(result.flags, dtype=np.int32),
            "continuum": np.asarray(extra["cont_runmed"], dtype=np.float64) if "cont_runmed" in extra else None,
        }
    if isinstance(result, dict):
        wave = result.get("wave_A", result.get("wavelengths_A", result.get("wavelengths")))
        err = result.get("flux_err", result.get("error", result.get("sigma")))
        if wave is None or result.get("flux") is None or err is None:
            raise ValueError("Extractor result dict must include wave_A/wavelengths, flux, and flux_err/error.")
        flux = np.asarray(result["flux"], dtype=np.float64)
        return {
            "wave_A": np.asarray(wave, dtype=np.float64),
            "flux": flux,
            "flux_err": np.asarray(err, dtype=np.float64),
            "flags": np.asarray(result.get("flags", np.zeros(flux.size, dtype=np.int32)), dtype=np.int32),
            "continuum": None if result.get("continuum") is None else np.asarray(result["continuum"], dtype=np.float64),
        }
    if isinstance(result, tuple) and len(result) >= 3:
        wave, flux, err = result[:3]
        flux = np.asarray(flux, dtype=np.float64)
        return {
            "wave_A": np.asarray(wave, dtype=np.float64),
            "flux": flux,
            "flux_err": np.asarray(err, dtype=np.float64),
            "flags": np.zeros(flux.size, dtype=np.int32),
            "continuum": None,
        }
    raise TypeError(f"Unsupported extractor result: {type(result)!r}")


def measure_recovery_with_h01_estimator(result, *, line_center_A, line_fwhm_A, continuum_window_A=80.0):
    product = _coerce_spectrum_product(result)
    flux = np.asarray(product["flux"], dtype=np.float64)
    wave = np.asarray(product["wave_A"], dtype=np.float64)
    base_good = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & np.isfinite(product["flux_err"])
        & (np.asarray(product["flux_err"]) > 0)
        & (np.asarray(product["flags"], dtype=np.int32) == 0)
    )
    if product["continuum"] is not None:
        flux = flux - np.asarray(product["continuum"], dtype=np.float64)
    else:
        # Match E1's residual construction (stage_h01 _control_residuals): subtract a
        # broad running-median continuum. The in-memory C2/C3/C4 products carry no D2
        # cont_runmed column, so without this the matched filter sees the continuum
        # pedestal (position-dependent star-halo level) as a spurious signal.
        min_pixels = max(3, min(15, int(np.count_nonzero(base_good))))
        cont = _CONTINUUM_RUNMED(wave, flux, base_good, window_A=float(continuum_window_A), min_pixels=min_pixels)
        flux = flux - cont
    good = base_good & np.isfinite(flux)
    recovered, sigma, z = H01_MATCHED_FILTER_POINT(
        product["wave_A"],
        flux,
        product["flux_err"],
        float(line_center_A),
        float(line_fwhm_A),
        good,
    )
    return {"recovered_flux": recovered, "recovered_sigma": sigma, "recovered_snr": z}


def _call_extractor(extractor, cube, wave_A, case, method, config):
    try:
        return extractor(cube=cube, wave_A=wave_A, case=case, method=method, config=config)
    except TypeError:
        try:
            return extractor(cube, wave_A, case, method, config)
        except TypeError:
            return extractor(cube, wave_A, case)


def _row_for_method(case, method, injected_flux, measurement, threshold_snr):
    recovered = float(measurement["recovered_flux"])
    sigma = float(measurement["recovered_sigma"])
    snr = float(measurement["recovered_snr"])
    throughput = np.nan if float(injected_flux) == 0 else recovered / float(injected_flux)
    bias = np.nan if float(injected_flux) == 0 else 100.0 * (recovered - float(injected_flux)) / float(injected_flux)
    return {
        "injection_id": case.injection_id,
        "variant": case.variant,
        "method": method,
        "position_label": case.position_label,
        "position_y": float(case.position_y),
        "position_x": float(case.position_x),
        "template_width": template_width_label(case.template_factor),
        "template_factor": float(case.template_factor),
        "continuum_mode": case.continuum_mode,
        "psf_fwhm_scale": float(case.psf_fwhm_scale),
        "snr": float(case.input_snr),
        "input_snr": float(case.input_snr),
        "injected_flux": float(injected_flux),
        "recovered_flux": recovered,
        "recovered_sigma": sigma,
        "recovered_snr": snr,
        "throughput": throughput,
        "throughput_err": 0.0,
        "complete": bool(np.isfinite(snr) and snr >= float(threshold_snr)),
        "bias_flux_pct": bias,
        "estimator": "stage_h01_detect.matched_filter_point",
    }


def _finite_values(rows, key):
    arr = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _float_or_nan(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _median_or_none(values):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.nanmedian(vals))


def _std_or_zero(values):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return 0.0
    return float(np.nanstd(vals, ddof=1))


def historic_regression_check(config, paths=None):
    expected = float(config.get("h04_historic_expected_snr", 8.97))
    recovered = config.get("h04_historic_recovered_snr")
    source = "config.h04_historic_recovered_snr"
    if recovered is None and paths is not None:
        truth = _read_optional_json(paths["stage06_truth_json"])
        if truth and truth.get("nominal_local_surface_matched_snr") is not None:
            recovered = truth["nominal_local_surface_matched_snr"]
            source = str(paths["stage06_truth_json"])
    if recovered is None:
        return {
            "expected_snr": expected,
            "recovered": None,
            "tolerance_snr": float(config.get("h04_historic_tolerance_snr", 0.25)),
            "source": "unavailable",
            "verdict": "unavailable",
        }
    recovered = float(recovered)
    tol = float(config.get("h04_historic_tolerance_snr", 0.25))
    return {
        "expected_snr": expected,
        "recovered": recovered,
        "tolerance_snr": tol,
        "source": source,
        "verdict": "pass" if abs(recovered - expected) <= tol else "fail",
    }


def _throughput_at_snr5(rows, methods):
    out = {}
    for method in methods:
        subset = [
            row
            for row in rows
            if row["method"] == method
            and row["variant"] == "nominal"
            and np.isclose(float(row["input_snr"]), 5.0)
            and np.isclose(float(row["template_factor"]), 1.0)
            and row["continuum_mode"] == "none"
        ]
        vals = _finite_values(subset, "throughput")
        if vals.size:
            out[method] = {"throughput": float(np.nanmedian(vals)), "err": _std_or_zero(vals), "n": int(vals.size)}
    return out


def _psf_perturbation_pct(rows, methods):
    pcts = []
    for method in methods:
        nominal = [
            row
            for row in rows
            if row["method"] == method
            and row["variant"] == "nominal"
            and row["position_label"] == "real"
            and row["continuum_mode"] == "none"
            and np.isclose(float(row["template_factor"]), 1.0)
            and float(row["input_snr"]) in {3.0, 5.0}
        ]
        for row in rows:
            if row["method"] != method or not str(row["variant"]).startswith("psf_"):
                continue
            candidates = [nom for nom in nominal if np.isclose(float(nom["input_snr"]), float(row["input_snr"]))]
            if not candidates:
                continue
            base = _median_or_none([candidate["throughput"] for candidate in candidates])
            if base is None or base == 0 or not np.isfinite(float(row["throughput"])):
                continue
            pcts.append(100.0 * abs(float(row["throughput"]) - base) / abs(base))
    return float(np.nanmedian(pcts)) if pcts else 0.0


def _bias_at_snr5(rows, methods):
    out = {}
    for method in methods:
        vals = [
            row["bias_flux_pct"]
            for row in rows
            if row["method"] == method and np.isclose(float(row["input_snr"]), 5.0) and row["variant"] == "nominal"
        ]
        med = _median_or_none(vals)
        if med is not None:
            out[method] = med
    return out


def _centroid_bias_placeholder(rows, methods):
    return {method: None for method in methods if any(row["method"] == method for row in rows)}


def _completeness(rows, methods, threshold_snr):
    out = {}
    for method in methods:
        subset = [
            row
            for row in rows
            if row["method"] == method
            and row["variant"] == "nominal"
            and np.isclose(float(row["template_factor"]), 1.0)
            and row["continuum_mode"] == "none"
            and np.isclose(float(row["input_snr"]), float(threshold_snr))
        ]
        if subset:
            out[method] = float(np.mean([bool(row["complete"]) for row in subset]))
    return out


def _v2_nulls_clean(rows, threshold_snr):
    nulls = [row for row in rows if row["variant"] == "nominal" and np.isclose(float(row["input_snr"]), 0.0)]
    hits = [row for row in nulls if np.isfinite(float(row["recovered_snr"])) and float(row["recovered_snr"]) >= threshold_snr]
    return {"status": "pass" if not hits else "fail", "n_nulls": len(nulls), "n_hits": len(hits)}


def _v3_monotonic(rows, methods):
    failures = []
    for method in methods:
        for factor in DEFAULT_TEMPLATE_FACTORS:
            subset = [
                row
                for row in rows
                if row["method"] == method
                and row["variant"] == "nominal"
                and row["continuum_mode"] == "none"
                and np.isclose(float(row["template_factor"]), factor)
                and float(row["input_snr"]) > 0
                and np.isfinite(float(row["throughput"]))
            ]
            by_snr = {}
            for row in subset:
                by_snr.setdefault(float(row["input_snr"]), []).append(float(row["throughput"]))
            if len(by_snr) < 2:
                continue
            snrs = sorted(by_snr)
            vals = [float(np.nanmedian(by_snr[snr])) for snr in snrs]
            if np.any(np.diff(vals) < -0.15):
                failures.append({"method": method, "template_factor": factor, "snr": snrs, "throughput": vals})
    return {"status": "pass" if not failures else "fail", "failures": failures}


def _v4_hierarchy(rows):
    med = {}
    for method in DEFAULT_METHODS:
        vals = [
            row["throughput"]
            for row in rows
            if row["method"] == method
            and row["variant"] == "nominal"
            and row["continuum_mode"] == "none"
            and np.isclose(float(row["template_factor"]), 1.0)
            and float(row["input_snr"]) >= 5.0
        ]
        if vals:
            med[method] = _median_or_none(vals)
    aperture = med.get("aperture")
    failures = []
    if aperture is not None:
        for method in ("optimal_ls", "optimal_psfsub", "psffit"):
            if med.get(method) is not None and med[method] + 0.05 < aperture:
                failures.append(method)
    return {"status": "pass" if not failures else "fail", "median_throughput": med, "failures": failures}


def _v5_continuum(rows, methods):
    out = {}
    failures = []
    for method in methods:
        no_cont = [
            row["throughput"]
            for row in rows
            if row["method"] == method
            and row["variant"] == "nominal"
            and row["continuum_mode"] == "none"
            and np.isclose(float(row["input_snr"]), 5.0)
            and np.isclose(float(row["template_factor"]), 1.0)
        ]
        flat = [
            row["throughput"]
            for row in rows
            if row["method"] == method
            and row["variant"] == "nominal"
            and row["continuum_mode"] == "flat"
            and np.isclose(float(row["input_snr"]), 5.0)
            and np.isclose(float(row["template_factor"]), 1.0)
        ]
        base = _median_or_none(no_cont)
        cont = _median_or_none(flat)
        if base is None or cont is None or base == 0:
            continue
        degradation = 100.0 * (base - cont) / abs(base)
        out[method] = degradation
        if degradation > 20.0:
            failures.append(method)
    return {"status": "pass" if not failures else "fail", "degradation_pct": out, "failures": failures}


def _qc_from_rows(config, paths, rows, methods, cases, budget, regression):
    threshold = float(config.get("h04_detection_threshold_snr", 5.0))
    psf_pct = _psf_perturbation_pct(rows, methods)
    checks = {
        "v1_regression": {"status": regression["verdict"], **regression},
        "v2_nulls_clean": _v2_nulls_clean(rows, threshold),
        "v3_monotonic": _v3_monotonic(rows, methods),
        "v4_hierarchy": _v4_hierarchy(rows),
        "v5_continuum": _v5_continuum(rows, methods),
    }
    open_issues = []
    if regression["verdict"] != "pass":
        open_issues.append("Historic Stage06 regression did not pass; H04 is not valid for E3.")
    for key, check in checks.items():
        if check.get("status") == "fail":
            open_issues.append(f"{key} failed.")
    return {
        "stage": "h04_injection_recovery",
        "run_id_base": str(config["run_id"]),
        "status": "complete",
        "clones_created": [],
        "regression_historic": regression,
        "grid": {
            "n_injections": int(
                len(
                    [
                        case
                        for case in cases
                        if case.variant == "nominal"
                    ]
                )
            ),
            "n_psf_perturbation": int(len([case for case in cases if case.variant != "nominal"])),
            "methods": list(methods),
        },
        "runtime_budget": budget,
        "throughput": {
            "per_method_at_snr5": _throughput_at_snr5(rows, methods),
            "psf_perturbation_pct": psf_pct,
        },
        "bias": {
            "flux_pct_at_snr5": _bias_at_snr5(rows, methods),
            "centroid_A": _centroid_bias_placeholder(rows, methods),
            "fwhm_pct": {method: None for method in methods},
        },
        "completeness_at_5sigma": _completeness(rows, methods, threshold),
        "checks": checks,
        "open_issues": open_issues,
    }


def _blocked_product(config, paths, positions, cases, methods, budget, reason):
    regression = historic_regression_check(config, paths)
    qc = {
        "stage": "h04_injection_recovery",
        "run_id_base": str(config["run_id"]),
        "status": "blocked",
        "blocked_reason": reason,
        "clones_created": [],
        "regression_historic": regression,
        "grid": {
            "n_injections": int(len([case for case in cases if case.variant == "nominal"])),
            "n_psf_perturbation": int(len([case for case in cases if case.variant != "nominal"])),
            "methods": list(methods),
            "positions": positions,
        },
        "runtime_budget": budget,
        "throughput": {"per_method_at_snr5": {}, "psf_perturbation_pct": 0.0},
        "bias": {"flux_pct_at_snr5": {}, "centroid_A": {}, "fwhm_pct": {}},
        "completeness_at_5sigma": {},
        "checks": {"v1_regression": {"status": regression["verdict"], **regression}},
        "open_issues": [reason],
    }
    return StageH04Product(rows=[], qc=_json_ready(qc))


def compute_stage_h04_products(config, paths=None, *, extractors=None, base_cube=None, wavelengths_A=None, psf_model=None):
    cfg = dict(config)
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_h04_paths(cfg["run_id"], root) if paths is None else paths
    methods = [str(method) for method in cfg.get("h04_methods", DEFAULT_METHODS)]
    positions = resolve_h04_positions(cfg, paths)
    cases = build_h04_cases(cfg, positions)
    budget = estimate_runtime_budget(cfg, len([case for case in cases if case.variant == "nominal"]), len(methods))
    if budget["requires_checkpoint"] and extractors is None:
        return _blocked_product(
            cfg,
            paths,
            positions,
            cases,
            methods,
            budget,
            "Estimated H04 runtime exceeds the checkpoint threshold; set h04_allow_long_run=true after review.",
        )
    if extractors is None:
        return _blocked_product(
            cfg,
            paths,
            positions,
            cases,
            methods,
            budget,
            "No H04 extractor runner was supplied; base run was not modified.",
        )
    if base_cube is None or wavelengths_A is None:
        base_cube, wavelengths_A, _cube_path = _load_stage02_cube(paths, cfg)
    if psf_model is None:
        psf_model, _psf_source = _load_psf_model(paths, cfg)

    line_center = _line_center_from_config(cfg)
    lsf_fwhm, _lsf_source = _lsf_fwhm_from_config_or_qc(cfg, paths)
    sigma_flux = _injection_sigma(cfg)
    continuum_flux_density = float(cfg.get("h04_continuum_flux_density", 0.0))
    threshold = float(cfg.get("h04_detection_threshold_snr", 5.0))
    continuum_window_A = float(cfg.get("h04_continuum_window_A", cfg.get("h01_continuum_window_A", 80.0)))

    def _rows_for_case(case):
        # Each case injects into its own cube copy and reads only shared,
        # read-only state (base_cube, psf_model, extractor closures) → the
        # cases are independent, so the per-case row list is deterministic and
        # order is restored by the caller regardless of execution order.
        injected_flux = float(case.input_snr) * sigma_flux
        continuum = continuum_flux_density if case.continuum_mode == "flat" else 0.0
        source = InjectionSource(
            y=case.position_y,
            x=case.position_x,
            total_line_flux=injected_flux,
            line_center_A=line_center,
            line_fwhm_A=lsf_fwhm * float(case.template_factor),
            label=case.injection_id,
            continuum_flux_density=continuum,
            psf_fwhm_scale=case.psf_fwhm_scale,
        )
        cube_injected = inject(base_cube, [source], wavelengths_A=wavelengths_A, psf_model=psf_model, copy=True)
        case_rows = []
        for method in methods:
            extractor = extractors[method] if isinstance(extractors, dict) else extractors
            result = _call_extractor(extractor, cube_injected, wavelengths_A, case, method, cfg)
            measurement = measure_recovery_with_h01_estimator(
                result,
                line_center_A=line_center,
                line_fwhm_A=lsf_fwhm * float(case.template_factor),
                continuum_window_A=continuum_window_A,
            )
            case_rows.append(_row_for_method(case, method, injected_flux, measurement, threshold))
        return case_rows

    case_n_jobs = _resolve_h04_n_jobs(cfg, len(cases))
    if case_n_jobs == 1:
        rows = [row for case in cases for row in _rows_for_case(case)]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=case_n_jobs) as pool:
            per_case = list(pool.map(_rows_for_case, cases))  # map preserves input order
        rows = [row for case_rows in per_case for row in case_rows]

    regression = historic_regression_check(cfg, paths)
    if bool(cfg.get("h04_require_historic_regression", True)) and regression["verdict"] != "pass":
        raise RuntimeError("H04 historic regression V1 must pass before publishing throughput.")
    qc = _qc_from_rows(cfg, paths, rows, methods, cases, budget, regression)
    return StageH04Product(rows=_json_ready(rows), qc=_json_ready(qc))


def clone_run_for_case(config, case, *, project_root=None):
    root = Path(project_root or config.get("project_root") or Path.cwd()).resolve()
    base_paths = RunPaths.from_project_root(config["run_id"], root)
    clone_id = f"{config['run_id']}_H04Inject_{case.injection_id}"
    clone_paths = RunPaths.from_project_root(clone_id, root)
    before = tree_sha256(base_paths.run_dir)
    create_run_clone(base_paths.run_dir, clone_paths.run_dir)
    after = tree_sha256(base_paths.run_dir)
    if before != after:
        raise RuntimeError("Base run hash changed while creating H04 clone.")
    return {"clone_id": clone_id, "clone_dir": clone_paths.run_dir, "base_hash": before}


def _plot_throughput(rows, path):
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    nominal = [row for row in rows if row.get("variant") == "nominal" and row.get("continuum_mode") == "none"]
    if nominal:
        for method in sorted({row["method"] for row in nominal}):
            subset = [row for row in nominal if row["method"] == method and np.isclose(float(row["template_factor"]), 1.0)]
            by_snr = {}
            for row in subset:
                throughput = _float_or_nan(row.get("throughput"))
                if np.isfinite(throughput):
                    by_snr.setdefault(float(row["input_snr"]), []).append(throughput)
            xs = sorted(by_snr)
            ys = [float(np.nanmedian(by_snr[x])) for x in xs]
            if xs:
                ax.plot(xs, ys, marker="o", label=method)
        ax.set_xlabel("Injected matched-filter S/N")
        ax.set_ylabel("Throughput")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "H04 not executed", ha="center", va="center")
        ax.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_recovered(rows, path):
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    finite = [
        row
        for row in rows
        if row.get("variant") == "nominal"
        and _float_or_nan(row.get("injected_flux")) > 0
        and np.isfinite(_float_or_nan(row.get("recovered_flux")))
    ]
    if finite:
        xs = np.asarray([_float_or_nan(row["injected_flux"]) for row in finite], dtype=np.float64)
        ys = np.asarray([_float_or_nan(row["recovered_flux"]) for row in finite], dtype=np.float64)
        ax.scatter(xs, ys, s=18, alpha=0.7)
        lo = min(float(np.nanmin(xs)), float(np.nanmin(ys)))
        hi = max(float(np.nanmax(xs)), float(np.nanmax(ys)))
        ax.plot([lo, hi], [lo, hi], color="0.3", ls="--", lw=1)
        ax.set_xlabel("Injected flux")
        ax.set_ylabel("Recovered flux")
    else:
        ax.text(0.5, 0.5, "H04 not executed", ha="center", va="center")
        ax.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_completeness(rows, path):
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    nominal = [row for row in rows if row.get("variant") == "nominal" and row.get("continuum_mode") == "none"]
    if nominal:
        for method in sorted({row["method"] for row in nominal}):
            subset = [row for row in nominal if row["method"] == method and np.isclose(float(row["template_factor"]), 1.0)]
            by_snr = {}
            for row in subset:
                by_snr.setdefault(float(row["input_snr"]), []).append(bool(row["complete"]))
            xs = sorted(by_snr)
            ys = [float(np.mean(by_snr[x])) for x in xs]
            if xs:
                ax.plot(xs, ys, marker="o", label=method)
        ax.set_xlabel("Injected matched-filter S/N")
        ax.set_ylabel("Completeness")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "H04 not executed", ha="center", va="center")
        ax.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_stage_h04_products(product: StageH04Product, config, paths):
    paths["paths"].ensure_base_dirs()
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    write_csv(paths["throughput_csv"], product.rows, fieldnames=TABLE_FIELDS)
    throughput_plot = _plot_throughput(product.rows, paths["throughput_plot"])
    recovery_plot = _plot_recovered(product.rows, paths["recovery_plot"])
    completeness_plot = _plot_completeness(product.rows, paths["completeness_plot"])
    qc = dict(product.qc)
    qc["tables"] = {"throughput_by_method": str(paths["throughput_csv"])}
    qc["figures"] = {
        "throughput": str(throughput_plot),
        "recovered_vs_injected": str(recovery_plot),
        "completeness": str(completeness_plot),
    }
    write_json(paths["stage_h04_qc_json"], _json_ready(qc))
    return {"table": paths["throughput_csv"], "qc_json": paths["stage_h04_qc_json"], "qc": qc}


def _derive_injection_sigma(cfg, paths, extractors, base_cube, wave_A, psf_model):
    """Self-consistent flux scale so input_snr == the reference method's S/N.

    The injected source is a broad AO PSF; a given aperture/optimal/psffit
    estimator only recovers a fraction of the total line flux, so the input S/N
    cannot be read off the total flux directly (spec E4 §4.1). We calibrate:
    inject a bright test source of total line flux F_cal at the real position,
    measure the reference method's recovered flux R and noise sigma, and set
    ``sigma_flux = sigma * F_cal / R``. Then ``input_snr * sigma_flux`` is the
    total flux whose reference-method matched-filter S/N equals ``input_snr``.
    Also returns the reference throughput R/F_cal for logging.
    """
    method = str(cfg.get("h04_sigma_reference_method", "psffit"))
    extractor = extractors[method] if isinstance(extractors, dict) else extractors
    line_center = _line_center_from_config(cfg)
    lsf_fwhm, _ = _lsf_fwhm_from_config_or_qc(cfg, paths)
    real = cfg["h04_real_position_yx"]

    # Baseline noise scale to size the calibration injection well above noise.
    baseline = _ZeroCase(position_y=float(real[0]), position_x=float(real[1]))
    base_meas = measure_recovery_with_h01_estimator(
        _call_extractor(extractor, base_cube, wave_A, baseline, method, cfg),
        line_center_A=line_center, line_fwhm_A=lsf_fwhm,
        continuum_window_A=float(cfg.get("h04_continuum_window_A", cfg.get("h01_continuum_window_A", 80.0))),
    )
    noise0 = float(base_meas["recovered_sigma"])
    if not np.isfinite(noise0) or noise0 <= 0:
        raise RuntimeError("Could not size the H04 calibration injection (baseline noise invalid).")

    f_cal = float(cfg.get("h04_calibration_snr", 100.0)) * noise0
    src = InjectionSource(
        y=float(real[0]), x=float(real[1]), total_line_flux=f_cal,
        line_center_A=line_center, line_fwhm_A=lsf_fwhm, label="h04_calib",
        continuum_flux_density=0.0, psf_fwhm_scale=1.0,
    )
    cube_cal = inject(base_cube, [src], wavelengths_A=wave_A, psf_model=psf_model, copy=True)
    cal_meas = measure_recovery_with_h01_estimator(
        _call_extractor(extractor, cube_cal, wave_A, baseline, method, cfg),
        line_center_A=line_center, line_fwhm_A=lsf_fwhm,
        continuum_window_A=float(cfg.get("h04_continuum_window_A", cfg.get("h01_continuum_window_A", 80.0))),
    )
    recovered = float(cal_meas["recovered_flux"])
    sigma_cal = float(cal_meas["recovered_sigma"])
    if not np.isfinite(recovered) or recovered <= 0 or not np.isfinite(sigma_cal) or sigma_cal <= 0:
        raise RuntimeError("H04 calibration injection did not recover positive flux; cannot set sigma_flux.")
    throughput_ref = recovered / f_cal
    sigma_flux = sigma_cal * f_cal / recovered
    return sigma_flux, {"reference_method": method, "throughput_ref": throughput_ref,
                        "f_cal": f_cal, "recovered_cal": recovered, "sigma_cal": sigma_cal}


class _ZeroCase:
    """Minimal case for a zero-injection baseline extraction."""

    def __init__(self, position_y, position_x):
        self.injection_id = "baseline"
        self.variant = "nominal"
        self.template_factor = 1.0
        self.psf_fwhm_scale = 1.0
        self.continuum_mode = "none"
        self.input_snr = 0.0
        self.position_y = float(position_y)
        self.position_x = float(position_x)
        self.position_label = "real"


def run_stage_h04(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False, extractors=None):
    cfg = stage_h04_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_h04_paths(cfg["run_id"], project_root=cfg.get("project_root"))

    base_cube = wave_A = psf_model = None
    if extractors is None and bool(cfg.get("h04_use_production_extractors", True)):
        from .stage_h04_extractors import build_production_extractors

        base_cube, wave_A, _cube_path = _load_stage02_cube(paths, cfg)
        base_cube = np.asarray(base_cube, dtype=np.float64)
        if base_cube.ndim == 4:
            if base_cube.shape[0] != 1:
                raise RuntimeError(
                    f"H04 in-memory extractors support a single cube; got {base_cube.shape[0]} "
                    "stacked cubes. Combine cubes upstream or supply extractors explicitly."
                )
            base_cube = base_cube[0]
        psf_model, _psf_source = _load_psf_model(paths, cfg)
        extractors = build_production_extractors(cfg, paths, wave_A=wave_A, psf_model=psf_model)
        if cfg.get("h04_injection_flux_sigma") is None:
            positions = resolve_h04_positions(cfg, paths)
            sigma_flux, calib_info = _derive_injection_sigma(
                {**cfg, "h04_real_position_yx": [positions[0]["y"], positions[0]["x"]]},
                paths, extractors, base_cube, wave_A, psf_model,
            )
            cfg["h04_injection_flux_sigma"] = sigma_flux
            cfg["h04_sigma_calibration"] = calib_info

    product = compute_stage_h04_products(
        cfg, paths, extractors=extractors, base_cube=base_cube,
        wavelengths_A=wave_A, psf_model=psf_model,
    )
    written = write_stage_h04_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="stage_h04_injection.py",
        description="Run Stage H04/E4 injection-recovery throughput calibration.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    args = parser.parse_args(argv)
    overrides = {"h04_allow_long_run": True} if args.allow_long_run else None
    result = run_stage_h04(
        args.run_id,
        project_root=args.project_root,
        overrides=overrides,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage_h04_qc_json"])


__all__ = [
    "DEFAULT_METHODS",
    "DEFAULT_SNR_GRID",
    "H01_MATCHED_FILTER_POINT",
    "H04Case",
    "StageH04Product",
    "build_h04_cases",
    "clone_run_for_case",
    "compute_stage_h04_products",
    "estimate_runtime_budget",
    "historic_regression_check",
    "measure_recovery_with_h01_estimator",
    "resolve_h04_positions",
    "run_stage_h04",
    "stage_h04_config_from_run",
    "stage_h04_paths",
    "write_stage_h04_products",
]


if __name__ == "__main__":
    main()
