"""Stage X11/D2: spectral calibration and final error budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..extraction.aperture import FLAG_BAD_WINDOW, FLAG_SKYLINE
from ..extraction.product import SpectrumProduct
from ..io import read_json, write_json
from ..paths import RunPaths
from ..reduction.sky_zap import SKYLINE_WINDOWS
from ..reduction.telluric import TELLURIC_BANDS
from ..spectral import (
    continuum_polyfit_loglambda,
    continuum_running_median,
    median_filter_1d,
    standard_line_free_mask,
)
from .stage_x10_compare import METHOD_ORDER


BAD_CONTINUUM_FLAGS = FLAG_BAD_WINDOW | FLAG_SKYLINE


@dataclass(frozen=True)
class CalibrationCorrections:
    wavelength_status: str = "unavailable"
    wavelength_offset_A: float = 0.0
    wavelength_linear_a_A: float = np.nan
    wavelength_linear_b: float = np.nan
    wavelength_source: str = "stage00q_qc.m1_wavelength"
    wavelength_apply: bool = False
    frame_final: str = "unknown"
    flux_scale: float = 1.0
    flux_scale_err_frac: float = 0.0
    flux_source: str = "stage00q_qc.m3_flux"
    variability_caveat: bool = True
    psf_frac: float = 0.0
    psf_source: str = "stage_e01_qc.companion_ring_metric.residual_pct_median"
    sky_frac: float = 0.0
    sky_windows_A: tuple[tuple[float, float], ...] = tuple(SKYLINE_WINDOWS)
    sky_source: str = "stage00s_qc.systematic_frac"
    telluric_frac_by_band: dict[str, float] = field(default_factory=dict)
    telluric_bands_A: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(TELLURIC_BANDS))
    telluric_source: str = "stage00t_qc.verification.v1_residual_pct_by_band"
    open_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalibratedProduct:
    method: str
    product: SpectrumProduct
    already_applied: dict
    continuum_summary: dict
    error_budget: list[dict]


@dataclass(frozen=True)
class StageX11Product:
    canonical_method: str
    products: dict[str, SpectrumProduct]
    calibrated: dict[str, CalibratedProduct]
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


def stage_x11_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    return {
        "paths": paths,
        "stage00q_qc_json": paths.stage_dir / "stage00q_qc.json",
        "stage00s_qc_json": paths.stage_dir / "stage00s_qc.json",
        "stage00t_qc_json": paths.stage_dir / "stage00t_qc.json",
        "stage_e01_qc_json": paths.stage_dir / "stage_e01_qc.json",
        "stage_x10_qc_json": paths.stage_dir / "stage_x10_qc.json",
        "spec_aperture_object": paths.stage_dir / "spec_aperture_object.fits",
        "spec_optimal_object": paths.stage_dir / "spec_optimal_object.fits",
        "spec_optimal_psfsub_object": paths.stage_dir / "spec_optimal_psfsub_object.fits",
        "spec_psffit_object": paths.stage_dir / "spec_psffit_object.fits",
        "spec_final_object": paths.stage_dir / "spec_final_object.fits",
        "spec_calibrated_aperture_object": paths.stage_dir / "spec_calibrated_aperture_object.fits",
        "spec_calibrated_optimal_ls_object": paths.stage_dir / "spec_calibrated_optimal_ls_object.fits",
        "spec_calibrated_optimal_psfsub_object": paths.stage_dir / "spec_calibrated_optimal_psfsub_object.fits",
        "spec_calibrated_psffit_object": paths.stage_dir / "spec_calibrated_psffit_object.fits",
        "stage_x11_qc_json": paths.stage_dir / "stage_x11_qc.json",
        "stage_x11_error_budget_png": paths.plot_dir / "stage_x11_error_budget.png",
        "stage_x11_continuum_png": paths.plot_dir / "stage_x11_continuum.png",
        "stage_x11_multimethod_png": paths.plot_dir / "stage_x11_multimethod.png",
    }


def stage_x11_config_from_run(
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
    cfg.setdefault("x11_canonical_method", cfg.get("canonical_method"))
    cfg.setdefault("x11_continuum_window_A", 80.0)
    cfg.setdefault("x11_continuum_poly_deg", 5)
    cfg.setdefault("x11_error_smooth_channels", 21)
    cfg.setdefault("x11_allow_red_wavelength", False)
    cfg.setdefault("x11_allow_red_flux", False)
    return cfg


def _get_path(paths, key):
    path = Path(paths[key])
    if not path.exists():
        return None
    return path


def _read_optional_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return read_json(path)


def _median_finite(values, default=np.nan):
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.nanmedian(arr))


def _scatter_frac(values):
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < 2:
        return 0.0
    med = float(np.nanmedian(arr))
    if med <= 0 or not np.isfinite(med):
        return 0.0
    return float(np.nanstd(arr) / med)


def _flux_scale_from_m3(m3):
    if not m3:
        return 1.0, 0.0, True, "stage00q_qc.m3_flux unavailable", ["A4/M3 flux QC unavailable; using scale=1."]
    issues = []
    if "scale_factor" in m3:
        scale = float(m3["scale_factor"])
        source = "stage00q_qc.m3_flux.scale_factor"
    elif "factor_median" in m3:
        scale = float(m3["factor_median"])
        source = "stage00q_qc.m3_flux.factor_median"
    elif "median_factor" in m3:
        scale = float(m3["median_factor"])
        source = "stage00q_qc.m3_flux.median_factor"
    elif isinstance(m3.get("factor_by_band"), dict):
        vals = [float(v) for v in m3["factor_by_band"].values()]
        median_factor = _median_finite(vals, default=1.0)
        scale = 1.0 / median_factor if np.isfinite(median_factor) and median_factor != 0 else 1.0
        source = "inverse(stage00q_qc.m3_flux.factor_by_band_median)"
    elif "flux_factor" in m3 and m3.get("flux_factor") is not None:
        # A4/M3 orchestrator (compute_m3_flux): synthetic/catalog flux factor.
        factor = float(m3["flux_factor"])
        if np.isfinite(factor) and 0.8 <= factor <= 1.25:
            # Consistent with 1 within M3's uncertainty (yellow/green): the cube
            # is absolute-flux-validated against Gaia; no correction applied and
            # NOT an open issue (a positive cross-check, recorded in source).
            scale = 1.0
            source = (f"stage00q_qc.m3_flux.flux_factor={factor:.3f} ({m3.get('band','?')} band) "
                      "validated vs Gaia DR3, consistent with 1 -> scale=1")
        else:
            scale = 1.0 / factor if factor != 0 else 1.0
            source = f"inverse(stage00q_qc.m3_flux.flux_factor={factor:.3f})"
            issues.append(f"A4/M3 flux factor {factor:.3f} off unity; applied scale={scale:.3f}.")
    else:
        scale = 1.0
        source = "stage00q_qc.m3_flux missing scale; scale=1"
        issues.append("A4/M3 flux scale unavailable; using scale=1.")
    err_frac = 0.0
    for key in ("scale_err_frac", "scale_error_frac", "flux_scale_err_frac"):
        if key in m3:
            err_frac = float(m3[key])
            break
    else:
        for key in ("scale_err_pct", "scale_error_pct", "factor_scatter_pct"):
            if key in m3:
                err_frac = float(m3[key]) / 100.0
                break
        else:
            if isinstance(m3.get("factor_by_band"), dict):
                err_frac = _scatter_frac([float(v) for v in m3["factor_by_band"].values()])
    return float(scale), float(max(err_frac, 0.0)), bool(m3.get("variability_caveat", True)), source, issues


def _wavelength_from_m1(m1, *, allow_red=False):
    issues = []
    if not m1:
        issues.append("A4/M1 wavelength QC unavailable; no wavelength correction applied.")
        return "unavailable", 0.0, np.nan, np.nan, False, issues
    status = str(m1.get("status", "unavailable")).lower()
    if status == "red" and not allow_red:
        raise RuntimeError("A4/M1 wavelength status is red; D2 must stop before applying wavelength correction.")
    if status in {"unavailable", "unknown", "none"}:
        issues.append("A4/M1 wavelength correction unavailable; wavelength grid left in input frame.")
        return status, 0.0, np.nan, np.nan, False, issues
    offset = float(m1.get("offset_median_A", m1.get("dlambda_A", 0.0)))
    linear_a = float(m1.get("linear_a_A", np.nan))
    linear_b = float(m1.get("linear_b", m1.get("linear_term", np.nan)))
    return status, offset, linear_a, linear_b, True, issues


def _psf_frac_from_qc(qc):
    if not qc:
        return 0.0, ["C1 PSF QC unavailable; psf systematic set to zero."]
    metric = qc.get("companion_ring_metric", {})
    pct = metric.get("residual_pct_median", metric.get("median_pct", 0.0))
    if pct is None:
        return 0.0, ["C1 PSF QC lacks companion_ring_metric residual; psf systematic set to zero."]
    return max(float(pct), 0.0) / 100.0, []


def _sky_frac_from_qc(qc):
    if not qc:
        return 0.0, []
    for key in ("systematic_frac", "sky_systematic_frac", "residual_systematic_frac"):
        if key in qc:
            return max(float(qc[key]), 0.0), []
    decision = qc.get("decision", {})
    for key in ("systematic_frac", "sky_systematic_frac", "residual_systematic_frac"):
        if key in decision:
            return max(float(decision[key]), 0.0), []
    return 0.0, ["A2 sky QC has no explicit systematic fraction; sky term set to zero."]


def _telluric_fracs_from_qc(qc):
    if not qc:
        return {}, []
    fracs = {}
    explicit = qc.get("telluric_systematic_frac_by_band") or qc.get("systematic_frac_by_band")
    if isinstance(explicit, dict):
        for key, value in explicit.items():
            fracs[str(key)] = max(float(value), 0.0)
        return fracs, []
    residual = qc.get("verification", {}).get("v1_residual_pct_by_band", {})
    if isinstance(residual, dict):
        for key, value in residual.items():
            if value is not None:
                fracs[str(key)] = max(float(value), 0.0) / 100.0
    if fracs:
        return fracs, []
    return {}, ["A3 telluric QC has no residual/systematic by band; telluric term set to zero."]


def calibration_corrections_from_qc(qc00, qc_psf=None, qc_sky=None, qc_telluric=None, *, config=None):
    cfg = {} if config is None else dict(config)
    qc00 = {} if qc00 is None else qc00
    issues = []
    m1 = qc00.get("m1_wavelength", {})
    status, offset, linear_a, linear_b, apply_wl, wl_issues = _wavelength_from_m1(
        m1,
        allow_red=bool(cfg.get("x11_allow_red_wavelength", False)),
    )
    issues.extend(wl_issues)
    m3 = qc00.get("m3_flux", {})
    if str(m3.get("status", "unavailable")).lower() == "red" and not bool(cfg.get("x11_allow_red_flux", False)):
        raise RuntimeError("A4/M3 flux status is red; D2 must stop before applying flux scale.")
    flux_scale, flux_err, variability, flux_source, flux_issues = _flux_scale_from_m3(m3)
    issues.extend(flux_issues)
    psf_frac, psf_issues = _psf_frac_from_qc(qc_psf)
    issues.extend(psf_issues)
    sky_frac, sky_issues = _sky_frac_from_qc(qc_sky)
    issues.extend(sky_issues)
    telluric_fracs, telluric_issues = _telluric_fracs_from_qc(qc_telluric)
    issues.extend(telluric_issues)
    cube = qc00.get("cube", {})
    frame = str(cube.get("wavelength_frame", cfg.get("wavelength_frame", "unknown")))
    return CalibrationCorrections(
        wavelength_status=status,
        wavelength_offset_A=offset,
        wavelength_linear_a_A=linear_a,
        wavelength_linear_b=linear_b,
        wavelength_apply=apply_wl,
        frame_final=frame,
        flux_scale=flux_scale,
        flux_scale_err_frac=flux_err,
        flux_source=flux_source,
        variability_caveat=variability,
        psf_frac=float(cfg.get("x11_psf_frac", psf_frac)),
        sky_frac=float(cfg.get("x11_sky_frac", sky_frac)),
        telluric_frac_by_band={**telluric_fracs, **cfg.get("x11_telluric_frac_by_band", {})},
        open_issues=tuple(issues),
    )


def audit_already_applied(product: SpectrumProduct, corrections: CalibrationCorrections | None = None) -> dict:
    product.validate()
    header = product.header
    missing = [key for key in ("WFRAME", "APCMODE", "ERRMODE") if key not in header]
    if missing:
        raise ValueError(f"SpectrumProduct header is ambiguous for D2; missing {missing}.")
    wframe = str(header["WFRAME"])
    if corrections is not None and corrections.frame_final not in {"unknown", "unavailable", "", None}:
        if wframe != str(corrections.frame_final):
            raise ValueError(f"Product WFRAME={wframe!r} differs from A4 frame {corrections.frame_final!r}.")
    errmode = str(header["ERRMODE"]).lower()
    stat_in_flux_err = errmode == "stat"
    if stat_in_flux_err:
        missing_stat = [key for key in ("STATFAC", "COVFAC") if key not in header]
        if missing_stat:
            raise ValueError(f"STAT error mode declared but header lacks {missing_stat}.")
    return {
        "wframe": wframe,
        "apcorr": True,
        "apcorr_mode": str(header["APCMODE"]),
        "stat_factors_in_flux_err": bool(stat_in_flux_err),
        "errmode": str(header["ERRMODE"]),
    }


def wavelength_offset_model(wave_A, corrections: CalibrationCorrections):
    wave = np.asarray(wave_A, dtype=np.float64)
    if not corrections.wavelength_apply:
        return np.zeros_like(wave)
    if np.isfinite(corrections.wavelength_linear_a_A) and np.isfinite(corrections.wavelength_linear_b):
        return corrections.wavelength_linear_a_A + corrections.wavelength_linear_b * wave
    return np.full_like(wave, float(corrections.wavelength_offset_A), dtype=np.float64)


def apply_wavelength_correction(wave_A, corrections: CalibrationCorrections):
    wave = np.asarray(wave_A, dtype=np.float64)
    return wave - wavelength_offset_model(wave, corrections)


def corrected_skyline_residuals(offsets_A, corrections: CalibrationCorrections):
    offsets = np.asarray(offsets_A, dtype=np.float64)
    applied = -float(corrections.wavelength_offset_A) if corrections.wavelength_apply else 0.0
    return offsets + applied


def verify_wavelength_residuals(offsets_A, corrections: CalibrationCorrections, *, tolerance_A=0.05):
    residuals = corrected_skyline_residuals(offsets_A, corrections)
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        return {"ran": False, "ok": None, "max_abs_residual_A": None, "n_lines": 0}
    max_abs = float(np.nanmax(np.abs(finite)))
    return {"ran": True, "ok": bool(max_abs < float(tolerance_A)), "max_abs_residual_A": max_abs, "n_lines": int(finite.size)}


def conservative_stat_error(flux_err, flux_err_emp, *, smooth_channels=21):
    stat = np.asarray(flux_err, dtype=np.float64)
    emp = np.asarray(flux_err_emp, dtype=np.float64)
    if stat.shape != emp.shape:
        raise ValueError("flux_err and flux_err_emp must have matching shapes.")
    combined = np.fmax(stat, emp)
    stat_only = np.isfinite(stat) & ~np.isfinite(emp)
    emp_only = np.isfinite(emp) & ~np.isfinite(stat)
    combined[stat_only] = stat[stat_only]
    combined[emp_only] = emp[emp_only]
    combined[(~np.isfinite(stat)) & (~np.isfinite(emp))] = np.nan
    return median_filter_1d(combined, smooth_channels)


def _continuum_good_mask(product, wave, flux):
    flags = np.asarray(product.flags, dtype=np.int32)
    base = np.isfinite(wave) & np.isfinite(flux) & ((flags & BAD_CONTINUUM_FLAGS) == 0)
    return standard_line_free_mask(wave, base)


def _window_mask(wave, windows):
    wave = np.asarray(wave, dtype=np.float64)
    mask = np.zeros(wave.size, dtype=bool)
    for lo, hi in windows or ():
        mask |= (wave >= float(lo)) & (wave <= float(hi))
    return mask


def _telluric_sys(wave, flux, corrections):
    out = np.zeros_like(np.asarray(flux, dtype=np.float64))
    for name, frac in corrections.telluric_frac_by_band.items():
        if name not in corrections.telluric_bands_A:
            continue
        lo, hi = corrections.telluric_bands_A[name]
        mask = (wave >= float(lo)) & (wave <= float(hi))
        out[mask] = np.maximum(out[mask], np.abs(flux[mask]) * float(frac))
    return out


def _error_budget_rows(flux_err_stat, sys_fluxcal, sys_psf, sys_sky, sys_telluric, sys_continuum, corrections):
    return [
        {
            "term": "stat",
            "type": "per_channel",
            "median": _finite_or_none(np.nanmedian(flux_err_stat)),
            "source": "max(flux_err, flux_err_emp) smoothed",
        },
        {
            "term": "psf",
            "type": "localized",
            "median": _finite_or_none(np.nanmedian(sys_psf)),
            "value": float(corrections.psf_frac),
            "source": corrections.psf_source,
        },
        {
            "term": "fluxcal",
            "type": "global_pct",
            "value": float(corrections.flux_scale_err_frac),
            "median": _finite_or_none(np.nanmedian(sys_fluxcal)),
            "source": corrections.flux_source,
        },
        {
            "term": "sky",
            "type": "windows",
            "value": float(corrections.sky_frac),
            "median": _finite_or_none(np.nanmedian(sys_sky)),
            "source": corrections.sky_source,
        },
        {
            "term": "telluric",
            "type": "bands",
            "value": dict(corrections.telluric_frac_by_band),
            "median": _finite_or_none(np.nanmedian(sys_telluric)),
            "source": corrections.telluric_source,
        },
        {
            "term": "continuum",
            "type": "column_only",
            "median": _finite_or_none(np.nanmedian(sys_continuum)),
            "source": "D2 continuum runmed/poly difference",
        },
    ]


def calibrate_spectrum_product(
    product: SpectrumProduct,
    corrections: CalibrationCorrections,
    *,
    method: str,
    canonical: bool = False,
    continuum_window_A: float = 80.0,
    continuum_poly_deg: int = 5,
    error_smooth_channels: int = 21,
):
    already = audit_already_applied(product, corrections)
    wave = apply_wavelength_correction(product.wave_A, corrections)
    scale = float(corrections.flux_scale)
    flux = np.asarray(product.flux, dtype=np.float64) * scale
    flux_err_scaled = np.asarray(product.flux_err, dtype=np.float64) * abs(scale)
    flux_err_emp_scaled = np.asarray(product.flux_err_emp, dtype=np.float64) * abs(scale)
    flux_err_stat = conservative_stat_error(
        flux_err_scaled,
        flux_err_emp_scaled,
        smooth_channels=error_smooth_channels,
    )
    good_cont = _continuum_good_mask(product, wave, flux)
    min_pixels = max(5, min(15, int(np.count_nonzero(good_cont))))
    cont_runmed = continuum_running_median(
        wave,
        flux,
        good_cont,
        window_A=continuum_window_A,
        min_pixels=min_pixels,
    )
    cont_poly = continuum_polyfit_loglambda(
        wave,
        flux,
        good_cont,
        degree=continuum_poly_deg,
    )
    sys_continuum = np.abs(cont_runmed - cont_poly)
    sys_fluxcal = np.abs(flux) * float(corrections.flux_scale_err_frac)
    sys_psf = np.abs(flux) * float(corrections.psf_frac)
    sys_sky = np.zeros_like(flux)
    if corrections.sky_frac > 0:
        mask = _window_mask(wave, corrections.sky_windows_A)
        sys_sky[mask] = np.abs(flux[mask]) * float(corrections.sky_frac)
    sys_telluric = _telluric_sys(wave, flux, corrections)
    flux_err_total = np.sqrt(
        flux_err_stat**2 + sys_fluxcal**2 + sys_psf**2 + sys_sky**2 + sys_telluric**2
    )

    header = dict(product.header)
    header["SRCERRM"] = str(header.get("ERRMODE", "unknown"))
    header["ERRMODE"] = "total"
    header["CALSTAGE"] = "x11"
    header["CALMETH"] = str(method)
    header["CANON"] = bool(canonical)
    header["WLCORR"] = bool(corrections.wavelength_apply)
    header["DLAM_A"] = float(-corrections.wavelength_offset_A if corrections.wavelength_apply else 0.0)
    header["WLSRC"] = corrections.wavelength_source
    header["FLXSCL"] = float(scale)
    header["FLXSRC"] = corrections.flux_source
    header["SYSFLX"] = float(corrections.flux_scale_err_frac)
    header["ERRTOT"] = "stat+sys"
    header["CONTRUN"] = float(continuum_window_A)
    header["CONTPOL"] = int(continuum_poly_deg)

    extra = dict(product.extra_columns or {})
    extra.update(
        {
            "flux_err_stat": flux_err_stat,
            "flux_err_total": flux_err_total,
            "cont_runmed": cont_runmed,
            "cont_poly": cont_poly,
            "sys_continuum": sys_continuum,
            "sys_fluxcal": sys_fluxcal,
            "sys_psf": sys_psf,
            "sys_sky": sys_sky,
            "sys_telluric": sys_telluric,
        }
    )
    calibrated = SpectrumProduct(
        wave_A=wave,
        flux=flux,
        flux_err=flux_err_total,
        flux_err_emp=flux_err_emp_scaled,
        apcorr=np.asarray(product.apcorr, dtype=np.float64),
        npix_eff=np.asarray(product.npix_eff, dtype=np.float64),
        flags=np.asarray(product.flags, dtype=np.int32),
        header=header,
        covariance=None if product.covariance is None else np.asarray(product.covariance, dtype=np.float64) * scale**2,
        extra_columns=extra,
    )
    calibrated.validate()
    halpha = (wave >= 6540.0) & (wave <= 6590.0)
    ratio = np.nan
    if np.any(halpha):
        denom = np.nanmedian(flux_err_stat[halpha])
        numer = np.nanmedian(sys_continuum[halpha])
        if np.isfinite(denom) and denom > 0:
            ratio = float(numer / denom)
    stable = np.nan
    with np.errstate(invalid="ignore"):
        stable = float(np.nanmean((sys_continuum <= flux_err_stat)[good_cont])) if np.any(good_cont) else np.nan
    continuum_summary = {
        "runmed_window_A": float(continuum_window_A),
        "poly_deg": int(continuum_poly_deg),
        "sys_at_halpha_vs_staterr": _finite_or_none(ratio),
        "fraction_good_channels_sys_lt_staterr": _finite_or_none(stable),
    }
    budget = _error_budget_rows(
        flux_err_stat,
        sys_fluxcal,
        sys_psf,
        sys_sky,
        sys_telluric,
        sys_continuum,
        corrections,
    )
    return CalibratedProduct(
        method=method,
        product=calibrated,
        already_applied=already,
        continuum_summary=continuum_summary,
        error_budget=budget,
    )


def _product_paths_from_config(cfg, paths):
    return {
        "aperture": Path(cfg.get("x11_spec_aperture_object", paths["spec_aperture_object"])),
        "optimal_ls": Path(cfg.get("x11_spec_optimal_object", paths["spec_optimal_object"])),
        "optimal_psfsub": Path(cfg.get("x11_spec_optimal_psfsub_object", paths["spec_optimal_psfsub_object"])),
        "psffit": Path(cfg.get("x11_spec_psffit_object", paths["spec_psffit_object"])),
    }


def load_x11_products(product_paths):
    products = {}
    for method in METHOD_ORDER:
        path = Path(product_paths[method])
        if not path.exists():
            raise FileNotFoundError(path)
        products[method] = SpectrumProduct.read(path)
    return products


def _resolve_canonical_method(cfg, qc_x10):
    method = cfg.get("x11_canonical_method")
    if method:
        method = str(method)
    elif qc_x10:
        method = qc_x10.get("canonical_method") or qc_x10.get("selected_method")
    if not method:
        raise RuntimeError("D2 requires an explicit x11_canonical_method or a D1 QC selected_method.")
    aliases = {"optimal": "optimal_ls", "optimal-ls": "optimal_ls", "psfsub": "optimal_psfsub"}
    method = aliases.get(str(method).lower(), str(method).lower())
    if method not in METHOD_ORDER:
        raise ValueError(f"Unknown canonical method {method!r}; expected one of {METHOD_ORDER}.")
    return method


def _qc_for_wavelength_v1(qc00, corrections):
    m1 = (qc00 or {}).get("m1_wavelength", {})
    offsets = []
    if isinstance(m1.get("offsets_A"), list):
        offsets = m1["offsets_A"]
    else:
        try:
            offset = float(m1.get("offset_median_A", np.nan))
        except (TypeError, ValueError):
            offset = np.nan
        if np.isfinite(offset):
            offsets = [offset]
    return verify_wavelength_residuals(offsets, corrections)


def _intermethod_continuum_report(calibrated, canonical_method, other_method, *, red_band_A=(8600.0, 9100.0)):
    """Continuum systematic isolated from real companion signal (D2 v3).

    Both extraction methods share the REAL companion spectrum, so their continuum
    DIFFERENCE (beyond the combined statistical error) is the method-dependent
    systematic — chromatic residual halo subtraction — not real molecular/SED
    structure. This is the physically meaningful "is the continuum trustworthy"
    metric; ``|runmed - poly|`` conflates real broadband structure with it.
    """

    if other_method not in calibrated or canonical_method not in calibrated:
        return None
    a = calibrated[canonical_method].product
    b = calibrated[other_method].product
    wave = np.asarray(a.wave_A, dtype=np.float64)
    ca = np.asarray(a.extra_columns["cont_runmed"], dtype=np.float64)
    cb = np.asarray(b.extra_columns["cont_runmed"], dtype=np.float64)
    ea = np.asarray(a.extra_columns["flux_err_stat"], dtype=np.float64)
    eb = np.asarray(b.extra_columns["flux_err_stat"], dtype=np.float64)
    good = np.isfinite(ca) & np.isfinite(cb) & np.isfinite(ea) & np.isfinite(eb) & (ea > 0) & (eb > 0)
    if not np.any(good):
        return None
    comb = np.sqrt(ea**2 + eb**2)
    inter = np.abs(ca - cb)
    frac_agree = float(np.mean((inter <= comb)[good]))
    red = good & (wave >= float(red_band_A[0])) & (wave <= float(red_band_A[1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        red_ratio = float(np.nanmedian((ca / cb)[red])) if np.any(red) else None
    red_frac_agree = float(np.mean((inter <= comb)[red])) if np.any(red) else None
    return {
        "canonical_vs": other_method,
        "fraction_channels_methods_agree": frac_agree,
        "red_band_A": [float(red_band_A[0]), float(red_band_A[1])],
        "red_band_median_ratio": _finite_or_none(red_ratio),
        "red_band_fraction_agree": _finite_or_none(red_frac_agree),
        "interpretation": (
            "Continuum difference between the two G1-validated methods beyond their combined stat "
            "error = method-dependent systematic (residual chromatic halo subtraction). The companion "
            "red SPECTRAL SHAPE is real and shared by both methods (not flagged); only the "
            "method-dependent LEVEL discrepancy is."
        ),
    }


def compute_stage_x11_products(config, paths=None) -> StageX11Product:
    cfg = dict(config)
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_x11_paths(cfg["run_id"], root) if paths is None else paths
    qc00 = read_json(Path(cfg.get("x11_stage00q_qc_json", paths["stage00q_qc_json"])))
    qc_sky = _read_optional_json(Path(cfg.get("x11_stage00s_qc_json", paths["stage00s_qc_json"])))
    qc_telluric = _read_optional_json(Path(cfg.get("x11_stage00t_qc_json", paths["stage00t_qc_json"])))
    qc_psf = _read_optional_json(Path(cfg.get("x11_stage_e01_qc_json", paths["stage_e01_qc_json"])))
    qc_x10 = _read_optional_json(Path(cfg.get("x11_stage_x10_qc_json", paths["stage_x10_qc_json"])))
    canonical_method = _resolve_canonical_method(cfg, qc_x10)
    corrections = calibration_corrections_from_qc(
        qc00,
        qc_psf=qc_psf,
        qc_sky=qc_sky,
        qc_telluric=qc_telluric,
        config=cfg,
    )
    products = load_x11_products(_product_paths_from_config(cfg, paths))
    calibrated = {}
    for method, product in products.items():
        calibrated[method] = calibrate_spectrum_product(
            product,
            corrections,
            method=method,
            canonical=method == canonical_method,
            continuum_window_A=float(cfg.get("x11_continuum_window_A", 80.0)),
            continuum_poly_deg=int(cfg.get("x11_continuum_poly_deg", 5)),
            error_smooth_channels=int(cfg.get("x11_error_smooth_channels", 21)),
        )
    canonical = calibrated[canonical_method]
    # Inter-method continuum systematic (isolates the real systematic from the
    # companion's real red spectral structure). The comparison method is the
    # other G1-validated method from D1's primary pair (default optimal_psfsub).
    other_method = None
    for pair in (qc_x10 or {}).get("primary_pairs", []) or []:
        members = str(pair).split("_vs_")
        if canonical_method in members:
            other_method = members[0] if members[1] == canonical_method else members[1]
            break
    if other_method is None or other_method not in calibrated:
        other_method = "optimal_psfsub" if "optimal_psfsub" in calibrated else None
    intermethod = _intermethod_continuum_report(calibrated, canonical_method, other_method) if other_method else None
    v3_threshold = float(cfg.get("x11_continuum_agree_threshold", 0.90))
    qc = {
        "stage": "x11_spectral_calibration",
        "run_id": str(cfg["run_id"]),
        "canonical_method": canonical_method,
        "already_applied": canonical.already_applied,
        "wavelength": {
            "dlambda_A": float(-corrections.wavelength_offset_A if corrections.wavelength_apply else 0.0),
            "measured_offset_A": float(corrections.wavelength_offset_A),
            "linear_term": _finite_or_none(-corrections.wavelength_linear_b),
            "status": corrections.wavelength_status,
            "source": corrections.wavelength_source,
            "frame_final": corrections.frame_final,
        },
        "flux": {
            "scale_factor": float(corrections.flux_scale),
            "scale_err": float(corrections.flux_scale_err_frac),
            "source": corrections.flux_source,
            "variability_caveat": bool(corrections.variability_caveat),
        },
        "continuum": {
            **canonical.continuum_summary,
            # The runmed-vs-poly metric conflates the companion's REAL red
            # molecular/SED structure with systematics — kept as a secondary,
            # signal-contaminated diagnostic, NOT the v3 gate.
            "poly_smoothness_note": (
                "fraction_good_channels_sys_lt_staterr uses |runmed-poly| which also flags REAL "
                "broadband companion structure (cool-dwarf red SED); see intermethod_systematic for "
                "the signal-free continuum systematic."
            ),
            "intermethod_systematic": intermethod,
        },
        "error_budget": canonical.error_budget,
        "also_calibrated": [method for method in METHOD_ORDER if method != canonical_method],
        "checks": {
            "v1_skylines": _qc_for_wavelength_v1(qc00, corrections),
            # v3 now gates on the SIGNAL-FREE inter-method continuum agreement
            # (real systematic), not on the signal-contaminated runmed-vs-poly.
            "v3_continuum_stable": {
                "ok": None if intermethod is None else bool(
                    intermethod["fraction_channels_methods_agree"] >= v3_threshold
                ),
                "metric": "intermethod_continuum_agreement",
                "fraction_channels_methods_agree": None if intermethod is None else intermethod["fraction_channels_methods_agree"],
                "threshold": v3_threshold,
                "legacy_runmed_poly_fraction": canonical.continuum_summary["fraction_good_channels_sys_lt_staterr"],
            },
        },
        "open_issues": list(corrections.open_issues),
    }
    if intermethod is not None and intermethod.get("red_band_median_ratio") is not None:
        rr = intermethod["red_band_median_ratio"]
        if abs(rr - 1.0) > 0.25:
            qc["open_issues"].append(
                f"Red-band ({intermethod['red_band_A']} A) continuum LEVEL differs {rr:.2f}x between "
                f"{canonical_method} and {other_method}: a residual chromatic halo-subtraction systematic "
                "on the faint companion (the red SHAPE is real and method-consistent). Characterized, "
                "not removable at the current PSF ring residual (~4-5%)."
            )
    if canonical.continuum_summary["sys_at_halpha_vs_staterr"] is not None:
        if canonical.continuum_summary["sys_at_halpha_vs_staterr"] > 0.2:
            qc["open_issues"].append("Continuum systematic around Halpha exceeds 20 pct of local statistical error.")
    if qc["checks"]["v1_skylines"]["ok"] is False:
        qc["open_issues"].append("V1 skyline residuals exceed 0.05 A after wavelength correction.")
    return StageX11Product(
        canonical_method=canonical_method,
        products=products,
        calibrated=calibrated,
        qc=_json_ready(qc),
    )


def _calibrated_output_path(paths, method):
    return paths[f"spec_calibrated_{method}_object"]


def _write_stage_x11_plots(product: StageX11Product, paths):
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths["paths"].plot_dir.mkdir(parents=True, exist_ok=True)
    canonical = product.calibrated[product.canonical_method].product
    wave = canonical.wave_A
    extra = canonical.extra_columns or {}

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(wave, canonical.flux, lw=0.8, label="flux")
    ax.plot(wave, extra.get("cont_runmed", np.full_like(wave, np.nan)), lw=1.0, label="runmed")
    ax.plot(wave, extra.get("cont_poly", np.full_like(wave, np.nan)), lw=1.0, label="poly")
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("Flux")
    ax.legend(fontsize=8)
    fig.savefig(paths["stage_x11_continuum_png"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for key in ("flux_err_stat", "sys_fluxcal", "sys_psf", "sys_sky", "sys_telluric", "flux_err_total"):
        if key in extra:
            ax.plot(wave, extra[key], lw=0.9, label=key)
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("Error / systematic")
    ax.legend(fontsize=8)
    fig.savefig(paths["stage_x11_error_budget_png"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for method in METHOD_ORDER:
        cal = product.calibrated[method].product
        ax.plot(cal.wave_A, cal.flux, lw=0.8, label=method)
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("Calibrated flux")
    ax.legend(fontsize=8)
    fig.savefig(paths["stage_x11_multimethod_png"], dpi=160)
    plt.close(fig)
    return {
        "continuum": str(paths["stage_x11_continuum_png"]),
        "error_budget": str(paths["stage_x11_error_budget_png"]),
        "multimethod": str(paths["stage_x11_multimethod_png"]),
    }


def write_stage_x11_products(product: StageX11Product, config, paths):
    paths["paths"].ensure_base_dirs()
    output_products = {}
    for method in METHOD_ORDER:
        out = _calibrated_output_path(paths, method)
        product.calibrated[method].product.write(out, overwrite=True)
        output_products[method] = str(out)
    product.calibrated[product.canonical_method].product.write(paths["spec_final_object"], overwrite=True)
    plots = _write_stage_x11_plots(product, paths)
    qc = dict(product.qc)
    qc["products"] = {
        "final_object": str(paths["spec_final_object"]),
        "calibrated_by_method": output_products,
    }
    qc["plots"] = plots
    write_json(paths["stage_x11_qc_json"], _json_ready(qc))
    return {"products": qc["products"], "plots": plots, "qc_json": paths["stage_x11_qc_json"], "qc": qc}


def run_stage_x11(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_x11_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_x11_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_x11_products(cfg, paths)
    written = write_stage_x11_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="stage_x11_calibrate.py",
        description="Run Stage X11/D2 spectral calibration.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--canonical-method", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)
    overrides = {}
    if args.canonical_method:
        overrides["x11_canonical_method"] = args.canonical_method
    result = run_stage_x11(
        args.run_id,
        project_root=args.project_root,
        overrides=overrides,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage_x11_qc_json"])


__all__ = [
    "CalibrationCorrections",
    "CalibratedProduct",
    "StageX11Product",
    "apply_wavelength_correction",
    "audit_already_applied",
    "calibrate_spectrum_product",
    "calibration_corrections_from_qc",
    "compute_stage_x11_products",
    "conservative_stat_error",
    "corrected_skyline_residuals",
    "run_stage_x11",
    "stage_x11_config_from_run",
    "stage_x11_paths",
    "verify_wavelength_residuals",
    "write_stage_x11_products",
]


if __name__ == "__main__":
    main()
