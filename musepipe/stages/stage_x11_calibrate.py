"""Stage X11/D2: spectral calibration and final error budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..extraction.aperture import (
    FLAG_BAD_WINDOW,
    FLAG_SKYLINE,
    aperture_correction_from_psf,
    aperture_spectrum,
)
from ..extraction.product import SpectrumProduct
from ..io import (
    bunit_to_cgs_scale,
    flux_unit_conflict,
    flux_unit_from_m3_qc,
    read_json,
    resolve_flux_unit,
    write_json,
)
from ..paths import RunPaths
from ..reduction.sky_zap import SKYLINE_WINDOWS
from ..reduction.telluric import TELLURIC_BANDS
from ..spectral import (
    continuum_polyfit_loglambda,
    continuum_running_median,
    control_reference_bias,
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
    #: Calibracion absoluta DECLARADA y no plegada en `flux_err_total`: A4/M3 da
    #: `flux_factor` sin barra de error, asi que `flux_scale_err_frac` (el termino
    #: que si entra en el presupuesto) sale 0 y el desvio medido frente a Gaia se
    #: quedaba invisible. Se conserva como |1 - flux_factor| en su propia columna
    #: para poder citarlo sin mover el error de la ciencia congelada.
    flux_declared_err_frac: float = 0.0
    flux_declared_source: str = ""
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
    #: primaria calibrada (None si C4 no dejo `spec_psffit_star.fits`)
    star: CalibratedProduct | None = None


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
        "spec_sgf_object": paths.stage_dir / "spec_sgf_object.fits",
        "spec_lpm_object": paths.stage_dir / "spec_lpm_object.fits",
        "spec_final_object": paths.stage_dir / "spec_final_object.fits",
        "spec_calibrated_aperture_object": paths.stage_dir / "spec_calibrated_aperture_object.fits",
        "spec_calibrated_optimal_ls_object": paths.stage_dir / "spec_calibrated_optimal_ls_object.fits",
        "spec_calibrated_optimal_psfsub_object": paths.stage_dir / "spec_calibrated_optimal_psfsub_object.fits",
        "spec_calibrated_psffit_object": paths.stage_dir / "spec_calibrated_psffit_object.fits",
        "spec_calibrated_sgf_object": paths.stage_dir / "spec_calibrated_sgf_object.fits",
        "spec_calibrated_lpm_object": paths.stage_dir / "spec_calibrated_lpm_object.fits",
        # Primaria: C4 la extrae junto al compañero pero nadie la calibraba, asi
        # que quedaba sin correccion en lambda, sin escala de flujo y sin
        # presupuesto de error (7 columnas frente a las 18 del compañero).
        "spec_psffit_star": paths.stage_dir / "spec_psffit_star.fits",
        "spec_calibrated_psffit_star": paths.stage_dir / "spec_calibrated_psffit_star.fits",
        "stage_x11_qc_json": paths.stage_dir / "stage_x11_qc.json",
        "stage_x11_error_budget_png": paths.plot_dir / "stage_x11_error_budget.png",
        "stage_x11_continuum_png": paths.plot_dir / "stage_x11_continuum.png",
        "stage_x11_multimethod_png": paths.plot_dir / "stage_x11_multimethod.png",
        # Los espectros definitivos (6 metodos + primaria) como entregable, no
        # como diagnostico: `stage_x11_multimethod.png` superpone los 6 crudos
        # y sin unidad, que sirve para mirar de reojo pero no para presentar.
        "stage_x11_spectra_png": paths.plot_dir / "stage_x11_spectra.png",
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
    declared_frac, declared_source = _declared_fluxcal_from_m3(m3, err_frac)
    return (float(scale), float(max(err_frac, 0.0)), bool(m3.get("variability_caveat", True)),
            source, issues, declared_frac, declared_source)


def _declared_fluxcal_from_m3(m3, err_frac):
    """El desvio de calibracion absoluta medido, para DECLARARLO sin aplicarlo.

    M3 compara el flujo sintetico de la primaria con el catalogo de Gaia y deja
    `flux_factor`, pero **sin barra de error**: `err_frac` sale 0 y el ~3% que
    M3 acaba de medir no aparece en ningun sitio del presupuesto. Se toma
    |1 - flux_factor| como la magnitud declarada de ese sistematico.

    No se pliega en `flux_err_total` a proposito: hacerlo cambiaria el error del
    compañero, que sostiene decisiones congeladas (E1/E3/G3). Nota: G3 ya asume
    su propio 10% de calibracion absoluta (`g3_sys_fluxcal_frac`), asi que este
    valor es una cota inferior de lo que el ajuste atmosferico ya se cree.
    """
    if err_frac > 0:
        # Si M3 llega a publicar una barra de error, esa manda y ya va aplicada.
        return 0.0, ""
    factor = m3.get("flux_factor") if m3 else None
    if factor is None:
        return 0.0, ""
    factor = float(factor)
    if not np.isfinite(factor) or factor <= 0:
        return 0.0, ""
    return (
        float(abs(1.0 - factor)),
        f"|1 - stage00q_qc.m3_flux.flux_factor| = |1 - {factor:.3f}| ({m3.get('band', '?')} band, "
        "vs Gaia DR3); declarado, NO sumado a flux_err_total",
    )


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
    (flux_scale, flux_err, variability, flux_source, flux_issues,
     flux_declared, flux_declared_source) = _flux_scale_from_m3(m3)
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
        flux_declared_err_frac=float(cfg.get("x11_fluxcal_declared_frac", flux_declared)),
        flux_declared_source=flux_declared_source,
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


def _error_budget_rows(flux_err_stat, sys_fluxcal, sys_psf, sys_sky, sys_telluric, sys_continuum,
                       corrections, sys_fluxcal_declared=None):
    rows = [
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
    if corrections.flux_declared_err_frac > 0:
        rows.append({
            "term": "fluxcal_declared",
            "type": "declared_not_applied",
            "value": float(corrections.flux_declared_err_frac),
            "median": (None if sys_fluxcal_declared is None
                       else _finite_or_none(np.nanmedian(sys_fluxcal_declared))),
            "source": corrections.flux_declared_source,
            "note": (
                "Columna `sys_fluxcal_declared`. NO entra en `flux_err_total`: plegarlo moveria el "
                "error del compañero, que sostiene decisiones congeladas (E1/E3/G3). G3 ya asume su "
                "propio 10% de calibracion absoluta (g3_sys_fluxcal_frac), mayor que este valor."
            ),
        })
    return rows


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
    # DECLARADO y fuera de la suma: ver `_declared_fluxcal_from_m3`. Va despues
    # de `flux_err_total` justamente para que se vea que no entra en el.
    sys_fluxcal_declared = np.abs(flux) * float(corrections.flux_declared_err_frac)

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
    header["SYSFLXD"] = float(corrections.flux_declared_err_frac)
    header["SYSFLXDN"] = "sys_fluxcal_declared NOT in flux_err_total"
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
            "sys_fluxcal_declared": sys_fluxcal_declared,
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
        sys_fluxcal_declared=sys_fluxcal_declared,
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
        "sgf": Path(cfg.get("x11_spec_sgf_object", paths["spec_sgf_object"])),
        "lpm": Path(cfg.get("x11_spec_lpm_object", paths["spec_lpm_object"])),
    }


def calibrate_star_product(cfg, paths, corrections):
    """Calibra el espectro de la PRIMARIA con la misma cadena que el compañero.

    Devuelve `(CalibratedProduct, resumen_qc)` o `(None, resumen_qc)` si C4 no
    dejo el producto (p.ej. una cadena que no corrio psffit).

    Los dos terminos de error quedan en columnas SEPARADAS, no fundidos:

    * `flux_err_emp` — empirico de anillo. Es la dispersion del coeficiente de
      la primaria entre los N ajustes psffit de control, colocados en un anillo
      a la separacion del compañero alrededor de la estrella
      (`extraction.psffit.control_psffit_spectra`). Mide cuanto se mueve el
      flujo de la primaria segun donde se ponga la segunda componente: es un
      sistematico del ajuste medido sobre el dato, no ruido de fotones.
    * `flux_err_stat` + `sys_fluxcal` / `sys_psf` / `sys_sky` / `sys_telluric` /
      `sys_continuum` — el presupuesto de sistematicos de D2, identico al del
      compañero.
    * `flux_err_total` — la suma en cuadratura de ambos bloques.

    La primaria tiene S/N enorme, asi que su error NO esta dominado por el
    termino estadistico sino por el presupuesto (calibracion absoluta de flujo
    de A4/M3, PSF, telurico). Por eso importa poder mirarlos por separado.
    """
    path = Path(cfg.get("x11_spec_psffit_star", paths["spec_psffit_star"]))
    if not path.exists():
        return None, {"available": False, "reason": f"C4 no dejo {path.name} en este run"}
    product = SpectrumProduct.read(path)
    calibrated = calibrate_spectrum_product(
        product,
        corrections,
        method="psffit",
        canonical=False,
        continuum_window_A=float(cfg.get("x11_continuum_window_A", 80.0)),
        continuum_poly_deg=int(cfg.get("x11_continuum_poly_deg", 5)),
        error_smooth_channels=int(cfg.get("x11_error_smooth_channels", 21)),
    )
    header = calibrated.product.header
    header["SOURCE"] = "primary"
    header["EMPSRC"] = "psffit control ring at the companion separation (C4)"
    header["ERRSEP"] = "flux_err_emp (ring) and sys_* (budget) kept separate"

    flux = np.asarray(calibrated.product.flux, dtype=np.float64)
    extra = calibrated.product.extra_columns or {}
    summary = {
        "available": True,
        "input": str(path),
        "output": str(paths["spec_calibrated_psffit_star"]),
        "n_channels": int(flux.size),
        "median_snr_total": _finite_or_none(
            _median_finite(np.abs(flux) / np.asarray(extra["flux_err_total"], dtype=np.float64))
        ),
        "median_snr_emp_only": _finite_or_none(
            _median_finite(np.abs(flux) / np.asarray(calibrated.product.flux_err_emp, dtype=np.float64))
        ),
        "error_terms": {
            "empirical_ring": "flux_err_emp",
            "budget": ["flux_err_stat", "sys_fluxcal", "sys_psf", "sys_sky",
                       "sys_telluric", "sys_continuum"],
            "combined": "flux_err_total",
            "declared_not_combined": ["sys_fluxcal_declared"],
        },
        "median_error_fraction": {
            name: _finite_or_none(
                _median_finite(np.asarray(extra[name], dtype=np.float64) / np.abs(flux))
            )
            for name in ("flux_err_stat", "sys_fluxcal", "sys_fluxcal_declared", "sys_psf",
                         "sys_sky", "sys_telluric", "sys_continuum")
        },
        "median_empirical_fraction": _finite_or_none(
            _median_finite(np.asarray(calibrated.product.flux_err_emp, dtype=np.float64) / np.abs(flux))
        ),
        "note": (
            "La primaria es la fuente brillante: su error lo domina el presupuesto de "
            "sistematicos, no el termino estadistico. El empirico de anillo se conserva "
            "aparte porque mide otra cosa (estabilidad del ajuste), no ruido de fotones."
        ),
    }
    return calibrated, summary


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


# Raw (pre-calibration) control npz per method, used as a fallback when the
# calibrated control npz (written by the E1/E2 glue) is not present yet.
_RAW_CONTROL_NPZ = {
    "aperture": "spec_aperture_controls.npz",
    "optimal_ls": "spec_optimal_controls.npz",
    "optimal_psfsub": "spec_optimal_psfsub_controls.npz",
    "psffit": "spec_psffit_controls.npz",
    "sgf": "spec_sgf_controls.npz",
    "lpm": "spec_lpm_controls.npz",
}


def _method_control_bias(stage_dir, method, waves, good_mask, window_A, *, flux_scale=1.0):
    """Control-mean continuum bias for a method on the calibrated flux scale.

    Prefers the calibrated control npz; falls back to the raw control npz scaled
    by the D2 flux scale (so a fresh run without the E1/E2 glue still works).
    Returns None if no controls are available.
    """

    if stage_dir is None:
        return None
    stage_dir = Path(stage_dir)
    cal = stage_dir / f"spec_calibrated_{method}_controls.npz"
    if cal.exists():
        try:
            controls = np.load(cal)["control_spectra"]
            return control_reference_bias(waves, controls, good_mask, window_A=float(window_A))
        except (OSError, KeyError, ValueError):
            pass
    raw_name = _RAW_CONTROL_NPZ.get(method)
    if raw_name and (stage_dir / raw_name).exists():
        try:
            controls = np.load(stage_dir / raw_name)["control_spectra"] * float(flux_scale)
            return control_reference_bias(waves, controls, good_mask, window_A=float(window_A))
        except (OSError, KeyError, ValueError):
            pass
    return None


def _intermethod_continuum_report(
    calibrated, canonical_method, other_method, *, red_band_A=(8600.0, 9100.0), stage_dir=None, window_A=80.0, flux_scale=1.0
):
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

    # Control-mean referencing (D1 v2 §3.1, model-free): subtract each method's
    # own control-mean continuum bias, then re-measure the inter-method continuum
    # agreement. Both extractions carry a chromatic residual-halo bias of OPPOSITE
    # sign (psffit +, psfsub -); referencing removes the source-free baseline and
    # isolates the genuine method-dependent systematic. Reported alongside (does
    # not overwrite flux or drive the gate).
    after_ref = None
    ba = _method_control_bias(stage_dir, canonical_method, wave, good, window_A, flux_scale=flux_scale)
    bb = _method_control_bias(stage_dir, other_method, wave, good, window_A, flux_scale=flux_scale)
    if ba is not None and bb is not None:
        car, cbr = ca - ba, cb - bb
        inter_r = np.abs(car - cbr)
        frac_r = float(np.mean((inter_r <= comb)[good]))
        red_frac_r = float(np.mean((inter_r <= comb)[red])) if np.any(red) else None
        with np.errstate(divide="ignore", invalid="ignore"):
            red_ratio_r = float(np.nanmedian((car / cbr)[red])) if np.any(red) else None
        after_ref = {
            "method": "control_mean_referenced (each method minus its own control-mean continuum)",
            "fraction_channels_methods_agree": frac_r,
            "red_band_fraction_agree": _finite_or_none(red_frac_r),
            "red_band_median_ratio": _finite_or_none(red_ratio_r),
            "canonical_control_bias_red": _finite_or_none(
                float(np.nanmedian(ba[red])) if np.any(red) else None
            ),
            "other_control_bias_red": _finite_or_none(
                float(np.nanmedian(bb[red])) if np.any(red) else None
            ),
            "interpretation": (
                "Referencing each method to its own same-radius controls removes the source-free "
                "halo-subtraction pedestal; the remaining ratio is the genuine chromatic systematic."
            ),
            "note_d1": (
                "D1's per-band t-test already subtracts the control-mean difference (control-centered), "
                "so its divergent_continuum verdict already reflects this referenced systematic; only "
                "D2's raw-continuum v3 metric needed the same referencing (now applied)."
            ),
        }
    return {
        "canonical_vs": other_method,
        "fraction_channels_methods_agree": frac_agree,
        "red_band_A": [float(red_band_A[0]), float(red_band_A[1])],
        "red_band_median_ratio": _finite_or_none(red_ratio),
        "red_band_fraction_agree": _finite_or_none(red_frac_agree),
        "after_control_reference": after_ref,
        "interpretation": (
            "Continuum difference between the two G1-validated methods beyond their combined stat "
            "error = method-dependent systematic (residual chromatic halo subtraction). The companion "
            "red SPECTRAL SHAPE is real and shared by both methods (not flagged); only the "
            "method-dependent LEVEL discrepancy is."
        ),
    }


def _v3_continuum_block(intermethod, v3_threshold, canonical):
    """v3 continuum-stability gate on the control-referenced agreement.

    Uses the control-referenced inter-method agreement when available (consistent
    with D1's control-centered t-test); falls back to the raw agreement. Keeps the
    raw value and the legacy runmed-vs-poly fraction as secondary diagnostics.
    """

    if intermethod is None:
        return {
            "ok": None,
            "metric": "intermethod_continuum_agreement",
            "fraction_channels_methods_agree": None,
            "threshold": v3_threshold,
            "legacy_runmed_poly_fraction": canonical.continuum_summary["fraction_good_channels_sys_lt_staterr"],
        }
    raw_frac = intermethod["fraction_channels_methods_agree"]
    after = intermethod.get("after_control_reference")
    referenced = after is not None
    frac = after["fraction_channels_methods_agree"] if referenced else raw_frac
    return {
        "ok": bool(frac >= v3_threshold),
        "metric": (
            "intermethod_continuum_agreement_control_referenced" if referenced
            else "intermethod_continuum_agreement"
        ),
        "fraction_channels_methods_agree": frac,
        "raw_fraction_channels_methods_agree": raw_frac,
        "control_referenced": referenced,
        "threshold": v3_threshold,
        "legacy_runmed_poly_fraction": canonical.continuum_summary["fraction_good_channels_sys_lt_staterr"],
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
    # Step 1: deliver a control-referenced continuum for characterization (G3).
    # Add `continuum_bias` (each method's own control-mean continuum) and
    # `cont_runmed_biasref` = cont_runmed - bias as labeled extra columns; the
    # detection flux and cont_runmed are left untouched (E1/E3 unaffected).
    _stage_dir = paths["spec_final_object"].parent
    _cont_window_A = float(cfg.get("x11_continuum_window_A", 80.0))
    for method, cal in calibrated.items():
        prod = cal.product
        cont = np.asarray(prod.extra_columns["cont_runmed"], dtype=np.float64)
        bias = _method_control_bias(
            _stage_dir, method, np.asarray(prod.wave_A, dtype=np.float64),
            np.isfinite(cont), _cont_window_A, flux_scale=corrections.flux_scale,
        )
        if bias is not None:
            prod.extra_columns["continuum_bias"] = bias
            prod.extra_columns["cont_runmed_biasref"] = cont - bias

    canonical = calibrated[canonical_method]
    # Inter-method continuum systematic (isolates the real systematic from the
    # companion's real red spectral structure). The comparison method is the
    # other G1-validated method from D1's primary pair (default optimal_psfsub).
    # Spec D2 erratum (2026-07-15, era D1 v3): the intermethod CONTINUUM gate
    # needs a continuum-PRESERVING comparator. sgf destroys the companion
    # continuum by construction (Julo et al. 2025 Sect. 2.1; D1 v3
    # method_caveats), so a canonical-vs-sgf level ratio would measure the
    # method, not the systematic.
    continuum_comparator_excluded = {"sgf"}
    other_method = None
    for pair in (qc_x10 or {}).get("primary_pairs", []) or []:
        members = str(pair).split("_vs_")
        if canonical_method in members:
            partner = members[0] if members[1] == canonical_method else members[1]
            if partner in continuum_comparator_excluded:
                continue
            other_method = partner
            break
    if other_method is None or other_method not in calibrated:
        for fallback in ("lpm", "optimal_psfsub"):
            if fallback in calibrated and fallback != canonical_method:
                other_method = fallback
                break
        else:
            other_method = None
    intermethod = (
        _intermethod_continuum_report(
            calibrated,
            canonical_method,
            other_method,
            stage_dir=paths["spec_final_object"].parent,
            window_A=float(cfg.get("x11_continuum_window_A", 80.0)),
            flux_scale=corrections.flux_scale,
        )
        if other_method
        else None
    )
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
            # El desvio absoluto que M3 mide frente a Gaia, declarado aparte
            # porque M3 no publica barra de error y `scale_err` sale 0.
            "declared_err_frac": float(corrections.flux_declared_err_frac),
            "declared_source": corrections.flux_declared_source,
            "declared_applied": False,
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
            # v3 gates on the SIGNAL-FREE inter-method continuum agreement (real
            # systematic), not the signal-contaminated runmed-vs-poly. It now uses
            # the CONTROL-REFERENCED agreement (consistent with D1, whose per-band
            # t-test is already control-centered); comparing raw continua
            # double-counts the source-free halo pedestal. Raw kept as diagnostic.
            "v3_continuum_stable": _v3_continuum_block(intermethod, v3_threshold, canonical),
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
    star_calibrated, star_summary = calibrate_star_product(cfg, paths, corrections)
    qc["primary_star"] = star_summary
    if not star_summary.get("available"):
        qc["open_issues"].append(
            "Primary-star spectrum not calibrated: " + str(star_summary.get("reason", "unknown"))
        )
    qc["spectra"] = _spectra_summary(canonical_method, calibrated, star_calibrated)
    if not qc["spectra"]["unit_consistent"]:
        qc["open_issues"].append(
            "Los espectros calibrados no comparten una unidad declarada: "
            f"{qc['spectra']['unit']} (ver musepipe.io.resolve_bunit y B1/B2)."
        )
    qc["flux"]["unit"] = _flux_unit_block(cfg, qc00, calibrated[canonical_method].product)
    if qc["flux"]["unit"]["conflict"]:
        qc["open_issues"].append(qc["flux"]["unit"]["conflict"])
    return StageX11Product(
        canonical_method=canonical_method,
        products=products,
        calibrated=calibrated,
        qc=_json_ready(qc),
        star=star_calibrated,
    )


def _flux_unit_block(cfg, qc00, product) -> dict:
    """Las tres fuentes de la unidad de flujo, resueltas y contrastadas en D2.

    D2 no convierte a cgs (aplica un factor adimensional), pero es la etapa que
    publica los espectros definitivos: dejar aqui que escala van a resolver E3 y
    G3, y de donde sale, evita que cada una lo deduzca por su cuenta.

    El contraste importa mas que el valor: `flux_factor` se midio dividiendo por
    la unidad de M3 y D2 lo aplica a un flujo expresado en el `BUNIT` del
    producto. Si no son la misma, la escala absoluta sale mal por ese cociente y
    nadie se enteraria (ver `io.flux_unit_conflict`).
    """
    bunit = str(product.header.get("BUNIT", "")) or None
    block = {
        "bunit": bunit,
        "from_bunit": bunit_to_cgs_scale(bunit),
        "from_m3_qc": flux_unit_from_m3_qc(qc00),
        "conflict": flux_unit_conflict(bunit, qc00),
        "note": (
            "Escala a erg/s/cm2/A que resolveran E3/G3 (io.resolve_flux_unit): knob de config "
            "-> BUNIT del producto -> `m3_flux.flux_unit_cgs`. D2 no la aplica."
        ),
    }
    try:
        block["cgs"], block["source"] = resolve_flux_unit(cfg, bunit=bunit, qc_m3=qc00)
    except ValueError as exc:
        block["cgs"], block["source"] = None, None
        block["reason"] = str(exc)
    return block


#: Banda roja donde el compañero SE detecta. Las cifras de la tabla de espectros
#: se dan aqui y no sobre toda la rejilla: en el azul su S/N < 1, asi que una
#: mediana global mediria ruido y haria parecer inconsistentes a los 6 metodos.
SPECTRA_RED_BAND_A = (7500.0, 9000.0)

#: sgf destruye el continuo por construccion (Julo et al. 2025 Sect. 2.1): su
#: nivel no es comparable con el de los demas, solo su linea.
CONTINUUM_FREE_METHODS = ("sgf",)


def _spectrum_row(name, role, cal, *, canonical=False, red_reference=None):
    """Una fila de la tabla de espectros definitivos."""
    product = cal.product
    wave = np.asarray(product.wave_A, dtype=np.float64)
    flux = np.asarray(product.flux, dtype=np.float64)
    extra = product.extra_columns or {}
    err = np.asarray(extra.get("flux_err_total", product.flux_err), dtype=np.float64)
    red = (wave >= SPECTRA_RED_BAND_A[0]) & (wave <= SPECTRA_RED_BAND_A[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(flux) / err
    cont = np.asarray(
        extra.get("cont_runmed_biasref", extra.get("cont_runmed", flux)), dtype=np.float64
    )
    red_cont = _median_finite(cont[red])
    row = {
        "name": str(name),
        "role": str(role),
        "canonical": bool(canonical),
        "n_channels": int(flux.size),
        "wave_min_A": float(np.nanmin(wave)) if wave.size else None,
        "wave_max_A": float(np.nanmax(wave)) if wave.size else None,
        "bunit": str(product.header.get("BUNIT", "")),
        "flux_median": _finite_or_none(_median_finite(flux)),
        "flux_err_total_median": _finite_or_none(_median_finite(err)),
        "snr_median": _finite_or_none(_median_finite(snr)),
        "red_band_A": list(SPECTRA_RED_BAND_A),
        "red_flux_median": _finite_or_none(_median_finite(flux[red])),
        "red_snr_median": _finite_or_none(_median_finite(snr[red])),
        "red_continuum_median": _finite_or_none(red_cont),
        "ratio_to_canonical_red": None,
        "caveat": None,
    }
    if red_reference not in (None, 0.0) and np.isfinite(red_cont) and np.isfinite(red_reference):
        row["ratio_to_canonical_red"] = _finite_or_none(red_cont / float(red_reference))
    if name in CONTINUUM_FREE_METHODS:
        row["caveat"] = (
            "sgf filtra el continuo por construccion: su nivel (y su cociente al canonico) "
            "no es comparable; su valor esta en la linea, no en el continuo."
        )
    elif np.isfinite(red_cont) and red_cont < -abs(row["flux_err_total_median"] or 0.0):
        # Un cociente negativo desconcierta si no se dice de donde viene: es el
        # residuo de halo AO cromatico sobre-sustraido, no un error de signo.
        # Solo se avisa si el continuo negativo SUPERA el error total: un nivel
        # compatible con cero (sgf/lpm filtran el continuo) no es sobre-sustraccion.
        row["caveat"] = (
            "continuo negativo en el rojo: sobre-sustraccion del halo AO cromatico "
            "(docs/d2_red_continuum_diagnosis.md), emparejada en los controles. El cociente "
            "al canonico sale negativo por eso."
        )
    return row


def _spectra_summary(canonical_method, calibrated, star=None) -> dict:
    """Tabla de los espectros definitivos: los 6 metodos + la primaria.

    Los espectros calibrados son un resultado en si mismos, no solo la entrada
    de E1: el compañero por los 6 metodos (misma rejilla, superponibles) y la
    primaria, todos con unidad y error total. Esta seccion los deja listados en
    el QC para que el notebook los presente sin recalcular nada.
    """
    canonical_row = _spectrum_row(
        canonical_method, "companion", calibrated[canonical_method], canonical=True
    )
    reference = canonical_row["red_continuum_median"]
    rows = [canonical_row]
    for method in METHOD_ORDER:
        if method == canonical_method:
            continue
        rows.append(
            _spectrum_row(method, "companion", calibrated[method], red_reference=reference)
        )
    if star is not None:
        rows.append(_spectrum_row("psffit_star", "primary", star))
    waves = [np.asarray(cal.product.wave_A, dtype=np.float64) for cal in calibrated.values()]
    same_grid = all(
        w.size == waves[0].size and np.allclose(w, waves[0], rtol=0, atol=1e-6) for w in waves[1:]
    )
    units = {row["bunit"] for row in rows}
    return {
        "n_companion": sum(1 for row in rows if row["role"] == "companion"),
        "n_primary": sum(1 for row in rows if row["role"] == "primary"),
        "unit": sorted(units)[0] if len(units) == 1 else sorted(units),
        "unit_consistent": len(units) == 1 and "" not in units,
        "companions_share_grid": bool(same_grid),
        "red_band_A": list(SPECTRA_RED_BAND_A),
        "table": rows,
        "note": (
            "Medianas en la banda roja porque el compañero solo se detecta ahi. El cociente "
            "al canonico usa el continuo referenciado a controles (cont_runmed_biasref) "
            "cuando existe, que es la comparacion inter-metodo que gatea D1/v3."
        ),
    }


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

    plots = {
        "continuum": str(paths["stage_x11_continuum_png"]),
        "error_budget": str(paths["stage_x11_error_budget_png"]),
        "multimethod": str(paths["stage_x11_multimethod_png"]),
    }
    fig, _ = definitive_spectra_figure(paths["spec_final_object"].parent, plt=plt)
    out = paths["stage_x11_spectra_png"]
    fig.savefig(out, dpi=160)
    plt.close(fig)
    plots["spectra"] = str(out)
    return plots


#: Cubo que entra a 04b: el ultimo estado del dato ANTES de restarle nada al
#: compañero. `cube_residual_local_object.fits` es ese mismo cubo ya con la
#: superficie local quitada, y es de donde extrae C2.
UNSUBTRACTED_CUBE_NAME = "cube_input_local_object.fits"


def unsubtracted_aperture_reference(stage_dir, *, box_size=3, window_A=80.0):
    """La MISMA apertura simple de C2, pero sobre el cubo sin sustraer.

    Sirve de referencia comun para los 6 metodos: todos son, en el fondo, una
    forma distinta de quitar el halo de la primaria en la posicion del
    compañero, y sin una medida de *lo que habia antes de quitarlo* no se puede
    decir cuanto quito cada uno.

    Se replica la operacion de C2 pieza por pieza para que la comparacion sea
    de manzanas con manzanas: misma posicion (la de B3, no la redondeada de
    04b), misma apertura (box3), y la misma correccion de apertura por curva de
    crecimiento de C1 — que aqui vale ~40x, porque una caja 3x3 recoge una
    fraccion minuscula de la PSF de NFM. Lo unico que cambia es el cubo.

    Devuelve `None` (con motivo) si al run le falta alguna pieza, para que la
    figura degrade en vez de romperse.
    """
    from astropy.io import fits

    stage_dir = Path(stage_dir)
    cube_path = stage_dir / UNSUBTRACTED_CUBE_NAME
    if not cube_path.exists():
        return None, f"falta {UNSUBTRACTED_CUBE_NAME} (04b no dejo su cubo de entrada)"
    qc_b3 = stage_dir / "stage01c_qc.json"
    if not qc_b3.exists():
        return None, "falta stage01c_qc.json (B3 no localizo al compañero)"
    try:
        yx = read_json(qc_b3)["companion"]["pos_yx"]
    except (KeyError, TypeError, ValueError):
        return None, "stage01c_qc.json no declara companion.pos_yx"

    aperture = {"kind": "box", "size": int(box_size)}
    half = int(box_size) + 2
    row, col = int(round(float(yx[0]))), int(round(float(yx[1])))
    with fits.open(cube_path, memmap=True) as hdul:
        hdu = next((h for h in hdul if getattr(h.data, "ndim", 0) == 3), None)
        if hdu is None:
            return None, f"{UNSUBTRACTED_CUBE_NAME} no contiene un cubo 3D"
        # Solo se lee la ventana alrededor del compañero: el cubo entero son
        # ~400 MB y la apertura mira 3x3 pixeles.
        y0, x0 = max(row - half, 0), max(col - half, 0)
        stamp = np.asarray(hdu.data[:, y0:row + half + 1, x0:col + half + 1], dtype=np.float64)
    local_yx = (float(yx[0]) - y0, float(yx[1]) - x0)
    flux_box, _ = aperture_spectrum(stamp, local_yx, aperture)

    canonical_path = stage_dir / "spec_calibrated_psffit_object.fits"
    any_product = next(iter(sorted(stage_dir.glob("spec_calibrated_*_object.fits"))), None)
    ref_product = canonical_path if canonical_path.exists() else any_product
    if ref_product is None:
        return None, "no hay productos calibrados de los que tomar la rejilla de λ"
    wave = np.asarray(SpectrumProduct.read(ref_product).wave_A, dtype=np.float64)
    if wave.size != flux_box.size:
        return None, (f"la rejilla del cubo ({flux_box.size} canales) no coincide con la del "
                      f"producto ({wave.size}): no son comparables")

    psf_path = stage_dir / "psf_model.json"
    psf_model = read_json(psf_path) if psf_path.exists() else None
    apcorr, apcorr_mode, _ = aperture_correction_from_psf(
        wave, aperture, psf_model, center_yx=(float(yx[0]), float(yx[1]))
    )
    flux = flux_box * apcorr
    good = np.isfinite(flux)
    continuum = continuum_running_median(wave, flux, good, window_A=float(window_A))
    return {
        "wave_A": wave,
        "flux": flux,
        "flux_box": flux_box,
        "apcorr": apcorr,
        "apcorr_mode": apcorr_mode,
        "continuum": continuum,
        "aperture": f"box{int(box_size)}",
        "position_yx": [float(yx[0]), float(yx[1])],
        "source": str(cube_path),
    }, None


def halo_removal_figure(stage_dir, *, plt=None, canonical_method=None, window_A=80.0):
    """Los 6 metodos contra la apertura simple SIN sustraer (6 filas x 2 columnas).

    La comparacion inter-metodo del gate v3 usa dos metodos; esta usa los seis y
    contra una referencia externa a todos ellos, que es lo unico que permite
    responder "¿cuanto quito cada uno?" en vez de solo "¿se parecen entre si?".

    Por fila (un metodo):

    - izquierda, escala **symlog** compartida: el continuo sin sustraer (gris) y
      el del metodo. Hacen falta dos ordenes de magnitud en el mismo eje — en
      esa apertura el pedestal de halo es varias veces el compañero — y el
      residuo cruza el cero, asi que log a secas no vale.
    - derecha: **lo que QUEDA**, en % de lo que habia. Se dice asi y no "cuanto
      se quito" porque el 100% no es la meta: la referencia incluye tambien al
      compañero, asi que lo que debe quedar es justamente el (~18% en el rojo
      para psffit en ROXs 12 b). Lo que no admite discusion es el cero: por
      debajo se quito MAS de lo que habia, y eso es sobre-sustraccion
      (`optimal_ls` deja -47% en el rojo).

    Devuelve `(fig, axes)`, o `(None, motivo)` si falta la referencia.
    """
    if plt is None:  # pragma: no cover - conveniencia para uso interactivo
        import matplotlib.pyplot as plt
    stage_dir = Path(stage_dir)
    reference, reason = unsubtracted_aperture_reference(stage_dir, window_A=window_A)
    if reference is None:
        return None, reason

    products = {}
    for method in METHOD_ORDER:
        path = stage_dir / f"spec_calibrated_{method}_object.fits"
        if path.exists():
            products[method] = SpectrumProduct.read(path)
    if not products:
        return None, f"no hay productos calibrados en {stage_dir}"
    if canonical_method is None:
        canonical_method = next(
            (m for m, p in products.items() if bool(p.header.get("CANON", False))),
            next(iter(products)),
        )
    order = [canonical_method] + [m for m in products if m != canonical_method]

    wave = reference["wave_A"]
    ref_cont = np.asarray(reference["continuum"], dtype=np.float64)
    fig, axes = plt.subplots(
        len(order), 2, figsize=(13, 2.0 * len(order)), sharex=True,
        gridspec_kw={"width_ratios": [1.35, 1.0]}, constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    # Umbral lineal del symlog: el nivel del propio compañero, para que su
    # continuo NO quede aplastado contra el cero por el pedestal de halo.
    linthresh = float(np.nanpercentile(np.abs(ref_cont), 1)) or 1.0
    for i, method in enumerate(order):
        extra = products[method].extra_columns or {}
        cont = np.asarray(
            extra.get("cont_runmed", products[method].flux), dtype=np.float64
        )
        axl, axr = axes[i, 0], axes[i, 1]
        axl.plot(wave, ref_cont, lw=1.0, color="0.55",
                 label="sin sustraer" if i == 0 else None)
        axl.plot(wave, cont, lw=1.2, color="k" if method == canonical_method else "tab:blue",
                 label="tras restar" if i == 0 else None)
        axl.set_yscale("symlog", linthresh=linthresh)
        axl.axhline(0.0, color="0.8", lw=0.6)
        axl.set_ylabel(method + ("\n(canónico)" if method == canonical_method else ""), fontsize=8)
        if i == 0:
            axl.legend(fontsize=7, loc="lower right", ncol=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            remaining_pct = 100.0 * cont / ref_cont
        axr.plot(wave, remaining_pct, lw=1.0, color="tab:purple")
        axr.axhline(0.0, color="tab:red", lw=1.0, ls="--")
        axr.axhline(100.0, color="0.7", lw=0.7, ls=":")
        axr.set_ylim(*_robust_limits([remaining_pct], low=2, high=98, pad=0.25))
        axr.set_ylabel("% que queda", fontsize=7)
    axes[0, 0].set_title(
        "continuo: lo que hay en la apertura SIN restar (gris) vs lo que deja el método\n"
        "(eje symlog: el pedestal de halo es varias veces el compañero)", fontsize=9,
    )
    axes[0, 1].set_title(
        "lo que QUEDA, en % de lo que había\n"
        "en el rojo eso es el compañero · por debajo de 0 (rojo) se quitó de más", fontsize=9,
    )
    for ax in axes[-1, :]:
        ax.set_xlabel("λ [Å]")
    fig.suptitle(
        f"D2 · los {len(order)} métodos contra la misma apertura sin sustraer "
        f"({reference['aperture']} en la posición de B3, apcorr {reference['apcorr_mode']})",
        fontsize=11,
    )
    return fig, axes


def _robust_limits(series, *, low=0.5, high=99.5, pad=0.15):
    """Limites por percentil, para que un canal aislado no fije la escala."""
    values = np.concatenate([np.asarray(s, dtype=np.float64).ravel() for s in series])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if hi <= lo:
        return (lo - 1.0, hi + 1.0)
    margin = pad * (hi - lo)
    return (float(lo - margin), float(hi + margin))


def definitive_spectra_figure(stage_dir, *, canonical_method=None, plt=None, smooth_channels=15):
    """Los espectros definitivos, presentables: los 6 metodos + la primaria.

    Lee los productos calibrados de `stage_dir` (no re-ejecuta nada), asi que el
    notebook D2 la dibuja igual que la etapa en vez de duplicar el codigo.
    Devuelve `(fig, axes)`; quien llama decide si la guarda o la muestra.

    Tres paneles con el mismo eje λ porque son tres escalas distintas: la
    primaria es ~1e3-1e4 veces mas brillante que el compañero, y lo que se
    compara entre metodos es el continuo, no el flujo canal a canal. El flujo
    del compañero va suavizado para que se lean los 6 a la vez, sobre la banda
    de error total del canonico (sin suavizar).
    """
    if plt is None:  # pragma: no cover - conveniencia para uso interactivo
        import matplotlib.pyplot as plt
    stage_dir = Path(stage_dir)
    products = {}
    for method in METHOD_ORDER:
        path = stage_dir / f"spec_calibrated_{method}_object.fits"
        if path.exists():
            products[method] = SpectrumProduct.read(path)
    if not products:
        raise FileNotFoundError(f"No hay productos calibrados de D2 en {stage_dir}")
    star_path = stage_dir / "spec_calibrated_psffit_star.fits"
    star = SpectrumProduct.read(star_path) if star_path.exists() else None
    if canonical_method is None:
        canonical_method = next(
            (m for m, p in products.items() if bool(p.header.get("CANON", False))),
            next(iter(products)),
        )

    def _err(product):
        extra = product.extra_columns or {}
        return np.asarray(extra.get("flux_err_total", product.flux_err), dtype=np.float64)

    canonical = products[canonical_method]
    unit = str(canonical.header.get("BUNIT", "")) or "sin unidad declarada"
    nrows = 3 if star is not None else 2
    heights = ([1.0] if star is not None else []) + [1.5, 1.0]
    fig, axes = plt.subplots(
        nrows, 1, figsize=(11, 3.1 * nrows), sharex=True,
        gridspec_kw={"height_ratios": heights}, constrained_layout=True,
    )
    axes = list(np.atleast_1d(axes))
    panels = iter(axes)

    if star is not None:
        ax = next(panels)
        swave = np.asarray(star.wave_A, dtype=np.float64)
        sflux = np.asarray(star.flux, dtype=np.float64)
        serr = _err(star)
        ax.fill_between(swave, sflux - serr, sflux + serr, color="tab:orange", alpha=0.30,
                        lw=0, label="± error total (stat + sistematicos)")
        ax.plot(swave, sflux, lw=0.7, color="k", label="primaria (psffit)")
        ax.set_ylabel(f"flujo primaria\n[{unit}]", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")

    ax = next(panels)
    wave = np.asarray(canonical.wave_A, dtype=np.float64)
    cerr = _err(canonical)
    ax.fill_between(wave, -cerr, cerr, color="0.75", alpha=0.45, lw=0,
                    label="± error total (canonico)")
    drawn = []
    for method, product in products.items():
        is_canonical = method == canonical_method
        smoothed = median_filter_1d(np.asarray(product.flux, dtype=np.float64), width=smooth_channels)
        drawn.append(smoothed)
        ax.plot(
            np.asarray(product.wave_A, dtype=np.float64), smoothed,
            lw=1.4 if is_canonical else 0.8, color="k" if is_canonical else None,
            zorder=3 if is_canonical else 2,
            label=f"{method} (canonico)" if is_canonical else method,
        )
    # Escala robusta: un solo canal en el borde del notch AO (~6000 A) es 20
    # veces el continuo del compañero y aplastaria los 6 espectros.
    ax.set_ylim(*_robust_limits(drawn + [cerr, -cerr]))
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.axvline(6562.8, color="tab:red", ls=":", lw=1.0)
    ax.set_ylabel(f"flujo compañero\n[{unit}]", fontsize=8)
    ax.set_title(f"compañero: flujo suavizado {smooth_channels} canales (Hα en rojo)", fontsize=9)
    ax.legend(fontsize=7, ncol=4, loc="upper left")

    ax = next(panels)
    ax.axvspan(*SPECTRA_RED_BAND_A, color="tab:red", alpha=0.06, lw=0)
    for method, product in products.items():
        extra = product.extra_columns or {}
        cont = extra.get("cont_runmed_biasref", extra.get("cont_runmed"))
        if cont is None:
            continue
        ax.plot(
            np.asarray(product.wave_A, dtype=np.float64),
            np.asarray(cont, dtype=np.float64),
            lw=1.4 if method == canonical_method else 0.9,
            color="k" if method == canonical_method else None, label=method,
        )
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_xlabel("λ [Å]")
    ax.set_ylabel(f"continuo\n[{unit}]", fontsize=8)
    ax.set_title(
        "continuo referenciado a controles: la comparacion inter-metodo "
        f"(banda sombreada {SPECTRA_RED_BAND_A[0]:.0f}-{SPECTRA_RED_BAND_A[1]:.0f} A = donde el "
        "compañero se detecta)", fontsize=9,
    )
    ax.legend(fontsize=7, ncol=6, loc="upper left")
    fig.suptitle(
        f"D2 · espectros definitivos: {len(products)} metodos"
        + (" + la primaria" if star is not None else ""),
        fontsize=11,
    )
    return fig, axes


def write_stage_x11_products(product: StageX11Product, config, paths):
    paths["paths"].ensure_base_dirs()
    output_products = {}
    for method in METHOD_ORDER:
        out = _calibrated_output_path(paths, method)
        product.calibrated[method].product.write(out, overwrite=True)
        output_products[method] = str(out)
    product.calibrated[product.canonical_method].product.write(paths["spec_final_object"], overwrite=True)
    star_out = None
    if product.star is not None:
        star_out = paths["spec_calibrated_psffit_star"]
        product.star.product.write(star_out, overwrite=True)
    plots = _write_stage_x11_plots(product, paths)
    qc = dict(product.qc)
    qc["products"] = {
        "final_object": str(paths["spec_final_object"]),
        "calibrated_by_method": output_products,
        "calibrated_star": None if star_out is None else str(star_out),
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
