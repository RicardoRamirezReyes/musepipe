"""Stage H01/E1: empirical Halpha detection on calibrated spectra."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..extraction.aperture import FLAG_BAD_WINDOW, FLAG_SKYLINE
from ..extraction.product import SpectrumProduct
from ..io import load_calibrated_controls, read_json, write_csv, write_json
from ..paths import RunPaths
from ..spectral import continuum_running_median
from .stage08c_look_elsewhere import empirical_fap, parametric_fap
from .stage_x10_compare import METHOD_ORDER


C_KMS = 299792.458
HALPHA_REST_A = 6562.8
DEFAULT_TEMPLATE_WIDTH_FACTORS = (1.0, 2.0, 4.0)
DEFAULT_ADMISSIBLE_PAIRS = (("psffit", "aperture"),)

#: de que columna sale la FAP que decide el veredicto. `empirical` cuenta
#: excedencias y no puede bajar de 1/(n+1); `parametric` ajusta una cola.
#: La eleccion se DECLARA en el config (`h01_fap_estimator`) y viaja al QC:
#: no hay defecto silencioso, porque cambia el veredicto.
FAP_ESTIMATORS = {"empirical": "global_empirical_fap", "parametric": "global_parametric_fap"}

#: por debajo de esto el ajuste no describe la nula y su FAP no vale
PARAMETRIC_FAP_MIN_KS_P = 0.05
BAD_DETECTION_FLAGS = FLAG_BAD_WINDOW | FLAG_SKYLINE


TABLE_FIELDS = [
    "method",
    "template_factor",
    "line_center_expected_A",
    "peak_wave_A",
    "peak_velocity_kms",
    "matched_flux",
    "matched_sigma",
    "matched_z",
    "local_empirical_fap",
    "global_empirical_fap",
    "centroid_A",
    "centroid_velocity_kms",
    "fwhm_A",
    "rv_consistent",
    "n_controls",
    "minimum_resolvable_fap",
    "global_parametric_fap",
    "parametric_fap_ks_p",
    "parametric_fap_extrapolation_sd",
]


@dataclass(frozen=True)
class H01MethodResult:
    method: str
    row: dict
    product: SpectrumProduct
    object_scan: dict
    null_maxima: np.ndarray
    control_scans: np.ndarray
    controls: np.ndarray


@dataclass(frozen=True)
class StageH01Product:
    method_results: dict[str, H01MethodResult]
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


def stage_h01_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage_h01")
    return {
        "paths": paths,
        "stage00q_qc_json": paths.stage_dir / "stage00q_qc.json",
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
        "halpha_detection_csv": paths.table_dir / "halpha_detection_by_method.csv",
        "null_maxima_npz": paths.stage_dir / "stage_h01_null_maxima.npz",
        "stage_h01_qc_json": paths.stage_dir / "stage_h01_qc.json",
        "plot_dir": plot_dir,
        "summary_plot": plot_dir / "stage_h01_summary.png",
    }


def stage_h01_config_from_run(
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
    cfg.setdefault("h01_halpha_rest_A", HALPHA_REST_A)
    cfg.setdefault("h01_search_half_width_kms", 500.0)
    cfg.setdefault("h01_template_width_factors", list(DEFAULT_TEMPLATE_WIDTH_FACTORS))
    cfg.setdefault("h01_detection_fap", 0.01)
    cfg.setdefault("h01_min_controls", 3)
    cfg.setdefault("h01_expected_controls", 31)
    cfg.setdefault("h01_admissible_pairs", [list(pair) for pair in DEFAULT_ADMISSIBLE_PAIRS])
    cfg.setdefault("h01_continuum_window_A", 80.0)
    cfg.setdefault("h01_placebo_centers_A", [6400.0, 6700.0])
    return cfg


def _product_paths_from_config(cfg, paths):
    return {
        "aperture": Path(cfg.get("h01_spec_calibrated_aperture_object", paths["spec_calibrated_aperture_object"])),
        "optimal_ls": Path(cfg.get("h01_spec_calibrated_optimal_ls_object", paths["spec_calibrated_optimal_ls_object"])),
        "optimal_psfsub": Path(
            cfg.get("h01_spec_calibrated_optimal_psfsub_object", paths["spec_calibrated_optimal_psfsub_object"])
        ),
        "psffit": Path(cfg.get("h01_spec_calibrated_psffit_object", paths["spec_calibrated_psffit_object"])),
        "sgf": Path(cfg.get("h01_spec_calibrated_sgf_object", paths["spec_calibrated_sgf_object"])),
        "lpm": Path(cfg.get("h01_spec_calibrated_lpm_object", paths["spec_calibrated_lpm_object"])),
    }


def _control_paths_from_config(cfg, paths):
    return {
        "aperture": Path(cfg.get("h01_controls_aperture_npz", paths["controls_calibrated_aperture_npz"])),
        "optimal_ls": Path(cfg.get("h01_controls_optimal_ls_npz", paths["controls_calibrated_optimal_ls_npz"])),
        "optimal_psfsub": Path(
            cfg.get("h01_controls_optimal_psfsub_npz", paths["controls_calibrated_optimal_psfsub_npz"])
        ),
        "psffit": Path(cfg.get("h01_controls_psffit_npz", paths["controls_calibrated_psffit_npz"])),
        "sgf": Path(cfg.get("h01_controls_sgf_npz", paths["controls_calibrated_sgf_npz"])),
        "lpm": Path(cfg.get("h01_controls_lpm_npz", paths["controls_calibrated_lpm_npz"])),
    }


def load_h01_products(product_paths):
    products = {}
    for method in METHOD_ORDER:
        path = Path(product_paths[method])
        if not path.exists():
            raise FileNotFoundError(path)
        products[method] = SpectrumProduct.read(path)
    return products


def load_h01_controls(control_paths, products, *, min_controls=3, product_paths=None):
    """Controles calibrados por metodo, con la guardia de procedencia de D2.

    Comprobar solo `shape[1] == n_wave` no basta: el eje espectral no cambia
    cuando cambia el cubo, asi que unos controles de otra reduccion pasan el
    filtro. Y aqui importa mas que en ningun sitio, porque cada control se
    normaliza con el error del OBJETO (ver `analyze_halpha_method`): sigma se
    cancela y el limite de E3 se reduce a `q99(flujo de los controles)`.
    """

    controls = {}
    for method in METHOD_ORDER:
        path = Path(control_paths[method])
        arr = load_calibrated_controls(
            path, object_path=None if product_paths is None else product_paths.get(method)
        )
        n_wave = products[method].wave_A.size
        if arr.ndim != 2 or arr.shape[1] != n_wave:
            raise ValueError(f"Controls for {method} must have shape (n_controls, {n_wave}).")
        if arr.shape[0] < int(min_controls):
            raise ValueError(f"Controls for {method} must include at least {min_controls} spectra.")
        controls[method] = arr
    return controls


def expected_line_center_A(rest_A, rv_sys_kms):
    return float(rest_A) * (1.0 + float(rv_sys_kms) / C_KMS)


def _channel_widths(wave_A):
    wave = np.asarray(wave_A, dtype=np.float64)
    if wave.size == 1:
        return np.ones(1, dtype=np.float64)
    edges = np.empty(wave.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
    edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
    return np.diff(edges)


def gaussian_flux_template(wave_A, center_A, fwhm_A):
    wave = np.asarray(wave_A, dtype=np.float64)
    sigma = float(fwhm_A) / 2.354820045
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Template FWHM must be positive.")
    profile = np.exp(-0.5 * ((wave - float(center_A)) / sigma) ** 2)
    norm = float(np.nansum(profile * _channel_widths(wave)))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Template has invalid normalization.")
    return profile / norm


def matched_filter_point(wave_A, residual, error, center_A, fwhm_A, good_mask):
    wave = np.asarray(wave_A, dtype=np.float64)
    resid = np.asarray(residual, dtype=np.float64)
    err = np.asarray(error, dtype=np.float64)
    template = gaussian_flux_template(wave, center_A, fwhm_A)
    good = np.asarray(good_mask, dtype=bool) & np.isfinite(resid) & np.isfinite(err) & (err > 0)
    good &= np.isfinite(template) & (template > 0)
    if int(np.count_nonzero(good)) < 2:
        return np.nan, np.nan, np.nan
    inv_var = 1.0 / (err[good] ** 2)
    denom = float(np.nansum(template[good] ** 2 * inv_var))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan, np.nan, np.nan
    flux = float(np.nansum(resid[good] * template[good] * inv_var) / denom)
    sigma = float(1.0 / np.sqrt(denom))
    return flux, sigma, flux / sigma


def _good_detection_mask(product):
    flags = np.asarray(product.flags, dtype=np.int32)
    err = np.asarray(product.extra_columns.get("flux_err_total", product.flux_err) if product.extra_columns else product.flux_err)
    return (
        np.isfinite(product.wave_A)
        & np.isfinite(product.flux)
        & np.isfinite(err)
        & (err > 0)
        & ((flags & BAD_DETECTION_FLAGS) == 0)
    )


def _object_residual(product):
    extra = product.extra_columns or {}
    if "cont_runmed" not in extra:
        raise ValueError("H01 requires D2 column cont_runmed in calibrated SpectrumProduct.")
    return np.asarray(product.flux, dtype=np.float64) - np.asarray(extra["cont_runmed"], dtype=np.float64)


def _control_residuals(wave, controls, good_mask, *, continuum_window_A=80.0):
    out = np.empty_like(np.asarray(controls, dtype=np.float64))
    min_pixels = max(3, min(15, int(np.count_nonzero(good_mask))))
    for i, spec in enumerate(np.asarray(controls, dtype=np.float64)):
        cont = continuum_running_median(
            wave,
            spec,
            good_mask & np.isfinite(spec),
            window_A=float(continuum_window_A),
            min_pixels=min_pixels,
        )
        out[i] = spec - cont
    return out


def search_centers_A(wave_A, expected_A, half_width_kms, good_mask):
    wave = np.asarray(wave_A, dtype=np.float64)
    half_A = float(expected_A) * float(half_width_kms) / C_KMS
    mask = np.asarray(good_mask, dtype=bool) & (wave >= expected_A - half_A) & (wave <= expected_A + half_A)
    centers = wave[mask]
    if centers.size == 0:
        raise ValueError("No good wavelength channels inside the Halpha search window.")
    return centers


def matched_filter_scan(
    wave_A,
    residual,
    error,
    centers_A,
    lsf_fwhm_A,
    width_factors=DEFAULT_TEMPLATE_WIDTH_FACTORS,
    good_mask=None,
):
    wave = np.asarray(wave_A, dtype=np.float64)
    good = np.isfinite(wave) if good_mask is None else np.asarray(good_mask, dtype=bool)
    centers = np.asarray(centers_A, dtype=np.float64)
    factors = tuple(float(value) for value in width_factors)
    z = np.full((len(factors), centers.size), np.nan, dtype=np.float64)
    flux = np.full_like(z, np.nan)
    sigma = np.full_like(z, np.nan)
    for i, factor in enumerate(factors):
        fwhm = float(lsf_fwhm_A) * factor
        for j, center in enumerate(centers):
            flux[i, j], sigma[i, j], z[i, j] = matched_filter_point(wave, residual, error, center, fwhm, good)
    return {"z": z, "flux": flux, "sigma": sigma, "centers_A": centers, "width_factors": factors}


def _scan_maximum(scan):
    z = np.asarray(scan["z"], dtype=np.float64)
    if not np.any(np.isfinite(z)):
        return None
    flat = int(np.nanargmax(z))
    i, j = np.unravel_index(flat, z.shape)
    return {
        "template_index": int(i),
        "center_index": int(j),
        "template_factor": float(scan["width_factors"][i]),
        "center_A": float(scan["centers_A"][j]),
        "z": float(z[i, j]),
        "flux": float(scan["flux"][i, j]),
        "sigma": float(scan["sigma"][i, j]),
    }


def _line_moments(wave, residual, center_A, fwhm_A):
    wave = np.asarray(wave, dtype=np.float64)
    resid = np.asarray(residual, dtype=np.float64)
    mask = np.isfinite(wave) & np.isfinite(resid) & (np.abs(wave - float(center_A)) <= 2.0 * float(fwhm_A))
    if int(np.count_nonzero(mask)) < 2:
        return None, None
    local_wave = wave[mask]
    positive = np.clip(resid[mask], 0.0, None)
    total = float(np.nansum(positive))
    if total <= 0 or not np.isfinite(total):
        return None, None
    centroid = float(np.nansum(local_wave * positive) / total)
    variance = float(np.nansum(((local_wave - centroid) ** 2) * positive) / total)
    return centroid, float(2.354820045 * np.sqrt(max(variance, 0.0)))


def analyze_halpha_method(
    method,
    product,
    controls,
    *,
    rv_sys_kms,
    rv_sys_err_kms=0.0,
    lsf_fwhm_A,
    rest_A=HALPHA_REST_A,
    search_half_width_kms=500.0,
    width_factors=DEFAULT_TEMPLATE_WIDTH_FACTORS,
    continuum_window_A=80.0,
    parametric_family="gumbel",
    parametric_n_boot=0,
    parametric_seed=None,
):
    product.validate()
    if str(product.header.get("WFRAME", "")).lower() not in {"barycentric", "topocentric"}:
        raise ValueError(f"Unknown wavelength frame for {method}: {product.header.get('WFRAME')!r}.")
    wave = np.asarray(product.wave_A, dtype=np.float64)
    good = _good_detection_mask(product)
    expected = expected_line_center_A(rest_A, rv_sys_kms)
    centers = search_centers_A(wave, expected, search_half_width_kms, good)
    error = np.asarray((product.extra_columns or {}).get("flux_err_total", product.flux_err), dtype=np.float64)
    object_resid = _object_residual(product)
    object_scan = matched_filter_scan(wave, object_resid, error, centers, lsf_fwhm_A, width_factors, good)
    maximum = _scan_maximum(object_scan)
    if maximum is None:
        raise ValueError(f"No finite matched-filter value for {method}.")
    control_resid = _control_residuals(wave, controls, good, continuum_window_A=continuum_window_A)
    control_scans = []
    null_maxima = []
    for control in control_resid:
        scan = matched_filter_scan(wave, control, error, centers, lsf_fwhm_A, width_factors, good)
        control_scans.append(scan["z"])
        row = _scan_maximum(scan)
        null_maxima.append(np.nan if row is None else row["z"])
    control_scans = np.asarray(control_scans, dtype=np.float64)
    null_maxima = np.asarray(null_maxima, dtype=np.float64)
    local_null = control_scans[:, maximum["template_index"], maximum["center_index"]]
    local_fap = empirical_fap(maximum["z"], local_null)
    global_fap = empirical_fap(maximum["z"], null_maxima)
    parametrica = parametric_fap(maximum["z"], null_maxima,
                                 family=parametric_family, n_boot=int(parametric_n_boot),
                                 seed=parametric_seed)
    peak_velocity = C_KMS * (maximum["center_A"] / float(rest_A) - 1.0)
    centroid, fwhm = _line_moments(
        wave,
        object_resid,
        maximum["center_A"],
        float(lsf_fwhm_A) * maximum["template_factor"],
    )
    centroid_velocity = None if centroid is None else C_KMS * (centroid / float(rest_A) - 1.0)
    rv_tol = np.sqrt(float(rv_sys_err_kms) ** 2 + (C_KMS * float(lsf_fwhm_A) / float(rest_A)) ** 2)
    velocity_for_check = peak_velocity if centroid_velocity is None else centroid_velocity
    rv_consistent = bool(abs(float(velocity_for_check) - float(rv_sys_kms)) <= rv_tol)
    row = {
        "method": method,
        "template_factor": maximum["template_factor"],
        "line_center_expected_A": float(expected),
        "peak_wave_A": maximum["center_A"],
        "peak_velocity_kms": float(peak_velocity),
        "matched_flux": maximum["flux"],
        "matched_sigma": maximum["sigma"],
        "matched_z": maximum["z"],
        "local_empirical_fap": _finite_or_none(local_fap),
        "global_empirical_fap": _finite_or_none(global_fap),
        "centroid_A": _finite_or_none(centroid),
        "centroid_velocity_kms": _finite_or_none(centroid_velocity),
        "fwhm_A": _finite_or_none(fwhm),
        "rv_consistent": bool(rv_consistent),
        "n_controls": int(np.asarray(controls).shape[0]),
        "minimum_resolvable_fap": float(1.0 / (np.asarray(controls).shape[0] + 1)),
        "global_parametric_fap": _finite_or_none(parametrica["fap"]),
        "parametric_fap_ks_p": _finite_or_none(parametrica["ks_p"]),
        "parametric_fap_extrapolation_sd": _finite_or_none(parametrica["extrapolation_sd"]),
    }
    row["_parametric"] = parametrica
    return H01MethodResult(
        method=method,
        row=row,
        product=product,
        object_scan=object_scan,
        null_maxima=null_maxima,
        control_scans=control_scans,
        controls=np.asarray(controls, dtype=np.float64),
    )


def classify_h01_verdict(rows, *, detection_fap=0.01, admissible_pairs=DEFAULT_ADMISSIBLE_PAIRS,
                         fap_estimator="empirical"):
    """El veredicto, leyendo la FAP del estimador DECLARADO.

    `empirical` no puede bajar de `1/(n+1)`; con los 33 controles que caben a la
    separacion de la companera su suelo es 0.029 y el criterio de 0.01 es
    inalcanzable con cualquier dato. `parametric` ajusta una cola a la misma
    nula. Cual manda se declara en el config y viaja al QC: cambia el veredicto,
    asi que no puede decidirse por defecto.
    """
    if str(fap_estimator) not in FAP_ESTIMATORS:
        raise ValueError(
            f"`h01_fap_estimator` desconocido: {fap_estimator!r}. "
            f"Admitidos: {sorted(FAP_ESTIMATORS)}."
        )
    campo = FAP_ESTIMATORS[str(fap_estimator)]
    significant = {
        row["method"]
        for row in rows
        if row.get(campo) is not None
        and float(row[campo]) < float(detection_fap)
        and bool(row["rv_consistent"])
    }
    fap_hits = {
        row["method"]
        for row in rows
        if row.get(campo) is not None and float(row[campo]) < float(detection_fap)
    }
    for left, right in admissible_pairs:
        if left in significant and right in significant:
            return {
                "verdict": "detection",
                "reason": "admissible_independent_pair_passes_fap_and_rv",
                "significant_methods": sorted(significant),
                "fap_estimator": str(fap_estimator),
            }
    if fap_hits:
        return {
            "verdict": "candidate",
            "reason": "fap_hit_without_two_rv_consistent_admissible_methods",
            "significant_methods": sorted(significant),
            "fap_hit_methods": sorted(fap_hits),
            "fap_estimator": str(fap_estimator),
        }
    return {"verdict": "non_detection", "reason": "no_method_passes_global_fap",
            "significant_methods": [], "fap_estimator": str(fap_estimator)}


def _lsf_fwhm_from_qc_or_config(qc00, cfg):
    if cfg.get("h01_lsf_fwhm_A") is not None:
        return float(cfg["h01_lsf_fwhm_A"]), "config.h01_lsf_fwhm_A"
    m2 = (qc00 or {}).get("m2_lsf", {})
    for key in ("fwhm_at_halpha_A", "halpha_fwhm_A", "lsf_fwhm_A"):
        if key in m2 and m2[key] is not None:
            return float(m2[key]), f"stage00q_qc.m2_lsf.{key}"
    coeffs = m2.get("poly2_coeffs")
    if coeffs:
        return float(np.polyval(np.asarray(coeffs, dtype=np.float64), HALPHA_REST_A)), "stage00q_qc.m2_lsf.poly2_coeffs"
    table = m2.get("table_A_fwhm")
    if table:
        waves = np.asarray([row[0] if isinstance(row, (list, tuple)) else row.get("wave_A") for row in table], dtype=float)
        fwhm = np.asarray([row[1] if isinstance(row, (list, tuple)) else row.get("fwhm_A") for row in table], dtype=float)
        return float(np.interp(HALPHA_REST_A, waves, fwhm)), "stage00q_qc.m2_lsf.table_A_fwhm"
    raise RuntimeError("H01 requires LSF FWHM at Halpha from A4/M2 or h01_lsf_fwhm_A config.")


def _rv_from_config(cfg):
    for key in ("h01_rv_sys_kms", "rv_sys_kms", "systemic_rv_kms"):
        if cfg.get(key) is not None:
            return float(cfg[key]), key
    raise RuntimeError("H01 requires systemic RV in h01_rv_sys_kms, rv_sys_kms, or systemic_rv_kms.")


def _rv_err_from_config(cfg):
    for key in ("h01_rv_sys_err_kms", "rv_sys_err_kms", "systemic_rv_err_kms"):
        if cfg.get(key) is not None:
            return float(cfg[key])
    return 0.0


def _control_summary(result: H01MethodResult):
    vals = result.null_maxima[np.isfinite(result.null_maxima)]
    if vals.size == 0:
        return {"n": 0, "median": None, "p95": None, "max": None, "outlier_count": None}
    med = float(np.nanmedian(vals))
    mad = float(1.4826 * np.nanmedian(np.abs(vals - med)))
    outliers = 0 if not np.isfinite(mad) or mad <= 0 else int(np.count_nonzero(vals > med + 8.0 * mad))
    return {
        "n": int(vals.size),
        "median": med,
        "p95": float(np.nanpercentile(vals, 95.0)),
        "max": float(np.nanmax(vals)),
        "outlier_count": outliers,
    }


def _placebo_checks(products, controls, cfg, *, rv_sys, rv_err, lsf_fwhm, detection_fap):
    rows = []
    for center_A in cfg.get("h01_placebo_centers_A", [6400.0, 6700.0]):
        for method in METHOD_ORDER:
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
                    width_factors=cfg.get("h01_template_width_factors", DEFAULT_TEMPLATE_WIDTH_FACTORS),
                    continuum_window_A=float(cfg.get("h01_continuum_window_A", 80.0)),
                )
            except ValueError:
                rows.append({"center_A": float(center_A), "method": method, "status": "skipped_outside_range"})
                continue
            rows.append(
                {
                    "center_A": float(center_A),
                    "method": method,
                    "status": "ran",
                    "global_empirical_fap": result.row["global_empirical_fap"],
                    "matched_z": result.row["matched_z"],
                }
            )
    ran = [row for row in rows if row["status"] == "ran" and row.get("global_empirical_fap") is not None]
    hits = [
        row
        for row in ran
        if float(row["global_empirical_fap"]) < float(detection_fap)
    ]
    return {"ok": None if not ran else len(hits) == 0, "rows": rows}


def compute_stage_h01_products(config, paths=None) -> StageH01Product:
    cfg = dict(config)
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_h01_paths(cfg["run_id"], root) if paths is None else paths
    qc00 = read_json(Path(cfg.get("h01_stage00q_qc_json", paths["stage00q_qc_json"])))
    rv_sys, rv_source = _rv_from_config(cfg)
    rv_err = _rv_err_from_config(cfg)
    lsf_fwhm, lsf_source = _lsf_fwhm_from_qc_or_config(qc00, cfg)
    products = load_h01_products(_product_paths_from_config(cfg, paths))
    controls = load_h01_controls(
        _control_paths_from_config(cfg, paths),
        products,
        min_controls=int(cfg.get("h01_min_controls", 3)),
        product_paths=_product_paths_from_config(cfg, paths),
    )
    method_results = {}
    rows = []
    for method in METHOD_ORDER:
        result = analyze_halpha_method(
            method,
            products[method],
            controls[method],
            rv_sys_kms=rv_sys,
            rv_sys_err_kms=rv_err,
            lsf_fwhm_A=lsf_fwhm,
            rest_A=float(cfg.get("h01_halpha_rest_A", HALPHA_REST_A)),
            search_half_width_kms=float(cfg.get("h01_search_half_width_kms", 500.0)),
            width_factors=cfg.get("h01_template_width_factors", DEFAULT_TEMPLATE_WIDTH_FACTORS),
            continuum_window_A=float(cfg.get("h01_continuum_window_A", 80.0)),
            parametric_family=str(cfg.get("h01_fap_family", "gumbel")),
            parametric_n_boot=int(cfg.get("h01_fap_n_boot", 2000)),
            parametric_seed=cfg.get("h01_fap_seed", 20260902),
        )
        method_results[method] = result
        rows.append(result.row)
    admissible_pairs = [tuple(pair) for pair in cfg.get("h01_admissible_pairs", DEFAULT_ADMISSIBLE_PAIRS)]
    fap_estimator = str(cfg.get("h01_fap_estimator", "empirical"))
    verdict = classify_h01_verdict(
        rows,
        detection_fap=float(cfg.get("h01_detection_fap", 0.01)),
        admissible_pairs=admissible_pairs,
        fap_estimator=fap_estimator,
    )
    open_issues = []
    parametrico = {r["method"]: r.pop("_parametric") for r in rows}
    detection_fap = float(cfg.get("h01_detection_fap", 0.01))
    if fap_estimator == "empirical":
        # El suelo del contador: si el criterio cae por debajo, no lo puede
        # pasar ningun dato y el veredicto no significa lo que parece.
        for r in rows:
            suelo = r.get("minimum_resolvable_fap")
            if suelo is not None and float(suelo) >= detection_fap:
                open_issues.append(
                    f"{r['method']}: el criterio FAP<{detection_fap:g} es INALCANZABLE con "
                    f"{r['n_controls']} controles (suelo {float(suelo):.4f}). El veredicto "
                    "'non_detection' puede no significar ausencia de senal.")
    else:
        # La cola solo vale si describe la nula y si no se extrapola a ciegas.
        for r in rows:
            ks = r.get("parametric_fap_ks_p")
            if ks is not None and float(ks) < PARAMETRIC_FAP_MIN_KS_P:
                open_issues.append(
                    f"{r['method']}: la cola {cfg.get('h01_fap_family', 'gumbel')} NO describe la "
                    f"nula (KS p={float(ks):.3f} < {PARAMETRIC_FAP_MIN_KS_P}); su FAP no vale.")
            ex = r.get("parametric_fap_extrapolation_sd")
            if ex is not None and float(ex) > 3.0:
                open_issues.append(
                    f"{r['method']}: la FAP parametrica extrapola {float(ex):.1f} sd por encima del "
                    "mayor control; su orden de magnitud es defendible, su cifra no.")
    expected_controls = int(cfg.get("h01_expected_controls", 31))
    for method, arr in controls.items():
        if arr.shape[0] != expected_controls:
            open_issues.append(f"{method} has {arr.shape[0]} controls; expected {expected_controls}.")
    qc = {
        "stage": "h01_halpha_detection",
        "run_id": str(cfg["run_id"]),
        "criterion": {
            "global_fap_lt": float(cfg.get("h01_detection_fap", 0.01)),
            "requires_admissible_pair": True,
            "requires_rv_within_lsf": True,
            "admissible_pairs": [[a, b] for a, b in admissible_pairs],
            "fap_estimator": fap_estimator,
            "fap_estimator_column": FAP_ESTIMATORS[fap_estimator],
            "fap_estimator_source": ("config.h01_fap_estimator" if "h01_fap_estimator" in cfg
                                     else "default (empirical)"),
        },
        "parametric_fap": {
            "family": str(cfg.get("h01_fap_family", "gumbel")),
            "n_boot": int(cfg.get("h01_fap_n_boot", 2000)),
            "seed": cfg.get("h01_fap_seed", 20260902),
            "min_ks_p": PARAMETRIC_FAP_MIN_KS_P,
            "by_method": parametrico,
        },
        "execution_order": ["controls", "object"],
        "line": {
            "rest_A": float(cfg.get("h01_halpha_rest_A", HALPHA_REST_A)),
            "rv_sys_kms": float(rv_sys),
            "rv_sys_err_kms": float(rv_err),
            "rv_source": rv_source,
            "search_half_width_kms": float(cfg.get("h01_search_half_width_kms", 500.0)),
        },
        "templates": {
            "lsf_fwhm_A": float(lsf_fwhm),
            "lsf_source": lsf_source,
            "width_factors": [float(v) for v in cfg.get("h01_template_width_factors", DEFAULT_TEMPLATE_WIDTH_FACTORS)],
        },
        "faps": {row["method"]: row["global_empirical_fap"] for row in rows},
        "verdict": verdict,
        "checks": {
            "v2_controls": {method: _control_summary(result) for method, result in method_results.items()},
            "v3_placebo": _placebo_checks(
                products,
                controls,
                cfg,
                rv_sys=rv_sys,
                rv_err=rv_err,
                lsf_fwhm=lsf_fwhm,
                detection_fap=float(cfg.get("h01_detection_fap", 0.01)),
            ),
            "v4_multimethod": {
                "global_faps": {row["method"]: row["global_empirical_fap"] for row in rows},
                "rv_consistent": {row["method"]: row["rv_consistent"] for row in rows},
            },
        },
        "open_issues": open_issues,
    }
    return StageH01Product(method_results=method_results, rows=_json_ready(rows), qc=_json_ready(qc))


def _write_stage_h01_plot(product: StageH01Product, paths):
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    n_rows = (len(METHOD_ORDER) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 3.5 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    for ax, method in zip(axes, METHOD_ORDER):
        result = product.method_results[method]
        spec = result.product
        wave = spec.wave_A
        resid = _object_residual(spec)
        row = result.row
        center = row["line_center_expected_A"]
        mask = (wave >= center - 100.0) & (wave <= center + 100.0)
        ax.plot(wave[mask], resid[mask], lw=0.8, label=method)
        ax.axvline(center, color="0.2", ls="--", lw=0.8)
        ax.axvline(row["peak_wave_A"], color="tab:red", ls=":", lw=1.0)
        ax.set_title(f"{method}: FAP={row['global_empirical_fap']:.3g}")
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylabel("Flux - continuum")
    fig.savefig(paths["summary_plot"], dpi=160)
    plt.close(fig)
    return paths["summary_plot"]


def write_stage_h01_products(product: StageH01Product, config, paths):
    paths["paths"].ensure_base_dirs()
    write_csv(paths["halpha_detection_csv"], product.rows, fieldnames=TABLE_FIELDS)
    np.savez(
        paths["null_maxima_npz"],
        **{
            f"{method}_null_maxima": result.null_maxima.astype(np.float64)
            for method, result in product.method_results.items()
        },
    )
    plot = _write_stage_h01_plot(product, paths)
    qc = dict(product.qc)
    qc["tables"] = {"by_method": str(paths["halpha_detection_csv"])}
    qc["null_maxima_npz"] = str(paths["null_maxima_npz"])
    qc["figures"] = {"summary": str(plot)}
    write_json(paths["stage_h01_qc_json"], _json_ready(qc))
    return {"table": paths["halpha_detection_csv"], "plot": plot, "qc_json": paths["stage_h01_qc_json"], "qc": qc}


def run_stage_h01(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_h01_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_h01_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_h01_products(cfg, paths)
    written = write_stage_h01_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="stage_h01_detect.py",
        description="Run Stage H01/E1 empirical Halpha detection.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--rv-sys-kms", type=float, default=None)
    parser.add_argument("--lsf-fwhm-A", type=float, default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)
    overrides = {}
    if args.rv_sys_kms is not None:
        overrides["h01_rv_sys_kms"] = args.rv_sys_kms
    if args.lsf_fwhm_A is not None:
        overrides["h01_lsf_fwhm_A"] = args.lsf_fwhm_A
    result = run_stage_h01(
        args.run_id,
        project_root=args.project_root,
        overrides=overrides,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage_h01_qc_json"])


__all__ = [
    "DEFAULT_ADMISSIBLE_PAIRS",
    "DEFAULT_TEMPLATE_WIDTH_FACTORS",
    "HALPHA_REST_A",
    "H01MethodResult",
    "StageH01Product",
    "analyze_halpha_method",
    "classify_h01_verdict",
    "compute_stage_h01_products",
    "expected_line_center_A",
    "gaussian_flux_template",
    "load_h01_controls",
    "load_h01_products",
    "matched_filter_point",
    "matched_filter_scan",
    "run_stage_h01",
    "stage_h01_config_from_run",
    "stage_h01_paths",
    "write_stage_h01_products",
]


if __name__ == "__main__":
    main()
