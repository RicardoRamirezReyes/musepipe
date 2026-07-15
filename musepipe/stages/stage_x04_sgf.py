"""Stage X04/C5: SGF stellar-halo subtraction (spec_C5_codex_sgf_subtraction).

Literature-faithful Savitzky-Golay spectral-diversity subtraction (Haffert
et al. 2019; Julo et al. 2025 App. A.3): deliberately no line masking and no
PCA, so the method's known biases stay measurable (Eq. 1 predictor in QC) and
comparable in D1 v3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..halosub import (
    DEFAULT_SGF_DEGREE,
    DEFAULT_SGF_WINDOW,
    sgf_self_subtraction_ratio,
    sgf_subtract,
)
from ..spectral import STANDARD_LINE_WINDOWS_A, standard_line_free_mask
from ..stats import robust_sigma
from .stage_x01_aperture import _load_positions
from .halosub_stage import (
    HalosubStageProduct,
    base_qc_payload,
    combine_residuals,
    extract_halosub_product,
    halosub_config_from_run,
    halosub_paths,
    load_stage02_exposures,
    lsf_fwhm_A_from_qc_or_config,
    subtract_exposures,
    write_halosub_products,
)

SPEC_VERSION = "C5_v1"
STANDARD_LINE_NAMES = ("halpha", "hbeta", "oi8446")


def stage_x04_paths(run_id, project_root=None):
    return halosub_paths(run_id, "sgf", project_root=project_root)


def stage_x04_config_from_run(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = halosub_config_from_run(
        run_id,
        method="sgf",
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    cfg.setdefault("sgf_window", DEFAULT_SGF_WINDOW)
    cfg.setdefault("sgf_degree", DEFAULT_SGF_DEGREE)
    return cfg


def self_subtraction_predictors(reference, wave, *, lsf_fwhm_A, window_channels, lines=None):
    """Eq. 1 predictor per science line, from the measured reference spectrum.

    ``C_S/L_S`` is measured on the reference (sidebands vs line window) and
    ``R`` is the expected line FWHM in channels over the filter window width
    (spec C5 §3.3). Exact in the paper's toy model (test_halosub_toy).
    """

    wave = np.asarray(wave, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    dl = float(np.nanmedian(np.diff(wave)))
    r_ratio = (float(lsf_fwhm_A) / dl) / float(window_channels)
    finite_wave = wave[np.isfinite(wave)]
    w_lo, w_hi = float(np.min(finite_wave)), float(np.max(finite_wave))
    rows = []
    line_defs = lines if lines is not None else list(zip(STANDARD_LINE_NAMES, STANDARD_LINE_WINDOWS_A))
    for name, (center, half) in line_defs:
        in_range = bool(w_lo <= float(center) <= w_hi)
        in_line = np.abs(wave - float(center)) <= float(lsf_fwhm_A)
        sideband = (np.abs(wave - float(center)) > float(half) + 8.0) & (
            np.abs(wave - float(center)) <= 40.0
        )
        l_s = float(np.nanmedian(ref[in_line])) if np.any(in_line) else np.nan
        c_s = float(np.nanmedian(ref[sideband])) if np.any(sideband) else np.nan
        predictor = None
        cs_over_ls = None
        if np.isfinite(l_s) and np.isfinite(c_s) and l_s != 0.0 and 0.0 <= r_ratio < 1.0:
            cs_over_ls = c_s / l_s
            predictor = sgf_self_subtraction_ratio(r_ratio, cs_over_ls)
        rows.append(
            {
                "line": str(name),
                "rest_A": float(center),
                "in_range": in_range,
                "cs_over_ls": None if cs_over_ls is None else float(cs_over_ls),
                "R": float(r_ratio),
                "predictor": None if predictor is None else float(predictor),
            }
        )
    return rows


def negative_continuum_fractions(product, *, lines=None):
    """Fraction of sideband channels below -2 sigma per science line (§3.3)."""

    wave = np.asarray(product.wave_A, dtype=np.float64)
    flux = np.asarray(product.flux, dtype=np.float64)
    err = np.asarray(product.flux_err, dtype=np.float64)
    rows = []
    line_defs = lines if lines is not None else list(zip(STANDARD_LINE_NAMES, STANDARD_LINE_WINDOWS_A))
    for name, (center, half) in line_defs:
        sideband = (
            (np.abs(wave - float(center)) > float(half))
            & (np.abs(wave - float(center)) <= 40.0)
            & np.isfinite(flux)
            & np.isfinite(err)
            & (err > 0)
        )
        frac = None
        if np.any(sideband):
            frac = float(np.mean(flux[sideband] < -2.0 * err[sideband]))
        rows.append({"line": str(name), "frac_below_minus2sigma": frac})
    return rows


def _far_continuum_check(extraction):
    """v2: control residuals compatible with 0 in line-free continuum."""

    controls = np.asarray(extraction.control_spectra_cal, dtype=np.float64)
    if controls.size == 0:
        return None, None
    wave = np.asarray(extraction.product.wave_A, dtype=np.float64)
    mask = standard_line_free_mask(wave) & np.all(np.isfinite(controls), axis=0)
    if not np.any(mask):
        return None, None
    values = controls[:, mask].ravel()
    median = float(np.nanmedian(values))
    sigma = float(robust_sigma(values))
    n = int(values.size)
    ok = bool(abs(median) < 2.0 * sigma / max(np.sqrt(n), 1.0)) if sigma > 0 else None
    return median, ok


def compute_stage_x04_products(config, paths=None):
    cfg = dict(config)
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_x04_paths(cfg["run_id"], root) if paths is None else paths
    open_issues = []

    exposures_raw, wave, _stage02_path, _bunit, _best = load_stage02_exposures(paths, cfg)
    object_yx, _star_yx, _p, _q = _load_positions(paths, cfg)
    window = int(cfg.get("sgf_window", DEFAULT_SGF_WINDOW))
    degree = int(cfg.get("sgf_degree", DEFAULT_SGF_DEGREE))

    def _subtract(cube, wave_A, s_hat):  # noqa: ARG001
        res = sgf_subtract(cube, s_hat, window=window, degree=degree)
        return res.residual_cube, {"window": window, "degree": degree}

    exposures = subtract_exposures(exposures_raw, wave, cfg, object_yx, _subtract)
    residual = combine_residuals(exposures, combine=cfg.get("halosub_combine", "mean"))
    extraction, _obj_yx, _sta_yx = extract_halosub_product(
        residual, wave, cfg, paths, method="sgf", open_issues=open_issues
    )

    references = np.stack([e.reference for e in exposures], axis=0)
    mean_reference = np.nanmean(references, axis=0)
    lsf_fwhm_A = lsf_fwhm_A_from_qc_or_config(paths, cfg)
    predictors = self_subtraction_predictors(
        mean_reference, wave, lsf_fwhm_A=lsf_fwhm_A, window_channels=window
    )
    negatives = negative_continuum_fractions(extraction.product)
    far_median, far_ok = _far_continuum_check(extraction)

    qc = base_qc_payload(cfg, "sgf", SPEC_VERSION, exposures, extraction, open_issues)
    qc["sgf"] = {
        "window": window,
        "degree": degree,
        "lsf_fwhm_A": float(lsf_fwhm_A),
        "ref_exclude_radius_px": float(cfg.get("halosub_exclude_radius_px", 3.0)),
    }
    qc["self_subtraction_predictor"] = predictors
    qc["negative_continuum"] = negatives
    qc["far_continuum_median"] = far_median
    qc["checks"] = {
        "v1_reference_ok": bool(all(e.n_spaxels_kept >= 50 for e in exposures)),
        "v2_far_continuum_ok": far_ok,
        "v3_predictor_written": bool(
            any(row["in_range"] for row in predictors)
            and all(row["predictor"] is not None for row in predictors if row["in_range"])
        ),
        "v4_scale_convention_ok": bool(
            extraction.product.header.get("SCALEREF") == "normrad_total_flux"
            and str(extraction.product.header.get("BKGMODE", "")).startswith("sgf_residual")
            and np.asarray(extraction.control_spectra_cal).size > 0
        ),
        "v5_no_pca": True,
    }
    return HalosubStageProduct(
        extraction=extraction,
        residual_cube=residual,
        references=references,
        exposures=exposures,
        qc=qc,
    )


def write_stage_x04_products(product, config, paths):
    return write_halosub_products(product, config, paths, method="sgf")


def run_stage_x04(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_x04_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_x04_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_x04_products(cfg, paths)
    written = write_stage_x04_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage X04/C5 SGF halo subtraction.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage_x04(
        args.run_id,
        project_root=args.project_root,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["qc_json"])


__all__ = [
    "SPEC_VERSION",
    "compute_stage_x04_products",
    "negative_continuum_fractions",
    "run_stage_x04",
    "self_subtraction_predictors",
    "stage_x04_config_from_run",
    "stage_x04_paths",
    "write_stage_x04_products",
]


if __name__ == "__main__":
    main()
