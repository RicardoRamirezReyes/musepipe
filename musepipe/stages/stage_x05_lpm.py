"""Stage X05/C6: LPM stellar-halo subtraction (spec_C6_codex_lpm_subtraction).

Legendre polynomial modulation (Julo et al. 2025 App. A.4): each spaxel is a
degree-4 polynomial modulation of the stellar reference, fitted by orthogonal
projection with the science lines masked, so line fluxes and profiles survive
the subtraction. Degree diagnostics (energy share, analytic MSE curve,
coefficient maps) are QC only — the degree never changes within a run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..constants import fwhm_to_sigma
from ..halosub import (
    DEFAULT_LPM_DEGREE,
    lpm_coefficient_energy_share,
    lpm_design_matrix,
    lpm_mse_terms,
    lpm_subtract,
)
from ..spectral import STANDARD_LINE_WINDOWS_A
from ..stats import robust_sigma
from .stage_x01_aperture import _load_positions
from .halosub_stage import (
    HalosubStageProduct,
    base_qc_payload,
    combine_residuals,
    expected_scaleref,
    extract_halosub_product,
    halosub_config_from_run,
    halosub_paths,
    load_stage02_exposures,
    lsf_fwhm_A_from_qc_or_config,
    subtract_exposures,
    write_halosub_products,
)

SPEC_VERSION = "C6_v1.1"
MAX_CONDITION_NUMBER = 1e8
MAX_SLOW_FRACTION = 0.20
DIAG_DEGREE = 9


def stage_x05_paths(run_id, project_root=None):
    return halosub_paths(run_id, "lpm", project_root=project_root)


def stage_x05_config_from_run(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = halosub_config_from_run(
        run_id,
        method="lpm",
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    cfg.setdefault("lpm_degree", DEFAULT_LPM_DEGREE)
    cfg.setdefault("lpm_masked_lines_A", [[c, h] for c, h in STANDARD_LINE_WINDOWS_A])
    cfg.setdefault("lpm_energy_share_warn_factor", 2.0)
    return cfg


def masked_line_windows(cfg):
    windows = cfg.get("lpm_masked_lines_A") or []
    return tuple((float(c), float(h)) for c, h in windows)


def _median_flux_spaxel(cube, wave, cfg, companion_yx=None):
    """Deterministic REAL stellar spaxel for the MSE oracle (spec C6 v1.1 §0b.2):
    the reference-selection spaxel whose integrated flux is the median."""

    from ..halosub import select_reference_spaxels

    keep, _ = select_reference_spaxels(
        cube,
        flux_lo_frac=float(cfg.get("halosub_flux_mask_lo", 0.01)),
        flux_hi_frac=float(cfg.get("halosub_flux_mask_hi", 0.1)),
        wave_mask=np.isfinite(np.asarray(wave, dtype=np.float64)),
        exclude_yx=[companion_yx] if companion_yx is not None else None,
        exclude_radius_px=float(cfg.get("halosub_exclude_radius_px", 3.0)),
    )
    with np.errstate(all="ignore"):
        totals = np.nansum(np.where(np.isfinite(cube), cube, 0.0), axis=0)
    values = totals[keep]
    target = float(np.median(values))
    ys, xs = np.where(keep)
    idx = int(np.argmin(np.abs(values - target)))
    return cube[:, ys[idx], xs[idx]]


def degree_diagnostics(first_cube, wave, reference, residual_cube, cfg, *, lsf_fwhm_A, companion_yx=None):
    """Spec C6 §3.3 (v1.1): energy share (Fig. 8), analytic MSE curve (Fig. 6/B.5).

    The MSE oracle star is a REAL spaxel (median-flux member of the reference
    selection), NOT the reference itself: the reference lies in the model's
    column space by construction (degree-0 column), which would make the
    underfit term identically zero and the argmin degenerate (v1 erratum).
    """

    line_windows = masked_line_windows(cfg)
    warn_factor = float(cfg.get("lpm_energy_share_warn_factor", 2.0))

    # Energy share from a degree-9 diagnostic fit on the first exposure.
    diag = lpm_subtract(first_cube, wave, reference, degree=DIAG_DEGREE, line_windows_A=line_windows)
    share = lpm_coefficient_energy_share(diag.coeffs)  # degrees 1..9
    tail_median = float(np.nanmedian(share[5:]))  # degrees 6..9
    degree_warn = bool(share[4] > warn_factor * tail_median) if np.isfinite(tail_median) else None

    # Analytic MSE curve: real spaxel as star, robust residual noise as sigma,
    # 5-sigma Gaussian line at Halpha (FWHM = LSF) as the planet spaxel.
    finite_res = residual_cube[np.isfinite(residual_cube)]
    sigma = float(robust_sigma(finite_res)) if finite_res.size else float("nan")
    center_A = float(line_windows[0][0]) if line_windows else 6562.8
    lsf_sigma_A = fwhm_to_sigma(lsf_fwhm_A)
    planet = 5.0 * sigma * np.exp(-0.5 * ((wave - center_A) / lsf_sigma_A) ** 2)
    star_spaxel = _median_flux_spaxel(first_cube, wave, cfg, companion_yx=companion_yx)
    # Smooth the real spaxel (spec v1.1): the chromatic deformation is smooth
    # by hypothesis; unsmoothed, the spaxel's own noise inflates the underfit
    # term with a trivial -sigma^2-per-degree slope.
    from ..spectral import median_filter_1d

    star_smooth = median_filter_1d(star_spaxel, width=301)
    good = np.isfinite(star_smooth) & np.isfinite(np.asarray(reference, dtype=np.float64)) & np.isfinite(planet)
    curve = []
    for degree in range(1, DIAG_DEGREE + 1):
        design = lpm_design_matrix(wave, reference, degree=degree)[good]
        terms = lpm_mse_terms(design, star_smooth[good], planet[good], sigma)
        curve.append({"degree": degree, **{k: float(v) for k, v in terms.items()}})
    totals = [row["total"] for row in curve]
    argmin = int(curve[int(np.argmin(totals))]["degree"]) if curve else None
    # Only warn when the curve expresses a real preference (spec v1.1): a
    # flat curve's argmin is noise, not evidence about the degree.
    flat = None
    mse_warn = None
    if argmin is not None:
        t_min, t_max = float(np.min(totals)), float(np.max(totals))
        flat = bool(t_min > 0 and (t_max - t_min) / t_min < 0.01)
        mse_warn = None if flat else bool(not 3 <= argmin <= 7)
    return {
        "energy_share": [float(v) for v in share],
        "degree_check_warn": degree_warn,
        "mse_curve": curve,
        "mse_argmin": argmin,
        "mse_curve_flat": flat,
        "mse_check_warn": mse_warn,
        "noise_sigma": sigma,
    }, diag.coeffs


def line_preservation_smoke(first_cube, wave, reference, cfg, control_yx, *, lsf_fwhm_A):
    """Spec C6 §6 v2: inject a masked-line Gaussian at a control spaxel and
    check the subtraction returns >= 90% of its flux."""

    line_windows = masked_line_windows(cfg)
    if not line_windows:
        return None
    center_A = float(line_windows[0][0])
    lsf_sigma_A = fwhm_to_sigma(lsf_fwhm_A)
    y, x = int(round(control_yx[0])), int(round(control_yx[1]))
    base_spax = first_cube[:, y, x]
    scale = np.nanmedian(np.abs(base_spax))
    if not np.isfinite(scale) or scale <= 0:
        return None
    line = 0.5 * scale * np.exp(-0.5 * ((wave - center_A) / lsf_sigma_A) ** 2)

    injected = first_cube.copy()
    injected[:, y, x] = injected[:, y, x] + line
    degree = int(cfg.get("lpm_degree", DEFAULT_LPM_DEGREE))
    base = lpm_subtract(first_cube, wave, reference, degree=degree, line_windows_A=line_windows)
    with_line = lpm_subtract(injected, wave, reference, degree=degree, line_windows_A=line_windows)
    delta = with_line.residual_cube[:, y, x] - base.residual_cube[:, y, x]
    in_line = np.abs(wave - center_A) <= 3.0 * lsf_sigma_A
    good = in_line & np.isfinite(delta) & np.isfinite(line)
    injected_flux = float(np.sum(line[good]))
    if injected_flux <= 0:
        return None
    return float(np.sum(delta[good]) / injected_flux)


def compute_stage_x05_products(config, paths=None):
    cfg = dict(config)
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_x05_paths(cfg["run_id"], root) if paths is None else paths
    open_issues = []

    exposures_raw, wave, _stage02_path, _bunit, _best = load_stage02_exposures(paths, cfg)
    object_yx, _star_yx, _p, _q = _load_positions(paths, cfg)
    degree = int(cfg.get("lpm_degree", DEFAULT_LPM_DEGREE))
    line_windows = masked_line_windows(cfg)

    def _subtract(cube, wave_A, s_hat):
        res = lpm_subtract(cube, wave_A, s_hat, degree=degree, line_windows_A=line_windows)
        info = {
            "condition_number": res.condition_number,
            "n_slow_spaxels": res.n_slow_spaxels,
            "coeffs": res.coeffs,
        }
        return res.residual_cube, info

    exposures = subtract_exposures(exposures_raw, wave, cfg, object_yx, _subtract)
    for e in exposures:
        if e.method_info["condition_number"] > MAX_CONDITION_NUMBER:
            raise RuntimeError(
                f"LPM design matrix condition number {e.method_info['condition_number']:.3e} "
                f"> {MAX_CONDITION_NUMBER:.0e}; degenerate basis (spec C6 §4)."
            )
    residual = combine_residuals(exposures, combine=cfg.get("halosub_combine", "mean"))
    extraction, _obj_yx, _sta_yx = extract_halosub_product(
        residual, wave, cfg, paths, method="lpm", open_issues=open_issues
    )

    references = np.stack([e.reference for e in exposures], axis=0)
    lsf_fwhm_A = lsf_fwhm_A_from_qc_or_config(paths, cfg)
    diagnostics, diag_coeffs = degree_diagnostics(
        exposures_raw[0], wave, exposures[0].reference, residual, cfg,
        lsf_fwhm_A=lsf_fwhm_A, companion_yx=object_yx,
    )
    recovery = None
    if extraction.controls_yx:
        recovery = line_preservation_smoke(
            exposures_raw[0], wave, exposures[0].reference, cfg, extraction.controls_yx[0],
            lsf_fwhm_A=lsf_fwhm_A,
        )
    if recovery is None:
        open_issues.append("Line-preservation smoke test unavailable (no usable control spaxel).")

    n_spax = exposures_raw[0].shape[1] * exposures_raw[0].shape[2]
    slow_fractions = [e.method_info["n_slow_spaxels"] / max(n_spax, 1) for e in exposures]

    qc = base_qc_payload(cfg, "lpm", SPEC_VERSION, exposures, extraction, open_issues)
    qc["lpm"] = {
        "degree": degree,
        "masked_lines_A": [[c, h] for c, h in line_windows],
        "lsf_fwhm_A": float(lsf_fwhm_A),
        "condition_number": [float(e.method_info["condition_number"]) for e in exposures],
        "n_slow_spaxels": [int(e.method_info["n_slow_spaxels"]) for e in exposures],
        "slow_fraction_max": float(max(slow_fractions)) if slow_fractions else None,
        "line_preservation_recovery": recovery,
    }
    qc["degree_diagnostics"] = diagnostics
    qc["checks"] = {
        "v1_reference_ok": bool(all(e.n_spaxels_kept >= 50 for e in exposures)),
        "v2_line_preservation_ok": None if recovery is None else bool(recovery >= 0.9),
        "v3_condition_ok": True,  # enforced above (hard abort otherwise)
        "v4_slow_path_ok": bool(max(slow_fractions) <= MAX_SLOW_FRACTION) if slow_fractions else None,
        "v5_scale_convention_ok": bool(
            extraction.product.header.get("SCALEREF") == expected_scaleref(cfg, knob="x05_flux_convention")
            and str(extraction.product.header.get("BKGMODE", "")).startswith("lpm_residual")
            and np.asarray(extraction.control_spectra_cal).size > 0
        ),
        "v6_degree_diagnostics_written": True,
    }
    product = HalosubStageProduct(
        extraction=extraction,
        residual_cube=residual,
        references=references,
        exposures=exposures,
        qc=qc,
    )
    # Coefficient maps of the production fit (first exposure) plus the
    # degree-9 diagnostic fit (paper Fig. 7), persisted by the writer.
    product.qc["_coeff_maps"] = {
        "production": exposures[0].method_info["coeffs"],
        "diagnostic_deg9": diag_coeffs,
    }
    return product


def write_stage_x05_products(product, config, paths):
    maps = product.qc.pop("_coeff_maps", None)
    extra_hdus = None
    if maps is not None:
        extra_hdus = [
            fits.ImageHDU(np.asarray(maps["production"], dtype=np.float32), name="COEFFS"),
            fits.ImageHDU(np.asarray(maps["diagnostic_deg9"], dtype=np.float32), name="COEFFS_DEG9"),
        ]
    return write_halosub_products(product, config, paths, method="lpm", extra_hdus=extra_hdus)


def run_stage_x05(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_x05_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_x05_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_x05_products(cfg, paths)
    written = write_stage_x05_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage X05/C6 LPM halo subtraction.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage_x05(
        args.run_id,
        project_root=args.project_root,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["qc_json"])


__all__ = [
    "SPEC_VERSION",
    "compute_stage_x05_products",
    "degree_diagnostics",
    "line_preservation_smoke",
    "run_stage_x05",
    "stage_x05_config_from_run",
    "stage_x05_paths",
    "write_stage_x05_products",
]


if __name__ == "__main__":
    main()
