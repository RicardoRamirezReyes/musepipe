"""Shared core for stages X04/C5 (SGF) and X05/C6 (LPM).

Both stages share everything except the per-exposure subtraction kernel and
their method-specific QC: reference-spectrum estimation, the per-exposure
loop, residual combination, box3 product extraction (D1 v2 §3.1 scale
convention) and product/QC writing. Specs:
``docs/spec_C5_codex_sgf_subtraction.md`` and
``docs/spec_C6_codex_lpm_subtraction.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from ..extraction.aperture import ApertureExtraction, make_aperture_product
from ..halosub import (
    DEFAULT_FLUX_MASK_HI,
    DEFAULT_FLUX_MASK_LO,
    reference_spectrum,
    select_reference_spaxels,
)
from ..io import read_json, write_json
from ..paths import RunPaths
from .stage_x01_aperture import (
    _best_indices_from_stage04b,
    _load_positions,
    _load_stat_cube,
    _read_optional_json,
    _wavelength_frame,
)
from .stage_x02_optimal import _bad_windows_from_config, _stat_metadata

MIN_REFERENCE_SPAXELS = 50
BOX3_APERTURE = {"kind": "box", "size": 3, "name": "box3"}


@dataclass(frozen=True)
class HalosubExposure:
    reference: np.ndarray
    residual: np.ndarray
    n_spaxels_kept: int
    method_info: dict


@dataclass(frozen=True)
class HalosubStageProduct:
    extraction: ApertureExtraction
    residual_cube: np.ndarray
    references: np.ndarray  # (n_exposures, nz)
    exposures: list[HalosubExposure]
    qc: dict


def halosub_paths(run_id, method, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    stage = {"sgf": "stage_x04_sgf", "lpm": "stage_x05_lpm"}[method]
    return {
        "paths": paths,
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "stage01_qc_json": paths.stage_dir / "stage01_qc.json",
        "stage00q_qc_json": paths.stage_dir / "stage00q_qc.json",
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "stage04b_qc_json": paths.stage_dir / "stage04b_qc.json",
        "psf_model_json": paths.stage_dir / "psf_model.json",
        "product_fits": paths.stage_dir / f"spec_{method}_object.fits",
        "controls_npz": paths.stage_dir / f"spec_{method}_controls.npz",
        "residual_cube_fits": paths.stage_dir / f"{stage}_residual_cube.fits",
        "reference_npy": paths.stage_dir / f"{stage}_reference_spectra.npy",
        "coeff_maps_fits": paths.stage_dir / f"{stage}_coeff_maps.fits",
        "qc_json": paths.stage_dir / f"spec_{method}_qc.json",
    }


def halosub_config_from_run(
    run_id=None,
    *,
    method,
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
    cfg.setdefault("halosub_flux_mask_lo", DEFAULT_FLUX_MASK_LO)
    cfg.setdefault("halosub_flux_mask_hi", DEFAULT_FLUX_MASK_HI)
    cfg.setdefault("halosub_exclude_radius_px", 3.0)
    cfg.setdefault("halosub_wave_range_A", None)
    cfg.setdefault("halosub_combine", "mean")  # frozen default (spec C5 §1.4)
    prefix = {"sgf": "x04", "lpm": "x05"}[method]
    cfg.setdefault(f"{prefix}_source_key", "companion")
    cfg.setdefault(f"{prefix}_star_key", "primary")
    cfg.setdefault(f"{prefix}_error_mode", "auto")
    cfg.setdefault(f"{prefix}_aperture_correction", "auto")
    # Same control convention as C2/C3/C4 (D1 pairs demand identical control
    # positions across methods): inherit the run's x01 count, fallback 8.
    cfg.setdefault(f"{prefix}_control_apertures", cfg.get("x01_control_apertures", cfg.get("stage07_control_apertures", 8)))
    cfg.setdefault(f"{prefix}_control_exclude_angle_deg", cfg.get("x01_control_exclude_angle_deg", 25.0))
    cfg.setdefault(f"{prefix}_bad_windows_A", _bad_windows_from_config(cfg))
    cfg.setdefault(f"{prefix}_skyline_windows_A", cfg.get("skyline_windows_A", []))
    cfg.setdefault(f"{prefix}_interpolated_windows_A", cfg.get("stage_e01_interpolated_windows_A", []))
    cfg.setdefault(f"{prefix}_annulus_bkg_px", cfg.get("x01_annulus_bkg_px", [8.0, 14.0, 30.0]))
    # x01 keys reused by the shared position/STAT loaders.
    cfg.setdefault("x01_source_key", cfg[f"{prefix}_source_key"])
    cfg.setdefault("x01_star_key", cfg[f"{prefix}_star_key"])
    return cfg


def load_stage02_exposures(paths, cfg):
    """Stage02 stack as a per-exposure list (spec C5 §3.1: never pre-averaged).

    Returns ``(exposure_cubes, wave, path, bunit, best_indices)``; 3D stacks
    yield a single exposure.
    """

    path = Path(cfg.get("halosub_stage02_cube_fits") or paths["stage02_cube_fits"])
    if not path.exists():
        raise FileNotFoundError(path)
    with fits.open(path, memmap=True) as hdul:
        if "CUBES" not in hdul or "WAVELENGTH" not in hdul:
            raise RuntimeError(f"{path} must contain CUBES and WAVELENGTH HDUs.")
        cubes = hdul["CUBES"].data.astype(np.float64)
        wave = hdul["WAVELENGTH"].data.astype(np.float64)
        bunit = str(hdul["CUBES"].header.get("BUNIT", ""))
    if cubes.ndim == 3:
        return [cubes], wave, path, bunit, [0]
    if cubes.ndim != 4:
        raise RuntimeError(f"Unexpected CUBES shape in {path}: {cubes.shape}")
    best = _best_indices_from_stage04b(paths)
    if best is None:
        best = list(range(cubes.shape[0]))
    best = [i for i in best if 0 <= i < cubes.shape[0]]
    if not best:
        raise RuntimeError("No valid Stage04b best indices for the halosub exposure loop.")
    return [cubes[i] for i in best], wave, path, bunit, best


def wave_range_mask(wave, wave_range_A):
    """Working spectral window (``halosub_wave_range_A``); None = full band."""

    mask = np.isfinite(np.asarray(wave, dtype=np.float64))
    if wave_range_A is not None:
        lo, hi = float(wave_range_A[0]), float(wave_range_A[1])
        mask &= (wave >= lo) & (wave <= hi)
    return mask


def subtract_exposures(exposures, wave, cfg, companion_yx, subtract_fn):
    """Per-exposure reference estimation + subtraction (specs C5/C6 §3.1).

    ``subtract_fn(cube, wave, s_hat)`` returns ``(residual, method_info)``.
    """

    wmask = wave_range_mask(wave, cfg.get("halosub_wave_range_A"))
    results = []
    for cube in exposures:
        keep, keep_qc = select_reference_spaxels(
            cube,
            flux_lo_frac=float(cfg.get("halosub_flux_mask_lo", DEFAULT_FLUX_MASK_LO)),
            flux_hi_frac=float(cfg.get("halosub_flux_mask_hi", DEFAULT_FLUX_MASK_HI)),
            wave_mask=wmask,
            exclude_yx=[companion_yx],
            exclude_radius_px=float(cfg.get("halosub_exclude_radius_px", 3.0)),
        )
        if keep_qc["n_spaxels_kept"] < MIN_REFERENCE_SPAXELS:
            raise RuntimeError(
                f"Reference selection kept {keep_qc['n_spaxels_kept']} spaxels "
                f"(< {MIN_REFERENCE_SPAXELS}); field too small for spectral diversity."
            )
        s_hat = reference_spectrum(cube, keep)
        residual, info = subtract_fn(cube, wave, s_hat)
        # Outside the working window the model is undefined: keep NaN so the
        # extraction flags those channels instead of reading zeros as flux.
        residual = residual.copy()
        residual[~wmask] = np.nan
        results.append(
            HalosubExposure(
                reference=s_hat,
                residual=residual,
                n_spaxels_kept=int(keep_qc["n_spaxels_kept"]),
                method_info=dict(info),
            )
        )
    return results


def combine_residuals(exposures, combine="mean"):
    stack = np.stack([e.residual for e in exposures], axis=0)
    with np.errstate(all="ignore"):
        if str(combine) == "median":
            return np.nanmedian(stack, axis=0)
        if str(combine) == "mean":
            return np.nanmean(stack, axis=0)
    raise ValueError(f"Unknown halosub_combine={combine!r}; expected 'mean' or 'median'.")


def extract_halosub_product(residual_cube, wave, cfg, paths, *, method, open_issues):
    """box3 aperture product on the combined residual cube (specs §3.2)."""

    prefix = {"sgf": "x04", "lpm": "x05"}[method]
    object_yx, star_yx, _positions_path, _positions_qc = _load_positions(paths, cfg)
    psf_model = read_json(paths["psf_model_json"]) if paths["psf_model_json"].exists() else None
    if psf_model is None:
        open_issues.append("psf_model.json missing; product written without aperture correction.")
    stat_cube, stat_state = _load_stat_cube(paths, cfg, residual_cube.shape)
    stat_factor, covariance_factor, stat_status, qc00, _qc01 = _stat_metadata(paths, cfg)
    if stat_cube is None:
        open_issues.append(f"STAT unavailable for {prefix} ({stat_state}); using empirical errors.")
    if str(stat_status).lower() == "red":
        open_issues.append("A4/M5 STAT status is red; products use empirical flux_err.")
    wframe = _wavelength_frame(cfg, qc00, open_issues)

    extraction = make_aperture_product(
        residual_cube,
        wave,
        object_yx,
        dict(BOX3_APERTURE),
        run_id=cfg["run_id"],
        input_cube_path=paths["stage02_cube_fits"],
        star_yx=star_yx,
        stat_zyx=stat_cube,
        stat_factor=stat_factor,
        covariance_factor=covariance_factor,
        stat_status=stat_status,
        error_mode=cfg.get(f"{prefix}_error_mode", "auto"),
        psf_model=psf_model,
        aperture_correction=cfg.get(f"{prefix}_aperture_correction", "auto"),
        wframe=wframe,
        bunit=cfg.get("cube_bunit", ""),
        bad_windows_A=cfg.get(f"{prefix}_bad_windows_A", []),
        skyline_windows_A=cfg.get(f"{prefix}_skyline_windows_A", []),
        interpolated_windows_A=cfg.get(f"{prefix}_interpolated_windows_A", []),
        n_controls=int(cfg.get(f"{prefix}_control_apertures", 8)),
        exclude_angle_deg=float(cfg.get(f"{prefix}_control_exclude_angle_deg", 25.0)),
        annulus_bkg_px=cfg.get(f"{prefix}_annulus_bkg_px"),
    )
    # Method identity on top of the aperture conventions (specs §1.5).
    extraction.product.header["METHOD"] = method
    extraction.product.header["BKGMODE"] = f"{method}_residual+{extraction.bkg_mode}"
    return extraction, object_yx, star_yx


def base_qc_payload(cfg, method, spec_version, exposures, extraction, open_issues):
    ratio = None
    err = np.asarray(extraction.product.flux_err, dtype=np.float64)
    emp = np.asarray(extraction.product.flux_err_emp, dtype=np.float64)
    good = np.isfinite(err) & np.isfinite(emp) & (emp > 0)
    if np.any(good):
        ratio = float(np.nanmedian(err[good] / emp[good]))
    return {
        "stage": {"sgf": "x04_sgf", "lpm": "x05_lpm"}[method],
        "spec_version": spec_version,
        "run_id": str(cfg["run_id"]),
        "method": method,
        "halosub": {
            "flux_mask_lo": float(cfg.get("halosub_flux_mask_lo", DEFAULT_FLUX_MASK_LO)),
            "flux_mask_hi": float(cfg.get("halosub_flux_mask_hi", DEFAULT_FLUX_MASK_HI)),
            "exclude_radius_px": float(cfg.get("halosub_exclude_radius_px", 3.0)),
            "wave_range_A": cfg.get("halosub_wave_range_A"),
            "combine": str(cfg.get("halosub_combine", "mean")),
            "n_exposures": len(exposures),
            "n_spaxels_kept": [e.n_spaxels_kept for e in exposures],
        },
        "pca_applied": False,
        "errors": {
            "mode": extraction.error_mode,
            "stat_vs_empirical_median_ratio": ratio,
        },
        "open_issues": list(open_issues),
    }


def write_halosub_products(product: HalosubStageProduct, cfg, paths, *, method, extra_hdus=None):
    paths["paths"].ensure_base_dirs()
    product.extraction.product.write(paths["product_fits"], overwrite=True)
    np.savez(
        paths["controls_npz"],
        control_spectra=np.asarray(product.extraction.control_spectra_cal, dtype=np.float64),
        control_spectra_raw=np.asarray(product.extraction.control_spectra, dtype=np.float64),
        bkg_mode=np.asarray(product.extraction.bkg_mode),
        apcorr_median=np.asarray(float(np.nanmedian(product.extraction.product.apcorr))),
    )
    wave = np.asarray(product.extraction.product.wave_A, dtype=np.float64)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(product.residual_cube.astype(np.float32), name="RESIDUAL"),
            fits.ImageHDU(wave, name="WAVELENGTH"),
        ]
    ).writeto(paths["residual_cube_fits"], overwrite=True)
    np.save(paths["reference_npy"], product.references.astype(np.float64))
    if extra_hdus:
        fits.HDUList([fits.PrimaryHDU()] + list(extra_hdus)).writeto(
            paths["coeff_maps_fits"], overwrite=True
        )
    qc = dict(product.qc)
    qc["products"] = {
        "object": str(paths["product_fits"]),
        "controls_npz": str(paths["controls_npz"]),
        "residual_cube": str(paths["residual_cube_fits"]),
        "reference_spectra_npy": str(paths["reference_npy"]),
    }
    if extra_hdus:
        qc["products"]["coeff_maps"] = str(paths["coeff_maps_fits"])
    write_json(paths["qc_json"], qc)
    return {"products": qc["products"], "qc_json": paths["qc_json"], "qc": qc}


def lsf_fwhm_A_from_qc_or_config(paths, cfg, default=2.6):
    qc00 = _read_optional_json(paths["stage00q_qc_json"])
    value = None
    if isinstance(qc00, dict):
        value = qc00.get("lsf", {}).get("fwhm_A") or qc00.get("cube", {}).get("lsf_fwhm_A")
    if value is None:
        value = cfg.get("lsf_fwhm_A", default)
    try:
        value = float(value)
    except Exception:
        value = float(default)
    if not np.isfinite(value) or value <= 0:
        value = float(default)
    return value


__all__ = [
    "BOX3_APERTURE",
    "HalosubExposure",
    "HalosubStageProduct",
    "MIN_REFERENCE_SPAXELS",
    "base_qc_payload",
    "combine_residuals",
    "extract_halosub_product",
    "halosub_config_from_run",
    "halosub_paths",
    "load_stage02_exposures",
    "lsf_fwhm_A_from_qc_or_config",
    "subtract_exposures",
    "wave_range_mask",
    "write_halosub_products",
]
