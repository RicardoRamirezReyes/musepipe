"""Stage X01/C2: aperture spectrum product from the Stage04b residual cube."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from ..extraction.aperture import (
    ApertureExtraction,
    default_apertures,
    extract_aperture_products,
)
from ..extraction.product import SpectrumProduct
from ..io import read_json, read_wavelength_axis, resolve_bunit, write_json
from ..paths import RunPaths
from .stage04b_local_surface import run_stage04b, stage04b_config_from_run


@dataclass(frozen=True)
class StageX01Product:
    extractions: dict[str, ApertureExtraction]
    qc: dict


def stage_x01_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    return {
        "paths": paths,
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "stage01_qc_json": paths.stage_dir / "stage01_qc.json",
        "stage00q_qc_json": paths.stage_dir / "stage00q_qc.json",
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "stage04b_cube_fits": paths.stage_dir / "stage04b_local_surface_cube_stack.fits",
        "stage04b_qc_json": paths.stage_dir / "stage04b_qc.json",
        "stage04b_good_mask_npy": paths.stage_dir / "stage04b_good_wavelength_mask.npy",
        "stage04b_bad_mask_npy": paths.stage_dir / "stage04b_bad_wavelength_mask.npy",
        "cube_residual_object": paths.stage_dir / "cube_residual_local_object.fits",
        "psf_model_json": paths.stage_dir / "psf_model.json",
        "spec_aperture_object": paths.stage_dir / "spec_aperture_object.fits",
        "spec_aperture_box3": paths.stage_dir / "spec_aperture_object_box3.fits",
        "spec_aperture_box5": paths.stage_dir / "spec_aperture_object_box5.fits",
        "spec_aperture_qc_json": paths.stage_dir / "spec_aperture_qc.json",
    }


def _bad_windows_from_config(cfg):
    windows = cfg.get("x01_bad_windows_A") or cfg.get("bad_wavelength_ranges_A")
    if windows is not None:
        return [[float(lo), float(hi)] for lo, hi in windows]
    if cfg.get("drop_wave_min_A") is not None and cfg.get("drop_wave_max_A") is not None:
        return [[float(cfg["drop_wave_min_A"]), float(cfg["drop_wave_max_A"])]]
    return []


def stage_x01_config_from_run(
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
    cfg.setdefault("x01_reuse_stage04b", True)
    cfg.setdefault("x01_source_key", "companion")
    cfg.setdefault("x01_star_key", "primary")
    cfg.setdefault("x01_apertures", default_apertures())
    cfg.setdefault("x01_error_mode", "auto")
    cfg.setdefault("x01_aperture_correction", "auto")
    cfg.setdefault("x01_control_apertures", cfg.get("stage07_control_apertures", 8))
    cfg.setdefault("x01_control_exclude_angle_deg", 25.0)
    cfg.setdefault("x01_bad_windows_A", _bad_windows_from_config(cfg))
    cfg.setdefault("x01_skyline_windows_A", cfg.get("skyline_windows_A", []))
    cfg.setdefault("x01_interpolated_windows_A", cfg.get("stage_e01_interpolated_windows_A", []))
    cfg.setdefault("x01_wframe", cfg.get("wavelength_frame", "topocentric"))
    cfg.setdefault("x01_bunit", cfg.get("cube_bunit", ""))
    return cfg


def _source_pos_yx(qc: dict, key: str) -> tuple[float, float]:
    aliases = {
        "object": "companion",
        "target": "companion",
        "science": "companion",
        "c": "companion",
        "star": "primary",
        "b": "primary",
    }
    key = aliases.get(str(key), str(key))
    item = qc.get(key)
    if isinstance(item, dict) and "pos_yx" in item:
        yx = item["pos_yx"]
        return (float(yx[0]), float(yx[1]))
    raise KeyError(f"stage01c_qc.json is missing {key!r}.pos_yx")


def _load_positions(paths, cfg):
    qc_path = Path(cfg.get("x01_positions_qc", paths["stage01c_qc_json"]))
    if not qc_path.exists():
        raise FileNotFoundError(qc_path)
    qc = read_json(qc_path)
    object_yx = _source_pos_yx(qc, cfg.get("x01_source_key", "companion"))
    star_yx = _source_pos_yx(qc, cfg.get("x01_star_key", "primary"))
    return object_yx, star_yx, qc_path, qc


def _load_masks(paths, n_wave):
    good = None
    bad = None
    if paths["stage04b_good_mask_npy"].exists():
        good = np.load(paths["stage04b_good_mask_npy"]).astype(bool)
    if paths["stage04b_bad_mask_npy"].exists():
        bad = np.load(paths["stage04b_bad_mask_npy"]).astype(bool)
    if good is not None and good.size != n_wave:
        good = None
    if bad is not None and bad.size != n_wave:
        bad = None
    return good, bad


def _load_residual_cube(paths, cfg):
    explicit = cfg.get("x01_residual_cube_fits")
    candidates = [Path(explicit)] if explicit else []
    candidates.append(paths["cube_residual_object"])
    for path in candidates:
        if path.exists():
            with fits.open(path, memmap=True) as hdul:
                if hdul[0].data is None:
                    raise RuntimeError(f"No primary cube data in {path}.")
                cube = hdul[0].data.astype(np.float64)
                wave = read_wavelength_axis(hdul, data_shape=cube.shape)
                header = hdul[0].header.copy()
            if cube.ndim != 3:
                raise RuntimeError(f"Expected residual cube shape (nz,ny,nx), got {cube.shape}.")
            good, bad = _load_masks(paths, cube.shape[0])
            bunit = resolve_bunit(cfg, stack_bunit=header.get("BUNIT"),
                                  override_key="x01_bunit")
            return cube, wave, good, bad, path, bunit

    stack = paths["stage04b_cube_fits"]
    if not stack.exists():
        raise FileNotFoundError(stack)
    with fits.open(stack, memmap=True) as hdul:
        if "RESIDUALS" not in hdul or "WAVELENGTH" not in hdul:
            raise RuntimeError(f"{stack} must contain RESIDUALS and WAVELENGTH HDUs.")
        residuals = hdul["RESIDUALS"].data.astype(np.float64)
        wave = hdul["WAVELENGTH"].data.astype(np.float64)
        good = hdul["GOOD_WAVE_MASK"].data.astype(bool) if "GOOD_WAVE_MASK" in hdul else None
        bad = hdul["BAD_WAVE_MASK"].data.astype(bool) if "BAD_WAVE_MASK" in hdul else None
    if residuals.ndim == 4:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cube = np.nanmean(residuals, axis=0)
    elif residuals.ndim == 3:
        cube = residuals
    else:
        raise RuntimeError(f"Unexpected RESIDUALS shape in {stack}: {residuals.shape}")
    return cube, wave, good, bad, stack, resolve_bunit(cfg, override_key="x01_bunit")


def _best_indices_from_stage04b(paths):
    if paths["stage04b_qc_json"].exists():
        qc = read_json(paths["stage04b_qc_json"])
        best = qc.get("best4")
        if best:
            return [int(x) for x in best]
    return None


def _load_stat_cube(paths, cfg, expected_shape):
    stat_path = Path(cfg.get("x01_stat_cube_fits") or paths["stage02_cube_fits"])
    if not stat_path.exists():
        return None, "missing"
    with fits.open(stat_path, memmap=True) as hdul:
        if "STAT" not in hdul:
            return None, "missing_stat_hdu"
        stat = hdul["STAT"].data.astype(np.float64)
    if stat.ndim == 4:
        best = _best_indices_from_stage04b(paths)
        if best is None:
            best = list(range(stat.shape[0]))
        best = [i for i in best if 0 <= i < stat.shape[0]]
        if not best:
            return None, "no_valid_best_indices"
        stat = np.nansum(stat[best], axis=0) / float(len(best) ** 2)
    if stat.shape != tuple(expected_shape):
        return None, f"shape_mismatch:{stat.shape}!={tuple(expected_shape)}"
    return stat, "loaded"


def _read_optional_json(path):
    return read_json(path) if Path(path).exists() else {}


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _stat_metadata(paths, cfg):
    qc00 = _read_optional_json(paths["stage00q_qc_json"])
    qc01 = _read_optional_json(paths["stage01_qc_json"])
    m5 = qc00.get("m5_stat", {}) if isinstance(qc00, dict) else {}
    stat_factor = cfg.get("x01_stat_factor_box3", cfg.get("stat_factor_box3", m5.get("factor_box3_median", 1.0)))
    covariance_factor = cfg.get(
        "x01_covariance_factor_box3",
        cfg.get("covariance_factor_box3", qc01.get("stat", {}).get("covariance_factor_box3", 1.0)),
    )
    stat_status = cfg.get("x01_stat_status", m5.get("status", "unknown"))
    try:
        stat_factor = float(stat_factor)
    except Exception:
        stat_factor = 1.0
    try:
        covariance_factor = float(covariance_factor)
    except Exception:
        covariance_factor = 1.0
    if not np.isfinite(stat_factor) or stat_factor <= 0:
        stat_factor = 1.0
    if not np.isfinite(covariance_factor) or covariance_factor <= 0:
        covariance_factor = 1.0
    return stat_factor, covariance_factor, str(stat_status), qc00, qc01


def _wavelength_frame(cfg, qc00, open_issues):
    frame = cfg.get("x01_wframe", "topocentric")
    if isinstance(qc00, dict):
        frame = qc00.get("cube", {}).get("wavelength_frame", frame)
    frame = str(frame or "unknown").lower()
    if frame not in {"topocentric", "barycentric"}:
        open_issues.append(f"Wavelength frame {frame!r} unavailable; using topocentric in product header.")
        frame = "topocentric"
    return frame


def _load_psf_model(paths, cfg, open_issues):
    mode = str(cfg.get("x01_aperture_correction", "auto")).lower()
    if mode in {"none", "off", "false"}:
        return None
    path = Path(cfg.get("x01_psf_model_json") or paths["psf_model_json"])
    if path.exists():
        return read_json(path)
    if mode in {"auto", "optional"}:
        open_issues.append("Aperture correction fell back to none because psf_model.json was not found.")
        return None
    raise FileNotFoundError(path)


def _median_ratio(numerator, denominator):
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    good = np.isfinite(num) & np.isfinite(den) & (den > 0)
    if not np.any(good):
        return None
    return float(np.nanmedian(num[good] / den[good]))


def _aperture_qc(extractions, cfg, paths, positions_path, cube_path, stat_state, open_issues):
    labels = list(extractions)
    primary = extractions.get("box3") or next(iter(extractions.values()))
    ratio = _median_ratio(primary.product.flux_err, primary.product.flux_err_emp)
    apcorr_median = _finite_or_none(np.nanmedian(primary.product.apcorr))
    apcorr_max = _finite_or_none(np.nanmax(primary.product.apcorr))
    apcorr_mode = primary.apcorr_mode
    v2_ok = None if ratio is None else bool(0.7 <= ratio <= 1.4)
    v4_ok = None
    if apcorr_mode == "psf_growth_curve":
        v4_ok = bool(apcorr_median is not None and apcorr_max is not None and apcorr_median >= 1.0 and apcorr_max <= 1.8)

    return {
        "stage": "x01_aperture",
        "run_id": str(cfg["run_id"]),
        "positions_from": str(positions_path),
        "input_cube": str(cube_path),
        "reuse_stage04b": bool(cfg.get("x01_reuse_stage04b", True)),
        "apertures": labels,
        "products": {},
        "errors": {
            "mode": primary.error_mode,
            "stat_factor_box3": float(primary.product.header.get("STATFAC", 1.0)),
            "covariance_factor_box3": float(primary.product.header.get("COVFAC", 1.0)),
            "stat_vs_empirical_median_ratio": ratio,
            "stat_input": stat_state,
        },
        "aperture_correction": {
            "mode": apcorr_mode,
            "median": apcorr_median,
            "max": apcorr_max,
            "norm_radius_px": _finite_or_none(primary.norm_radius_px),
        },
        "flags": {
            "bad_window_channels": int(np.count_nonzero(primary.product.flags & 1)),
            "skyline_channels": int(np.count_nonzero(primary.product.flags & 2)),
            "interpolated_channels": int(np.count_nonzero(primary.product.flags & 4)),
            "clipped_channels": int(np.count_nonzero(primary.product.flags & 8)),
        },
        "equivalence": {"baseline": "stage07 product", "verdict": "not_checked", "max_rel_diff": None},
        "checks": {
            "v2_error_ratio_ok": v2_ok,
            "v3_roundtrip_ok": True,
            "v4_apcorr_range_ok": v4_ok,
        },
        "open_issues": list(open_issues),
    }


def compute_stage_x01_products(config, paths=None):
    cfg = dict(config)
    run_id = cfg["run_id"]
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_x01_paths(run_id, root) if paths is None else paths

    reuse_stage04b = bool(cfg.get("x01_reuse_stage04b", True))
    stage04b_available = paths["cube_residual_object"].exists() or paths["stage04b_cube_fits"].exists()
    if not reuse_stage04b:
        stage04b_cfg = stage04b_config_from_run(run_id, project_root=root)
        run_stage04b(config=stage04b_cfg, project_root=root, save_plots=False)
    elif not stage04b_available:
        raise FileNotFoundError(
            f"Stage04b products are missing for run {run_id!r}. "
            "Run Stage04b first or set x01_reuse_stage04b=false."
        )

    object_yx, star_yx, positions_path, _positions_qc = _load_positions(paths, cfg)
    cube, wave, good_mask, bad_mask, cube_path, bunit = _load_residual_cube(paths, cfg)
    stat_cube, stat_state = _load_stat_cube(paths, cfg, cube.shape)
    stat_factor, covariance_factor, stat_status, qc00, _qc01 = _stat_metadata(paths, cfg)
    open_issues = []
    if stat_cube is None:
        open_issues.append(f"STAT unavailable for X01 ({stat_state}); using empirical errors.")
    if str(stat_status).lower() == "red":
        open_issues.append("A4/M5 STAT status is red; using empirical errors.")
    wframe = _wavelength_frame(cfg, qc00, open_issues)
    psf_model = _load_psf_model(paths, cfg, open_issues)

    # Wings-intact aperture correction: the PSF growth-curve apcorr assumes the
    # companion wings are present, but the stage04b local-surface residual
    # over-subtracts them (box5<box3), breaking box3/box5 consistency (V4).
    # When apcorr is active, extract from the raw (stage02) cube with a distant
    # annulus background instead of the 04b residual.
    annulus_bkg = None
    apcorr_cfg = str(cfg.get("x01_aperture_correction", "auto")).lower()
    if psf_model is not None and apcorr_cfg in ("auto", "psf_growth_curve") and bool(cfg.get("x01_wings_intact_apcorr", True)):
        raw_path = Path(cfg.get("stage_e01_input_cube_fits") or (paths["paths"].stage_dir / "stage02_xcorr_cube_stack.fits"))
        if raw_path.exists():
            with fits.open(raw_path) as h:
                rc = np.asarray(h["CUBES"].data if "CUBES" in h else h[1].data, dtype=float)
            if rc.ndim == 4:
                rc = rc[0]
            if rc.shape[0] == wave.size:
                cube, cube_path = rc, str(raw_path)
                annulus_bkg = list(cfg.get("x01_annulus_bkg_px", [8.0, 14.0, 30.0]))
                open_issues.append(
                    "Aperture correction uses a wings-intact extraction (raw cube + annulus "
                    "background) so the PSF growth curve stays self-consistent; the stage04b "
                    "residual would over-subtract the companion wings (box5<box3)."
                )

    extractions = extract_aperture_products(
        cube,
        wave,
        object_yx,
        run_id=run_id,
        input_cube_path=cube_path,
        apertures=cfg.get("x01_apertures", default_apertures()),
        star_yx=star_yx,
        stat_zyx=stat_cube,
        stat_factor_box3=stat_factor,
        covariance_factor_box3=covariance_factor,
        stat_status=stat_status,
        error_mode=cfg.get("x01_error_mode", "auto"),
        psf_model=psf_model,
        aperture_correction=cfg.get("x01_aperture_correction", "auto"),
        wframe=wframe,
        bunit=bunit,
        bad_windows_A=cfg.get("x01_bad_windows_A", []),
        skyline_windows_A=cfg.get("x01_skyline_windows_A", []),
        interpolated_windows_A=cfg.get("x01_interpolated_windows_A", []),
        good_mask=good_mask,
        bad_mask=bad_mask,
        n_controls=int(cfg.get("x01_control_apertures", 8)),
        exclude_angle_deg=float(cfg.get("x01_control_exclude_angle_deg", 25.0)),
        annulus_bkg_px=annulus_bkg,
    )
    qc = _aperture_qc(extractions, cfg, paths, positions_path, cube_path, stat_state, open_issues)
    return StageX01Product(extractions=extractions, qc=qc)


def write_stage_x01_products(product: StageX01Product, config, paths):
    paths["paths"].ensure_base_dirs()
    product_paths = {}
    for label, extraction in product.extractions.items():
        if label == "box3":
            targets = [paths["spec_aperture_object"], paths["spec_aperture_box3"]]
        elif label == "box5":
            targets = [paths["spec_aperture_box5"]]
        else:
            targets = [paths["paths"].stage_dir / f"spec_aperture_object_{label}.fits"]
        for target in targets:
            extraction.product.write(target, overwrite=True)
            SpectrumProduct.read(target)
        product_paths[label] = str(targets[0])
        if label == "box3":
            product_paths["box3_alias"] = str(paths["spec_aperture_box3"])
            product_paths["default"] = str(paths["spec_aperture_object"])
    # Persist the box3 control spectra for D1 (empirical sigma_diff source).
    # `control_spectra` (what D1 reads) is on the SAME physical scale as the
    # product flux (background-referenced + apcorr, D1 v2 §3.1); the raw sums
    # are kept alongside for diagnostics.
    if "box3" in product.extractions:
        box3 = product.extractions["box3"]
        np.savez(
            paths["paths"].stage_dir / "spec_aperture_controls.npz",
            control_spectra=np.asarray(box3.control_spectra_cal, dtype=np.float64),
            control_spectra_raw=np.asarray(box3.control_spectra, dtype=np.float64),
            bkg_mode=np.asarray(box3.bkg_mode),
            apcorr_median=np.asarray(float(np.nanmedian(box3.product.apcorr))),
        )
    qc = dict(product.qc)
    qc["products"] = product_paths
    write_json(paths["spec_aperture_qc_json"], qc)
    return {
        "products": product_paths,
        "qc_json": paths["spec_aperture_qc_json"],
        "qc": qc,
    }


def run_stage_x01(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_x01_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_x01_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_x01_products(cfg, paths)
    written = write_stage_x01_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage X01/C2 aperture extraction.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    parser.add_argument("--no-reuse-stage04b", action="store_true")
    parser.add_argument(
        "--aperture-correction",
        choices=["auto", "none", "psf_growth_curve"],
        default=None,
    )
    args = parser.parse_args(argv)
    overrides = {}
    if args.no_reuse_stage04b:
        overrides["x01_reuse_stage04b"] = False
    if args.aperture_correction is not None:
        overrides["x01_aperture_correction"] = args.aperture_correction
    result = run_stage_x01(
        args.run_id,
        project_root=args.project_root,
        overrides=overrides or None,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["spec_aperture_qc_json"])


__all__ = [
    "StageX01Product",
    "compute_stage_x01_products",
    "run_stage_x01",
    "stage_x01_config_from_run",
    "stage_x01_paths",
    "write_stage_x01_products",
]


if __name__ == "__main__":
    main()
