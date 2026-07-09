"""Stage X03/C4: simultaneous linear star+companion PSF fitting."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..apertures import aperture_weights
from ..config import load_run_config
from ..extraction.product import SpectrumProduct
from ..extraction.psffit import PsfFitProducts, crosstalk_metric, make_psffit_products
from ..io import read_json, write_json
from ..paths import RunPaths
from .stage_x01_aperture import (
    _load_masks,
    _load_positions,
    _load_stat_cube,
    _read_optional_json,
    _wavelength_frame,
)
from .stage_x02_optimal import _load_stage02_cube, _median_ratio, _stat_metadata


@dataclass(frozen=True)
class StageX03Product:
    products: PsfFitProducts
    qc: dict


def stage_x03_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    return {
        "paths": paths,
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "stage01_qc_json": paths.stage_dir / "stage01_qc.json",
        "stage00q_qc_json": paths.stage_dir / "stage00q_qc.json",
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "stage04b_qc_json": paths.stage_dir / "stage04b_qc.json",
        "stage04b_good_mask_npy": paths.stage_dir / "stage04b_good_wavelength_mask.npy",
        "stage04b_bad_mask_npy": paths.stage_dir / "stage04b_bad_wavelength_mask.npy",
        "psf_model_json": paths.stage_dir / "psf_model.json",
        "spec_aperture_object": paths.stage_dir / "spec_aperture_object.fits",
        "spec_optimal_object": paths.stage_dir / "spec_optimal_object.fits",
        "spec_psffit_object": paths.stage_dir / "spec_psffit_object.fits",
        "spec_psffit_star": paths.stage_dir / "spec_psffit_star.fits",
        "cube_psffit_residual": paths.stage_dir / "cube_psffit_residual.fits",
        "spec_psffit_qc_json": paths.stage_dir / "spec_psffit_qc.json",
    }


def _bad_windows_from_config(cfg):
    windows = cfg.get("x03_bad_windows_A") or cfg.get("bad_wavelength_ranges_A")
    if windows is not None:
        return [[float(lo), float(hi)] for lo, hi in windows]
    if cfg.get("drop_wave_min_A") is not None and cfg.get("drop_wave_max_A") is not None:
        return [[float(cfg["drop_wave_min_A"]), float(cfg["drop_wave_max_A"])]]
    return []


def stage_x03_config_from_run(
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
    cfg.setdefault("x03_source_key", "companion")
    cfg.setdefault("x03_star_key", "primary")
    cfg.setdefault("x03_star_radius_px", 20.0)
    cfg.setdefault("x03_comp_radius_px", 12.0)
    cfg.setdefault("x03_error_mode", "auto")
    cfg.setdefault("x03_control_apertures", cfg.get("stage07_control_apertures", 8))
    cfg.setdefault("x03_control_exclude_angle_deg", 25.0)
    cfg.setdefault("x03_bad_windows_A", _bad_windows_from_config(cfg))
    cfg.setdefault("x03_skyline_windows_A", cfg.get("skyline_windows_A", []))
    cfg.setdefault("x03_interpolated_windows_A", cfg.get("stage_e01_interpolated_windows_A", []))
    cfg.setdefault("x03_wframe", cfg.get("wavelength_frame", "topocentric"))
    cfg.setdefault("x03_bunit", cfg.get("cube_bunit", ""))
    cfg.setdefault("x03_crosstalk_windows_A", [[4860.0, 4880.0], [5880.0, 5905.0], [6560.0, 6570.0]])
    return cfg


def _load_psf_model(paths, cfg):
    path = Path(cfg.get("x03_psf_model_json") or paths["psf_model_json"])
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path), path


def _finite_percentiles(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": None, "p90": None}
    return {"median": float(np.nanmedian(arr)), "p90": float(np.nanpercentile(arr, 90.0))}


def _count_gt(values, threshold):
    arr = np.asarray(values, dtype=np.float64)
    return int(np.count_nonzero(np.isfinite(arr) & (np.abs(arr) > float(threshold))))


def _star_large_aperture_check(cube, star_yx, psf_model, star_product):
    radius = float(psf_model.get("norm_radius_px", 25.0))
    _, ny, nx = cube.shape
    weights = aperture_weights(ny, nx, star_yx, {"kind": "circle", "radius_px": radius})
    aperture_flux = np.nansum(np.asarray(cube) * weights[None, :, :], axis=(1, 2))
    ratio = _median_ratio(star_product.flux, aperture_flux)
    return {"vs_large_aperture_median_ratio": ratio}


def _comparison_metric(product, path):
    if not Path(path).exists():
        return None
    other = SpectrumProduct.read(path)
    return _median_ratio(product.flux, other.flux)


def _qc_payload(products: PsfFitProducts, cfg, paths, psf_model_path, stat_state, star_check, open_issues):
    fit = products.result
    crosstalk = crosstalk_metric(
        products.companion.wave_A,
        products.star.flux,
        products.companion.flux,
        cfg.get("x03_crosstalk_windows_A", []),
    )
    ratio = _median_ratio(products.companion.flux_err, products.companion.flux_err_emp)
    return {
        "stage": "x03_psffit",
        "run_id": str(cfg["run_id"]),
        "psf_model": str(psf_model_path),
        "fit_region": {
            "star_radius_px": float(cfg.get("x03_star_radius_px", 20.0)),
            "comp_radius_px": float(cfg.get("x03_comp_radius_px", 12.0)),
            "n_pixels_median": int(np.nanmedian(fit.npix)),
        },
        "positions": {"from": str(paths["stage01c_qc_json"]), "chromatic": False},
        "conditioning": {
            "condition_number_median": None if not np.any(np.isfinite(fit.condition_number)) else float(np.nanmedian(fit.condition_number)),
            "rho_ab_median": None if not np.any(np.isfinite(fit.rho_ab)) else float(np.nanmedian(fit.rho_ab)),
            "rho_bc_median": None if not np.any(np.isfinite(fit.rho_bc)) else float(np.nanmedian(fit.rho_bc)),
            "channels_rho_gt_0p5": _count_gt(fit.rho_ab, 0.5),
            "channels_rho_bc_gt_0p5": _count_gt(fit.rho_bc, 0.5),
        },
        "chi2r": {
            **_finite_percentiles(fit.chi2r),
            "inflation_column_written": False,
        },
        "crosstalk": crosstalk,
        "errors": {
            "mode": products.error_mode,
            "stat_vs_empirical_median_ratio": ratio,
            "n_controls": int(len(products.controls_yx)),
            "stat_input": stat_state,
        },
        "star_product_check": star_check,
        "quick_compare": {
            "vs_aperture_median_ratio": _comparison_metric(products.companion, paths["spec_aperture_object"]),
            "vs_optimal_median_ratio": _comparison_metric(products.companion, paths["spec_optimal_object"]),
        },
        "checks": {
            "v3_star_scale_ok": None
            if star_check["vs_large_aperture_median_ratio"] is None
            else bool(0.97 <= star_check["vs_large_aperture_median_ratio"] <= 1.03),
            "v4_rho_ab_ok": None
            if not np.any(np.isfinite(fit.rho_ab))
            else bool(np.nanmedian(np.abs(fit.rho_ab)) < 0.3),
            "rho_bc_warning": bool(np.any(np.isfinite(fit.rho_bc)) and np.nanmedian(np.abs(fit.rho_bc)) > 0.5),
        },
        "open_issues": list(open_issues),
    }


def compute_stage_x03_products(config, paths=None):
    cfg = dict(config)
    run_id = cfg["run_id"]
    root = Path(cfg.get("project_root") or Path.cwd()).resolve()
    paths = stage_x03_paths(run_id, root) if paths is None else paths
    star_yx_key = cfg.get("x03_star_key", "primary")
    comp_yx_key = cfg.get("x03_source_key", "companion")
    comp_yx, star_yx, _positions_path, _positions_qc = _load_positions(
        {
            **paths,
            "stage01c_qc_json": Path(cfg.get("x03_positions_qc", paths["stage01c_qc_json"])),
        },
        {"x01_source_key": comp_yx_key, "x01_star_key": star_yx_key},
    )
    psf_model, psf_model_path = _load_psf_model(paths, cfg)
    cube, wave, cube_path, bunit = _load_stage02_cube(paths, cfg)
    stat_cube, stat_state = _load_stat_cube(paths, cfg, cube.shape)
    stat_factor, covariance_factor, stat_status, qc00, _qc01 = _stat_metadata(paths, cfg)
    good_mask, bad_mask = _load_masks(paths, wave.size)
    open_issues = []
    if stat_cube is None:
        open_issues.append(f"STAT unavailable for X03 ({stat_state}); using empirical errors and estimated variance weights.")
    if str(stat_status).lower() == "red":
        open_issues.append("A4/M5 STAT status is red; products use empirical flux_err.")
    wframe = _wavelength_frame(cfg, qc00, open_issues)
    products = make_psffit_products(
        cube,
        wave,
        star_yx,
        comp_yx,
        psf_model,
        run_id=run_id,
        input_cube_path=cube_path,
        variance_zyx=stat_cube,
        stat_factor_spaxel=stat_factor,
        covariance_factor_box3=covariance_factor,
        stat_status=stat_status,
        error_mode=cfg.get("x03_error_mode", "auto"),
        wframe=wframe,
        bunit=bunit or cfg.get("x03_bunit", ""),
        star_radius_px=float(cfg.get("x03_star_radius_px", 20.0)),
        comp_radius_px=float(cfg.get("x03_comp_radius_px", 12.0)),
        bad_windows_A=cfg.get("x03_bad_windows_A", []),
        skyline_windows_A=cfg.get("x03_skyline_windows_A", []),
        interpolated_windows_A=cfg.get("x03_interpolated_windows_A", []),
        good_mask=good_mask,
        bad_mask=bad_mask,
        n_controls=int(cfg.get("x03_control_apertures", 8)),
        exclude_angle_deg=float(cfg.get("x03_control_exclude_angle_deg", 25.0)),
    )
    star_check = _star_large_aperture_check(cube, star_yx, psf_model, products.star)
    qc = _qc_payload(products, cfg, paths, psf_model_path, stat_state, star_check, open_issues)
    return StageX03Product(products=products, qc=qc)


def write_stage_x03_products(product: StageX03Product, config, paths):
    paths["paths"].ensure_base_dirs()
    product.products.companion.write(paths["spec_psffit_object"], overwrite=True)
    product.products.star.write(paths["spec_psffit_star"], overwrite=True)
    SpectrumProduct.read(paths["spec_psffit_object"])
    SpectrumProduct.read(paths["spec_psffit_star"])
    wave = product.products.companion.wave_A
    hdr = fits.Header()
    hdr["RUNID"] = str(config["run_id"])
    hdr["STAGE"] = "x03_psffit"
    hdr["WMIN"] = float(wave[0])
    hdr["WMAX"] = float(wave[-1])
    hdr["DW"] = float(np.nanmedian(np.diff(wave))) if wave.size > 1 else 0.0
    fits.HDUList(
        [
            fits.PrimaryHDU(product.products.result.residual_cube.astype(np.float32), header=hdr),
            fits.ImageHDU(wave.astype(np.float64), name="WAVELENGTH"),
            fits.ImageHDU(product.products.result.fit_mask.astype(np.uint8), name="FIT_MASK"),
        ]
    ).writeto(paths["cube_psffit_residual"], overwrite=True)
    # Persist companion control spectra for D1 (empirical sigma_diff source).
    # psffit amplitudes are already total NORMRAD flux (apcorr=1): calibrated
    # and raw coincide; both keys kept for uniformity with x01/x02 (D1 v2 §3.1).
    control_comp = np.asarray(product.products.control_comp_spectra, dtype=np.float64)
    np.savez(
        paths["paths"].stage_dir / "spec_psffit_controls.npz",
        control_spectra=control_comp,
        control_spectra_raw=control_comp,
        bkg_mode=np.asarray("psffit_plane"),
        apcorr_median=np.asarray(1.0),
    )
    qc = dict(product.qc)
    qc["products"] = {
        "object": str(paths["spec_psffit_object"]),
        "star": str(paths["spec_psffit_star"]),
        "residual_cube": str(paths["cube_psffit_residual"]),
    }
    write_json(paths["spec_psffit_qc_json"], qc)
    return {"products": qc["products"], "qc_json": paths["spec_psffit_qc_json"], "qc": qc}


def run_stage_x03(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage_x03_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage_x03_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage_x03_products(cfg, paths)
    written = write_stage_x03_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": written["qc"], "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage X03/C4 linear PSF fitting.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage_x03(
        args.run_id,
        project_root=args.project_root,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["spec_psffit_qc_json"])


__all__ = [
    "StageX03Product",
    "compute_stage_x03_products",
    "run_stage_x03",
    "stage_x03_config_from_run",
    "stage_x03_paths",
    "write_stage_x03_products",
]


if __name__ == "__main__":
    main()
