"""Stage 04b: local surface subtraction for far-object extraction."""

from __future__ import annotations

import argparse
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from ..io import read_json, write_csv, write_json
from ..localfit import fit_local_surface_2d, subtract_local_surface_cube
from ..paths import RunPaths
from ..spectral import nearest_channel_indices


HALPHA_CHANNELS_A = [6560.96, 6562.21, 6563.46]
HBETA_CHANNELS_A = [4860.96]
CONT_HA_RANGE_A = (6570.0, 6700.0)
CONT_HB_RANGE_A = (4800.0, 4930.0)


STAGE04B_DEFAULTS = {
    "input_mode": "native_stage02",
    "peak_selection_mode": "stage04_qc",
    "target_object": "c",
    "target_source_label": "object",
    "local_model_kind": "plane",
    "fit_radius_px": 12.0,
    "mask_radius_px": 3.0,
    "local_fit_sigma_clip": 3.0,
    "local_fit_max_iter": 3,
    "local_fit_min_pixels": 30,
    "mask_other_objects": True,
    "other_mask_radius_px": 3.0,
    "box_size": 3,
    "halpha_channels_A": HALPHA_CHANNELS_A,
    "hbeta_channels_A": HBETA_CHANNELS_A,
    "cont_ha_min_A": CONT_HA_RANGE_A[0],
    "cont_ha_max_A": CONT_HA_RANGE_A[1],
    "cont_hb_min_A": CONT_HB_RANGE_A[0],
    "cont_hb_max_A": CONT_HB_RANGE_A[1],
    "fig_convolve_sigma_px": 2.0,
    "n_best_cubes": 4,
}


def _as_yx(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if "y" in value and "x" in value:
            return (int(value["y"]), int(value["x"]))
        return None
    if len(value) != 2:
        return None
    return (int(value[0]), int(value[1]))


def _as_xy(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return (int(value["x"]), int(value["y"]))
        return None
    if len(value) != 2:
        return None
    return (int(value[0]), int(value[1]))


def _first_yx_from_config(cfg, yx_keys, xy_keys=()):
    for key in yx_keys:
        yx = _as_yx(cfg.get(key))
        if yx is not None:
            return yx
    for key in xy_keys:
        xy = _as_xy(cfg.get(key))
        if xy is not None:
            return (xy[1], xy[0])
    return None


def center_yx(ny, nx):
    return (int(ny) // 2, int(nx) // 2)


def peak_inside(yx, ny, nx, box_half=0):
    if yx is None or len(yx) != 2:
        return False
    y, x = int(yx[0]), int(yx[1])
    return (int(box_half) <= y < int(ny) - int(box_half)) and (
        int(box_half) <= x < int(nx) - int(box_half)
    )


def validate_peak_yx(label, yx, ny, nx, box_half=0, allow_center_fallback=False):
    """Validate one source position inside the current cube footprint."""

    if yx is None or len(yx) != 2:
        if allow_center_fallback:
            fallback = center_yx(ny, nx)
            warnings.warn(f"{label} is missing; using crop center {fallback} as fallback.")
            return fallback
        raise ValueError(f"{label} is missing. Set it in config or Stage04 QC before running Stage 04b.")

    y, x = int(yx[0]), int(yx[1])
    if peak_inside((y, x), ny, nx, box_half=box_half):
        return (y, x)

    if allow_center_fallback:
        fallback = center_yx(ny, nx)
        warnings.warn(
            f"{label}={(y, x)} is outside cube bounds for shape (ny={ny}, nx={nx}); "
            f"using crop center {fallback} as fallback."
        )
        return fallback

    raise ValueError(
        f"{label}={(y, x)} is outside cube bounds for shape (ny={ny}, nx={nx}) "
        f"with box_half={box_half}."
    )


def _read_peak_from_qc(peaks, key):
    if not isinstance(peaks, dict):
        return None
    item = peaks.get(key)
    if item is None:
        return None
    return _as_yx(item)


def stage04b_paths(run_id, project_root=None):
    """Return the Stage 04b paths, preserving the historical filenames."""

    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage04b")
    return {
        "paths": paths,
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "stage03_cube_fits": paths.stage_dir / "stage03_fakecont_cube_stack.fits",
        "stage03_qc_json": paths.stage_dir / "stage03_qc.json",
        "stage04_pca_qc_json": paths.stage_dir / "stage04_qc.json",
        "stage04b_cube_fits": paths.stage_dir / "stage04b_local_surface_cube_stack.fits",
        "stage04b_qc_json": paths.stage_dir / "stage04b_qc.json",
        "stage04_qc_json": paths.stage_dir / "stage04_qc.json",
        "best4_npy": paths.stage_dir / "stage04_best4_indices.npy",
        "worst2_npy": paths.stage_dir / "stage04_worst2_indices.npy",
        "cube_residual_best4": paths.stage_dir / "cube_residual_best4.fits",
        "cube_localmodel_best4": paths.stage_dir / "cube_localmodel_best4.fits",
        "cube_fakecont_best4": paths.stage_dir / "cube_fakecont_best4.fits",
        "cube_residual_object": paths.stage_dir / "cube_residual_local_object.fits",
        "cube_localmodel_object": paths.stage_dir / "cube_localmodel_object.fits",
        "cube_input_object": paths.stage_dir / "cube_input_local_object.fits",
        "good_wave_mask_npy": paths.stage_dir / "stage04b_good_wavelength_mask.npy",
        "bad_wave_mask_npy": paths.stage_dir / "stage04b_bad_wavelength_mask.npy",
        "ha_image_best4": paths.stage_dir / "Ha_image_best4.fits",
        "hb_image_best4": paths.stage_dir / "Hb_image_best4.fits",
        "ha_snr_best4": paths.stage_dir / "Ha_SNR_best4.fits",
        "hb_snr_best4": paths.stage_dir / "Hb_SNR_best4.fits",
        "coordinate_check_cube": paths.stage_dir / "stage04b_input_cube_used_for_coordinate_check.fits",
        "ranking_csv": paths.table_dir / "stage04b_cube_ranking.csv",
        "plot_dir": plot_dir,
        "summary_fig": plot_dir / "stage04b_summary_best4.png",
    }


def _bad_wavelength_ranges_from_config(cfg):
    ranges = cfg.get("stage04b_bad_wavelength_ranges_A") or cfg.get("bad_wavelength_ranges_A")
    if ranges is not None:
        return [[float(lo), float(hi)] for lo, hi in ranges]
    drop_min = cfg.get("drop_wave_min_A")
    drop_max = cfg.get("drop_wave_max_A")
    if drop_min is None or drop_max is None:
        return []
    return [[float(drop_min), float(drop_max)]]


def stage04b_config_from_run(
    run_id=None,
    *,
    project_root=None,
    overrides=None,
    allow_run_id_mismatch=False,
):
    """Build Stage 04b config from the active run config."""

    run_config = load_run_config(
        run_id,
        project_root=project_root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    cfg = run_config.config
    paths = run_config.paths

    n_jobs_default = min(4, int(cfg.get("xcorr_n_jobs", os.cpu_count() or 1)))
    config = {
        **STAGE04B_DEFAULTS,
        "run_id": run_config.run_id,
        "target_name": cfg.get("target_name", run_config.run_id),
        "project_root": str(paths.project_root),
        "n_jobs": int(cfg.get("stage04b_n_jobs", n_jobs_default)),
        "bad_wavelength_ranges_A": _bad_wavelength_ranges_from_config(cfg),
        "manual_peak_b_yx": _first_yx_from_config(
            cfg,
            (
                "stage04b_manual_peak_b_yx",
                "manual_peak_b_yx",
                "peak_b_yx",
                "star_yx",
                "reference_geometry_center_yx",
            ),
            (
                "stage04b_manual_peak_b_xy",
                "manual_peak_b_xy",
                "peak_b_xy",
                "star_xy",
                "reference_geometry_center_xy",
            ),
        ),
        "manual_peak_c_yx": _first_yx_from_config(
            cfg,
            (
                "stage04b_manual_peak_c_yx",
                "manual_peak_c_yx",
                "peak_c_yx",
                "object_yx",
                "target_yx",
                "science_object_yx",
            ),
            (
                "stage04b_manual_peak_c_xy",
                "manual_peak_c_xy",
                "peak_c_xy",
                "object_xy",
                "target_xy",
                "science_object_xy",
            ),
        ),
    }

    for key, default_value in STAGE04B_DEFAULTS.items():
        config[key] = cfg.get(f"stage04b_{key}", cfg.get(key, default_value))
    config["input_mode"] = cfg.get("stage04b_input_mode", config["input_mode"])
    config["target_object"] = cfg.get("stage04b_target_object", config["target_object"])
    config["target_source_label"] = cfg.get("stage04b_target_source_label", config["target_source_label"])
    config["halpha_channels_A"] = cfg.get("ha_channels_A", config["halpha_channels_A"])
    config["hbeta_channels_A"] = cfg.get("hb_channels_A", config["hbeta_channels_A"])
    config["box_size"] = int(cfg.get("box_aperture_size_px", config["box_size"]))
    config["fig_convolve_sigma_px"] = float(cfg.get("fig_convolve_sigma_px", config["fig_convolve_sigma_px"]))

    if overrides:
        config.update(overrides)
    return config


def build_bad_wavelength_mask(wavelengths, bad_wavelength_ranges_A):
    waves = np.asarray(wavelengths, dtype=np.float64)
    bad = np.zeros(waves.shape, dtype=bool)
    for wmin, wmax in bad_wavelength_ranges_A or []:
        bad |= (waves >= float(wmin)) & (waves <= float(wmax))
    return bad


def _normalise_medpix_stack(med_pix_stack, n_cubes, ny, nx):
    if med_pix_stack is None:
        return np.ones((n_cubes, ny, nx), dtype=np.float32)
    med = np.asarray(med_pix_stack, dtype=np.float32)
    if med.ndim == 2:
        med = np.broadcast_to(med[None, :, :], (n_cubes, ny, nx)).copy()
    if med.shape != (n_cubes, ny, nx):
        raise RuntimeError(f"MEDPIX shape {med.shape} does not match {(n_cubes, ny, nx)}.")
    return med


def load_stage04b_input(config, paths):
    """Load Stage 04b input cubes from Stage 02 or Stage 03."""

    input_mode = config["input_mode"]
    if input_mode == "native_stage02":
        input_cube_fits = paths["stage02_cube_fits"]
        if not input_cube_fits.exists():
            raise FileNotFoundError(input_cube_fits)
        with fits.open(input_cube_fits, memmap=True) as hdul:
            cubes = hdul["CUBES"].data.astype(np.float32)
            wavelengths = hdul["WAVELENGTH"].data.astype(np.float64)
        input_units_label = "native_stage02_physical_like"
        med_pix_stack = None
    elif input_mode == "fakecont_stage03":
        input_cube_fits = paths["stage03_cube_fits"]
        if not input_cube_fits.exists():
            raise FileNotFoundError(input_cube_fits)
        with fits.open(input_cube_fits, memmap=True) as hdul:
            cubes = hdul["CUBES"].data.astype(np.float32)
            wavelengths = hdul["WAVELENGTH"].data.astype(np.float64)
            med_pix_stack = hdul["MEDPIX"].data.astype(np.float32) if "MEDPIX" in hdul else None
        input_units_label = "stage03_fakecont_rescaled"
    else:
        raise ValueError(f"Unknown Stage 04b input_mode: {input_mode}")

    if cubes.ndim == 3:
        cubes = cubes[None, :, :, :]
    if cubes.ndim != 4:
        raise RuntimeError(f"Expected input cubes with shape (N,nz,ny,nx), got {cubes.shape}.")

    n_cubes, n_wave, ny, nx = cubes.shape
    if n_wave != len(wavelengths):
        raise RuntimeError(f"Cube wavelength axis {n_wave} != wavelength array {len(wavelengths)}.")

    med_pix_stack = _normalise_medpix_stack(med_pix_stack, n_cubes, ny, nx)
    bad_wave_mask = build_bad_wavelength_mask(wavelengths, config.get("bad_wavelength_ranges_A", []))
    good_wave_mask = np.isfinite(wavelengths) & ~bad_wave_mask
    if np.any(bad_wave_mask):
        cubes[:, bad_wave_mask, :, :] = np.nan

    return {
        "cubes": cubes,
        "wavelengths": wavelengths,
        "med_pix_stack": med_pix_stack,
        "good_wave_mask": good_wave_mask,
        "bad_wave_mask": bad_wave_mask,
        "input_cube_fits": input_cube_fits,
        "input_units_label": input_units_label,
    }


def _smooth_image(image, sigma_px):
    if sigma_px is None or float(sigma_px) <= 0:
        return image
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(image, sigma=float(sigma_px))


def make_line_image(cube_zyx, line_idxs, sigma_px=2.0):
    img = np.nanmean(np.asarray(cube_zyx)[np.asarray(line_idxs, dtype=int)], axis=0)
    img = _smooth_image(img, sigma_px)
    return img.astype(np.float32)


def continuum_sigma_map(cube_zyx, cont_mask, sigma_px=2.0):
    sig = np.nanstd(np.asarray(cube_zyx)[np.asarray(cont_mask, dtype=bool)], axis=0)
    sig = _smooth_image(sig, sigma_px)
    return sig.astype(np.float32)


def make_line_snr_map(cube_zyx, line_idxs, cont_mask, sigma_px=2.0):
    line_img = make_line_image(cube_zyx, line_idxs, sigma_px=sigma_px)
    noise = continuum_sigma_map(cube_zyx, cont_mask, sigma_px=sigma_px)
    snr = line_img / np.maximum(noise, 1e-8)
    snr[~np.isfinite(snr)] = np.nan
    return line_img, snr.astype(np.float32)


def _nanmean_axis0(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=0)


def resolve_stage04b_peak_candidates(config, stage04_qc=None):
    """Resolve b/c candidate positions from Stage04 QC or manual config."""

    stage04_qc = {} if stage04_qc is None else stage04_qc
    manual_b = _as_yx(config.get("manual_peak_b_yx"))
    manual_c = _as_yx(config.get("manual_peak_c_yx"))
    peak_selection_mode = config.get("peak_selection_mode", "stage04_qc")

    if peak_selection_mode == "manual":
        return manual_b, manual_c, "manual"
    if peak_selection_mode != "stage04_qc":
        raise ValueError(f"Unknown peak_selection_mode: {peak_selection_mode}")

    peaks = stage04_qc.get("detected_peaks", {})
    b_yx = _read_peak_from_qc(peaks, "b") or _read_peak_from_qc(peaks, "star")
    c_yx = _read_peak_from_qc(peaks, "c") or _read_peak_from_qc(peaks, "object")

    if b_yx is None and c_yx is None:
        if manual_b is not None or manual_c is not None:
            warnings.warn("No usable detected_peaks in Stage04 QC; falling back to manual Stage 04b peaks.")
        return manual_b, manual_c, "manual_fallback_no_qc_peaks"

    return b_yx or manual_b, c_yx or manual_c, str(stage04_qc.get("stage", "stage04_qc"))


def resolve_stage04b_positions(config, stage04_qc, spatial_shape):
    """Validate b/c positions and return target/companion coordinates."""

    ny, nx = map(int, spatial_shape)
    box_half = int(config.get("box_size", 3)) // 2
    candidate_b_yx, candidate_c_yx, peak_source = resolve_stage04b_peak_candidates(config, stage04_qc)

    target_object = config.get("target_object", "c")
    peak_b_yx = validate_peak_yx(
        "PEAK_B_YX",
        candidate_b_yx,
        ny,
        nx,
        box_half=box_half,
        allow_center_fallback=(target_object != "b"),
    )
    peak_c_yx = validate_peak_yx(
        "PEAK_C_YX",
        candidate_c_yx,
        ny,
        nx,
        box_half=box_half,
        allow_center_fallback=(target_object != "c"),
    )

    if target_object == "b":
        target_yx = peak_b_yx
        other_yx = peak_c_yx
    elif target_object == "c":
        target_yx = peak_c_yx
        other_yx = peak_b_yx
    else:
        raise ValueError("target_object must be 'b' or 'c'.")

    return {
        "peak_b_yx": peak_b_yx,
        "peak_c_yx": peak_c_yx,
        "target_yx": target_yx,
        "other_yx": other_yx,
        "peak_source": peak_source,
    }


def compute_stage04b_products(
    cubes_fc_norm,
    wavelengths,
    med_pix_stack,
    config,
    *,
    stage04_qc=None,
    good_wave_mask=None,
    bad_wave_mask=None,
    copy_input=True,
):
    """Compute Stage 04b residual/model cubes, ranking, line maps, and metadata."""

    config = {**STAGE04B_DEFAULTS, **dict(config)}
    cubes = np.array(cubes_fc_norm, dtype=np.float32, copy=copy_input)
    if cubes.ndim != 4:
        raise ValueError(f"Expected cubes with shape (N,nz,ny,nx), got {cubes.shape}.")
    n_cubes, n_wave, ny, nx = cubes.shape
    if n_cubes <= 0:
        raise ValueError("Stage 04b requires at least one cube.")

    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if n_wave != len(wavelengths):
        raise RuntimeError(f"Cube wavelength axis {n_wave} != wavelength array {len(wavelengths)}.")

    if bad_wave_mask is None:
        bad_wave_mask = build_bad_wavelength_mask(wavelengths, config.get("bad_wavelength_ranges_A", []))
    else:
        bad_wave_mask = np.asarray(bad_wave_mask, dtype=bool)
    if good_wave_mask is None:
        good_wave_mask = np.isfinite(wavelengths) & ~bad_wave_mask
    else:
        good_wave_mask = np.asarray(good_wave_mask, dtype=bool) & ~bad_wave_mask

    if np.any(bad_wave_mask):
        cubes[:, bad_wave_mask, :, :] = np.nan

    med_pix_stack = _normalise_medpix_stack(med_pix_stack, n_cubes, ny, nx)
    positions = resolve_stage04b_positions(config, stage04_qc or {}, (ny, nx))
    target_yx = positions["target_yx"]
    other_yx = positions["other_yx"]

    cubes_res_norm = np.empty_like(cubes, dtype=np.float32)
    cubes_mod_norm = np.empty_like(cubes, dtype=np.float32)
    nfit_per_cube = [None] * n_cubes
    extra_exclusion_yx = [other_yx] if config.get("mask_other_objects", True) and other_yx is not None else None

    def subtract_one_cube(index):
        residual, model, nfit = subtract_local_surface_cube(
            cubes[index],
            target_yx=target_yx,
            fit_radius_px=float(config["fit_radius_px"]),
            mask_radius_px=float(config["mask_radius_px"]),
            model_kind=config["local_model_kind"],
            extra_exclusion_yx=extra_exclusion_yx,
            extra_exclusion_radius_px=float(config["other_mask_radius_px"]),
            sigma_clip=float(config["local_fit_sigma_clip"]),
            max_iter=int(config["local_fit_max_iter"]),
            min_fit_pixels=int(config["local_fit_min_pixels"]),
        )
        return index, residual, model, nfit

    n_jobs = max(1, min(int(config.get("n_jobs", 1)), n_cubes))
    t0 = time.perf_counter()
    if n_jobs == 1:
        for index in range(n_cubes):
            out_index, residual, model, nfit = subtract_one_cube(index)
            cubes_res_norm[out_index] = residual
            cubes_mod_norm[out_index] = model
            nfit_per_cube[out_index] = nfit
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(subtract_one_cube, index): index for index in range(n_cubes)}
            for fut in as_completed(futures):
                out_index, residual, model, nfit = fut.result()
                cubes_res_norm[out_index] = residual
                cubes_mod_norm[out_index] = model
                nfit_per_cube[out_index] = nfit
    elapsed_s = time.perf_counter() - t0
    nfit_per_cube = np.asarray(nfit_per_cube)

    cubes_res_phys = cubes_res_norm * med_pix_stack[:, None, :, :]
    cubes_mod_phys = cubes_mod_norm * med_pix_stack[:, None, :, :]
    cubes_input_phys = cubes * med_pix_stack[:, None, :, :]

    ha_idxs = nearest_channel_indices(wavelengths, config["halpha_channels_A"])
    hb_idxs = nearest_channel_indices(wavelengths, config["hbeta_channels_A"])
    cont_ha = (
        good_wave_mask
        & (wavelengths >= float(config["cont_ha_min_A"]))
        & (wavelengths <= float(config["cont_ha_max_A"]))
    )
    cont_hb = (
        good_wave_mask
        & (wavelengths >= float(config["cont_hb_min_A"]))
        & (wavelengths <= float(config["cont_hb_max_A"]))
    )

    ha_imgs = []
    hb_imgs = []
    ha_snr_maps = []
    hb_snr_maps = []
    for index in range(n_cubes):
        ha_img_i, ha_snr_i = make_line_snr_map(
            cubes_res_phys[index],
            ha_idxs,
            cont_ha,
            sigma_px=float(config["fig_convolve_sigma_px"]),
        )
        hb_img_i, hb_snr_i = make_line_snr_map(
            cubes_res_phys[index],
            hb_idxs,
            cont_hb,
            sigma_px=float(config["fig_convolve_sigma_px"]),
        )
        ha_imgs.append(ha_img_i)
        hb_imgs.append(hb_img_i)
        ha_snr_maps.append(ha_snr_i)
        hb_snr_maps.append(hb_snr_i)

    ha_imgs = np.asarray(ha_imgs, dtype=np.float32)
    hb_imgs = np.asarray(hb_imgs, dtype=np.float32)
    ha_snr_maps = np.asarray(ha_snr_maps, dtype=np.float32)
    hb_snr_maps = np.asarray(hb_snr_maps, dtype=np.float32)

    yb, xb = positions["peak_b_yx"]
    yc, xc = positions["peak_c_yx"]
    cube_scores = []
    for index in range(n_cubes):
        score_b = float(ha_snr_maps[index, yb, xb])
        score_c = float(ha_snr_maps[index, yc, xc])
        score_target = score_c if config["target_object"] == "c" else score_b
        cube_scores.append((index, score_target, score_b, score_c))

    cube_scores_sorted = sorted(
        cube_scores,
        key=lambda row: -np.nan_to_num(row[1], nan=-np.inf),
    )
    n_best = min(int(config.get("n_best_cubes", 4)), n_cubes)
    best4 = [int(row[0]) for row in cube_scores_sorted[:n_best]]
    worst2 = [int(row[0]) for row in cube_scores_sorted[n_best:]]

    cube_res_best4 = _nanmean_axis0(cubes_res_phys[best4]).astype(np.float32)
    cube_mod_best4 = _nanmean_axis0(cubes_mod_phys[best4]).astype(np.float32)
    cube_fc_best4 = _nanmean_axis0(cubes_input_phys[best4]).astype(np.float32)

    ha_img_best4, ha_snr_best4 = make_line_snr_map(
        cube_res_best4,
        ha_idxs,
        cont_ha,
        sigma_px=float(config["fig_convolve_sigma_px"]),
    )
    hb_img_best4, hb_snr_best4 = make_line_snr_map(
        cube_res_best4,
        hb_idxs,
        cont_hb,
        sigma_px=float(config["fig_convolve_sigma_px"]),
    )

    return {
        **positions,
        "elapsed_s": float(elapsed_s),
        "cubes_res_phys": cubes_res_phys.astype(np.float32),
        "cubes_mod_phys": cubes_mod_phys.astype(np.float32),
        "cubes_input_phys": cubes_input_phys.astype(np.float32),
        "cube_res_best4": cube_res_best4,
        "cube_mod_best4": cube_mod_best4,
        "cube_fc_best4": cube_fc_best4,
        "ha_imgs": ha_imgs,
        "hb_imgs": hb_imgs,
        "ha_snr_maps": ha_snr_maps,
        "hb_snr_maps": hb_snr_maps,
        "ha_img_best4": ha_img_best4,
        "hb_img_best4": hb_img_best4,
        "ha_snr_best4": ha_snr_best4,
        "hb_snr_best4": hb_snr_best4,
        "wavelengths": wavelengths,
        "good_wave_mask": good_wave_mask,
        "bad_wave_mask": bad_wave_mask,
        "ha_idxs": ha_idxs,
        "hb_idxs": hb_idxs,
        "cont_ha_mask": cont_ha,
        "cont_hb_mask": cont_hb,
        "cube_scores_sorted": cube_scores_sorted,
        "best4": best4,
        "worst2": worst2,
        "nfit_per_cube": nfit_per_cube,
    }


def make_basic_header(wavelengths, config, input_cube_fits, bad_wave_mask):
    hdr = fits.Header()
    hdr["WMIN"] = float(wavelengths[0])
    hdr["WMAX"] = float(wavelengths[-1])
    hdr["DW"] = float(np.nanmedian(np.diff(wavelengths)))
    hdr["BUNIT"] = "physical_like"
    hdr["HIERARCH REDUCTION"] = "local_surface_subtraction"
    hdr["HIERARCH INPUT_MODE"] = config["input_mode"]
    hdr["HIERARCH INPUT_FILE"] = Path(input_cube_fits).name
    hdr["HIERARCH LOCAL_MODEL"] = config["local_model_kind"]
    hdr["HIERARCH FIT_RADIUS"] = float(config["fit_radius_px"])
    hdr["HIERARCH MASK_RADIUS"] = float(config["mask_radius_px"])
    hdr["HIERARCH SIGCLIP"] = float(config["local_fit_sigma_clip"])
    hdr["HIERARCH CLIPITER"] = int(config["local_fit_max_iter"])
    hdr["HIERARCH MINFITPX"] = int(config["local_fit_min_pixels"])
    hdr["HIERARCH TARGET_LABEL"] = config["target_source_label"]
    hdr["HIERARCH TARGET_OBJECT"] = config["target_object"]
    hdr["HIERARCH TARGET_Y"] = int(config["target_yx"][0])
    hdr["HIERARCH TARGET_X"] = int(config["target_yx"][1])
    hdr["HIERARCH BAD_NCHAN"] = int(np.count_nonzero(bad_wave_mask))
    hdr["HIERARCH BAD_WAVE_RANGES"] = str(config.get("bad_wavelength_ranges_A", []))
    return hdr


def save_stage04b_coordinate_check_cube(cube_zyx, path, wavelengths, peak_b_yx, peak_c_yx, source_name):
    hdr = fits.Header()
    hdr["WMIN"] = float(wavelengths[0])
    hdr["WMAX"] = float(wavelengths[-1])
    hdr["DW"] = float(np.nanmedian(np.diff(wavelengths)))
    hdr["CRPIX3"] = 1
    hdr["CRVAL3"] = float(wavelengths[0])
    hdr["CDELT3"] = float(np.nanmedian(np.diff(wavelengths)))
    hdr["CTYPE3"] = "WAVE"
    hdr["CUNIT3"] = "Angstrom"
    hdr["BUNIT"] = "physical_like"
    hdr["HIERARCH SOURCE"] = source_name
    hdr["HIERARCH STAGE"] = "04b_local_surface_subtraction"
    hdr["HIERARCH COORD_NOTE"] = "Python arrays use cube[z,y,x]. DS9 displays x,y."
    hdr["HIERARCH B_Y"] = int(peak_b_yx[0])
    hdr["HIERARCH B_X"] = int(peak_b_yx[1])
    hdr["HIERARCH C_Y"] = int(peak_c_yx[0])
    hdr["HIERARCH C_X"] = int(peak_c_yx[1])
    fits.PrimaryHDU(data=np.asarray(cube_zyx, dtype=np.float32), header=hdr).writeto(path, overwrite=True)


def stage04b_qc_payload(products, config, paths):
    n_cubes = products["cubes_res_phys"].shape[0]
    peak_b_yx = products["peak_b_yx"]
    peak_c_yx = products["peak_c_yx"]
    target_yx = products["target_yx"]
    return {
        "run_id": config["run_id"],
        "target_name": config.get("target_name", config["run_id"]),
        "stage": "stage04b_local_surface_subtraction",
        "input_shape": [int(x) for x in products["cubes_input_phys"].shape],
        "output_shape": [int(x) for x in products["cubes_res_phys"].shape],
        "method": "local_surface_subtraction",
        "input_mode": config["input_mode"],
        "input_cube_fits": str(config["input_cube_fits"]),
        "input_units_label": config["input_units_label"],
        "object_cube_fits": str(paths["cube_residual_object"]),
        "local_model_kind": config["local_model_kind"],
        "target_object": config["target_object"],
        "target_source_label": config["target_source_label"],
        "target_yx": [int(target_yx[0]), int(target_yx[1])],
        "target_xy": [int(target_yx[1]), int(target_yx[0])],
        "fit_radius_px": float(config["fit_radius_px"]),
        "mask_radius_px": float(config["mask_radius_px"]),
        "local_fit_sigma_clip": float(config["local_fit_sigma_clip"]),
        "local_fit_max_iter": int(config["local_fit_max_iter"]),
        "local_fit_min_pixels": int(config["local_fit_min_pixels"]),
        "stage04b_n_jobs": int(config["n_jobs"]),
        "mask_other_objects": bool(config["mask_other_objects"]),
        "other_mask_radius_px": float(config["other_mask_radius_px"]),
        "best4": [int(x) for x in products["best4"]],
        "worst2": [int(x) for x in products["worst2"]],
        "detected_peaks": {
            "star": {"y": int(peak_b_yx[0]), "x": int(peak_b_yx[1]), "role": "central_star_placeholder"},
            "object": {"y": int(target_yx[0]), "x": int(target_yx[1]), "role": "science_object"},
            "b": {"y": int(peak_b_yx[0]), "x": int(peak_b_yx[1]), "role": "central_star_placeholder"},
            "c": {"y": int(peak_c_yx[0]), "x": int(peak_c_yx[1]), "role": "science_object"},
        },
        "bad_wavelength_ranges_A": [
            [float(wmin), float(wmax)] for wmin, wmax in config.get("bad_wavelength_ranges_A", [])
        ],
        "bad_channel_count": int(np.count_nonzero(products["bad_wave_mask"])),
        "good_wavelength_mask_npy": str(paths["good_wave_mask_npy"]),
        "bad_wavelength_mask_npy": str(paths["bad_wave_mask_npy"]),
        "ha_idxs": [int(x) for x in products["ha_idxs"]],
        "hb_idxs": [int(x) for x in products["hb_idxs"]],
        "halpha_channels_A": [float(x) for x in config["halpha_channels_A"]],
        "hbeta_channels_A": [float(x) for x in config["hbeta_channels_A"]],
        "continuum_windows_A": {
            "Ha": [float(config["cont_ha_min_A"]), float(config["cont_ha_max_A"])],
            "Hb": [float(config["cont_hb_min_A"]), float(config["cont_hb_max_A"])],
        },
        "cube_scores": [
            {
                "cube_index": int(index),
                "score_target": float(score),
                "score_b": float(score_b),
                "score_c": float(score_c),
            }
            for index, score, score_b, score_c in products["cube_scores_sorted"]
        ],
        "nfit_median_per_cube": [
            float(np.nanmedian(products["nfit_per_cube"][index])) for index in range(n_cubes)
        ],
        "peak_source": products["peak_source"],
        "elapsed_s": float(products["elapsed_s"]),
    }


def show_image_with_safe_colorbar(ax, img, title):
    import matplotlib.pyplot as plt

    finite = np.isfinite(img)
    if np.any(finite):
        vmin, vmax = np.nanpercentile(img[finite], [1, 99])
        im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(img, origin="lower")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def save_stage04b_summary_plot(plot_path, products, config, show_plots=False):
    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    show_image_with_safe_colorbar(axes[0, 0], products["ha_img_best4"], "Halpha image local-sub best")
    show_image_with_safe_colorbar(axes[0, 1], products["ha_snr_best4"], "Halpha S/N local-sub best")
    show_image_with_safe_colorbar(axes[1, 0], products["hb_img_best4"], "Hbeta image local-sub best")
    show_image_with_safe_colorbar(axes[1, 1], products["hb_snr_best4"], "Hbeta S/N local-sub best")

    yb, xb = products["peak_b_yx"]
    yc, xc = products["peak_c_yx"]
    for ax in axes.ravel():
        ax.scatter([xb, xc], [yb, yc], marker="x", s=90)
        ax.text(xb + 1, yb + 1, "b", fontsize=11)
        ax.text(xc + 1, yc + 1, "c", fontsize=11)

    fig.suptitle(
        f"Stage04b local surface subtraction | target={config['target_object']} | best={products['best4']}",
        fontsize=13,
    )
    fig.savefig(plot_path, dpi=200)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    return plot_path


def write_stage04b_products(products, config, paths, *, save_plots=True, show_plots=False):
    """Write FITS, CSV, NPY, QC JSON, and optional summary plot products."""

    stage_paths = paths["paths"]
    stage_paths.stage_dir.mkdir(parents=True, exist_ok=True)
    stage_paths.table_dir.mkdir(parents=True, exist_ok=True)
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)

    wavelengths = products["wavelengths"]
    config = dict(config)
    config["target_yx"] = products["target_yx"]
    hdr_cube = make_basic_header(wavelengths, config, config["input_cube_fits"], products["bad_wave_mask"])

    np.save(paths["good_wave_mask_npy"], products["good_wave_mask"])
    np.save(paths["bad_wave_mask_npy"], products["bad_wave_mask"])
    np.save(paths["best4_npy"], np.asarray(products["best4"], dtype=int))
    np.save(paths["worst2_npy"], np.asarray(products["worst2"], dtype=int))

    fits.PrimaryHDU(products["cube_res_best4"], header=hdr_cube).writeto(paths["cube_residual_best4"], overwrite=True)
    fits.PrimaryHDU(products["cube_res_best4"], header=hdr_cube).writeto(paths["cube_residual_object"], overwrite=True)
    fits.PrimaryHDU(products["cube_mod_best4"], header=hdr_cube).writeto(paths["cube_localmodel_best4"], overwrite=True)
    fits.PrimaryHDU(products["cube_mod_best4"], header=hdr_cube).writeto(paths["cube_localmodel_object"], overwrite=True)
    fits.PrimaryHDU(products["cube_fc_best4"], header=hdr_cube).writeto(paths["cube_fakecont_best4"], overwrite=True)
    fits.PrimaryHDU(products["cube_fc_best4"], header=hdr_cube).writeto(paths["cube_input_object"], overwrite=True)

    fits.PrimaryHDU(products["ha_img_best4"]).writeto(paths["ha_image_best4"], overwrite=True)
    fits.PrimaryHDU(products["hb_img_best4"]).writeto(paths["hb_image_best4"], overwrite=True)
    fits.PrimaryHDU(products["ha_snr_best4"]).writeto(paths["ha_snr_best4"], overwrite=True)
    fits.PrimaryHDU(products["hb_snr_best4"]).writeto(paths["hb_snr_best4"], overwrite=True)

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(products["cubes_res_phys"].astype(np.float32), name="RESIDUALS"),
            fits.ImageHDU(products["cubes_mod_phys"].astype(np.float32), name="LOCAL_MODEL"),
            fits.ImageHDU(wavelengths.astype(np.float64), name="WAVELENGTH"),
            fits.ImageHDU(products["good_wave_mask"].astype(np.uint8), name="GOOD_WAVE_MASK"),
            fits.ImageHDU(products["bad_wave_mask"].astype(np.uint8), name="BAD_WAVE_MASK"),
        ]
    ).writeto(paths["stage04b_cube_fits"], overwrite=True)

    save_stage04b_coordinate_check_cube(
        products["cube_fc_best4"],
        paths["coordinate_check_cube"],
        wavelengths,
        products["peak_b_yx"],
        products["peak_c_yx"],
        source_name="cube_fc_best4",
    )

    ranking_rows = [
        {
            "rank": rank,
            "cube_index": int(index),
            "score_target": float(score),
            "score_b": float(score_b),
            "score_c": float(score_c),
        }
        for rank, (index, score, score_b, score_c) in enumerate(products["cube_scores_sorted"], start=1)
    ]
    write_csv(paths["ranking_csv"], ranking_rows, ["rank", "cube_index", "score_target", "score_b", "score_c"])

    qc = stage04b_qc_payload(products, config, paths)
    write_json(paths["stage04b_qc_json"], qc)
    write_json(paths["stage04_qc_json"], qc)

    plot_paths = []
    if save_plots:
        plot_paths.append(save_stage04b_summary_plot(paths["summary_fig"], products, config, show_plots=show_plots))

    return {
        "stage04b_cube_fits": paths["stage04b_cube_fits"],
        "summary_csv": paths["ranking_csv"],
        "qc_json": paths["stage04b_qc_json"],
        "stage04_qc_json": paths["stage04_qc_json"],
        "plots": plot_paths,
        "qc": qc,
        "ranking_rows": ranking_rows,
    }


def run_stage04b(
    config=None,
    *,
    show_plots=False,
    save_plots=True,
    project_root=None,
    allow_run_id_mismatch=False,
):
    """Run Stage 04b and write products compatible with the historical notebook."""

    if config is None:
        config = stage04b_config_from_run(
            project_root=project_root,
            allow_run_id_mismatch=allow_run_id_mismatch,
        )
    else:
        config = dict(config)

    run_id = config["run_id"]
    root = Path(project_root or config.get("project_root") or Path.cwd()).resolve()
    paths = stage04b_paths(run_id, root)
    stage04_qc = read_json(paths["stage04_pca_qc_json"]) if paths["stage04_pca_qc_json"].exists() else {}
    input_payload = load_stage04b_input(config, paths)
    config["input_cube_fits"] = str(input_payload["input_cube_fits"])
    config["input_units_label"] = input_payload["input_units_label"]

    products = compute_stage04b_products(
        input_payload["cubes"],
        input_payload["wavelengths"],
        input_payload["med_pix_stack"],
        config,
        stage04_qc=stage04_qc,
        good_wave_mask=input_payload["good_wave_mask"],
        bad_wave_mask=input_payload["bad_wave_mask"],
        copy_input=False,
    )
    written = write_stage04b_products(products, config, paths, save_plots=save_plots, show_plots=show_plots)
    return {
        **written,
        "products": products,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage 04b local-surface subtraction.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    args = parser.parse_args(argv)
    config = stage04b_config_from_run(
        args.run_id,
        project_root=args.project_root,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    result = run_stage04b(
        config,
        project_root=args.project_root,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["qc_json"])


__all__ = [
    "STAGE04B_DEFAULTS",
    "build_bad_wavelength_mask",
    "center_yx",
    "compute_stage04b_products",
    "continuum_sigma_map",
    "load_stage04b_input",
    "make_line_image",
    "make_line_snr_map",
    "peak_inside",
    "main",
    "resolve_stage04b_peak_candidates",
    "resolve_stage04b_positions",
    "run_stage04b",
    "save_stage04b_summary_plot",
    "stage04b_config_from_run",
    "stage04b_paths",
    "validate_peak_yx",
    "write_stage04b_products",
]


if __name__ == "__main__":
    main()
