"""Stage 01: load, align, and crop MUSE cubes.

Notebook-to-module mapping:

- ``01_load_align_crop.ipynb`` cells 4-6 -> wavelength helpers, centering,
  spatial alignment, crop/regrid loop.
- cells 9-11 -> FITS, CSV, and QC writers.

B1a preserves the historical DATA products. B1b extends the product with STAT
when available and records shifts, crop metadata, and a coarse covariance factor.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy import units as u
from astropy.io import fits
from scipy.ndimage import shift as ndi_shift

from ..config import load_run_config
from ..io import write_json
from ..paths import RunPaths


@dataclass(frozen=True)
class Stage01Product:
    cubes: np.ndarray | list[np.ndarray]
    wavelengths: np.ndarray | list[np.ndarray]
    stat: np.ndarray | list[np.ndarray] | None
    qc: dict


def stage01_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage01")
    return {
        "paths": paths,
        "stage01_cube_fits": paths.stage_dir / "stage01_cropped_cube_stack.fits",
        "stage01_wave_npy": paths.stage_dir / "stage01_wavelengths.npy",
        "stage01_centers_csv": paths.table_dir / "stage01_centers.csv",
        "stage01_shifts_csv": paths.table_dir / "stage01_shifts_spatial.csv",
        "stage01_qc_json": paths.stage_dir / "stage01_qc.json",
        "stage01_cropbounds_csv": paths.table_dir / "stage01_crop_bounds.csv",
        "stage01_initial_cropbounds_csv": paths.table_dir / "stage01_initial_crop_bounds.csv",
        "plot_dir": plot_dir,
    }


def build_wavelength_axis(header_cube, header_primary, nz: int) -> np.ndarray:
    crval3 = header_cube.get("CRVAL3", None)
    crpix3 = header_cube.get("CRPIX3", None)
    cdelt3 = header_cube.get("CDELT3", header_cube.get("CD3_3", None))
    cunit3 = header_cube.get("CUNIT3", None)

    if (crval3 is not None) and (crpix3 is not None) and (cdelt3 is not None):
        if cunit3 is not None:
            try:
                unit = u.Unit(str(cunit3))
                crval_a = (crval3 * unit).to(u.AA).value
                cdelt_a = (cdelt3 * unit).to(u.AA).value
                k = np.arange(nz, dtype=np.float64) + 1.0
                return crval_a + (k - float(crpix3)) * cdelt_a
            except Exception:
                pass
        k = np.arange(nz, dtype=np.float64) + 1.0
        return float(crval3) + (k - float(crpix3)) * float(cdelt3)

    wmin = header_primary.get("WAVELMIN", None)
    wmax = header_primary.get("WAVELMAX", None)
    if (wmin is None) or (wmax is None):
        raise RuntimeError("Cannot reconstruct wavelength axis.")
    return np.linspace(float(wmin), float(wmax), nz, dtype=np.float64)


def sanitize_wavelength_indices(wavelengths_A: np.ndarray, nz: int):
    wave = np.asarray(wavelengths_A, dtype=np.float64).ravel()
    n_use = min(int(nz), wave.size)
    if int(nz) != wave.size:
        warnings.warn(
            f"[sanitize] nz cube={int(nz)} != len(wave)={wave.size}. Trimming to {n_use}."
        )
    channel_index = np.arange(n_use, dtype=int)
    wave = wave[:n_use]
    if wave.size < 2:
        raise RuntimeError("Wavelength axis has fewer than 2 channels.")
    good = np.isfinite(wave)
    channel_index = channel_index[good]
    wave = wave[good]
    if np.nanmedian(np.diff(wave)) < 0:
        wave = wave[::-1].copy()
        channel_index = channel_index[::-1].copy()
    keep = np.ones(wave.size, dtype=bool)
    keep[1:] = np.diff(wave) > 0
    if not np.all(keep):
        warnings.warn(f"[sanitize] Removing {np.sum(~keep)} repeated/non-increasing channels.")
        wave = wave[keep]
        channel_index = channel_index[keep]
    if wave.size < 2:
        raise RuntimeError("Invalid wavelength axis after sanitization.")
    return channel_index, wave


def apply_wavelength_indices(cube_zyx: np.ndarray, channel_index: np.ndarray) -> np.ndarray:
    channel_index = np.asarray(channel_index, dtype=int)
    if channel_index.size == cube_zyx.shape[0] and np.array_equal(
        channel_index, np.arange(cube_zyx.shape[0])
    ):
        return np.asarray(cube_zyx, dtype=np.float32)
    return np.asarray(cube_zyx[channel_index], dtype=np.float32)


def inspect_cube_spectral_grid(fpath: str | Path, data_ext: int):
    with fits.open(fpath, memmap=True) as hdul:
        data = hdul[data_ext].data
        hdr_cube = hdul[data_ext].header
        hdr_prim = hdul[0].header
        nz = int(data.shape[0])
        wave = build_wavelength_axis(hdr_cube, hdr_prim, nz)
        _, wave = sanitize_wavelength_indices(wave, nz)
        return {
            "file": str(fpath),
            "nz_native": nz,
            "nz_wave": int(wave.size),
            "wmin": float(wave[0]),
            "wmax": float(wave[-1]),
            "dw": float(np.nanmedian(np.diff(wave))),
        }


def build_common_wavelength_grid(cube_files, data_ext: int):
    infos = [inspect_cube_spectral_grid(fp, data_ext) for fp in cube_files]
    dw_common = float(np.nanmedian([d["dw"] for d in infos]))
    wmin_common = float(max(d["wmin"] for d in infos))
    wmax_common = float(min(d["wmax"] for d in infos))
    if not np.isfinite(dw_common) or dw_common <= 0:
        raise RuntimeError("Invalid common wavelength step.")
    if wmax_common <= wmin_common:
        raise RuntimeError("No common spectral overlap among cubes.")
    nz_common = int(np.floor((wmax_common - wmin_common) / dw_common)) + 1
    wave_common = wmin_common + np.arange(nz_common, dtype=np.float64) * dw_common
    if wave_common.size < 10:
        raise RuntimeError("Common wavelength grid is unexpectedly short.")
    return wave_common, infos


def make_white_light_image(cube_zyx, wavelengths_A, drop_min, drop_max, collapse="median"):
    keep = ~((wavelengths_A >= drop_min) & (wavelengths_A <= drop_max))
    if np.sum(keep) < 5:
        raise RuntimeError("Too few channels remain after excluding Na-LGS region.")
    if collapse == "median":
        return np.nanmedian(cube_zyx[keep], axis=0)
    if collapse == "mean":
        return np.nanmean(cube_zyx[keep], axis=0)
    raise ValueError("collapse must be 'median' or 'mean'")


def find_centroid_peak(image2d: np.ndarray, box_half_size: int = 4):
    img = np.asarray(image2d, dtype=np.float64)
    if not np.isfinite(img).any():
        raise RuntimeError("White-light image is all-NaN.")
    y_peak, x_peak = np.unravel_index(np.nanargmax(img), img.shape)
    y1 = max(0, y_peak - box_half_size)
    y2 = min(img.shape[0], y_peak + box_half_size + 1)
    x1 = max(0, x_peak - box_half_size)
    x2 = min(img.shape[1], x_peak + box_half_size + 1)
    cut = np.nan_to_num(img[y1:y2, x1:x2], nan=0.0)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    denom = np.sum(cut)
    if denom <= 0:
        return float(y_peak), float(x_peak)
    return float(np.sum(yy * cut) / denom), float(np.sum(xx * cut) / denom)


def find_centroid_maoppy_moffat(image2d, y_init=None, x_init=None, stamp_half_size=10, max_nfev=120):
    if (y_init is None) or (x_init is None):
        y_init, x_init = find_centroid_peak(image2d, box_half_size=4)
    y_init_i = int(np.round(y_init))
    x_init_i = int(np.round(x_init))
    y1 = max(0, y_init_i - stamp_half_size)
    y2 = min(image2d.shape[0], y_init_i + stamp_half_size + 1)
    x1 = max(0, x_init_i - stamp_half_size)
    x2 = min(image2d.shape[1], x_init_i + stamp_half_size + 1)
    stamp = np.nan_to_num(image2d[y1:y2, x1:x2], nan=0.0).astype(np.float64)
    bkg = np.median(stamp)
    stamp = stamp - bkg
    stamp[stamp < 0] = 0.0
    cy_global = y1 + (stamp.shape[0] - 1) / 2.0
    cx_global = x1 + (stamp.shape[1] - 1) / 2.0
    try:
        import maoppy
        from maoppy.psfmodel import Moffat

        model = Moffat(npix=stamp.shape, norm=np.inf)
        x0_model = np.array([2.0, 2.0, 0.0, 2.5], dtype=float)
        res = maoppy.psffit(
            stamp,
            model,
            x0_model,
            dxdy=(0.0, 0.0),
            flux_bck=(True, True),
            positive_bck=True,
            max_nfev=max_nfev,
        )
        return cy_global + float(res.dxdy[0]), cx_global + float(res.dxdy[1]), False
    except Exception:
        return float(y_init), float(x_init), True


def get_cube_center(white_image, cfg):
    method = cfg["centering_method"]
    if method == "peak":
        y, x = find_centroid_peak(white_image, box_half_size=4)
        return y, x, False
    if method == "maoppy":
        y_guess, x_guess = find_centroid_peak(white_image, box_half_size=4)
        return find_centroid_maoppy_moffat(
            white_image,
            y_init=y_guess,
            x_init=x_guess,
            stamp_half_size=int(cfg["maoppy_stamp_half_size"]),
            max_nfev=int(cfg["maoppy_max_nfev"]),
        )
    raise ValueError(f"Unknown centering_method: {method}")


def crop_cube_around_center(cube_zyx: np.ndarray, y0: float, x0: float, npix: int):
    nz, ny, nx = cube_zyx.shape
    half = npix // 2
    yc = int(np.round(y0))
    xc = int(np.round(x0))
    y1 = max(0, min(yc - half, ny - npix))
    x1 = max(0, min(xc - half, nx - npix))
    y2 = y1 + npix
    x2 = x1 + npix
    cube_crop = cube_zyx[:, y1:y2, x1:x2]
    if cube_crop.shape[1:] != (npix, npix):
        raise RuntimeError(f"Crop shape mismatch: got {cube_crop.shape}")
    return cube_crop, (y1, y2, x1, x2)


def regrid_cube_spectral_axis(cube_zyx, wave_in_A, wave_out_A) -> np.ndarray:
    cube = np.asarray(cube_zyx, dtype=np.float32)
    wave_in = np.asarray(wave_in_A, dtype=np.float64)
    wave_out = np.asarray(wave_out_A, dtype=np.float64)
    nz_in, ny, nx = cube.shape
    flat_in = cube.reshape(nz_in, ny * nx)
    flat_out = np.full((wave_out.size, ny * nx), np.nan, dtype=np.float32)
    for j in range(flat_in.shape[1]):
        spec = flat_in[:, j]
        good = np.isfinite(spec)
        if np.count_nonzero(good) < 2:
            continue
        x = wave_in[good]
        y = spec[good].astype(np.float64)
        uniq, idx = np.unique(x, return_index=True)
        if uniq.size < 2:
            continue
        flat_out[:, j] = np.interp(wave_out, uniq, y[idx], left=np.nan, right=np.nan).astype(np.float32)
    return flat_out.reshape(wave_out.size, ny, nx)


def apply_spatial_alignment(cube_big, shift_y, shift_x, mode):
    if mode == "none":
        return cube_big.copy()
    if mode == "integer":
        sy = int(np.round(shift_y))
        sx = int(np.round(shift_x))
        return np.roll(np.roll(cube_big, sy, axis=1), sx, axis=2)
    if mode == "subpixel":
        cube_big0 = np.nan_to_num(cube_big, nan=0.0).astype(np.float32, copy=False)
        valid = np.isfinite(cube_big).astype(np.float32)
        cube_s = ndi_shift(cube_big0, shift=(0.0, shift_y, shift_x), order=3, mode="constant", cval=0.0, prefilter=True)
        valid_s = ndi_shift(valid, shift=(0.0, shift_y, shift_x), order=0, mode="constant", cval=0.0, prefilter=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = cube_s / np.maximum(valid_s, 1e-6)
        out[valid_s < 0.5] = np.nan
        return out.astype(np.float32)
    raise ValueError(f"Unknown spatial_shift_mode: {mode}")


def propagate_stat_alignment(stat_big, shift_y, shift_x, mode):
    """Propagate variance through spatial shift.

    For subpixel shifts this uses an exact bilinear weight-squared propagation.
    DATA equivalence remains tied to the historical cubic shift; the QC records
    this STAT kernel explicitly.
    """

    if stat_big is None:
        return None, "unavailable"
    stat = np.asarray(stat_big, dtype=np.float32)
    if mode == "none":
        return stat.copy(), "none"
    if mode == "integer":
        sy = int(np.round(shift_y))
        sx = int(np.round(shift_x))
        return np.roll(np.roll(stat, sy, axis=1), sx, axis=2), "integer_roll"
    if mode == "subpixel":
        return bilinear_shift_variance(stat, shift_y, shift_x).astype(np.float32), "bilinear_kernel_squared"
    raise ValueError(f"Unknown spatial_shift_mode: {mode}")


def bilinear_shift_variance(var_zyx: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(var_zyx, dtype=np.float64), nan=0.0)
    nz, ny, nx = arr.shape
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    src_y = yy - float(shift_y)
    src_x = xx - float(shift_x)
    y0 = np.floor(src_y).astype(int)
    x0 = np.floor(src_x).astype(int)
    fy = src_y - y0
    fx = src_x - x0
    out = np.zeros_like(arr, dtype=np.float64)
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            yi = y0 + dy
            xi = x0 + dx
            valid = (yi >= 0) & (yi < ny) & (xi >= 0) & (xi < nx)
            weight2 = (wy * wx) ** 2
            for z in range(nz):
                plane = out[z]
                source = arr[z]
                plane[valid] += weight2[valid] * source[yi[valid], xi[valid]]
    out[out == 0] = np.nan
    return out


def native_wavelengths_match(wave_list, atol=1e-8):
    if len(wave_list) == 0:
        return True
    ref = np.asarray(wave_list[0], dtype=np.float64)
    for w in wave_list[1:]:
        w = np.asarray(w, dtype=np.float64)
        if w.shape != ref.shape or not np.allclose(w, ref, rtol=0.0, atol=atol):
            return False
    return True


def covariance_factor_box3(cube_zyx: np.ndarray, *, n_samples: int = 25) -> float:
    cube = np.asarray(cube_zyx, dtype=np.float64)
    if cube.ndim == 4:
        cube = np.nanmean(cube, axis=0)
    _, ny, nx = cube.shape
    centers = []
    for y in np.linspace(2, ny - 3, int(np.sqrt(n_samples)), dtype=int):
        for x in np.linspace(2, nx - 3, int(np.sqrt(n_samples)), dtype=int):
            centers.append((int(y), int(x)))
    if len(centers) < 2:
        return np.nan
    spectra = []
    spaxels = []
    for y, x in centers:
        box = cube[:, y - 1 : y + 2, x - 1 : x + 2]
        spectra.append(np.nansum(box, axis=(1, 2)))
        spaxels.append(cube[:, y, x])
    aperture_var = np.nanmedian(np.nanvar(np.asarray(spectra), axis=0))
    spaxel_var = np.nanmedian(np.nanvar(np.asarray(spaxels), axis=0))
    if spaxel_var <= 0 or not np.isfinite(spaxel_var):
        return np.nan
    return float(aperture_var / (9.0 * spaxel_var))


def stage01_config_from_run(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    run_config = load_run_config(
        run_id,
        project_root=project_root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    cfg = dict(run_config.config)
    if overrides:
        cfg.update(overrides)
    cfg.setdefault("entry_point", "cropped_cube")
    cfg.setdefault("stage01_profile", "maoppy_refined")
    cfg.setdefault("centering_method", "maoppy")
    cfg.setdefault("maoppy_stamp_half_size", 10)
    cfg.setdefault("maoppy_max_nfev", 120)
    cfg.setdefault("spatial_shift_mode", "subpixel")
    cfg.setdefault("spectral_grid_mode", "common_grid")
    cfg.setdefault("stage01_initial_crop_npix", 80)
    cfg.setdefault("stage01_initial_crop_edge_guard_pix", 12)
    cfg.setdefault("drop_wave_min_A", 5780.0)
    cfg.setdefault("drop_wave_max_A", 6050.0)
    cfg["run_id"] = run_config.run_id
    cfg["project_root"] = str(run_config.paths.project_root)
    return cfg


def _stat_ext_index(hdul, cfg):
    stat_ext = cfg.get("stat_ext", "STAT")
    if stat_ext in (None, "", "null"):
        return None
    if isinstance(stat_ext, int) and stat_ext < len(hdul):
        return stat_ext
    if isinstance(stat_ext, str) and stat_ext in hdul:
        return stat_ext
    if "STAT" in hdul:
        return "STAT"
    if len(hdul) > 2 and getattr(hdul[2].data, "shape", None) == getattr(hdul[int(cfg["data_ext"])].data, "shape", None):
        return 2
    return None


def compute_stage01_products(config) -> Stage01Product:
    cfg = dict(config)
    drop_wmin = float(cfg["drop_wave_min_A"])
    drop_wmax = float(cfg["drop_wave_max_A"])
    crop_npix = int(cfg["crop_npix"])
    initial_cfg = cfg.get("stage01_initial_crop_npix", 80)
    initial_npix = None if initial_cfg in (None, 0, "") else int(initial_cfg)
    initial_guard = int(cfg.get("stage01_initial_crop_edge_guard_pix", 12))
    data_ext = int(cfg["data_ext"])
    center_method = str(cfg["centering_method"])
    spatial_mode = str(cfg["spatial_shift_mode"])
    spectral_mode = str(cfg["spectral_grid_mode"])

    if spectral_mode == "common_grid":
        wavelengths_common, spectral_grid_info = build_common_wavelength_grid(cfg["cube_files"], data_ext)
    else:
        wavelengths_common = None
        spectral_grid_info = [inspect_cube_spectral_grid(fp, data_ext) for fp in cfg["cube_files"]]

    cropped_native = []
    stat_cropped_native = []
    native_wave_list = []
    crop_bounds = []
    initial_bounds = []
    initial_used = []
    star_centers = []
    fallback_used = []
    shifts = []
    native_wave_info_used = []
    inputs = []
    ref_center = None
    interp_kernels = set()

    for i, fpath_raw in enumerate(cfg["cube_files"]):
        fpath = Path(fpath_raw)
        with fits.open(fpath, memmap=True) as hdul:
            data_full = hdul[data_ext].data
            stat_idx = _stat_ext_index(hdul, cfg)
            stat_full = hdul[stat_idx].data if stat_idx is not None else None
            hdr_cube = hdul[data_ext].header
            hdr_prim = hdul[0].header
            nz_native, ny, nx = data_full.shape
            wave_native = build_wavelength_axis(hdr_cube, hdr_prim, nz_native)
            channel_index, wave_native = sanitize_wavelength_indices(wave_native, nz_native)
            native_wave_info_used.append(
                {
                    "file": fpath.name,
                    "nz": int(wave_native.size),
                    "wmin": float(wave_native[0]),
                    "wmax": float(wave_native[-1]),
                    "dw": float(np.nanmedian(np.diff(wave_native))),
                }
            )
            inputs.append({"file": str(fpath), "data_ext": data_ext, "stat_ext": None if stat_idx is None else str(stat_idx)})

            def estimate_center(bounds):
                y1_c, y2_c, x1_c, x2_c = bounds
                cube_center_raw = data_full[:, y1_c:y2_c, x1_c:x2_c]
                cube_center = apply_wavelength_indices(cube_center_raw, channel_index)
                white = make_white_light_image(cube_center, wave_native, drop_wmin, drop_wmax)
                y_local, x_local, fallback = get_cube_center(white, cfg)
                return white, float(y_local + y1_c), float(x_local + x1_c), float(y_local), float(x_local), fallback

            center_bounds = (0, int(ny), 0, int(nx))
            used_initial = False
            if initial_npix is not None and initial_npix < min(ny, nx):
                _, center_bounds = crop_cube_around_center(
                    data_full,
                    y0=(ny - 1) / 2.0,
                    x0=(nx - 1) / 2.0,
                    npix=initial_npix,
                )
                used_initial = True
            white, y0, x0, y_local, x_local, fallback = estimate_center(center_bounds)
            if used_initial and initial_guard > 0:
                y1_c, y2_c, x1_c, x2_c = center_bounds
                near_edge = (
                    y_local < initial_guard
                    or x_local < initial_guard
                    or y_local > (y2_c - y1_c - 1 - initial_guard)
                    or x_local > (x2_c - x1_c - 1 - initial_guard)
                )
                if near_edge:
                    center_bounds = (0, int(ny), 0, int(nx))
                    used_initial = False
                    white, y0, x0, y_local, x_local, fallback = estimate_center(center_bounds)

            initial_bounds.append(center_bounds)
            initial_used.append(bool(used_initial))
            star_centers.append((y0, x0))
            fallback_used.append(bool(fallback))

            if ref_center is None:
                ref_center = (y0, x0)
            shift_y = float(ref_center[0] - y0)
            shift_x = float(ref_center[1] - x0)
            shifts.append((shift_y, shift_x))

            if spatial_mode == "subpixel":
                pad = int(np.ceil(max(abs(shift_y), abs(shift_x)))) + 4
            elif spatial_mode == "integer":
                pad = int(np.ceil(max(abs(np.round(shift_y)), abs(np.round(shift_x))))) + 2
            else:
                pad = 2
            npix_big = int(crop_npix + 2 * pad)
            cube_big_raw, bounds_big = crop_cube_around_center(data_full, ref_center[0], ref_center[1], npix_big)
            cube_big = apply_wavelength_indices(cube_big_raw, channel_index)
            cube_big_aligned = apply_spatial_alignment(cube_big, shift_y, shift_x, spatial_mode)
            cube_crop_native = cube_big_aligned[:, pad : pad + crop_npix, pad : pad + crop_npix]

            if stat_full is not None:
                stat_big_raw, _ = crop_cube_around_center(stat_full, ref_center[0], ref_center[1], npix_big)
                stat_big = apply_wavelength_indices(stat_big_raw, channel_index)
                stat_aligned, kernel = propagate_stat_alignment(stat_big, shift_y, shift_x, spatial_mode)
                interp_kernels.add(kernel)
                stat_crop = stat_aligned[:, pad : pad + crop_npix, pad : pad + crop_npix]
                stat_cropped_native.append(stat_crop.astype(np.float32, copy=False))
            else:
                stat_cropped_native.append(None)
                interp_kernels.add("unavailable")

            if cube_crop_native.shape[1:] != (crop_npix, crop_npix):
                raise RuntimeError(f"Final crop shape mismatch in {fpath.name}: {cube_crop_native.shape}")
            y1_big, _, x1_big, _ = bounds_big
            bounds_final = (y1_big + pad, y1_big + pad + crop_npix, x1_big + pad, x1_big + pad + crop_npix)
            cropped_native.append(cube_crop_native.astype(np.float32, copy=False))
            native_wave_list.append(wave_native.astype(np.float64, copy=True))
            crop_bounds.append(bounds_final)

    native_match = native_wavelengths_match(native_wave_list, atol=1e-8)
    if spectral_mode == "common_grid":
        cropped = []
        stats = []
        for cube_crop, stat_crop, wave_native in zip(cropped_native, stat_cropped_native, native_wave_list):
            if cube_crop.shape[0] == wavelengths_common.size and np.allclose(wave_native, wavelengths_common, rtol=0.0, atol=1e-8):
                cropped.append(cube_crop.astype(np.float32, copy=False))
                stats.append(None if stat_crop is None else stat_crop.astype(np.float32, copy=False))
            else:
                cropped.append(regrid_cube_spectral_axis(cube_crop, wave_native, wavelengths_common))
                stats.append(None if stat_crop is None else regrid_cube_spectral_axis(stat_crop, wave_native, wavelengths_common))
        cropped_cubes = np.stack(cropped, axis=0).astype(np.float32, copy=False)
        stat_cubes = None if any(item is None for item in stats) else np.stack(stats, axis=0).astype(np.float32, copy=False)
        wavelengths = wavelengths_common.astype(np.float64, copy=True)
        storage_mode = "stack_common_grid"
    elif spectral_mode == "native_if_possible" and native_match:
        cropped_cubes = np.stack(cropped_native, axis=0).astype(np.float32, copy=False)
        stat_cubes = None if any(item is None for item in stat_cropped_native) else np.stack(stat_cropped_native, axis=0).astype(np.float32, copy=False)
        wavelengths = native_wave_list[0].astype(np.float64, copy=True)
        storage_mode = "stack_native_grid"
    else:
        cropped_cubes = cropped_native
        stat_cubes = None if any(item is None for item in stat_cropped_native) else stat_cropped_native
        wavelengths = native_wave_list
        storage_mode = "list_per_cube_native"

    cube_shape = [int(x) for x in cropped_cubes.shape] if isinstance(cropped_cubes, np.ndarray) else [[int(x) for x in c.shape] for c in cropped_cubes]
    finite_fraction = (
        [float(np.isfinite(cropped_cubes[i]).mean()) for i in range(cropped_cubes.shape[0])]
        if isinstance(cropped_cubes, np.ndarray)
        else [float(np.isfinite(cube_i).mean()) for cube_i in cropped_cubes]
    )
    if isinstance(cropped_cubes, np.ndarray):
        wavelength_summary = {
            "type": "shared_axis",
            "nz": int(len(wavelengths)),
            "wmin_A": float(wavelengths[0]),
            "wmax_A": float(wavelengths[-1]),
            "dw_A": float(np.nanmedian(np.diff(wavelengths))),
        }
    else:
        wavelength_summary = {
            "type": "per_cube_axes",
            "per_cube": [
                {
                    "cube_index": int(i),
                    "nz": int(len(wave_i)),
                    "wmin_A": float(wave_i[0]),
                    "wmax_A": float(wave_i[-1]),
                    "dw_A": float(np.nanmedian(np.diff(wave_i))),
                }
                for i, wave_i in enumerate(wavelengths)
            ],
        }

    stat_propagated = stat_cubes is not None
    cov_factor = covariance_factor_box3(cropped_cubes) if isinstance(cropped_cubes, np.ndarray) else np.nan
    qc = {
        "run_id": cfg["run_id"],
        "target_name": cfg.get("target_name", cfg["run_id"]),
        "stage": "stage01",
        "entry_point": cfg.get("entry_point", "cropped_cube"),
        "inputs": inputs,
        "profile": cfg["stage01_profile"],
        "centering_method": center_method,
        "spatial_shift_mode": spatial_mode,
        "spectral_grid_mode": spectral_mode,
        "stage01_storage_mode": storage_mode,
        "native_grids_match": bool(native_match),
        "n_cubes": int(len(cfg["cube_files"])),
        "crop_npix": int(crop_npix),
        "crop": {"npix": int(crop_npix), "center_yx": [float(ref_center[0]), float(ref_center[1])], "center_source": "measured"},
        "stage01_initial_crop_npix": None if initial_npix is None else int(initial_npix),
        "stage01_initial_crop_edge_guard_pix": int(initial_guard),
        "cube_shape": cube_shape,
        "ref_center": [float(ref_center[0]), float(ref_center[1])],
        "wavelength_info": wavelength_summary,
        "native_wave_info_used": native_wave_info_used,
        "star_centers": [
            {"cube_index": int(i), "y_center": float(y0), "x_center": float(x0)}
            for i, (y0, x0) in enumerate(star_centers)
        ],
        "spatial_shifts": [
            {"cube_index": int(i), "shift_y": float(sy), "shift_x": float(sx)}
            for i, (sy, sx) in enumerate(shifts)
        ],
        "shifts": [
            {
                "cube": int(i),
                "dy": float(sy),
                "dx": float(sx),
                "method": str(cfg["stage01_profile"]),
                "fallback_used": bool(fallback_used[i]),
            }
            for i, (sy, sx) in enumerate(shifts)
        ],
        "crop_bounds_per_cube": [
            {"cube_index": int(i), "y1": int(b[0]), "y2": int(b[1]), "x1": int(b[2]), "x2": int(b[3])}
            for i, b in enumerate(crop_bounds)
        ],
        "initial_crop_bounds_per_cube": [
            {
                "cube_index": int(i),
                "used_initial_crop": bool(initial_used[i]),
                "y1": int(b[0]),
                "y2": int(b[1]),
                "x1": int(b[2]),
                "x2": int(b[3]),
            }
            for i, b in enumerate(initial_bounds)
        ],
        "finite_fraction_per_cube": finite_fraction,
        "stat": {
            "propagated": bool(stat_propagated),
            "interp_kernel": ",".join(sorted(interp_kernels)),
            "covariance_factor_box3": None if not np.isfinite(cov_factor) else float(cov_factor),
        },
        "equivalence": {"baseline": "", "max_abs_diff": None, "verdict": ""},
        "open_issues": [],
    }
    return Stage01Product(cropped_cubes, wavelengths, stat_cubes, qc)


def write_stage01_products(product: Stage01Product, config, paths_dict):
    cfg = config
    cropped_cubes = product.cubes
    wavelengths = product.wavelengths
    stat_cubes = product.stat
    qc = product.qc
    crop_npix = int(cfg["crop_npix"])
    storage_mode = qc["stage01_storage_mode"]

    hdr = fits.Header()
    hdr["RUNID"] = str(cfg["run_id"])
    hdr["TARGET"] = str(cfg.get("target_name", cfg["run_id"]))
    hdr["STAGE"] = "stage01"
    hdr["MODE"] = str(storage_mode)
    hdr["NCUBES"] = int(len(cfg["cube_files"]))
    hdr["CROPPIX"] = int(cfg["crop_npix"])
    initial_npix = qc.get("stage01_initial_crop_npix")
    hdr["ICROP"] = -1 if initial_npix is None else int(initial_npix)
    hdr["CENTER"] = str(cfg["centering_method"])
    hdr["SPATIAL"] = str(cfg["spatial_shift_mode"])
    hdr["SPECMD"] = str(cfg["spectral_grid_mode"])
    hdus = [fits.PrimaryHDU(header=hdr)]

    if isinstance(cropped_cubes, np.ndarray):
        hdr["NZ"] = int(cropped_cubes.shape[1])
        hdr["NY"] = int(cropped_cubes.shape[2])
        hdr["NX"] = int(cropped_cubes.shape[3])
        hdr["WMIN"] = float(wavelengths[0])
        hdr["WMAX"] = float(wavelengths[-1])
        hdr["DW"] = float(np.nanmedian(np.diff(wavelengths)))
        hdus.append(fits.ImageHDU(data=cropped_cubes.astype(np.float32), name="CUBES"))
        hdus.append(fits.ImageHDU(data=np.asarray(wavelengths, dtype=np.float64), name="WAVELENGTH"))
        if isinstance(stat_cubes, np.ndarray):
            hdus.append(fits.ImageHDU(data=stat_cubes.astype(np.float32), name="STAT"))
    else:
        hdr["NZ"] = -1
        hdr["NY"] = int(crop_npix)
        hdr["NX"] = int(crop_npix)
        hdr["NOTE"] = "Per-cube native spectral axes"
        for i, (cube_i, wave_i, fp) in enumerate(zip(cropped_cubes, wavelengths, cfg["cube_files"])):
            cube_hdr = fits.Header()
            cube_hdr["CUBEIDX"] = int(i)
            cube_hdr["FNAME"] = Path(fp).name
            cube_hdr["NZ"] = int(cube_i.shape[0])
            cube_hdr["NY"] = int(cube_i.shape[1])
            cube_hdr["NX"] = int(cube_i.shape[2])
            cube_hdr["WMIN"] = float(wave_i[0])
            cube_hdr["WMAX"] = float(wave_i[-1])
            cube_hdr["DW"] = float(np.nanmedian(np.diff(wave_i)))
            hdus.append(fits.ImageHDU(data=np.asarray(cube_i, dtype=np.float32), header=cube_hdr, name=f"CUBE{i:02d}"))
            hdus.append(fits.ImageHDU(data=np.asarray(wave_i, dtype=np.float64), name=f"WAVE{i:02d}"))

    fits.HDUList(hdus).writeto(paths_dict["stage01_cube_fits"], overwrite=True)
    if isinstance(wavelengths, np.ndarray):
        np.save(paths_dict["stage01_wave_npy"], wavelengths)

    _write_stage01_csvs(cfg, qc, paths_dict)
    write_json(paths_dict["stage01_qc_json"], qc)


def _write_stage01_csvs(cfg, qc, paths_dict):
    with open(paths_dict["stage01_centers_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cube_index", "file", "y_center", "x_center"])
        for row, fp in zip(qc["star_centers"], cfg["cube_files"]):
            writer.writerow([row["cube_index"], Path(fp).name, row["y_center"], row["x_center"]])
    with open(paths_dict["stage01_shifts_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cube_index", "file", "shift_y", "shift_x"])
        for row, fp in zip(qc["spatial_shifts"], cfg["cube_files"]):
            writer.writerow([row["cube_index"], Path(fp).name, row["shift_y"], row["shift_x"]])
    with open(paths_dict["stage01_cropbounds_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cube_index", "file", "y1", "y2", "x1", "x2"])
        for row, fp in zip(qc["crop_bounds_per_cube"], cfg["cube_files"]):
            writer.writerow([row["cube_index"], Path(fp).name, row["y1"], row["y2"], row["x1"], row["x2"]])
    with open(paths_dict["stage01_initial_cropbounds_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cube_index", "file", "used_initial_crop", "y1", "y2", "x1", "x2"])
        for row, fp in zip(qc["initial_crop_bounds_per_cube"], cfg["cube_files"]):
            writer.writerow([row["cube_index"], Path(fp).name, row["used_initial_crop"], row["y1"], row["y2"], row["x1"], row["x2"]])


def run_stage01(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage01_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths_dict = stage01_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    paths_dict["paths"].ensure_base_dirs()
    paths_dict["plot_dir"].mkdir(parents=True, exist_ok=True)
    product = compute_stage01_products(cfg)
    write_stage01_products(product, cfg, paths_dict)
    return {"config": cfg, "paths": paths_dict, "qc": product.qc}


__all__ = [
    "Stage01Product",
    "bilinear_shift_variance",
    "build_common_wavelength_grid",
    "build_wavelength_axis",
    "compute_stage01_products",
    "crop_cube_around_center",
    "find_centroid_peak",
    "propagate_stat_alignment",
    "run_stage01",
    "stage01_config_from_run",
    "stage01_paths",
]
