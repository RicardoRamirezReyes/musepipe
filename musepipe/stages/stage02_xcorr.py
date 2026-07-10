"""Stage 02: stripe spectral-shift correction by cross-correlation.

This module is the thin driver around the pure stripe helpers. It preserves the
historical Stage02 product names while adding B2 QC for stripe amplitudes and
STAT propagation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from ..io import write_json
from ..paths import RunPaths
from ..stripes import (
    _build_stripe_geometry,
    _parse_stripe_geometry_config,
    _parse_xcorr_group_keys,
    _stripe_group_sort_value,
    _validate_manual_range_list_for_stage02,
    _xcorr_shift_pixels,
    apply_stripe_spectral_shifts,
    stripe_channel_amplitudes,
    stripe_metric_exclusion_mask,
    stripe_metric_summary,
    xcorr_shift_map,
)


@dataclass(frozen=True)
class Stage02Product:
    cubes: np.ndarray
    wavelengths: np.ndarray
    stat: np.ndarray | None
    shifts: np.ndarray
    shiftmaps: np.ndarray
    shift_rows: list[dict]
    metric_rows: list[dict]
    qc: dict


def stage02_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage02")
    return {
        "paths": paths,
        "stage01_cube_fits": paths.stage_dir / "stage01_cropped_cube_stack.fits",
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "stage02_shifts_npy": paths.stage_dir / "stage02_xcorr_shifts.npy",
        "stage02_shiftmaps_fits": paths.stage_dir / "stage02_shiftmaps.fits",
        "stage02_qc_json": paths.stage_dir / "stage02_qc.json",
        "stage02_shift_csv": paths.table_dir / "stage02_xcorr_shift_summary.csv",
        "stage02_metric_csv": paths.table_dir / "stage02_stripe_metric.csv",
        "stage02_metric_plot": plot_dir / "stage02_stripe_metric.png",
        "plot_dir": plot_dir,
    }


def stage02_config_from_run(
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
    cfg.setdefault(
        "stage01_cube_fits",
        str(run_config.paths.stage_dir / "stage01_cropped_cube_stack.fits"),
    )
    cfg.setdefault("xcorr_wmin_A", 4650.0)
    cfg.setdefault("xcorr_wmax_A", 9300.0)
    cfg.setdefault("xcorr_max_lag_ch", 6)
    cfg.setdefault("xcorr_nstripes", 3)
    cfg.setdefault("xcorr_apply_mode", "subpixel")
    cfg.setdefault("xcorr_shift_estimator", "stripe_median_spectrum")
    cfg.setdefault("xcorr_ref_mode", "auto_by_orientation")
    cfg.setdefault("xcorr_metric_excluded_windows_A", [[5780.0, 6050.0]])
    cfg.setdefault("xcorr_metric_edge_trim_pix", 0)
    cfg.setdefault("xcorr_metric_spectral_edge_channels", 0)
    cfg.setdefault("xcorr_metric_normalization", "median_abs")
    cfg.setdefault("stage02_save_metric_plot", bool(cfg.get("save_intermediate_plots", False)))
    return cfg


def _load_stage01_stack(path):
    with fits.open(path, memmap=True) as hdul:
        if "CUBES" not in hdul or "WAVELENGTH" not in hdul:
            raise RuntimeError("Stage02 expects CUBES and WAVELENGTH HDUs from Stage01.")
        cubes = hdul["CUBES"].data.astype(np.float32)
        wavelengths = hdul["WAVELENGTH"].data.astype(np.float64)
        stat = hdul["STAT"].data.astype(np.float32) if "STAT" in hdul else None
    if cubes.ndim != 4:
        raise RuntimeError(f"Stage02 expects stacked cubes (N,nz,ny,nx), got {cubes.shape}.")
    if wavelengths.ndim != 1 or wavelengths.size != cubes.shape[1]:
        raise RuntimeError("Stage02 wavelength axis does not match CUBES spectral dimension.")
    if stat is not None and stat.shape != cubes.shape:
        raise RuntimeError("STAT shape does not match CUBES shape.")
    return cubes, wavelengths, stat


def _excluded_windows_from_config(cfg):
    windows = cfg.get("xcorr_metric_excluded_windows_A")
    if windows is None:
        windows = []
    return [[float(a), float(b)] for a, b in windows]


def _fit_mask_from_config(wavelengths_A, cfg):
    wave = np.asarray(wavelengths_A, dtype=np.float64)
    mask = (
        np.isfinite(wave)
        & (wave >= float(cfg.get("xcorr_wmin_A", 4650.0)))
        & (wave <= float(cfg.get("xcorr_wmax_A", 9300.0)))
    )
    if bool(cfg.get("xcorr_exclude_nalgs", True)):
        wmin = float(cfg.get("nalgs_wmin_A", cfg.get("drop_wave_min_A", 5780.0)))
        wmax = float(cfg.get("nalgs_wmax_A", cfg.get("drop_wave_max_A", 6050.0)))
        mask &= ~((wave >= wmin) & (wave <= wmax))
    if bool(cfg.get("xcorr_exclude_strong_line", False)) and cfg.get("halpha_A") is not None:
        half = float(cfg.get("line_exclude_halfwidth_A", 2.0))
        center = float(cfg["halpha_A"])
        mask &= ~((wave >= center - half) & (wave <= center + half))
    if np.count_nonzero(mask) < 10:
        raise RuntimeError("Too few wavelength channels remain for Stage02 xcorr.")
    return mask


def _manual_ranges_per_cube(cfg, n_cubes):
    ranges = cfg.get("xcorr_stripe_ranges_per_cube", None)
    if ranges is None:
        return [None] * n_cubes
    ranges = list(ranges)
    if len(ranges) != n_cubes:
        raise ValueError(f"xcorr_stripe_ranges_per_cube must have length {n_cubes}.")
    out = []
    nstripes = int(cfg["xcorr_nstripes"])
    for i, item in enumerate(ranges):
        out.append(
            _validate_manual_range_list_for_stage02(
                item,
                nstripes,
                f"xcorr_stripe_ranges_per_cube[{i}]",
            )
            if item is not None
            else None
        )
    return out


def _median_spec_from_stripe(cube_zyx, a1, a2, axis, stripe_mask=None):
    if axis == "y":
        block = cube_zyx[:, int(a1):int(a2), :]
        reduce_axis = (1, 2)
    elif axis == "x":
        block = cube_zyx[:, :, int(a1):int(a2)]
        reduce_axis = (1, 2)
    elif axis == "angle":
        if stripe_mask is None:
            raise ValueError("stripe_mask is required for angled stripes.")
        block = cube_zyx[:, stripe_mask]
        reduce_axis = 1
    else:
        raise ValueError(f"Invalid axis: {axis}")
    with np.errstate(all="ignore"):
        return np.nanmedian(block, axis=reduce_axis).astype(np.float64)


def _stripe_geometry_for_cube(ny, nx, cfg, orientations, angles, ranges_all, cube_index):
    return _build_stripe_geometry(
        ny,
        nx,
        int(cfg["xcorr_nstripes"]),
        orientations[cube_index],
        manual_stripe_ranges=ranges_all[cube_index],
        stripe_angle_deg=angles[cube_index],
    )


def _choose_ref_index_map(group_keys, cfg):
    unique_groups = sorted(set(group_keys), key=_stripe_group_sort_value)
    fixed = cfg.get("xcorr_fixed_ref_index", cfg.get("xcorr_ref_index", None))
    ref_map = {}
    for group in unique_groups:
        idxs = [i for i, key in enumerate(group_keys) if key == group]
        ref = int(fixed) if fixed is not None else int(idxs[0])
        if ref not in idxs:
            raise RuntimeError(f"Reference {ref} is not inside stripe group {group}: {idxs}")
        ref_map[group] = ref
    return ref_map


def _measure_shifts_for_group(
    cubes,
    fit_mask,
    cfg,
    orientations,
    angles,
    ranges_all,
    group_indices,
    ref_index,
):
    _, nz, ny, nx = cubes.shape
    max_lag = int(cfg.get("xcorr_max_lag_ch", 6))
    estimator = str(cfg.get("xcorr_shift_estimator", "stripe_median_spectrum"))
    if estimator not in {"stripe_median_spectrum", "spaxel_map_region_median"}:
        raise ValueError(
            "xcorr_shift_estimator must be 'stripe_median_spectrum' or "
            f"'spaxel_map_region_median'; got {estimator!r}."
        )
    ref_ranges, ref_axis, _, ref_masks = _stripe_geometry_for_cube(
        ny,
        nx,
        cfg,
        orientations,
        angles,
        ranges_all,
        ref_index,
    )
    nstripes = len(ref_ranges)
    ref_specs = []
    for s, (a1, a2) in enumerate(ref_ranges):
        ref_specs.append(
            _median_spec_from_stripe(
                cubes[ref_index],
                a1,
                a2,
                ref_axis,
                stripe_mask=ref_masks[s],
            )[fit_mask]
        )

    shifts = {}
    for cube_index in group_indices:
        if cube_index == ref_index:
            shifts[cube_index] = np.zeros(nstripes, dtype=np.float32)
            continue
        ranges_i, axis_i, _, masks_i = _stripe_geometry_for_cube(
            ny,
            nx,
            cfg,
            orientations,
            angles,
            ranges_all,
            cube_index,
        )
        if len(ranges_i) != nstripes:
            raise RuntimeError(
                f"Cube {cube_index}: number of stripes {len(ranges_i)} != reference {nstripes}."
            )
        dz = np.full(nstripes, np.nan, dtype=np.float32)
        for s, (a1, a2) in enumerate(ranges_i):
            if estimator == "spaxel_map_region_median":
                dz_map = xcorr_shift_map(
                    cubes[cube_index][fit_mask],
                    ref_spec=ref_specs[s],
                    max_lag=max_lag,
                )
                vals = dz_map[masks_i[s]]
                dz[s] = np.nanmedian(vals) if np.isfinite(vals).any() else np.nan
            else:
                tgt = _median_spec_from_stripe(
                    cubes[cube_index],
                    a1,
                    a2,
                    axis_i,
                    stripe_mask=masks_i[s],
                )[fit_mask]
                dz[s] = _xcorr_shift_pixels(ref_specs[s], tgt, max_lag=max_lag)
        shifts[cube_index] = dz
    return shifts


def _shiftmaps_from_shifts(shifts, ny, nx, cfg, orientations, angles, ranges_all):
    n_cubes, nstripes = shifts.shape
    maps = np.full((n_cubes, ny, nx), np.nan, dtype=np.float32)
    for i in range(n_cubes):
        ranges_i, axis_i, _, masks_i = _stripe_geometry_for_cube(
            ny,
            nx,
            cfg,
            orientations,
            angles,
            ranges_all,
            i,
        )
        if len(ranges_i) != nstripes:
            raise RuntimeError(f"Cube {i}: stripe geometry changed while making shift maps.")
        for s, (a1, a2) in enumerate(ranges_i):
            dz = shifts[i, s]
            if axis_i == "y":
                maps[i, int(a1):int(a2), :] = dz
            elif axis_i == "x":
                maps[i, :, int(a1):int(a2)] = dz
            else:
                maps[i, masks_i[s]] = dz
    return maps


def _central_source_mask(cfg, ny, nx):
    path = cfg.get("stage02_metric_source_mask")
    if path:
        path = Path(path)
        if path.suffix == ".npy":
            mask = np.load(path).astype(bool)
        else:
            with fits.open(path, memmap=True) as hdul:
                mask = hdul[0].data.astype(bool)
        if mask.shape != (ny, nx):
            raise ValueError(f"stage02_metric_source_mask shape {mask.shape} != {(ny, nx)}")
        return mask
    if not bool(cfg.get("apply_central_mask", False)):
        return None
    radius = float(cfg.get("central_mask_radius_px", 0.0))
    if radius <= 0:
        return None
    y0 = float(cfg.get("stage02_metric_center_y", (ny - 1) / 2.0))
    x0 = float(cfg.get("stage02_metric_center_x", (nx - 1) / 2.0))
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    return (yy - y0) ** 2 + (xx - x0) ** 2 <= radius**2


def _metric_rows_and_summary(cubes_pre, cubes_post, wavelengths, cfg, orientations, angles, ranges_all):
    n_cubes, _, ny, nx = cubes_pre.shape
    windows = _excluded_windows_from_config(cfg)
    edge_channels = int(cfg.get("xcorr_metric_spectral_edge_channels", 0))
    excluded = stripe_metric_exclusion_mask(
        wavelengths,
        excluded_windows_A=windows,
        spectral_edge_channels=edge_channels,
    )
    source_mask = _central_source_mask(cfg, ny, nx)
    amp_pre_all = []
    amp_post_all = []
    rows = []
    for i in range(n_cubes):
        amp_pre = stripe_channel_amplitudes(
            cubes_pre[i],
            nstripes=int(cfg["xcorr_nstripes"]),
            stripe_orientation=orientations[i],
            manual_stripe_ranges=ranges_all[i],
            stripe_angle_deg=angles[i],
            source_mask=source_mask,
            edge_trim_pix=int(cfg.get("xcorr_metric_edge_trim_pix", 0)),
            normalization=cfg.get("xcorr_metric_normalization", "median_abs"),
        )
        amp_post = stripe_channel_amplitudes(
            cubes_post[i],
            nstripes=int(cfg["xcorr_nstripes"]),
            stripe_orientation=orientations[i],
            manual_stripe_ranges=ranges_all[i],
            stripe_angle_deg=angles[i],
            source_mask=source_mask,
            edge_trim_pix=int(cfg.get("xcorr_metric_edge_trim_pix", 0)),
            normalization=cfg.get("xcorr_metric_normalization", "median_abs"),
        )
        amp_pre_all.append(amp_pre)
        amp_post_all.append(amp_post)
        for ch, wave in enumerate(wavelengths):
            rows.append(
                {
                    "cube_index": int(i),
                    "channel": int(ch),
                    "wavelength_A": float(wave),
                    "amplitude_pre": None if not np.isfinite(amp_pre[ch]) else float(amp_pre[ch]),
                    "amplitude_post": None if not np.isfinite(amp_post[ch]) else float(amp_post[ch]),
                    "excluded_from_summary": bool(excluded[ch]),
                }
            )

    with np.errstate(all="ignore"):
        amp_pre_global = np.nanmedian(np.vstack(amp_pre_all), axis=0)
        amp_post_global = np.nanmedian(np.vstack(amp_post_all), axis=0)
    summary = stripe_metric_summary(
        amp_pre_global,
        amp_post_global,
        wavelengths,
        excluded_windows_A=windows,
        spectral_edge_channels=edge_channels,
    )
    return rows, summary


def _shift_rows(shifts, cfg, orientations, angles, ranges_all, group_keys, ref_map, ny, nx):
    rows = []
    for i in range(shifts.shape[0]):
        group = group_keys[i]
        ranges_i = ranges_all[i]
        if ranges_i is None:
            ranges_i, _, _, _ = _build_stripe_geometry(
                ny,
                nx,
                int(cfg["xcorr_nstripes"]),
                orientations[i],
                stripe_angle_deg=angles[i],
            )
        for s in range(shifts.shape[1]):
            r0, r1 = ranges_i[s] if ranges_i is not None else (np.nan, np.nan)
            rows.append(
                {
                    "cube_index": int(i),
                    "stripe_index": int(s),
                    "dz_ch": None if not np.isfinite(shifts[i, s]) else float(shifts[i, s]),
                    "stripe_orientation": orientations[i],
                    "stripe_angle_deg": float(angles[i]),
                    "stripe_range_start": float(r0),
                    "stripe_range_stop": float(r1),
                    "stripe_group_key": group,
                    "ref_index": int(ref_map[group]),
                }
            )
    return rows


def _equivalence_payload(cubes, baseline_fits):
    if not baseline_fits:
        return {"baseline": "", "max_abs_diff": None, "verdict": "not_checked"}
    baseline_fits = Path(baseline_fits)
    if not baseline_fits.exists():
        raise RuntimeError(f"Stage02 equivalence baseline does not exist: {baseline_fits}")
    with fits.open(baseline_fits, memmap=True) as hdul:
        baseline = hdul["CUBES"].data.astype(np.float32)
    if baseline.shape != cubes.shape:
        return {
            "baseline": str(baseline_fits),
            "max_abs_diff": None,
            "verdict": "shape_mismatch",
        }
    diff = np.abs(cubes.astype(np.float64) - baseline.astype(np.float64))
    max_abs = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
    if max_abs == 0.0:
        verdict = "identical"
    elif np.allclose(cubes, baseline, rtol=1e-6, atol=1e-6, equal_nan=True):
        verdict = "allclose"
    else:
        verdict = "mismatch"
    return {"baseline": str(baseline_fits), "max_abs_diff": max_abs, "verdict": verdict}


def compute_stage02_products(config) -> Stage02Product:
    cfg = dict(config)
    cubes_in, wavelengths, stat_in = _load_stage01_stack(cfg["stage01_cube_fits"])
    n_cubes, nz, ny, nx = cubes_in.shape
    orientations, angles = _parse_stripe_geometry_config(
        cfg.get("xcorr_stripe_orientation", "vertical"),
        cfg.get("xcorr_stripe_angle_deg", None),
        n_cubes,
    )
    group_keys = _parse_xcorr_group_keys(
        angles,
        cfg.get("xcorr_stripe_group_key", None),
        n_cubes=n_cubes,
    )
    ranges_all = _manual_ranges_per_cube(cfg, n_cubes)
    fit_mask = _fit_mask_from_config(wavelengths, cfg)
    ref_map = _choose_ref_index_map(group_keys, cfg)

    shifts = np.full((n_cubes, int(cfg["xcorr_nstripes"])), np.nan, dtype=np.float32)
    for group in sorted(set(group_keys), key=_stripe_group_sort_value):
        group_indices = [i for i, key in enumerate(group_keys) if key == group]
        group_shifts = _measure_shifts_for_group(
            cubes_in,
            fit_mask,
            cfg,
            orientations,
            angles,
            ranges_all,
            group_indices,
            ref_map[group],
        )
        for i, dz in group_shifts.items():
            shifts[i] = dz

    apply_mode = str(cfg.get("xcorr_apply_mode", "subpixel"))
    cubes_out = np.empty_like(cubes_in, dtype=np.float32)
    stat_out = None if stat_in is None else np.empty_like(stat_in, dtype=np.float32)
    stat_kernels = set()
    for i in range(n_cubes):
        cubes_out[i], _ = apply_stripe_spectral_shifts(
            cubes_in[i],
            shifts[i],
            nstripes=int(cfg["xcorr_nstripes"]),
            stripe_orientation=orientations[i],
            manual_stripe_ranges=ranges_all[i],
            stripe_angle_deg=angles[i],
            apply_mode=apply_mode,
            is_variance=False,
        )
        if stat_in is not None:
            stat_out[i], kernel = apply_stripe_spectral_shifts(
                stat_in[i],
                shifts[i],
                nstripes=int(cfg["xcorr_nstripes"]),
                stripe_orientation=orientations[i],
                manual_stripe_ranges=ranges_all[i],
                stripe_angle_deg=angles[i],
                apply_mode=apply_mode,
                is_variance=True,
            )
            stat_kernels.add(kernel)

    shiftmaps = _shiftmaps_from_shifts(shifts, ny, nx, cfg, orientations, angles, ranges_all)
    metric_rows, metric_summary = _metric_rows_and_summary(
        cubes_in,
        cubes_out,
        wavelengths,
        cfg,
        orientations,
        angles,
        ranges_all,
    )
    shift_rows = _shift_rows(shifts, cfg, orientations, angles, ranges_all, group_keys, ref_map, ny, nx)
    finite_fraction = [float(np.isfinite(cubes_out[i]).mean()) for i in range(n_cubes)]
    qc = {
        "run_id": cfg["run_id"],
        "target_name": cfg.get("target_name", cfg["run_id"]),
        "stage": "stage02",
        "input_stage01_mode": "stack_common_grid",
        "ref_mode": cfg.get("xcorr_ref_mode", "auto_by_orientation"),
        "apply_mode": apply_mode,
        "fit_range_A": [float(cfg["xcorr_wmin_A"]), float(cfg["xcorr_wmax_A"])],
        "trim_edge_pix": int(cfg.get("xcorr_metric_edge_trim_pix", 0)),
        "max_lag_ch": int(cfg.get("xcorr_max_lag_ch", 6)),
        "shift_estimator": str(cfg.get("xcorr_shift_estimator", "stripe_median_spectrum")),
        "stripe_orientation": list(orientations),
        "stripe_angle_deg": [float(x) for x in angles],
        "stripe_group_key": list(group_keys),
        "stripe_ranges_per_cube": ranges_all,
        "cube_shape": [int(x) for x in cubes_out.shape],
        "mean_shift_per_cube_ch": [float(np.nanmean(shifts[i])) for i in range(n_cubes)],
        "std_shift_per_cube_ch": [float(np.nanstd(shifts[i])) for i in range(n_cubes)],
        "finite_fraction_per_cube": finite_fraction,
        "ref_index": None,
        "ref_index_map": {str(k): int(v) for k, v in ref_map.items()},
        "ref_scores": [None for _ in range(n_cubes)],
        "stripe_metric": metric_summary,
        "stat": {
            "present": bool(stat_in is not None),
            "shift_applied": bool(stat_in is not None),
            "interp_kernel": ",".join(sorted(stat_kernels)) if stat_in is not None else "unavailable",
        },
        "equivalence": _equivalence_payload(
            cubes_out,
            cfg.get("stage02_equivalence_baseline_fits"),
        ),
    }
    return Stage02Product(
        cubes=cubes_out,
        wavelengths=wavelengths,
        stat=stat_out,
        shifts=shifts,
        shiftmaps=shiftmaps,
        shift_rows=shift_rows,
        metric_rows=metric_rows,
        qc=qc,
    )


def write_stage02_products(product: Stage02Product, config, paths_dict):
    cfg = dict(config)
    overwrite = bool(cfg.get("overwrite_existing_stage_files", False))
    for key in (
        "stage02_cube_fits",
        "stage02_shifts_npy",
        "stage02_shiftmaps_fits",
        "stage02_qc_json",
        "stage02_shift_csv",
        "stage02_metric_csv",
    ):
        path = Path(paths_dict[key])
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing Stage02 product: {path}")

    hdr = fits.Header()
    hdr["RUNID"] = str(cfg["run_id"])
    hdr["TARGET"] = str(cfg.get("target_name", cfg["run_id"]))
    hdr["STAGE"] = "stage02"
    hdr["NCUBES"] = int(product.cubes.shape[0])
    hdr["NZ"] = int(product.cubes.shape[1])
    hdr["NY"] = int(product.cubes.shape[2])
    hdr["NX"] = int(product.cubes.shape[3])
    hdus = [
        fits.PrimaryHDU(header=hdr),
        fits.ImageHDU(data=product.cubes.astype(np.float32), name="CUBES"),
        fits.ImageHDU(data=product.wavelengths.astype(np.float64), name="WAVELENGTH"),
    ]
    if product.stat is not None:
        hdus.append(fits.ImageHDU(data=product.stat.astype(np.float32), name="STAT"))
    fits.HDUList(hdus).writeto(paths_dict["stage02_cube_fits"], overwrite=overwrite)

    fits.HDUList(
        [
            fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(data=product.shiftmaps.astype(np.float32), name="SHIFTMAPS"),
        ]
    ).writeto(paths_dict["stage02_shiftmaps_fits"], overwrite=overwrite)
    np.save(paths_dict["stage02_shifts_npy"], product.shifts.astype(np.float32))

    _write_csv(paths_dict["stage02_shift_csv"], product.shift_rows)
    _write_csv(paths_dict["stage02_metric_csv"], product.metric_rows)
    product.qc["stripe_metric"]["table"] = str(
        Path("tables") / Path(paths_dict["stage02_metric_csv"]).name
    )
    if bool(cfg.get("stage02_save_metric_plot", cfg.get("save_intermediate_plots", False))):
        _write_metric_plot(paths_dict["stage02_metric_plot"], product.metric_rows, product.qc)
        product.qc["stripe_metric"]["plot"] = str(
            Path("plots") / "stage02" / Path(paths_dict["stage02_metric_plot"]).name
        )
    write_json(paths_dict["stage02_qc_json"], product.qc)


def _write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_metric_plot(path, rows, qc):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    by_channel = {}
    for row in rows:
        ch = int(row["channel"])
        by_channel.setdefault(
            ch,
            {
                "wave": float(row["wavelength_A"]),
                "pre": [],
                "post": [],
                "excluded": bool(row["excluded_from_summary"]),
            },
        )
        if row["amplitude_pre"] is not None:
            by_channel[ch]["pre"].append(float(row["amplitude_pre"]))
        if row["amplitude_post"] is not None:
            by_channel[ch]["post"].append(float(row["amplitude_post"]))

    channels = np.array(sorted(by_channel), dtype=int)
    wave = np.array([by_channel[ch]["wave"] for ch in channels], dtype=float)
    pre = np.array(
        [np.nanmedian(by_channel[ch]["pre"]) if by_channel[ch]["pre"] else np.nan for ch in channels],
        dtype=float,
    )
    post = np.array(
        [np.nanmedian(by_channel[ch]["post"]) if by_channel[ch]["post"] else np.nan for ch in channels],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wave, pre, lw=1.0, label="pre")
    ax.plot(wave, post, lw=1.0, label="post")
    for wmin, wmax in qc.get("stripe_metric", {}).get("mask_excluded_windows_A", []):
        ax.axvspan(float(wmin), float(wmax), color="0.8", alpha=0.35)
    dirty = set(qc.get("stripe_metric", {}).get("dirty_channels", []))
    if dirty:
        dirty_idx = np.array([i for i, ch in enumerate(channels) if int(ch) in dirty], dtype=int)
        ax.scatter(wave[dirty_idx], post[dirty_idx], s=12, color="tab:red", zorder=3, label="dirty")
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("Stripe amplitude")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_stage02(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage02_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths_dict = stage02_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    paths_dict["paths"].ensure_base_dirs()
    paths_dict["plot_dir"].mkdir(parents=True, exist_ok=True)
    product = compute_stage02_products(cfg)
    write_stage02_products(product, cfg, paths_dict)
    return {"config": cfg, "paths": paths_dict, "qc": product.qc}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage02 xcorr stripe correction.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--baseline-fits", default=None)
    args = parser.parse_args(argv)
    overrides = {}
    if args.overwrite:
        overrides["overwrite_existing_stage_files"] = True
    if args.baseline_fits:
        overrides["stage02_equivalence_baseline_fits"] = args.baseline_fits
    result = run_stage02(
        args.run_id,
        project_root=args.project_root,
        overrides=overrides,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage02_qc_json"])


__all__ = [
    "Stage02Product",
    "compute_stage02_products",
    "run_stage02",
    "stage02_config_from_run",
    "stage02_paths",
    "write_stage02_products",
]


if __name__ == "__main__":
    main()
