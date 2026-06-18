from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from musepipe.apertures import same_radius_control_positions
from musepipe.config import load_run_config
from musepipe.io import get_cube_data, read_json, read_wavelengths_and_masks, write_csv, write_json
from musepipe.localfit import local_surface_spectra_fast
from musepipe.paths import RunPaths
from musepipe.spectral import continuum_running_median
from musepipe.stats import finite_percentile, robust_sigma_axis0


def default_apertures():
    return [
        {"name": "pixel", "kind": "pixel"},
        {"name": "box3_sum", "kind": "box", "size": 3},
        {"name": "circle_r1p5_sum", "kind": "circle", "radius_px": 1.5},
        {"name": "circle_r2p0_sum", "kind": "circle", "radius_px": 2.0},
        {"name": "gauss_sig1p0_r3_sum", "kind": "gaussian", "sigma_px": 1.0, "radius_px": 3.0},
    ]


STAGE08_DEFAULTS = {
    "default_aperture": "box3_sum",
    "continuum_window_A": 80.0,
    "control_apertures": 8,
    "control_exclude_angle_deg": 25.0,
    "control_margin_px": 4,
}


CONFIG = {
    "run_id": "ROXs12b",
    "object_xy": (72, 152),
    "star_yx": (85, 85),
    **STAGE08_DEFAULTS,
    "apertures": default_apertures(),
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


def _spatial_shape_from_shape(shape):
    if shape is None:
        return None
    vals = list(shape)
    if vals and isinstance(vals[0], (list, tuple)):
        return _spatial_shape_from_shape(vals[0])
    if len(vals) < 2:
        return None
    return (int(vals[-2]), int(vals[-1]))


def _resolve_spatial_shape(cfg, qc04b):
    for key in ("output_shape", "input_shape", "cube_shape", "spatial_shape"):
        shape = _spatial_shape_from_shape(qc04b.get(key))
        if shape is not None:
            return shape
    crop_npix = cfg.get("crop_npix")
    if crop_npix is not None:
        npix = int(crop_npix)
        return (npix, npix)
    return None


def _resolve_object_yx(cfg, qc04b, fallback_xy=None):
    for source in (cfg, qc04b):
        for key in ("stage08_object_yx", "object_yx", "target_yx", "stage04b_target_yx"):
            yx = _as_yx(source.get(key))
            if yx is not None:
                return yx
        for key in ("stage08_object_xy", "object_xy", "target_xy", "stage04b_target_xy", "science_object_xy"):
            xy = _as_xy(source.get(key))
            if xy is not None:
                return (xy[1], xy[0])

    target_object = qc04b.get("target_object")
    detected_peaks = qc04b.get("detected_peaks")
    if target_object is not None and isinstance(detected_peaks, dict):
        yx = _as_yx(detected_peaks.get(str(target_object)))
        if yx is not None:
            return yx

    xy = _as_xy(fallback_xy)
    if xy is not None:
        return (xy[1], xy[0])
    raise RuntimeError("Could not resolve Stage 8 object position from config or stage04b_qc.json.")


def _resolve_star_yx(cfg, qc04b):
    for source in (cfg, qc04b):
        for key in ("stage08_star_yx", "star_yx", "reference_geometry_center_yx"):
            yx = _as_yx(source.get(key))
            if yx is not None:
                return yx
        for key in ("stage08_star_xy", "star_xy", "reference_geometry_center_xy"):
            xy = _as_xy(source.get(key))
            if xy is not None:
                return (xy[1], xy[0])

    shape = _resolve_spatial_shape(cfg, qc04b)
    if shape is None:
        raise RuntimeError("Could not resolve Stage 8 star center; set stage08_star_yx in config.")
    ny, nx = shape
    return (int(ny // 2), int(nx // 2))


def stage08_config_from_run(
    run_id=None,
    *,
    project_root=None,
    overrides=None,
    allow_run_id_mismatch=False,
):
    """Build a Stage 8 config from the active run config and Stage04b QC."""

    run_config = load_run_config(
        run_id,
        project_root=project_root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    cfg = run_config.config
    paths = run_config.paths
    qc04b = read_json(paths.stage_dir / "stage04b_qc.json") if (paths.stage_dir / "stage04b_qc.json").exists() else {}

    object_yx = _resolve_object_yx(cfg, qc04b)
    star_yx = _resolve_star_yx(cfg, qc04b)

    config = {
        "run_id": run_config.run_id,
        "project_root": str(paths.project_root),
        "object_xy": (int(object_yx[1]), int(object_yx[0])),
        "star_yx": (int(star_yx[0]), int(star_yx[1])),
        **STAGE08_DEFAULTS,
        "apertures": default_apertures(),
    }

    for key, default_value in STAGE08_DEFAULTS.items():
        config[key] = cfg.get(f"stage08_{key}", cfg.get(key, default_value))
    if "stage08_apertures" in cfg:
        config["apertures"] = cfg["stage08_apertures"]

    if overrides:
        config.update(overrides)
    return config


def median_stack_spectrum(spectra):
    stack = np.vstack(spectra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(stack, axis=0)


def make_products_for_aperture(waves, good_mask, object_spec, control_specs, continuum_window_A):
    control_median = median_stack_spectrum(control_specs)
    control_sigma = robust_sigma_axis0(np.vstack(control_specs))
    flux_native = object_spec - control_median
    continuum = continuum_running_median(waves, flux_native, good_mask, window_A=continuum_window_A)
    flux_contsub = flux_native - continuum
    snr_like = flux_contsub / control_sigma
    snr_like[(~np.asarray(good_mask, dtype=bool)) | (~np.isfinite(control_sigma)) | (control_sigma <= 0)] = np.nan
    return {
        "object_local_residual": object_spec,
        "control_median_local_residual": control_median,
        "control_sigma": control_sigma,
        "flux_native": flux_native,
        "continuum": continuum,
        "flux_continuum_sub": flux_contsub,
        "snr_like": snr_like,
    }


def table_rows(waves, good_mask, bad_mask, products):
    rows = []
    for i, wave in enumerate(waves):
        rows.append(
            {
                "wavelength_A": float(wave),
                "good_wave_mask": int(bool(good_mask[i])),
                "bad_wave_mask": int(bool(bad_mask[i])),
                "object_local_residual": float(products["object_local_residual"][i]),
                "control_median_local_residual": float(products["control_median_local_residual"][i]),
                "flux_native": float(products["flux_native"][i]),
                "continuum": float(products["continuum"][i]),
                "flux_continuum_sub": float(products["flux_continuum_sub"][i]),
                "control_sigma": float(products["control_sigma"][i]),
                "snr_like": float(products["snr_like"][i]),
            }
        )
    return rows


def all_aperture_rows(waves, good_mask, bad_mask, products_by_aperture):
    names = list(products_by_aperture.keys())
    rows = []
    for i, wave in enumerate(waves):
        row = {
            "wavelength_A": float(wave),
            "good_wave_mask": int(bool(good_mask[i])),
            "bad_wave_mask": int(bool(bad_mask[i])),
        }
        for name in names:
            products = products_by_aperture[name]
            row[f"{name}_flux_native"] = float(products["flux_native"][i])
            row[f"{name}_flux_continuum_sub"] = float(products["flux_continuum_sub"][i])
            row[f"{name}_control_sigma"] = float(products["control_sigma"][i])
            row[f"{name}_snr_like"] = float(products["snr_like"][i])
        rows.append(row)
    return rows


def save_default_fits(path, waves, good_mask, bad_mask, products):
    cols = [
        fits.Column(name="wavelength_A", format="D", array=np.asarray(waves, dtype=np.float64)),
        fits.Column(name="good_wave_mask", format="L", array=np.asarray(good_mask, dtype=bool)),
        fits.Column(name="bad_wave_mask", format="L", array=np.asarray(bad_mask, dtype=bool)),
        fits.Column(name="object_local_residual", format="D", array=products["object_local_residual"]),
        fits.Column(name="control_median_local_residual", format="D", array=products["control_median_local_residual"]),
        fits.Column(name="flux_native", format="D", array=products["flux_native"]),
        fits.Column(name="continuum", format="D", array=products["continuum"]),
        fits.Column(name="flux_continuum_sub", format="D", array=products["flux_continuum_sub"]),
        fits.Column(name="control_sigma", format="D", array=products["control_sigma"]),
        fits.Column(name="snr_like", format="D", array=products["snr_like"]),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="SPECTRUM")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def shade_bad_ranges(ax, bad_ranges):
    for lo, hi in bad_ranges:
        ax.axvspan(float(lo), float(hi), color="0.85", alpha=0.4, lw=0)


def save_plots(plot_dir, waves, good_mask, bad_ranges, default_products, products_by_aperture, default_aperture):
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    shade_bad_ranges(ax, bad_ranges)
    ax.plot(waves, default_products["flux_native"], lw=0.9, color="0.45", label="object - median controls")
    ax.plot(waves, default_products["continuum"], lw=1.0, color="tab:orange", label="running continuum")
    ax.plot(waves, default_products["flux_continuum_sub"], lw=0.9, color="tab:blue", label="continuum-subtracted")
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("Native units")
    ax.set_title(f"Stage 8 full spectrum ({default_aperture})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    overview_png = plot_dir / "stage08_full_spectrum_default.png"
    fig.savefig(overview_png, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    shade_bad_ranges(ax, bad_ranges)
    ax.plot(waves, default_products["snr_like"], lw=0.8, color="tab:red")
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    ax.axhline(3, color="0.3", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(-3, color="0.3", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("SNR-like")
    ax.set_title("Stage 8 continuum-subtracted spectrum / control scatter")
    fig.tight_layout()
    snr_png = plot_dir / "stage08_full_spectrum_snr_like.png"
    fig.savefig(snr_png, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    shade_bad_ranges(ax, bad_ranges)
    for name, products in products_by_aperture.items():
        ax.plot(waves, products["flux_continuum_sub"], lw=0.75, label=name)
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("Continuum-subtracted native units")
    ax.set_title("Stage 8 aperture comparison")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    apertures_png = plot_dir / "stage08_aperture_comparison_overview.png"
    fig.savefig(apertures_png, bbox_inches="tight")
    plt.close(fig)
    return [overview_png, snr_png, apertures_png]


def run_stage08(config=None, show_plots=False, *, project_root=None, allow_run_id_mismatch=False):
    if config is None:
        config = stage08_config_from_run(
            project_root=project_root,
            allow_run_id_mismatch=allow_run_id_mismatch,
        )
    else:
        config = dict(config)
    run_id = config["run_id"]
    root = Path(project_root or config.get("project_root") or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    stage_dir = paths.stage_dir
    table_dir = paths.table_dir
    plot_dir = paths.plot_stage_dir("stage08_full_spectrum")
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    stage04b_stack_fits = stage_dir / "stage04b_local_surface_cube_stack.fits"
    stage04b_qc_json = stage_dir / "stage04b_qc.json"
    input_cube_fits = stage_dir / "cube_input_local_object.fits"
    residual_cube_fits = stage_dir / "cube_residual_local_object.fits"
    good_mask_npy = stage_dir / "stage04b_good_wavelength_mask.npy"
    bad_mask_npy = stage_dir / "stage04b_bad_wavelength_mask.npy"

    qc04b = read_json(stage04b_qc_json)
    if not input_cube_fits.exists():
        input_cube_fits = Path(qc04b["input_cube_fits"])

    local_model_kind = qc04b.get("local_model_kind", "plane")
    fit_radius_px = float(qc04b.get("fit_radius_px", 12.0))
    mask_radius_px = float(qc04b.get("mask_radius_px", 3.0))
    min_pixels = int(qc04b.get("local_fit_min_pixels", 30))
    bad_ranges = qc04b.get("bad_wavelength_ranges_A", [])

    waves, good_mask, bad_mask = read_wavelengths_and_masks(stage04b_stack_fits, good_mask_npy, bad_mask_npy)
    wave_indices = np.where(good_mask)[0]

    with fits.open(input_cube_fits, memmap=True) as hdul:
        cube = get_cube_data(hdul)
        nz, ny, nx = cube.shape
        if nz != waves.size:
            raise RuntimeError(f"Cube spectral axis {nz} != wavelength axis {waves.size}.")

        object_yx = _resolve_object_yx(config, qc04b, fallback_xy=config.get("object_xy"))
        object_yx = (int(object_yx[0]), int(object_yx[1]))
        star_yx = _resolve_star_yx(config, qc04b)
        star_yx = (int(star_yx[0]), int(star_yx[1]))
        control_yx = same_radius_control_positions(
            object_yx,
            star_yx,
            ny,
            nx,
            n_positions=config["control_apertures"],
            exclude_angle_deg=config["control_exclude_angle_deg"],
            margin_px=config["control_margin_px"],
        )

        apertures = list(config["apertures"])
        print(f"Stage 8 run: {run_id}")
        print(f"Input cube: {input_cube_fits}")
        print(f"Cube shape: {cube.shape}")
        print(f"Object yx={object_yx}, xy={(object_yx[1], object_yx[0])}")
        print(f"Controls: {len(control_yx)}")
        print(f"Good wavelengths: {len(wave_indices)} / {waves.size}")
        print(f"Local model: {local_model_kind}, fit_radius={fit_radius_px}, mask_radius={mask_radius_px}")

        object_spectra, aperture_meta, object_fit_meta = local_surface_spectra_fast(
            cube,
            object_yx,
            apertures,
            wave_indices,
            fit_radius_px,
            mask_radius_px,
            local_model_kind,
            min_pixels,
        )

        control_spectra_by_aperture = {ap["name"]: [] for ap in apertures}
        control_fit_meta = []
        for pos in control_yx:
            spectra_i, _, fit_meta_i = local_surface_spectra_fast(
                cube,
                pos,
                apertures,
                wave_indices,
                fit_radius_px,
                mask_radius_px,
                local_model_kind,
                min_pixels,
            )
            control_fit_meta.append(fit_meta_i)
            for ap in apertures:
                control_spectra_by_aperture[ap["name"]].append(spectra_i[ap["name"]])

    products_by_aperture = {}
    for ap in config["apertures"]:
        name = ap["name"]
        products_by_aperture[name] = make_products_for_aperture(
            waves,
            good_mask,
            object_spectra[name],
            control_spectra_by_aperture[name],
            config["continuum_window_A"],
        )

    default_aperture = config["default_aperture"]
    default_products = products_by_aperture[default_aperture]

    model_csv = table_dir / "stage08_full_spectrum_model_input.csv"
    all_aperture_csv = table_dir / "stage08_full_spectrum_all_apertures.csv"
    model_fits = stage_dir / "stage08_full_spectrum_model_input.fits"
    all_npz = stage_dir / "stage08_full_spectrum_all_apertures.npz"
    qc_json = stage_dir / "stage08_full_spectrum_qc.json"

    default_fields = [
        "wavelength_A",
        "good_wave_mask",
        "bad_wave_mask",
        "object_local_residual",
        "control_median_local_residual",
        "flux_native",
        "continuum",
        "flux_continuum_sub",
        "control_sigma",
        "snr_like",
    ]
    write_csv(model_csv, table_rows(waves, good_mask, bad_mask, default_products), default_fields)

    all_fields = ["wavelength_A", "good_wave_mask", "bad_wave_mask"]
    for name in products_by_aperture:
        all_fields.extend(
            [
                f"{name}_flux_native",
                f"{name}_flux_continuum_sub",
                f"{name}_control_sigma",
                f"{name}_snr_like",
            ]
        )
    write_csv(all_aperture_csv, all_aperture_rows(waves, good_mask, bad_mask, products_by_aperture), all_fields)
    save_default_fits(model_fits, waves, good_mask, bad_mask, default_products)

    npz_payload = {
        "wavelength_A": waves,
        "good_wave_mask": good_mask,
        "bad_wave_mask": bad_mask,
        "control_yx": np.asarray(control_yx, dtype=np.int16),
        "object_yx": np.asarray(object_yx, dtype=np.int16),
    }
    for name, products in products_by_aperture.items():
        for key, values in products.items():
            npz_payload[f"{name}_{key}"] = values
        npz_payload[f"{name}_control_local_residuals"] = np.vstack(control_spectra_by_aperture[name])
    np.savez_compressed(all_npz, **npz_payload)

    plot_paths = save_plots(
        plot_dir,
        waves,
        good_mask,
        bad_ranges,
        default_products,
        products_by_aperture,
        default_aperture,
    )
    if show_plots:
        for path in plot_paths:
            print(path)

    snr = default_products["snr_like"]
    qc = {
        "run_id": run_id,
        "input_cube_fits": str(input_cube_fits),
        "residual_cube_fits_reference": str(residual_cube_fits),
        "stage04b_stack_fits": str(stage04b_stack_fits),
        "object_yx": list(map(int, object_yx)),
        "object_xy": [int(object_yx[1]), int(object_yx[0])],
        "star_yx": list(map(int, star_yx)),
        "control_yx": [list(map(int, p)) for p in control_yx],
        "default_aperture": default_aperture,
        "apertures": config["apertures"],
        "aperture_meta": aperture_meta,
        "local_model_kind": local_model_kind,
        "fit_radius_px": fit_radius_px,
        "mask_radius_px": mask_radius_px,
        "local_fit_min_pixels": min_pixels,
        "local_fit_note": "Fast fixed-mask local fit; sigma clipping is not rerun in Stage 8.",
        "continuum_window_A": float(config["continuum_window_A"]),
        "bad_wavelength_ranges_A": bad_ranges,
        "n_wavelengths": int(waves.size),
        "n_good_wavelengths": int(np.sum(good_mask)),
        "n_bad_wavelengths": int(np.sum(bad_mask)),
        "object_fit_meta": object_fit_meta,
        "control_fit_meta": control_fit_meta,
        "snr_like_percentiles": {
            "p01": finite_percentile(snr[good_mask], 1),
            "p05": finite_percentile(snr[good_mask], 5),
            "p50": finite_percentile(snr[good_mask], 50),
            "p95": finite_percentile(snr[good_mask], 95),
            "p99": finite_percentile(snr[good_mask], 99),
        },
        "outputs": {
            "model_csv": str(model_csv),
            "all_aperture_csv": str(all_aperture_csv),
            "model_fits": str(model_fits),
            "all_npz": str(all_npz),
            "plots": [str(p) for p in plot_paths],
        },
    }
    write_json(qc_json, qc)

    print(f"Saved: {model_csv}")
    print(f"Saved: {all_aperture_csv}")
    print(f"Saved: {model_fits}")
    print(f"Saved: {all_npz}")
    print(f"Saved: {qc_json}")
    return {
        "model_csv": model_csv,
        "all_aperture_csv": all_aperture_csv,
        "model_fits": model_fits,
        "all_npz": all_npz,
        "qc_json": qc_json,
        "plots": plot_paths,
        "qc": qc,
    }


if __name__ == "__main__":
    run_stage08(show_plots=True)
