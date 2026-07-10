"""Stage 08c: empirical false-alarm and look-elsewhere calibration."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits

from ..apertures import angular_separation_deg, same_radius_control_positions
from ..config import load_run_config
from ..io import get_cube_data, read_json, write_csv, write_json
from ..localfit import local_surface_spectra_fast
from ..paths import RunPaths
from ..spectral import continuum_running_median
from ..stats import robust_sigma_axis0


STAGE08C_DEFAULTS = {
    "n_control_positions": 36,
    "object_exclude_angle_deg": 25.0,
    "control_reference_exclude_angle_deg": 15.0,
    "control_margin_px": 4,
    "fap_levels": [0.10, 0.05, 0.01],
    "top_n_peaks": 10,
    "peak_min_separation_A": 5.0,
    "halpha_A": 6562.8,
    "halpha_half_width_A": 5.0,
    "search_wavelength_min_A": None,
    "search_wavelength_max_A": None,
}


def stage08c_paths(run_id, project_root=None):
    """Return Stage 08c input and output paths."""

    paths = RunPaths.from_project_root(run_id, project_root=project_root)
    plot_dir = paths.plot_stage_dir("stage08c_look_elsewhere")
    return {
        "stage08_npz": paths.stage_dir / "stage08_full_spectrum_all_apertures.npz",
        "stage08_qc": paths.stage_dir / "stage08_full_spectrum_qc.json",
        "null_maxima_csv": paths.table_dir / "stage08c_null_maxima.csv",
        "candidates_csv": paths.table_dir / "stage08c_object_candidates.csv",
        "thresholds_csv": paths.table_dir / "stage08c_global_thresholds.csv",
        "spectra_npz": paths.stage_dir / "stage08c_look_elsewhere_spectra.npz",
        "qc_json": paths.stage_dir / "stage08c_look_elsewhere_qc.json",
        "summary_plot": plot_dir / "stage08c_look_elsewhere_summary.png",
    }


def empirical_fap(observed, null_values):
    """Return the finite-sample corrected one-sided empirical false-alarm rate."""

    values = np.asarray(null_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(observed):
        return np.nan
    exceedances = int(np.sum(values >= float(observed)))
    return float((exceedances + 1) / (values.size + 1))


def empirical_threshold_rows(null_values, fap_levels):
    """Return order-statistic thresholds when the requested FAP is resolvable."""

    values = np.asarray(null_values, dtype=np.float64)
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        raise ValueError("At least one finite null maximum is required.")
    resolution = float(1.0 / (values.size + 1))
    rows = []
    for level in map(float, fap_levels):
        if not 0.0 < level < 1.0:
            raise ValueError(f"FAP levels must lie between zero and one: {level}")
        resolvable = bool(level >= resolution)
        threshold = None
        if resolvable:
            try:
                threshold = float(np.quantile(values, 1.0 - level, method="higher"))
            except TypeError:  # NumPy < 1.22 compatibility.
                threshold = float(np.quantile(values, 1.0 - level, interpolation="higher"))
        rows.append(
            {
                "fap_level": level,
                "resolvable": int(resolvable),
                "threshold_snr_like": threshold,
                "n_null_spectra": int(values.size),
                "minimum_resolvable_fap": resolution,
            }
        )
    return rows


def _position_angle_deg(position_yx, star_yx):
    y, x = map(float, position_yx)
    sy, sx = map(float, star_yx)
    return float(np.degrees(np.arctan2(y - sy, x - sx)) % 360.0)


def _snr_like_from_target(
    wavelengths,
    good_mask,
    target_spectrum,
    reference_spectra,
    continuum_window_A,
):
    references = np.asarray(reference_spectra, dtype=np.float64)
    if references.ndim != 2 or references.shape[0] < 3:
        raise ValueError("At least three reference spectra are required.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_reference = np.nanmedian(references, axis=0)
        sigma_reference = robust_sigma_axis0(references)
    flux_native = np.asarray(target_spectrum, dtype=np.float64) - median_reference
    continuum = continuum_running_median(
        wavelengths,
        flux_native,
        good_mask,
        window_A=float(continuum_window_A),
    )
    flux_continuum_sub = flux_native - continuum
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_like = flux_continuum_sub / sigma_reference
    invalid = (
        ~np.asarray(good_mask, dtype=bool)
        | ~np.isfinite(sigma_reference)
        | (sigma_reference <= 0)
    )
    snr_like[invalid] = np.nan
    return {
        "snr_like": snr_like,
        "flux_native": flux_native,
        "continuum": continuum,
        "flux_continuum_sub": flux_continuum_sub,
        "reference_median": median_reference,
        "reference_sigma": sigma_reference,
    }


def _search_mask(wavelengths, good_mask, config):
    mask = np.asarray(good_mask, dtype=bool).copy()
    wave_min = config.get("search_wavelength_min_A")
    wave_max = config.get("search_wavelength_max_A")
    if wave_min is not None:
        mask &= np.asarray(wavelengths) >= float(wave_min)
    if wave_max is not None:
        mask &= np.asarray(wavelengths) <= float(wave_max)
    if not np.any(mask):
        raise ValueError("Stage 08c search mask contains no wavelengths.")
    return mask


def _maximum_row(wavelengths, snr_like, search_mask):
    valid = np.asarray(search_mask, dtype=bool) & np.isfinite(snr_like)
    if not np.any(valid):
        return {"index": None, "wavelength_A": np.nan, "snr_like": np.nan}
    indices = np.where(valid)[0]
    index = int(indices[np.argmax(np.asarray(snr_like)[indices])])
    return {
        "index": index,
        "wavelength_A": float(wavelengths[index]),
        "snr_like": float(snr_like[index]),
    }


def top_spectral_peaks(
    wavelengths,
    snr_like,
    search_mask,
    *,
    n_peaks=10,
    min_separation_A=5.0,
):
    """Return greedily separated positive spectral maxima."""

    waves = np.asarray(wavelengths, dtype=np.float64)
    values = np.asarray(snr_like, dtype=np.float64)
    valid = np.asarray(search_mask, dtype=bool) & np.isfinite(values)
    order = np.where(valid)[0]
    order = order[np.argsort(values[order])[::-1]]
    selected = []
    for index in order:
        if any(abs(waves[index] - waves[other]) < float(min_separation_A) for other in selected):
            continue
        selected.append(int(index))
        if len(selected) >= int(n_peaks):
            break
    return selected


def compute_stage08c_products(
    wavelengths,
    good_mask,
    object_spectrum,
    control_spectra,
    control_positions_yx,
    star_yx,
    config,
):
    """Compute object and leave-neighborhood-out null search statistics."""

    waves = np.asarray(wavelengths, dtype=np.float64)
    good = np.asarray(good_mask, dtype=bool)
    controls = np.asarray(control_spectra, dtype=np.float64)
    positions = [tuple(map(int, value)) for value in control_positions_yx]
    if controls.ndim != 2 or controls.shape != (len(positions), waves.size):
        raise ValueError("Control spectra and position counts do not match.")
    if np.asarray(object_spectrum).shape != waves.shape or good.shape != waves.shape:
        raise ValueError("Object spectrum, mask, and wavelength shapes must match.")

    search_mask = _search_mask(waves, good, config)
    continuum_window_A = float(config["continuum_window_A"])
    object_products = _snr_like_from_target(
        waves,
        good,
        object_spectrum,
        controls,
        continuum_window_A,
    )

    control_angles = np.asarray(
        [_position_angle_deg(position, star_yx) for position in positions],
        dtype=np.float64,
    )
    exclude_deg = float(config["control_reference_exclude_angle_deg"])
    null_snr = np.full_like(controls, np.nan, dtype=np.float64)
    null_rows = []
    reference_counts = []
    for index, position in enumerate(positions):
        reference_indices = [
            other
            for other in range(len(positions))
            if other != index
            and angular_separation_deg(
                np.radians(control_angles[index]),
                np.radians(control_angles[other]),
            )
            >= exclude_deg
        ]
        if len(reference_indices) < 3:
            raise ValueError(
                f"Control {index} has only {len(reference_indices)} independent references."
            )
        products = _snr_like_from_target(
            waves,
            good,
            controls[index],
            controls[reference_indices],
            continuum_window_A,
        )
        null_snr[index] = products["snr_like"]
        maximum = _maximum_row(waves, null_snr[index], search_mask)
        reference_counts.append(len(reference_indices))
        null_rows.append(
            {
                "control_id": index + 1,
                "control_y": int(position[0]),
                "control_x": int(position[1]),
                "control_pa_deg": float(control_angles[index]),
                "n_reference_controls": len(reference_indices),
                "max_snr_like": maximum["snr_like"],
                "max_wavelength_A": maximum["wavelength_A"],
            }
        )

    null_maxima = np.asarray([row["max_snr_like"] for row in null_rows], dtype=np.float64)
    object_maximum = _maximum_row(waves, object_products["snr_like"], search_mask)
    threshold_rows = empirical_threshold_rows(null_maxima, config["fap_levels"])

    candidate_rows = []
    peak_indices = top_spectral_peaks(
        waves,
        object_products["snr_like"],
        search_mask,
        n_peaks=config["top_n_peaks"],
        min_separation_A=config["peak_min_separation_A"],
    )
    for rank, index in enumerate(peak_indices, start=1):
        candidate_rows.append(
            {
                "candidate_type": "ranked_peak",
                "rank": rank,
                "wavelength_A": float(waves[index]),
                "snr_like": float(object_products["snr_like"][index]),
                "local_empirical_fap": empirical_fap(
                    object_products["snr_like"][index], null_snr[:, index]
                ),
                "global_empirical_fap": empirical_fap(
                    object_products["snr_like"][index], null_maxima
                ),
            }
        )

    halpha_mask = (
        search_mask
        & (waves >= float(config["halpha_A"]) - float(config["halpha_half_width_A"]))
        & (waves <= float(config["halpha_A"]) + float(config["halpha_half_width_A"]))
    )
    halpha_maximum = _maximum_row(waves, object_products["snr_like"], halpha_mask)
    null_halpha_maxima = np.asarray(
        [_maximum_row(waves, row, halpha_mask)["snr_like"] for row in null_snr],
        dtype=np.float64,
    )
    halpha_pointwise_fap = np.nan
    halpha_window_fap = np.nan
    if halpha_maximum["index"] is not None:
        index = halpha_maximum["index"]
        halpha_pointwise_fap = empirical_fap(
            halpha_maximum["snr_like"], null_snr[:, index]
        )
        halpha_window_fap = empirical_fap(
            halpha_maximum["snr_like"], null_halpha_maxima
        )
        candidate_rows.append(
            {
                "candidate_type": "Halpha_window",
                "rank": None,
                "wavelength_A": halpha_maximum["wavelength_A"],
                "snr_like": halpha_maximum["snr_like"],
                "local_empirical_fap": halpha_window_fap,
                "global_empirical_fap": empirical_fap(
                    halpha_maximum["snr_like"], null_maxima
                ),
            }
        )

    return {
        "search_mask": search_mask,
        "object_products": object_products,
        "object_maximum": object_maximum,
        "object_global_fap": empirical_fap(object_maximum["snr_like"], null_maxima),
        "null_snr_like": null_snr,
        "null_maxima": null_maxima,
        "null_rows": null_rows,
        "candidate_rows": candidate_rows,
        "threshold_rows": threshold_rows,
        "halpha_maximum": halpha_maximum,
        "null_halpha_maxima": null_halpha_maxima,
        "halpha_pointwise_empirical_fap": halpha_pointwise_fap,
        "halpha_window_empirical_fap": halpha_window_fap,
        "reference_counts": reference_counts,
        "control_angles_deg": control_angles,
    }


def stage08c_config_from_run(run_id=None, *, project_root=None, overrides=None):
    """Build Stage 08c configuration from the frozen run and Stage 8 QC."""

    run_config = load_run_config(run_id, project_root=project_root)
    paths = run_config.paths
    qc_path = paths.stage_dir / "stage08_full_spectrum_qc.json"
    if not qc_path.exists():
        raise FileNotFoundError(f"Stage 8 QC is required before C4: {qc_path}")
    qc08 = read_json(qc_path)
    cfg = run_config.config
    config = {
        "run_id": run_config.run_id,
        "project_root": str(paths.project_root),
        "default_aperture": str(qc08["default_aperture"]),
        "continuum_window_A": float(qc08["continuum_window_A"]),
        **STAGE08C_DEFAULTS,
    }
    for key, default_value in STAGE08C_DEFAULTS.items():
        config[key] = cfg.get(f"stage08c_{key}", default_value)
    if overrides:
        config.update(overrides)
    return config


def _extract_real_inputs(config, paths):
    qc08 = read_json(paths["stage08_qc"])
    with np.load(paths["stage08_npz"]) as payload:
        wavelengths = np.asarray(payload["wavelength_A"], dtype=np.float64)
        good_mask = np.asarray(payload["good_wave_mask"], dtype=bool)
        object_key = f"{config['default_aperture']}_object_local_residual"
        if object_key not in payload:
            raise KeyError(f"Stage 8 NPZ does not contain {object_key}.")
        object_spectrum = np.asarray(payload[object_key], dtype=np.float64)

    aperture = next(
        (value for value in qc08["apertures"] if value["name"] == config["default_aperture"]),
        None,
    )
    if aperture is None:
        raise ValueError(f"Unknown Stage 8 aperture: {config['default_aperture']}")
    object_yx = tuple(map(int, qc08["object_yx"]))
    star_yx = tuple(map(int, qc08["star_yx"]))
    input_cube_fits = Path(qc08["input_cube_fits"])
    if not input_cube_fits.exists():
        raise FileNotFoundError(f"Stage 8 input cube is missing: {input_cube_fits}")

    with fits.open(input_cube_fits, memmap=True) as hdul:
        cube = get_cube_data(hdul)
        if cube.shape[0] != wavelengths.size:
            raise ValueError("Stage 8 cube and wavelength axes do not match.")
        positions = same_radius_control_positions(
            object_yx,
            star_yx,
            cube.shape[1],
            cube.shape[2],
            n_positions=config["n_control_positions"],
            exclude_angle_deg=config["object_exclude_angle_deg"],
            margin_px=config["control_margin_px"],
        )
        positions = list(dict.fromkeys(positions))
        wave_indices = np.where(good_mask)[0]
        control_spectra = []
        for index, position in enumerate(positions, start=1):
            print(f"Stage 08c control {index}/{len(positions)} at {position}")
            spectra, _, _ = local_surface_spectra_fast(
                cube,
                position,
                [aperture],
                wave_indices,
                float(qc08["fit_radius_px"]),
                float(qc08["mask_radius_px"]),
                str(qc08["local_model_kind"]),
                int(qc08["local_fit_min_pixels"]),
            )
            control_spectra.append(np.asarray(spectra[config["default_aperture"]]))

    return {
        "wavelengths": wavelengths,
        "good_mask": good_mask,
        "object_spectrum": object_spectrum,
        "control_spectra": np.asarray(control_spectra, dtype=np.float64),
        "control_positions_yx": positions,
        "star_yx": star_yx,
        "object_yx": object_yx,
        "stage08_qc": qc08,
    }


def save_stage08c_plot(products, wavelengths, path, config):
    """Save the C4 object/null diagnostic figure."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    waves = np.asarray(wavelengths)
    object_snr = products["object_products"]["snr_like"]
    null_maxima = products["null_maxima"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    axes[0, 0].plot(waves, object_snr, color="0.25", linewidth=0.7)
    for row in products["threshold_rows"]:
        if row["threshold_snr_like"] is not None:
            axes[0, 0].axhline(
                row["threshold_snr_like"],
                linestyle="--",
                linewidth=1,
                label=f"global FAP {100 * row['fap_level']:.0f}%",
            )
    axes[0, 0].axvline(float(config["halpha_A"]), color="tab:red", alpha=0.7, label="Halpha")
    axes[0, 0].set(xlabel="Wavelength [A]", ylabel="Object S/N-like", title="Full spectral search")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.2)

    sorted_maxima = np.sort(null_maxima)
    empirical_cdf = np.arange(1, len(sorted_maxima) + 1) / len(sorted_maxima)
    axes[0, 1].step(sorted_maxima, empirical_cdf, where="post", label="Null maxima")
    axes[0, 1].axvline(
        products["object_maximum"]["snr_like"], color="black", linestyle="--", label="Object max"
    )
    axes[0, 1].axvline(
        products["halpha_maximum"]["snr_like"], color="tab:red", linestyle=":", label="Halpha max"
    )
    axes[0, 1].set(xlabel="Maximum S/N-like", ylabel="Empirical CDF", title="Look-elsewhere null")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].scatter(products["control_angles_deg"], null_maxima, color="tab:blue")
    axes[1, 0].axhline(
        products["object_maximum"]["snr_like"], color="black", linestyle="--", label="Object max"
    )
    axes[1, 0].set(xlabel="Control PA [deg]", ylabel="Maximum S/N-like", title="Spatial null maxima")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.2)

    ranked = [row for row in products["candidate_rows"] if row["candidate_type"] == "ranked_peak"]
    axes[1, 1].scatter(
        [row["wavelength_A"] for row in ranked],
        [row["global_empirical_fap"] for row in ranked],
        c=[row["snr_like"] for row in ranked],
        cmap="viridis",
        edgecolor="black",
    )
    axes[1, 1].axhline(0.05, color="tab:red", linestyle="--", label="FAP 5%")
    axes[1, 1].set(
        xlabel="Candidate wavelength [A]",
        ylabel="Global empirical FAP",
        ylim=(-0.02, 1.02),
        title="Top object peaks after trial correction",
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)

    figure.suptitle(
        f"{config['run_id']} Stage 08c: empirical look-elsewhere calibration "
        f"({len(null_maxima)} controls)"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_stage08c_products(products, inputs, config, paths):
    """Write C4 tables, compact spectra, QC, and diagnostic plot."""

    paths["null_maxima_csv"].parent.mkdir(parents=True, exist_ok=True)
    paths["spectra_npz"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_plot"].parent.mkdir(parents=True, exist_ok=True)
    write_csv(paths["null_maxima_csv"], products["null_rows"], list(products["null_rows"][0]))
    write_csv(
        paths["candidates_csv"],
        products["candidate_rows"],
        list(products["candidate_rows"][0]),
    )
    write_csv(
        paths["thresholds_csv"],
        products["threshold_rows"],
        list(products["threshold_rows"][0]),
    )
    np.savez_compressed(
        paths["spectra_npz"],
        wavelength_A=inputs["wavelengths"],
        good_wave_mask=inputs["good_mask"],
        search_mask=products["search_mask"],
        object_snr_like=products["object_products"]["snr_like"],
        null_snr_like=products["null_snr_like"],
        null_maxima=products["null_maxima"],
        null_halpha_maxima=products["null_halpha_maxima"],
        control_yx=np.asarray(inputs["control_positions_yx"], dtype=np.int16),
        control_pa_deg=products["control_angles_deg"],
    )
    save_stage08c_plot(products, inputs["wavelengths"], paths["summary_plot"], config)

    fap_resolution = float(1.0 / (len(products["null_maxima"]) + 1))
    qc = {
        "run_id": config["run_id"],
        "stage": "stage08c_look_elsewhere",
        "input_stage08_npz": str(paths["stage08_npz"]),
        "input_stage08_qc": str(paths["stage08_qc"]),
        "default_aperture": config["default_aperture"],
        "object_yx": list(map(int, inputs["object_yx"])),
        "star_yx": list(map(int, inputs["star_yx"])),
        "control_yx": [list(map(int, value)) for value in inputs["control_positions_yx"]],
        "n_control_spectra": len(products["null_rows"]),
        "n_search_channels": int(np.sum(products["search_mask"])),
        "continuum_window_A": float(config["continuum_window_A"]),
        "control_reference_exclude_angle_deg": float(
            config["control_reference_exclude_angle_deg"]
        ),
        "reference_count_min_max": [
            int(min(products["reference_counts"])),
            int(max(products["reference_counts"])),
        ],
        "one_sided_search": "positive emission peaks",
        "minimum_resolvable_fap": fap_resolution,
        "thresholds": products["threshold_rows"],
        "object_global_max": products["object_maximum"],
        "object_global_empirical_fap": products["object_global_fap"],
        "halpha_window_max": products["halpha_maximum"],
        "halpha_pointwise_empirical_fap": products["halpha_pointwise_empirical_fap"],
        "halpha_window_empirical_fap": products["halpha_window_empirical_fap"],
        "halpha_global_empirical_fap": next(
            row["global_empirical_fap"]
            for row in products["candidate_rows"]
            if row["candidate_type"] == "Halpha_window"
        ),
        "outputs": {
            "null_maxima_csv": str(paths["null_maxima_csv"]),
            "candidates_csv": str(paths["candidates_csv"]),
            "thresholds_csv": str(paths["thresholds_csv"]),
            "spectra_npz": str(paths["spectra_npz"]),
            "qc_json": str(paths["qc_json"]),
            "summary_plot": str(paths["summary_plot"]),
        },
        "scope_note": (
            "FAP values are empirical ranks among same-radius spatial controls and include "
            "the spectral look-elsewhere effect over the configured search mask. Nearby "
            "control positions may remain spatially correlated; thresholds below the stated "
            "minimum resolvable FAP are not reported."
        ),
    }
    write_json(paths["qc_json"], qc)
    return qc


def run_stage08c(config=None, *, project_root=None):
    """Run C4 without replacing any Stage 8 product."""

    if config is None:
        config = stage08c_config_from_run(project_root=project_root)
    else:
        config = dict(config)
    root = Path(project_root or config.get("project_root") or Path.cwd()).resolve()
    paths = stage08c_paths(config["run_id"], project_root=root)
    for key in ("stage08_npz", "stage08_qc"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing Stage 8 input: {paths[key]}")
    inputs = _extract_real_inputs(config, paths)
    products = compute_stage08c_products(
        inputs["wavelengths"],
        inputs["good_mask"],
        inputs["object_spectrum"],
        inputs["control_spectra"],
        inputs["control_positions_yx"],
        inputs["star_yx"],
        config,
    )
    qc = write_stage08c_products(products, inputs, config, paths)
    return {"paths": paths, "inputs": inputs, "products": products, "qc": qc}


__all__ = [
    "STAGE08C_DEFAULTS",
    "compute_stage08c_products",
    "empirical_fap",
    "empirical_threshold_rows",
    "run_stage08c",
    "save_stage08c_plot",
    "stage08c_config_from_run",
    "stage08c_paths",
    "top_spectral_peaks",
    "write_stage08c_products",
]
