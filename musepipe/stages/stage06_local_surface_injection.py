"""Stage 06: Halpha injection-recovery through local-surface subtraction."""

from __future__ import annotations

import math
from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits

from ..apertures import box_spectrum_sum
from ..config import load_run_config
from ..io import read_json, write_csv, write_json
from ..localfit import subtract_local_surface_cube
from ..paths import RunPaths
from ..spectral import nearest_channel_indices
from ..stats import robust_sigma
from .stage04b_local_surface import (
    load_stage04b_input,
    stage04b_config_from_run,
    stage04b_paths,
)
from .stage07b_halpha_robustness import resolve_object_yx, resolve_star_yx


STAGE06_LOCAL_DEFAULTS = {
    "line_center_A": 6562.8,
    "halpha_channels_A": [6560.96, 6562.21, 6563.46],
    "cont_ha_min_A": 6570.0,
    "cont_ha_max_A": 6700.0,
    "muse_mode": "NFM",
    "pixel_scale_arcsec": 0.025,
    "spatial_psf_fwhm_arcsec": 0.070,
    "muse_R_at_halpha": 2484.0,
    "template_radius_nsigma": 5.0,
    "match_reference_separation": True,
    "injection_yx": None,
    "injection_pa_offset_deg": 180.0,
    "injection_radius_arcsec": 0.45,
    "injection_pa_deg": 90.0,
    "inject_in_all_cubes": True,
    "cube_index_to_inject": 0,
    "use_stage04_selection": True,
    "n_best_cubes": None,
    "injection_snr_grid": [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0],
    "nominal_snr": 3.0,
    "injection_snr_reference": "local_surface_matched_filter",
    "recovery_mode": "realistic",
    "box_size": 3,
    "n_control_angles": 12,
    "control_exclude_pa_deg": 30.0,
    "diagnostic_halfwidth_A": 75.0,
    "save_diagnostic_fits": True,
    "fig_convolve_sigma_px": 2.0,
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


def cube_center_yx(ny, nx):
    return ((float(ny) - 1.0) / 2.0, (float(nx) - 1.0) / 2.0)


def separation_pa_from_center(yx, center_yx, pixel_scale_arcsec):
    y, x = map(float, yx)
    cy, cx = map(float, center_yx)
    dy = y - cy
    dx = x - cx
    radius_px = float(math.hypot(dy, dx))
    pa_deg = float((math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)
    return radius_px, radius_px * float(pixel_scale_arcsec), pa_deg


def yx_from_center_radius_pa(center_yx, radius_px, pa_deg):
    cy, cx = map(float, center_yx)
    pa_rad = math.radians(float(pa_deg))
    y = int(round(cy + float(radius_px) * math.cos(pa_rad)))
    x = int(round(cx + float(radius_px) * math.sin(pa_rad)))
    return (y, x)


def validate_yx(yx, ny, nx, *, margin_px=0, label="position"):
    if yx is None:
        raise ValueError(f"{label} is missing.")
    y, x = map(int, yx)
    margin = int(math.ceil(float(margin_px)))
    if not (margin <= y < int(ny) - margin and margin <= x < int(nx) - margin):
        raise ValueError(
            f"{label}={(y, x)} is outside shape {(int(ny), int(nx))} "
            f"with margin_px={margin}."
        )
    return (y, x)


def resolve_injection_geometry(config, ny, nx):
    star_yx = tuple(map(float, config.get("star_yx") or cube_center_yx(ny, nx)))
    reference_yx = _as_yx(config.get("reference_object_yx"))
    manual_yx = _as_yx(config.get("injection_yx"))
    pixel_scale = float(config["pixel_scale_arcsec"])

    reference_sep_px = np.nan
    reference_sep_arcsec = np.nan
    reference_pa_deg = np.nan
    requested_pa_deg = np.nan

    if manual_yx is not None:
        injection_yx = manual_yx
        source = "manual"
    elif bool(config.get("match_reference_separation", True)):
        if reference_yx is None:
            raise ValueError("reference_object_yx is required when match_reference_separation=True.")
        reference_sep_px, reference_sep_arcsec, reference_pa_deg = separation_pa_from_center(
            reference_yx, star_yx, pixel_scale
        )
        requested_pa_deg = (reference_pa_deg + float(config["injection_pa_offset_deg"])) % 360.0
        injection_yx = yx_from_center_radius_pa(star_yx, reference_sep_px, requested_pa_deg)
        source = "reference_separation"
    else:
        reference_sep_arcsec = float(config["injection_radius_arcsec"])
        reference_sep_px = reference_sep_arcsec / pixel_scale
        requested_pa_deg = float(config["injection_pa_deg"])
        injection_yx = yx_from_center_radius_pa(star_yx, reference_sep_px, requested_pa_deg)
        source = "radius_pa"

    injection_yx = validate_yx(
        injection_yx,
        ny,
        nx,
        margin_px=float(config["fit_radius_px"]),
        label="injection_yx",
    )
    injection_sep_px, injection_sep_arcsec, actual_pa_deg = separation_pa_from_center(
        injection_yx, star_yx, pixel_scale
    )
    return {
        "star_yx": star_yx,
        "reference_object_yx": reference_yx,
        "injection_yx": injection_yx,
        "injection_source": source,
        "reference_separation_px": float(reference_sep_px),
        "reference_separation_arcsec": float(reference_sep_arcsec),
        "reference_pa_deg": float(reference_pa_deg),
        "requested_injection_pa_deg": float(requested_pa_deg),
        "injection_separation_px": float(injection_sep_px),
        "injection_separation_arcsec": float(injection_sep_arcsec),
        "actual_injection_pa_deg": float(actual_pa_deg),
    }


def make_spatial_psf_patch(ny, nx, y0, x0, fwhm_px, nsigma=5.0):
    sigma_px = float(fwhm_px) / 2.354820045
    if not np.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError("Spatial PSF FWHM must be positive.")
    radius = int(math.ceil(float(nsigma) * sigma_px))
    y1 = max(0, int(y0) - radius)
    y2 = min(int(ny), int(y0) + radius + 1)
    x1 = max(0, int(x0) - radius)
    x2 = min(int(nx), int(x0) + radius + 1)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    psf = np.exp(-0.5 * (((yy - float(y0)) / sigma_px) ** 2 + ((xx - float(x0)) / sigma_px) ** 2))
    psf = np.asarray(psf, dtype=np.float64)
    norm = float(np.sum(psf))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Invalid spatial PSF normalization.")
    psf /= norm
    return slice(y1, y2), slice(x1, x2), psf.astype(np.float32)


def make_spectral_profile_patch(wavelengths, center_A, fwhm_A, dlam_A, nsigma=5.0):
    waves = np.asarray(wavelengths, dtype=np.float64)
    sigma_A = float(fwhm_A) / 2.354820045
    indices = np.where(np.abs(waves - float(center_A)) <= float(nsigma) * sigma_A)[0]
    if indices.size == 0:
        indices = np.asarray([int(np.nanargmin(np.abs(waves - float(center_A))))])
    z1, z2 = int(indices.min()), int(indices.max()) + 1
    profile = np.exp(-0.5 * ((waves[z1:z2] - float(center_A)) / sigma_A) ** 2)
    norm = float(np.sum(profile * float(dlam_A)))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Invalid spectral profile normalization.")
    profile /= norm
    return slice(z1, z2), profile.astype(np.float32)


def make_signal_template(
    wavelengths,
    ny,
    nx,
    y0,
    x0,
    pixel_scale_arcsec,
    spatial_fwhm_arcsec,
    line_center_A,
    line_fwhm_A,
    dlam_A,
    nsigma=5.0,
):
    fwhm_px = float(spatial_fwhm_arcsec) / float(pixel_scale_arcsec)
    ysl, xsl, spatial_psf = make_spatial_psf_patch(ny, nx, y0, x0, fwhm_px, nsigma=nsigma)
    zsl, spectral_profile = make_spectral_profile_patch(
        wavelengths, line_center_A, line_fwhm_A, dlam_A, nsigma=nsigma
    )
    template = spectral_profile[:, None, None] * spatial_psf[None, :, :]
    return zsl, ysl, xsl, template.astype(np.float32), spectral_profile, spatial_psf


def local_surface_control_positions(
    star_yx,
    injection_yx,
    reference_yx,
    ny,
    nx,
    *,
    n_angles=12,
    exclude_pa_deg=30.0,
    margin_px=4,
    min_injection_distance_px=0.0,
):
    sy, sx = map(float, star_yx)
    iy, ix = map(float, injection_yx)
    radius = float(math.hypot(iy - sy, ix - sx))
    pa_inj = math.atan2(iy - sy, ix - sx)
    pa_ref = None
    if reference_yx is not None:
        ry, rx = map(float, reference_yx)
        pa_ref = math.atan2(ry - sy, rx - sx)
    exclude_rad = math.radians(float(exclude_pa_deg))
    margin = int(math.ceil(float(margin_px)))

    controls = []
    for index in range(int(n_angles)):
        theta = 2.0 * math.pi * index / float(n_angles)
        if abs(math.atan2(math.sin(theta - pa_inj), math.cos(theta - pa_inj))) < exclude_rad:
            continue
        if pa_ref is not None:
            delta_ref = abs(math.atan2(math.sin(theta - pa_ref), math.cos(theta - pa_ref)))
            if delta_ref < exclude_rad:
                continue
        y = int(round(sy + radius * math.sin(theta)))
        x = int(round(sx + radius * math.cos(theta)))
        if not (margin <= y < int(ny) - margin and margin <= x < int(nx) - margin):
            continue
        if math.hypot(y - iy, x - ix) < float(min_injection_distance_px):
            continue
        if (y, x) not in controls:
            controls.append((y, x))
    return controls


def inject_physical_template(
    cube_norm,
    med_pix,
    zsl,
    ysl,
    xsl,
    template,
    total_line_flux_native,
):
    """Inject a physical-units template into one normalized cube."""

    out = np.array(cube_norm, dtype=np.float32, copy=True)
    scale = np.asarray(med_pix[ysl, xsl], dtype=np.float32)
    valid = np.isfinite(scale) & (np.abs(scale) > 1e-12)
    addition = np.full(template.shape, np.nan, dtype=np.float32)
    addition[:, valid] = (
        float(total_line_flux_native) * np.asarray(template[:, valid], dtype=np.float32) / scale[valid][None, :]
    )
    patch = out[zsl, ysl, xsl]
    finite_add = np.isfinite(addition)
    patch[finite_add] += addition[finite_add]
    out[zsl, ysl, xsl] = patch
    return out


def _nanmean_stack(values):
    stack = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(stack)
    count = np.sum(finite, axis=0)
    total = np.nansum(stack, axis=0)
    out = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out.astype(np.float32)


def combined_local_surface_residual(
    cubes_diag,
    med_pix_stack,
    selected_indices,
    center_yx,
    config,
    *,
    injection_flux_native=None,
    injection_zsl=None,
    injection_ysl=None,
    injection_xsl=None,
    injection_template=None,
    extra_exclusion_yx=None,
):
    """Subtract local surfaces and combine selected cubes in physical units."""

    cubes = np.asarray(cubes_diag, dtype=np.float32)
    med = np.asarray(med_pix_stack, dtype=np.float32)
    selected = [int(index) for index in selected_indices]
    if not selected:
        raise ValueError("At least one selected cube is required.")

    residuals = []
    for cube_index in selected:
        work = np.array(cubes[cube_index], dtype=np.float32, copy=True)
        should_inject = bool(config.get("inject_in_all_cubes", True)) or (
            cube_index == int(config.get("cube_index_to_inject", 0))
        )
        if injection_flux_native is not None and should_inject:
            work = inject_physical_template(
                work,
                med[cube_index],
                injection_zsl,
                injection_ysl,
                injection_xsl,
                injection_template,
                injection_flux_native,
            )

        residual_norm, _, _ = subtract_local_surface_cube(
            work,
            center_yx,
            fit_radius_px=float(config["fit_radius_px"]),
            mask_radius_px=float(config["mask_radius_px"]),
            model_kind=config["local_model_kind"],
            extra_exclusion_yx=extra_exclusion_yx,
            extra_exclusion_radius_px=float(config["other_mask_radius_px"]),
            sigma_clip=float(config["local_fit_sigma_clip"]),
            max_iter=int(config["local_fit_max_iter"]),
            min_fit_pixels=int(config["local_fit_min_pixels"]),
        )
        residuals.append(residual_norm * med[cube_index][None, :, :])
    return _nanmean_stack(residuals)


def integrated_line_flux(spectrum, line_indices, dlam_A, continuum_mask=None):
    spec = np.asarray(spectrum, dtype=np.float64)
    continuum = 0.0
    if continuum_mask is not None and np.any(continuum_mask):
        continuum = float(np.nanmedian(spec[np.asarray(continuum_mask, dtype=bool)]))
    return float(np.nansum(spec[np.asarray(line_indices, dtype=int)] - continuum) * float(dlam_A))


def aperture_line_flux(cube, center_yx, line_indices, continuum_mask, dlam_A, box_size=3):
    spectrum = box_spectrum_sum(cube, center_yx[0], center_yx[1], box_size=box_size)
    return integrated_line_flux(spectrum, line_indices, dlam_A, continuum_mask=continuum_mask)


def matched_filter_flux(
    cube,
    zsl,
    ysl,
    xsl,
    template,
    continuum_mask=None,
):
    patch = np.asarray(cube[zsl, ysl, xsl], dtype=np.float64)
    temp = np.asarray(template, dtype=np.float64)
    if continuum_mask is not None and np.any(continuum_mask):
        continuum = np.nanmedian(np.asarray(cube)[np.asarray(continuum_mask, dtype=bool), ysl, xsl], axis=0)
        patch = patch - continuum[None, :, :]
    denom = float(np.nansum(temp * temp))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.nansum(patch * temp) / denom)


def _matched_flux_sigma(cube, zsl, ysl, xsl, template, continuum_mask):
    local = np.asarray(cube)[np.asarray(continuum_mask, dtype=bool), ysl, xsl]
    voxel_sigma = robust_sigma(local.ravel())
    denom = float(np.nansum(np.asarray(template, dtype=np.float64) ** 2))
    if not np.isfinite(voxel_sigma) or voxel_sigma <= 0 or denom <= 0:
        return np.nan
    return float(voxel_sigma / math.sqrt(denom))


def _expected_combined_flux(flux, selected_indices, inject_all, cube_index):
    if inject_all:
        return float(flux)
    selected = [int(index) for index in selected_indices]
    return float(flux) / len(selected) if int(cube_index) in selected else 0.0


def _selected_cube_indices(config, n_cubes, paths=None):
    selected = None
    source = "all_cubes"
    if bool(config.get("use_stage04_selection", True)) and paths is not None:
        path = paths["stage04_best4_indices"]
        if path.exists():
            selected = [int(value) for value in np.load(path).ravel()]
            source = path.name
    if selected is None:
        selected = list(range(int(n_cubes)))
    selected = [index for index in selected if 0 <= index < int(n_cubes)]
    n_best = config.get("n_best_cubes")
    if n_best is not None:
        selected = selected[: max(1, int(n_best))]
    if not selected:
        raise RuntimeError("No valid selected cubes for Stage 06 local-surface injection.")
    return selected, source


def compute_stage06_local_products(
    cubes_norm,
    wavelengths,
    med_pix_stack,
    config,
    *,
    selected_indices=None,
    selection_source="explicit",
):
    """Compute a local-surface Halpha injection-recovery grid."""

    config = {**STAGE06_LOCAL_DEFAULTS, **dict(config)}
    cubes = np.asarray(cubes_norm, dtype=np.float32)
    waves = np.asarray(wavelengths, dtype=np.float64)
    med = np.asarray(med_pix_stack, dtype=np.float32)
    if cubes.ndim != 4:
        raise ValueError(f"Expected cubes with shape (N,nz,ny,nx), got {cubes.shape}.")
    n_cubes, n_wave, ny, nx = cubes.shape
    if waves.shape != (n_wave,):
        raise ValueError(f"Wavelength shape {waves.shape} does not match n_wave={n_wave}.")
    if med.shape != (n_cubes, ny, nx):
        raise ValueError(f"MEDPIX shape {med.shape} does not match {(n_cubes, ny, nx)}.")

    recovery_mode = str(config["recovery_mode"]).strip().lower()
    if recovery_mode not in {"realistic", "deterministic"}:
        raise ValueError("recovery_mode must be 'realistic' or 'deterministic'.")
    snr_reference = str(config["injection_snr_reference"]).strip().lower()
    allowed_references = {"local_surface_matched_filter", "local_surface_aperture_3x3"}
    if snr_reference not in allowed_references:
        raise ValueError(f"injection_snr_reference must be one of {sorted(allowed_references)}.")

    dlam_A = float(np.nanmedian(np.diff(waves)))
    if not np.isfinite(dlam_A) or dlam_A <= 0:
        raise ValueError("Wavelength axis must be finite and increasing.")
    line_fwhm_A = float(config["line_center_A"]) / float(config["muse_R_at_halpha"])
    geometry = resolve_injection_geometry(config, ny, nx)
    inj_y, inj_x = geometry["injection_yx"]

    zsl_full, ysl, xsl, template, spectral_profile, spatial_psf = make_signal_template(
        waves,
        ny,
        nx,
        inj_y,
        inj_x,
        float(config["pixel_scale_arcsec"]),
        float(config["spatial_psf_fwhm_arcsec"]),
        float(config["line_center_A"]),
        line_fwhm_A,
        dlam_A,
        nsigma=float(config["template_radius_nsigma"]),
    )

    diag_min_A = min(
        float(config["line_center_A"]) - float(config["diagnostic_halfwidth_A"]),
        float(config["cont_ha_min_A"]),
    )
    diag_max_A = max(
        float(config["line_center_A"]) + float(config["diagnostic_halfwidth_A"]),
        float(config["cont_ha_max_A"]),
    )
    diag_indices = np.where((waves >= diag_min_A) & (waves <= diag_max_A))[0]
    if diag_indices.size == 0:
        raise RuntimeError("No channels in the Stage 06 diagnostic wavelength window.")
    diag_start, diag_stop = int(diag_indices.min()), int(diag_indices.max()) + 1
    if not (diag_start <= zsl_full.start and zsl_full.stop <= diag_stop):
        raise RuntimeError("Diagnostic window does not contain the complete injection template.")
    diag_waves = waves[diag_start:diag_stop]
    cubes_diag = cubes[:, diag_start:diag_stop]
    zsl_diag = slice(zsl_full.start - diag_start, zsl_full.stop - diag_start)
    line_indices = nearest_channel_indices(diag_waves, config["halpha_channels_A"])
    continuum_mask = (diag_waves >= float(config["cont_ha_min_A"])) & (
        diag_waves <= float(config["cont_ha_max_A"])
    )
    if not np.any(continuum_mask):
        raise RuntimeError("No continuum channels in the Stage 06 diagnostic window.")

    if selected_indices is None:
        selected_indices = list(range(n_cubes))
        selection_source = "all_cubes"
    selected_indices = [int(index) for index in selected_indices]

    reference_yx = geometry["reference_object_yx"]
    base_target = combined_local_surface_residual(
        cubes_diag,
        med,
        selected_indices,
        geometry["injection_yx"],
        config,
        extra_exclusion_yx=[reference_yx] if reference_yx is not None else None,
    )

    template_radius_px = max(
        int(inj_y - ysl.start),
        int(ysl.stop - inj_y - 1),
        int(inj_x - xsl.start),
        int(xsl.stop - inj_x - 1),
    )
    control_margin = int(math.ceil(float(config["fit_radius_px"])))
    controls = local_surface_control_positions(
        geometry["star_yx"],
        geometry["injection_yx"],
        reference_yx,
        ny,
        nx,
        n_angles=int(config["n_control_angles"]),
        exclude_pa_deg=float(config["control_exclude_pa_deg"]),
        margin_px=control_margin,
        min_injection_distance_px=float(config["fit_radius_px"]) + template_radius_px,
    )
    if recovery_mode == "realistic" and len(controls) < 4:
        raise RuntimeError(f"Only {len(controls)} valid same-radius controls; at least 4 are required.")

    control_aperture_fluxes = []
    control_matched_fluxes = []
    for control_yx in controls:
        exclusions = [geometry["injection_yx"]]
        if reference_yx is not None:
            exclusions.append(reference_yx)
        control_cube = combined_local_surface_residual(
            cubes_diag,
            med,
            selected_indices,
            control_yx,
            config,
            extra_exclusion_yx=exclusions,
        )
        control_aperture_fluxes.append(
            aperture_line_flux(
                control_cube,
                control_yx,
                line_indices,
                continuum_mask,
                dlam_A,
                box_size=int(config["box_size"]),
            )
        )
        dy = int(control_yx[0] - inj_y)
        dx = int(control_yx[1] - inj_x)
        shifted_ysl = slice(ysl.start + dy, ysl.stop + dy)
        shifted_xsl = slice(xsl.start + dx, xsl.stop + dx)
        control_matched_fluxes.append(
            matched_filter_flux(
                control_cube,
                zsl_diag,
                shifted_ysl,
                shifted_xsl,
                template,
                continuum_mask=continuum_mask,
            )
        )

    realistic_aperture_noise = robust_sigma(control_aperture_fluxes)
    realistic_matched_noise = robust_sigma(control_matched_fluxes)
    base_spectrum = box_spectrum_sum(base_target, inj_y, inj_x, box_size=int(config["box_size"]))
    channel_noise = robust_sigma(base_spectrum[continuum_mask])
    deterministic_aperture_noise = float(channel_noise * math.sqrt(len(line_indices)) * dlam_A)
    deterministic_matched_noise = _matched_flux_sigma(
        base_target, zsl_diag, ysl, xsl, template, continuum_mask
    )

    if not np.isfinite(realistic_aperture_noise) or realistic_aperture_noise <= 0:
        realistic_aperture_noise = deterministic_aperture_noise
        aperture_noise_source = "target_continuum_fallback"
    else:
        aperture_noise_source = "same_radius_controls"
    if not np.isfinite(realistic_matched_noise) or realistic_matched_noise <= 0:
        realistic_matched_noise = deterministic_matched_noise
        matched_noise_source = "target_continuum_fallback"
    else:
        matched_noise_source = "same_radius_controls"

    noise_by_reference = {
        "local_surface_aperture_3x3": float(
            realistic_aperture_noise if recovery_mode == "realistic" else deterministic_aperture_noise
        ),
        "local_surface_matched_filter": float(
            realistic_matched_noise if recovery_mode == "realistic" else deterministic_matched_noise
        ),
    }
    injection_sigma = noise_by_reference[snr_reference]
    if not np.isfinite(injection_sigma) or injection_sigma <= 0:
        raise RuntimeError(f"Invalid injection 1-sigma flux for {snr_reference}: {injection_sigma}.")

    snr_grid = np.asarray(config["injection_snr_grid"], dtype=np.float64)
    if snr_grid.ndim != 1 or snr_grid.size == 0 or np.any(~np.isfinite(snr_grid)):
        raise ValueError("injection_snr_grid must be a non-empty finite 1D sequence.")
    flux_grid = snr_grid * injection_sigma
    nominal_index = int(np.nanargmin(np.abs(snr_grid - float(config["nominal_snr"]))))

    rows = []
    nominal_injected = None
    nominal_delta = None
    for grid_index, (target_snr, flux_native) in enumerate(zip(snr_grid, flux_grid)):
        injected_target = combined_local_surface_residual(
            cubes_diag,
            med,
            selected_indices,
            geometry["injection_yx"],
            config,
            injection_flux_native=float(flux_native),
            injection_zsl=zsl_diag,
            injection_ysl=ysl,
            injection_xsl=xsl,
            injection_template=template,
            extra_exclusion_yx=[reference_yx] if reference_yx is not None else None,
        )
        delta = injected_target - base_target
        expected_flux = _expected_combined_flux(
            flux_native,
            selected_indices,
            bool(config["inject_in_all_cubes"]),
            int(config["cube_index_to_inject"]),
        )

        delta_aperture_flux = aperture_line_flux(
            delta,
            geometry["injection_yx"],
            line_indices,
            None,
            dlam_A,
            box_size=int(config["box_size"]),
        )
        delta_matched_flux = matched_filter_flux(delta, zsl_diag, ysl, xsl, template)
        realistic_aperture_flux = aperture_line_flux(
            injected_target,
            geometry["injection_yx"],
            line_indices,
            continuum_mask,
            dlam_A,
            box_size=int(config["box_size"]),
        )
        realistic_matched_value = matched_filter_flux(
            injected_target,
            zsl_diag,
            ysl,
            xsl,
            template,
            continuum_mask=continuum_mask,
        )

        if recovery_mode == "realistic":
            primary_aperture_flux = realistic_aperture_flux
            primary_aperture_noise = realistic_aperture_noise
            primary_matched_flux = realistic_matched_value
            primary_matched_noise = realistic_matched_noise
        else:
            primary_aperture_flux = delta_aperture_flux
            primary_aperture_noise = deterministic_aperture_noise
            primary_matched_flux = delta_matched_flux
            primary_matched_noise = deterministic_matched_noise

        row = {
            "grid_index": int(grid_index),
            "recovery_mode": recovery_mode,
            "target_snr_input": float(target_snr),
            "injected_total_line_flux_native": float(flux_native),
            "expected_combined_line_flux_native": float(expected_flux),
            "local_surface_aperture_flux_native": float(primary_aperture_flux),
            "local_surface_aperture_line_noise_native": float(primary_aperture_noise),
            "local_surface_aperture_snr": float(primary_aperture_flux / primary_aperture_noise),
            "local_surface_matched_flux_native": float(primary_matched_flux),
            "local_surface_matched_flux_sigma_native": float(primary_matched_noise),
            "local_surface_matched_snr": float(primary_matched_flux / primary_matched_noise),
            "local_surface_matched_throughput": float(primary_matched_flux / expected_flux)
            if expected_flux != 0
            else np.nan,
            "local_surface_aperture_throughput": float(primary_aperture_flux / expected_flux)
            if expected_flux != 0
            else np.nan,
            "delta_aperture_flux_native": float(delta_aperture_flux),
            "delta_matched_flux_native": float(delta_matched_flux),
            "delta_matched_signal_transfer": float(delta_matched_flux / expected_flux)
            if expected_flux != 0
            else np.nan,
            "delta_aperture_signal_transfer": float(delta_aperture_flux / expected_flux)
            if expected_flux != 0
            else np.nan,
            "realistic_aperture_flux_native": float(realistic_aperture_flux),
            "realistic_aperture_line_noise_native": float(realistic_aperture_noise),
            "realistic_aperture_snr": float(realistic_aperture_flux / realistic_aperture_noise),
            "realistic_matched_flux_native": float(realistic_matched_value),
            "realistic_matched_flux_sigma_native": float(realistic_matched_noise),
            "realistic_matched_snr": float(realistic_matched_value / realistic_matched_noise),
            "n_controls": int(len(controls)),
        }
        rows.append(row)
        if grid_index == nominal_index:
            nominal_injected = injected_target
            nominal_delta = delta

    return {
        "wavelengths": waves,
        "diag_wavelengths": diag_waves,
        "diag_start": diag_start,
        "diag_stop": diag_stop,
        "line_indices": np.asarray(line_indices, dtype=int),
        "continuum_mask": np.asarray(continuum_mask, dtype=bool),
        "zsl_full": zsl_full,
        "zsl_diag": zsl_diag,
        "ysl": ysl,
        "xsl": xsl,
        "template": template,
        "spectral_profile": spectral_profile,
        "spatial_psf": spatial_psf,
        "template_integral": float(np.sum(template) * dlam_A),
        "line_fwhm_A": line_fwhm_A,
        "dlam_A": dlam_A,
        "geometry": geometry,
        "selected_indices": selected_indices,
        "selection_source": selection_source,
        "control_positions_yx": controls,
        "control_aperture_fluxes": control_aperture_fluxes,
        "control_matched_fluxes": control_matched_fluxes,
        "aperture_noise_source": aperture_noise_source,
        "matched_noise_source": matched_noise_source,
        "injection_flux_sigma_by_reference": noise_by_reference,
        "injection_flux_sigma_native": float(injection_sigma),
        "snr_grid": snr_grid,
        "flux_grid": flux_grid,
        "nominal_index": nominal_index,
        "base_target_diag": base_target,
        "nominal_injected_diag": nominal_injected,
        "nominal_delta_diag": nominal_delta,
        "rows": rows,
        "recovery_mode": recovery_mode,
    }


def stage06_local_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    injection_dir = paths.stage_dir / "injections"
    plot_dir = paths.plot_stage_dir("stage06_local_surface_injection")
    return {
        "paths": paths,
        "stage04_best4_indices": paths.stage_dir / "stage04_best4_indices.npy",
        "template_fits": injection_dir / "stage06_local_surface_halpha_signal_template.fits",
        "diagnostic_fits": injection_dir / "stage06_local_surface_recovered_halpha_nominal_diag.fits",
        "results_csv": paths.table_dir / "stage06_local_surface_halpha_injection_recovery.csv",
        "qc_json": paths.stage_dir / "stage06_local_surface_halpha_injection_qc.json",
        "truth_json": paths.table_dir / "stage06_local_surface_halpha_injection_truth_summary.json",
        "plot_dir": plot_dir,
        "summary_plot": plot_dir / "stage06_local_surface_halpha_injection_summary.png",
    }


def stage06_local_config_from_run(
    run_id=None,
    *,
    project_root=None,
    overrides=None,
    allow_run_id_mismatch=False,
):
    """Build Stage 06 local-surface config from run config and Stage04b QC."""

    run_config = load_run_config(
        run_id,
        project_root=project_root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    cfg = run_config.config
    stage04_cfg = stage04b_config_from_run(
        run_config.run_id,
        project_root=run_config.paths.project_root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    qc_path = run_config.paths.stage_dir / "stage04b_qc.json"
    qc04b = read_json(qc_path) if qc_path.exists() else {}
    reference_yx = resolve_object_yx(cfg, qc04b)
    star_yx = resolve_star_yx(cfg, qc04b)

    defaults = dict(STAGE06_LOCAL_DEFAULTS)
    defaults["line_center_A"] = float(cfg.get("halpha_A", defaults["line_center_A"]))
    defaults["halpha_channels_A"] = cfg.get("ha_channels_A", defaults["halpha_channels_A"])
    defaults["pixel_scale_arcsec"] = float(cfg.get("pixel_scale_mas", 25.0)) / 1000.0
    defaults["fig_convolve_sigma_px"] = float(cfg.get("fig_convolve_sigma_px", 2.0))

    config = {
        **defaults,
        "run_id": run_config.run_id,
        "target_name": cfg.get("target_name", run_config.run_id),
        "project_root": str(run_config.paths.project_root),
        "crop_npix": int(cfg.get("crop_npix", 0)),
        "reference_object_yx": reference_yx,
        "star_yx": star_yx,
        "input_mode": stage04_cfg["input_mode"],
        "bad_wavelength_ranges_A": stage04_cfg["bad_wavelength_ranges_A"],
        "local_model_kind": stage04_cfg["local_model_kind"],
        "fit_radius_px": stage04_cfg["fit_radius_px"],
        "mask_radius_px": stage04_cfg["mask_radius_px"],
        "local_fit_sigma_clip": stage04_cfg["local_fit_sigma_clip"],
        "local_fit_max_iter": stage04_cfg["local_fit_max_iter"],
        "local_fit_min_pixels": stage04_cfg["local_fit_min_pixels"],
        "other_mask_radius_px": stage04_cfg["other_mask_radius_px"],
    }

    for key, default_value in defaults.items():
        config[key] = cfg.get(f"stage06_local_{key}", default_value)
    config["reference_object_yx"] = _as_yx(
        cfg.get("stage06_local_reference_object_yx", config["reference_object_yx"])
    )
    config["star_yx"] = _as_yx(cfg.get("stage06_local_star_yx", config["star_yx"]))
    config["injection_yx"] = _as_yx(cfg.get("stage06_local_injection_yx", config["injection_yx"]))
    config["injection_snr_grid"] = cfg.get(
        "stage06_local_injection_snr_grid",
        cfg.get("stage06_pca_injection_snr_grid", config["injection_snr_grid"]),
    )
    config["nominal_snr"] = cfg.get(
        "stage06_local_nominal_snr", cfg.get("stage06_pca_nominal_snr", config["nominal_snr"])
    )
    config["recovery_mode"] = cfg.get(
        "stage06_local_recovery_mode", cfg.get("stage06_pca_recovery_mode", config["recovery_mode"])
    )
    if overrides:
        config.update(overrides)
    return config


def _base_header(config, products):
    geometry = products["geometry"]
    header = fits.Header()
    header["RUNID"] = str(config["run_id"])
    header["TARGET"] = str(config.get("target_name", ""))
    header["STAGE"] = "STAGE06"
    header["MODE"] = "LOCSURF"
    header["INJ_Y"] = int(geometry["injection_yx"][0])
    header["INJ_X"] = int(geometry["injection_yx"][1])
    header["LCEN_A"] = float(config["line_center_A"])
    header["LFWHM_A"] = float(products["line_fwhm_A"])
    header["INJFLUX"] = float(products["flux_grid"][products["nominal_index"]])
    header["PIXSCALE"] = float(config["pixel_scale_arcsec"])
    header["PSFFWHM"] = float(config["spatial_psf_fwhm_arcsec"])
    header["DIAGZ0"] = int(products["diag_start"])
    header["DIAGZ1"] = int(products["diag_stop"])
    return header


def _line_image(cube, line_indices, sigma_px=0.0):
    image = np.nanmean(np.asarray(cube)[np.asarray(line_indices, dtype=int)], axis=0)
    if sigma_px is not None and float(sigma_px) > 0:
        from scipy.ndimage import gaussian_filter

        image = gaussian_filter(image, sigma=float(sigma_px))
    return image


def save_stage06_local_plot(path, products, config, *, show_plots=False):
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry = products["geometry"]
    inj_y, inj_x = geometry["injection_yx"]
    radius = int(math.ceil(float(config["fit_radius_px"]))) + 3
    base = _line_image(
        products["base_target_diag"], products["line_indices"], config.get("fig_convolve_sigma_px", 0.0)
    )
    injected = _line_image(
        products["nominal_injected_diag"],
        products["line_indices"],
        config.get("fig_convolve_sigma_px", 0.0),
    )
    delta = _line_image(
        products["nominal_delta_diag"], products["line_indices"], config.get("fig_convolve_sigma_px", 0.0)
    )

    yy, xx = np.mgrid[: base.shape[0], : base.shape[1]]
    local_display = np.hypot(yy - inj_y, xx - inj_x) <= float(config["fit_radius_px"])
    base = np.where(local_display, base, np.nan)
    injected = np.where(local_display, injected, np.nan)
    delta = np.where(local_display, delta, np.nan)

    y1, y2 = max(0, inj_y - radius), min(base.shape[0], inj_y + radius + 1)
    x1, x2 = max(0, inj_x - radius), min(base.shape[1], inj_x + radius + 1)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].imshow(products["spatial_psf"], origin="lower")
    axes[0, 0].set_title("Spatial PSF template")
    axes[0, 1].plot(products["diag_wavelengths"][products["zsl_diag"]], products["spectral_profile"])
    axes[0, 1].axvline(float(config["line_center_A"]), color="0.3", linestyle="--")
    axes[0, 1].set_title("Spectral line profile")
    axes[0, 1].set_xlabel("Wavelength [A]")

    snr_input = np.asarray([row["target_snr_input"] for row in products["rows"]])
    matched_snr = np.asarray([row["local_surface_matched_snr"] for row in products["rows"]])
    aperture_snr = np.asarray([row["local_surface_aperture_snr"] for row in products["rows"]])
    axes[0, 2].plot(snr_input, matched_snr, marker="s", label="matched filter")
    axes[0, 2].plot(snr_input, aperture_snr, marker="o", label="3x3 aperture")
    axes[0, 2].set_xlabel("Injected S/N")
    axes[0, 2].set_ylabel("Recovered S/N")
    axes[0, 2].set_title(f"Recovery ({products['recovery_mode']})")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(alpha=0.25)

    for axis, image, title in zip(
        axes[1],
        [base, injected, delta],
        ["Baseline local residual", "Nominal injected residual", "Signal-transfer delta"],
    ):
        crop = image[y1:y2, x1:x2]
        finite = crop[np.isfinite(crop)]
        if finite.size:
            if title.startswith("Signal"):
                limit = float(np.nanpercentile(np.abs(finite), 99))
                vmin, vmax = -limit, limit
            else:
                vmin, vmax = np.nanpercentile(finite, [1, 99])
        else:
            vmin, vmax = None, None
        axis.imshow(
            crop,
            origin="lower",
            extent=(x1, x2 - 1, y1, y2 - 1),
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm" if title.startswith("Signal") else None,
        )
        axis.scatter([inj_x], [inj_y], marker="x", color="black")
        axis.set_title(title)
        axis.set_xlabel("x [px]")
        axis.set_ylabel("y [px]")

    fig.suptitle(
        f"{config['run_id']} - Halpha injection through local-surface subtraction",
        fontsize=13,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    return path


def write_stage06_local_products(products, config, paths, *, save_plots=True, show_plots=False):
    run_paths = paths["paths"]
    run_paths.ensure_base_dirs()
    paths["template_fits"].parent.mkdir(parents=True, exist_ok=True)
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    header = _base_header(config, products)

    fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(products["template"].astype(np.float32), name="LOCAL_TEMPLATE"),
            fits.ImageHDU(products["spatial_psf"].astype(np.float32), name="SPATIAL_PSF"),
            fits.ImageHDU(products["spectral_profile"].astype(np.float32), name="SPECTRAL_PROFILE"),
            fits.ImageHDU(
                products["diag_wavelengths"][products["zsl_diag"]].astype(np.float64),
                name="WAVELENGTH",
            ),
        ]
    ).writeto(paths["template_fits"], overwrite=True)

    diagnostic_fits = None
    if bool(config.get("save_diagnostic_fits", True)):
        fits.HDUList(
            [
                fits.PrimaryHDU(header=header),
                fits.ImageHDU(products["base_target_diag"].astype(np.float32), name="BASE_TARGET_DIAG"),
                fits.ImageHDU(
                    products["nominal_injected_diag"].astype(np.float32), name="INJECTED_TARGET_DIAG"
                ),
                fits.ImageHDU(products["nominal_delta_diag"].astype(np.float32), name="DELTA_TARGET_DIAG"),
                fits.ImageHDU(products["diag_wavelengths"].astype(np.float64), name="WAVELENGTH"),
            ]
        ).writeto(paths["diagnostic_fits"], overwrite=True)
        diagnostic_fits = paths["diagnostic_fits"]

    write_csv(paths["results_csv"], products["rows"], list(products["rows"][0].keys()))
    nominal_row = products["rows"][products["nominal_index"]]
    geometry = products["geometry"]
    qc = {
        "run_id": config["run_id"],
        "target_name": config.get("target_name"),
        "stage": "stage06_local_surface_halpha_injection",
        "input_mode": config["input_mode"],
        "input_cube_fits": str(config["input_cube_fits"]),
        "input_shape": [int(value) for value in config["input_shape"]],
        "recovery_mode": products["recovery_mode"],
        "selection_source": products["selection_source"],
        "selected_indices": [int(value) for value in products["selected_indices"]],
        "injection_yx": list(map(int, geometry["injection_yx"])),
        "star_yx": [float(value) for value in geometry["star_yx"]],
        "reference_object_yx": list(map(int, geometry["reference_object_yx"]))
        if geometry["reference_object_yx"] is not None
        else None,
        "injection_geometry": geometry,
        "control_positions_yx": [list(map(int, value)) for value in products["control_positions_yx"]],
        "n_control_positions": len(products["control_positions_yx"]),
        "local_model_kind": config["local_model_kind"],
        "fit_radius_px": float(config["fit_radius_px"]),
        "mask_radius_px": float(config["mask_radius_px"]),
        "local_fit_sigma_clip": float(config["local_fit_sigma_clip"]),
        "local_fit_max_iter": int(config["local_fit_max_iter"]),
        "line_center_A": float(config["line_center_A"]),
        "line_fwhm_A": float(products["line_fwhm_A"]),
        "halpha_channel_indices_diag": [int(value) for value in products["line_indices"]],
        "diagnostic_wavelength_range_A": [
            float(products["diag_wavelengths"][0]),
            float(products["diag_wavelengths"][-1]),
        ],
        "template_integral": float(products["template_integral"]),
        "pixel_scale_arcsec": float(config["pixel_scale_arcsec"]),
        "spatial_psf_fwhm_arcsec": float(config["spatial_psf_fwhm_arcsec"]),
        "injection_snr_reference": config["injection_snr_reference"],
        "injection_flux_sigma_native": float(products["injection_flux_sigma_native"]),
        "injection_flux_sigma_by_reference": products["injection_flux_sigma_by_reference"],
        "aperture_noise_source": products["aperture_noise_source"],
        "matched_noise_source": products["matched_noise_source"],
        "nominal_target_snr_input": float(nominal_row["target_snr_input"]),
        "nominal_total_line_flux_native": float(nominal_row["injected_total_line_flux_native"]),
        "nominal_row": nominal_row,
        "results_csv": str(paths["results_csv"]),
        "template_fits": str(paths["template_fits"]),
        "diagnostic_fits": str(diagnostic_fits) if diagnostic_fits else None,
    }
    write_json(paths["qc_json"], qc)

    truth = {
        "run_id": config["run_id"],
        "target_name": config.get("target_name"),
        "line": "Halpha",
        "recovery_method": "local_surface_subtraction",
        "recovery_mode": products["recovery_mode"],
        "injection_yx": list(map(int, geometry["injection_yx"])),
        "control_positions_yx": [list(map(int, value)) for value in products["control_positions_yx"]],
        "selected_indices": [int(value) for value in products["selected_indices"]],
        "injection_snr_reference": config["injection_snr_reference"],
        "injection_flux_sigma_native": float(products["injection_flux_sigma_native"]),
        "nominal_target_snr_input": float(nominal_row["target_snr_input"]),
        "nominal_total_line_flux_native": float(nominal_row["injected_total_line_flux_native"]),
        "nominal_expected_combined_line_flux_native": float(
            nominal_row["expected_combined_line_flux_native"]
        ),
        "nominal_local_surface_matched_flux_native": float(
            nominal_row["local_surface_matched_flux_native"]
        ),
        "nominal_local_surface_matched_snr": float(nominal_row["local_surface_matched_snr"]),
        "nominal_local_surface_matched_flux_ratio": float(
            nominal_row["local_surface_matched_throughput"]
        ),
        "nominal_delta_matched_signal_transfer": float(
            nominal_row["delta_matched_signal_transfer"]
        ),
        "native_units_note": (
            "Fluxes are integrated Stage04b physical-like native units. "
            "A calibrated BUNIT is required before interpreting them as cgs fluxes."
        ),
    }
    write_json(paths["truth_json"], truth)

    plot_paths = []
    if save_plots:
        plot_paths.append(
            save_stage06_local_plot(paths["summary_plot"], products, config, show_plots=show_plots)
        )
    qc["plots"] = [str(path) for path in plot_paths]
    write_json(paths["qc_json"], qc)
    return {
        "results_csv": paths["results_csv"],
        "qc_json": paths["qc_json"],
        "truth_json": paths["truth_json"],
        "template_fits": paths["template_fits"],
        "diagnostic_fits": diagnostic_fits,
        "plots": plot_paths,
        "qc": qc,
        "truth": truth,
    }


def run_stage06_local(
    config=None,
    *,
    show_plots=False,
    save_plots=True,
    project_root=None,
    allow_run_id_mismatch=False,
):
    """Run local-surface injection recovery without overwriting PCA products."""

    if config is None:
        config = stage06_local_config_from_run(
            project_root=project_root,
            allow_run_id_mismatch=allow_run_id_mismatch,
        )
    else:
        config = dict(config)
    root = Path(project_root or config.get("project_root") or Path.cwd()).resolve()
    paths = stage06_local_paths(config["run_id"], root)
    stage04_paths = stage04b_paths(config["run_id"], root)
    stage04_cfg = stage04b_config_from_run(
        config["run_id"],
        project_root=root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    stage04_cfg.update(
        {
            "input_mode": config["input_mode"],
            "bad_wavelength_ranges_A": config.get("bad_wavelength_ranges_A", []),
        }
    )
    input_payload = load_stage04b_input(stage04_cfg, stage04_paths)
    selected_indices, selection_source = _selected_cube_indices(
        config,
        input_payload["cubes"].shape[0],
        paths=paths,
    )
    config["input_cube_fits"] = str(input_payload["input_cube_fits"])
    config["input_shape"] = list(map(int, input_payload["cubes"].shape))
    products = compute_stage06_local_products(
        input_payload["cubes"],
        input_payload["wavelengths"],
        input_payload["med_pix_stack"],
        config,
        selected_indices=selected_indices,
        selection_source=selection_source,
    )
    written = write_stage06_local_products(
        products,
        config,
        paths,
        save_plots=save_plots,
        show_plots=show_plots,
    )
    return {**written, "products": products}


__all__ = [
    "STAGE06_LOCAL_DEFAULTS",
    "aperture_line_flux",
    "combined_local_surface_residual",
    "compute_stage06_local_products",
    "cube_center_yx",
    "inject_physical_template",
    "local_surface_control_positions",
    "make_signal_template",
    "matched_filter_flux",
    "resolve_injection_geometry",
    "run_stage06_local",
    "save_stage06_local_plot",
    "separation_pa_from_center",
    "stage06_local_config_from_run",
    "stage06_local_paths",
    "validate_yx",
    "write_stage06_local_products",
    "yx_from_center_radius_pa",
]
