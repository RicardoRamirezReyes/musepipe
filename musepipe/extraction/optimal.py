"""Horne-style optimal extraction on MUSE cube residuals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from ..apertures import same_radius_control_positions
from ..parallel import run_channel_chunks
from ..psf import evaluate_psf_model, scaled_psf_model as _scaled_psf_model
from ..stats import robust_sigma, robust_sigma_axis0
from .aperture import (
    FLAG_CLIPPED,
    annulus_background_spectrum,
    azimuthal_background_spectrum,
    local_plane_background_spectrum,
    aperture_correction_from_psf,
    channel_flags,
    sha256_file,
)
from .product import FORMAT_VERSION, SpectrumProduct


@dataclass(frozen=True)
class OptimalExtraction:
    product: SpectrumProduct
    raw_flux: np.ndarray
    raw_flux_err: np.ndarray
    raw_flux_err_emp: np.ndarray
    raw_variance: np.ndarray
    controls_yx: list[tuple[int, int]]
    control_spectra: np.ndarray
    clip_fraction: np.ndarray
    rejection_map: np.ndarray
    covariance_factor: np.ndarray
    error_mode: str
    apcorr_mode: str
    norm_radius_px: float
    variant: str
    # Controls processed exactly like the object (same local background
    # reference, same apcorr): the physical scale of ``product.flux``
    # (D1 v2 §3.1 "control = object").
    control_spectra_cal: np.ndarray | None = None
    bkg_mode: str = "none"


def circular_window_indices(shape, center_yx, radius_px):
    ny, nx = map(int, shape)
    cy, cx = map(float, center_yx)
    radius = float(radius_px)
    half = int(math.ceil(radius))
    y1 = max(0, int(math.floor(cy)) - half)
    y2 = min(ny, int(math.floor(cy)) + half + 2)
    x1 = max(0, int(math.floor(cx)) - half)
    x2 = min(nx, int(math.floor(cx)) + half + 2)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    return yy[mask].astype(int), xx[mask].astype(int)


def normalized_psf_window(psf_model, wavelength_A, center_yx, ypix, xpix):
    p = evaluate_psf_model(
        psf_model,
        float(wavelength_A),
        np.asarray(ypix, dtype=np.float64) - float(center_yx[0]),
        np.asarray(xpix, dtype=np.float64) - float(center_yx[1]),
    ).astype(np.float64)
    p[~np.isfinite(p)] = 0.0
    p[p < 0] = 0.0
    norm = float(np.sum(p))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("PSF window has invalid normalization.")
    return p / norm


def covariance_factor_for_npix(npix_eff, covariance_factor_box3=1.0):
    """Linearly interpolate covariance inflation between one pixel and box3."""

    vals = np.asarray(npix_eff, dtype=np.float64)
    box3 = float(covariance_factor_box3)
    if not np.isfinite(box3) or box3 <= 0:
        box3 = 1.0
    t = np.clip((vals - 1.0) / 8.0, 0.0, 1.0)
    return 1.0 + t * (box3 - 1.0)


def _channel_estimate(data, variance, p, valid, *, clip_sigma=4.0, clip_max_iter=2):
    data = np.asarray(data, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    mask = valid.copy()
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return np.nan, np.nan, np.nan, 1.0, np.zeros_like(valid, dtype=bool)

    flux = np.nan
    raw_var = np.nan
    max_iter = max(0, int(clip_max_iter))
    for iteration in range(max_iter + 1):
        denom = np.sum((p[mask] ** 2) / variance[mask])
        if not np.isfinite(denom) or denom <= 0:
            return np.nan, np.nan, np.nan, 1.0, valid.copy()
        flux = float(np.sum(p[mask] * data[mask] / variance[mask]) / denom)
        raw_var = float(1.0 / denom)
        if clip_sigma is None or iteration >= max_iter:
            break
        z = np.zeros_like(data, dtype=np.float64)
        z[valid] = (data[valid] - flux * p[valid]) / np.sqrt(variance[valid])
        new_mask = valid & (np.abs(z) <= float(clip_sigma))
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    clipped = valid & ~mask
    p_kept = p[mask]
    npix_eff = np.nan
    if p_kept.size and np.sum(p_kept**2) > 0:
        npix_eff = float((np.sum(p_kept) ** 2) / np.sum(p_kept**2))
    frac_clip = float(np.count_nonzero(clipped) / max(n_valid, 1))
    return flux, raw_var, npix_eff, frac_clip, clipped


def estimate_variance_cube(cube_zyx):
    cube = np.asarray(cube_zyx, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (nz,ny,nx), got {cube.shape}.")
    out = np.empty_like(cube, dtype=np.float64)
    for i in range(cube.shape[0]):
        sigma = robust_sigma(cube[i])
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        out[i] = sigma**2
    return out


def optimal_raw_spectrum(
    cube_zyx,
    variance_zyx,
    wave_A,
    center_yx,
    psf_model,
    *,
    window_radius_px=8.0,
    clip_sigma=4.0,
    clip_max_iter=2,
    n_jobs=1,
    bkg_spectrum=None,
):
    cube = np.asarray(cube_zyx, dtype=np.float64)
    variance = np.asarray(variance_zyx, dtype=np.float64)
    wave = np.asarray(wave_A, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (nz,ny,nx), got {cube.shape}.")
    if variance.shape != cube.shape:
        raise ValueError("variance_zyx shape must match cube_zyx.")
    if wave.ndim != 1 or wave.size != cube.shape[0]:
        raise ValueError("wave_A must be 1D and match cube spectral length.")
    if bkg_spectrum is not None:
        bkg_spectrum = np.asarray(bkg_spectrum, dtype=np.float64)
        if bkg_spectrum.shape != wave.shape:
            raise ValueError("bkg_spectrum must match wave_A length.")

    nz, ny, nx = cube.shape
    ypix, xpix = circular_window_indices((ny, nx), center_yx, window_radius_px)
    flux = np.full(nz, np.nan, dtype=np.float64)
    raw_var = np.full(nz, np.nan, dtype=np.float64)
    npix_eff = np.full(nz, np.nan, dtype=np.float64)
    clip_fraction = np.zeros(nz, dtype=np.float64)
    chunk_rejection = {}

    def _estimate_range(z0, z1):
        # Per-channel work identical to the serial loop; the rejection counts
        # accumulate in a per-chunk map (integer-valued, so the final sum is
        # exact regardless of chunk order).
        local_map = np.zeros((ny, nx), dtype=np.float64)
        for z in range(z0, z1):
            data = cube[z, ypix, xpix]
            if bkg_spectrum is not None and np.isfinite(bkg_spectrum[z]):
                # Local background reference: subtracting a per-channel scalar
                # is separable from the Horne estimator (D1 v2 §3.1).
                data = data - bkg_spectrum[z]
            var = variance[z, ypix, xpix]
            p = normalized_psf_window(psf_model, wave[z], center_yx, ypix, xpix)
            valid = np.isfinite(data) & np.isfinite(var) & (var > 0) & np.isfinite(p) & (p > 0)
            f, v, neff, frac, clipped = _channel_estimate(
                data,
                var,
                p,
                valid,
                clip_sigma=clip_sigma,
                clip_max_iter=clip_max_iter,
            )
            flux[z] = f
            raw_var[z] = v
            npix_eff[z] = neff
            clip_fraction[z] = frac
            if np.any(clipped):
                np.add.at(local_map, (ypix[clipped], xpix[clipped]), 1.0)
        chunk_rejection[z0] = local_map

    run_channel_chunks(_estimate_range, nz, n_jobs=n_jobs)
    rejection_map = np.zeros((ny, nx), dtype=np.float64)
    for z0 in sorted(chunk_rejection):
        rejection_map += chunk_rejection[z0]

    return {
        "flux": flux,
        "variance": raw_var,
        "npix_eff": npix_eff,
        "clip_fraction": clip_fraction,
        "rejection_map": rejection_map,
    }


#: Modos de fondo local. `annulus` es el historico y el que sigue por defecto;
#: `azimuthal` se anade en 2026-07-27 tras medir que el anillo centrado en el
#: compañero atraviesa el gradiente del halo y su mediana queda sesgada por el
#: lado interior. Cual es mejor NO es universal: medido en los dos objetos del
#: proyecto el cambio va en direcciones opuestas (ver el informe del 2026-07-27),
#: asi que se elige por config y el defecto no se mueve solo.
BACKGROUND_MODES = ("annulus", "azimuthal", "local_plane")


def local_background_spectrum(
    cube,
    center_yx,
    star_yx,
    *,
    mode="annulus",
    annulus_px=None,
    azimuthal_width_px=3.0,
    azimuthal_exclude_px=10.0,
    plane_fit_radius_px=14.0,
    plane_mask_radius_px=3.0,
):
    """El fondo local de una posicion, en el modo pedido, o None si no hay.

    Vive aqui y no repetido en el objeto y en los controles porque el principio
    `control = objeto` exige que a los dos se les aplique EXACTAMENTE el mismo
    estimador: en `azimuthal` el radio estelar se toma de cada posicion, que por
    construccion es el mismo para el compañero y sus controles, y lo que se
    excluye es el entorno de esa posicion y no siempre el del compañero.
    """
    mode = str(mode or "annulus").lower()
    if mode not in BACKGROUND_MODES:
        raise ValueError(f"background mode must be one of {BACKGROUND_MODES}, got {mode!r}")
    if mode == "local_plane":
        return local_plane_background_spectrum(
            cube, center_yx,
            fit_radius_px=float(plane_fit_radius_px),
            mask_radius_px=float(plane_mask_radius_px),
        )
    if mode == "azimuthal":
        radius = float(np.hypot(float(center_yx[0]) - float(star_yx[0]),
                                float(center_yx[1]) - float(star_yx[1])))
        return azimuthal_background_spectrum(
            cube, star_yx, radius,
            width_px=float(azimuthal_width_px),
            exclude_yx=center_yx,
            exclude_radius=float(azimuthal_exclude_px),
        )
    if annulus_px is None:
        return None
    return annulus_background_spectrum(
        cube, center_yx, annulus_px[0], annulus_px[1],
        exclude_yx=star_yx,
        exclude_radius=float(annulus_px[2]) if len(annulus_px) > 2 else 30.0,
    )


def control_optimal_spectra(
    cube_zyx,
    variance_zyx,
    wave_A,
    object_yx,
    star_yx,
    psf_model,
    *,
    window_radius_px=8.0,
    clip_sigma=4.0,
    clip_max_iter=2,
    n_controls=8,
    exclude_angle_deg=25.0,
    n_jobs=1,
    local_bkg_annulus_px=None,
    background_mode="annulus",
    azimuthal_width_px=3.0,
    azimuthal_exclude_px=10.0,
    plane_fit_radius_px=14.0,
    plane_mask_radius_px=3.0,
):
    cube = np.asarray(cube_zyx, dtype=np.float64)
    _, ny, nx = cube.shape
    controls = same_radius_control_positions(
        object_yx,
        star_yx,
        ny,
        nx,
        n_positions=int(n_controls),
        exclude_angle_deg=float(exclude_angle_deg),
        margin_px=int(math.ceil(window_radius_px)) + 1,
    )
    spectra = []
    for center in controls:
        bkg = local_background_spectrum(
            cube, center, star_yx,
            mode=background_mode,
            annulus_px=local_bkg_annulus_px,
            azimuthal_width_px=azimuthal_width_px,
            azimuthal_exclude_px=azimuthal_exclude_px,
            plane_fit_radius_px=plane_fit_radius_px,
            plane_mask_radius_px=plane_mask_radius_px,
        )
        raw = optimal_raw_spectrum(
            cube,
            variance_zyx,
            wave_A,
            center,
            psf_model,
            window_radius_px=window_radius_px,
            clip_sigma=clip_sigma,
            clip_max_iter=clip_max_iter,
            n_jobs=n_jobs,
            bkg_spectrum=bkg,
        )
        spectra.append(raw["flux"])
    if not spectra:
        return controls, np.empty((0, cube.shape[0]), dtype=np.float64)
    return controls, np.asarray(spectra, dtype=np.float64)


def make_optimal_product(
    cube_zyx,
    wave_A,
    object_yx,
    psf_model,
    *,
    run_id: str,
    input_cube_path: str | Path,
    input_cube_sha: str | None = None,
    input_cube_source: str | None = None,
    variant: str = "ls",
    star_yx=None,
    variance_zyx=None,
    stat_factor_spaxel: float = 1.0,
    covariance_factor_box3: float = 1.0,
    stat_status: str = "unknown",
    error_mode: str = "auto",
    aperture_correction: str = "auto",
    growth_curve=None,
    wframe: str = "topocentric",
    bunit: str = "",
    window_radius_px: float = 8.0,
    clip_sigma: float = 4.0,
    clip_max_iter: int = 2,
    bad_windows_A: Sequence[Sequence[float]] = (),
    skyline_windows_A: Sequence[Sequence[float]] = (),
    interpolated_windows_A: Sequence[Sequence[float]] = (),
    good_mask=None,
    bad_mask=None,
    n_controls: int = 8,
    exclude_angle_deg: float = 25.0,
    clip_flag_fraction: float = 0.05,
    n_jobs: int = 1,
    local_bkg_annulus_px: Sequence[float] | None = None,
    background_mode: str = "annulus",
    azimuthal_width_px: float = 3.0,
    azimuthal_exclude_px: float = 10.0,
    plane_fit_radius_px: float = 14.0,
    plane_mask_radius_px: float = 3.0,
) -> OptimalExtraction:
    cube = np.asarray(cube_zyx, dtype=np.float64)
    wave = np.asarray(wave_A, dtype=np.float64)
    if variance_zyx is None:
        variance = estimate_variance_cube(cube)
        variance_source = "empirical_unit_weight"
    else:
        variance = np.asarray(variance_zyx, dtype=np.float64)
        variance_source = "stat"
    if variance.shape != cube.shape:
        raise ValueError("variance_zyx shape must match cube_zyx.")

    stat_factor = float(stat_factor_spaxel)
    if not np.isfinite(stat_factor) or stat_factor <= 0:
        stat_factor = 1.0
    variance = variance * stat_factor
    # Same local background reference for object and controls: re-references
    # any residual pedestal (e.g. the stage04b local-surface residual for the
    # LS variant) so all methods share the flux convention (D1 v2 §3.1).
    object_bkg = local_background_spectrum(
        cube, object_yx, star_yx,
        mode=background_mode,
        annulus_px=local_bkg_annulus_px,
        azimuthal_width_px=azimuthal_width_px,
        azimuthal_exclude_px=azimuthal_exclude_px,
        plane_fit_radius_px=plane_fit_radius_px,
        plane_mask_radius_px=plane_mask_radius_px,
    )
    raw = optimal_raw_spectrum(
        cube,
        variance,
        wave,
        object_yx,
        psf_model,
        window_radius_px=window_radius_px,
        clip_sigma=clip_sigma,
        clip_max_iter=clip_max_iter,
        n_jobs=n_jobs,
        bkg_spectrum=object_bkg,
    )
    cov_factor = covariance_factor_for_npix(raw["npix_eff"], covariance_factor_box3)
    raw_variance = raw["variance"] * cov_factor

    if star_yx is None:
        sigma = robust_sigma(raw["flux"])
        raw_flux_err_emp = np.full(wave.size, sigma, dtype=np.float64)
        controls_yx = []
        control_spectra = np.empty((0, wave.size), dtype=np.float64)
    else:
        controls_yx, control_spectra = control_optimal_spectra(
            cube,
            variance,
            wave,
            object_yx,
            star_yx,
            psf_model,
            window_radius_px=window_radius_px,
            clip_sigma=clip_sigma,
            clip_max_iter=clip_max_iter,
            n_controls=n_controls,
            exclude_angle_deg=exclude_angle_deg,
            n_jobs=n_jobs,
            local_bkg_annulus_px=local_bkg_annulus_px,
        )
        if control_spectra.shape[0] >= 2:
            raw_flux_err_emp = robust_sigma_axis0(control_spectra)
        else:
            sigma = robust_sigma(raw["flux"])
            raw_flux_err_emp = np.full(wave.size, sigma, dtype=np.float64)

    requested_error_mode = str(error_mode or "auto").lower()
    stat_usable = (
        variance_zyx is not None
        and requested_error_mode != "empirical"
        and str(stat_status).lower() != "red"
    )
    if stat_usable:
        raw_flux_err = np.sqrt(np.clip(raw_variance, 0.0, np.inf))
        mode = "stat"
    else:
        raw_flux_err = np.asarray(raw_flux_err_emp, dtype=np.float64)
        mode = "empirical"

    aperture = {"kind": "circle", "radius_px": float(window_radius_px), "name": f"optimal_r{float(window_radius_px):g}"}
    apcorr, apcorr_mode, norm_radius = aperture_correction_from_psf(
        wave,
        aperture,
        psf_model,
        center_yx=object_yx,
        correction_mode=aperture_correction,
        growth_curve=growth_curve,
    )
    clipped_channels = raw["clip_fraction"] > float(clip_flag_fraction)
    flags = channel_flags(
        wave,
        bad_windows_A=bad_windows_A,
        skyline_windows_A=skyline_windows_A,
        interpolated_windows_A=interpolated_windows_A,
        good_mask=good_mask,
        bad_mask=bad_mask,
        clipped_mask=clipped_channels,
    )

    input_cube_path = Path(input_cube_path)
    if input_cube_sha is None:
        input_cube_sha = sha256_file(input_cube_path) if input_cube_path.exists() else ""
    label = f"optimal_{variant}_r{float(window_radius_px):g}"
    if str(background_mode).lower() == "local_plane":
        bkg_mode = f"local_plane_r{float(plane_fit_radius_px):g}_m{float(plane_mask_radius_px):g}"
    elif str(background_mode).lower() == "azimuthal":
        sep_px = float(np.hypot(float(object_yx[0]) - float(star_yx[0]),
                                float(object_yx[1]) - float(star_yx[1])))
        bkg_mode = f"azimuthal_r{sep_px:.1f}_w{float(azimuthal_width_px):g}"
    elif local_bkg_annulus_px is not None:
        bkg_mode = f"annulus_{float(local_bkg_annulus_px[0]):g}_{float(local_bkg_annulus_px[1]):g}"
    else:
        bkg_mode = "none"
    header = {
        "FORMATV": FORMAT_VERSION,
        "METHOD": "optimal",
        "RUNID": str(run_id),
        "SRCPOS_Y": float(object_yx[0]),
        "SRCPOS_X": float(object_yx[1]),
        "APERTURE": label,
        "WFRAME": str(wframe),
        "INCUBE": str(input_cube_path),
        "INCUBESH": str(input_cube_sha),
        "NORMRAD": float(norm_radius),
        "BUNIT": str(bunit or ""),
        "ERRMODE": mode,
        "APCMODE": apcorr_mode,
        "STATFAC": float(stat_factor),
        "COVFAC": float(np.nanmedian(cov_factor)),
        "WINDOW": float(window_radius_px),
        "CLIPSIG": -1.0 if clip_sigma is None else float(clip_sigma),
        "CLIPIT": int(clip_max_iter),
        "PNORM": "window_renorm_plus_apcorr",
        "VARIANT": str(variant),
        "VARSRC": variance_source,
        "BKGMODE": bkg_mode,
        "SCALEREF": ("empirical_total_flux" if "empirical_total" in str(apcorr_mode)
                     else "normrad_total_flux"),
    }
    # De que cubo sale el espectro, dicho por el producto y no deducido del
    # hash: cuando C1b resta la primaria en cada exposicion, el psfsub consume
    # SU cubo y `INCUBESH` deja de coincidir con el de los otros metodos a
    # proposito. D1 solo puede distinguir eso de una mezcla de cubos erronea si
    # el producto lo declara (`validate_product_set`). Sin valor no se estampa:
    # los productos que no tienen nada especial que decir no cambian.
    if input_cube_source:
        header["CUBESRC"] = str(input_cube_source)
    product = SpectrumProduct(
        wave_A=wave,
        flux=raw["flux"] * apcorr,
        flux_err=raw_flux_err * apcorr,
        flux_err_emp=raw_flux_err_emp * apcorr,
        apcorr=apcorr,
        npix_eff=raw["npix_eff"],
        flags=flags.astype(np.int32),
        header=header,
    )
    product.validate()
    return OptimalExtraction(
        product=product,
        raw_flux=raw["flux"],
        raw_flux_err=raw_flux_err,
        raw_flux_err_emp=raw_flux_err_emp,
        raw_variance=raw_variance,
        controls_yx=controls_yx,
        control_spectra=control_spectra,
        clip_fraction=raw["clip_fraction"],
        rejection_map=raw["rejection_map"],
        covariance_factor=cov_factor,
        error_mode=mode,
        apcorr_mode=apcorr_mode,
        norm_radius_px=float(norm_radius),
        variant=str(variant),
        control_spectra_cal=control_spectra * apcorr[None, :],
        bkg_mode=bkg_mode,
    )


def psf_image(shape, wavelength_A, center_yx, psf_model):
    yy, xx = np.indices(shape, dtype=np.float64)
    return evaluate_psf_model(
        psf_model,
        float(wavelength_A),
        yy - float(center_yx[0]),
        xx - float(center_yx[1]),
    )


def fit_primary_psf_model_cube(
    cube_zyx,
    wave_A,
    primary_yx,
    psf_model,
    *,
    variance_zyx=None,
    fit_radius_px: float | None = None,
    exclude_centers_yx=(),
    exclude_radius_px: float = 8.0,
    n_jobs: int = 1,
) -> tuple[np.ndarray, dict]:
    model, meta, _amplitudes, _backgrounds, _n_fit = fit_primary_psf_amplitudes(
        cube_zyx,
        wave_A,
        primary_yx,
        psf_model,
        variance_zyx=variance_zyx,
        fit_radius_px=fit_radius_px,
        exclude_centers_yx=exclude_centers_yx,
        exclude_radius_px=exclude_radius_px,
        n_jobs=n_jobs,
    )
    return model, meta


def fit_primary_psf_amplitudes(
    cube_zyx,
    wave_A,
    primary_yx,
    psf_model,
    *,
    variance_zyx=None,
    fit_radius_px: float | None = None,
    exclude_centers_yx=(),
    exclude_radius_px: float = 8.0,
    n_jobs: int = 1,
):
    """Igual que ``fit_primary_psf_model_cube``, pero devolviendo los vectores.

    Existe porque C1b resta la primaria **exposición a exposición** y necesita
    la amplitud por canal de cada una: es el flujo de la estrella en esa
    exposición, y con él se comprueba que la resta no se ha comido nada. La
    matemática es la misma, sin una línea distinta; ``fit_primary_psf_model_cube``
    se limita a tirar los vectores, que es lo que hacía antes.

    Devuelve ``(model, meta, amplitudes, backgrounds, n_fit)``. El modelo NO
    lleva el fondo sumado, igual que antes.
    """

    cube = np.asarray(cube_zyx, dtype=np.float64)
    wave = np.asarray(wave_A, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (nz,ny,nx), got {cube.shape}.")
    if wave.size != cube.shape[0]:
        raise ValueError("wave_A must match cube spectral length.")
    variance = None if variance_zyx is None else np.asarray(variance_zyx, dtype=np.float64)
    if variance is not None and variance.shape != cube.shape:
        raise ValueError("variance_zyx shape must match cube_zyx.")

    nz, ny, nx = cube.shape
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    py, px = map(float, primary_yx)
    fit_radius = float(fit_radius_px or psf_model.get("norm_radius_px", 25.0))
    fit_mask = (yy - py) ** 2 + (xx - px) ** 2 <= fit_radius**2
    for center in exclude_centers_yx or ():
        if center is None:
            continue
        cy, cx = map(float, center)
        fit_mask &= (yy - cy) ** 2 + (xx - cx) ** 2 > float(exclude_radius_px) ** 2

    model = np.zeros_like(cube, dtype=np.float64)
    amplitudes = np.full(nz, np.nan, dtype=np.float64)
    backgrounds = np.full(nz, np.nan, dtype=np.float64)
    n_fit = np.zeros(nz, dtype=np.int32)

    def _fit_range(z0, z1):
        # Per-channel work identical to the serial loop; disjoint output slots.
        for z in range(z0, z1):
            psf = psf_image((ny, nx), wave[z], primary_yx, psf_model)
            data = cube[z]
            valid = fit_mask & np.isfinite(data) & np.isfinite(psf)
            if variance is not None:
                valid &= np.isfinite(variance[z]) & (variance[z] > 0)
                weight = 1.0 / variance[z][valid]
            else:
                weight = np.ones(np.count_nonzero(valid), dtype=np.float64)
            if np.count_nonzero(valid) < 3:
                continue
            a = np.column_stack([psf[valid], np.ones(np.count_nonzero(valid), dtype=np.float64)])
            sw = np.sqrt(weight)
            try:
                coeff, *_ = np.linalg.lstsq(a * sw[:, None], data[valid] * sw, rcond=None)
            except np.linalg.LinAlgError:
                continue
            amp = float(coeff[0])
            bg = float(coeff[1])
            amplitudes[z] = amp
            backgrounds[z] = bg
            n_fit[z] = int(np.count_nonzero(valid))
            model[z] = amp * psf

    run_channel_chunks(_fit_range, nz, n_jobs=n_jobs)
    meta = {
        "amplitude_median": None if not np.any(np.isfinite(amplitudes)) else float(np.nanmedian(amplitudes)),
        "background_median": None if not np.any(np.isfinite(backgrounds)) else float(np.nanmedian(backgrounds)),
        "n_fit_median": int(np.nanmedian(n_fit)) if n_fit.size else 0,
        "fit_radius_px": float(fit_radius),
        "exclude_radius_px": float(exclude_radius_px),
    }
    return model, meta, amplitudes, backgrounds, n_fit


# Re-exportado desde `musepipe.psf`: esta copia solo escalaba coeficientes
# Moffat, asi que con el modelo psfao (el que C1 elige cuando hay maoppy) la
# prueba de sensibilidad de C3 salia 0.0% por no perturbar nada.
scaled_psf_model = _scaled_psf_model


__all__ = [
    "OptimalExtraction",
    "circular_window_indices",
    "control_optimal_spectra",
    "covariance_factor_for_npix",
    "estimate_variance_cube",
    "fit_primary_psf_amplitudes",
    "fit_primary_psf_model_cube",
    "make_optimal_product",
    "normalized_psf_window",
    "optimal_raw_spectrum",
    "psf_image",
    "scaled_psf_model",
]
