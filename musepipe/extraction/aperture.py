"""Aperture extraction helpers for the frozen spectrum product format."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np

from ..apertures import aperture_weights, same_radius_control_positions
from ..localfit import fit_local_surface_2d
from ..psf import evaluate_psf_model
from ..stats import robust_sigma, robust_sigma_axis0
from .product import FORMAT_VERSION, SpectrumProduct


FLAG_BAD_WINDOW = 1
FLAG_SKYLINE = 2
FLAG_INTERPOLATED = 4
FLAG_CLIPPED = 8


@dataclass(frozen=True)
class ApertureExtraction:
    product: SpectrumProduct
    raw_flux: np.ndarray
    raw_flux_err: np.ndarray
    raw_flux_err_emp: np.ndarray
    controls_yx: list[tuple[int, int]]
    control_spectra: np.ndarray
    aperture: dict
    aperture_label: str
    error_mode: str
    apcorr_mode: str
    norm_radius_px: float
    # Controls processed exactly like the object (same background, same
    # apcorr), i.e. on the physical scale of ``product.flux`` (D1 v2 §3.1).
    control_spectra_cal: np.ndarray | None = None
    bkg_mode: str = "none"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def aperture_label(aperture: dict) -> str:
    if aperture.get("name"):
        return str(aperture["name"])
    kind = str(aperture.get("kind", "box"))
    if kind == "box":
        return f"box{int(aperture.get('size', 3))}"
    if kind == "circle":
        return f"r{float(aperture['radius_px']):g}"
    if kind == "gaussian":
        return f"gauss{float(aperture['sigma_px']):g}"
    return kind


def default_apertures() -> list[dict]:
    return [
        {"name": "box3", "kind": "box", "size": 3},
        {"name": "box5", "kind": "box", "size": 5},
    ]


def _as_cube(cube_zyx, name="cube") -> np.ndarray:
    cube = np.asarray(cube_zyx, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected {name} with shape (nz,ny,nx), got {cube.shape}.")
    return cube


def _npix_eff(cube_zyx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(cube_zyx) & (weights[None, :, :] > 0)
    sumw = np.sum(weights[None, :, :] * valid, axis=(1, 2))
    sumw2 = np.sum((weights[None, :, :] ** 2) * valid, axis=(1, 2))
    out = np.full(cube_zyx.shape[0], np.nan, dtype=np.float64)
    good = sumw2 > 0
    out[good] = (sumw[good] ** 2) / sumw2[good]
    return out


def aperture_spectrum(cube_zyx, center_yx, aperture: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted-sum spectrum and per-channel effective pixel count."""

    cube = _as_cube(cube_zyx)
    _, ny, nx = cube.shape
    weights = aperture_weights(ny, nx, center_yx, aperture)
    weighted = cube * weights[None, :, :]
    with np.errstate(invalid="ignore"):
        flux = np.nansum(weighted, axis=(1, 2)).astype(np.float64)
    npix_eff = _npix_eff(cube, weights)
    flux[~np.isfinite(npix_eff)] = np.nan
    return flux, npix_eff


def annulus_background_spectrum(cube_zyx, center_yx, r_in, r_out, *, exclude_yx=None, exclude_radius=0.0):
    """Per-channel local background = median of a source-free annulus.

    Used for the wings-intact aperture-correction path: subtracting a distant
    annulus (rather than a local surface, stage04b) preserves the companion's
    PSF wings so the PSF growth-curve aperture correction stays self-consistent
    (box3<box5). Excludes a region around ``exclude_yx`` (the primary)."""

    cube = _as_cube(cube_zyx)
    _, ny, nx = cube.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - float(center_yx[0]), xx - float(center_yx[1]))
    mask = (r >= float(r_in)) & (r <= float(r_out))
    if exclude_yx is not None and float(exclude_radius) > 0:
        mask &= np.hypot(yy - float(exclude_yx[0]), xx - float(exclude_yx[1])) > float(exclude_radius)
    if not mask.any():
        return np.zeros(cube.shape[0], dtype=np.float64)
    vals = cube[:, mask]
    with np.errstate(all="ignore"):
        return np.nanmedian(vals, axis=1).astype(np.float64)


def azimuthal_background_spectrum(
    cube_zyx,
    star_yx,
    radius_px,
    *,
    width_px=3.0,
    exclude_yx=None,
    exclude_radius=10.0,
):
    """Fondo por canal = mediana del anillo centrado en la PRIMARIA, al radio dado.

    La alternativa a `annulus_background_spectrum` cuando el fondo es el halo de
    una estrella brillante. Un anillo centrado en el **compañero** atraviesa el
    gradiente radial del halo —a 71 px de la primaria cae un factor ~1.6 cada 10
    px—; uno centrado en la **primaria**, a la misma separación, se mantiene a
    halo constante por construcción.

    Ojo con el signo del sesgo del anillo, que no es el que parece: la curvatura
    del arco mete más área por fuera, así que la mediana de su radio *estelar*
    cae 0.7 px MÁS LEJOS de la estrella y el anillo **sub**-estima. La intuición
    de que el lado interior tira la mediana hacia arriba es falsa.

    A cambio esto asume que el halo es **azimutalmente simétrico**, y no lo es: la
    PSF de AO tiene speckles y spikes. Medido en los dos objetos del proyecto el
    cambio va en direcciones opuestas y **el anillo sale mejor que este** — ver
    `reports/20260727/sesgo_anillo_y_ventana_2026-07-27.md`. Por eso es una
    opción seleccionable y no el comportamiento por defecto.

    `exclude_radius` quita del anillo el entorno del propio compañero
    (`exclude_yx`), que si no contaminaría su propio fondo.
    """

    cube = _as_cube(cube_zyx)
    _, ny, nx = cube.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r_star = np.hypot(yy - float(star_yx[0]), xx - float(star_yx[1]))
    mask = np.abs(r_star - float(radius_px)) <= float(width_px)
    if exclude_yx is not None and float(exclude_radius) > 0:
        mask &= np.hypot(yy - float(exclude_yx[0]), xx - float(exclude_yx[1])) > float(exclude_radius)
    if not mask.any():
        return np.zeros(cube.shape[0], dtype=np.float64)
    vals = cube[:, mask]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # Un canal enteramente NaN (hueco del laser) da un aviso por cada uno:
        # se devuelve NaN, que es lo correcto, sin llenar la salida de ruido.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(vals, axis=1).astype(np.float64)


def local_plane_background_spectrum(
    cube_zyx,
    center_yx,
    *,
    fit_radius_px=14.0,
    mask_radius_px=3.0,
    exclude_yx=None,
    exclude_radius_px=3.0,
    model_kind="plane",
):
    """Fondo por canal = plano local ajustado alrededor de la fuente y **evaluado en ella**.

    La diferencia con los otros dos estimadores no es el tamaño de la región: es
    *dónde se lee el resultado*. Una mediana de anillo devuelve el nivel **del
    anillo**, que está en otro sitio; el plano se ajusta al entorno y luego se
    evalúa en la posición de la fuente, así que sigue el gradiente del halo en vez
    de promediarlo. Medido en la banda azul de ROXs 12 b, por píxel: el plano da
    4.179 donde el anillo da 4.280 y el valor local es ~4.126.

    Es además el mismo estimador que ya usan **04b** (que resta esta superficie del
    cubo entero) y **C4** (que lleva el plano dentro de su matriz de diseño), así
    que ponerlo aquí hace a `optimal_ls` consistente con ellos en vez de ser el
    único con una receta propia.

    Lo que sigue sin poder describir es una **asimetría local** —el compañero cae
    en un mínimo del patrón de speckles—: un plano solo puede seguir un gradiente
    lineal. Ver `reports/20260727/sesgo_anillo_y_ventana_2026-07-27.md`.

    `mask_radius_px` excluye del ajuste la propia fuente; `exclude_yx` cualquier
    otra que caiga dentro del radio de ajuste.
    """

    cube = _as_cube(cube_zyx)
    # `fit_local_surface_2d` trunca el centro a entero para armar la region; se
    # lee el modelo en ESE pixel para que el punto de evaluacion y el origen del
    # ajuste sean el mismo.
    yc_i, xc_i = int(float(center_yx[0])), int(float(center_yx[1]))
    extra = None if exclude_yx is None else [(float(exclude_yx[0]), float(exclude_yx[1]))]
    out = np.full(cube.shape[0], np.nan, dtype=np.float64)
    for z in range(cube.shape[0]):
        model, _n_good = fit_local_surface_2d(
            cube[z],
            float(center_yx[0]),
            float(center_yx[1]),
            fit_radius_px=float(fit_radius_px),
            mask_radius_px=float(mask_radius_px),
            model_kind=str(model_kind),
            extra_exclusion_yx=extra,
            extra_exclusion_radius_px=float(exclude_radius_px),
        )
        out[z] = model[yc_i, xc_i]
    return out


def aperture_stat_error(
    stat_zyx,
    center_yx,
    aperture: dict,
    *,
    stat_factor: float = 1.0,
    covariance_factor: float = 1.0,
) -> np.ndarray:
    """Propagate a variance cube through the same aperture weights."""

    stat = _as_cube(stat_zyx, name="stat")
    _, ny, nx = stat.shape
    weights = aperture_weights(ny, nx, center_yx, aperture)
    valid = np.isfinite(stat) & (weights[None, :, :] > 0)
    variance = np.nansum(stat * (weights[None, :, :] ** 2), axis=(1, 2))
    variance[np.sum(valid, axis=(1, 2)) == 0] = np.nan
    factor = float(stat_factor) * float(covariance_factor)
    if not np.isfinite(factor) or factor <= 0:
        factor = 1.0
    variance *= factor
    return np.sqrt(np.clip(variance, 0.0, np.inf)).astype(np.float64)


def control_aperture_spectra(
    cube_zyx,
    object_yx,
    star_yx,
    aperture: dict,
    *,
    n_controls: int = 8,
    exclude_angle_deg: float = 25.0,
    margin_px: int | None = None,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    cube = _as_cube(cube_zyx)
    _, ny, nx = cube.shape
    if margin_px is None:
        if str(aperture.get("kind", "box")) == "box":
            margin_px = int(aperture.get("size", 3)) // 2 + 1
        else:
            margin_px = int(math.ceil(float(aperture.get("radius_px", 3.0)))) + 1
    controls = same_radius_control_positions(
        object_yx,
        star_yx,
        ny,
        nx,
        n_positions=int(n_controls),
        exclude_angle_deg=float(exclude_angle_deg),
        margin_px=int(margin_px),
    )
    spectra = []
    npix = []
    for yx in controls:
        flux, npix_eff = aperture_spectrum(cube, yx, aperture)
        spectra.append(flux)
        npix.append(npix_eff)
    if not spectra:
        empty = np.empty((0, cube.shape[0]), dtype=np.float64)
        return controls, empty, empty.copy()
    return controls, np.asarray(spectra, dtype=np.float64), np.asarray(npix, dtype=np.float64)


def empirical_error_spectrum(
    cube_zyx,
    object_yx,
    star_yx,
    aperture: dict,
    *,
    n_controls: int = 8,
    exclude_angle_deg: float = 25.0,
    fallback_spectrum: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Estimate 1-sigma error from same-radius control apertures."""

    controls, control_specs, _npix = control_aperture_spectra(
        cube_zyx,
        object_yx,
        star_yx,
        aperture,
        n_controls=n_controls,
        exclude_angle_deg=exclude_angle_deg,
    )
    if control_specs.shape[0] >= 2:
        err = robust_sigma_axis0(control_specs)
    else:
        if fallback_spectrum is None:
            fallback_spectrum, _ = aperture_spectrum(cube_zyx, object_yx, aperture)
        sigma = robust_sigma(fallback_spectrum)
        err = np.full(np.asarray(fallback_spectrum).size, sigma, dtype=np.float64)
    return np.asarray(err, dtype=np.float64), controls, control_specs


def _flag_window(wave_A: np.ndarray, windows_A: Sequence[Sequence[float]], bit: int, flags: np.ndarray) -> None:
    for window in windows_A or ():
        if window is None or len(window) != 2:
            continue
        lo, hi = window
        if lo is None or hi is None:
            continue
        flags[(wave_A >= float(lo)) & (wave_A <= float(hi))] |= int(bit)


def channel_flags(
    wave_A,
    *,
    bad_windows_A: Sequence[Sequence[float]] = (),
    skyline_windows_A: Sequence[Sequence[float]] = (),
    interpolated_windows_A: Sequence[Sequence[float]] = (),
    clipped_mask=None,
    good_mask=None,
    bad_mask=None,
) -> np.ndarray:
    wave = np.asarray(wave_A, dtype=np.float64)
    flags = np.zeros(wave.size, dtype=np.int32)
    _flag_window(wave, bad_windows_A, FLAG_BAD_WINDOW, flags)
    _flag_window(wave, skyline_windows_A, FLAG_SKYLINE, flags)
    _flag_window(wave, interpolated_windows_A, FLAG_INTERPOLATED, flags)
    if good_mask is not None:
        flags[~np.asarray(good_mask, dtype=bool)] |= FLAG_BAD_WINDOW
    if bad_mask is not None:
        flags[np.asarray(bad_mask, dtype=bool)] |= FLAG_BAD_WINDOW
    if clipped_mask is not None:
        flags[np.asarray(clipped_mask, dtype=bool)] |= FLAG_CLIPPED
    return flags


def aperture_correction_from_psf(
    wave_A,
    aperture: dict,
    psf_model: dict | None,
    *,
    center_yx=(0.0, 0.0),
    correction_mode: str = "auto",
    growth_curve=None,
) -> tuple[np.ndarray, str, float]:
    """Return wavelength-dependent aperture correction from a C1 PSF model.

    By default the PSF is normalized to 1 inside ``norm_radius_px`` (25 px =
    0.63" in NFM), so the recovered "total flux" is really *the flux inside that
    radius*. Measured on the A2-size cube, 40-55% of the modelled light lies
    outside it, and the missing factor is chromatic (~2.5 blue, ~1.9 red), so it
    does not cancel -- it tilts the continuum.

    Passing ``growth_curve`` (A2's ``growth_curve`` QC block) multiplies the
    correction by the empirically measured ``F_total / F(<=norm_radius)`` and
    switches the convention to genuine total flux. A2 measures it; the caller
    decides -- see ``x01_flux_convention``.
    """

    wave = np.asarray(wave_A, dtype=np.float64)
    mode = str(correction_mode or "auto").lower()
    if mode in {"none", "off", "false"}:
        return np.ones(wave.size, dtype=np.float64), "none", 0.0
    if psf_model is None:
        if mode in {"auto", "optional"}:
            return np.ones(wave.size, dtype=np.float64), "none", 0.0
        raise RuntimeError("Aperture correction requested but no psf_model was supplied.")

    norm_radius = float(psf_model.get("norm_radius_px", 25.0))
    half = int(math.ceil(norm_radius))
    frac_y = float(center_yx[0]) - round(float(center_yx[0]))
    frac_x = float(center_yx[1]) - round(float(center_yx[1]))
    source_center = (half + frac_y, half + frac_x)
    yy, xx = np.indices((2 * half + 1, 2 * half + 1), dtype=np.float64)
    dy = yy - source_center[0]
    dx = xx - source_center[1]
    weights = aperture_weights(2 * half + 1, 2 * half + 1, source_center, aperture)
    fractions = np.empty(wave.size, dtype=np.float64)
    for i, w in enumerate(wave):
        psf = evaluate_psf_model(psf_model, float(w), dy, dx)
        frac = float(np.nansum(psf * weights))
        if not np.isfinite(frac) or frac <= 0:
            raise RuntimeError(f"Invalid aperture PSF fraction at wave={w}.")
        fractions[i] = frac
    apcorr = (1.0 / fractions).astype(np.float64)
    if growth_curve:
        # Import ABSOLUTO y dentro de la funcion: esta funcion se COPIA
        # literalmente dentro de los notebooks de `debug/`, donde un import
        # relativo (`from ..growth_curve`) revienta con ImportError por no
        # haber paquete padre. El absoluto funciona en los dos sitios.
        from musepipe.growth_curve import factor_at_wavelengths

        factor = np.asarray(factor_at_wavelengths(growth_curve, wave), dtype=np.float64)
        if not np.all(np.isfinite(factor)) or np.any(factor <= 0):
            raise RuntimeError("Growth-curve total-flux factor is not finite and positive.")
        return apcorr * factor, "psf_growth_curve+empirical_total", norm_radius
    return apcorr, "psf_growth_curve", norm_radius


def make_aperture_product(
    cube_zyx,
    wave_A,
    object_yx,
    aperture: dict,
    *,
    run_id: str,
    input_cube_path: str | Path,
    input_cube_sha: str | None = None,
    star_yx=None,
    stat_zyx=None,
    stat_factor: float = 1.0,
    covariance_factor: float = 1.0,
    stat_status: str = "unknown",
    error_mode: str = "auto",
    psf_model: dict | None = None,
    aperture_correction: str = "auto",
    growth_curve=None,
    wframe: str = "topocentric",
    bunit: str = "",
    bad_windows_A: Sequence[Sequence[float]] = (),
    skyline_windows_A: Sequence[Sequence[float]] = (),
    interpolated_windows_A: Sequence[Sequence[float]] = (),
    good_mask=None,
    bad_mask=None,
    clipped_mask=None,
    n_controls: int = 8,
    exclude_angle_deg: float = 25.0,
    annulus_bkg_px: Sequence[float] | None = None,
) -> ApertureExtraction:
    cube = _as_cube(cube_zyx)
    wave = np.asarray(wave_A, dtype=np.float64)
    if wave.ndim != 1 or wave.size != cube.shape[0]:
        raise ValueError("wave_A must be 1D and match cube spectral length.")

    raw_flux, npix_eff = aperture_spectrum(cube, object_yx, aperture)
    if annulus_bkg_px is not None:
        # Wings-intact extraction: subtract a distant annulus background per
        # channel so the PSF wings survive for a self-consistent growth-curve
        # aperture correction (see annulus_background_spectrum).
        bkg = annulus_background_spectrum(
            cube, object_yx, annulus_bkg_px[0], annulus_bkg_px[1],
            exclude_yx=star_yx, exclude_radius=float(annulus_bkg_px[2]) if len(annulus_bkg_px) > 2 else 30.0,
        )
        raw_flux = raw_flux - bkg * npix_eff
    if star_yx is None:
        raw_flux_err_emp = np.full(wave.size, robust_sigma(raw_flux), dtype=np.float64)
        controls_yx = []
        control_spectra = np.empty((0, wave.size), dtype=np.float64)
        control_spectra_bkgsub = control_spectra
    else:
        controls_yx, control_spectra, control_npix = control_aperture_spectra(
            cube,
            object_yx,
            star_yx,
            aperture,
            n_controls=n_controls,
            exclude_angle_deg=exclude_angle_deg,
        )
        # Controls processed like the object: same annulus background per
        # control position (D1 v2 §3.1 "control = object").
        control_spectra_bkgsub = control_spectra
        if annulus_bkg_px is not None and control_spectra.shape[0]:
            control_spectra_bkgsub = control_spectra.copy()
            exclude_radius = float(annulus_bkg_px[2]) if len(annulus_bkg_px) > 2 else 30.0
            for k, yx in enumerate(controls_yx):
                ctrl_bkg = annulus_background_spectrum(
                    cube, yx, annulus_bkg_px[0], annulus_bkg_px[1],
                    exclude_yx=star_yx, exclude_radius=exclude_radius,
                )
                control_spectra_bkgsub[k] = control_spectra[k] - ctrl_bkg * control_npix[k]
        if control_spectra_bkgsub.shape[0] >= 2:
            raw_flux_err_emp = robust_sigma_axis0(control_spectra_bkgsub)
        else:
            raw_flux_err_emp = np.full(wave.size, robust_sigma(raw_flux), dtype=np.float64)

    requested_error_mode = str(error_mode or "auto").lower()
    stat_usable = (
        stat_zyx is not None
        and requested_error_mode != "empirical"
        and str(stat_status).lower() != "red"
    )
    if stat_usable:
        raw_flux_err = aperture_stat_error(
            stat_zyx,
            object_yx,
            aperture,
            stat_factor=stat_factor,
            covariance_factor=covariance_factor,
        )
        mode = "stat"
    else:
        raw_flux_err = np.asarray(raw_flux_err_emp, dtype=np.float64)
        mode = "empirical"

    apcorr, apcorr_mode, norm_radius = aperture_correction_from_psf(
        wave,
        aperture,
        psf_model,
        center_yx=object_yx,
        correction_mode=aperture_correction,
        growth_curve=growth_curve,
    )
    flags = channel_flags(
        wave,
        bad_windows_A=bad_windows_A,
        skyline_windows_A=skyline_windows_A,
        interpolated_windows_A=interpolated_windows_A,
        good_mask=good_mask,
        bad_mask=bad_mask,
        clipped_mask=clipped_mask,
    )

    input_cube_path = Path(input_cube_path)
    if input_cube_sha is None:
        input_cube_sha = sha256_file(input_cube_path) if input_cube_path.exists() else ""
    label = aperture_label(aperture)
    if annulus_bkg_px is not None:
        bkg_mode = f"annulus_{float(annulus_bkg_px[0]):g}_{float(annulus_bkg_px[1]):g}"
    else:
        bkg_mode = "none"
    header = {
        "FORMATV": FORMAT_VERSION,
        "METHOD": "aperture",
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
        "COVFAC": float(covariance_factor),
        "BKGMODE": bkg_mode,
        # La convencion de flujo se DECLARA, no se supone: con la curva de
        # crecimiento de A2 aplicada el "1" deja de ser el radio de
        # normalizacion y pasa a ser el flujo total medido.
        "SCALEREF": ("empirical_total_flux" if "empirical_total" in str(apcorr_mode)
                     else "normrad_total_flux"),
    }
    product = SpectrumProduct(
        wave_A=wave,
        flux=raw_flux * apcorr,
        flux_err=raw_flux_err * apcorr,
        flux_err_emp=raw_flux_err_emp * apcorr,
        apcorr=apcorr,
        npix_eff=npix_eff,
        flags=flags,
        header=header,
    )
    product.validate()
    return ApertureExtraction(
        product=product,
        raw_flux=raw_flux,
        raw_flux_err=raw_flux_err,
        raw_flux_err_emp=raw_flux_err_emp,
        controls_yx=controls_yx,
        control_spectra=control_spectra,
        aperture=dict(aperture),
        aperture_label=label,
        error_mode=mode,
        apcorr_mode=apcorr_mode,
        norm_radius_px=float(norm_radius),
        control_spectra_cal=control_spectra_bkgsub * apcorr[None, :],
        bkg_mode=bkg_mode,
    )


def extract_aperture_products(
    cube_zyx,
    wave_A,
    object_yx,
    *,
    run_id: str,
    input_cube_path: str | Path,
    apertures: Sequence[dict] | None = None,
    star_yx=None,
    stat_zyx=None,
    stat_factor_box3: float = 1.0,
    covariance_factor_box3: float = 1.0,
    stat_status: str = "unknown",
    error_mode: str = "auto",
    psf_model: dict | None = None,
    aperture_correction: str = "auto",
    growth_curve=None,
    wframe: str = "topocentric",
    bunit: str = "",
    bad_windows_A: Sequence[Sequence[float]] = (),
    skyline_windows_A: Sequence[Sequence[float]] = (),
    interpolated_windows_A: Sequence[Sequence[float]] = (),
    good_mask=None,
    bad_mask=None,
    clipped_mask=None,
    n_controls: int = 8,
    exclude_angle_deg: float = 25.0,
    annulus_bkg_px: Sequence[float] | None = None,
) -> dict[str, ApertureExtraction]:
    outputs = {}
    input_cube_sha = sha256_file(input_cube_path) if Path(input_cube_path).exists() else ""
    for aperture in list(default_apertures() if apertures is None else apertures):
        label = aperture_label(aperture)
        stat_factor = float(aperture.get("stat_factor", stat_factor_box3))
        cov_factor = float(aperture.get("covariance_factor", covariance_factor_box3))
        outputs[label] = make_aperture_product(
            cube_zyx,
            wave_A,
            object_yx,
            aperture,
            run_id=run_id,
            input_cube_path=input_cube_path,
            input_cube_sha=input_cube_sha,
            star_yx=star_yx,
            stat_zyx=stat_zyx,
            stat_factor=stat_factor,
            covariance_factor=cov_factor,
            stat_status=stat_status,
            error_mode=error_mode,
            psf_model=psf_model,
            aperture_correction=aperture_correction,
            growth_curve=growth_curve,
            wframe=wframe,
            bunit=bunit,
            bad_windows_A=bad_windows_A,
            skyline_windows_A=skyline_windows_A,
            interpolated_windows_A=interpolated_windows_A,
            good_mask=good_mask,
            bad_mask=bad_mask,
            clipped_mask=clipped_mask,
            n_controls=n_controls,
            exclude_angle_deg=exclude_angle_deg,
            annulus_bkg_px=annulus_bkg_px,
        )
    return outputs


__all__ = [
    "FLAG_BAD_WINDOW",
    "FLAG_CLIPPED",
    "FLAG_INTERPOLATED",
    "FLAG_SKYLINE",
    "ApertureExtraction",
    "aperture_correction_from_psf",
    "aperture_label",
    "aperture_spectrum",
    "aperture_stat_error",
    "channel_flags",
    "control_aperture_spectra",
    "default_apertures",
    "empirical_error_spectrum",
    "extract_aperture_products",
    "make_aperture_product",
    "sha256_file",
]
