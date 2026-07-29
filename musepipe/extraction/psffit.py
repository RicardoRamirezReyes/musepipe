"""Linear two-source PSF fitting for fixed-position MUSE spectra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..apertures import aperture_weights, same_radius_control_positions
from ..parallel import run_channel_chunks
from ..psf import evaluate_psf_model
from ..stats import robust_sigma, robust_sigma_axis0
from .aperture import channel_flags, sha256_file
from .optimal import covariance_factor_for_npix, estimate_variance_cube
from .product import FORMAT_VERSION, SpectrumProduct


@dataclass(frozen=True)
class PsfFitCubeResult:
    coeffs: np.ndarray
    covariance: np.ndarray
    chi2r: np.ndarray
    condition_number: np.ndarray
    rho_ab: np.ndarray
    rho_bc: np.ndarray
    npix: np.ndarray
    npix_eff_comp: np.ndarray
    residual_cube: np.ndarray
    model_cube: np.ndarray
    fit_mask: np.ndarray


@dataclass(frozen=True)
class PsfFitProducts:
    companion: SpectrumProduct
    star: SpectrumProduct
    result: PsfFitCubeResult
    comp_flux_err_emp: np.ndarray
    star_flux_err_emp: np.ndarray
    control_comp_spectra: np.ndarray
    control_star_spectra: np.ndarray
    controls_yx: list[tuple[int, int]]
    error_mode: str


def fit_region_mask(shape, star_yx, comp_yx, *, star_radius_px=20.0, comp_radius_px=12.0):
    ny, nx = map(int, shape)
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    sy, sx = map(float, star_yx)
    cy, cx = map(float, comp_yx)
    return ((yy - sy) ** 2 + (xx - sx) ** 2 <= float(star_radius_px) ** 2) | (
        (yy - cy) ** 2 + (xx - cx) ** 2 <= float(comp_radius_px) ** 2
    )


def psf_pair_design(shape, wave_A, star_yx, comp_yx, psf_model):
    yy, xx = np.indices(shape, dtype=np.float64)
    p_star = evaluate_psf_model(
        psf_model,
        float(wave_A),
        yy - float(star_yx[0]),
        xx - float(star_yx[1]),
    )
    p_comp = evaluate_psf_model(
        psf_model,
        float(wave_A),
        yy - float(comp_yx[0]),
        xx - float(comp_yx[1]),
    )
    y0 = 0.5 * (float(star_yx[0]) + float(comp_yx[0]))
    x0 = 0.5 * (float(star_yx[1]) + float(comp_yx[1]))
    scale = max(float(np.hypot(float(comp_yx[0]) - float(star_yx[0]), float(comp_yx[1]) - float(star_yx[1]))), 1.0)
    y_scaled = (yy - y0) / scale
    x_scaled = (xx - x0) / scale
    return np.stack([p_star, p_comp, np.ones(shape), y_scaled, x_scaled], axis=-1)


def _correlation(cov, i, j):
    denom = float(cov[i, i] * cov[j, j])
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(cov[i, j] / np.sqrt(denom))


def _fit_one_channel(data_2d, variance_2d, design_3d, fit_mask):
    data = np.asarray(data_2d, dtype=np.float64)
    variance = np.asarray(variance_2d, dtype=np.float64)
    design = np.asarray(design_3d, dtype=np.float64)
    valid = np.asarray(fit_mask, dtype=bool) & np.isfinite(data) & np.isfinite(variance) & (variance > 0)
    valid &= np.all(np.isfinite(design), axis=-1)
    n = int(np.count_nonzero(valid))
    p = design.shape[-1]
    if n <= p:
        coeff = np.full(p, np.nan, dtype=np.float64)
        cov = np.full((p, p), np.nan, dtype=np.float64)
        return coeff, cov, np.nan, np.inf, np.nan, np.nan, n, np.nan, np.full_like(data, np.nan), np.full_like(data, np.nan)

    a = design[valid].reshape(n, p)
    y = data[valid]
    var = variance[valid]
    sw = 1.0 / np.sqrt(var)
    aw = a * sw[:, None]
    yw = y * sw
    coeff, *_ = np.linalg.lstsq(aw, yw, rcond=None)
    normal = aw.T @ aw
    cov = np.linalg.pinv(normal)
    model = np.tensordot(design, coeff, axes=([-1], [0]))
    resid = data - model
    dof = max(1, n - p)
    chi2r = float(np.nansum((resid[valid] ** 2) / var) / dof)

    col_norm = np.linalg.norm(aw, axis=0)
    safe = col_norm > 0
    if np.all(safe):
        condition = float(np.linalg.cond(aw / col_norm[None, :]))
    else:
        condition = np.inf
    rho_ab = _correlation(cov, 0, 1)
    rho_bc_vals = [_correlation(cov, 1, j) for j in (2, 3, 4)]
    rho_bc = float(np.nanmax(np.abs(rho_bc_vals))) if np.any(np.isfinite(rho_bc_vals)) else np.nan
    pcomp = a[:, 1]
    npix_eff = np.nan
    if np.sum(pcomp**2) > 0:
        npix_eff = float((np.sum(pcomp) ** 2) / np.sum(pcomp**2))
    return coeff, cov, chi2r, condition, rho_ab, rho_bc, n, npix_eff, model, resid


def fit_psffit_cube(
    cube_zyx,
    variance_zyx,
    wave_A,
    star_yx,
    comp_yx,
    psf_model,
    *,
    star_radius_px=20.0,
    comp_radius_px=12.0,
    n_jobs=1,
) -> PsfFitCubeResult:
    cube = np.asarray(cube_zyx, dtype=np.float64)
    variance = np.asarray(variance_zyx, dtype=np.float64)
    wave = np.asarray(wave_A, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (nz,ny,nx), got {cube.shape}.")
    if variance.shape != cube.shape:
        raise ValueError("variance_zyx shape must match cube_zyx.")
    if wave.ndim != 1 or wave.size != cube.shape[0]:
        raise ValueError("wave_A must match cube spectral length.")

    nz, ny, nx = cube.shape
    fit_mask = fit_region_mask((ny, nx), star_yx, comp_yx, star_radius_px=star_radius_px, comp_radius_px=comp_radius_px)
    coeffs = np.full((nz, 5), np.nan, dtype=np.float64)
    cov = np.full((nz, 5, 5), np.nan, dtype=np.float64)
    chi2r = np.full(nz, np.nan, dtype=np.float64)
    cond = np.full(nz, np.nan, dtype=np.float64)
    rho_ab = np.full(nz, np.nan, dtype=np.float64)
    rho_bc = np.full(nz, np.nan, dtype=np.float64)
    npix = np.zeros(nz, dtype=np.int32)
    npix_eff = np.full(nz, np.nan, dtype=np.float64)
    model_cube = np.full_like(cube, np.nan, dtype=np.float64)
    residual_cube = np.full_like(cube, np.nan, dtype=np.float64)

    def _fit_range(z0, z1):
        # Per-channel work identical to the serial loop; disjoint output slots.
        for z in range(z0, z1):
            design = psf_pair_design((ny, nx), wave[z], star_yx, comp_yx, psf_model)
            row = _fit_one_channel(cube[z], variance[z], design, fit_mask)
            coeffs[z], cov[z], chi2r[z], cond[z], rho_ab[z], rho_bc[z], npix[z], npix_eff[z], model_cube[z], residual_cube[z] = row

    run_channel_chunks(_fit_range, nz, n_jobs=n_jobs)
    return PsfFitCubeResult(
        coeffs=coeffs,
        covariance=cov,
        chi2r=chi2r,
        condition_number=cond,
        rho_ab=rho_ab,
        rho_bc=rho_bc,
        npix=npix,
        npix_eff_comp=npix_eff,
        residual_cube=residual_cube,
        model_cube=model_cube,
        fit_mask=fit_mask,
    )


def control_psffit_spectra(
    cube_zyx,
    variance_zyx,
    wave_A,
    star_yx,
    comp_yx,
    psf_model,
    *,
    star_radius_px=20.0,
    comp_radius_px=12.0,
    n_controls=8,
    exclude_angle_deg=25.0,
    n_jobs=1,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    cube = np.asarray(cube_zyx, dtype=np.float64)
    _, ny, nx = cube.shape
    controls = same_radius_control_positions(
        comp_yx,
        star_yx,
        ny,
        nx,
        n_positions=int(n_controls),
        exclude_angle_deg=float(exclude_angle_deg),
        margin_px=int(np.ceil(comp_radius_px)) + 1,
    )
    star_specs = []
    comp_specs = []
    for center in controls:
        fit = fit_psffit_cube(
            cube,
            variance_zyx,
            wave_A,
            star_yx,
            center,
            psf_model,
            star_radius_px=star_radius_px,
            comp_radius_px=comp_radius_px,
            n_jobs=n_jobs,
        )
        star_specs.append(fit.coeffs[:, 0])
        comp_specs.append(fit.coeffs[:, 1])
    if not controls:
        return controls, np.empty((0, cube.shape[0])), np.empty((0, cube.shape[0]))
    return controls, np.asarray(star_specs, dtype=np.float64), np.asarray(comp_specs, dtype=np.float64)


def _spectrum_header(run_id, source_label, source_yx, input_cube_path, input_cube_sha, psf_model, wframe, bunit, error_mode, stat_factor, cov_factor, chi2r, scaleref="normrad_total_flux", apcmode="psf_model_norm_radius"):
    return {
        "FORMATV": FORMAT_VERSION,
        "METHOD": "psffit",
        "RUNID": str(run_id),
        "SRCPOS_Y": float(source_yx[0]),
        "SRCPOS_X": float(source_yx[1]),
        "APERTURE": f"psffit_{source_label}",
        "WFRAME": str(wframe),
        "INCUBE": str(input_cube_path),
        "INCUBESH": str(input_cube_sha),
        "NORMRAD": float(psf_model.get("norm_radius_px", 25.0)),
        "BUNIT": str(bunit or ""),
        "ERRMODE": str(error_mode),
        "APCMODE": str(apcmode),
        "STATFAC": float(stat_factor),
        "COVFAC": float(cov_factor),
        "ERRINFL": float(np.nanmedian(np.sqrt(np.clip(chi2r, 0.0, np.inf)))),
        "INFLAPP": False,
        # The [p_star, p_comp, 1, y, x] design's plane IS this method's
        # background convention; amplitudes are already total NORMRAD flux.
        "BKGMODE": "psffit_plane",
        "SCALEREF": str(scaleref),
    }


def make_psffit_products(
    cube_zyx,
    wave_A,
    star_yx,
    comp_yx,
    psf_model,
    *,
    run_id: str,
    input_cube_path: str | Path,
    input_cube_sha: str | None = None,
    variance_zyx=None,
    stat_factor_spaxel: float = 1.0,
    covariance_factor_box3: float = 1.0,
    stat_status: str = "unknown",
    error_mode: str = "auto",
    wframe: str = "topocentric",
    bunit: str = "",
    star_radius_px: float = 20.0,
    growth_curve=None,
    comp_radius_px: float = 12.0,
    bad_windows_A: Sequence[Sequence[float]] = (),
    skyline_windows_A: Sequence[Sequence[float]] = (),
    interpolated_windows_A: Sequence[Sequence[float]] = (),
    good_mask=None,
    bad_mask=None,
    n_controls: int = 8,
    exclude_angle_deg: float = 25.0,
    n_jobs: int = 1,
) -> PsfFitProducts:
    cube = np.asarray(cube_zyx, dtype=np.float64)
    wave = np.asarray(wave_A, dtype=np.float64)
    if variance_zyx is None:
        variance = estimate_variance_cube(cube)
    else:
        variance = np.asarray(variance_zyx, dtype=np.float64)
    if variance.shape != cube.shape:
        raise ValueError("variance_zyx shape must match cube_zyx.")
    stat_factor = float(stat_factor_spaxel)
    if not np.isfinite(stat_factor) or stat_factor <= 0:
        stat_factor = 1.0
    variance = variance * stat_factor

    result = fit_psffit_cube(
        cube,
        variance,
        wave,
        star_yx,
        comp_yx,
        psf_model,
        star_radius_px=star_radius_px,
        comp_radius_px=comp_radius_px,
        n_jobs=n_jobs,
    )
    cov_factor = covariance_factor_for_npix(result.npix_eff_comp, covariance_factor_box3)
    star_var = result.covariance[:, 0, 0] * cov_factor
    comp_var = result.covariance[:, 1, 1] * cov_factor

    controls_yx, star_controls, comp_controls = control_psffit_spectra(
        cube,
        variance,
        wave,
        star_yx,
        comp_yx,
        psf_model,
        star_radius_px=star_radius_px,
        comp_radius_px=comp_radius_px,
        n_controls=n_controls,
        exclude_angle_deg=exclude_angle_deg,
        n_jobs=n_jobs,
    )
    if comp_controls.shape[0] >= 2:
        comp_err_emp = robust_sigma_axis0(comp_controls)
    else:
        comp_err_emp = np.full(wave.size, robust_sigma(result.coeffs[:, 1]), dtype=np.float64)
    if star_controls.shape[0] >= 2:
        star_err_emp = robust_sigma_axis0(star_controls)
    else:
        star_err_emp = np.full(wave.size, robust_sigma(result.coeffs[:, 0]), dtype=np.float64)

    requested_error_mode = str(error_mode or "auto").lower()
    stat_usable = (
        variance_zyx is not None
        and requested_error_mode != "empirical"
        and str(stat_status).lower() != "red"
    )
    if stat_usable:
        comp_err = np.sqrt(np.clip(comp_var, 0.0, np.inf))
        star_err = np.sqrt(np.clip(star_var, 0.0, np.inf))
        mode = "stat"
    else:
        comp_err = comp_err_emp
        star_err = star_err_emp
        mode = "empirical"

    flags = channel_flags(
        wave,
        bad_windows_A=bad_windows_A,
        skyline_windows_A=skyline_windows_A,
        interpolated_windows_A=interpolated_windows_A,
        good_mask=good_mask,
        bad_mask=bad_mask,
    )
    # C4 no tiene apertura: el ajuste con P normalizada ya devuelve flujo
    # "total" en la convencion NORMRAD, y por eso su apcorr es 1 (spec C4 §3.4).
    # Para pasar a flujo total EMPIRICO se aplica el mismo factor al objeto,
    # la estrella, sus errores y los controles, igual que en C2/C3.
    apcorr = np.ones(wave.size, dtype=np.float64)
    if growth_curve:
        # Import ABSOLUTO y dentro de la funcion: esta funcion se COPIA
        # literalmente dentro de los notebooks de `debug/`, donde un import
        # relativo (`from ..growth_curve`) revienta con ImportError por no
        # haber paquete padre. El absoluto funciona en los dos sitios.
        from musepipe.growth_curve import factor_at_wavelengths

        factor = np.asarray(factor_at_wavelengths(growth_curve, wave), dtype=np.float64)
        if not np.all(np.isfinite(factor)) or np.any(factor <= 0):
            raise RuntimeError("Growth-curve total-flux factor is not finite and positive.")
        apcorr = apcorr * factor
    input_cube_path = Path(input_cube_path)
    if input_cube_sha is None:
        input_cube_sha = sha256_file(input_cube_path) if input_cube_path.exists() else ""
    cov_median = float(np.nanmedian(cov_factor))
    scaleref = "empirical_total_flux" if growth_curve else "normrad_total_flux"
    # El apcorr nominal de C4 es 1 (spec C4 §3.4), pero cuando lleva el factor
    # empirico encima el modo tiene que decirlo: un header que declara
    # `psf_model_norm_radius` mientras el flujo esta en escala total confunde
    # a quien audite el fichero.
    apcmode = "psf_model_norm_radius+empirical_total" if growth_curve else "psf_model_norm_radius"
    comp_header = _spectrum_header(run_id, "object", comp_yx, input_cube_path, input_cube_sha, psf_model, wframe, bunit, mode, stat_factor, cov_median, result.chi2r, scaleref, apcmode)
    star_header = _spectrum_header(run_id, "star", star_yx, input_cube_path, input_cube_sha, psf_model, wframe, bunit, mode, stat_factor, cov_median, result.chi2r, scaleref, apcmode)
    comp_flux_cal = result.coeffs[:, 1] * apcorr
    star_flux_cal = result.coeffs[:, 0] * apcorr
    comp_err_cal = comp_err * apcorr
    star_err_cal = star_err * apcorr
    comp_err_emp_cal = comp_err_emp * apcorr
    star_err_emp_cal = star_err_emp * apcorr
    comp_controls_cal = comp_controls * apcorr[None, :]
    star_controls_cal = star_controls * apcorr[None, :]
    companion = SpectrumProduct(
        wave_A=wave,
        flux=comp_flux_cal,
        flux_err=comp_err_cal,
        flux_err_emp=comp_err_emp_cal,
        apcorr=apcorr,
        npix_eff=result.npix_eff_comp,
        flags=flags,
        header=comp_header,
    )
    star = SpectrumProduct(
        wave_A=wave,
        flux=star_flux_cal,
        flux_err=star_err_cal,
        flux_err_emp=star_err_emp_cal,
        apcorr=apcorr,
        npix_eff=result.npix_eff_comp,
        flags=flags,
        header=star_header,
    )
    companion.validate()
    star.validate()
    return PsfFitProducts(
        companion=companion,
        star=star,
        result=result,
        comp_flux_err_emp=comp_err_emp_cal,
        star_flux_err_emp=star_err_emp_cal,
        control_comp_spectra=comp_controls_cal,
        control_star_spectra=star_controls_cal,
        controls_yx=controls_yx,
        error_mode=mode,
    )


def continuum_running_median(values, width=7):
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    half = max(1, int(width) // 2)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.nanmedian(arr[lo:hi])
    return out


def crosstalk_metric(wave_A, star_flux, comp_flux, windows_A):
    wave = np.asarray(wave_A, dtype=np.float64)
    a = np.asarray(star_flux, dtype=np.float64) - continuum_running_median(star_flux)
    b = np.asarray(comp_flux, dtype=np.float64) - continuum_running_median(comp_flux)
    mask = np.zeros(wave.size, dtype=bool)
    used = []
    for lo, hi in windows_A or ():
        this = (wave >= float(lo)) & (wave <= float(hi))
        if np.count_nonzero(this) >= 2:
            mask |= this
            used.append([float(lo), float(hi)])
    good = mask & np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(good) < 3:
        return {"metric_corr_b_vs_a_lines": None, "windows_used_A": used}
    corr = float(np.corrcoef(a[good], b[good])[0, 1])
    return {"metric_corr_b_vs_a_lines": corr, "windows_used_A": used}


__all__ = [
    "PsfFitCubeResult",
    "PsfFitProducts",
    "continuum_running_median",
    "control_psffit_spectra",
    "crosstalk_metric",
    "fit_psffit_cube",
    "fit_region_mask",
    "make_psffit_products",
    "psf_pair_design",
]
