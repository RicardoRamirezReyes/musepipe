"""Chromatic PSF primitives for MUSE cube extraction stages."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import warnings
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from .stats import finite_percentile, robust_sigma


PSF_SHAPE_PARAMS = ("y0", "x0", "fwhm_maj", "fwhm_min", "theta_deg", "beta")


@dataclass(frozen=True)
class MoffatFit:
    success: bool
    params: dict[str, float]
    errors: dict[str, float]
    background: float
    chi2r: float
    clip_frac: float
    n_fit: int
    message: str


MOFFAT_BETA_FLOOR = 1.05  # Moffat is only a normalizable PSF for beta > 1.


def moffat_alpha_from_fwhm(fwhm, beta):
    fwhm = float(fwhm)
    beta = float(beta)
    # A Moffat has finite integral only for beta > 1; a degree-N beta(lambda)
    # polynomial from C1 can extrapolate to beta <= 0 at band edges outside its
    # fit range (seen on the LkCa 15 Moffat fit: 382/3681 channels beta<=0),
    # which sends 2**(1/beta) to an OverflowError and crashes every downstream
    # apcorr. Clamp to the physical floor: a no-op for any healthy PSF (beta>1),
    # and it keeps the aperture correction finite where the model is being
    # extrapolated into the non-normalizable regime.
    if not math.isfinite(beta) or beta < MOFFAT_BETA_FLOOR:
        beta = MOFFAT_BETA_FLOOR
    denom = 2.0 * math.sqrt(max(2.0 ** (1.0 / beta) - 1.0, 1e-12))
    return fwhm / denom


def moffat_elliptical_profile(dy, dx, fwhm_maj, fwhm_min, theta_deg, beta):
    """Unit-peak elliptical Moffat profile."""

    dy = np.asarray(dy, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    theta = np.deg2rad(float(theta_deg))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    alpha_maj = moffat_alpha_from_fwhm(fwhm_maj, beta)
    alpha_min = moffat_alpha_from_fwhm(fwhm_min, beta)
    rr = (x_rot / alpha_maj) ** 2 + (y_rot / alpha_min) ** 2
    return (1.0 + rr) ** (-float(beta))


def moffat_image(shape, y0, x0, fwhm_maj, fwhm_min, theta_deg, beta, amplitude=1.0, background=0.0):
    yy, xx = np.indices(shape, dtype=np.float64)
    profile = moffat_elliptical_profile(
        yy - float(y0),
        xx - float(x0),
        fwhm_maj,
        fwhm_min,
        theta_deg,
        beta,
    )
    return float(background) + float(amplitude) * profile


def fixed_radius_grid(norm_radius_px):
    radius = float(norm_radius_px)
    half = int(math.ceil(radius))
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
    mask = (yy**2 + xx**2) <= radius**2
    return yy, xx, mask


def moffat_norm(params, norm_radius_px=25.0):
    yy, xx, mask = fixed_radius_grid(norm_radius_px)
    profile = moffat_elliptical_profile(
        yy,
        xx,
        params["fwhm_maj"],
        params["fwhm_min"],
        params.get("theta_deg", 0.0),
        params["beta"],
    )
    norm = float(np.nansum(profile[mask]))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Invalid Moffat normalization.")
    return norm


def normalized_moffat_psf(dy, dx, params, norm_radius_px=25.0):
    profile = moffat_elliptical_profile(
        dy,
        dx,
        params["fwhm_maj"],
        params["fwhm_min"],
        params.get("theta_deg", 0.0),
        params["beta"],
    )
    return profile / moffat_norm(params, norm_radius_px=norm_radius_px)


def source_mask(shape, centers_yx, radius_px):
    yy, xx = np.indices(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    for center in centers_yx or ():
        if center is None:
            continue
        y, x = map(float, center)
        mask |= (yy - y) ** 2 + (xx - x) ** 2 <= float(radius_px) ** 2
    return mask


def corner_background(image, corner_size=12):
    img = np.asarray(image, dtype=np.float64)
    c = int(min(corner_size, max(1, img.shape[0] // 4), max(1, img.shape[1] // 4)))
    vals = np.concatenate(
        [
            img[:c, :c].ravel(),
            img[:c, -c:].ravel(),
            img[-c:, :c].ravel(),
            img[-c:, -c:].ravel(),
        ]
    )
    vals = vals[np.isfinite(vals)]
    return float(np.nanmedian(vals)) if vals.size else 0.0


def _initial_fit_params(image, center_yx, background, fit_mask):
    img = np.asarray(image, dtype=np.float64)
    y0, x0 = map(float, center_yx)
    peak_region = fit_mask & np.isfinite(img)
    peak = float(np.nanmax(img[peak_region] - float(background))) if np.any(peak_region) else 1.0
    if not np.isfinite(peak) or peak <= 0:
        peak = 1.0
    return np.array([peak, y0, x0, 4.0, 4.0, 0.0, 2.5], dtype=np.float64)


def _pack_params(values):
    amp, y0, x0, fmaj, fmin, theta, beta = values
    if fmin > fmaj:
        fmaj, fmin = fmin, fmaj
        theta += 90.0
    theta = ((float(theta) + 90.0) % 180.0) - 90.0
    return {
        "amplitude": float(amp),
        "y0": float(y0),
        "x0": float(x0),
        "fwhm_maj": float(fmaj),
        "fwhm_min": float(fmin),
        "theta_deg": theta,
        "beta": float(beta),
    }


def fit_moffat_image(
    image,
    *,
    center_yx,
    fit_radius_px=28.0,
    mask=None,
    background=None,
    core_mask_px=0.0,
    sigma_clip=3.0,
    max_iter=3,
    min_pixels=40,
):
    """Fit a fixed-background elliptical Moffat image model."""

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got {img.shape}.")
    ny, nx = img.shape
    cy, cx = map(float, center_yx)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    fit_mask = (rr <= float(fit_radius_px)) & np.isfinite(img)
    if mask is not None:
        fit_mask &= ~np.asarray(mask, dtype=bool)
    if float(core_mask_px) > 0:
        fit_mask &= rr > float(core_mask_px)
    if int(np.count_nonzero(fit_mask)) < int(min_pixels):
        raise RuntimeError("Too few pixels for Moffat fit.")

    background = corner_background(img) if background is None else float(background)
    y = img[fit_mask] - background
    ypix = yy[fit_mask]
    xpix = xx[fit_mask]
    good = np.ones(y.size, dtype=bool)
    x0 = _initial_fit_params(img, center_yx, background, fit_mask)
    lower = [0.0, cy - 3.0, cx - 3.0, 0.6, 0.6, -90.0, 1.05]
    upper = [
        max(float(np.nanmax(y)) * 3.0, 1.0),
        cy + 3.0,
        cx + 3.0,
        max(2.0, fit_radius_px),
        max(2.0, fit_radius_px),
        90.0,
        12.0,
    ]

    fit = None
    for _ in range(int(max_iter)):
        if int(np.count_nonzero(good)) < int(min_pixels):
            break

        def resid(values):
            params = _pack_params(values)
            model = params["amplitude"] * moffat_elliptical_profile(
                ypix[good] - params["y0"],
                xpix[good] - params["x0"],
                params["fwhm_maj"],
                params["fwhm_min"],
                params["theta_deg"],
                params["beta"],
            )
            return model - y[good]

        fit = least_squares(resid, x0=x0, bounds=(lower, upper), max_nfev=500)
        full_params = _pack_params(fit.x)
        model_all = full_params["amplitude"] * moffat_elliptical_profile(
            ypix - full_params["y0"],
            xpix - full_params["x0"],
            full_params["fwhm_maj"],
            full_params["fwhm_min"],
            full_params["theta_deg"],
            full_params["beta"],
        )
        residual_all = model_all - y
        sigma = robust_sigma(residual_all[good])
        if sigma_clip is None or not np.isfinite(sigma) or sigma <= 0:
            break
        new_good = np.abs(residual_all) <= float(sigma_clip) * sigma
        if np.array_equal(new_good, good):
            break
        good = new_good
        x0 = fit.x

    if fit is None:
        raise RuntimeError("Moffat fit did not run.")
    params = _pack_params(fit.x)
    model = params["amplitude"] * moffat_elliptical_profile(
        ypix[good] - params["y0"],
        xpix[good] - params["x0"],
        params["fwhm_maj"],
        params["fwhm_min"],
        params["theta_deg"],
        params["beta"],
    )
    resid = model - y[good]
    sigma = robust_sigma(resid)
    dof = max(1, int(np.count_nonzero(good)) - 7)
    chi2r = float(np.nansum((resid / sigma) ** 2) / dof) if np.isfinite(sigma) and sigma > 0 else np.nan
    errors = {key: np.nan for key in ("amplitude",) + PSF_SHAPE_PARAMS}
    if fit.jac is not None and fit.jac.size and np.isfinite(sigma) and sigma > 0:
        try:
            cov = np.linalg.pinv(fit.jac.T @ fit.jac) * sigma**2
            err_values = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
            for key, err in zip(("amplitude", "y0", "x0", "fwhm_maj", "fwhm_min", "theta_deg", "beta"), err_values):
                errors[key] = float(err)
        except Exception:
            pass

    return MoffatFit(
        success=bool(fit.success),
        params=params,
        errors=errors,
        background=background,
        chi2r=chi2r,
        clip_frac=float(1.0 - np.count_nonzero(good) / y.size),
        n_fit=int(np.count_nonzero(good)),
        message=str(fit.message),
    )


def evaluate_moffat_fit(shape, fit: MoffatFit):
    p = fit.params
    return moffat_image(
        shape,
        p["y0"],
        p["x0"],
        p["fwhm_maj"],
        p["fwhm_min"],
        p["theta_deg"],
        p["beta"],
        amplitude=p["amplitude"],
        background=fit.background,
    )


def companion_ring_metric(image, model, primary_yx, companion_yx, *, width_px=3.0, source_exclusion_radius_px=0.0):
    img = np.asarray(image, dtype=np.float64)
    mod = np.asarray(model, dtype=np.float64)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    py, px = map(float, primary_yx)
    cy, cx = map(float, companion_yx)
    radius = math.hypot(cy - py, cx - px)
    rr = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
    ann = np.abs(rr - radius) <= float(width_px) / 2.0
    if source_exclusion_radius_px and source_exclusion_radius_px > 0:
        ann &= (yy - cy) ** 2 + (xx - cx) ** 2 > float(source_exclusion_radius_px) ** 2
    halo = np.abs(mod)
    vals = np.abs(img - mod) / np.maximum(halo, np.nanmedian(halo[ann]) * 0.05)
    vals = vals[ann & np.isfinite(vals)]
    if vals.size == 0:
        return {"radius_px": float(radius), "median_pct": np.nan, "p90_pct": np.nan}
    return {
        "radius_px": float(radius),
        "median_pct": float(100.0 * np.nanmedian(vals)),
        "p90_pct": float(100.0 * finite_percentile(vals, 90.0)),
    }


def smooth_parameter(wavelengths_A, values, *, max_degree=2, wave_ref_A=None, wave_scale_A=1000.0):
    wave = np.asarray(wavelengths_A, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    good = np.isfinite(wave) & np.isfinite(vals)
    if int(np.count_nonzero(good)) == 0:
        raise ValueError("No finite values to smooth.")
    wave_ref = float(np.nanmedian(wave[good])) if wave_ref_A is None else float(wave_ref_A)
    x = (wave[good] - wave_ref) / float(wave_scale_A)
    y = vals[good]
    best = None
    for deg in range(0, min(int(max_degree), y.size - 1) + 1):
        coeff_high = np.polyfit(x, y, deg)
        pred = np.polyval(coeff_high, x)
        rss = float(np.nansum((y - pred) ** 2))
        k = deg + 1
        aic = y.size * math.log(max(rss / max(y.size, 1), 1e-24)) + 2 * k
        if best is None or aic < best["aic"]:
            best = {"degree": deg, "coeff_high": coeff_high, "rss": rss, "aic": aic}
    coeff_low = best["coeff_high"][::-1].astype(float).tolist()
    return {
        "degree": int(best["degree"]),
        "coefficients": coeff_low,
        "wave_ref_A": wave_ref,
        "wave_scale_A": float(wave_scale_A),
        "model": "polynomial",
    }


def eval_smoothed_parameter(spec, wavelength_A):
    x = (float(wavelength_A) - float(spec["wave_ref_A"])) / float(spec.get("wave_scale_A", 1000.0))
    coeff = np.asarray(spec["coefficients"], dtype=np.float64)
    return float(np.polynomial.polynomial.polyval(x, coeff))


def build_psf_model_document(
    wavelength_bins_A,
    fit_rows,
    *,
    form="moffat",
    norm_radius_px=25.0,
    hybrid=False,
):
    waves = np.asarray(wavelength_bins_A, dtype=np.float64)
    smoothing = {}
    for key in PSF_SHAPE_PARAMS:
        smoothing[key] = smooth_parameter(waves, [row[key] for row in fit_rows])
    return {
        "form": str(form),
        "norm_radius_px": float(norm_radius_px),
        "coefficients": smoothing,
        "hybrid": bool(hybrid),
    }


_PSFAO_PARAM_NAMES = ("r0", "C", "A", "alpha", "ratio", "theta", "beta")


@functools.lru_cache(maxsize=16384)
def _psfao_image_cached(x_key, npix, system_name, samp, norm_radius):
    """Build (and cache) the normalised Psfao image for one parameter set.

    Building the Psfao model is an FFT (~6 ms). During per-channel PSF fitting
    (C3/C4) the optimiser evaluates the SAME wavelength/params many times while
    varying only flux/position, so caching the image (keyed on the params, grid
    size, sampling and norm radius) turns hours into minutes. Returns the even
    image, its norm_radius integral, and the grid centre."""

    from maoppy.instrument import muse_nfm, muse_wfm
    from maoppy.psfmodel import Psfao

    system = muse_wfm if str(system_name).lower().endswith("wfm") else muse_nfm
    model = Psfao((npix, npix), system=system, samp=samp)
    # Clip to Psfao's physical bounds (smoothed/interpolated params can drift out
    # of range at edge/gap wavelengths).
    low, high = model.bounds
    eps = 1e-6
    x = [
        float(np.clip(
            xi,
            low[i] + eps if np.isfinite(low[i]) else -np.inf,
            high[i] - eps if np.isfinite(high[i]) else np.inf,
        ))
        for i, xi in enumerate(x_key)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = np.asarray(model(x), dtype=np.float64)  # peak at (npix//2, npix//2)
    c = npix // 2
    gy, gx = np.mgrid[0:npix, 0:npix]
    rr = np.hypot(gy - c, gx - c)
    total = float(np.nansum(img[rr <= float(norm_radius)]))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Psfao normalization within norm_radius failed.")
    return img, total, c


def _evaluate_psfao(model_doc, wavelength_A, dy, dx):
    """Evaluate a physical AO PSF (maoppy Psfao) on the (dy, dx) offsets,
    normalised so it sums to 1 within ``norm_radius_px``. Mirrors the Moffat
    branch's contract so aperture-correction/growth-curve/optimal/psffit
    consumers are agnostic to the PSF form. Params come from the C1 per-bin
    ``param_table`` (interpolated) or ``smoothed_poly``. The expensive FFT build
    is cached in ``_psfao_image_cached``; here we only re-sample it."""

    from maoppy.instrument import muse_nfm
    from scipy.ndimage import map_coordinates

    dy = np.asarray(dy, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    if dy.shape != dx.shape:
        raise ValueError("psfao evaluation expects matching dy/dx offset arrays.")
    names = model_doc.get("param_names", _PSFAO_PARAM_NAMES)
    # Snap the wavelength to a coarse bin before building the PSF: C1 fits the
    # Psfao parameters in 100 A bins and the PSF varies <0.5% within ~50 A, so
    # binning lets consecutive channels share ONE cached FFT build (3681 builds
    # -> ~90) with negligible loss. Sampling still uses the exact per-call
    # offsets, so per-channel positions/flux stay exact.
    wave_bin = float(model_doc.get("psfao_wave_bin_A", 50.0))
    w_eff = round(float(wavelength_A) / wave_bin) * wave_bin if wave_bin > 0 else float(wavelength_A)
    table = model_doc.get("param_table")
    if table:
        # Interpolate the per-bin fits (Psfao PSD params are degenerate, so
        # smoothing them independently then reconstructing corrupts the PSF).
        lam = np.asarray(table["lambda_A"], dtype=np.float64)
        w = float(np.clip(w_eff, lam.min(), lam.max()))
        x = [float(np.interp(w, lam, np.asarray(table[name], dtype=np.float64))) for name in names]
    else:
        poly = model_doc.get("smoothed_poly") or {}
        x = [float(np.polyval(np.asarray(poly[name], dtype=np.float64), w_eff)) for name in names]
    system_name = "muse_wfm" if str(model_doc.get("system", "muse_nfm")).lower().endswith("wfm") else "muse_nfm"
    samp = float(muse_nfm.samp(w_eff * 1e-10))
    norm_radius = float(model_doc.get("norm_radius_px", 25.0))
    # Grid sizing: two regimes, both giving a source-position-INDEPENDENT npix so
    # the cache is reused across the star/companion/control evaluations at a given
    # wavelength.
    #   - apcorr/growth-curve callers pass offsets within norm_radius -> npix from
    #     norm_radius (small, fast).
    #   - psffit evaluates over the full image (offsets up to ~image size); the PSF
    #     is only needed over the joint fit region (source separation + fit radius),
    #     so use a fixed grid_reach (default 100 px, config `psfao_grid_reach_px`).
    #     This captures the star halo at the companion (~71 px) while avoiding the
    #     ~316 px FFTs the full-image offsets would otherwise force; offsets beyond
    #     the grid sample as 0 (negligible PSF there).
    max_off = 0.0
    if dy.size:
        max_off = max(float(np.nanmax(np.abs(dy))), float(np.nanmax(np.abs(dx))))
    if max_off <= norm_radius:
        reach = norm_radius
    else:
        # 140 px is where the PSF-fit companion flux converges for this geometry
        # (100 under-samples the AO halo → ~3% high; 140/180/250 agree to 0.2%).
        reach = max(norm_radius, float(model_doc.get("psfao_grid_reach_px", 140.0)))
    npix = 2 * (int(np.ceil(reach)) + 2)  # even, source at npix//2
    img, total, c = _psfao_image_cached(
        tuple(round(v, 10) for v in x), npix, system_name, round(samp, 10), round(norm_radius, 6)
    )
    rows = (c + dy).ravel()
    cols = (c + dx).ravel()
    vals = map_coordinates(img, [rows, cols], order=1, mode="constant", cval=0.0).reshape(dy.shape)
    return vals / total


def evaluate_psf_model(model_doc, wavelength_A, dy, dx):
    form = str(model_doc.get("form", "moffat")).lower()
    if form == "psfao":
        return _evaluate_psfao(model_doc, wavelength_A, dy, dx)
    if form != "moffat":
        raise ValueError(f"Unsupported psf_model form={form!r}; expected 'moffat' or 'psfao'.")
    params = {
        key: eval_smoothed_parameter(model_doc["coefficients"][key], wavelength_A)
        for key in PSF_SHAPE_PARAMS
    }
    return normalized_moffat_psf(
        dy,
        dx,
        params,
        norm_radius_px=float(model_doc.get("norm_radius_px", 25.0)),
    )


def psf_roundtrip_error(model_doc, wavelengths_A):
    yy, xx, mask = fixed_radius_grid(float(model_doc.get("norm_radius_px", 25.0)))
    errors = []
    for wave in wavelengths_A:
        vals = evaluate_psf_model(model_doc, float(wave), yy, xx)
        errors.append(abs(float(np.nansum(vals[mask])) - 1.0))
    return float(np.nanmax(errors)) if errors else np.nan


def radial_hybrid_profile(residual, center_yx, *, mask=None, bin_width_px=1.0, smoothing_scale_px=4.0):
    resid = np.asarray(residual, dtype=np.float64)
    yy, xx = np.indices(resid.shape, dtype=np.float64)
    rr = np.sqrt((yy - float(center_yx[0])) ** 2 + (xx - float(center_yx[1])) ** 2)
    valid = np.isfinite(resid)
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)
    bins = np.floor(rr / float(bin_width_px)).astype(int)
    nbin = int(np.nanmax(bins)) + 1
    profile = np.full(nbin, np.nan, dtype=np.float64)
    radii = (np.arange(nbin, dtype=np.float64) + 0.5) * float(bin_width_px)
    for b in range(nbin):
        pix = valid & (bins == b)
        if np.count_nonzero(pix) >= 3:
            profile[b] = np.nanmedian(resid[pix])
    finite = np.isfinite(profile)
    if np.count_nonzero(finite) >= 2:
        profile[~finite] = np.interp(radii[~finite], radii[finite], profile[finite])
    else:
        profile[~finite] = 0.0
    sigma_bins = max(float(smoothing_scale_px) / float(bin_width_px), 0.0)
    if sigma_bins > 0:
        profile = gaussian_filter1d(profile, sigma=sigma_bins, mode="nearest")
    return radii, profile


def evaluate_radial_profile(shape, center_yx, radii, profile):
    yy, xx = np.indices(shape, dtype=np.float64)
    rr = np.sqrt((yy - float(center_yx[0])) ** 2 + (xx - float(center_yx[1])) ** 2)
    return np.interp(rr.ravel(), np.asarray(radii), np.asarray(profile), left=profile[0], right=profile[-1]).reshape(shape)


__all__ = [
    "MoffatFit",
    "PSF_SHAPE_PARAMS",
    "build_psf_model_document",
    "companion_ring_metric",
    "corner_background",
    "evaluate_moffat_fit",
    "evaluate_psf_model",
    "evaluate_radial_profile",
    "fit_moffat_image",
    "moffat_elliptical_profile",
    "moffat_image",
    "moffat_norm",
    "normalized_moffat_psf",
    "psf_roundtrip_error",
    "radial_hybrid_profile",
    "source_mask",
    "smooth_parameter",
]
