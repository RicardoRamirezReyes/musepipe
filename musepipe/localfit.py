"""Local surface fitting and local residual spectrum extraction."""

from __future__ import annotations

import math

import numpy as np

from .apertures import aperture_weights
from .parallel import run_channel_chunks
from .stats import robust_sigma


def fit_local_surface_2d(
    image,
    yc,
    xc,
    fit_radius_px=12.0,
    mask_radius_px=3.0,
    model_kind="plane",
    extra_exclusion_yx=None,
    extra_exclusion_radius_px=3.0,
    sigma_clip=3.0,
    max_iter=3,
    min_fit_pixels=30,
):
    """Fit a local constant or planar background around one source.

    The model is defined only inside the local fit radius. Pixels within
    ``mask_radius_px`` of the target, non-finite pixels, and optional extra
    exclusions are omitted from the fit.
    """

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img.shape}.")

    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]

    yc_i = int(yc)
    xc_i = int(xc)
    rr = np.sqrt((yy - yc_i) ** 2 + (xx - xc_i) ** 2)
    local_region = rr <= float(fit_radius_px)
    fit_mask = local_region & (rr > float(mask_radius_px)) & np.isfinite(img)

    if extra_exclusion_yx is not None:
        for y2, x2 in extra_exclusion_yx:
            rr2 = np.sqrt((yy - int(y2)) ** 2 + (xx - int(x2)) ** 2)
            fit_mask &= rr2 > float(extra_exclusion_radius_px)

    n_good = int(np.count_nonzero(fit_mask))
    model = np.full_like(img, np.nan, dtype=np.float32)
    if n_good < int(min_fit_pixels):
        return model, n_good

    x = xx[fit_mask].astype(np.float64) - float(xc_i)
    y = yy[fit_mask].astype(np.float64) - float(yc_i)
    z = img[fit_mask].astype(np.float64)
    good = np.isfinite(z)

    if model_kind == "constant":
        for _ in range(int(max_iter)):
            if int(np.count_nonzero(good)) < int(min_fit_pixels):
                return model, int(np.count_nonzero(good))
            level = np.nanmedian(z[good])
            resid = z - level
            sig = robust_sigma(resid[good])
            if not np.isfinite(sig) or sig <= 0 or sigma_clip is None:
                break
            new_good = good & (np.abs(resid) <= float(sigma_clip) * sig)
            if np.array_equal(new_good, good):
                break
            good = new_good

        if int(np.count_nonzero(good)) < int(min_fit_pixels):
            return model, int(np.count_nonzero(good))
        level = np.nanmedian(z[good])
        model[local_region] = level

    elif model_kind == "plane":
        design = np.column_stack([np.ones_like(x), x, y])
        coeff = None
        for _ in range(int(max_iter)):
            if int(np.count_nonzero(good)) < int(min_fit_pixels):
                return model, int(np.count_nonzero(good))
            coeff, *_ = np.linalg.lstsq(design[good], z[good], rcond=None)
            resid = z - design @ coeff
            sig = robust_sigma(resid[good])
            if not np.isfinite(sig) or sig <= 0 or sigma_clip is None:
                break
            new_good = good & (np.abs(resid) <= float(sigma_clip) * sig)
            if np.array_equal(new_good, good):
                break
            good = new_good

        if coeff is None or int(np.count_nonzero(good)) < int(min_fit_pixels):
            return model, int(np.count_nonzero(good))

        local_model = coeff[0] + coeff[1] * (xx - float(xc_i)) + coeff[2] * (yy - float(yc_i))
        model[local_region] = local_model[local_region]

    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    return model.astype(np.float32), int(np.count_nonzero(good))


def subtract_local_surface_cube(
    cube_zyx,
    target_yx,
    *,
    fit_radius_px=12.0,
    mask_radius_px=3.0,
    model_kind="plane",
    extra_exclusion_yx=None,
    extra_exclusion_radius_px=3.0,
    sigma_clip=3.0,
    max_iter=3,
    min_fit_pixels=30,
    n_jobs=1,
):
    """Subtract a local surface model channel by channel."""

    cube = np.asarray(cube_zyx, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {cube.shape}.")

    nz, _, _ = cube.shape
    yc, xc = int(target_yx[0]), int(target_yx[1])
    residual = np.empty_like(cube, dtype=np.float32)
    model = np.empty_like(cube, dtype=np.float32)
    nfit = np.zeros(nz, dtype=int)

    def _subtract_range(z0, z1):
        # Per-channel work identical to the serial loop; disjoint output slots.
        for k in range(z0, z1):
            model_k, n_good = fit_local_surface_2d(
                cube[k],
                yc=yc,
                xc=xc,
                fit_radius_px=fit_radius_px,
                mask_radius_px=mask_radius_px,
                model_kind=model_kind,
                extra_exclusion_yx=extra_exclusion_yx,
                extra_exclusion_radius_px=extra_exclusion_radius_px,
                sigma_clip=sigma_clip,
                max_iter=max_iter,
                min_fit_pixels=min_fit_pixels,
            )
            model[k] = model_k
            residual[k] = cube[k].copy()
            local = np.isfinite(model_k)
            residual[k, local] = cube[k, local] - model_k[local]
            nfit[k] = n_good

    run_channel_chunks(_subtract_range, nz, n_jobs=n_jobs)
    return residual, model, nfit


def fit_fast_surface_coefficients(y_fit, a_fit, model_kind, min_pixels):
    """Fit per-wavelength local-surface coefficients for a fixed mask."""

    y_fit = np.asarray(y_fit, dtype=np.float64)
    a_fit = np.asarray(a_fit, dtype=np.float64)
    n_wave = y_fit.shape[0]

    if model_kind == "constant":
        coeffs = np.full((n_wave, 1), np.nan, dtype=np.float64)
        enough = np.sum(np.isfinite(y_fit), axis=1) >= int(min_pixels)
        coeffs[enough, 0] = np.nanmedian(y_fit[enough], axis=1)
        return coeffs

    coeffs = np.full((n_wave, a_fit.shape[1]), np.nan, dtype=np.float64)
    finite = np.isfinite(y_fit)
    all_finite = np.all(finite, axis=1)
    if np.any(all_finite):
        pinv = np.linalg.pinv(a_fit)
        coeffs[all_finite] = y_fit[all_finite] @ pinv.T

    partial = ~all_finite
    for row_index in np.where(partial)[0]:
        valid = finite[row_index]
        if int(np.sum(valid)) < int(min_pixels):
            continue
        coeffs[row_index], *_ = np.linalg.lstsq(a_fit[valid], y_fit[row_index, valid], rcond=None)
    return coeffs


def local_surface_spectra_fast(
    cube,
    center_yx,
    apertures,
    wave_indices,
    fit_radius_px,
    mask_radius_px,
    model_kind,
    min_pixels,
):
    """Extract local-surface residual spectra with a fixed fit mask.

    This is the vectorized Stage 8 path. It does not repeat sigma clipping per
    channel; it is intended for fast spectral extraction after Stage04b has
    already established the local model geometry.
    """

    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {cube.shape}.")

    nz, ny, nx = cube.shape
    cy, cx = map(float, center_yx)
    y1 = max(0, int(math.floor(cy - fit_radius_px)))
    y2 = min(ny, int(math.ceil(cy + fit_radius_px)) + 1)
    x1 = max(0, int(math.floor(cx - fit_radius_px)))
    x2 = min(nx, int(math.ceil(cx + fit_radius_px)) + 1)

    yy, xx = np.mgrid[y1:y2, x1:x2]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    fit_mask = (rr <= float(fit_radius_px)) & (rr > float(mask_radius_px))
    fit_idx = np.where(fit_mask.ravel())[0]
    if fit_idx.size < int(min_pixels):
        raise RuntimeError(f"Only {fit_idx.size} local fit pixels at {center_yx}.")

    a_patch = np.column_stack(
        [
            np.ones(yy.size, dtype=np.float64),
            yy.ravel().astype(np.float64) - cy,
            xx.ravel().astype(np.float64) - cx,
        ]
    )
    if model_kind == "constant":
        a_fit = np.ones((fit_idx.size, 1), dtype=np.float64)
    else:
        a_fit = a_patch[fit_idx]

    wave_indices = np.asarray(wave_indices, dtype=int)
    data = cube[wave_indices, y1:y2, x1:x2].reshape(len(wave_indices), -1)
    data = data.astype(np.float64, copy=False)
    y_fit = data[:, fit_idx]
    coeffs = fit_fast_surface_coefficients(y_fit, a_fit, model_kind, min_pixels)

    spectra = {}
    meta = {}
    for aperture in apertures:
        weights = aperture_weights(ny, nx, center_yx, aperture)[y1:y2, x1:x2].ravel()
        ap_idx = np.where(weights > 0)[0]
        w = weights[ap_idx]
        data_sum = np.nansum(data[:, ap_idx] * w[None, :], axis=1)
        if model_kind == "constant":
            model_sum = coeffs[:, 0] * np.sum(w)
        else:
            model_vector = a_patch[ap_idx].T @ w
            model_sum = coeffs @ model_vector

        spec = np.full(nz, np.nan, dtype=np.float64)
        spec[wave_indices] = data_sum - model_sum
        spectra[aperture["name"]] = spec
        meta[aperture["name"]] = {
            "n_pix": int(w.size),
            "weight_sum": float(np.sum(w)),
            "effective_n_pix": float((np.sum(w) ** 2) / np.sum(w**2)),
        }

    fit_meta = {
        "fit_y1": int(y1),
        "fit_y2": int(y2),
        "fit_x1": int(x1),
        "fit_x2": int(x2),
        "n_fit_pixels": int(fit_idx.size),
        "n_good_wavelengths_fit": int(np.sum(np.isfinite(coeffs[:, 0]))),
    }
    return spectra, meta, fit_meta


__all__ = [
    "fit_fast_surface_coefficients",
    "fit_local_surface_2d",
    "local_surface_spectra_fast",
    "subtract_local_surface_cube",
]
