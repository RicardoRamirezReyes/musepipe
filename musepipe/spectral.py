"""Spectral-axis and wavelength-mask helpers."""

from __future__ import annotations

import numpy as np


STANDARD_LINE_WINDOWS_A = (
    (6562.8, 15.0),
    (4861.3, 15.0),
    (8446.0, 15.0),
)


def nearest_channel_index(waves_A, target_A) -> int:
    """Return index of the finite wavelength channel closest to target_A."""

    waves = np.asarray(waves_A, dtype=np.float64)
    if not np.isfinite(target_A):
        raise RuntimeError("Cannot choose a wavelength channel from a non-finite target wavelength.")
    good = np.isfinite(waves)
    if not np.any(good):
        raise RuntimeError("Wavelength array has no finite values.")
    idx_good = np.where(good)[0]
    return int(idx_good[np.argmin(np.abs(waves[good] - float(target_A)))])


def nearest_channel_indices(waves_A, targets_A) -> np.ndarray:
    """Return nearest finite wavelength-channel indices for target wavelengths."""

    return np.array([nearest_channel_index(waves_A, target) for target in targets_A], dtype=int)


def continuum_running_median(waves, spec, good_mask, window_A=80.0, min_pixels=15) -> np.ndarray:
    """Estimate a broad continuum with a wavelength-window running median."""

    waves = np.asarray(waves, dtype=np.float64)
    spec = np.asarray(spec, dtype=np.float64)
    good = np.asarray(good_mask, dtype=bool) & np.isfinite(spec) & np.isfinite(waves)
    continuum = np.full(spec.shape, np.nan, dtype=np.float64)
    half_width = float(window_A) / 2.0
    for i, wave in enumerate(waves):
        local = good & (np.abs(waves - wave) <= half_width)
        if int(np.sum(local)) >= int(min_pixels):
            continuum[i] = np.nanmedian(spec[local])
    return continuum


def control_reference_bias(
    waves, control_spectra, good_mask=None, *, window_A=80.0, min_pixels=15
) -> np.ndarray:
    """Per-channel extraction bias estimated from same-radius control apertures.

    The controls are extracted identically to the object but contain no
    companion, so their MEAN spectrum is the extraction/background bias at the
    source radius (residual chromatic halo subtraction, pedestal). It is
    smoothed with the same running-median window used for the object continuum,
    because the bias is a smooth chromatic function with no spectral lines; the
    smoothing keeps the referenced continuum from inheriting per-channel control
    noise. Subtracting this bias from a method's continuum re-references it to a
    source-free baseline — the model-free "control = object" correction
    (D1 v2 §3.1); it does not touch emission lines (the mean control has none).
    """

    waves = np.asarray(waves, dtype=np.float64)
    controls = np.asarray(control_spectra, dtype=np.float64)
    if controls.ndim != 2 or controls.shape[1] != waves.size:
        raise ValueError("control_spectra must be (n_controls, n_channels) matching waves.")
    with np.errstate(all="ignore"):
        mean_control = np.nanmean(controls, axis=0)
    finite = np.isfinite(mean_control)
    mask = finite if good_mask is None else (np.asarray(good_mask, dtype=bool) & finite)
    return continuum_running_median(waves, mean_control, mask, window_A=window_A, min_pixels=min_pixels)


def median_filter_1d(values, width=21) -> np.ndarray:
    """Centered NaN-median filter with truncated edges."""

    arr = np.asarray(values, dtype=np.float64)
    width = int(width)
    if width <= 1 or arr.size == 0:
        return arr.copy()
    half = width // 2
    out = np.empty_like(arr)
    for i in range(arr.size):
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        with np.errstate(invalid="ignore"):
            out[i] = np.nanmedian(arr[lo:hi])
    return out


def standard_line_free_mask(waves_A, base_mask=None, *, line_windows_A=STANDARD_LINE_WINDOWS_A) -> np.ndarray:
    """Mask finite wavelengths outside the standard Halpha/Hbeta/OI windows."""

    waves = np.asarray(waves_A, dtype=np.float64)
    mask = np.isfinite(waves)
    if base_mask is not None:
        mask &= np.asarray(base_mask, dtype=bool)
    for center, half_width in line_windows_A:
        mask &= np.abs(waves - float(center)) > float(half_width)
    return mask


def continuum_polyfit_loglambda(
    waves,
    spec,
    good_mask,
    *,
    degree=5,
    max_iter=4,
    lower_sigma=3.0,
    upper_sigma=5.0,
) -> np.ndarray:
    """Fit a polynomial continuum in log(lambda) with asymmetric clipping."""

    waves = np.asarray(waves, dtype=np.float64)
    spec = np.asarray(spec, dtype=np.float64)
    good = np.asarray(good_mask, dtype=bool) & np.isfinite(waves) & np.isfinite(spec) & (waves > 0)
    continuum = np.full(spec.shape, np.nan, dtype=np.float64)
    if int(np.count_nonzero(good)) < 2:
        return continuum
    x_all = np.log(waves)
    x0 = float(np.nanmedian(x_all[good]))
    xscale = float(np.nanmax(np.abs(x_all[good] - x0)))
    if not np.isfinite(xscale) or xscale <= 0:
        xscale = 1.0
    x = (x_all - x0) / xscale
    fit_mask = good.copy()
    coeff = None
    for _ in range(max(1, int(max_iter))):
        n_good = int(np.count_nonzero(fit_mask))
        if n_good < 2:
            break
        deg = min(int(degree), n_good - 1)
        coeff = np.polyfit(x[fit_mask], spec[fit_mask], deg)
        model = np.polyval(coeff, x)
        resid = spec - model
        local = resid[fit_mask]
        sigma = 1.4826 * np.nanmedian(np.abs(local - np.nanmedian(local)))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = np.nanstd(local)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        new_mask = good & (resid >= -float(lower_sigma) * sigma) & (resid <= float(upper_sigma) * sigma)
        if np.array_equal(new_mask, fit_mask):
            break
        fit_mask = new_mask
    if coeff is not None:
        continuum = np.polyval(coeff, x).astype(np.float64)
    return continuum


def make_wavelength_mask(
    waves_A,
    cfg=None,
    *,
    wave_min_A=None,
    wave_max_A=None,
    exclude_drop_range=True,
    min_channels=5,
):
    """Build a finite wavelength mask with optional bounds and config drop range."""

    cfg = {} if cfg is None else cfg
    waves = np.asarray(waves_A, dtype=np.float64)
    mask = np.isfinite(waves)

    if wave_min_A is not None:
        mask &= waves >= float(wave_min_A)
    if wave_max_A is not None:
        mask &= waves <= float(wave_max_A)

    drop_min = cfg.get("drop_wave_min_A")
    drop_max = cfg.get("drop_wave_max_A")
    if exclude_drop_range and drop_min is not None and drop_max is not None:
        mask &= ~((waves >= float(drop_min)) & (waves <= float(drop_max)))

    if int(np.count_nonzero(mask)) < int(min_channels):
        raise RuntimeError("Too few wavelength channels remain after masking.")
    return mask


def spectrum_to_snr(spec_1d, noise):
    """Divide a spectrum by a scalar noise level.

    Returns an all-NaN array (shaped like the input) when ``noise`` is not
    finite or not positive; otherwise ``spec_1d / noise``.
    """

    spec_1d = np.asarray(spec_1d, dtype=np.float64)
    if not np.isfinite(noise) or noise <= 0:
        return np.full_like(spec_1d, np.nan, dtype=np.float64)
    return spec_1d / float(noise)


def integrated_line_flux(spec_1d, line_idxs, dlam_A):
    """Integrated line flux over the given channel indices.

    ``nansum(spec_1d[line_idxs]) * dlam_A``; ``line_idxs`` is cast to int.
    """

    idx = np.asarray(line_idxs, dtype=int)
    return float(np.nansum(np.asarray(spec_1d)[idx]) * dlam_A)


__all__ = [
    "continuum_running_median",
    "continuum_polyfit_loglambda",
    "integrated_line_flux",
    "make_wavelength_mask",
    "median_filter_1d",
    "nearest_channel_index",
    "nearest_channel_indices",
    "standard_line_free_mask",
    "STANDARD_LINE_WINDOWS_A",
    "spectrum_to_snr",
]
