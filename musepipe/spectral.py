"""Spectral-axis and wavelength-mask helpers."""

from __future__ import annotations

import numpy as np


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


__all__ = [
    "continuum_running_median",
    "make_wavelength_mask",
    "nearest_channel_index",
    "nearest_channel_indices",
]
