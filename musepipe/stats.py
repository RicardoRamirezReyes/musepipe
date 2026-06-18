"""Robust statistics helpers shared by notebooks and stage modules."""

from __future__ import annotations

import warnings

import numpy as np


def finite_values(values) -> np.ndarray:
    """Return finite values as a float64 1D array."""

    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def robust_sigma(values) -> float:
    """Robust 1D sigma estimate using MAD with std fallback."""

    vals = finite_values(values)
    if vals.size == 0:
        return np.nan
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.nanstd(vals)
    return float(sigma)


def robust_sigma_axis0(values) -> np.ndarray:
    """Robust sigma along axis 0 using MAD with std fallback per column."""

    arr = np.asarray(values, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(arr, axis=0)
        mad = np.nanmedian(np.abs(arr - med[None, :]), axis=0)
        sigma = 1.4826 * mad
        std = np.nanstd(arr, axis=0)
    bad = ~np.isfinite(sigma) | (sigma <= 0)
    sigma[bad] = std[bad]
    return sigma


def finite_percentile(values, q) -> float:
    """Percentile over finite values, returning NaN for empty input."""

    vals = finite_values(values)
    if vals.size == 0:
        return np.nan
    return float(np.nanpercentile(vals, q))


def median_finite(values) -> float:
    """Median over finite values, returning NaN for empty input."""

    vals = finite_values(values)
    if vals.size == 0:
        return np.nan
    return float(np.nanmedian(vals))


def empirical_z(value, reference_values) -> float:
    """Robust z-score of one value relative to finite reference values."""

    refs = finite_values(reference_values)
    if refs.size == 0 or not np.isfinite(value):
        return np.nan
    sigma = robust_sigma(refs)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    return float((float(value) - np.nanmedian(refs)) / sigma)


def robust_limits(values, p_lo=2.0, p_hi=98.0, *, symmetric=False):
    """Return percentile display limits over finite values."""

    vals = finite_values(values)
    if vals.size == 0:
        return (np.nan, np.nan)
    if symmetric:
        vmax = finite_percentile(np.abs(vals), p_hi)
        return (-vmax, vmax)
    return (finite_percentile(vals, p_lo), finite_percentile(vals, p_hi))


__all__ = [
    "empirical_z",
    "finite_percentile",
    "finite_values",
    "median_finite",
    "robust_limits",
    "robust_sigma",
    "robust_sigma_axis0",
]
