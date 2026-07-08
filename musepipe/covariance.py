"""Phase G1 validation math: empirical spectral/spatial covariance, bias budget.

Additive module (spec G1 §1.4): estimates the empirical channel-channel
correlation from the same-radius control spectra (the null population of
07b/08c), the resampling spatial-covariance inflation factor, and the small
pure-function helpers for the bias budget and the per-method validation
verdict. Covariance is estimated from data (controls/residuals), never from an
instrument model (spec §1.3).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Spectral covariance (channel-channel), estimated by lag over N controls
# --------------------------------------------------------------------------
def _demean_rows(residuals):
    r = np.asarray(residuals, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("residuals must be 2D (n_controls, n_channels).")
    return r - np.nanmean(r, axis=1, keepdims=True)


def correlation_by_lag(residuals, max_lag):
    """Average sample correlation rho(lag) over all control rows and channel pairs.

    With few controls a full channel x channel matrix is rank-deficient, so we
    assume local stationarity and average the normalized product of channels
    separated by ``lag`` across every row and every valid pair. rho(0)=1.
    """
    r = _demean_rows(residuals)
    n_rows, n_ch = r.shape
    max_lag = int(min(max_lag, n_ch - 1))
    var = np.nanmean(r * r)  # pooled variance
    rho = np.full(max_lag + 1, np.nan, dtype=np.float64)
    if not np.isfinite(var) or var <= 0:
        return rho
    for lag in range(max_lag + 1):
        a = r[:, : n_ch - lag]
        b = r[:, lag:]
        prod = a * b
        rho[lag] = float(np.nanmean(prod) / var)
    return rho


def correlation_length_channels(rho):
    """Effective correlation length = 1 + 2*sum_{lag>=1} max(rho_lag, 0) up to first sign change."""
    rho = np.asarray(rho, dtype=np.float64)
    total = 1.0
    for lag in range(1, rho.size):
        val = rho[lag]
        if not np.isfinite(val) or val <= 0.0:
            break
        total += 2.0 * val
    return float(total)


def n_eff_over_n(rho):
    """Fraction of independent channels ~ 1 / correlation_length (clipped to (0,1])."""
    length = correlation_length_channels(rho)
    if not np.isfinite(length) or length <= 0:
        return np.nan
    return float(min(1.0, 1.0 / length))


def spectral_covariance_blocks(control_spectra, *, block_size=200, max_lag=15, good_mask=None):
    """Per-block rho(lag), correlation length and n_eff/n from control spectra.

    Each control spectrum row is continuum-agnostic here: pass residuals
    (continuum already removed) for a clean estimate. Returns a dict with a
    per-block list and the median correlation length across blocks.
    """
    spectra = np.asarray(control_spectra, dtype=np.float64)
    if spectra.ndim != 2:
        raise ValueError("control_spectra must be 2D (n_controls, n_channels).")
    n_ctrl, n_ch = spectra.shape
    if good_mask is not None:
        spectra = spectra[:, np.asarray(good_mask, dtype=bool)]
        n_ch = spectra.shape[1]
    blocks = []
    for z0 in range(0, n_ch, int(block_size)):
        z1 = min(z0 + int(block_size), n_ch)
        if z1 - z0 < 3:
            continue
        rho = correlation_by_lag(spectra[:, z0:z1], max_lag)
        blocks.append({
            "z0": int(z0), "z1": int(z1),
            "rho_1": float(rho[1]) if rho.size > 1 else np.nan,
            "corr_length_channels": correlation_length_channels(rho),
            "n_eff_over_n": n_eff_over_n(rho),
            "rho": rho,
        })
    lengths = [b["corr_length_channels"] for b in blocks if np.isfinite(b["corr_length_channels"])]
    return {
        "n_controls": int(n_ctrl),
        "block_size": int(block_size),
        "max_lag": int(max_lag),
        "blocks": blocks,
        "corr_length_channels_median": float(np.median(lengths)) if lengths else np.nan,
        "rho_1_median": float(np.median([b["rho_1"] for b in blocks if np.isfinite(b["rho_1"])])) if blocks else np.nan,
    }


def bootstrap_corr_length_error(control_spectra, *, block_size=200, max_lag=15, n_boot=200, seed=0):
    """Bootstrap error on the median correlation length by resampling controls."""
    spectra = np.asarray(control_spectra, dtype=np.float64)
    n_ctrl = spectra.shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n_ctrl, size=n_ctrl)
        out = spectral_covariance_blocks(spectra[idx], block_size=block_size, max_lag=max_lag)
        if np.isfinite(out["corr_length_channels_median"]):
            vals.append(out["corr_length_channels_median"])
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan


# --------------------------------------------------------------------------
# Spatial covariance inflation from resampling (plan B1-R)
# --------------------------------------------------------------------------
def spatial_inflation_by_box(image, *, boxes=(1, 2, 3, 4, 5), n_samples=400, seed=0):
    """var(sum over NxN box) / (N^2 * var(single pixel)) over empty-region samples.

    ``image`` is a 2D (near-)source-free image (e.g. a median continuum slice of
    an empty cube region). Independent noise gives 1.0; resampling covariance
    inflates it above 1. Returns {N: factor}.
    """
    img = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(img)
    pix_var = float(np.nanvar(img[finite]))
    ny, nx = img.shape
    rng = np.random.default_rng(seed)
    out = {}
    for N in boxes:
        if N == 1:
            out[int(N)] = 1.0
            continue
        sums = []
        for _ in range(int(n_samples)):
            y = int(rng.integers(0, max(1, ny - N)))
            x = int(rng.integers(0, max(1, nx - N)))
            block = img[y:y + N, x:x + N]
            if np.all(np.isfinite(block)):
                sums.append(float(np.sum(block)))
        if len(sums) < 2 or pix_var <= 0:
            out[int(N)] = np.nan
        else:
            out[int(N)] = float(np.var(sums, ddof=1) / (N * N * pix_var))
    return out


# --------------------------------------------------------------------------
# Bias budget + verdict rules (spec §5/§6)
# --------------------------------------------------------------------------
def combine_budget_quadrature(terms):
    """Total fractional bias = signed sum of values; total error = quadrature sum.

    ``terms`` is a list of {"value_frac", "err_frac"}. Biases add (they are
    directional corrections); their uncertainties add in quadrature.
    """
    value = float(np.nansum([float(t.get("value_frac", 0.0)) for t in terms]))
    err = float(np.sqrt(np.nansum([float(t.get("err_frac", 0.0)) ** 2 for t in terms])))
    return {"total_value_frac": value, "total_err_frac": err}


def method_verdict(bias_frac, sensitivity_frac, stat_err_frac, *, bias_threshold=0.05,
                   bias_stable=True, bias_bounded=True):
    """Per-method validation verdict (spec §6).

    - ``validated``: |bias| < threshold AND sensitivity < 1 statistical sigma.
    - ``validated_with_bias``: bias stable and bounded (correctable/propagated).
    - ``rejected``: bias depends on shape params in an unbounded way.
    """
    bias = abs(float(bias_frac))
    sens = abs(float(sensitivity_frac))
    stat = abs(float(stat_err_frac))
    if not bias_bounded:
        return "rejected"
    if bias < float(bias_threshold) and sens < stat:
        return "validated"
    if bias_stable:
        return "validated_with_bias"
    return "rejected"


__all__ = [
    "bootstrap_corr_length_error",
    "combine_budget_quadrature",
    "correlation_by_lag",
    "correlation_length_channels",
    "method_verdict",
    "n_eff_over_n",
    "spatial_inflation_by_box",
    "spectral_covariance_blocks",
]
