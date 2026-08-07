"""Spectral-diversity stellar-halo subtraction kernels (SGF and LPM).

Shared infrastructure for stages C5 (``stage_x04_sgf``) and C6
(``stage_x05_lpm``), implementing the two methods analyzed in
Julo et al. 2025 (arXiv:2509.09878), plan
``docs/2026-07-15_plan_integracion_halosub.md`` (WP-H0):

* **SGF** — the state-of-the-art Savitzky-Golay filtering method
  (Haffert et al. 2019): each spaxel is divided by a stellar reference
  spectrum, the ratio is low-pass filtered with a Savitzky-Golay filter,
  and the reference times the filtered ratio is subtracted.
* **LPM** — Legendre polynomial modulation (the paper's proposal): each
  spaxel is modeled as a degree-d polynomial modulation of the reference
  spectrum, estimated by least squares (orthogonal projection) with the
  science lines masked out of the fit.

The paper's closed-form toy-model predictions (Fig. 2c/2d, Eq. 1,
App. B.1, Eq. B.5) are exposed as functions so tests and stage QC can use
them as analytical oracles.

Frozen defaults (paper Table 1; changing them requires a spec revision):
SGF d=1, window 101 channels; LPM degree 4; reference-spaxel flux mask
[0.01, 0.1] x Fmax.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre as npleg
from scipy.signal import savgol_filter

from .spectral import STANDARD_LINE_WINDOWS_A, standard_line_free_mask

# Frozen defaults (paper Table 1 / Sect. 3.2). Run config may override via
# sgf_* / lpm_* / halosub_* keys; the specs C5/C6 freeze the per-run values.
DEFAULT_SGF_WINDOW = 101
DEFAULT_SGF_DEGREE = 1
DEFAULT_LPM_DEGREE = 4
DEFAULT_FLUX_MASK_LO = 0.01
DEFAULT_FLUX_MASK_HI = 0.1


# ---------------------------------------------------------------------------
# Reference stellar spectrum
# ---------------------------------------------------------------------------

def select_reference_spaxels(
    cube_zyx,
    *,
    flux_lo_frac=DEFAULT_FLUX_MASK_LO,
    flux_hi_frac=DEFAULT_FLUX_MASK_HI,
    wave_mask=None,
    exclude_yx=None,
    exclude_radius_px=0.0,
):
    """Spatial mask of spaxels used for the stellar reference spectrum.

    Follows the paper's Table 1: spaxels with integrated flux below
    ``flux_lo_frac * Fmax`` (too noisy) or above ``flux_hi_frac * Fmax``
    (most distorted by the pipeline / star core) are excluded.

    ``exclude_yx`` optionally removes discs of ``exclude_radius_px`` around
    known sources (e.g. the companion), so its flux never contaminates the
    reference.

    Returns ``(keep_mask_2d, qc_dict)``.
    """

    cube = np.asarray(cube_zyx, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected cube (nz, ny, nx), got shape {cube.shape}.")
    if not 0.0 <= float(flux_lo_frac) < float(flux_hi_frac) <= 1.0:
        raise ValueError("Need 0 <= flux_lo_frac < flux_hi_frac <= 1.")

    nz = cube.shape[0]
    if wave_mask is None:
        wmask = np.ones(nz, dtype=bool)
    else:
        wmask = np.asarray(wave_mask, dtype=bool)
        if wmask.shape != (nz,):
            raise ValueError("wave_mask must match the spectral axis.")

    with np.errstate(all="ignore"):
        total = np.nansum(cube[wmask], axis=0)
        n_finite = np.sum(np.isfinite(cube[wmask]), axis=0)
    valid = n_finite > 0
    total = np.where(valid, total, np.nan)

    fmax = np.nanmax(total) if np.any(valid) else np.nan
    if not np.isfinite(fmax) or fmax <= 0:
        raise RuntimeError("Cannot select reference spaxels: no positive integrated flux.")

    keep = valid & (total > float(flux_lo_frac) * fmax) & (total < float(flux_hi_frac) * fmax)

    if exclude_yx:
        ny, nx = cube.shape[1:]
        yy, xx = np.indices((ny, nx), dtype=np.float64)
        for y0, x0 in exclude_yx:
            rr2 = (yy - float(y0)) ** 2 + (xx - float(x0)) ** 2
            keep &= rr2 > float(exclude_radius_px) ** 2

    qc = {
        "n_spaxels_kept": int(np.sum(keep)),
        "n_spaxels_valid": int(np.sum(valid)),
        "flux_lo_frac": float(flux_lo_frac),
        "flux_hi_frac": float(flux_hi_frac),
        "fmax": float(fmax),
    }
    if qc["n_spaxels_kept"] == 0:
        raise RuntimeError("Reference spaxel selection kept 0 spaxels; check flux mask fractions.")
    return keep, qc


def reference_spectrum(cube_zyx, keep_mask=None):
    """Median stellar reference spectrum over the selected spaxels.

    Median (not mean/sum) for robustness, as in the paper App. A.2; the
    flux-conservation constant is irrelevant because both methods estimate a
    multiplicative deformation per spaxel.
    """

    cube = np.asarray(cube_zyx, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected cube (nz, ny, nx), got shape {cube.shape}.")
    if keep_mask is None:
        spax = cube.reshape(cube.shape[0], -1)
    else:
        keep = np.asarray(keep_mask, dtype=bool)
        if keep.shape != cube.shape[1:]:
            raise ValueError("keep_mask must match the spatial axes.")
        spax = cube[:, keep]
    with np.errstate(all="ignore"):
        return np.nanmedian(spax, axis=1)


def safe_reference(s_hat, eps_frac=1e-6):
    """Reference with near-zero / non-finite channels set to NaN (safe divisor)."""

    ref = np.asarray(s_hat, dtype=np.float64).copy()
    scale = np.nanmedian(np.abs(ref))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Reference spectrum has no finite scale.")
    bad = ~np.isfinite(ref) | (np.abs(ref) < float(eps_frac) * scale)
    ref[bad] = np.nan
    return ref


# ---------------------------------------------------------------------------
# SGF (Savitzky-Golay filtering, state of the art)
# ---------------------------------------------------------------------------

def fill_nan_along_axis0(values):
    """Linear interpolation of NaNs along axis 0 (edges: nearest finite).

    Returns ``(filled, finite_mask)``. Columns without any finite value stay
    NaN in ``filled``.
    """

    arr = np.asarray(values, dtype=np.float64)
    flat = arr.reshape(arr.shape[0], -1)
    finite = np.isfinite(flat)
    filled = flat.copy()
    idx = np.arange(flat.shape[0], dtype=np.float64)
    needs = np.any(~finite, axis=0) & np.any(finite, axis=0)
    for j in np.where(needs)[0]:
        good = finite[:, j]
        filled[~good, j] = np.interp(idx[~good], idx[good], flat[good, j])
    return filled.reshape(arr.shape), finite.reshape(arr.shape)


@dataclass(frozen=True)
class SgfResult:
    stellar_cube: np.ndarray
    residual_cube: np.ndarray
    alpha_cube: np.ndarray
    ref_used: np.ndarray
    window: int
    degree: int


def sgf_subtract(cube_zyx, s_hat, *, window=DEFAULT_SGF_WINDOW, degree=DEFAULT_SGF_DEGREE):
    """SGF halo subtraction of one cube (paper App. A.3, Eq. A.10).

    Per spaxel: ``alpha = SG(d / s_hat)`` (low-pass Savitzky-Golay along the
    spectral axis), stellar estimate ``s_hat * alpha``, residual
    ``d - s_hat * alpha``. No line masking and no PCA — this is deliberately
    the literature-faithful baseline whose biases Eq. 1 quantifies.
    """

    cube = np.asarray(cube_zyx, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected cube (nz, ny, nx), got shape {cube.shape}.")
    nz = cube.shape[0]
    window = int(window)
    degree = int(degree)
    if window % 2 == 0 or window <= degree:
        raise ValueError(f"SGF window must be odd and > degree, got window={window} degree={degree}.")
    if window > nz:
        raise ValueError(f"SGF window ({window}) exceeds the spectral axis ({nz}).")

    ref = safe_reference(s_hat)
    if ref.shape != (nz,):
        raise ValueError("s_hat must match the spectral axis.")

    with np.errstate(all="ignore"):
        ratio = cube / ref[:, None, None]
    filled, finite = fill_nan_along_axis0(ratio)
    # Columns with no finite ratio at all: keep NaN through the filter.
    all_nan = ~np.any(finite, axis=0)
    filled[:, all_nan] = 0.0

    alpha = savgol_filter(filled, window_length=window, polyorder=degree, axis=0, mode="interp")
    alpha[:, all_nan] = np.nan

    stellar = ref[:, None, None] * alpha
    residual = cube - stellar
    return SgfResult(
        stellar_cube=stellar,
        residual_cube=residual,
        alpha_cube=alpha,
        ref_used=ref,
        window=window,
        degree=degree,
    )


# ---------------------------------------------------------------------------
# LPM (Legendre polynomial modulation, proposed method)
# ---------------------------------------------------------------------------

def lpm_design_matrix(wave_A, s_hat, degree=DEFAULT_LPM_DEGREE):
    """Design matrix M (paper Eq. A.13): Legendre polynomials modulating s_hat.

    Column k is ``P_k(lambda_scaled) * s_hat``; wavelengths are mapped to
    [-1, 1] over the finite channels so the basis is (near) orthogonal and
    the Gram matrix stays well conditioned (paper App. A.4).
    """

    wave = np.asarray(wave_A, dtype=np.float64)
    ref = np.asarray(s_hat, dtype=np.float64)
    if wave.shape != ref.shape:
        raise ValueError("wave_A and s_hat must have the same shape.")
    degree = int(degree)
    if degree < 0:
        raise ValueError("degree must be >= 0.")

    good = np.isfinite(wave)
    if not np.any(good):
        raise RuntimeError("No finite wavelengths for the LPM basis.")
    lo, hi = float(np.min(wave[good])), float(np.max(wave[good]))
    if hi <= lo:
        raise RuntimeError("Degenerate wavelength range for the LPM basis.")
    x = 2.0 * (wave - lo) / (hi - lo) - 1.0
    x = np.where(good, x, 0.0)

    vander = npleg.legvander(x, degree)  # (nz, degree+1)
    return vander * ref[:, None]


@dataclass(frozen=True)
class LpmResult:
    stellar_cube: np.ndarray
    residual_cube: np.ndarray
    coeffs: np.ndarray  # (degree+1, ny, nx)
    ref_used: np.ndarray
    fit_mask: np.ndarray  # channels used in the fit (common mask)
    degree: int
    condition_number: float
    n_slow_spaxels: int


def lpm_fit_mask(wave_A, s_hat, *, extra_mask=None, line_windows_A=STANDARD_LINE_WINDOWS_A):
    """Channels entering the LPM fit: finite reference, outside line windows.

    ``line_windows_A`` follows ``STANDARD_LINE_WINDOWS_A`` (Halpha/Hbeta/OI)
    by default; stages pass the per-target list frozen in their spec. This is
    the paper's key ingredient against self-subtraction: the lines never
    constrain the stellar model, which interpolates smoothly across them.
    """

    ref = safe_reference(s_hat)
    mask = standard_line_free_mask(wave_A, base_mask=np.isfinite(ref), line_windows_A=line_windows_A)
    if extra_mask is not None:
        mask &= np.asarray(extra_mask, dtype=bool)
    return mask


def lpm_subtract(
    cube_zyx,
    wave_A,
    s_hat,
    *,
    degree=DEFAULT_LPM_DEGREE,
    fit_mask=None,
    line_windows_A=STANDARD_LINE_WINDOWS_A,
    min_channel_finite_frac=0.75,
):
    """LPM halo subtraction of one cube (paper Eq. A.14/A.15).

    The pseudo-inverse of the masked design matrix is computed once and
    applied to every spaxel (the model is shared; only the data change).
    Channel hygiene (spec C6 v1.1): channels finite in less than
    ``min_channel_finite_frac`` of the spaxels are dropped from the fit mask
    (half-bad columns would push every spaxel to the slow path). Spaxels with
    additional non-finite channels inside the fit mask fall back to a
    per-spaxel least squares (slow path, counted in the result).
    """

    cube = np.asarray(cube_zyx, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected cube (nz, ny, nx), got shape {cube.shape}.")
    nz, ny, nx = cube.shape

    ref = safe_reference(s_hat)
    if fit_mask is None:
        fit_mask = lpm_fit_mask(wave_A, ref, line_windows_A=line_windows_A)
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool) & np.isfinite(ref)
    if min_channel_finite_frac is not None and ny * nx > 1:
        finite_all = np.isfinite(cube.reshape(nz, ny * nx))
        spax_has_data = np.any(finite_all, axis=0)
        if np.any(spax_has_data):
            channel_frac = finite_all[:, spax_has_data].mean(axis=1)
            fit_mask = fit_mask & (channel_frac >= float(min_channel_finite_frac))
    n_params = int(degree) + 1
    if int(np.sum(fit_mask)) <= n_params:
        raise RuntimeError(
            f"LPM fit mask keeps {int(np.sum(fit_mask))} channels for {n_params} parameters."
        )

    design = lpm_design_matrix(wave_A, ref, degree=degree)
    m_fit = design[fit_mask]
    pinv = np.linalg.pinv(m_fit)
    condition = float(np.linalg.cond(m_fit))

    data = cube.reshape(nz, ny * nx)
    data_fit = data[fit_mask]
    finite_fit = np.isfinite(data_fit)
    fast = np.all(finite_fit, axis=0)

    coeffs = np.full((n_params, ny * nx), np.nan, dtype=np.float64)
    if np.any(fast):
        coeffs[:, fast] = pinv @ data_fit[:, fast]

    slow_idx = np.where(~fast & np.any(finite_fit, axis=0))[0]
    for j in slow_idx:
        good = finite_fit[:, j]
        if int(np.sum(good)) <= n_params:
            continue
        beta, *_ = np.linalg.lstsq(m_fit[good], data_fit[good, j], rcond=None)
        coeffs[:, j] = beta

    stellar = design @ coeffs  # (nz, ny*nx); NaN coeffs propagate
    stellar = stellar.reshape(nz, ny, nx)
    residual = cube - stellar
    return LpmResult(
        stellar_cube=stellar,
        residual_cube=residual,
        coeffs=coeffs.reshape(n_params, ny, nx),
        ref_used=ref,
        fit_mask=fit_mask,
        degree=int(degree),
        condition_number=condition,
        n_slow_spaxels=int(slow_idx.size),
    )


def lpm_coefficient_energy_share(coeffs):
    """Per-degree energy share of the LPM coefficients (paper Fig. 8).

    Degree 0 carries the total flux and is excluded (as in the paper); the
    remaining energies are medians over the field of the squared coefficients,
    normalized to sum 1. Used by the C6 QC ``lpm_degree_check``.
    """

    arr = np.asarray(coeffs, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError("coeffs must be (degree+1, ny, nx) with degree >= 1.")
    with np.errstate(all="ignore"):
        energy = np.array([np.nanmedian(arr[k] ** 2) for k in range(1, arr.shape[0])])
    total = np.nansum(energy)
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Degenerate coefficient energies.")
    return energy / total


# ---------------------------------------------------------------------------
# Analytical oracles (paper toy model) — used by tests and stage QC
# ---------------------------------------------------------------------------

def sgf_peak_response(h_min, h_max, w_peak, w_max):
    """SG (d=1) value at the center of a rectangular peak (paper App. B.1).

    ``g_peak = h_min + R * (h_max - h_min)`` with
    ``R = (2*w_peak + 1) / (2*w_max + 1)``.
    """

    ratio = (2.0 * float(w_peak) + 1.0) / (2.0 * float(w_max) + 1.0)
    return float(h_min) + ratio * (float(h_max) - float(h_min))


def sgf_toy_line_estimate(l_p, r, l_s, c_s, c_p):
    """Post-SGF planetary line height in the toy model (paper Fig. 2d).

    ``L_P (1 - R) (1 - (L_S/L_P) * (C_P/C_S))`` — the line flux the SGF
    method retains; the deficit is the self-subtraction.
    """

    return float(l_p) * (1.0 - float(r)) * (1.0 - (float(l_s) / float(l_p)) * (float(c_p) / float(c_s)))


def sgf_toy_continuum_estimate(r, l_p, l_s, c_s, c_p):
    """Post-SGF planetary continuum next to the line (paper Fig. 2d).

    ``-C_P * R * ((C_S/C_P) * (L_P/L_S) - 1)`` — the spurious negative
    continuum the filtering leaves around the line.
    """

    return -float(c_p) * float(r) * ((float(c_s) / float(c_p)) * (float(l_p) / float(l_s)) - 1.0)


def sgf_self_subtraction_ratio(r, cs_over_ls):
    """Continuum-depth to line-estimate ratio (paper Eq. 1).

    ``C~_P / L^_P = -(R / (1 - R)) * (C_S / L_S)`` — exact in the toy model;
    the C5 QC evaluates it with the measured stellar continuum-to-line ratio
    as the per-line self-subtraction predictor.
    """

    r = float(r)
    if not 0.0 <= r < 1.0:
        raise ValueError("R must be in [0, 1).")
    return -(r / (1.0 - r)) * float(cs_over_ls)


def lpm_projector(design):
    """Orthogonal projector onto the column space of the design matrix."""

    m = np.asarray(design, dtype=np.float64)
    return m @ np.linalg.pinv(m)


def lpm_mse_terms(design, stellar_spaxel, planet_spaxel, sigma):
    """Expected MSE decomposition of the LPM residual (paper Eq. B.5).

    ``MSE = ||P_perp s||^2 + ||P m p||^2 + (l - (d+1)) sigma^2`` — star
    underfitting, planet overfitting, and noise overfitting. Used as the
    analytic oracle for the degree bias-variance trade-off (paper Fig. 6 and
    the C6 tuning QC).
    """

    m = np.asarray(design, dtype=np.float64)
    s = np.asarray(stellar_spaxel, dtype=np.float64)
    p = np.asarray(planet_spaxel, dtype=np.float64)
    proj = lpm_projector(m)
    perp = np.eye(m.shape[0]) - proj
    star_underfit = float(np.sum((perp @ s) ** 2))
    planet_overfit = float(np.sum((proj @ p) ** 2))
    noise = float(sigma) ** 2 * (m.shape[0] - m.shape[1])
    return {
        "star_underfit": star_underfit,
        "planet_overfit": planet_overfit,
        "noise_overfit": noise,
        "total": star_underfit + planet_overfit + noise,
    }


__all__ = [
    "DEFAULT_FLUX_MASK_HI",
    "DEFAULT_FLUX_MASK_LO",
    "DEFAULT_LPM_DEGREE",
    "DEFAULT_SGF_DEGREE",
    "DEFAULT_SGF_WINDOW",
    "LpmResult",
    "SgfResult",
    "fill_nan_along_axis0",
    "lpm_coefficient_energy_share",
    "lpm_design_matrix",
    "lpm_fit_mask",
    "lpm_mse_terms",
    "lpm_projector",
    "lpm_subtract",
    "reference_spectrum",
    "safe_reference",
    "select_reference_spaxels",
    "sgf_peak_response",
    "sgf_self_subtraction_ratio",
    "sgf_subtract",
    "sgf_toy_continuum_estimate",
    "sgf_toy_line_estimate",
]
