"""Per-spaxel wavelength-offset map (WP-S0, plan wavesol/stripes 2026-07-17).

Implements the diagnostic of Xie et al. 2020 (§4.2.2): measure, for every
spaxel of a combined cube, the spectral shift of its stellar absorption
features against a field-median reference spectrum. Slicer-aligned structure
in the resulting offset map is the signature of per-exposure / per-slice
wavelength-solution differences ("stripes", Hashimoto et al. 2020 Fig. 4).

Core functions only — target-agnostic and I/O-free (arrays in, dicts out).
CLI wiring, QC persistence and plots belong to the stage driver (paso S0b).

Conventions:
- Offsets are in spectral CHANNELS (positive = spaxel spectrum appears
  shifted to larger channel index, i.e. redshifted, relative to the field
  reference); `offset_map_A` converts with the median channel step.
- Default windows target stellar ABSORPTION features of an early-M primary
  and deliberately avoid: the NaLGS gap (5780-6050 A), Halpha emission
  (6540-6590, the primary is an emitter and its line-to-continuum ratio
  varies instrumentally; Xie+20 §4.1), the telluric O2 B (6860-6960) and
  A (7590-7700) bands, and strong H2O regions. Windows must NOT straddle
  the NaLGS gap: the xcorr estimator splices out NaN channels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence
import warnings

import numpy as np

from musepipe.spectral import continuum_running_median
from musepipe.stats import robust_sigma
from musepipe.stripes import _xcorr_shift_pixels


DEFAULT_WINDOWS_A: tuple[tuple[float, float], ...] = (
    (5100.0, 5550.0),   # Mg b + metal blends
    (6100.0, 6500.0),   # Ca I 6122/6162 + TiO edges (stops before Halpha)
    (6620.0, 6850.0),   # TiO (stops before O2 B band)
    (8480.0, 8680.0),   # Ca II triplet 8498/8542/8662
)

DEFAULT_MAX_LAG_CH = 6
DEFAULT_MIN_WINDOW_CHANNELS = 40
DEFAULT_MIN_FINITE_FRACTION = 0.7
DEFAULT_BRIGHTNESS_PERCENTILE = 50.0
DEFAULT_CONTINUUM_WINDOW_A = 80.0

# Gate G1 defaults (plan §2): p95 |offset| above 0.1 A, or slicer-aligned
# structure above 3x its expected noise, justifies phase 2.
GATE_P95_THRESHOLD_A = 0.1
GATE_SIGNIFICANCE_THRESHOLD = 3.0

# S/N cut for the p95 metric (paso S0b). The preliminary 2026-07-17 run showed
# the raw p95 is dominated by faint spaxels where the xcorr fails (global p95
# 3.9 A of pure noise vs 0.18 A in the r<1" core); the per-spaxel error lets us
# restrict the p95 to spaxels whose offset is actually measured. 0.08 ch is
# ~0.1 A at the MUSE channel step.
DEFAULT_MAX_ERR_CH = 0.08


def spaxel_brightness_map(cube: np.ndarray) -> np.ndarray:
    """Median finite flux per spaxel; NaN where nothing is finite."""

    data = np.asarray(cube, dtype=np.float64)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(data, axis=0)


def brightness_selection_mask(
    brightness: np.ndarray,
    percentile: float = DEFAULT_BRIGHTNESS_PERCENTILE,
) -> np.ndarray:
    """Spaxels bright enough to measure (>= the given finite percentile)."""

    finite = np.isfinite(brightness)
    if not finite.any():
        return np.zeros_like(finite)
    threshold = np.nanpercentile(brightness[finite], percentile)
    return finite & (brightness >= threshold)


def window_slices(
    waves_A: np.ndarray,
    windows_A: Sequence[Sequence[float]] = DEFAULT_WINDOWS_A,
    min_channels: int = DEFAULT_MIN_WINDOW_CHANNELS,
) -> list[slice]:
    """Index slices for the requested windows; short/absent windows dropped."""

    waves = np.asarray(waves_A, dtype=np.float64)
    slices: list[slice] = []
    for lo, hi in windows_A:
        idx = np.where((waves >= float(lo)) & (waves <= float(hi)))[0]
        if idx.size >= int(min_channels):
            slices.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return slices


def normalize_window(
    waves_w: np.ndarray,
    spec_w: np.ndarray,
    continuum_window_A: float = DEFAULT_CONTINUUM_WINDOW_A,
) -> np.ndarray | None:
    """Continuum-normalize one window: spec/continuum - 1.

    Division (not subtraction) removes the amplitude AND slope differences
    between halo radii (chromatic AO halo, Hashimoto+20 Fig. 5) so that the
    xcorr sees features, not continuum shape. Returns None when unusable.
    """

    spec = np.asarray(spec_w, dtype=np.float64)
    good = np.isfinite(spec)
    if good.sum() < DEFAULT_MIN_WINDOW_CHANNELS // 2:
        return None
    cont = continuum_running_median(np.asarray(waves_w, dtype=np.float64), spec, good,
                                    window_A=continuum_window_A)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = spec / cont - 1.0
    norm[~np.isfinite(norm)] = np.nan
    norm[np.asarray(cont) <= 0] = np.nan
    if np.isfinite(norm).sum() < DEFAULT_MIN_WINDOW_CHANNELS // 2:
        return None
    return norm


def normalize_window_cube(
    waves_w: np.ndarray,
    cube_w: np.ndarray,
    continuum_window_A: float = DEFAULT_CONTINUUM_WINDOW_A,
    *,
    block_spaxels: int = 4096,
) -> np.ndarray:
    """Vectorized :func:`normalize_window` over a stack of spectra.

    ``cube_w`` is ``(n_channels, n_spaxels)`` covering ONE window; the return
    is the same shape, each column continuum-normalized exactly as
    :func:`normalize_window` would do it (columns it would reject as ``None``
    come back all-NaN). This exists because the per-spaxel path calls
    ``continuum_running_median`` (an O(n_channels^2) Python loop) once per
    spaxel, which is hours at full resolution.

    The membership of the running-median window is taken straight from the
    wavelength axis with ``searchsorted`` (``|wave_j - wave_i| <= window/2``),
    so it reproduces the original edge truncation and the per-spaxel
    ``min_pixels`` gate exactly on any grid — not just a uniform one — instead
    of a fixed channel half-width. The heavy work is then a single
    ``np.nanmedian`` per channel over all spaxels, blocked in columns to bound
    the temporary allocation. Equivalence to the scalar path is asserted by the
    S0b test (``max|norm_vec - norm_ref| < 1e-9``).
    """

    waves = np.asarray(waves_w, dtype=np.float64)
    data = np.asarray(cube_w, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != waves.size:
        raise ValueError("cube_w must be (n_channels, n_spaxels) matching waves_w.")

    n_chan, n_spax = data.shape
    min_needed = DEFAULT_MIN_WINDOW_CHANNELS // 2
    min_pixels = 15  # continuum_running_median default
    half_width = float(continuum_window_A) / 2.0

    finite_data = np.isfinite(data)
    # For each channel i, the running-median pools channels j with
    # |wave_j - wave_i| <= half_width. searchsorted gives that index span
    # exactly (waves is ascending), matching continuum_running_median's mask.
    lo = np.searchsorted(waves, waves - half_width, side="left")
    hi = np.searchsorted(waves, waves + half_width, side="right")

    cont = np.full((n_chan, n_spax), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for start in range(0, n_spax, int(block_spaxels)):
            cols = slice(start, min(start + int(block_spaxels), n_spax))
            blk = data[:, cols]
            fblk = finite_data[:, cols]
            cblk = cont[:, cols]
            for i in range(n_chan):
                a, b = lo[i], hi[i]
                counts = fblk[a:b, :].sum(axis=0)
                med = np.nanmedian(blk[a:b, :], axis=0)
                med[counts < min_pixels] = np.nan
                cblk[i, :] = med

    with np.errstate(invalid="ignore", divide="ignore"):
        norm = data / cont - 1.0
    norm[~np.isfinite(norm)] = np.nan
    norm[cont <= 0] = np.nan

    # Reject whole columns the scalar path would return None for: too few finite
    # input channels, or too few finite normalized channels.
    good_spec = finite_data.sum(axis=0)
    good_norm = np.isfinite(norm).sum(axis=0)
    bad = (good_spec < min_needed) | (good_norm < min_needed)
    norm[:, bad] = np.nan
    return norm


def build_reference_spectrum(cube: np.ndarray, select_mask: np.ndarray) -> np.ndarray:
    """Per-channel median spectrum over the selected spaxels."""

    data = np.asarray(cube, dtype=np.float64)
    mask = np.asarray(select_mask, dtype=bool)
    if not mask.any():
        raise ValueError("build_reference_spectrum: empty spaxel selection")
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(data[:, mask], axis=1)


def measure_spaxel_offset(
    waves_A: np.ndarray,
    spec: np.ndarray,
    ref_norm_windows: Sequence[np.ndarray | None],
    slices: Sequence[slice],
    max_lag: int = DEFAULT_MAX_LAG_CH,
    min_finite_fraction: float = DEFAULT_MIN_FINITE_FRACTION,
    continuum_window_A: float = DEFAULT_CONTINUUM_WINDOW_A,
    precomputed_norms: Sequence[np.ndarray | None] | None = None,
) -> tuple[float, int, list[float], float]:
    """Offset (channels) of one spectrum vs the pre-normalized reference.

    Returns (median offset over usable windows, n windows used, per-window
    offsets, error in channels). The error is the robust scatter of the
    per-window offsets divided by sqrt(n) — NaN when fewer than 2 windows are
    usable (paso S0b: it feeds the S/N cut on the p95 metric). NaN offset when
    no window is usable.

    ``precomputed_norms`` lets the vectorized driver pass columns already
    normalized by :func:`normalize_window_cube` (one per window, or None);
    the raw finite-fraction gate is still applied here so the result is
    identical to normalizing per spaxel.
    """

    per_window: list[float] = []
    for k, (ref_norm, sl) in enumerate(zip(ref_norm_windows, slices)):
        if ref_norm is None:
            per_window.append(np.nan)
            continue
        spec_w = np.asarray(spec[sl], dtype=np.float64)
        finite_frac = np.isfinite(spec_w).mean() if spec_w.size else 0.0
        if finite_frac < min_finite_fraction:
            per_window.append(np.nan)
            continue
        if precomputed_norms is not None:
            spec_norm = precomputed_norms[k]
        else:
            spec_norm = normalize_window(waves_A[sl], spec_w, continuum_window_A)
        if spec_norm is None:
            per_window.append(np.nan)
            continue
        per_window.append(float(_xcorr_shift_pixels(ref_norm, spec_norm, max_lag=max_lag)))

    finite = [v for v in per_window if np.isfinite(v)]
    if not finite:
        return np.nan, 0, per_window, np.nan
    err = robust_sigma(finite) / np.sqrt(len(finite)) if len(finite) >= 2 else np.nan
    return float(np.median(finite)), len(finite), per_window, float(err)


def compute_offset_map(
    cube: np.ndarray,
    waves_A: np.ndarray,
    *,
    windows_A: Sequence[Sequence[float]] = DEFAULT_WINDOWS_A,
    max_lag: int = DEFAULT_MAX_LAG_CH,
    brightness_percentile: float = DEFAULT_BRIGHTNESS_PERCENTILE,
    min_finite_fraction: float = DEFAULT_MIN_FINITE_FRACTION,
    continuum_window_A: float = DEFAULT_CONTINUUM_WINDOW_A,
    vectorized: bool = True,
) -> dict[str, object]:
    """Per-spaxel wavelength-offset map of a (nlam, ny, nx) cube.

    Returns a dict with `offset_map_ch`, `offset_map_A`, `err_map_ch`,
    `err_map_A`, `nwin_map`, `select_mask`, `reference_spectrum`,
    `channel_step_A`, `windows_used_A`. Spaxels outside the brightness
    selection, or without a usable window, are NaN in the maps.

    ``vectorized`` (default) normalizes each window over all selected spaxels at
    once via :func:`normalize_window_cube`; ``vectorized=False`` runs the scalar
    per-spaxel path. Both produce the same maps (S0b equivalence gate); the flag
    exists so the test can compare them.
    """

    data = np.asarray(cube, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"compute_offset_map expects a 3D cube, got shape {data.shape}")
    waves = np.asarray(waves_A, dtype=np.float64)
    if waves.size != data.shape[0]:
        raise ValueError("wavelength axis length does not match cube")

    step_A = float(np.median(np.diff(waves)))
    slices = window_slices(waves, windows_A)
    if not slices:
        raise ValueError("no usable spectral window inside the wavelength range")

    brightness = spaxel_brightness_map(data)
    select = brightness_selection_mask(brightness, brightness_percentile)
    reference = build_reference_spectrum(data, select)
    ref_norm_windows = [
        normalize_window(waves[sl], reference[sl], continuum_window_A) for sl in slices
    ]

    ny, nx = data.shape[1:]
    offset_ch = np.full((ny, nx), np.nan)
    err_ch = np.full((ny, nx), np.nan)
    nwin = np.zeros((ny, nx), dtype=np.int16)

    sel_yx = np.argwhere(select)
    if vectorized and sel_yx.size:
        ys, xs = sel_yx[:, 0], sel_yx[:, 1]
        data_sel = data[:, ys, xs]  # (nlam, n_sel)
        norm_per_window = [
            normalize_window_cube(waves[sl], data_sel[sl, :], continuum_window_A)
            for sl in slices
        ]
        for j in range(sel_yx.shape[0]):
            precomp = [norm_per_window[k][:, j] for k in range(len(slices))]
            dz, n, _, err = measure_spaxel_offset(
                waves, data_sel[:, j], ref_norm_windows, slices,
                max_lag=max_lag, min_finite_fraction=min_finite_fraction,
                continuum_window_A=continuum_window_A, precomputed_norms=precomp,
            )
            offset_ch[ys[j], xs[j]] = dz
            err_ch[ys[j], xs[j]] = err
            nwin[ys[j], xs[j]] = n
    else:
        for iy, ix in sel_yx:
            dz, n, _, err = measure_spaxel_offset(
                waves, data[:, iy, ix], ref_norm_windows, slices,
                max_lag=max_lag, min_finite_fraction=min_finite_fraction,
                continuum_window_A=continuum_window_A,
            )
            offset_ch[iy, ix] = dz
            err_ch[iy, ix] = err
            nwin[iy, ix] = n

    return {
        "offset_map_ch": offset_ch,
        "offset_map_A": offset_ch * step_A,
        "err_map_ch": err_ch,
        "err_map_A": err_ch * step_A,
        "nwin_map": nwin,
        "select_mask": select,
        "reference_spectrum": reference,
        "channel_step_A": step_A,
        "windows_used_A": [
            (float(waves[sl.start]), float(waves[sl.stop - 1])) for sl in slices
        ],
    }


def stripe_profile(offset_map: np.ndarray, orientation: str = "vertical") -> dict[str, np.ndarray]:
    """Collapse the map along the stripe direction.

    'vertical' stripes (stage02 geometry for this dataset) are constant along
    y and vary along x: collapse rows -> one value per COLUMN. 'horizontal'
    is the transpose. Returns per-position median, robust scatter and count.
    """

    arr = np.asarray(offset_map, dtype=np.float64)
    if orientation == "vertical":
        axis = 0
    elif orientation == "horizontal":
        axis = 1
    else:
        raise ValueError(f"unknown orientation {orientation!r}")

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmedian(arr, axis=axis)
    npos = profile.size
    scatter = np.full(npos, np.nan)
    count = np.zeros(npos, dtype=int)
    for i in range(npos):
        vals = arr[:, i] if axis == 0 else arr[i, :]
        vals = vals[np.isfinite(vals)]
        count[i] = vals.size
        if vals.size >= 3:
            scatter[i] = robust_sigma(vals)
    return {"profile": profile, "scatter": scatter, "count": count}


def structure_metrics(
    offset_map_ch: np.ndarray,
    channel_step_A: float,
    orientation: str = "vertical",
    *,
    err_map_ch: np.ndarray | None = None,
    max_err_ch: float | None = None,
) -> dict[str, object]:
    """Amplitude and significance of stripe-aligned structure in the map.

    `structure_amp` is the robust sigma of the along-stripe-collapsed profile;
    `expected_noise` is what that profile would scatter by if the map were
    pure per-spaxel noise (median per-position scatter / sqrt(count));
    `structure_significance` is their ratio. The transverse profile is
    reported as a control: real stripes must show structure in the stripe
    profile, not in the transverse one.

    When `err_map_ch` and `max_err_ch` are given, the p95 |offset| is computed
    ONLY over spaxels with err < max_err_ch (paso S0b: the raw p95 is dominated
    by faint spaxels where the xcorr fails); `n_selected_low_err` records how
    many spaxels survive the cut. The column profile keeps every spaxel (its
    per-position median is already robust).
    """

    arr = np.asarray(offset_map_ch, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("structure_metrics: offset map has no finite values")

    if err_map_ch is not None and max_err_ch is not None:
        err = np.asarray(err_map_ch, dtype=np.float64)
        low_err = np.isfinite(arr) & np.isfinite(err) & (err < float(max_err_ch))
        selected = np.abs(arr[low_err])
        if selected.size == 0:  # nothing passed the cut: fall back to all finite
            selected = np.abs(finite)
    else:
        selected = np.abs(finite)
    n_selected = int(selected.size)
    p95_abs_ch = float(np.percentile(selected, 95.0))

    main = stripe_profile(arr, orientation)
    transverse = stripe_profile(
        arr, "horizontal" if orientation == "vertical" else "vertical"
    )

    def _amp_and_noise(prof: Mapping[str, np.ndarray]) -> tuple[float, float]:
        profile = prof["profile"][np.isfinite(prof["profile"])]
        amp = robust_sigma(profile) if profile.size >= 3 else float("nan")
        ok = np.isfinite(prof["scatter"]) & (prof["count"] > 0)
        if ok.any():
            noise = float(np.median(prof["scatter"][ok] / np.sqrt(prof["count"][ok])))
        else:
            noise = float("nan")
        return float(amp), noise

    amp, noise = _amp_and_noise(main)
    amp_t, noise_t = _amp_and_noise(transverse)
    significance = amp / noise if np.isfinite(amp) and noise > 0 else float("nan")
    significance_t = amp_t / noise_t if np.isfinite(amp_t) and noise_t > 0 else float("nan")

    return {
        "orientation": orientation,
        "n_spaxels_measured": int(finite.size),
        "n_selected_low_err": n_selected,
        "max_err_ch": None if max_err_ch is None else float(max_err_ch),
        "median_abs_offset_ch": float(np.median(np.abs(finite))),
        "p95_abs_offset_ch": p95_abs_ch,
        "p95_abs_offset_A": float(p95_abs_ch * channel_step_A),
        "structure_amp_ch": amp,
        "structure_amp_A": amp * channel_step_A if np.isfinite(amp) else float("nan"),
        "structure_expected_noise_ch": noise,
        "structure_significance": float(significance),
        "transverse_amp_ch": amp_t,
        "transverse_significance": float(significance_t),
        "stripe_profile_ch": main["profile"].tolist(),
        "stripe_profile_count": main["count"].tolist(),
    }


def evaluate_gate_g1(
    metrics: Mapping[str, object],
    *,
    p95_threshold_A: float = GATE_P95_THRESHOLD_A,
    significance_threshold: float = GATE_SIGNIFICANCE_THRESHOLD,
    max_err_ch: float = DEFAULT_MAX_ERR_CH,
) -> dict[str, object]:
    """Automatic RECOMMENDATION for gate G1 (the decision itself is human).

    Phase 2 (per-exposure re-reduction) is recommended when the offsets are
    large (p95 above threshold) or when the map shows significant
    stripe-aligned structure that the transverse control does not share.

    `max_err_ch` is recorded in `thresholds` for provenance — the p95 it is
    applied to is computed upstream in :func:`structure_metrics`.
    """

    p95_A = float(metrics["p95_abs_offset_A"])
    sig = float(metrics["structure_significance"])
    sig_t = float(metrics["transverse_significance"])

    reasons: list[str] = []
    if p95_A > p95_threshold_A:
        reasons.append(
            f"p95 |offset| = {p95_A:.3f} A > {p95_threshold_A:.3f} A"
        )
    aligned = np.isfinite(sig) and sig > significance_threshold
    if aligned and (not np.isfinite(sig_t) or sig > 2.0 * sig_t):
        reasons.append(
            f"stripe-aligned structure {sig:.1f}x noise "
            f"(transverse control {sig_t:.1f}x)"
        )

    recommend = bool(reasons)
    return {
        "recommendation": "fase2_justificada" if recommend else "fase2_descartable",
        "reasons": reasons if reasons else [
            f"p95 |offset| = {p95_A:.3f} A <= {p95_threshold_A:.3f} A y "
            f"estructura alineada {sig:.1f}x <= {significance_threshold:.1f}x ruido"
        ],
        "thresholds": {
            "p95_threshold_A": p95_threshold_A,
            "significance_threshold": significance_threshold,
            "max_err_ch": max_err_ch,
        },
        "decision": "pending_human",
    }


# --------------------------------------------------------------------------- #
# CLI (paso S0b): run S0 over a real cube at full resolution.
# --------------------------------------------------------------------------- #


def bin_cube_spatial(cube: np.ndarray, binning: int) -> np.ndarray:
    """Spatially block-average a (nlam, ny, nx) cube by an integer factor.

    Trims the trailing rows/cols that do not fill a full block (as the
    scratchpad preliminary did) and averages with ``nanmean`` so masked
    spaxels do not poison the bin.
    """

    data = np.asarray(cube, dtype=np.float64)
    b = int(binning)
    if b <= 1:
        return data
    nlam, ny, nx = data.shape
    ny2, nx2 = (ny // b) * b, (nx // b) * b
    if ny2 == 0 or nx2 == 0:
        raise ValueError(f"binning {b} too large for cube {ny}x{nx}")
    trimmed = data[:, :ny2, :nx2].reshape(nlam, ny2 // b, b, nx2 // b, b)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(trimmed, axis=(2, 4))


def parse_windows_arg(text: str) -> tuple[tuple[float, float], ...]:
    """Parse ``"5100:5550,6100:6500"`` into a tuple of (lo, hi) A pairs."""

    windows: list[tuple[float, float]] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        lo, hi = token.split(":")
        windows.append((float(lo), float(hi)))
    if not windows:
        raise ValueError(f"no windows parsed from {text!r}")
    return tuple(windows)


def _write_wavesol_plots(
    result: Mapping[str, object],
    metrics: Mapping[str, object],
    *,
    orientation: str,
    max_err_ch: float,
    out_path: Path,
) -> None:
    import os

    from musepipe.stats import robust_limits

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    step_A = float(result["channel_step_A"])
    offset_A = np.asarray(result["offset_map_A"], dtype=np.float64)
    err_A = np.asarray(result["err_map_A"], dtype=np.float64)
    offset_ch = np.asarray(result["offset_map_ch"], dtype=np.float64)
    err_ch = np.asarray(result["err_map_ch"], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # (1) offset map, robust symmetric scale in A
    vlo, vhi = robust_limits(offset_A, symmetric=True)
    im0 = axes[0, 0].imshow(offset_A, origin="lower", cmap="RdBu_r", vmin=vlo, vmax=vhi)
    axes[0, 0].set_title("wavelength offset [A]")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # (2) error map in A
    ehi = robust_limits(err_A, p_lo=2.0, p_hi=98.0)[1]
    im1 = axes[0, 1].imshow(err_A, origin="lower", cmap="viridis", vmin=0.0, vmax=ehi)
    axes[0, 1].set_title(f"per-spaxel error [A] (cut {max_err_ch * step_A:.3f} A)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # (3) column (stripe) profile with expected-noise band
    prof = stripe_profile(offset_ch, orientation)
    pos = np.arange(prof["profile"].size)
    p_A = prof["profile"] * step_A
    with np.errstate(invalid="ignore", divide="ignore"):
        noise_A = prof["scatter"] / np.sqrt(np.maximum(prof["count"], 1)) * step_A
    axes[1, 0].plot(pos, p_A, color="C0", lw=1.2)
    axes[1, 0].fill_between(pos, p_A - noise_A, p_A + noise_A, color="C0", alpha=0.25,
                            label="expected noise band")
    axes[1, 0].axhline(0.0, color="k", lw=0.6)
    axes[1, 0].set_xlabel("column" if orientation == "vertical" else "row")
    axes[1, 0].set_ylabel("median offset [A]")
    axes[1, 0].set_title(f"{orientation} stripe profile")
    axes[1, 0].legend(fontsize=8)

    # (4) histogram of low-error spaxel offsets only
    low = np.isfinite(offset_A) & np.isfinite(err_ch) & (err_ch < float(max_err_ch))
    vals = offset_A[low]
    if vals.size:
        axes[1, 1].hist(vals, bins=60, color="C2", alpha=0.8)
    axes[1, 1].axvline(0.0, color="k", lw=0.6)
    axes[1, 1].set_xlabel("offset [A]")
    axes[1, 1].set_ylabel("N spaxels")
    axes[1, 1].set_title(f"low-error offsets (n={int(vals.size)})")

    p95_A = float(metrics["p95_abs_offset_A"])
    sig = float(metrics["structure_significance"])
    fig.suptitle(
        f"S0 wavelength-offset map — p95|off|={p95_A:.3f} A, "
        f"stripe sig={sig:.1f}x (transverse {float(metrics['transverse_significance']):.1f}x)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_offset_map_fits(result: Mapping[str, object], out_path: Path, header_info: Mapping[str, object]) -> None:
    from astropy.io import fits

    primary = fits.PrimaryHDU()
    for key, value in header_info.items():
        primary.header[key] = value
    hdus = [primary]
    for name, key in (
        ("OFFSET_A", "offset_map_A"),
        ("OFFSET_CH", "offset_map_ch"),
        ("ERR_CH", "err_map_ch"),
        ("ERR_A", "err_map_A"),
    ):
        arr = np.asarray(result[key], dtype=np.float32)
        hdus.append(fits.ImageHDU(data=arr, name=name))
    hdus.append(fits.ImageHDU(data=np.asarray(result["nwin_map"], dtype=np.int16), name="NWIN"))
    hdus.append(fits.ImageHDU(data=np.asarray(result["select_mask"], dtype=np.uint8), name="SELECT"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(out_path, overwrite=True)


def run_phase(args: argparse.Namespace) -> int:
    from musepipe.qc.cube_qc import _load_cube_and_wave, sha256_file, utc_now_iso

    cube_path = Path(args.cube).expanduser()
    if not cube_path.exists():
        print(f"ERROR: cube does not exist: {cube_path}", file=sys.stderr)
        return 2

    t0 = time.time()
    cube, waves = _load_cube_and_wave(cube_path, data_ext=None)
    if args.binning and int(args.binning) > 1:
        cube = bin_cube_spatial(cube, int(args.binning))
    windows = parse_windows_arg(args.windows) if args.windows else DEFAULT_WINDOWS_A

    result = compute_offset_map(
        cube, waves,
        windows_A=windows,
        max_lag=args.max_lag,
        brightness_percentile=args.brightness_percentile,
        continuum_window_A=args.continuum_window_A,
    )
    metrics = structure_metrics(
        result["offset_map_ch"], result["channel_step_A"], args.orientation,
        err_map_ch=result["err_map_ch"], max_err_ch=args.max_err_ch,
    )
    gate = evaluate_gate_g1(metrics, max_err_ch=args.max_err_ch)
    runtime_s = time.time() - t0

    sha = "" if args.skip_checksum else sha256_file(cube_path)
    qc = {
        "stage": "S0_wavesol_map",
        "timestamp_utc": utc_now_iso(),
        "cube": str(cube_path),
        "sha256": sha,
        "binning": int(args.binning),
        "orientation": args.orientation,
        "brightness_percentile": float(args.brightness_percentile),
        "max_err_ch": float(args.max_err_ch),
        "windows_used_A": [list(w) for w in result["windows_used_A"]],
        "channel_step_A": float(result["channel_step_A"]),
        "metrics": metrics,
        "gate_g1": gate,
        "runtime_s": float(runtime_s),
    }

    if args.map_output:
        map_path = Path(args.map_output)
        _write_offset_map_fits(result, map_path, {
            "CUBE": str(cube_path)[-68:],
            "BINNING": int(args.binning),
            "STEP_A": float(result["channel_step_A"]),
            "ORIENT": args.orientation,
            "MAXERRCH": float(args.max_err_ch),
        })
        qc["map_output"] = str(map_path)

    if args.plot_output:
        plot_path = Path(args.plot_output)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        _write_wavesol_plots(result, metrics, orientation=args.orientation,
                             max_err_ch=args.max_err_ch, out_path=plot_path)
        qc["plot_output"] = str(plot_path)

    qc_path = Path(args.qc_output)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(
        f"S0 {cube_path.name}: p95|off|={metrics['p95_abs_offset_A']:.3f} A "
        f"(n_low_err={metrics['n_selected_low_err']}), "
        f"stripe sig={metrics['structure_significance']:.1f}x "
        f"(transverse {metrics['transverse_significance']:.1f}x) -> "
        f"{gate['recommendation']} | {runtime_s:.0f} s -> {qc_path}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S0 per-spaxel wavelength-offset map (wavesol/stripes plan)."
    )
    parser.add_argument("--cube", required=True)
    parser.add_argument("--qc-output", required=True)
    parser.add_argument("--map-output", default=None)
    parser.add_argument("--plot-output", default=None)
    parser.add_argument("--orientation", default="vertical", choices=["vertical", "horizontal"])
    parser.add_argument("--binning", type=int, default=1)
    parser.add_argument("--windows", default=None,
                        help='e.g. "5100:5550,6100:6500,6620:6850,8480:8680"')
    parser.add_argument("--brightness-percentile", type=float, default=DEFAULT_BRIGHTNESS_PERCENTILE)
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG_CH)
    parser.add_argument("--continuum-window-A", type=float, default=DEFAULT_CONTINUUM_WINDOW_A)
    parser.add_argument("--max-err-ch", type=float, default=DEFAULT_MAX_ERR_CH)
    parser.add_argument("--skip-checksum", action="store_true")
    parser.set_defaults(func=run_phase)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
