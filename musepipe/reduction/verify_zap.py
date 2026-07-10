"""Verification checks for A2 ZAP sky cleaning."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from astropy.io import fits

from ..stats import robust_sigma
from .sky_zap import SKYLINE_WINDOWS, compute_sky_residual_metrics, wavelength_mask
from .verify import (
    VerificationError,
    VerificationResult,
    circular_aperture_mask,
    extract_aperture_spectrum,
)


CONTINUUM_CHECK_WINDOWS = (
    (5100.0, 5500.0),
    (6100.0, 6250.0),
    (6600.0, 6800.0),
)
HALPHA_WINDOW = (6540.0, 6590.0)
HALPHA_CONTINUUM_WINDOWS = (
    (6500.0, 6530.0),
    (6600.0, 6630.0),
)


def _as_cube(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 3:
        raise VerificationError(f"Expected cube shape (wave,y,x), got {arr.shape}.")
    return arr


def _window_union_mask(wave: Sequence[float], windows: Sequence[tuple[float, float]]) -> np.ndarray:
    wave_arr = np.asarray(wave, dtype=np.float64)
    mask = np.zeros(wave_arr.shape, dtype=bool)
    for lo, hi in windows:
        mask |= (wave_arr >= lo) & (wave_arr <= hi)
    return mask & np.isfinite(wave_arr)


def verify_rms_reduction_skylines(
    pre_cube,
    post_cube,
    wave: Sequence[float],
    sky_mask: np.ndarray,
    *,
    min_reduction: float = 2.0,
) -> VerificationResult:
    """V1: skyline residual RMS is reduced by at least ``min_reduction``."""

    pre_metrics = compute_sky_residual_metrics(_as_cube(pre_cube), wave, sky_mask)
    post_metrics = compute_sky_residual_metrics(_as_cube(post_cube), wave, sky_mask)
    pre = float(pre_metrics["skyline_rms_median"])
    post = float(post_metrics["skyline_rms_median"])
    reduction = float(pre / post) if post > 0 else float("inf")
    return VerificationResult(
        "v1_rms_reduction_skylines",
        reduction >= min_reduction,
        reduction,
        f"skyline_rms_pre={pre:.6g}, post={post:.6g}, reduction={reduction:.3f}",
    )


def _relative_change_pct(pre_spec: np.ndarray, post_spec: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(pre_spec) & np.isfinite(post_spec) & (pre_spec != 0)
    if not valid.any():
        raise VerificationError("No valid continuum channels for source-continuum check.")
    rel = (post_spec[valid] - pre_spec[valid]) / pre_spec[valid]
    return float(100.0 * np.nanmedian(rel))


def verify_source_continuum_intact(
    pre_cube,
    post_cube,
    wave: Sequence[float],
    sources: Mapping[str, tuple[float, float]],
    *,
    aperture_radius_px: float,
    max_primary_change_pct: float = 1.0,
    max_companion_sigma: float = 1.0,
) -> VerificationResult:
    """V2: source continua are unchanged outside skyline windows."""

    pre = _as_cube(pre_cube)
    post = _as_cube(post_cube)
    if pre.shape != post.shape:
        raise VerificationError(f"Pre/post cube shapes differ: {pre.shape} vs {post.shape}.")
    wave_arr = np.asarray(wave, dtype=np.float64)
    cont_mask = _window_union_mask(wave_arr, CONTINUUM_CHECK_WINDOWS)
    changes: dict[str, float] = {}
    companion_z: float | None = None

    for name, yx in sources.items():
        pre_spec = extract_aperture_spectrum(pre, yx, aperture_radius_px)
        post_spec = extract_aperture_spectrum(post, yx, aperture_radius_px)
        change = _relative_change_pct(pre_spec, post_spec, cont_mask)
        changes[name] = change
        if name.lower() in {"companion", "object", "roxs12b"}:
            diff = post_spec[cont_mask] - pre_spec[cont_mask]
            sigma = robust_sigma(diff)
            med = float(np.nanmedian(diff))
            companion_z = float(med / sigma) if sigma > 0 and np.isfinite(sigma) else float("inf")

    primary_ok = True
    if "primary" in sources:
        primary_ok = abs(changes.get("primary", 0.0)) < max_primary_change_pct
    companion_ok = True if companion_z is None else abs(companion_z) < max_companion_sigma
    max_abs_change = max(abs(value) for value in changes.values()) if changes else 0.0
    return VerificationResult(
        "v2_source_continuum_change_pct",
        bool(primary_ok and companion_ok),
        max_abs_change,
        f"changes_pct={changes}, companion_z={companion_z}",
    )


def verify_halpha_intact(
    pre_cube,
    post_cube,
    wave: Sequence[float],
    companion_yx: tuple[float, float],
    *,
    aperture_radius_px: float,
    max_abs_z: float = 2.0,
) -> VerificationResult:
    """V3: the Halpha window at the companion is unchanged by ZAP."""

    pre = _as_cube(pre_cube)
    post = _as_cube(post_cube)
    pre_spec = extract_aperture_spectrum(pre, companion_yx, aperture_radius_px)
    post_spec = extract_aperture_spectrum(post, companion_yx, aperture_radius_px)
    diff = post_spec - pre_spec
    wave_arr = np.asarray(wave, dtype=np.float64)
    ha_mask = _window_union_mask(wave_arr, (HALPHA_WINDOW,))
    cont_mask = _window_union_mask(wave_arr, HALPHA_CONTINUUM_WINDOWS)
    if not ha_mask.any() or not cont_mask.any():
        raise VerificationError("Halpha or continuum windows select no channels.")
    sigma = robust_sigma(diff[cont_mask])
    signal = float(np.nanmedian(diff[ha_mask]))
    z = float(signal / sigma) if sigma > 0 and np.isfinite(sigma) else float("inf")
    return VerificationResult(
        "v3_halpha_window_change_sigma",
        abs(z) < max_abs_z,
        z,
        f"halpha_diff={signal:.6g}, continuum_sigma={sigma:.6g}, z={z:.3f}",
    )


def verify_sky_residual_symmetry(
    post_cube,
    sky_mask: np.ndarray,
    *,
    max_median_over_rms: float = 0.2,
) -> VerificationResult:
    """V4: post-ZAP sky residuals are roughly symmetric around zero."""

    cube = _as_cube(post_cube)
    values = cube[:, np.asarray(sky_mask, dtype=bool)]
    med = float(np.nanmedian(values))
    sigma = robust_sigma(values)
    ratio = float(abs(med) / sigma) if sigma > 0 and np.isfinite(sigma) else float("inf")
    return VerificationResult(
        "v4_sky_residual_symmetry",
        ratio < max_median_over_rms,
        ratio,
        f"median={med:.6g}, rms={sigma:.6g}, |median|/rms={ratio:.3f}",
    )


def continuum_normalize(spec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(spec, dtype=np.float64)
    med = np.nanmedian(arr)
    if not np.isfinite(med) or med == 0:
        med = 1.0
    return arr / med - 1.0


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    aa = a[valid] - np.nanmean(a[valid])
    bb = b[valid] - np.nanmean(b[valid])
    denom = np.sqrt(np.sum(aa**2) * np.sum(bb**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def verify_eigenspectra_not_stellar(
    eigenspectra: np.ndarray,
    star_spectrum: Sequence[float],
    *,
    max_corr: float = 0.5,
) -> VerificationResult:
    """V5: ZAP eigenspectra should not look like the primary-star spectrum."""

    eig = np.asarray(eigenspectra, dtype=np.float64)
    if eig.ndim == 1:
        eig = eig[None, :]
    star = continuum_normalize(star_spectrum)
    corrs = []
    for row in eig:
        corrs.append(abs(_corrcoef(continuum_normalize(row), star)))
    corr_max = float(np.nanmax(corrs)) if corrs else float("nan")
    return VerificationResult(
        "v5_eigen_vs_star_corr_max",
        corr_max < max_corr,
        corr_max,
        f"max_abs_corr={corr_max:.3f}",
    )


def _extension_sha256(path: str | Path, extname: str) -> str:
    digest = hashlib.sha256()
    with fits.open(path, memmap=True) as hdul:
        if extname not in hdul:
            raise VerificationError(f"{path} has no {extname} extension.")
        data = np.asarray(hdul[extname].data)
        digest.update(np.ascontiguousarray(data).view(np.uint8))
    return digest.hexdigest()


def verify_stat_untouched(
    pre_cube_path: str | Path,
    post_cube_path: str | Path,
    *,
    require_history: bool = True,
) -> VerificationResult:
    """V6: STAT extension is byte-identical after ZAP."""

    pre_hash = _extension_sha256(pre_cube_path, "STAT")
    post_hash = _extension_sha256(post_cube_path, "STAT")
    stat_same = pre_hash == post_hash
    history_ok = True
    if require_history:
        with fits.open(post_cube_path, memmap=True) as hdul:
            history = hdul[0].header.get("HISTORY", "")
            if isinstance(history, str):
                history_text = history
            else:
                history_text = " ".join(str(value) for value in history)
            history_ok = "ZAP" in history_text or "zap" in history_text
    return VerificationResult(
        "v6_stat_untouched",
        bool(stat_same and history_ok),
        stat_same,
        f"stat_same={stat_same}, history_ok={history_ok}",
    )


__all__ = [
    "verify_eigenspectra_not_stellar",
    "verify_halpha_intact",
    "verify_rms_reduction_skylines",
    "verify_sky_residual_symmetry",
    "verify_source_continuum_intact",
    "verify_stat_untouched",
]
