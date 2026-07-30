"""Verification helpers for A1 raw MUSE reductions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from astropy.io import fits


class VerificationError(RuntimeError):
    """Raised when a requested verification cannot be completed."""


@dataclass(frozen=True)
class VerificationResult:
    """Scalar result for one verification check."""

    name: str
    passed: bool
    value: float | bool | None
    message: str


DEFAULT_BAD_RANGES = (
    (5780.0, 6050.0),
    (6860.0, 6960.0),
    (7580.0, 7700.0),
    (8120.0, 8350.0),
)


def _image_hdu_by_name_or_shape(hdul: fits.HDUList, name: str | None = None):
    if name is not None and name in hdul:
        return hdul[name]
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        if data is not None and getattr(data, "ndim", 0) == 3:
            return hdu
    raise VerificationError("No 3D cube HDU found.")


def read_cube_data(path: str | Path, *, name: str | None = "DATA") -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        hdu = _image_hdu_by_name_or_shape(hdul, name)
        return np.asarray(hdu.data, dtype=np.float64), hdu.header.copy()


def read_stat_data(path: str | Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        if "STAT" not in hdul:
            raise VerificationError("STAT extension is missing.")
        return np.asarray(hdul["STAT"].data, dtype=np.float64)


def write_basic_plot(path: str | Path, image: np.ndarray, *, title: str = "") -> None:
    """Write a small diagnostic PNG without making matplotlib a hard import at module load."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(image, origin="lower", interpolation="nearest")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _wavelength_axis(header: fits.Header, nlam: int) -> np.ndarray | None:
    """Reconstruct the linear wavelength axis from a cube header, or None."""

    if "CRVAL3" not in header or "CRPIX3" not in header:
        return None
    step = _spectral_step(header)
    if step is None:
        return None
    return float(header["CRVAL3"]) + (np.arange(nlam) - (float(header["CRPIX3"]) - 1.0)) * step


# Sodium laser guide star (NaLGS) region: masked in ALL spaxels of NFM cubes.
LASER_RANGES = ((5780.0, 6050.0),)


def verify_stat(
    cube_path: str | Path,
    *,
    max_nan_fraction: float = 0.05,
    laser_ranges: Sequence[tuple[float, float]] = LASER_RANGES,
    plot_path: str | Path | None = None,
) -> VerificationResult:
    """V1: DATA and STAT exist, STAT is positive, and NaNs are not massive.

    The NFM DATACUBE has ~11.7% *structural* NaN that is not a defect: (a) the
    NaLGS laser region (``laser_ranges``, default 5780-6050 A) is masked in every
    spaxel, and (b) the NFM field edges are NaN at all wavelengths. The NaN
    fraction is therefore computed only over the *good* region: channels outside
    ``laser_ranges`` and spaxels that have any finite value along the spectral
    axis (i.e. excluding the all-NaN edge mask).
    """

    data, header = read_cube_data(cube_path)
    stat = read_stat_data(cube_path)
    if data.shape != stat.shape:
        return VerificationResult(
            "v1_stat_present",
            False,
            None,
            f"DATA shape {data.shape} != STAT shape {stat.shape}.",
        )

    # (b) edge mask: keep only spaxels with any finite value along the spectral axis.
    spatial_support = np.isfinite(data).any(axis=0) | np.isfinite(stat).any(axis=0)
    if not spatial_support.any():
        return VerificationResult("v1_stat_present", False, None, "No finite spatial support.")

    # (a) laser mask: drop channels whose wavelength falls in a laser range.
    wave = _wavelength_axis(header, data.shape[0])
    good_channel = np.ones(data.shape[0], dtype=bool)
    if wave is not None:
        for lo, hi in laser_ranges:
            good_channel &= ~((wave >= lo) & (wave <= hi))
    if not good_channel.any():
        return VerificationResult("v1_stat_present", False, None, "No good channels outside laser ranges.")

    region = good_channel[:, None, None] & spatial_support[None, :, :]
    bad = region & (~np.isfinite(data) | ~np.isfinite(stat))
    nan_fraction = float(bad.sum() / region.sum())
    finite_stat = stat[region & np.isfinite(stat)]
    stat_positive = bool(finite_stat.size > 0 and np.nanmin(finite_stat) > 0)
    passed = nan_fraction < max_nan_fraction and stat_positive

    if plot_path is not None:
        nan_map = bad.sum(axis=0).astype(float)
        write_basic_plot(plot_path, nan_map, title="NaN count in DATA/STAT (good region)")

    n_laser = int((~good_channel).sum())
    message = (
        f"nan_fraction={nan_fraction:.4f}, stat_positive={stat_positive}, "
        f"laser_channels_excluded={n_laser}, edge_spaxels_excluded={int((~spatial_support).sum())}"
    )
    return VerificationResult("v1_stat_present", passed, nan_fraction, message)


def good_wavelength_mask(
    wave: np.ndarray,
    *,
    bad_ranges: Sequence[tuple[float, float]] = DEFAULT_BAD_RANGES,
) -> np.ndarray:
    mask = np.isfinite(wave)
    for lo, hi in bad_ranges:
        mask &= ~((wave >= lo) & (wave <= hi))
    return mask


def verify_standard_response(
    wave: Sequence[float],
    response: Sequence[float],
    reference: Sequence[float],
    *,
    max_rms: float = 0.05,
    bad_ranges: Sequence[tuple[float, float]] = DEFAULT_BAD_RANGES,
) -> VerificationResult:
    """V2: response residual RMS against a reference response."""

    wave_arr = np.asarray(wave, dtype=np.float64)
    response_arr = np.asarray(response, dtype=np.float64)
    ref_arr = np.asarray(reference, dtype=np.float64)
    if wave_arr.shape != response_arr.shape or response_arr.shape != ref_arr.shape:
        raise VerificationError("wave, response, and reference must have the same shape.")
    mask = good_wavelength_mask(wave_arr, bad_ranges=bad_ranges)
    mask &= np.isfinite(response_arr) & np.isfinite(ref_arr) & (ref_arr != 0)
    if not mask.any():
        raise VerificationError("No good wavelengths available for response verification.")
    residual = response_arr[mask] / ref_arr[mask] - 1.0
    rms = float(np.sqrt(np.mean(residual**2)))
    return VerificationResult(
        "v2_std_residual_rms",
        rms < max_rms,
        rms,
        f"response_rms={rms:.4f}",
    )


def _spectral_step(header: fits.Header) -> float | None:
    for key in ("CD3_3", "CDELT3"):
        if key in header:
            return float(header[key])
    return None


def _has_spatial_wcs(header: fits.Header) -> bool:
    if not all(key in header for key in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2")):
        return False
    has_cd = any(key.startswith("CD1_") or key.startswith("CD2_") for key in header)
    has_cdelt = "CDELT1" in header and "CDELT2" in header
    return bool(has_cd or has_cdelt)


def verify_wcs_headers(
    cube_path: str | Path,
    adp_path: str | Path,
    *,
    crval3_tolerance: float = 1e-6,
    cd3_tolerance: float = 1e-9,
) -> VerificationResult:
    """V3: spatial/spectral WCS exists and spectral solution matches ADP."""

    _, header = read_cube_data(cube_path)
    _, adp_header = read_cube_data(adp_path)
    missing = []
    for key in ("CRVAL3", "CRPIX3"):
        if key not in header:
            missing.append(key)
    if _spectral_step(header) is None:
        missing.append("CD3_3/CDELT3")
    if not _has_spatial_wcs(header):
        missing.append("spatial WCS")
    if missing:
        return VerificationResult("v3_wcs_ok", False, False, f"Missing {', '.join(missing)}.")

    crval_ok = abs(float(header["CRVAL3"]) - float(adp_header["CRVAL3"])) <= crval3_tolerance
    step = _spectral_step(header)
    adp_step = _spectral_step(adp_header)
    step_ok = step is not None and adp_step is not None and abs(step - adp_step) <= cd3_tolerance
    passed = bool(crval_ok and step_ok)
    return VerificationResult(
        "v3_wcs_ok",
        passed,
        passed,
        f"crval3_ok={crval_ok}, spectral_step_ok={step_ok}",
    )


def whitelight_image(cube: np.ndarray) -> np.ndarray:
    return np.nanmean(cube, axis=0)


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(af) & np.isfinite(bf)
    if mask.sum() < 3:
        return float("nan")
    af = af[mask] - np.mean(af[mask])
    bf = bf[mask] - np.mean(bf[mask])
    denom = np.sqrt(np.sum(af**2) * np.sum(bf**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(af * bf) / denom)


def best_integer_shift_correlation(
    image: np.ndarray,
    reference: np.ndarray,
    *,
    max_shift: int = 3,
) -> tuple[float, tuple[int, int]]:
    """Return best Pearson correlation after small integer-pixel shifts."""

    if image.shape != reference.shape:
        raise VerificationError(f"Image shapes differ: {image.shape} vs {reference.shape}")
    best_corr = -np.inf
    best_shift = (0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            y0 = max(0, dy)
            y1 = min(image.shape[0], image.shape[0] + dy)
            x0 = max(0, dx)
            x1 = min(image.shape[1], image.shape[1] + dx)
            ref_y0 = max(0, -dy)
            ref_y1 = ref_y0 + (y1 - y0)
            ref_x0 = max(0, -dx)
            ref_x1 = ref_x0 + (x1 - x0)
            corr = _corrcoef(image[y0:y1, x0:x1], reference[ref_y0:ref_y1, ref_x0:ref_x1])
            if np.isfinite(corr) and corr > best_corr:
                best_corr = corr
                best_shift = (dy, dx)
    return float(best_corr), best_shift


def _peak_registered_windows(
    image: np.ndarray,
    reference: np.ndarray,
    *,
    half_window: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract equal-sized windows centred on the brightest pixel of each image.

    The primary star is the brightest point source in both scenes, so registering
    on its peak makes the comparison independent of the (differing) array sizes and
    field crops. Returns the two windows and the half-size actually used.
    """

    py, px = np.unravel_index(np.nanargmax(image), image.shape)
    ry, rx = np.unravel_index(np.nanargmax(reference), reference.shape)
    reach = min(
        py, image.shape[0] - 1 - py, px, image.shape[1] - 1 - px,
        ry, reference.shape[0] - 1 - ry, rx, reference.shape[1] - 1 - rx,
        half_window,
    )
    if reach < 1:
        raise VerificationError("Brightest pixel too close to an edge to build a comparison window.")
    win = image[py - reach:py + reach + 1, px - reach:px + reach + 1]
    ref = reference[ry - reach:ry + reach + 1, rx - reach:rx + reach + 1]
    return win, ref, int(reach)


def verify_whitelight_vs_adp(
    cube_path: str | Path,
    adp_path: str | Path,
    *,
    min_corr: float = 0.95,
    max_shift: int = 3,
    half_window: int = 20,
) -> VerificationResult:
    """V4: white-light scene matches the ADP after registering on the primary star.

    The re-reduced cube and the ADP have different array sizes (e.g. 323x365 vs
    330x338), so a pixel-to-pixel comparison on the raw grids is impossible.
    Instead we register both white-light images on their brightest pixel (the
    primary star, the same physical source in both), extract a common
    ``(2*half_window+1)`` window, and report the best Pearson correlation over a
    small residual integer-shift search. This is the simplest quantitative,
    shape-independent comparison (chosen over full WCS reprojection).
    """

    cube, _ = read_cube_data(cube_path)
    adp, _ = read_cube_data(adp_path)
    win, ref, reach = _peak_registered_windows(
        whitelight_image(cube),
        whitelight_image(adp),
        half_window=half_window,
    )
    corr, shift = best_integer_shift_correlation(win, ref, max_shift=min(max_shift, reach - 1) if reach > 1 else 0)
    return VerificationResult(
        "v4_adp_whitelight_corr",
        corr > min_corr,
        corr,
        f"corr={corr:.4f}, integer_shift={shift}, window=+-{reach}px",
    )


def verify_whitelight_vs_adp_psf_matched(
    cube_path: str | Path,
    adp_path: str | Path,
    *,
    min_corr: float = 0.95,
    max_shift: int = 3,
    half_window: int = 20,
    max_sigma_px: float = 8.0,
    sigma_step_px: float = 0.1,
    plot_path: str | Path | None = None,
) -> VerificationResult:
    """V4: compare the scene after matching a sharper new cube to the ADP PSF.

    The convolution is diagnostic only and never modifies the science cube.
    """

    from scipy.ndimage import gaussian_filter

    cube, _ = read_cube_data(cube_path)
    adp, _ = read_cube_data(adp_path)
    image, reference, reach = _peak_registered_windows(
        whitelight_image(cube),
        whitelight_image(adp),
        half_window=half_window,
    )
    valid = np.isfinite(image)
    filled = np.where(valid, image, 0.0)
    best_corr = -np.inf
    best_shift = (0, 0)
    best_sigma = 0.0
    best_image = image
    for sigma in np.arange(0.0, max_sigma_px + sigma_step_px / 2.0, sigma_step_px):
        if sigma == 0:
            matched = image
        else:
            weight = gaussian_filter(valid.astype(float), sigma, mode="constant", cval=0.0)
            smooth = gaussian_filter(filled, sigma, mode="constant", cval=0.0)
            matched = np.divide(smooth, weight, out=np.full_like(smooth, np.nan), where=weight > 0)
        corr, shift = best_integer_shift_correlation(
            matched,
            reference,
            max_shift=min(max_shift, reach - 1) if reach > 1 else 0,
        )
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_shift = shift
            best_sigma = float(sigma)
            best_image = matched

    if plot_path is not None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        output = Path(plot_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        image_norm = best_image / np.nanmax(best_image)
        reference_norm = reference / np.nanmax(reference)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_norm, origin="lower", cmap="magma")
        axes[0].set_title(f"New cube matched (sigma={best_sigma:.1f} px)")
        axes[1].imshow(reference_norm, origin="lower", cmap="magma")
        axes[1].set_title("ADP reference")
        axes[2].imshow(image_norm - reference_norm, origin="lower", cmap="coolwarm", vmin=-0.2, vmax=0.2)
        axes[2].set_title("Normalized difference")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(output, dpi=180)
        plt.close(fig)

    return VerificationResult(
        "v4_adp_whitelight_corr",
        best_corr > min_corr,
        float(best_corr),
        f"psf_matched_corr={best_corr:.4f}, sigma_px={best_sigma:.2f}, "
        f"integer_shift={best_shift}, window=+-{reach}px",
    )


def circular_aperture_mask(shape: tuple[int, int], yx: tuple[float, float], radius: float) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float64)
    cy, cx = yx
    return (y - cy) ** 2 + (x - cx) ** 2 <= radius**2


def extract_aperture_spectrum(cube: np.ndarray, yx: tuple[float, float], radius: float) -> np.ndarray:
    mask = circular_aperture_mask(cube.shape[1:], yx, radius)
    if not mask.any():
        raise VerificationError("Aperture contains no pixels.")
    return np.nansum(cube[:, mask], axis=1)


def verify_star_spectrum_vs_adp(
    cube_path: str | Path,
    adp_path: str | Path,
    *,
    star_yx: tuple[float, float],
    adp_star_yx: tuple[float, float] | None = None,
    radius: float,
    min_fraction_in_range: float = 0.80,
    ratio_range: tuple[float, float] = (0.9, 1.1),
) -> VerificationResult:
    """V5: primary-star spectrum ratio against ADP is mostly within range."""

    cube, _ = read_cube_data(cube_path)
    adp, _ = read_cube_data(adp_path)
    spec = extract_aperture_spectrum(cube, star_yx, radius)
    ref = extract_aperture_spectrum(adp, adp_star_yx or star_yx, radius)
    mask = np.isfinite(spec) & np.isfinite(ref) & (ref != 0)
    if not mask.any():
        raise VerificationError("No valid channels for star-spectrum comparison.")
    ratio = spec[mask] / ref[mask]
    lo, hi = ratio_range
    fraction = float(np.mean((ratio >= lo) & (ratio <= hi)))
    rms = float(np.sqrt(np.mean((ratio - np.nanmedian(ratio)) ** 2)))
    passed = fraction >= min_fraction_in_range
    return VerificationResult(
        "v5_adp_star_spec_ratio_rms",
        passed,
        rms,
        f"fraction_in_{lo:.2f}_{hi:.2f}={fraction:.3f}, ratio_rms={rms:.4f}",
    )


def read_mask(path: str | Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        data = np.asarray(hdu.data)
    if data.ndim != 2:
        raise VerificationError(f"Sky mask must be 2D, got {data.shape}.")
    return data.astype(bool)


def verify_sky_mask_clean(
    mask: np.ndarray | None,
    positions_yx: Iterable[tuple[float, float]],
    *,
    radius_px: float,
    true_means_sky: bool = True,
) -> VerificationResult:
    """V6: scipost sky mask does not select primary/companion pixels.

    ``true_means_sky`` documents the convention explicitly. If the caller cannot
    provide a verifiable mask, this function raises a hard error as required by
    the A1 spec.
    """

    if mask is None:
        raise VerificationError("No verifiable sky mask/provenance was provided for V6.")
    sky = np.asarray(mask, dtype=bool)
    if not true_means_sky:
        sky = ~sky

    overlap = 0
    for yx in positions_yx:
        source_region = circular_aperture_mask(sky.shape, yx, radius_px)
        overlap += int(np.count_nonzero(sky & source_region))
    passed = overlap == 0
    return VerificationResult(
        "v6_sky_mask_clean",
        passed,
        passed,
        f"sky_pixels_inside_source_regions={overlap}",
    )


__all__ = [
    "VerificationError",
    "VerificationResult",
    "best_integer_shift_correlation",
    "circular_aperture_mask",
    "extract_aperture_spectrum",
    "good_wavelength_mask",
    "read_cube_data",
    "read_mask",
    "read_stat_data",
    "verify_sky_mask_clean",
    "verify_standard_response",
    "verify_star_spectrum_vs_adp",
    "verify_stat",
    "verify_wcs_headers",
    "verify_whitelight_vs_adp",
    "verify_whitelight_vs_adp_psf_matched",
    "whitelight_image",
]
