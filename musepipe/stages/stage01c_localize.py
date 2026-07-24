"""Stage 01c: canonical target localization for downstream extractions.

This stage measures source positions from the working cube and writes them to
``stage01c_qc.json``. Existing historical stages may still read legacy config
coordinates; new stages should consume this QC instead of hardcoding target
positions. Coordinates are stored as ``pos_yx`` for array indexing, with
``pos_xy`` only where explicitly labelled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import least_squares

from ..config import load_run_config
from ..io import write_json
from ..paths import RunPaths
from ..stats import robust_sigma


class LocalizationError(RuntimeError):
    """Base class for Stage01c localization failures."""


class AmbiguousDetectionError(LocalizationError):
    """Raised when more than one significant peak is found in the search region."""


class SourceNotDetectedError(LocalizationError):
    """Raised when a required source is not detected above threshold."""


class AstrometryValidationError(LocalizationError):
    """Raised when measured astrometry fails configured validation gates."""


@dataclass(frozen=True)
class DetectionResult:
    pos_yx: tuple[float, float]
    err_px: float
    snr: float
    peak_yx: tuple[int, int]
    band_used_A: tuple[float | None, float | None]
    snr_map: np.ndarray
    image: np.ndarray


@dataclass(frozen=True)
class Stage01cProduct:
    qc: dict
    source_rows: list[dict]
    chromatic_rows: list[dict]
    images: dict[str, np.ndarray]


def stage01c_paths(run_id, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    paths = RunPaths.from_project_root(run_id, root)
    plot_dir = paths.plot_stage_dir("stage01c")
    return {
        "paths": paths,
        "stage02_cube_fits": paths.stage_dir / "stage02_xcorr_cube_stack.fits",
        "stage01_cube_fits": paths.stage_dir / "stage01_cropped_cube_stack.fits",
        "stage01c_qc_json": paths.stage_dir / "stage01c_qc.json",
        "stage01c_sources_csv": paths.table_dir / "stage01c_sources.csv",
        "stage01c_chromatic_csv": paths.table_dir / "stage01c_chromatic_centroids.csv",
        "plot_dir": plot_dir,
        "field_plot": plot_dir / "stage01c_field_sources.png",
        "companion_plot": plot_dir / "stage01c_companion_stamp.png",
        "chromatic_plot": plot_dir / "stage01c_chromatic_drift.png",
    }


def stage01c_config_from_run(
    run_id=None,
    *,
    project_root=None,
    overrides=None,
    allow_run_id_mismatch=False,
):
    run_config = load_run_config(
        run_id,
        project_root=project_root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    cfg = dict(run_config.config)
    if overrides:
        cfg.update(overrides)
    cfg["run_id"] = run_config.run_id
    cfg["project_root"] = str(run_config.paths.project_root)
    cfg.setdefault("stage01c_input_preference", "stage02")
    cfg.setdefault("stage01c_primary_bands", None)
    cfg.setdefault("stage01c_bad_windows_A", _bad_windows_from_config(cfg))
    cfg.setdefault("stage01c_red_band_A", [7500.0, 9000.0])
    cfg.setdefault("stage01c_fallback_bands_A", [[6000.0, 7000.0], [None, None]])
    cfg.setdefault("stage01c_detection_snr_min", 5.0)
    cfg.setdefault("stage01c_search_radius_px", 10.0)
    cfg.setdefault("stage01c_field_source_search_radius_px", cfg["stage01c_search_radius_px"])
    cfg.setdefault("stage01c_centroid_stamp_half_size", 4)
    cfg.setdefault("stage01c_primary_stamp_half_size", 6)
    cfg.setdefault("stage01c_chromatic_bins", 6)
    cfg.setdefault("stage01c_chromatic_threshold_px", 0.5)
    cfg.setdefault("stage01c_save_plots", bool(cfg.get("save_intermediate_plots", False)))
    cfg.setdefault("stage01c_validate_astrometry", True)
    cfg.setdefault("stage01c_validate_legacy", True)
    return cfg


def _bad_windows_from_config(cfg):
    windows = cfg.get("stage01c_bad_windows_A") or cfg.get("bad_wavelength_ranges_A")
    if windows is not None:
        return windows
    if cfg.get("drop_wave_min_A") is not None and cfg.get("drop_wave_max_A") is not None:
        return [[float(cfg["drop_wave_min_A"]), float(cfg["drop_wave_max_A"])]]
    return []


def _as_yx(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if "y" in value and "x" in value:
            return (float(value["y"]), float(value["x"]))
        return None
    if len(value) != 2:
        return None
    return (float(value[0]), float(value[1]))


def _as_xy(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return (float(value["x"]), float(value["y"]))
        return None
    if len(value) != 2:
        return None
    return (float(value[0]), float(value[1]))


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_input_cube_path(cfg, paths):
    explicit = cfg.get("stage01c_input_cube_fits")
    if explicit:
        return Path(explicit), "explicit"
    pref = str(cfg.get("stage01c_input_preference", "stage02")).lower()
    if pref == "stage02" and paths["stage02_cube_fits"].exists():
        return paths["stage02_cube_fits"], "stage02"
    if paths["stage01_cube_fits"].exists():
        return paths["stage01_cube_fits"], "stage01_fallback"
    if paths["stage02_cube_fits"].exists():
        return paths["stage02_cube_fits"], "stage02"
    raise FileNotFoundError("No Stage02 or Stage01 cube stack found for Stage01c.")


def load_cube_stack(path):
    with fits.open(path, memmap=True) as hdul:
        if "CUBES" not in hdul or "WAVELENGTH" not in hdul:
            raise RuntimeError("Stage01c expects CUBES and WAVELENGTH HDUs.")
        cubes = hdul["CUBES"].data.astype(np.float32)
        wavelengths = hdul["WAVELENGTH"].data.astype(np.float64)
        header0 = hdul[0].header.copy()
        header_cube = hdul["CUBES"].header.copy()
    if cubes.ndim != 4:
        raise RuntimeError(f"Expected CUBES shape (N,nz,ny,nx), got {cubes.shape}.")
    if wavelengths.ndim != 1 or wavelengths.size != cubes.shape[1]:
        raise RuntimeError("WAVELENGTH axis does not match CUBES spectral dimension.")
    with np.errstate(all="ignore"):
        cube = np.nanmedian(cubes, axis=0).astype(np.float32)
    return cubes, cube, wavelengths, header0, header_cube


def good_wavelength_mask(wavelengths, bad_windows_A=()):
    wave = np.asarray(wavelengths, dtype=np.float64)
    good = np.isfinite(wave)
    for lo, hi in bad_windows_A or ():
        if lo is None or hi is None:
            continue
        good &= ~((wave >= float(lo)) & (wave <= float(hi)))
    return good


def band_mask(wavelengths, band_A, bad_windows_A=()):
    wave = np.asarray(wavelengths, dtype=np.float64)
    good = good_wavelength_mask(wave, bad_windows_A=bad_windows_A)
    lo, hi = band_A
    if lo is not None:
        good &= wave >= float(lo)
    if hi is not None:
        good &= wave <= float(hi)
    return good


def collapse_band(cube_zyx, wavelengths, band_A, bad_windows_A=()):
    mask = band_mask(wavelengths, band_A, bad_windows_A=bad_windows_A)
    if np.count_nonzero(mask) < 2:
        raise RuntimeError(f"Band {band_A} has too few good channels.")
    with np.errstate(all="ignore"):
        return np.nanmedian(np.asarray(cube_zyx)[mask], axis=0).astype(np.float64)


def default_primary_bands(wavelengths, bad_windows_A=(), n_bands=4):
    wave = np.asarray(wavelengths, dtype=np.float64)
    good = good_wavelength_mask(wave, bad_windows_A=bad_windows_A)
    vals = wave[good]
    if vals.size < int(n_bands) * 2:
        raise RuntimeError("Too few wavelengths to build primary centroid bands.")
    edges = np.linspace(float(vals[0]), float(vals[-1]), int(n_bands) + 1)
    return [[float(edges[i]), float(edges[i + 1])] for i in range(int(n_bands))]


def centroid_2d(image, initial_yx=None, *, stamp_half_size=5, fwhm_px=None):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got {img.shape}.")
    ny, nx = img.shape
    if initial_yx is None:
        y0, x0 = np.unravel_index(np.nanargmax(img), img.shape)
    else:
        y0, x0 = initial_yx
    half = int(stamp_half_size)
    yc = int(round(float(y0)))
    xc = int(round(float(x0)))
    y1 = max(0, yc - half)
    y2 = min(ny, yc + half + 1)
    x1 = max(0, xc - half)
    x2 = min(nx, xc + half + 1)
    stamp = img[y1:y2, x1:x2].astype(np.float64)
    finite = np.isfinite(stamp)
    if np.count_nonzero(finite) < 6:
        raise SourceNotDetectedError("Centroid stamp has too few finite pixels.")

    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float64)
    bg0 = float(np.nanmedian(stamp[finite]))
    amp0 = float(np.nanmax(stamp[finite]) - bg0)
    if not np.isfinite(amp0) or amp0 == 0.0:
        amp0 = 1.0
    sigma0 = float(fwhm_px) / 2.355 if fwhm_px is not None else max(1.0, half / 2.0)
    sigma0 = max(0.4, sigma0)

    def resid(pars):
        bg, amp, y, x, sigma = pars
        sigma = max(abs(sigma), 0.25)
        model = bg + amp * np.exp(-0.5 * (((yy - y) ** 2 + (xx - x) ** 2) / sigma**2))
        return (model[finite] - stamp[finite]).ravel()

    lower = [bg0 - abs(amp0) * 5.0, -abs(amp0) * 10.0, y1 - 1.0, x1 - 1.0, 0.25]
    upper = [bg0 + abs(amp0) * 5.0, abs(amp0) * 10.0, y2, x2, max(2.0, half * 2.0)]
    try:
        fit = least_squares(
            resid,
            x0=[bg0, amp0, float(y0), float(x0), sigma0],
            bounds=(lower, upper),
            max_nfev=200,
        )
        bg, amp, y, x, sigma = fit.x
        noise = robust_sigma(resid(fit.x))
        snr = abs(float(amp)) / noise if np.isfinite(noise) and noise > 0 else np.inf
        err = max(0.02, abs(float(sigma)) / max(snr, 1.0))
        if not (y1 - 0.5 <= y <= y2 - 0.5 and x1 - 0.5 <= x <= x2 - 0.5):
            raise RuntimeError("Centroid fit left stamp.")
        return (float(y), float(x)), float(err)
    except Exception:
        weights = stamp - bg0
        weights[~finite] = 0.0
        weights[weights < 0.0] = 0.0
        total = float(np.sum(weights))
        if total <= 0:
            return (float(y0), float(x0)), float("nan")
        y = float(np.sum(yy * weights) / total)
        x = float(np.sum(xx * weights) / total)
        return (y, x), float("nan")


def measure_primary(cube_zyx, wavelengths, cfg):
    bad = cfg.get("stage01c_bad_windows_A", [])
    bands = cfg.get("stage01c_primary_bands")
    if bands is None:
        bands = default_primary_bands(wavelengths, bad_windows_A=bad, n_bands=4)
    approx = _as_yx(
        cfg.get("stage01c_primary_approx_yx")
        or cfg.get("primary_yx")
        or cfg.get("star_yx")
        or cfg.get("reference_geometry_center_yx")
    )
    centers = []
    errs = []
    for band in bands:
        image = collapse_band(cube_zyx, wavelengths, band, bad_windows_A=bad)
        init = approx
        if init is None:
            init = np.unravel_index(np.nanargmax(image), image.shape)
        center, err = centroid_2d(
            image,
            init,
            stamp_half_size=int(cfg.get("stage01c_primary_stamp_half_size", 6)),
        )
        centers.append(center)
        errs.append(err)
    centers = np.asarray(centers, dtype=np.float64)
    pos = tuple(np.nanmedian(centers, axis=0))
    scatter = float(np.nanmax(np.linalg.norm(centers - np.nanmedian(centers, axis=0), axis=1)))
    err = robust_sigma(np.linalg.norm(centers - np.asarray(pos), axis=1))
    if not np.isfinite(err) or err <= 0:
        err = float(np.nanmedian(errs)) if np.isfinite(errs).any() else 0.0
    return {
        "pos_yx": (float(pos[0]), float(pos[1])),
        "err_px": float(err),
        "per_band_scatter_px": scatter,
        "bands_A": [[None if v is None else float(v) for v in band] for band in bands],
        "centers": centers,
    }


def estimate_fwhm_px(image, center_yx, *, max_radius_px=12.0, default=3.0):
    img = np.asarray(image, dtype=np.float64)
    y0, x0 = map(float, center_yx)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    rr = np.sqrt((yy - y0) ** 2 + (xx - x0) ** 2)
    bg = np.nanmedian(img[(rr >= max_radius_px * 0.7) & (rr <= max_radius_px)])
    data = img - bg
    peak = float(np.nanmax(data[rr <= 2.0]))
    if not np.isfinite(peak) or peak <= 0:
        return float(default)
    half = 0.5 * peak
    radii = np.arange(0.0, float(max_radius_px) + 1.0, 1.0)
    prof = []
    for r0, r1 in zip(radii[:-1], radii[1:]):
        vals = data[(rr >= r0) & (rr < r1)]
        prof.append(np.nanmedian(vals) if vals.size else np.nan)
    prof = np.asarray(prof, dtype=float)
    below = np.where(prof <= half)[0]
    if below.size == 0:
        return float(default)
    i = int(below[0])
    radius_half = max(0.5, i + 0.5)
    return float(2.0 * radius_half)


def subtract_azimuthal_median(image, center_yx, *, bin_width_px=1.0, mask=None):
    img = np.asarray(image, dtype=np.float64)
    y0, x0 = map(float, center_yx)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    rr = np.sqrt((yy - y0) ** 2 + (xx - x0) ** 2)
    bins = np.floor(rr / float(bin_width_px)).astype(int)
    model = np.full_like(img, np.nan, dtype=np.float64)
    valid_base = np.isfinite(img)
    if mask is not None:
        valid_base &= ~np.asarray(mask, dtype=bool)
    for b in range(int(np.nanmax(bins)) + 1):
        pix = bins == b
        fit = pix & valid_base
        if np.count_nonzero(fit) < 3:
            continue
        model[pix] = np.nanmedian(img[fit])
    return (img - model).astype(np.float64), model.astype(np.float64)


def pixel_offset_from_sep_pa(sep_arcsec, pa_deg, pixel_scale_arcsec, north_angle_deg=0.0):
    sep_px = float(sep_arcsec) / float(pixel_scale_arcsec)
    theta = math.radians(float(pa_deg) + float(north_angle_deg))
    dy = -sep_px * math.cos(theta)
    dx = sep_px * math.sin(theta)
    return float(dy), float(dx)


def position_from_sep_pa(primary_yx, sep_arcsec, pa_deg, pixel_scale_arcsec, north_angle_deg=0.0):
    dy, dx = pixel_offset_from_sep_pa(sep_arcsec, pa_deg, pixel_scale_arcsec, north_angle_deg)
    return (float(primary_yx[0]) + dy, float(primary_yx[1]) + dx)


def sep_pa_from_positions(primary_yx, source_yx, pixel_scale_arcsec, north_angle_deg=0.0):
    dy = float(source_yx[0]) - float(primary_yx[0])
    dx = float(source_yx[1]) - float(primary_yx[1])
    sep_px = math.hypot(dy, dx)
    pa = math.degrees(math.atan2(dx, -dy)) - float(north_angle_deg)
    pa = pa % 360.0
    return float(sep_px * float(pixel_scale_arcsec)), float(pa)


def resolve_pixel_scale_and_orientation(cfg, header0, header_cube):
    source = "config"
    scale = cfg.get("stage01c_pixel_scale_arcsec", cfg.get("pixel_scale_arcsec"))
    if scale is None:
        for hdr in (header_cube, header0):
            cdelt1 = hdr.get("CDELT1")
            cdelt2 = hdr.get("CDELT2")
            if cdelt1 is not None or cdelt2 is not None:
                values = [abs(float(v)) * 3600.0 for v in (cdelt1, cdelt2) if v is not None]
                if values:
                    scale = float(np.nanmedian(values))
                    source = "header"
                    break
    if scale is None:
        mode = str(cfg.get("muse_mode", cfg.get("ins_mode", ""))).upper()
        if "NFM" in mode:
            scale = 0.025
            source = "mode_NFM_default"
        elif "WFM" in mode:
            scale = 0.2
            source = "mode_WFM_default"
    if scale is None:
        raise LocalizationError(
            "Pixel scale is missing. Set stage01c_pixel_scale_arcsec or pixel_scale_arcsec."
        )

    north_angle = cfg.get("stage01c_north_angle_deg", cfg.get("wcs_north_angle_deg", 0.0))
    orient_source = "config" if "stage01c_north_angle_deg" in cfg or "wcs_north_angle_deg" in cfg else "assumed_north_up"
    return float(scale), float(north_angle), source, orient_source


def _snr_map(filtered, primary_yx, *, ring_width_px=8.0):
    img = np.asarray(filtered, dtype=np.float64)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    rr = np.sqrt((yy - float(primary_yx[0])) ** 2 + (xx - float(primary_yx[1])) ** 2)
    bins = np.floor(rr / float(ring_width_px)).astype(int)
    out = np.full_like(img, np.nan, dtype=np.float64)
    for b in range(int(np.nanmax(bins)) + 1):
        pix = bins == b
        vals = img[pix]
        sig = robust_sigma(vals)
        med = np.nanmedian(vals)
        if np.isfinite(sig) and sig > 0:
            out[pix] = (img[pix] - med) / sig
    if not np.isfinite(out).any():
        sig = robust_sigma(img)
        med = np.nanmedian(img)
        if np.isfinite(sig) and sig > 0:
            out = (img - med) / sig
    return out


def _merge_sub_fwhm_peaks(coords, snr, merge_radius_px):
    """Collapse local maxima closer than one FWHM into their strongest member.

    A bright PSF over a speckly halo (e.g. an unresolved-binary primary) produces
    several 3x3 local maxima within a resolution element. Those are one source,
    not several: two peaks closer than a FWHM are not angularly resolved. Peaks
    are grouped greedily from the strongest; each group keeps its dominant pixel.
    Genuinely separate sources (> one FWHM apart) survive as distinct groups, so
    a real ambiguity still raises downstream.
    """

    ordered = sorted(coords, key=lambda yx: float(snr[yx[0], yx[1]]), reverse=True)
    representatives = []
    for candidate in ordered:
        if all(
            math.hypot(candidate[0] - rep[0], candidate[1] - rep[1]) > float(merge_radius_px)
            for rep in representatives
        ):
            representatives.append(candidate)
    return representatives


def detect_restricted_source(
    image,
    predicted_yx,
    *,
    search_radius_px,
    fwhm_px,
    snr_min=5.0,
    centroid_stamp_half_size=4,
    primary_yx=None,
    band_used_A=(None, None),
    peak_merge_radius_px=None,
):
    img = np.asarray(image, dtype=np.float64)
    sigma = max(float(fwhm_px) / 2.355, 0.5)
    filtered = gaussian_filter(np.nan_to_num(img, nan=np.nanmedian(img)), sigma=sigma)
    snr = _snr_map(filtered, primary_yx if primary_yx is not None else predicted_yx)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    region = (yy - float(predicted_yx[0])) ** 2 + (xx - float(predicted_yx[1])) ** 2 <= float(search_radius_px) ** 2
    if not np.any(region & np.isfinite(snr)):
        raise SourceNotDetectedError("Search region has no finite S/N pixels.")
    local_max = snr == maximum_filter(np.nan_to_num(snr, nan=-np.inf), size=3)
    candidates = region & local_max & (snr >= float(snr_min))
    coords = list(zip(*np.where(candidates)))
    merge_radius = float(peak_merge_radius_px) if peak_merge_radius_px is not None else float(fwhm_px)
    if len(coords) > 1:
        coords = _merge_sub_fwhm_peaks(coords, snr, merge_radius)
    if len(coords) > 1:
        coords = sorted(coords, key=lambda yx: float(snr[yx[0], yx[1]]), reverse=True)
        raise AmbiguousDetectionError(
            f"{len(coords)} significant peaks separated by more than {merge_radius:.1f} px "
            f"(1 FWHM) inside the search region; strongest={coords[:3]}"
        )
    if len(coords) == 0:
        peak = np.unravel_index(np.nanargmax(np.where(region, snr, np.nan)), snr.shape)
        peak_snr = float(snr[peak])
        if not np.isfinite(peak_snr) or peak_snr < float(snr_min):
            raise SourceNotDetectedError(f"Peak S/N {peak_snr:.2f} is below threshold {snr_min:.2f}.")
    else:
        peak = coords[0]
        peak_snr = float(snr[peak])

    center, err = centroid_2d(
        img,
        peak,
        stamp_half_size=int(centroid_stamp_half_size),
        fwhm_px=fwhm_px,
    )
    return DetectionResult(
        pos_yx=(float(center[0]), float(center[1])),
        err_px=float(err) if np.isfinite(err) else 0.0,
        snr=peak_snr,
        peak_yx=(int(peak[0]), int(peak[1])),
        band_used_A=(None if band_used_A[0] is None else float(band_used_A[0]), None if band_used_A[1] is None else float(band_used_A[1])),
        snr_map=snr.astype(np.float32),
        image=img.astype(np.float32),
    )


def _source_mask(shape, centers_yx, radius_px):
    yy, xx = np.indices(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    for center in centers_yx:
        if center is None:
            continue
        y, x = map(float, center)
        mask |= (yy - y) ** 2 + (xx - x) ** 2 <= float(radius_px) ** 2
    return mask


def _candidate_companion_bands(cfg):
    bands = [cfg.get("stage01c_red_band_A", [7500.0, 9000.0])]
    bands.extend(cfg.get("stage01c_fallback_bands_A", []))
    return bands


def locate_companion(cube_zyx, wavelengths, primary, cfg, pixel_scale, north_angle, fwhm_px):
    expected_sep = cfg.get("expected_sep_arcsec")
    expected_pa = cfg.get("expected_pa_deg")
    approx = _as_yx(cfg.get("stage01c_companion_approx_yx") or cfg.get("companion_yx") or cfg.get("object_yx"))
    if expected_sep is not None and expected_pa is not None:
        predicted = position_from_sep_pa(primary["pos_yx"], expected_sep, expected_pa, pixel_scale, north_angle)
    elif approx is not None:
        predicted = approx
    else:
        raise LocalizationError(
            "Companion prediction is missing. Set expected_sep_arcsec/expected_pa_deg or stage01c_companion_approx_yx."
        )

    last_error = None
    for band in _candidate_companion_bands(cfg):
        try:
            image = collapse_band(cube_zyx, wavelengths, band, bad_windows_A=cfg.get("stage01c_bad_windows_A", []))
            mask = _source_mask(image.shape, [primary["pos_yx"]], radius_px=max(2.0, fwhm_px))
            residual, _ = subtract_azimuthal_median(image, primary["pos_yx"], mask=mask)
            return detect_restricted_source(
                residual,
                predicted,
                search_radius_px=float(cfg.get("stage01c_search_radius_px", 10.0)),
                fwhm_px=fwhm_px,
                snr_min=float(cfg.get("stage01c_detection_snr_min", 5.0)),
                centroid_stamp_half_size=int(cfg.get("stage01c_centroid_stamp_half_size", 4)),
                primary_yx=primary["pos_yx"],
                band_used_A=band,
                peak_merge_radius_px=cfg.get("stage01c_peak_merge_radius_px"),
            ), predicted
        except SourceNotDetectedError as exc:
            last_error = exc
    raise SourceNotDetectedError(f"Companion not detected in any configured band: {last_error}")


def locate_field_source(cube_zyx, wavelengths, primary, companion_yx, cfg, fwhm_px):
    approx = _as_yx(
        cfg.get("stage01c_field_source_approx_yx")
        or cfg.get("field_source_yx")
        or cfg.get("second_source_yx")
    )
    if approx is None:
        return None
    band = cfg.get("stage01c_red_band_A", [7500.0, 9000.0])
    image = collapse_band(cube_zyx, wavelengths, band, bad_windows_A=cfg.get("stage01c_bad_windows_A", []))
    mask = _source_mask(image.shape, [primary["pos_yx"], companion_yx], radius_px=max(2.0, fwhm_px))
    residual, _ = subtract_azimuthal_median(image, primary["pos_yx"], mask=mask)
    return detect_restricted_source(
        residual,
        approx,
        search_radius_px=float(cfg.get("stage01c_field_source_search_radius_px", cfg.get("stage01c_search_radius_px", 10.0))),
        fwhm_px=fwhm_px,
        snr_min=float(cfg.get("stage01c_detection_snr_min", 5.0)),
        centroid_stamp_half_size=int(cfg.get("stage01c_centroid_stamp_half_size", 4)),
        primary_yx=primary["pos_yx"],
        band_used_A=band,
    )


def chromatic_centroids(cube_zyx, wavelengths, primary_yx, companion_yx, cfg, fwhm_px):
    wave = np.asarray(wavelengths, dtype=np.float64)
    good = good_wavelength_mask(wave, bad_windows_A=cfg.get("stage01c_bad_windows_A", []))
    vals = wave[good]
    n_bins = int(cfg.get("stage01c_chromatic_bins", 6))
    if vals.size < n_bins * 2:
        return [], {"primary": np.nan, "companion": np.nan}
    edges = np.linspace(float(vals[0]), float(vals[-1]), n_bins + 1)
    rows = []
    primary_pos = []
    companion_pos = []
    for i in range(n_bins):
        band = [float(edges[i]), float(edges[i + 1])]
        image = collapse_band(cube_zyx, wave, band, bad_windows_A=cfg.get("stage01c_bad_windows_A", []))
        p_yx, _ = centroid_2d(
            image,
            primary_yx,
            stamp_half_size=int(cfg.get("stage01c_primary_stamp_half_size", 6)),
            fwhm_px=fwhm_px,
        )
        residual, _ = subtract_azimuthal_median(
            image,
            p_yx,
            mask=_source_mask(image.shape, [p_yx, companion_yx], radius_px=max(2.0, fwhm_px)),
        )
        c_yx, _ = centroid_2d(
            residual,
            companion_yx,
            stamp_half_size=int(cfg.get("stage01c_centroid_stamp_half_size", 4)),
            fwhm_px=fwhm_px,
        )
        primary_pos.append(p_yx)
        companion_pos.append(c_yx)
        rows.append(
            {
                "bin_index": i,
                "wave_min_A": band[0],
                "wave_max_A": band[1],
                "primary_y": float(p_yx[0]),
                "primary_x": float(p_yx[1]),
                "companion_y": float(c_yx[0]),
                "companion_x": float(c_yx[1]),
            }
        )
    return rows, {
        "primary": peak_to_peak_drift(primary_pos),
        "companion": peak_to_peak_drift(companion_pos),
    }


def peak_to_peak_drift(positions_yx):
    arr = np.asarray(positions_yx, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        return np.nan
    dy = float(np.nanmax(arr[:, 0]) - np.nanmin(arr[:, 0]))
    dx = float(np.nanmax(arr[:, 1]) - np.nanmin(arr[:, 1]))
    return float(math.hypot(dy, dx))


def _deviation_sigma(measured, expected, err):
    if expected is None or err is None or float(err) <= 0 or measured is None:
        return None
    return float((float(measured) - float(expected)) / float(err))


def _pa_delta_deg(measured, expected):
    if measured is None or expected is None:
        return None
    return float(((float(measured) - float(expected) + 180.0) % 360.0) - 180.0)


def _legacy_check(companion_yx, cfg):
    legacy_xy = _as_xy(
        cfg.get("stage01c_legacy_companion_xy")
        or cfg.get("legacy_companion_xy")
        or cfg.get("hardcoded_companion_xy")
    )
    if legacy_xy is None:
        return {"hardcoded_companion_xy": None, "distance_px": None}
    legacy_yx = (legacy_xy[1], legacy_xy[0])
    dist = math.hypot(float(companion_yx[0]) - legacy_yx[0], float(companion_yx[1]) - legacy_yx[1])
    return {"hardcoded_companion_xy": [float(legacy_xy[0]), float(legacy_xy[1])], "distance_px": float(dist)}


def compute_stage01c_products(config, *, input_path=None) -> Stage01cProduct:
    cfg = dict(config)
    paths = stage01c_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    if input_path is None:
        cube_path, input_source = _select_input_cube_path(cfg, paths)
    else:
        cube_path, input_source = Path(input_path), "explicit"
    cubes, cube, wavelengths, header0, header_cube = load_cube_stack(cube_path)
    _, nz, ny, nx = cubes.shape
    pixel_scale, north_angle, scale_source, orient_source = resolve_pixel_scale_and_orientation(cfg, header0, header_cube)

    primary = measure_primary(cube, wavelengths, cfg)
    red_image = collapse_band(cube, wavelengths, cfg.get("stage01c_red_band_A", [7500.0, 9000.0]), cfg.get("stage01c_bad_windows_A", []))
    fwhm_px = float(cfg.get("stage01c_psf_fwhm_px") or estimate_fwhm_px(red_image, primary["pos_yx"]))
    companion, predicted_companion_yx = locate_companion(
        cube,
        wavelengths,
        primary,
        cfg,
        pixel_scale,
        north_angle,
        fwhm_px,
    )
    field_source = locate_field_source(cube, wavelengths, primary, companion.pos_yx, cfg, fwhm_px)
    chrom_rows, drift = chromatic_centroids(cube, wavelengths, primary["pos_yx"], companion.pos_yx, cfg, fwhm_px)

    sep, pa = sep_pa_from_positions(primary["pos_yx"], companion.pos_yx, pixel_scale, north_angle)
    sep_err = pixel_scale * math.hypot(float(primary["err_px"]), float(companion.err_px))
    pa_err = math.degrees(
        math.hypot(float(primary["err_px"]), float(companion.err_px))
        / max(math.hypot(companion.pos_yx[0] - primary["pos_yx"][0], companion.pos_yx[1] - primary["pos_yx"][1]), 1e-6)
    )
    expected_sep = cfg.get("expected_sep_arcsec")
    expected_pa = cfg.get("expected_pa_deg")
    expected_sep_err = cfg.get("expected_sep_err")
    expected_pa_err = cfg.get("expected_pa_err")
    sep_sigma = _deviation_sigma(sep, expected_sep, expected_sep_err)
    pa_delta = _pa_delta_deg(pa, expected_pa)
    pa_sigma = None if pa_delta is None or expected_pa_err in (None, 0) else float(pa_delta / float(expected_pa_err))

    open_issues = []
    if orient_source == "assumed_north_up":
        open_issues.append("WCS orientation not present; used north-up pixel convention.")
    if input_source == "stage01_fallback":
        open_issues.append("Stage02 cube missing; localized on Stage01 fallback cube.")
    if field_source is None:
        open_issues.append("No field-source approximate position configured; field_source was not measured.")
    if sep_sigma is not None and abs(sep_sigma) > 2.0:
        msg = f"Measured separation differs from expected by {sep_sigma:.2f} sigma."
        if bool(cfg.get("stage01c_validate_astrometry", True)):
            raise AstrometryValidationError(msg)
        open_issues.append(msg)
    if pa_sigma is not None and abs(pa_sigma) > 2.0:
        msg = f"Measured PA differs from expected by {pa_sigma:.2f} sigma."
        if bool(cfg.get("stage01c_validate_astrometry", True)):
            raise AstrometryValidationError(msg)
        open_issues.append(msg)

    legacy = _legacy_check(companion.pos_yx, cfg)
    if legacy["distance_px"] is not None and legacy["distance_px"] > 2.0:
        msg = (
            f"Measured companion is {legacy['distance_px']:.2f} px from legacy hardcoded "
            "coordinates; historical downstream products may have used the old position."
        )
        if bool(cfg.get("stage01c_validate_legacy", True)):
            raise AstrometryValidationError(msg)
        open_issues.append(msg)

    chrom_needed = bool(
        np.isfinite(drift["companion"])
        and drift["companion"] > float(cfg.get("stage01c_chromatic_threshold_px", 0.5))
    )
    qc = {
        "stage": "01c_target_localization",
        "run_id": cfg["run_id"],
        "input_cube": {"file": str(cube_path), "sha256": _sha256(cube_path), "source": input_source},
        "pixel_scale_arcsec": float(pixel_scale),
        "wcs_orientation": {"north_angle_deg": float(north_angle), "source": orient_source, "pixel_scale_source": scale_source},
        "primary": {
            "pos_yx": [float(primary["pos_yx"][0]), float(primary["pos_yx"][1])],
            "pos_xy": [float(primary["pos_yx"][1]), float(primary["pos_yx"][0])],
            "err_px": float(primary["err_px"]),
            "per_band_scatter_px": float(primary["per_band_scatter_px"]),
            "bands_A": primary["bands_A"],
        },
        "companion": {
            "pos_yx": [float(companion.pos_yx[0]), float(companion.pos_yx[1])],
            "pos_xy": [float(companion.pos_yx[1]), float(companion.pos_yx[0])],
            "err_px": float(companion.err_px),
            "snr_detection": float(companion.snr),
            "band_used_A": [companion.band_used_A[0], companion.band_used_A[1]],
            "predicted_pos_yx": [float(predicted_companion_yx[0]), float(predicted_companion_yx[1])],
        },
        "field_source": None
        if field_source is None
        else {
            "pos_yx": [float(field_source.pos_yx[0]), float(field_source.pos_yx[1])],
            "pos_xy": [float(field_source.pos_yx[1]), float(field_source.pos_yx[0])],
            "err_px": float(field_source.err_px),
            "snr_detection": float(field_source.snr),
        },
        "astrometry": {
            "sep_arcsec": float(sep),
            "sep_err": float(sep_err),
            "pa_deg": float(pa),
            "pa_err": float(pa_err),
            "expected_sep_arcsec": None if expected_sep is None else float(expected_sep),
            "expected_pa_deg": None if expected_pa is None else float(expected_pa),
            "sep_deviation_sigma": sep_sigma,
            "pa_deviation_sigma": pa_sigma,
        },
        "legacy_check": legacy,
        "chromatic": {
            "companion_drift_px_peak_to_peak": _finite_or_none(drift["companion"]),
            "primary_drift_px_peak_to_peak": _finite_or_none(drift["primary"]),
            "chromatic_centroid_needed": chrom_needed,
            "threshold_px": float(cfg.get("stage01c_chromatic_threshold_px", 0.5)),
        },
        "psf": {"fwhm_px": float(fwhm_px), "fwhm_arcsec": float(fwhm_px * pixel_scale)},
        "cube_shape": [int(x) for x in cubes.shape],
        "open_issues": open_issues,
    }
    source_rows = [
        _source_row("primary", primary["pos_yx"], primary["err_px"], None),
        _source_row("companion", companion.pos_yx, companion.err_px, companion.snr),
    ]
    if field_source is not None:
        source_rows.append(_source_row("field_source", field_source.pos_yx, field_source.err_px, field_source.snr))
    return Stage01cProduct(
        qc=qc,
        source_rows=source_rows,
        chromatic_rows=chrom_rows,
        images={
            "red_image": red_image.astype(np.float32),
            "companion_snr": companion.snr_map,
            "companion_image": companion.image,
        },
    )


def _source_row(label, pos_yx, err_px, snr):
    return {
        "source": label,
        "y": float(pos_yx[0]),
        "x": float(pos_yx[1]),
        "err_px": None if err_px is None else float(err_px),
        "snr_detection": None if snr is None else float(snr),
    }


def write_stage01c_products(product: Stage01cProduct, config, paths):
    paths["paths"].ensure_base_dirs()
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    _write_csv(paths["stage01c_sources_csv"], product.source_rows)
    _write_csv(paths["stage01c_chromatic_csv"], product.chromatic_rows)
    if bool(config.get("stage01c_save_plots", config.get("save_intermediate_plots", False))):
        _write_plots(product, paths)
        product.qc["figures"] = {
            "field": str(Path("plots") / "stage01c" / paths["field_plot"].name),
            "companion": str(Path("plots") / "stage01c" / paths["companion_plot"].name),
            "chromatic": str(Path("plots") / "stage01c" / paths["chromatic_plot"].name),
        }
    write_json(paths["stage01c_qc_json"], product.qc)
    return {
        "qc_json": paths["stage01c_qc_json"],
        "sources_csv": paths["stage01c_sources_csv"],
        "chromatic_csv": paths["stage01c_chromatic_csv"],
        "qc": product.qc,
    }


def _write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(product, paths):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    qc = product.qc
    red = product.images["red_image"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(red, origin="lower", cmap="magma")
    for label, color in (("primary", "cyan"), ("companion", "lime")):
        y, x = qc[label]["pos_yx"]
        ax.scatter([x], [y], marker="+", s=90, color=color, label=label)
    if qc["field_source"] is not None:
        y, x = qc["field_source"]["pos_yx"]
        ax.scatter([x], [y], marker="x", s=75, color="white", label="field_source")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(paths["field_plot"], dpi=150)
    plt.close(fig)

    snr = product.images["companion_snr"]
    cy, cx = qc["companion"]["pos_yx"]
    half = 12
    y1 = max(0, int(round(cy)) - half)
    y2 = min(snr.shape[0], int(round(cy)) + half + 1)
    x1 = max(0, int(round(cx)) - half)
    x2 = min(snr.shape[1], int(round(cx)) + half + 1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(snr[y1:y2, x1:x2], origin="lower", cmap="viridis")
    ax.scatter([cx - x1], [cy - y1], marker="+", s=90, color="red")
    fig.tight_layout()
    fig.savefig(paths["companion_plot"], dpi=150)
    plt.close(fig)

    rows = product.chromatic_rows
    if rows:
        wave = np.array([(r["wave_min_A"] + r["wave_max_A"]) / 2.0 for r in rows])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(wave, [r["primary_y"] for r in rows], label="primary y")
        ax.plot(wave, [r["primary_x"] for r in rows], label="primary x")
        ax.plot(wave, [r["companion_y"] for r in rows], label="companion y")
        ax.plot(wave, [r["companion_x"] for r in rows], label="companion x")
        ax.legend(loc="best")
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylabel("Centroid [px]")
        fig.tight_layout()
        fig.savefig(paths["chromatic_plot"], dpi=150)
        plt.close(fig)


def run_stage01c(run_id=None, *, project_root=None, overrides=None, allow_run_id_mismatch=False):
    cfg = stage01c_config_from_run(
        run_id,
        project_root=project_root,
        overrides=overrides,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = stage01c_paths(cfg["run_id"], project_root=cfg.get("project_root"))
    product = compute_stage01c_products(cfg)
    written = write_stage01c_products(product, cfg, paths)
    return {"config": cfg, "paths": paths, "qc": product.qc, "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage01c target localization.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--allow-run-id-mismatch", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args(argv)
    overrides = {"stage01c_save_plots": True} if args.save_plots else None
    result = run_stage01c(
        args.run_id,
        project_root=args.project_root,
        overrides=overrides,
        allow_run_id_mismatch=args.allow_run_id_mismatch,
    )
    print(result["paths"]["stage01c_qc_json"])


__all__ = [
    "AmbiguousDetectionError",
    "AstrometryValidationError",
    "DetectionResult",
    "LocalizationError",
    "SourceNotDetectedError",
    "Stage01cProduct",
    "band_mask",
    "chromatic_centroids",
    "collapse_band",
    "compute_stage01c_products",
    "detect_restricted_source",
    "measure_primary",
    "position_from_sep_pa",
    "run_stage01c",
    "sep_pa_from_positions",
    "stage01c_config_from_run",
    "stage01c_paths",
    "write_stage01c_products",
]


if __name__ == "__main__":
    main()
