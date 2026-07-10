"""A4 cube-quality measurements for MUSE datacubes.

The functions in this module are target-agnostic. Target-specific coordinates,
catalog photometry, and passbands must be supplied by config or caller code.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
from astropy.io import fits

from musepipe.reduction.sky_zap import (
    CONTINUUM_WINDOWS,
    SKYLINE_WINDOWS,
    compute_sky_residual_metrics,
    wavelength_mask,
)
from musepipe.reduction.verify import circular_aperture_mask, extract_aperture_spectrum
from musepipe.stats import robust_sigma


C_KMS = 299792.458
EXCLUDED_WINDOWS_A = ((5780.0, 6050.0),)


@dataclass(frozen=True)
class Skyline:
    name: str
    wave_A: float


@dataclass(frozen=True)
class LineMeasurement:
    name: str
    lab_wave_A: float
    expected_wave_A: float
    centroid_A: float
    fwhm_A: float
    snr: float
    offset_A: float


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_skylines_csv(path: str | Path) -> list[Skyline]:
    rows: list[Skyline] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        filtered = (line for line in handle if not line.lstrip().startswith("#"))
        for row in csv.DictReader(filtered):
            rows.append(Skyline(row["name"], float(row["wave_A"])))
    return rows


def wavelength_axis_from_header(header: fits.Header, n_wave: int) -> np.ndarray:
    if all(key in header for key in ("CRVAL3", "CDELT3")):
        crpix = float(header.get("CRPIX3", 1.0))
        return float(header["CRVAL3"]) + (
            np.arange(int(n_wave), dtype=np.float64) + 1.0 - crpix
        ) * float(header["CDELT3"])
    if all(key in header for key in ("CRVAL3", "CD3_3")):
        crpix = float(header.get("CRPIX3", 1.0))
        return float(header["CRVAL3"]) + (
            np.arange(int(n_wave), dtype=np.float64) + 1.0 - crpix
        ) * float(header["CD3_3"])
    raise RuntimeError("Could not recover wavelength axis from cube header.")


def detect_wavelength_frame(header: fits.Header) -> tuple[str, float | None]:
    """Detect topocentric/barycentric wavelength frame from common headers."""

    text = " ".join(str(value) for value in header.values()).lower()
    specsys = str(header.get("SPECSYS", "")).upper()
    if "BARY" in specsys:
        return "barycentric", None
    if "TOPO" in specsys:
        return "topocentric", 0.0
    for key in header:
        upper = key.upper()
        if "BARY" in upper or "RVCORR" in upper or "HELIO" in upper:
            try:
                value = float(header[key])
            except (TypeError, ValueError):
                continue
            return "barycentric", value
    if "bary" in text or "rvcorr" in text:
        return "barycentric", None
    if "topocentric" in text:
        return "topocentric", 0.0
    return "unknown", None


def expected_skyline_wave(lab_wave_A: float, *, frame: str, vbary_kms: float | None) -> float:
    """Expected observed skyline position in the cube wavelength frame."""

    if frame == "barycentric":
        if vbary_kms is None:
            raise RuntimeError("Barycentric wavelength frame requires vbary_kms.")
        return float(lab_wave_A) * (1.0 - float(vbary_kms) / C_KMS)
    if frame == "topocentric":
        return float(lab_wave_A)
    raise RuntimeError("Wavelength frame is unknown; do not assume topocentric/barycentric.")


def _line_window(wave: np.ndarray, center: float, half_width_A: float) -> np.ndarray:
    return np.isfinite(wave) & (wave >= center - half_width_A) & (wave <= center + half_width_A)


def measure_line_moments(
    wave: Sequence[float],
    spectrum: Sequence[float],
    *,
    expected_wave_A: float,
    half_width_A: float = 3.0,
) -> tuple[float, float, float]:
    """Measure centroid, FWHM, and S/N with continuum-subtracted moments."""

    wave_arr = np.asarray(wave, dtype=np.float64)
    spec = np.asarray(spectrum, dtype=np.float64)
    mask = _line_window(wave_arr, expected_wave_A, half_width_A)
    if mask.sum() < 3:
        raise RuntimeError("Not enough samples around skyline.")
    local_wave = wave_arr[mask]
    local_spec = spec[mask]
    edge = np.r_[local_spec[: max(1, mask.sum() // 5)], local_spec[-max(1, mask.sum() // 5) :]]
    continuum = float(np.nanmedian(edge))
    profile = local_spec - continuum
    profile[~np.isfinite(profile)] = 0.0
    profile = np.clip(profile, 0.0, None)
    total = float(np.sum(profile))
    noise = robust_sigma(edge - continuum)
    peak = float(np.nanmax(profile)) if profile.size else np.nan
    snr = peak / noise if noise > 0 and np.isfinite(noise) else np.inf
    if total <= 0 or not np.isfinite(total):
        raise RuntimeError("Skyline profile has no positive flux.")
    centroid = float(np.sum(local_wave * profile) / total)
    variance = float(np.sum(((local_wave - centroid) ** 2) * profile) / total)
    sigma = np.sqrt(max(variance, 0.0))
    fwhm = float(2.354820045 * sigma)
    return centroid, fwhm, float(snr)


def measure_skylines(
    wave: Sequence[float],
    sky_spectrum: Sequence[float],
    skylines: Sequence[Skyline],
    *,
    frame: str = "topocentric",
    vbary_kms: float | None = 0.0,
    min_snr: float = 10.0,
    half_width_A: float = 3.0,
) -> list[LineMeasurement]:
    measurements: list[LineMeasurement] = []
    for line in skylines:
        try:
            expected = expected_skyline_wave(line.wave_A, frame=frame, vbary_kms=vbary_kms)
            centroid, fwhm, snr = measure_line_moments(
                wave,
                sky_spectrum,
                expected_wave_A=expected,
                half_width_A=half_width_A,
            )
        except RuntimeError:
            continue
        if snr >= min_snr:
            measurements.append(
                LineMeasurement(
                    name=line.name,
                    lab_wave_A=float(line.wave_A),
                    expected_wave_A=float(expected),
                    centroid_A=float(centroid),
                    fwhm_A=float(fwhm),
                    snr=float(snr),
                    offset_A=float(centroid - expected),
                )
            )
    return measurements


def fit_wavelength_offsets(measurements: Sequence[LineMeasurement]) -> dict[str, object]:
    if not measurements:
        return {
            "n_lines": 0,
            "offset_median_A": np.nan,
            "offset_err_A": np.nan,
            "linear_a_A": np.nan,
            "linear_b": np.nan,
            "residual_scatter_A": np.nan,
            "status": "unavailable",
        }
    x = np.asarray([m.expected_wave_A for m in measurements], dtype=np.float64)
    y = np.asarray([m.offset_A for m in measurements], dtype=np.float64)
    median = float(np.nanmedian(y))
    err = float(robust_sigma(y) / np.sqrt(max(1, y.size)))
    if y.size >= 2:
        coeff = np.polyfit(x, y, 1)
        pred = np.polyval(coeff, x)
        residual_scatter = robust_sigma(y - pred)
        linear_b = float(coeff[0])
        linear_a = float(coeff[1])
    else:
        residual_scatter = 0.0
        linear_b = 0.0
        linear_a = median
    status = status_wavelength(abs(median), residual_scatter)
    return {
        "n_lines": int(y.size),
        "offset_median_A": median,
        "offset_err_A": err,
        "linear_a_A": linear_a,
        "linear_b": linear_b,
        "residual_scatter_A": float(residual_scatter),
        "status": status,
    }


def status_wavelength(abs_offset_A: float, residual_scatter_A: float) -> str:
    if abs_offset_A < 0.1 and residual_scatter_A < 0.1:
        return "green"
    if residual_scatter_A < 0.1:
        return "yellow"
    return "red"


def nominal_muse_fwhm_A(wave_A: Sequence[float]) -> np.ndarray:
    """Approximate nominal MUSE FWHM from R~1770 at 4800A to R~3590 at 9300A."""

    wave = np.asarray(wave_A, dtype=np.float64)
    resolution = np.interp(wave, [4800.0, 9300.0], [1770.0, 3590.0])
    return wave / resolution


def measure_lsf(measurements: Sequence[LineMeasurement]) -> dict[str, object]:
    if len(measurements) < 2:
        return {"table_A_fwhm": [], "poly2_coeffs": [], "max_dev_vs_nominal_pct": np.nan, "status": "unavailable"}
    wave = np.asarray([m.centroid_A for m in measurements], dtype=np.float64)
    fwhm = np.asarray([m.fwhm_A for m in measurements], dtype=np.float64)
    degree = min(2, len(measurements) - 1)
    coeffs = np.polyfit(wave, fwhm, degree).tolist()
    nominal = nominal_muse_fwhm_A(wave)
    dev_pct = np.abs(fwhm / nominal - 1.0) * 100.0
    max_dev = float(np.nanmax(dev_pct))
    if max_dev < 15.0:
        status = "green"
    elif max_dev <= 30.0:
        status = "yellow"
    else:
        status = "red"
    return {
        "table_A_fwhm": [
            {"wave_A": float(w), "fwhm_A": float(f), "nominal_fwhm_A": float(n)}
            for w, f, n in zip(wave, fwhm, nominal)
        ],
        "poly2_coeffs": coeffs,
        "max_dev_vs_nominal_pct": max_dev,
        "status": status,
    }


def synthetic_band_flux(
    wave: Sequence[float],
    spectrum: Sequence[float],
    passband_wave: Sequence[float],
    passband_response: Sequence[float],
) -> float:
    wave_arr = np.asarray(wave, dtype=np.float64)
    spec = np.asarray(spectrum, dtype=np.float64)
    pb_wave = np.asarray(passband_wave, dtype=np.float64)
    response = np.asarray(passband_response, dtype=np.float64)
    interp_response = np.interp(wave_arr, pb_wave, response, left=0.0, right=0.0)
    valid = np.isfinite(spec) & np.isfinite(interp_response) & (interp_response > 0)
    if not valid.any():
        return float("nan")
    return float(np.trapz(spec[valid] * interp_response[valid], wave_arr[valid]) / np.trapz(interp_response[valid], wave_arr[valid]))


def flux_factor_from_reference(synthetic_flux: float, reference_flux: float) -> float:
    if reference_flux <= 0 or not np.isfinite(reference_flux):
        return float("nan")
    return float(synthetic_flux / reference_flux)


def status_flux(factors: Mapping[str, float]) -> str:
    vals = np.asarray([value for value in factors.values() if np.isfinite(value)], dtype=np.float64)
    if vals.size == 0:
        return "unavailable"
    if np.all((vals >= 0.9) & (vals <= 1.1)):
        return "green"
    if np.all((vals >= 0.8) & (vals <= 1.25)):
        return "yellow"
    return "red"


def _module_passband_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "gaia_passbands"


def load_passband_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a Gaia passband CSV (``wavelength_A,response_photon``).

    Skips ``#`` comment lines and the column-name header row.
    """

    wave: list[float] = []
    resp: list[float] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(",")
            try:
                wave.append(float(parts[0]))
                resp.append(float(parts[1]))
            except (ValueError, IndexError):
                continue  # header row (e.g. "wavelength_A,response_photon")
    if not wave:
        raise ValueError(f"No numeric rows in passband file {path}.")
    return np.asarray(wave, dtype=np.float64), np.asarray(resp, dtype=np.float64)


def detect_primary_yx(cube: np.ndarray, wave=None, band_A=None) -> tuple[float, float]:
    """Return the (y, x) of the brightest spaxel (the primary star)."""

    data = np.asarray(cube, dtype=np.float64)
    if band_A is not None and wave is not None:
        wv = np.asarray(wave, dtype=np.float64)
        mask = (wv >= float(band_A[0])) & (wv <= float(band_A[1]))
        image = np.nanmedian(data[mask], axis=0) if mask.any() else np.nanmedian(data, axis=0)
    else:
        image = np.nanmedian(data, axis=0)
    idx = np.unravel_index(int(np.nanargmax(image)), image.shape)
    return (float(idx[0]), float(idx[1]))


def _resolve_passband_dir(config: Mapping[str, object], passband_dir, project_root=None) -> Path:
    if passband_dir is not None:
        candidate = Path(passband_dir)
        if candidate.is_dir():
            return candidate
    cfg_dir = config.get("m3_passband_dir")
    if cfg_dir:
        candidate = Path(cfg_dir)
        if not candidate.is_absolute() and project_root is not None:
            candidate = Path(project_root) / cfg_dir
        if candidate.is_dir():
            return candidate
    return _module_passband_dir()


def compute_m3_flux(
    cube,
    wave,
    config: Mapping[str, object],
    *,
    primary_yx=None,
    aperture_radius_px=None,
    passband_dir=None,
    project_root=None,
) -> dict[str, object]:
    """A4/M3: absolute flux-scale check of the PRIMARY vs its Gaia catalog flux.

    Extracts a large-aperture total-flux spectrum of the primary from a
    flux-calibrated cube, integrates it through the recommended Gaia passband
    (default RP — the least MUSE-truncated), and compares the synthetic band
    flux to the catalog reference F_lambda (Vega-at-pivot). See the ``m3_*``
    config block and ``musepipe/qc/data/gaia_passbands/README.md``.

    Target-agnostic: all star-specific numbers come from ``config``.
    """

    passbands = config.get("m3_passbands")
    if not isinstance(passbands, Mapping) or not passbands:
        return {"status": "unavailable", "reason": "no_m3_passbands_in_config", "variability_caveat": True}

    band = str(config.get("m3_recommended_band") or next(iter(passbands)))
    pb = passbands.get(band)
    if not isinstance(pb, Mapping):
        return {"status": "unavailable", "reason": f"passband_{band}_missing", "variability_caveat": True}

    data = np.asarray(cube, dtype=np.float64)
    wave_arr = np.asarray(wave, dtype=np.float64)
    if data.ndim != 3 or wave_arr.size != data.shape[0]:
        raise ValueError("cube must be (nz,ny,nx) with wave matching the spectral axis.")

    pb_dir = _resolve_passband_dir(config, passband_dir, project_root=project_root)
    pb_file = pb_dir / str(pb["file"])
    if not pb_file.exists():
        return {"status": "unavailable", "reason": f"passband_file_missing:{pb_file}", "variability_caveat": True}
    pb_wave, pb_resp = load_passband_csv(pb_file)

    if primary_yx is None:
        override = config.get("m3_primary_yx")
        if override is not None:
            primary_yx = (float(override[0]), float(override[1]))
        else:
            band_A = config.get("m3_primary_detect_band_A")
            primary_yx = detect_primary_yx(data, wave_arr, band_A=band_A)
    if aperture_radius_px is None:
        aperture_radius_px = float(config.get("m3_aperture_radius_px", config.get("psf_norm_radius_px", 25.0)))

    spectrum = extract_aperture_spectrum(data, (float(primary_yx[0]), float(primary_yx[1])), float(aperture_radius_px))

    flux_unit_cgs = float(config.get("m3_flux_unit_cgs", 1.0e-20))
    synthetic_native = synthetic_band_flux(wave_arr, spectrum, pb_wave, pb_resp)
    synthetic_cgs = synthetic_native * flux_unit_cgs
    reference_cgs = float(pb["ref_flambda_cgs"])
    factor = flux_factor_from_reference(synthetic_cgs, reference_cgs)
    status = status_flux({band: factor})
    overlap = pb.get("muse_overlap_frac")

    return {
        "status": status,
        "band": band,
        "flux_factor": None if not np.isfinite(factor) else float(factor),
        "synthetic_flux_cgs": None if not np.isfinite(synthetic_cgs) else float(synthetic_cgs),
        "reference_flux_cgs": reference_cgs,
        "catalog_mag": pb.get("mag"),
        "primary_yx": [float(primary_yx[0]), float(primary_yx[1])],
        "aperture_radius_px": float(aperture_radius_px),
        "flux_unit_cgs": flux_unit_cgs,
        "muse_overlap_frac": overlap,
        "source": config.get("m3_passband_source"),
        "variability_caveat": True,
        "caveats": [
            config.get("m3_caveat"),
            "Aperture-summed primary flux may miss the broad NFM AO halo (factor biased low); "
            "aperture_radius_px is recorded.",
            None if overlap is None else f"{band} band is {float(overlap) * 100:.1f}% inside the MUSE range.",
        ],
    }


def measure_sky_statistics(
    cube: np.ndarray,
    wave: Sequence[float],
    sky_mask: np.ndarray,
) -> dict[str, object]:
    metrics = compute_sky_residual_metrics(cube, wave, sky_mask)
    values = np.asarray(cube)[:, np.asarray(sky_mask, dtype=bool)]
    med_by_channel = np.nanmedian(values, axis=1)
    rms_by_channel = metrics["channel_rms"]
    cont_mask = wavelength_mask(wave, CONTINUUM_WINDOWS)
    rms_cont = float(np.nanmedian(rms_by_channel[cont_mask]))
    median_bias = float(np.nanmedian(np.abs(med_by_channel[cont_mask])))
    ratio = median_bias / rms_cont if rms_cont > 0 else np.inf
    r_value = float(metrics["R_skyline_over_continuum"])
    if ratio < 0.2 and r_value < 1.5:
        status = "green"
    elif r_value <= 2.0 and ratio < 0.5:
        status = "yellow"
    else:
        status = "red"
    return {
        "rms_continuum": rms_cont,
        "rms_skylines": float(metrics["skyline_rms_median"]),
        "R": r_value,
        "median_bias": median_bias,
        "n_apertures": None,
        "status": status,
    }


def _box3_sums(arr: np.ndarray, centers_yx: Sequence[tuple[int, int]]) -> np.ndarray:
    out = []
    for y, x in centers_yx:
        y1 = max(0, int(y) - 1)
        y2 = min(arr.shape[1], int(y) + 2)
        x1 = max(0, int(x) - 1)
        x2 = min(arr.shape[2], int(x) + 2)
        out.append(np.nansum(arr[:, y1:y2, x1:x2], axis=(1, 2)))
    return np.asarray(out, dtype=np.float64).T


def measure_stat_factors(
    cube: np.ndarray,
    stat: np.ndarray,
    sky_mask: np.ndarray,
    *,
    box_centers_yx: Sequence[tuple[int, int]] | None = None,
) -> dict[str, object]:
    data = np.asarray(cube, dtype=np.float64)
    variance = np.asarray(stat, dtype=np.float64)
    mask = np.asarray(sky_mask, dtype=bool)
    if data.shape != variance.shape:
        raise RuntimeError("DATA and STAT shapes differ.")
    if mask.shape != data.shape[1:]:
        raise RuntimeError("sky_mask shape differs from cube spatial shape.")
    values = data[:, mask]
    stat_values = variance[:, mask]
    emp_var = np.nanvar(values, axis=1, ddof=1)
    med_stat = np.nanmedian(stat_values, axis=1)
    factor_spaxel = emp_var / med_stat
    if box_centers_yx:
        box_data = _box3_sums(data, box_centers_yx)
        box_stat = _box3_sums(variance, box_centers_yx)
        factor_box = np.nanvar(box_data, axis=1, ddof=1) / np.nanmedian(box_stat, axis=1)
    else:
        factor_box = factor_spaxel
    spaxel_median = float(np.nanmedian(factor_spaxel))
    box_median = float(np.nanmedian(factor_box))
    trend = trend_with_lambda(factor_spaxel)
    status = status_stat(spaxel_median, box_median, trend)
    return {
        "factor_spaxel_by_channel": factor_spaxel,
        "factor_box3_by_channel": factor_box,
        "factor_spaxel_median": spaxel_median,
        "factor_box3_median": box_median,
        "trend_with_lambda": trend,
        "status": status,
    }


def trend_with_lambda(values: Sequence[float], *, tolerance: float = 0.2) -> str:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() < 3:
        return "flat"
    x = np.linspace(-1.0, 1.0, arr.size)[finite]
    y = arr[finite]
    slope = np.polyfit(x, y, 1)[0]
    scale = np.nanmedian(np.abs(y))
    if not np.isfinite(scale) or scale == 0:
        return "flat"
    rel = slope / scale
    if rel > tolerance:
        return "rising"
    if rel < -tolerance:
        return "falling"
    return "flat"


def status_stat(spaxel_factor: float, box_factor: float, trend: str) -> str:
    vals = np.asarray([spaxel_factor, box_factor], dtype=np.float64)
    if np.all((vals >= 0.8) & (vals <= 1.5)) and trend == "flat":
        return "green"
    if np.all((vals >= 0.5) & (vals <= 2.0)):
        return "yellow"
    return "red"


def stage00q_qc_skeleton(
    *,
    run_id: str,
    cube_file: str | Path,
    provenance: str,
    sha256: str = "",
    wavelength_frame: str = "unknown",
    vbary_kms: float | None = None,
    skyline_source_cube: str = "same",
) -> dict[str, object]:
    return {
        "stage": "00q_cube_qc",
        "run_id": str(run_id),
        "timestamp_utc": utc_now_iso(),
        "cube": {
            "file": str(cube_file),
            "sha256": sha256,
            "provenance": str(provenance),
            "wavelength_frame": wavelength_frame,
            "vbary_kms": vbary_kms,
            "skyline_source_cube": skyline_source_cube,
        },
        "m1_wavelength": {"status": "unavailable"},
        "m2_lsf": {"status": "unavailable"},
        "m3_flux": {"status": "unavailable", "variability_caveat": True},
        "m4_sky": {"status": "unavailable"},
        "m5_stat": {"status": "unavailable"},
        "excluded_windows_A": [list(window) for window in EXCLUDED_WINDOWS_A],
        "comparison_adp": {"ran": False, "table": ""},
        "open_issues": [],
    }


def check_cube_phase(args: argparse.Namespace) -> int:
    cube = Path(args.cube).expanduser()
    if not cube.exists():
        print(f"ERROR: cube does not exist: {cube}", file=sys.stderr)
        return 2
    with fits.open(cube, memmap=True) as hdul:
        data_hdu = hdul["DATA"] if "DATA" in hdul else next(
            (hdu for hdu in hdul if getattr(hdu.data, "ndim", 0) == 3),
            None,
        )
        if data_hdu is None:
            print(f"ERROR: no 3D DATA cube found in {cube}", file=sys.stderr)
            return 2
        frame, vbary = detect_wavelength_frame(data_hdu.header)
        if frame == "unknown":
            frame, vbary = detect_wavelength_frame(hdul[0].header)
    qc = stage00q_qc_skeleton(
        run_id=args.run_id,
        cube_file=cube,
        provenance=args.provenance,
        sha256="" if args.skip_checksum else sha256_file(cube),
        wavelength_frame=frame,
        vbary_kms=vbary,
        skyline_source_cube=args.skyline_source_cube,
    )
    output = Path(args.qc_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qc["cube"], indent=2))
    return 0


def _load_cube_and_wave(cube_path: Path, data_ext, stat_ext=None):
    with fits.open(cube_path, memmap=True) as hdul:
        data_hdu = None
        for key in ("CUBES", "DATA", "RESIDUALS"):
            if key in hdul and getattr(hdul[key].data, "ndim", 0) in (3, 4):
                data_hdu = hdul[key]
                break
        if data_hdu is None and data_ext is not None:
            try:
                cand = hdul[data_ext]
                if getattr(cand.data, "ndim", 0) in (3, 4):
                    data_hdu = cand
            except (KeyError, IndexError):
                data_hdu = None
        if data_hdu is None:
            data_hdu = next((hdu for hdu in hdul if getattr(hdu.data, "ndim", 0) in (3, 4)), None)
        if data_hdu is None:
            raise ValueError(f"No 3D/4D DATA cube found in {cube_path}.")
        cube = np.asarray(data_hdu.data, dtype=np.float64)
        if cube.ndim == 4:  # e.g. stage02 CUBES (1, nz, ny, nx)
            cube = cube[0]
        header = data_hdu.header
        nz = cube.shape[0]
        if "WAVELENGTH" in hdul:
            wave = np.asarray(hdul["WAVELENGTH"].data, dtype=np.float64).ravel()
            if wave.size != nz:
                wave = None
        else:
            wave = None
        if wave is None:
            crval = float(header.get("CRVAL3", header.get("CRVAL1", 0.0)))
            cdelt = float(header.get("CD3_3", header.get("CDELT3", header.get("CDELT1", 1.0))))
            crpix = float(header.get("CRPIX3", header.get("CRPIX1", 1.0)))
            wave = crval + (np.arange(nz, dtype=np.float64) - (crpix - 1.0)) * cdelt
    return cube, wave


def m3_flux_phase(args: argparse.Namespace) -> int:
    from musepipe.config import load_run_config

    rc = load_run_config(args.run_id, project_root=args.project_root, allow_run_id_mismatch=True)
    cfg = dict(rc.config)
    cube_path = Path(args.cube).expanduser()
    if not cube_path.exists():
        print(f"ERROR: cube does not exist: {cube_path}", file=sys.stderr)
        return 2
    data_ext = args.data_ext if args.data_ext is not None else cfg.get("data_ext", 1)
    cube, wave = _load_cube_and_wave(cube_path, data_ext)
    primary_yx = None
    if args.primary_yx is not None:
        primary_yx = tuple(float(v) for v in args.primary_yx)
    m3 = compute_m3_flux(
        cube,
        wave,
        cfg,
        primary_yx=primary_yx,
        aperture_radius_px=args.aperture_radius,
        project_root=str(rc.paths.project_root),
    )
    m3["cube_file"] = str(cube_path)
    qc_path = Path(args.qc_output)
    if qc_path.exists():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        qc["m3_flux"] = m3
        qc_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
        print(f"Patched m3_flux in {qc_path}: status={m3.get('status')} factor={m3.get('flux_factor')}")
    else:
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(json.dumps({"m3_flux": m3}, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote m3_flux to {qc_path}: status={m3.get('status')} factor={m3.get('flux_factor')}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A4 cube QC helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check-cube", help="Create a stage00q QC skeleton for one cube.")
    check_parser.add_argument("--cube", required=True)
    check_parser.add_argument("--provenance", choices=["raw_reduction", "adp", "historic"], required=True)
    check_parser.add_argument("--run-id", required=True)
    check_parser.add_argument("--qc-output", required=True)
    check_parser.add_argument("--skyline-source-cube", default="same")
    check_parser.add_argument("--skip-checksum", action="store_true")
    check_parser.set_defaults(func=check_cube_phase)

    m3_parser = subparsers.add_parser("m3-flux", help="Compute A4/M3 absolute flux scale from the primary vs Gaia and patch the QC.")
    m3_parser.add_argument("--cube", required=True)
    m3_parser.add_argument("--run-id", required=True)
    m3_parser.add_argument("--project-root", default=None)
    m3_parser.add_argument("--qc-output", required=True)
    m3_parser.add_argument("--data-ext", default=None)
    m3_parser.add_argument("--primary-yx", nargs=2, type=float, default=None, metavar=("Y", "X"))
    m3_parser.add_argument("--aperture-radius", type=float, default=None)
    m3_parser.set_defaults(func=m3_flux_phase)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LineMeasurement",
    "Skyline",
    "detect_wavelength_frame",
    "expected_skyline_wave",
    "fit_wavelength_offsets",
    "compute_m3_flux",
    "detect_primary_yx",
    "flux_factor_from_reference",
    "load_passband_csv",
    "measure_line_moments",
    "measure_lsf",
    "measure_sky_statistics",
    "measure_skylines",
    "measure_stat_factors",
    "nominal_muse_fwhm_A",
    "read_skylines_csv",
    "stage00q_qc_skeleton",
    "status_flux",
    "synthetic_band_flux",
    "wavelength_axis_from_header",
]
