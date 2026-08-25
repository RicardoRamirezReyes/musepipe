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
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from astropy.io import fits

from musepipe.reduction.sky_zap import (
    CONTINUUM_WINDOWS,
    SKYLINE_WINDOWS,
    compute_sky_residual_metrics,
    wavelength_mask,
)
from musepipe.io import cube_bunit, resolve_bunit, resolve_flux_unit
from musepipe.reduction.verify import circular_aperture_mask, extract_aperture_spectrum
from musepipe.stats import robust_sigma


C_KMS = 299792.458
EXCLUDED_WINDOWS_A = ((5780.0, 6050.0),)

#: LSF de referencia de MUSE: Bacon et al. 2017, A&A 608, A1 (MUSE HUDF Survey I),
#: Ec. 8 — FWHM(λ) = 5.866e-8 λ² − 9.187e-4 λ + 6.040, con λ y FWHM en Å.
#: Es la mediana de la LSF medida sobre los cubos UDF (dispersión 1–3%, ~0.05 Å).
#: Sustituye a la interpolación lineal en R (1770@4800Å → 3590@9300Å) usada antes,
#: que no procedía de ninguna publicación. Salvedad: es una referencia WFM; se usa
#: como patrón de comparación, no como la LSF del cubo — aguas abajo (E1/E3/G2) se
#: usa siempre la LSF *medida* del airglow.
MUSE_LSF_POLY_BACON2017 = (5.866e-8, -9.187e-4, 6.040)
MUSE_LSF_REFERENCE = "Bacon et al. 2017, A&A 608, A1, Eq. 8"
MUSE_LSF_REFERENCE_SHORT = "Bacon+2017"


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


# Clean, isolated atomic airglow lines with precise single-component laboratory
# wavelengths — the reliable set for the M1 wavelength-offset fit (OH bands are
# unresolved blends, good for the LSF width but not for absolute wavelength).
M1_CLEAN_AIRGLOW = (
    Skyline("OI_5577", 5577.338),
    Skyline("OI_6300", 6300.304),
    Skyline("OI_6363", 6363.776),
)


def read_sky_spectrum_fits(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a MUSE SKY_SPECTRUM product (BinTable lambda/data) as (wave, flux)."""

    with fits.open(path) as hdul:
        table = None
        for hdu in hdul:
            cols = getattr(getattr(hdu, "columns", None), "names", None)
            if cols and "lambda" in cols and "data" in cols:
                table = hdu.data
                break
        if table is None:
            raise ValueError(f"{path} has no BinTable with 'lambda'/'data' columns.")
        wave = np.asarray(table["lambda"], dtype=np.float64)
        flux = np.asarray(table["data"], dtype=np.float64)
    return wave, flux


def measure_m1_m2_from_sky_spectrum(
    sky_wave,
    sky_flux,
    *,
    lsf_skylines: Sequence[Skyline] | None = None,
    m1_skylines: Sequence[Skyline] = M1_CLEAN_AIRGLOW,
    frame: str = "topocentric",
    vbary_kms: float = 0.0,
    min_snr_m1: float = 10.0,
    min_snr_lsf: float = 15.0,
    half_width_A: float = 4.0,
    halpha_A: float = 6562.8,
) -> dict[str, object]:
    """A4 M1 (wavelength) + M2 (LSF) from an airglow sky spectrum.

    Airglow lines are at rest in the TOPOCENTRIC frame, so the offsets measured
    here are the cube's wavelength-solution residual. M1 uses clean isolated
    atomic lines; M2 (LSF) uses all lines above ``min_snr_lsf`` and reports the
    FWHM interpolated to Halpha.
    """

    sky_wave = np.asarray(sky_wave, dtype=np.float64)
    sky_flux = np.asarray(sky_flux, dtype=np.float64)
    if lsf_skylines is None:
        lsf_skylines = list(m1_skylines)

    m1_meas = measure_skylines(
        sky_wave, sky_flux, m1_skylines, frame=frame, vbary_kms=vbary_kms,
        min_snr=min_snr_m1, half_width_A=half_width_A,
    )
    m1 = fit_wavelength_offsets(m1_meas)
    m1["lines"] = [
        {"name": m.name, "expected_A": m.expected_wave_A, "centroid_A": m.centroid_A,
         "offset_A": m.offset_A, "snr": m.snr}
        for m in m1_meas
    ]
    m1["frame"] = frame
    m1["note"] = "airglow rest = topocentric; offset is the cube wavelength-solution residual"

    lsf_meas = measure_skylines(
        sky_wave, sky_flux, lsf_skylines, frame=frame, vbary_kms=vbary_kms,
        min_snr=min_snr_lsf, half_width_A=half_width_A,
    )
    m2 = measure_lsf(lsf_meas)
    lsf_at_halpha = None
    coeffs = m2.get("poly2_coeffs") or []
    if coeffs:
        lsf_at_halpha = float(np.polyval(coeffs, float(halpha_A)))
    elif m2.get("table_A_fwhm"):
        lsf_at_halpha = float(np.nanmedian([r["fwhm_A"] for r in m2["table_A_fwhm"]]))
    m2["lsf_fwhm_at_halpha_A"] = lsf_at_halpha
    m2["n_lines"] = len(lsf_meas)

    return {"m1_wavelength": m1, "m2_lsf": m2, "lsf_fwhm_at_halpha_A": lsf_at_halpha}


def nominal_muse_fwhm_A(wave_A: Sequence[float]) -> np.ndarray:
    """Reference MUSE LSF FWHM in A, from Bacon et al. 2017 (A&A 608, A1), Eq. 8.

    ``FWHM(lambda) = 5.866e-8 lambda^2 - 9.187e-4 lambda + 6.040`` (lambda in A),
    the median LSF measured on the MUSE UDF cubes. See ``MUSE_LSF_REFERENCE``.
    """

    wave = np.asarray(wave_A, dtype=np.float64)
    return np.polyval(MUSE_LSF_POLY_BACON2017, wave)


def measure_lsf(measurements: Sequence[LineMeasurement]) -> dict[str, object]:
    if len(measurements) < 2:
        return {"table_A_fwhm": [], "poly2_coeffs": [], "max_dev_vs_nominal_pct": np.nan,
                "nominal_reference": MUSE_LSF_REFERENCE,
                "nominal_poly_coeffs": list(MUSE_LSF_POLY_BACON2017),
                "status": "unavailable"}
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
        "nominal_reference": MUSE_LSF_REFERENCE,
        "nominal_poly_coeffs": list(MUSE_LSF_POLY_BACON2017),
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


def _growth_curve_total_spectrum(cube, wave, yx, radii, pb_wave, pb_resp, *, tol=0.01):
    """Grow the aperture until the band flux plateaus (captures the AO halo).

    Returns (total_spectrum, plateau_radius, growth_curve). The plateau flux is
    the growth-curve total: once the band flux stops changing by more than
    ``tol`` (relative) it has captured essentially all the source light.
    """

    ny, nx = np.asarray(cube).shape[1:]
    max_r = int(min(yx[0], yx[1], ny - 1 - yx[0], nx - 1 - yx[1]))
    curve = []
    best_spec = None
    best_r = None
    prev = None
    for r in radii:
        if r > max_r:
            continue
        spec = extract_aperture_spectrum(np.asarray(cube, dtype=np.float64), (float(yx[0]), float(yx[1])), float(r))
        band = synthetic_band_flux(wave, spec, pb_wave, pb_resp)
        curve.append({"radius_px": float(r), "band_flux": None if not np.isfinite(band) else float(band)})
        best_spec, best_r = spec, float(r)
        if prev is not None and np.isfinite(band) and prev > 0 and abs(band - prev) / prev < float(tol):
            break  # converged: this (larger) radius is the plateau
        prev = band
    return best_spec, best_r, curve


def _rp_truncation_correction(wave, spectrum, pb_wave, pb_resp, *, fit_lo=8200.0, fit_hi=9300.0):
    """Multiplicative correction for the passband tail beyond the MUSE cutoff.

    Extrapolates the red continuum (log F_lambda linear in log lambda) across
    the missing tail and returns full-band / in-band response-weighted mean.
    Returns (correction, slope) or (1.0, None) if it cannot be estimated.
    """

    wave = np.asarray(wave, dtype=np.float64)
    spec = np.asarray(spectrum, dtype=np.float64)
    pb_wave = np.asarray(pb_wave, dtype=np.float64)
    muse_hi = float(np.nanmax(wave))
    if float(np.nanmax(pb_wave)) <= muse_hi:
        return 1.0, None  # passband fully inside the cube; no truncation
    red = (wave >= fit_lo) & (wave <= fit_hi) & np.isfinite(spec) & (spec > 0)
    if int(np.count_nonzero(red)) < 10:
        return 1.0, None
    slope, intercept = np.polyfit(np.log(wave[red]), np.log(spec[red]), 1)
    grid = np.linspace(float(np.nanmin(pb_wave)), float(np.nanmax(pb_wave)), 2000)
    resp = np.interp(grid, pb_wave, pb_resp, left=0.0, right=0.0)
    in_muse = grid <= muse_hi
    model_tail = np.exp(slope * np.log(grid) + intercept)
    flux_on_grid = np.where(in_muse, np.interp(grid, wave, spec, left=0.0, right=0.0), model_tail)
    denom_full = np.trapz(resp, grid)
    denom_in = np.trapz(resp[in_muse], grid[in_muse])
    if denom_full <= 0 or denom_in <= 0:
        return 1.0, None
    mean_full = np.trapz(flux_on_grid * resp, grid) / denom_full
    mean_in = np.trapz(flux_on_grid[in_muse] * resp[in_muse], grid[in_muse]) / denom_in
    if not np.isfinite(mean_in) or mean_in <= 0:
        return 1.0, None
    return float(mean_full / mean_in), float(slope)


def compute_m3_flux(
    cube,
    wave,
    config: Mapping[str, object],
    *,
    primary_yx=None,
    aperture_radius_px=None,
    aperture_correction=None,
    growth_radii_px=None,
    growth_tol=0.01,
    apply_truncation_correction=None,
    passband_dir=None,
    project_root=None,
    bunit=None,
) -> dict[str, object]:
    """A4/M3: absolute flux-scale check of the PRIMARY vs its Gaia catalog flux.

    Extracts a large-aperture total-flux spectrum of the primary from a
    flux-calibrated cube, integrates it through the recommended Gaia passband
    (default RP — the least MUSE-truncated), and compares the synthetic band
    flux to the catalog reference F_lambda (Vega-at-pivot). See the ``m3_*``
    config block and ``musepipe/qc/data/gaia_passbands/README.md``.

    Target-agnostic: all star-specific numbers come from ``config``.

    ``bunit``: unidad del cubo que se esta midiendo. La comparacion con el
    catalogo es en cgs, asi que sin unidad no hay factor: se resuelve con
    ``musepipe.io.resolve_flux_unit`` (knob ``m3_flux_unit_cgs`` -> ``BUNIT`` del
    cubo -> cubo de entrada del run) y, si no hay ninguna, M3 sale
    ``unavailable`` en vez de suponer la nativa de MUSE — suponerla convertia un
    cubo en otras unidades en un ``flux_factor`` mal por 1e20 sin decirlo.
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
    if aperture_correction is None:
        aperture_correction = str(config.get("m3_aperture_correction", "none"))
    if apply_truncation_correction is None:
        apply_truncation_correction = bool(config.get("m3_apply_truncation_correction", False))

    # Aperture correction: grow the aperture until the band flux plateaus so the
    # AO halo is captured (the finite aperture otherwise biases the factor low).
    growth_curve = None
    plateau_radius = None
    if str(aperture_correction) == "growth_curve":
        if growth_radii_px is None:
            growth_radii_px = config.get("m3_growth_radii_px") or [25, 50, 80, 110, 140, 170, 200]
        spectrum, plateau_radius, growth_curve = _growth_curve_total_spectrum(
            data, wave_arr, (float(primary_yx[0]), float(primary_yx[1])),
            [float(r) for r in growth_radii_px], pb_wave, pb_resp, tol=float(growth_tol),
        )
        effective_radius = plateau_radius
    else:
        spectrum = extract_aperture_spectrum(data, (float(primary_yx[0]), float(primary_yx[1])), float(aperture_radius_px))
        effective_radius = float(aperture_radius_px)

    cube_unit = resolve_bunit(config, stack_bunit=bunit, override_key="m3_bunit")
    try:
        flux_unit_cgs, flux_unit_source = resolve_flux_unit(
            config, bunit=cube_unit, key="m3_flux_unit_cgs"
        )
    except ValueError as exc:
        return {"status": "unavailable", "reason": "flux_unit_unknown",
                "detail": str(exc), "bunit": cube_unit, "variability_caveat": True}
    synthetic_native = synthetic_band_flux(wave_arr, spectrum, pb_wave, pb_resp)

    truncation_correction = 1.0
    truncation_slope = None
    if apply_truncation_correction:
        truncation_correction, truncation_slope = _rp_truncation_correction(wave_arr, spectrum, pb_wave, pb_resp)

    synthetic_cgs = synthetic_native * flux_unit_cgs * truncation_correction
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
        "aperture_correction": str(aperture_correction),
        "aperture_radius_px": None if effective_radius is None else float(effective_radius),
        "plateau_radius_px": plateau_radius,
        "growth_curve": growth_curve,
        "truncation_correction": float(truncation_correction),
        "truncation_slope_flam_vs_lam": truncation_slope,
        "flux_unit_cgs": flux_unit_cgs,
        # De donde salio la unidad, para que D2/E3/G3 puedan citarla o
        # contrastarla con el BUNIT del producto (`io.flux_unit_conflict`).
        "flux_unit_source": flux_unit_source,
        "bunit": cube_unit,
        "muse_overlap_frac": overlap,
        "source": config.get("m3_passband_source"),
        "variability_caveat": True,
        "caveats": [
            config.get("m3_caveat"),
            ("Aperture flux corrected to the growth-curve plateau (AO halo captured)."
             if aperture_correction == "growth_curve"
             else "Fixed aperture may miss the broad NFM AO halo (factor biased low); use growth_curve."),
            None if overlap is None else f"{band} band is {float(overlap) * 100:.1f}% inside the MUSE range"
            + ("" if not apply_truncation_correction else f"; tail corrected x{truncation_correction:.4f}."),
        ],
    }


def measure_sky_statistics(
    cube: np.ndarray,
    wave: Sequence[float],
    sky_mask: np.ndarray,
) -> dict[str, object]:
    metrics = compute_sky_residual_metrics(cube, wave, sky_mask)
    mask = np.asarray(sky_mask, dtype=bool)
    values = np.asarray(cube)[:, mask]
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
        "rms_by_channel": rms_by_channel,
        "median_by_channel": med_by_channel,
        "rms_continuum": rms_cont,
        "rms_skylines": float(metrics["skyline_rms_median"]),
        "R": r_value,
        # Fraction of the FOV used as empty sky; feeds the min_sky_fraction
        # gate of sky_zap.classify_zap_decision (A2 pre-registered rule).
        "sky_fraction": float(mask.mean()),
        "median_bias": median_bias,
        "n_apertures": None,
        "status": status,
    }


def measure_sky_radial_profile(
    cube: np.ndarray,
    wave: Sequence[float],
    source_mask: np.ndarray,
    valid_mask: np.ndarray,
    primary_yx: Sequence[float],
    *,
    bin_px: float = 2.0,
    min_pixels: int = 20,
) -> dict[str, object]:
    """El suelo de continuo por anillos alrededor de la primaria.

    M4 publica su `median_bias` sobre TODA la mascara de cielo (~43 % del campo)
    sin dependencia radial: mide que hay un suelo, pero no puede decir de que
    es. Esto lo separa por radio. Si cae con el radio es el halo AO de la
    primaria; si es plano, es cielo residual — y en un campo NFM de 7.5" la
    mascara de "cielo" esta entera dentro del halo, asi que la pregunta no es
    retorica.

    Reutiliza `measure_sky_statistics` anillo a anillo: el MISMO estimador de
    M4, con las mismas ventanas de continuo, para que el diagnostico sea
    conmensurable con la metrica que explica. La unica diferencia es
    deliberada: aqui el suelo se colapsa CON SIGNO, mientras que M4 publica
    |mediana| y por tanto no distingue sobre- de sub-sustraccion.

    Salvedades que NO se redescubren aqui: la mediana azimutal subestima por la
    curvatura del arco (~0.7 px, `reports/20260727/sesgo_anillo_y_ventana_2026-07-27.md`)
    y el anillo tiene asimetria azimutal por speckles y spikes.
    """

    data = np.asarray(cube)
    sources = np.asarray(source_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if sources.shape != data.shape[1:] or valid.shape != data.shape[1:]:
        raise RuntimeError(
            f"source/valid masks {sources.shape}/{valid.shape} do not match the cube frame "
            f"{data.shape[1:]}: a position is not valid outside its frame."
        )
    cy, cx = float(primary_yx[0]), float(primary_yx[1])
    ny, nx = data.shape[1:]
    yy, xx = np.ogrid[:ny, :nx]
    radius = np.hypot(yy - cy, xx - cx)
    # `r_complete` es hasta donde el anillo cabe ENTERO en el campo. No se corta
    # ahi: la mascara de fuentes de A2 tapa todo el interior (en ROXs 12 b, hasta
    # r ~ 85 px), asi que cortar en r_complete = 99 dejaria 8 anillos y tiraria
    # las esquinas, que es donde esta la mayor parte del cielo. Para una MEDIANA
    # la cobertura parcial no sesga como sesgaria una curva de crecimiento —solo
    # importa si el halo es azimutalmente asimetrico— asi que se mide hasta el
    # borde y cada anillo declara su cobertura.
    r_complete = float(min(cy, ny - 1 - cy, cx, nx - 1 - cx))
    r_max = float(radius.max())
    cont_mask = wavelength_mask(wave, CONTINUUM_WINDOWS)

    edges = np.arange(0.0, r_max + float(bin_px), float(bin_px))
    centers: list[float] = []
    floors: list[float] = []
    rms: list[float] = []
    counts: list[int] = []
    coverage: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_frame = int(((radius >= lo) & (radius < hi)).sum())
        expected = float(np.pi * (hi ** 2 - lo ** 2))
        coverage.append(float(in_frame / expected) if expected > 0 else 0.0)
        ring = (radius >= lo) & (radius < hi) & (~sources) & valid
        n_pix = int(ring.sum())
        counts.append(n_pix)
        if n_pix < int(min_pixels):
            centers.append(float(0.5 * (lo + hi)))
            floors.append(float("nan"))
            rms.append(float("nan"))
            continue
        # El radio MEDIDO de los pixeles que contribuyen, no el centro
        # geometrico del anillo: en un anillo hay mas pixeles fuera que dentro,
        # asi que el mediano cae mas lejos, y usar (lo+hi)/2 sesga la pendiente
        # de una ley de potencias. Es la curvatura del arco de
        # `reports/20260727/sesgo_anillo_y_ventana_2026-07-27.md`, medida en vez
        # de arrastrada.
        centers.append(float(np.median(radius[ring])))
        stats = measure_sky_statistics(data, wave, ring)
        median_by_channel = np.asarray(stats["median_by_channel"], dtype=np.float64)
        floors.append(float(np.nanmedian(median_by_channel[cont_mask])))
        rms.append(float(stats["rms_continuum"]))

    r_centers = np.asarray(centers, dtype=np.float64)
    profile = np.asarray(floors, dtype=np.float64)
    cov = np.asarray(coverage, dtype=np.float64)
    finite = np.isfinite(profile)

    def _fit(selection):
        from musepipe.growth_curve import fit_halo_and_sky

        usable = selection & finite
        if int(usable.sum()) < 10:
            return {"n_annuli": int(usable.sum()), "fit": None}
        radii = np.where(usable, r_centers, np.nan)
        values = np.where(usable, profile, np.nan)
        lo = float(np.nanmin(radii[usable]))
        hi = float(np.nanmax(radii[usable]))
        try:
            fit = fit_halo_and_sky(radii, values, (max(lo, 1.0), hi))
        except Exception:  # noqa: BLE001 - un ajuste que no converge no es un fallo de la medida
            fit = None
        return {
            "n_annuli": int(usable.sum()),
            "r_range_px": [lo, hi],
            "fit": None if fit is None else {"amp": fit[0], "power": fit[1], "sky": fit[2]},
        }

    # El ajuste `A*r^-p + S` NO es estable: con poco brazo radial, p y S se
    # canjean y p se dispara (medido en ROXs 12 b: p = 32.4 con cobertura > 0.4,
    # 3.17 con > 0.1, 2.80 con todo). Publicar un solo numero invitaria a citarlo
    # como si estuviera medido, asi que se publica la familia y se deja ver que
    # no lo esta. Lo que SI es robusto es la caida del suelo con el radio.
    stability = {
        f"coverage_gt_{int(round(threshold * 100)):02d}": _fit(cov > threshold)
        for threshold in (0.9, 0.4, 0.1, 0.0)
    }
    best = stability["coverage_gt_00"]
    halo_fit = best["fit"]
    fit_range = best.get("r_range_px")

    def _decline(selection, threshold):
        usable = selection & finite
        if int(usable.sum()) < 2:
            return None
        radii = r_centers[usable]
        values = profile[usable]
        return {
            "coverage_threshold": threshold,
            "n_annuli": int(usable.sum()),
            "r_inner_px": float(radii[0]),
            "r_outer_px": float(radii[-1]),
            "floor_inner": float(values[0]),
            "floor_outer": float(values[-1]),
            "floor_ratio_inner_over_outer": (
                float(values[0] / values[-1]) if values[-1] != 0 else None
            ),
            "monotonic_decreasing": bool(np.all(np.diff(values) <= 1e-9)),
        }

    # `decline` se mide donde el anillo esta bien cubierto. Mas afuera solo
    # quedan las esquinas del campo (cobertura < 0.15) y ahi el perfil de
    # ROXs 42B b se aplana o repunta: puede ser un suelo real o el sesgo de
    # muestrear cuatro direcciones azimutales. No se mezcla lo uno con lo otro.
    decline = _decline(cov > 0.4, 0.4)
    decline_full = _decline(np.ones_like(cov, dtype=bool), 0.0)

    return {
        "r_centers_px": [float(v) for v in r_centers],
        "floor_by_annulus": [None if not np.isfinite(v) else float(v) for v in profile],
        "rms_by_annulus": [None if not np.isfinite(v) else float(v) for v in rms],
        "n_pixels_by_annulus": counts,
        "azimuthal_coverage_frac": [round(float(v), 4) for v in coverage],
        "bin_px": float(bin_px),
        "r_last_complete_annulus_px": r_complete,
        "r_max_px": r_max,
        "fit_range_px": fit_range,
        "n_annuli_fitted": int(finite.sum()),
        "primary_yx": [cy, cx],
        "halo_fit": halo_fit,
        "halo_fit_stability": stability,
        "decline": decline,
        "decline_full_range": decline_full,
        # Diagnostico, no compuerta: decidir "halo" contra "cielo" pediria un
        # umbral que la spec A4 no define, y la §4 prohibe inventarlos.
        "note": (
            "Signed continuum floor per annulus around the primary; diagnostic only, "
            "no threshold and no semaphore. M4's own status is unaffected. The robust "
            "result is `decline` (how much the floor falls with radius); `halo_fit` is "
            "NOT stable against the fitted radial range -- see halo_fit_stability before "
            "quoting its power."
        ),
    }


def empty_aperture_centers(
    source_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    radius_px: int = 2,
    spacing_px: int = 10,
    max_apertures: int = 32,
) -> list[tuple[int, int]]:
    """Select deterministic empty-aperture centers outside all masked sources."""

    sources = np.asarray(source_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if sources.shape != valid.shape:
        raise RuntimeError("source and valid masks differ in shape")
    radius = int(radius_px)
    margin = radius + 1
    candidates: list[tuple[int, int]] = []
    for y in range(margin, sources.shape[0] - margin, int(spacing_px)):
        for x in range(margin, sources.shape[1] - margin, int(spacing_px)):
            ys = slice(y - radius, y + radius + 1)
            xs = slice(x - radius, x + radius + 1)
            if not sources[ys, xs].any() and valid[ys, xs].all():
                candidates.append((y, x))
    if len(candidates) <= int(max_apertures):
        return candidates
    indices = np.linspace(0, len(candidates) - 1, int(max_apertures), dtype=int)
    return [candidates[index] for index in indices]


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
        source = "cube header"
        if frame == "unknown":
            frame, vbary = detect_wavelength_frame(hdul[0].header)
    if frame == "unknown":
        # La cabecera del cubo de ROXs 42B b no trae SPECSYS ni RVCORR, asi que
        # la deteccion devolvia "unknown" — y ese "unknown" viajaba hasta C2,
        # que descartaba en silencio el knob del config y estampaba
        # `WFRAME = topocentric` en los espectros definitivos de un cubo
        # barycentrico. El knob declarado manda, igual que con la unidad de
        # flujo: nunca un default silencioso.
        frame, vbary, source = _frame_from_config(args, frame, vbary)
    output = Path(args.qc_output)
    if output.exists() and not getattr(args, "force", False):
        existing = json.loads(output.read_text(encoding="utf-8"))
        measured = [k for k in _M_KEYS if _metric_status(existing, k) != "unavailable"]
        if measured:
            print(
                f"ERROR: {output} already has measured metrics ({', '.join(measured)}) and "
                "check-cube rewrites the whole document. Re-run with --force to discard them.",
                file=sys.stderr,
            )
            return 2
    qc = stage00q_qc_skeleton(
        run_id=args.run_id,
        cube_file=cube,
        provenance=args.provenance,
        sha256="" if args.skip_checksum else sha256_file(cube),
        wavelength_frame=frame,
        vbary_kms=vbary,
        skyline_source_cube=args.skyline_source_cube,
    )
    qc["cube"]["wavelength_frame_source"] = source
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qc["cube"], indent=2))
    return 0


def _frame_from_config(args: argparse.Namespace, frame: str, vbary: float | None):
    """El marco de lambda declarado en el config, cuando la cabecera no lo trae."""

    try:
        from musepipe.config import load_run_config

        rc = load_run_config(
            args.run_id, project_root=getattr(args, "project_root", None), allow_run_id_mismatch=True
        )
        cfg = dict(rc.config)
    except Exception:  # noqa: BLE001 - sin config resoluble, se queda en unknown
        return frame, vbary, "undetermined (no SPECSYS/RVCORR in header, no run config)"
    declared = str(cfg.get("wavelength_frame") or "").strip().lower()
    if declared in {"topocentric", "barycentric"}:
        return declared, cfg.get("vbary_kms", vbary), "run config knob 'wavelength_frame'"
    return frame, vbary, "undetermined (no SPECSYS/RVCORR in header, none declared in config)"


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
        aperture_correction=args.aperture_correction,
        apply_truncation_correction=args.truncation_correction or None,
        project_root=str(rc.paths.project_root),
        # La unidad del cubo que M3 mide de verdad, que no tiene por que ser el
        # `cube_files[0]` del config (A4 corre sobre el telurico, el ADP...).
        bunit=cube_bunit(cube_path, ext=data_ext),
    )
    m3["cube_file"] = str(cube_path)
    m3["measured_utc"] = utc_now_iso()
    qc_path = Path(args.qc_output)
    if qc_path.exists():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        qc["m3_flux"] = m3
        qc["timestamp_utc"] = m3["measured_utc"]
        finalize_qc(qc)
        qc_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
        print(f"Patched m3_flux in {qc_path}: status={m3.get('status')} factor={m3.get('flux_factor')}")
    else:
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(json.dumps({"m3_flux": m3}, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote m3_flux to {qc_path}: status={m3.get('status')} factor={m3.get('flux_factor')}")
    return 0


def m1m2_sky_phase(args: argparse.Namespace) -> int:
    lsf_lines = read_skylines_csv(args.skylines) if args.skylines else list(M1_CLEAN_AIRGLOW)
    # Pool the airglow line measurements over all exposures, then fit once — a
    # single robust offset/scatter (M1) and LSF(lambda) poly (M2) from many points.
    m1_meas: list[LineMeasurement] = []
    lsf_meas: list[LineMeasurement] = []
    n_files = 0
    for sky_path in args.sky_spectrum:
        p = Path(sky_path).expanduser()
        if not p.exists():
            print(f"ERROR: sky spectrum does not exist: {p}", file=sys.stderr)
            return 2
        wave, flux = read_sky_spectrum_fits(p)
        m1_meas.extend(measure_skylines(wave, flux, M1_CLEAN_AIRGLOW, frame=args.frame,
                                        vbary_kms=args.vbary_kms, min_snr=10.0, half_width_A=4.0))
        lsf_meas.extend(measure_skylines(wave, flux, lsf_lines, frame=args.frame,
                                         vbary_kms=args.vbary_kms, min_snr=15.0, half_width_A=4.0))
        n_files += 1

    m1 = fit_wavelength_offsets(m1_meas)
    by_line: dict[str, list[float]] = {}
    for m in m1_meas:
        by_line.setdefault(m.name, []).append(m.offset_A)
    m1["offset_by_line_A"] = {k: float(np.median(v)) for k, v in by_line.items()}
    m1["n_measurements"] = len(m1_meas)
    m1["n_exposures"] = n_files
    m1["frame"] = args.frame
    m1["note"] = "airglow rest = topocentric; offset = cube wavelength-solution residual"
    m1["source"] = "MUSE SKY_SPECTRUM (esoreflex scipost cache), pooled over exposures"

    m2 = measure_lsf(lsf_meas)
    coeffs = m2.get("poly2_coeffs") or []
    lsf_at_halpha = float(np.polyval(coeffs, float(args.halpha_A))) if coeffs else None
    m2["lsf_fwhm_at_halpha_A"] = lsf_at_halpha
    m2["n_measurements"] = len(lsf_meas)
    m2["n_exposures"] = n_files
    m2["source"] = "MUSE SKY_SPECTRUM (esoreflex scipost cache) airglow LSF, pooled over exposures"

    stamp = utc_now_iso()
    m1["measured_utc"] = stamp
    m2["measured_utc"] = stamp

    qc_path = Path(args.qc_output)
    payload = {"m1_wavelength": m1, "m2_lsf": m2}
    if qc_path.exists():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        qc.update(payload)
        qc["timestamp_utc"] = stamp
        finalize_qc(qc)
        qc_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    else:
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"m1 offset={m1.get('offset_median_A'):+.4f} A scatter={m1.get('residual_scatter_A'):.4f} ({m1.get('status')}) | "
          f"m2 LSF@Halpha={m2.get('lsf_fwhm_at_halpha_A'):.3f} A ({m2.get('status')}) | "
          f"exposures={n_files}, m1_meas={m1.get('n_measurements')} -> {qc_path}")
    return 0


#: Orden de severidad de los estados de M1-M5. `unavailable` no es un fallo,
#: pero impide declarar el cubo en verde: no se puede certificar lo que no se
#: ha medido.
_M_SEVERITY = ("green", "yellow", "red")
_M_KEYS = ("m1_wavelength", "m2_lsf", "m3_flux", "m4_sky", "m5_stat")


def aggregate_status(qc: MutableMapping[str, object]) -> dict:
    """Estado global del cubo a partir de M1-M5, con quien lo decide.

    Hasta ahora nadie lo calculaba: el `status: "red"` del QC de ROXs 12 b se
    escribio a mano, y el de ROXs 42B b sencillamente no existia, asi que la
    rojez de M5 solo se veia leyendo las cinco metricas una por una.

    Rojo si alguna esta roja; amarillo si alguna esta amarilla; verde solo si
    las cinco estan medidas y verdes. `unavailable` se cuenta aparte y degrada a
    `incomplete`, que no es lo mismo que un fallo.
    """

    states = {}
    for key in _M_KEYS:
        block = qc.get(key) or {}
        states[key] = str(block.get("status") or "unavailable") if isinstance(block, Mapping) else "unavailable"
    missing = sorted(k for k, v in states.items() if v not in _M_SEVERITY)
    worst = "green"
    driver = None
    for key, state in states.items():
        if state in _M_SEVERITY and _M_SEVERITY.index(state) > _M_SEVERITY.index(worst):
            worst, driver = state, key
    if worst == "green" and missing:
        worst = "incomplete"
    return {"status": worst, "driven_by": driver, "by_metric": states, "unavailable": missing}


def patch_status(qc: MutableMapping[str, object]) -> str:
    summary = aggregate_status(qc)
    qc["status"] = summary["status"]
    qc["status_detail"] = summary
    return summary["status"]


#: Alias historico. `m4m5_phase` lo llamaba cuando era privado y era su unico
#: uso; ahora `finalize_qc` tambien lo necesita desde fuera del modulo.
_patch_status = patch_status


_METRIC_LABEL = {
    "m1_wavelength": "M1 (wavelength solution)",
    "m2_lsf": "M2 (LSF)",
    "m3_flux": "M3 (flux scale)",
    "m4_sky": "M4 (sky statistics)",
    "m5_stat": "M5 (STAT validation)",
}


def _metric_status(qc: Mapping[str, object], key: str) -> str:
    block = qc.get(key)
    if not isinstance(block, Mapping):
        return "unavailable"
    return str(block.get("status") or "unavailable")


def _metric_block(qc: Mapping[str, object], key: str) -> Mapping[str, object]:
    block = qc.get(key)
    return block if isinstance(block, Mapping) else {}


def _fmt(value: object, spec: str = ".3f") -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "unavailable"
    if not np.isfinite(number):
        return "unavailable"
    return format(number, spec)


def derive_open_issues(qc: Mapping[str, object]) -> list[dict]:
    """Los `open_issues` de A4, derivados de M1-M5 en vez de escritos a mano.

    Hasta ahora no los escribia nadie: el esqueleto los dejaba en `[]`, y los
    dos que tenia ROXs 12 b eran manuscritos — asi que ROXs 42B b salia con la
    lista vacia teniendo la MISMA M4 en amarillo y la MISMA M5 en rojo. Que
    sean derivados es lo que hace que los seis runs digan lo mismo del mismo
    hecho.

    Cada entrada declara su `priority`, y eso es lo que impide que una nota
    informativa acabe publicada como bloqueante: F1 escalaba a `blocking`
    cualquier `open_issue` de una etapa roja, incluida la de M4 cuyo propio
    texto decia «retain as a systematic diagnostic»
    (ver `report.aggregate_open_issues`).
    """

    issues: list[dict] = []

    for key in _M_KEYS:
        state = _metric_status(qc, key)
        block = _metric_block(qc, key)
        label = _METRIC_LABEL[key]

        if state == "unavailable":
            if key == "m5_stat":
                text = (
                    f"{label} is unavailable: the STAT factors were never measured. Not measuring "
                    "is not passing — C2/C3 fall back to native STAT with stat_factor_box3 = 1.0, "
                    "which is an error budget half the real one."
                )
            else:
                text = f"{label} is unavailable: not measured, so the cube cannot be certified on it."
            issues.append({"issue": text, "priority": "major", "metric": key, "source": "derived"})
            continue

        if state == "red":
            if key == "m5_stat":
                # La limitacion ya aceptada por la politica de compuerta de F1
                # (`ACCEPTED_LIMITATIONS["A4_cube_qc"]["m5_stat.status"]`): es la
                # covarianza del remuestreo, inherente al formato del cubo.
                text = (
                    f"M5 is red: native STAT underestimates empirical variance by factors "
                    f"{_fmt(block.get('factor_spaxel_median'))} at spaxel scale and "
                    f"{_fmt(block.get('factor_box3_median'))} for 3x3 apertures. Downstream "
                    "significance must use identically processed empirical controls, never "
                    "native STAT as sigma."
                )
                issues.append({"issue": text, "priority": "accepted", "metric": key, "source": "derived"})
            else:
                issues.append({
                    "issue": f"{label} is red; the cube fails this gate.",
                    "priority": "blocking",
                    "metric": key,
                    "source": "derived",
                })
            continue

        if state == "yellow":
            if key == "m4_sky":
                rms = block.get("rms_continuum")
                bias = block.get("median_bias")
                ratio = None
                try:
                    ratio = float(bias) / float(rms)  # type: ignore[arg-type]
                except (TypeError, ValueError, ZeroDivisionError):
                    ratio = None
                text = (
                    f"M4 is yellow: residual-sky median bias is {_fmt(ratio)} times the continuum "
                    f"RMS despite R={_fmt(block.get('R'), '.3f')}; retain as a systematic diagnostic."
                )
                radial = block.get("radial")
                if isinstance(radial, Mapping):
                    decline = radial.get("decline")
                    if isinstance(decline, Mapping):
                        # Se cita la CAIDA, que es lo robusto, y no la potencia
                        # del ajuste, que depende del brazo radial admitido.
                        text += (
                            " Radial diagnostic: the continuum floor falls x"
                            f"{_fmt(decline.get('floor_ratio_inner_over_outer'), '.2f')} between "
                            f"r={_fmt(decline.get('r_inner_px'), '.0f')} and "
                            f"r={_fmt(decline.get('r_outer_px'), '.0f')} px from the primary, so it "
                            "is not a flat sky residual (see m4_sky.radial)."
                        )
                issues.append({"issue": text, "priority": "info", "metric": key, "source": "derived"})
            elif key == "m2_lsf":
                issues.append({
                    "issue": (
                        f"M2 is yellow: the measured LSF deviates "
                        f"{_fmt(block.get('max_dev_vs_nominal_pct'), '.1f')}% from "
                        f"{MUSE_LSF_REFERENCE_SHORT}; downstream stages use the MEASURED LSF, "
                        "so this is a comparison caveat, not a defect."
                    ),
                    "priority": "info",
                    "metric": key,
                    "source": "derived",
                })
            else:
                issues.append({
                    "issue": f"{label} is yellow; retain as a systematic diagnostic.",
                    "priority": "info",
                    "metric": key,
                    "source": "derived",
                })

    return issues


def _normalized_issue_text(text: object) -> str:
    return " ".join(str(text).split()).lower()


def _preserved_open_issues(previous: object, derived: Sequence[Mapping[str, object]]) -> list[dict]:
    """Las notas manuscritas que `derive_open_issues` no puede reconstruir.

    `ROXs12b_raw` guarda cuatro que no salen de ningun campo del QC (que
    `gaia_passbands/` solo tiene un README, que `factor_box3` puede salir Inf
    en los bordes NaN...). Derivar no puede significar tirarlas, asi que se
    conservan marcadas `source: manual`.

    Lo unico que se descarta es el DUPLICADO LITERAL: las dos notas de
    ROXs 12 b son palabra por palabra lo que la derivacion reconstruye, y
    conservarlas seria repetir el mismo hecho dos veces, que es justo el
    defecto que este trabajo arregla. La comparacion es por texto exacto
    (normalizando espacios) a proposito: una nota que diga algo distinto,
    aunque sea de la misma metrica, se queda.
    """

    if not isinstance(previous, list):
        return []
    derived_texts = {_normalized_issue_text(item.get("issue")) for item in derived}
    kept: list[dict] = []
    for item in previous:
        if isinstance(item, Mapping):
            if str(item.get("source", "")) == "derived":
                continue
            entry = dict(item)
            entry.setdefault("priority", "info")
            entry.setdefault("source", "manual")
        elif str(item).strip():
            entry = {"issue": str(item), "priority": "info", "source": "manual"}
        else:
            continue
        if _normalized_issue_text(entry.get("issue")) in derived_texts:
            continue
        kept.append(entry)
    return kept


def derive_downstream_decision(qc: Mapping[str, object]) -> dict | None:
    """La decision de ruido que M5 impone, derivada en vez de manuscrita.

    `docs/structure/04_results_block_a.md` la presenta como algo que «A4
    escribe» y la llama el origen formal de las aperturas de control, pero no
    la escribia ningun codigo del repo: solo ROXs 12 b la tenia, a mano.
    """

    state = _metric_status(qc, "m5_stat")
    if state == "unavailable":
        return None
    previous = qc.get("downstream_decision")
    previous = previous if isinstance(previous, Mapping) else {}
    empirical = state in {"red", "yellow"}
    decision = {
        "decision": "use_empirical_controls" if empirical else "native_stat_ok",
        "native_stat_as_sigma": not empirical,
        "require_identically_processed_controls": empirical,
        "driven_by": f"m5_stat.status={state}",
        "reference": "docs/noise_model.md",
    }
    # No se reescribe la fecha de una decision ya tomada. Sin ella, la de la
    # medida: inventar `utc_now_iso()` aqui haria que `finalize` no fuese
    # idempotente.
    approved = previous.get("approved_utc") or _metrics_measured_utc(qc)
    if approved:
        decision["approved_utc"] = str(approved)
    return decision


def _metrics_measured_utc(qc: Mapping[str, object]) -> str | None:
    stamps = []
    for key in _M_KEYS:
        stamp = _metric_block(qc, key).get("measured_utc")
        if stamp:
            stamps.append(str(stamp))
    return max(stamps) if stamps else None


def finalize_qc(qc: MutableMapping[str, object]) -> MutableMapping[str, object]:
    """Recalcula todo lo DERIVADO del QC de A4. Funcion pura JSON -> JSON.

    No abre el cubo: cuanto necesita esta ya en el documento. Eso es lo que
    permite homogeneizar los seis QC en segundos en vez de re-medir 2 GB por
    run, y lo que la hace verificable — sobre un QC ya correcto no cambia un
    byte.
    """

    patch_status(qc)
    derived = derive_open_issues(qc)
    qc["open_issues"] = derived + _preserved_open_issues(qc.get("open_issues"), derived)
    decision = derive_downstream_decision(qc)
    if decision is not None:
        qc["downstream_decision"] = decision
    measured = _metrics_measured_utc(qc)
    if measured:
        qc["metrics_measured_utc"] = measured
    return qc


#: Cuanto puede alejarse del pico de brillo una posicion declarada antes de
#: considerarse de OTRO frame. Cinco pixeles: el pico esta a nivel de spaxel y
#: un centroide fino no se aleja tanto.
PRIMARY_YX_PEAK_TOLERANCE_PX = 5.0


def resolve_primary_yx(data: np.ndarray, wave, declared=None):
    """La primaria de ESTE cubo, verificada contra su pico de brillo.

    No se toma `m3_primary_yx` a ciegas: en ROXs 12 b esa clave vale [166, 168]
    y su propia nota dice que es del `cube_telcorr.fits` SIN recortar, de
    338x330 — mientras que A4 mide sobre un `DATACUBE_FINAL.fits` de 200x200,
    donde la primaria esta en [100, 100]. Usarla habria centrado los anillos en
    una esquina vacia sin que nada protestara. Una posicion no vale fuera de su
    frame, asi que aqui se comprueba contra el dato en vez de creerse.
    """

    peak = detect_primary_yx(data, wave)
    if declared is None:
        return peak, "brightest spaxel of this cube"
    candidate = (float(declared[0]), float(declared[1]))
    offset = float(np.hypot(candidate[0] - peak[0], candidate[1] - peak[1]))
    if offset > PRIMARY_YX_PEAK_TOLERANCE_PX:
        return None, (
            f"declared primary {list(candidate)} is {offset:.1f} px from this cube's brightness "
            f"peak {list(peak)}: it belongs to a different frame"
        )
    return candidate, f"declared, {offset:.2f} px from this cube's brightness peak"


def m4m5_phase(args: argparse.Namespace) -> int:
    cube_path = Path(args.cube).expanduser()
    mask_path = Path(args.source_mask).expanduser()
    if not cube_path.exists() or not mask_path.exists():
        print("ERROR: cube and source mask must exist", file=sys.stderr)
        return 2
    with fits.open(cube_path, memmap=True) as hdul:
        data_hdu = hdul["DATA"] if "DATA" in hdul else hdul[1]
        stat_hdu = hdul["STAT"] if "STAT" in hdul else None
        if stat_hdu is None:
            print("ERROR: cube has no STAT extension", file=sys.stderr)
            return 2
        data = np.asarray(data_hdu.data, dtype=np.float64)
        stat = np.asarray(stat_hdu.data, dtype=np.float64)
        wave = wavelength_axis_from_header(data_hdu.header, data.shape[0])
    with fits.open(mask_path, memmap=True) as hdul:
        source_mask = np.asarray(hdul[0].data, dtype=bool)

    valid_mask = np.mean(np.isfinite(data[::200]), axis=0) >= 0.9
    centers = empty_aperture_centers(
        source_mask,
        valid_mask,
        radius_px=args.aperture_radius,
        spacing_px=args.spacing_px,
        max_apertures=args.max_apertures,
    )
    if len(centers) < 10:
        print(f"ERROR: only {len(centers)} valid empty apertures; A4 requires at least 10", file=sys.stderr)
        return 2

    yy, xx = np.ogrid[: data.shape[1], : data.shape[2]]
    sky_mask = np.zeros(data.shape[1:], dtype=bool)
    for y, x in centers:
        sky_mask |= (yy - y) ** 2 + (xx - x) ** 2 <= float(args.aperture_radius) ** 2

    m4 = measure_sky_statistics(data, wave, sky_mask)
    m5 = measure_stat_factors(data, stat, sky_mask, box_centers_yx=centers)
    m4["sampled_pixel_fraction"] = m4.pop("sky_fraction")
    m4["sky_fraction"] = float(np.mean((~source_mask) & valid_mask))

    primary_yx, primary_source = resolve_primary_yx(data, wave, getattr(args, "primary_yx", None))
    if primary_yx is None:
        print(f"ERROR: {primary_source}", file=sys.stderr)
        return 2
    radial = measure_sky_radial_profile(
        data, wave, source_mask, valid_mask, primary_yx, bin_px=args.radial_bin_px
    )
    radial["primary_yx_source"] = primary_source
    m4["radial"] = radial
    products_dir = Path(args.products_dir)
    products_dir.mkdir(parents=True, exist_ok=True)
    curves_path = products_dir / "stage00q_m4_m5_curves.npz"
    np.savez_compressed(
        curves_path,
        wavelength_A=wave,
        m4_rms=np.asarray(m4.pop("rms_by_channel")),
        m4_median=np.asarray(m4.pop("median_by_channel")),
        m5_factor_spaxel=np.asarray(m5.pop("factor_spaxel_by_channel")),
        m5_factor_box3=np.asarray(m5.pop("factor_box3_by_channel")),
        aperture_centers_yx=np.asarray(centers, dtype=np.int16),
    )
    m4["n_apertures"] = len(centers)
    m4["aperture_radius_px"] = float(args.aperture_radius)
    m4["source_mask"] = str(mask_path)
    m4["curves"] = str(curves_path)
    m5["n_apertures"] = len(centers)
    m5["curves"] = str(curves_path)
    stamp = utc_now_iso()
    m4["measured_utc"] = stamp
    m5["measured_utc"] = stamp

    qc_path = Path(args.qc_output)
    qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}
    qc["m4_sky"] = m4
    qc["m5_stat"] = m5
    qc["timestamp_utc"] = stamp
    finalize_qc(qc)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(
        f"m4 R={m4['R']:.4f} bias/rms={m4['median_bias'] / m4['rms_continuum']:.4f} "
        f"({m4['status']}) | m5 spaxel={m5['factor_spaxel_median']:.3f} "
        f"box3={m5['factor_box3_median']:.3f} ({m5['status']}) | apertures={len(centers)}"
    )
    return 0


def resolve_qc_wavelength_frame(qc: MutableMapping[str, object], cfg: Mapping[str, object]) -> bool:
    """Rellena `cube.wavelength_frame` desde el config cuando A4 dice `unknown`.

    La §5 de la spec declara el enum `barycentric|topocentric`: `"unknown"` esta
    fuera de contrato, y viajaba hasta C2 para estampar `WFRAME = topocentric`
    en los espectros de un cubo barycentrico. Solo rellena lo que falta: un
    marco ya resuelto no se toca, asi que la funcion es idempotente.
    """

    cube = qc.get("cube")
    if not isinstance(cube, MutableMapping):
        return False
    current = str(cube.get("wavelength_frame") or "").strip().lower()
    if current in {"topocentric", "barycentric"}:
        return False
    declared = str(cfg.get("wavelength_frame") or "").strip().lower()
    if declared not in {"topocentric", "barycentric"}:
        return False
    cube["wavelength_frame"] = declared
    if cube.get("vbary_kms") is None and cfg.get("vbary_kms") is not None:
        cube["vbary_kms"] = cfg.get("vbary_kms")
    cube["wavelength_frame_source"] = "run config knob 'wavelength_frame'"
    return True


def finalize_phase(args: argparse.Namespace) -> int:
    qc_path = Path(args.qc_output)
    if not qc_path.exists():
        print(f"ERROR: QC does not exist: {qc_path}", file=sys.stderr)
        return 2
    before = qc_path.read_text(encoding="utf-8")
    qc = json.loads(before)
    if getattr(args, "run_id", None):
        from musepipe.config import load_run_config

        rc = load_run_config(
            args.run_id, project_root=getattr(args, "project_root", None), allow_run_id_mismatch=True
        )
        if resolve_qc_wavelength_frame(qc, dict(rc.config)):
            print(f"  wavelength_frame resolved from config: {qc['cube']['wavelength_frame']}")
    finalize_qc(qc)
    after = json.dumps(qc, indent=2) + "\n"
    if after == before:
        print(f"{qc_path}: already finalized, unchanged")
        return 0
    qc_path.write_text(after, encoding="utf-8")
    detail = qc.get("status_detail") or {}
    issues = qc.get("open_issues") or []
    print(
        f"{qc_path}: status={qc.get('status')} (driven_by={detail.get('driven_by')}) "
        f"| open_issues={len(issues)} "
        f"| decision={(qc.get('downstream_decision') or {}).get('decision')}"
    )
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
    check_parser.add_argument("--project-root", default=None)
    check_parser.add_argument(
        "--force", action="store_true",
        help="Rewrite the QC even if it already carries measured M1-M5 (they are discarded).",
    )
    check_parser.set_defaults(func=check_cube_phase)

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="Recompute everything DERIVED in a stage00q QC (status, open_issues, decision). Pure JSON, no cube.",
    )
    finalize_parser.add_argument("--qc-output", required=True)
    finalize_parser.add_argument(
        "--run-id", default=None,
        help="If given, resolve cube.wavelength_frame from the run config when A4 left it unknown.",
    )
    finalize_parser.add_argument("--project-root", default=None)
    finalize_parser.set_defaults(func=finalize_phase)

    m3_parser = subparsers.add_parser("m3-flux", help="Compute A4/M3 absolute flux scale from the primary vs Gaia and patch the QC.")
    m3_parser.add_argument("--cube", required=True)
    m3_parser.add_argument("--run-id", required=True)
    m3_parser.add_argument("--project-root", default=None)
    m3_parser.add_argument("--qc-output", required=True)
    m3_parser.add_argument("--data-ext", default=None)
    m3_parser.add_argument("--primary-yx", nargs=2, type=float, default=None, metavar=("Y", "X"))
    m3_parser.add_argument("--aperture-radius", type=float, default=None)
    m3_parser.add_argument("--aperture-correction", choices=["none", "growth_curve"], default=None)
    m3_parser.add_argument("--truncation-correction", action="store_true")
    m3_parser.set_defaults(func=m3_flux_phase)

    sky_parser = subparsers.add_parser("m1m2-sky", help="Measure A4/M1 (wavelength) + M2 (LSF) from MUSE SKY_SPECTRUM airglow lines and patch the QC.")
    sky_parser.add_argument("--sky-spectrum", nargs="+", required=True, help="One or more SKY_SPECTRUM_*.fits (per exposure).")
    sky_parser.add_argument("--qc-output", required=True)
    sky_parser.add_argument("--skylines", default="musepipe/qc/data/skylines.csv")
    sky_parser.add_argument("--frame", default="topocentric", choices=["topocentric", "barycentric"])
    sky_parser.add_argument("--vbary-kms", type=float, default=0.0)
    sky_parser.add_argument("--halpha-A", type=float, default=6562.8)
    sky_parser.set_defaults(func=m1m2_sky_phase)

    m4m5_parser = subparsers.add_parser(
        "m4m5", help="Measure A4/M4 sky residuals and A4/M5 STAT factors."
    )
    m4m5_parser.add_argument("--cube", required=True)
    m4m5_parser.add_argument("--source-mask", required=True)
    m4m5_parser.add_argument("--qc-output", required=True)
    m4m5_parser.add_argument("--products-dir", required=True)
    m4m5_parser.add_argument("--aperture-radius", type=int, default=2)
    m4m5_parser.add_argument("--spacing-px", type=int, default=10)
    m4m5_parser.add_argument("--max-apertures", type=int, default=32)
    m4m5_parser.add_argument("--primary-yx", nargs=2, type=float, default=None, metavar=("Y", "X"),
                             help="Primary position for the M4 radial diagnostic, in THIS cube's frame.")
    m4m5_parser.add_argument("--radial-bin-px", type=float, default=2.0)
    m4m5_parser.set_defaults(func=m4m5_phase)

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
    "M1_CLEAN_AIRGLOW",
    "MUSE_LSF_POLY_BACON2017",
    "MUSE_LSF_REFERENCE",
    "MUSE_LSF_REFERENCE_SHORT",
    "aggregate_status",
    "compute_m3_flux",
    "derive_downstream_decision",
    "derive_open_issues",
    "detect_primary_yx",
    "empty_aperture_centers",
    "finalize_qc",
    "resolve_qc_wavelength_frame",
    "patch_status",
    "flux_factor_from_reference",
    "load_passband_csv",
    "measure_m1_m2_from_sky_spectrum",
    "read_sky_spectrum_fits",
    "measure_line_moments",
    "measure_lsf",
    "measure_sky_radial_profile",
    "resolve_primary_yx",
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
