"""A2 sky-residual decision and ZAP execution helpers.

The scientific guardrail for this stage is simple: decide from metrics whether
ZAP is needed, build a conservative source mask, and never let a successful
software call stand in for verification that source flux was preserved.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
from astropy.io import fits

from ..stats import robust_sigma
from .verify import circular_aperture_mask, whitelight_image


STAGE_NAME = "00s_sky_zap"

NALGS_RANGE = (5780.0, 6050.0)
SKYLINE_WINDOWS = (
    (5572.0, 5582.0),
    (6295.0, 6306.0),
    (6358.0, 6370.0),
    (6864.0, 6960.0),
    (7240.0, 9300.0),
)
CONTINUUM_WINDOWS = (
    (5100.0, 5500.0),
    (6600.0, 6800.0),
)


class SkyZapError(RuntimeError):
    """Raised when A2 must stop at a gate or checkpoint."""


@dataclass(frozen=True)
class InputCubeInfo:
    """Resolved input cube metadata for A2."""

    cube: Path
    provenance: str
    sha256: str
    a1_gates_ok: bool
    a1_not_applicable: bool
    has_data: bool
    has_stat: bool
    #: Verificaciones de A1 cuyo veredicto se dedujo de un escalar (§`verification_verdict`).
    a1_scalar_verdicts: tuple[str, ...] = ()


def verification_verdict(entry):
    """Veredicto de una verificación de A1, en cualquiera de sus dos formas.

    El QC de A1 publica cada V de dos maneras: como **dict** `{ok, value,
    message}` —la que escribe `a1_verify`, con `ok` booleano explícito— o como
    **escalar** heredado, que es lo que hay en las reducciones antiguas.

    La puerta hacía `bool(verification.get(key))` sobre las dos, y eso no
    verifica nada en cuanto el valor es un número: `bool(0.00122)` es `True`, y
    un residuo de 0.00122 no significa que la prueba pasara. Con la forma dict
    pasa lo mismo por otra vía: cualquier dict no vacío es verdadero.

    Aquí el dict manda (`ok`), el booleano se respeta, y `None` es «sin medir»
    —que **no** satisface una verificación exigida—. El escalar se sigue
    aceptando para no romper las reducciones ya validadas, pero quien lo use
    queda anotado en `a1_scalar_verdicts` y viaja al QC de A2.
    """

    if isinstance(entry, dict):
        return bool(entry.get("ok")) if entry.get("ok") is not None else False
    if isinstance(entry, bool):
        return entry
    if entry is None:
        return False
    return bool(entry)


@dataclass(frozen=True)
class ZapDecision:
    """Decision metrics for whether ZAP is needed."""

    r_skyline_over_continuum: float
    skyline_rms_median: float
    continuum_rms_median: float
    sky_fraction: float
    decision: str
    zap_applied: bool
    checkpoint_required: bool


@dataclass(frozen=True)
class SourceRegion:
    """A forced source region for ZAP masking."""

    name: str
    yx: tuple[float, float]
    radius_px: float
    auto_halo: bool = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_zap_environment(module_name: str = "zap") -> dict[str, object]:
    """Verify that the import named ``zap`` is the musevlt package interface."""

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SkyZapError(
            "ZAP is not importable. Install the musevlt 'zap' package in this environment "
            "or use a documented separate environment for A2."
        ) from exc

    process = getattr(module, "process", None)
    if process is None or not callable(process):
        raise SkyZapError(
            f"Imported {module_name!r}, but it does not expose callable zap.process; "
            "this is likely the wrong package."
        )

    return {
        "zap_version": str(getattr(module, "__version__", "unknown")),
        "zap_path": str(getattr(module, "__file__", "")),
        "separate_env": False,
    }


def resolve_input_cube(
    cube_path: str | Path,
    *,
    provenance: str,
    a1_qc_path: str | Path | None = None,
    checksum: bool = True,
) -> InputCubeInfo:
    """Resolve and gate-check the A2 input cube.

    ``provenance='adp'`` intentionally bypasses A1 QC while recording that A1 is
    not applicable. ``provenance='raw_reduction'`` requires A1 gates and green
    verification before A2 can proceed.
    """

    cube = Path(cube_path).expanduser()
    if not cube.exists():
        raise SkyZapError(f"Input cube does not exist: {cube}")
    provenance = str(provenance)
    if provenance not in {"raw_reduction", "adp"}:
        raise SkyZapError("provenance must be 'raw_reduction' or 'adp'.")

    with fits.open(cube, memmap=True) as hdul:
        has_data = "DATA" in hdul or any(getattr(hdu.data, "ndim", 0) == 3 for hdu in hdul)
        has_stat = "STAT" in hdul
    if not has_data or not has_stat:
        raise SkyZapError(f"Input cube must contain DATA and STAT; got DATA={has_data}, STAT={has_stat}.")

    a1_gates_ok = False
    a1_not_applicable = provenance == "adp"
    scalar_verdicts: list[str] = []
    if provenance == "raw_reduction":
        if a1_qc_path is None:
            raise SkyZapError("A1 QC path is required for provenance='raw_reduction'.")
        with Path(a1_qc_path).open("r", encoding="utf-8") as handle:
            qc = json.load(handle)
        gates = set(qc.get("gates_passed", []))
        verification = qc.get("verification", {})
        open_issues = qc.get("open_issues", [])
        required_gates = {"fase0", "fase1", "fase2", "fase3"}
        required_verifications = (
            "v1_stat_present",
            "v3_wcs_ok",
            "v4_adp_whitelight_corr",
            "v5_adp_star_spec_ratio_rms",
            "v6_sky_mask_clean",
        )
        verdicts = {key: verification_verdict(verification.get(key)) for key in required_verifications}
        scalar_verdicts = sorted(
            key for key in required_verifications
            if not isinstance(verification.get(key), (bool, dict, type(None)))
        )
        a1_gates_ok = required_gates.issubset(gates) and all(verdicts.values())
        if not a1_gates_ok or open_issues:
            failed = sorted(k for k, v in verdicts.items() if not v)
            detail = f" (V sin veredicto favorable: {', '.join(failed)})" if failed else ""
            raise SkyZapError(
                "A1 QC is not green for sky-sensitive gates; fix/report A1 before A2." + detail
            )

    return InputCubeInfo(
        cube=cube,
        provenance=provenance,
        sha256=sha256_file(cube) if checksum else "",
        a1_gates_ok=a1_gates_ok,
        a1_not_applicable=a1_not_applicable,
        has_data=has_data,
        has_stat=has_stat,
        a1_scalar_verdicts=tuple(scalar_verdicts),
    )


def read_cube_hdu(path: str | Path, *, name: str = "DATA"):
    """Return an open HDUList and the selected 3D cube HDU.

    The caller owns the returned HDUList and must close it.
    """

    hdul = fits.open(path, memmap=True)
    if name in hdul:
        return hdul, hdul[name]
    for hdu in hdul:
        if getattr(hdu.data, "ndim", 0) == 3:
            return hdul, hdu
    hdul.close()
    raise SkyZapError(f"No 3D DATA cube found in {path}.")


def wavelength_axis_from_header(header: fits.Header, n_wave: int) -> np.ndarray:
    """Build a wavelength axis from a FITS spectral WCS header."""

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
    raise SkyZapError("Could not recover wavelength axis from cube DATA header.")


def wavelength_mask(
    wave: Sequence[float],
    windows: Sequence[tuple[float, float]],
    *,
    exclude_ranges: Sequence[tuple[float, float]] = (NALGS_RANGE,),
) -> np.ndarray:
    wave_arr = np.asarray(wave, dtype=np.float64)
    mask = np.zeros(wave_arr.shape, dtype=bool)
    for lo, hi in windows:
        mask |= (wave_arr >= lo) & (wave_arr <= hi)
    for lo, hi in exclude_ranges:
        mask &= ~((wave_arr >= lo) & (wave_arr <= hi))
    return mask & np.isfinite(wave_arr)


def corner_empty_positions(shape: tuple[int, int], *, margin_px: int = 8) -> list[tuple[int, int]]:
    ny, nx = shape
    margin = int(margin_px)
    if ny <= 2 * margin or nx <= 2 * margin:
        margin = max(1, min(ny, nx) // 4)
    return [
        (margin, margin),
        (margin, nx - margin - 1),
        (ny - margin - 1, margin),
        (ny - margin - 1, nx - margin - 1),
    ]


def filter_positions_far_from_sources(
    positions: Iterable[tuple[float, float]],
    source_regions: Iterable[SourceRegion],
    *,
    min_gap_px: float = 2.0,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for pos in positions:
        py, px = map(float, pos)
        keep = True
        for region in source_regions:
            ry, rx = region.yx
            if np.hypot(py - ry, px - rx) <= float(region.radius_px) + float(min_gap_px):
                keep = False
                break
        if keep:
            out.append((py, px))
    return out


def aperture_pixel_mask(
    shape: tuple[int, int],
    positions_yx: Iterable[tuple[float, float]],
    *,
    radius_px: float,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for yx in positions_yx:
        mask |= circular_aperture_mask(shape, yx, radius_px)
    return mask


def channel_rms_from_mask(cube: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    """Robust RMS per wavelength channel over selected sky pixels."""

    data = np.asarray(cube)
    if data.ndim != 3:
        raise SkyZapError(f"Expected cube shape (wave,y,x), got {data.shape}.")
    if sky_mask.shape != data.shape[1:]:
        raise SkyZapError(f"Sky mask shape {sky_mask.shape} != cube spatial shape {data.shape[1:]}.")
    if not np.any(sky_mask):
        raise SkyZapError("Sky mask selects no pixels.")
    values = np.asarray(data[:, sky_mask], dtype=np.float64)
    med = np.nanmedian(values, axis=1)
    mad = np.nanmedian(np.abs(values - med[:, None]), axis=1)
    rms = 1.4826 * mad
    fallback = np.nanstd(values, axis=1)
    bad = ~np.isfinite(rms) | (rms <= 0)
    rms[bad] = fallback[bad]
    return rms.astype(np.float64)


def compute_sky_residual_metrics(
    cube: np.ndarray,
    wave: Sequence[float],
    sky_mask: np.ndarray,
    *,
    skyline_windows: Sequence[tuple[float, float]] = SKYLINE_WINDOWS,
    continuum_windows: Sequence[tuple[float, float]] = CONTINUUM_WINDOWS,
) -> dict[str, object]:
    """Measure skyline/continuum RMS ratio in empty sky apertures."""

    rms = channel_rms_from_mask(cube, sky_mask)
    wave_arr = np.asarray(wave, dtype=np.float64)
    if rms.shape != wave_arr.shape:
        raise SkyZapError(f"RMS length {rms.shape} does not match wavelength axis {wave_arr.shape}.")
    sky_mask_wave = wavelength_mask(wave_arr, skyline_windows)
    cont_mask_wave = wavelength_mask(wave_arr, continuum_windows)
    if not sky_mask_wave.any() or not cont_mask_wave.any():
        raise SkyZapError("Skyline/continuum windows select no wavelength channels.")
    skyline = float(np.nanmedian(rms[sky_mask_wave]))
    continuum = float(np.nanmedian(rms[cont_mask_wave]))
    ratio = float(skyline / continuum) if continuum > 0 else float("nan")
    return {
        "channel_rms": rms,
        "skyline_rms_median": skyline,
        "continuum_rms_median": continuum,
        "R_skyline_over_continuum": ratio,
    }


def classify_zap_decision(
    r_skyline_over_continuum: float,
    sky_fraction: float,
    *,
    low_threshold: float = 1.5,
    high_threshold: float = 2.0,
    min_sky_fraction: float = 0.25,
) -> ZapDecision:
    """Convert A2 metrics into the pre-registered ZAP decision."""

    if not np.isfinite(r_skyline_over_continuum):
        raise SkyZapError("R_skyline_over_continuum is not finite.")
    checkpoint = False
    if sky_fraction < min_sky_fraction:
        decision = "insufficient_sky_checkpoint"
        checkpoint = True
        zap = False
    elif r_skyline_over_continuum <= low_threshold:
        decision = "not_needed"
        zap = False
    elif r_skyline_over_continuum > high_threshold:
        decision = "needed"
        zap = True
    else:
        decision = "gray_zone_checkpoint"
        checkpoint = True
        zap = False

    return ZapDecision(
        r_skyline_over_continuum=float(r_skyline_over_continuum),
        skyline_rms_median=float("nan"),
        continuum_rms_median=float("nan"),
        sky_fraction=float(sky_fraction),
        decision=decision,
        zap_applied=zap,
        checkpoint_required=checkpoint,
    )


def decide_zap_from_metrics(metrics: Mapping[str, object], sky_fraction: float) -> ZapDecision:
    decision = classify_zap_decision(
        float(metrics["R_skyline_over_continuum"]),
        float(sky_fraction),
    )
    return ZapDecision(
        r_skyline_over_continuum=decision.r_skyline_over_continuum,
        skyline_rms_median=float(metrics["skyline_rms_median"]),
        continuum_rms_median=float(metrics["continuum_rms_median"]),
        sky_fraction=decision.sky_fraction,
        decision=decision.decision,
        zap_applied=decision.zap_applied,
        checkpoint_required=decision.checkpoint_required,
    )


def binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Small 3x3 binary dilation without making scipy mandatory."""

    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(iterations)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = expanded
    return out


def estimate_halo_radius(
    image: np.ndarray,
    center_yx: tuple[float, float],
    *,
    min_radius_px: float,
    max_radius_px: float | None = None,
    margin_px: float = 5.0,
    threshold_sigma: float = 1.0,
) -> float:
    """Estimate source halo radius from a white-light radial profile."""

    arr = np.asarray(image, dtype=np.float64)
    ny, nx = arr.shape
    cy, cx = center_yx
    yy, xx = np.indices(arr.shape, dtype=np.float64)
    rr = np.hypot(yy - cy, xx - cx)
    if max_radius_px is None:
        max_radius_px = float(min(ny, nx)) / 2.0
    outer = rr > 0.65 * float(max_radius_px)
    background = float(np.nanmedian(arr[outer])) if np.any(outer) else float(np.nanmedian(arr))
    sigma = robust_sigma(arr[outer]) if np.any(outer) else robust_sigma(arr)
    threshold = background + float(threshold_sigma) * sigma
    max_bin = int(max(1, np.ceil(max_radius_px)))
    last = float(min_radius_px)
    for radius in range(1, max_bin + 1):
        shell = (rr >= radius - 1) & (rr < radius)
        if np.any(shell) and np.nanmedian(arr[shell]) > threshold:
            last = float(radius)
    return max(float(min_radius_px), last + float(margin_px))


def build_source_mask(
    image: np.ndarray,
    source_regions: Sequence[SourceRegion],
    *,
    threshold_sigma: float = 3.0,
    dilation_px: int = 2,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a conservative True=source mask for ZAP."""

    arr = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        raise SkyZapError("White-light image has no finite pixels.")
    background = float(np.nanmedian(arr[finite]))
    sigma = robust_sigma(arr[finite])
    threshold = background + float(threshold_sigma) * sigma
    mask = finite & (arr > threshold)
    if dilation_px > 0:
        mask = binary_dilate(mask, int(dilation_px))

    effective_radii: dict[str, float] = {}
    for region in source_regions:
        radius = float(region.radius_px)
        if region.auto_halo:
            radius = estimate_halo_radius(arr, region.yx, min_radius_px=radius)
        effective_radii[region.name] = radius
        mask |= circular_aperture_mask(arr.shape, region.yx, radius)

    mask &= finite
    return mask, effective_radii


def sky_fraction_from_source_mask(source_mask: np.ndarray, finite_mask: np.ndarray | None = None) -> float:
    if finite_mask is None:
        finite_mask = np.ones_like(source_mask, dtype=bool)
    valid = np.asarray(finite_mask, dtype=bool)
    if not valid.any():
        return float("nan")
    return float(np.count_nonzero(valid & ~np.asarray(source_mask, dtype=bool)) / np.count_nonzero(valid))


def write_mask_fits(mask: np.ndarray, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.asarray(mask, dtype=np.uint8)).writeto(output, overwrite=True)
    return output


def write_mask_preview(
    image: np.ndarray,
    source_mask: np.ndarray,
    path: str | Path,
    *,
    positions: Sequence[SourceRegion] = (),
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    finite = np.isfinite(image)
    vmin, vmax = np.nanpercentile(image[finite], [1, 99]) if finite.any() else (0, 1)
    ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.contour(source_mask.astype(float), levels=[0.5], colors="tab:red", linewidths=1.0)
    for region in positions:
        y, x = region.yx
        ax.plot(x, y, marker="+", color="tab:cyan", ms=9, mew=1.5)
        ax.text(x + 2, y + 2, region.name, color="tab:cyan", fontsize=8)
    ax.set_title("ZAP source mask")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def copy_stat_to_output(
    input_cube: str | Path,
    output_cube: str | Path,
    *,
    history: str,
) -> None:
    """Copy the input STAT extension into an existing ZAP output cube."""

    with fits.open(input_cube, memmap=True) as src, fits.open(output_cube, mode="update") as dst:
        if "STAT" not in src:
            raise SkyZapError("Input cube has no STAT extension to preserve.")
        if "STAT" in dst:
            dst["STAT"].data[...] = src["STAT"].data
        else:
            dst.append(fits.ImageHDU(data=src["STAT"].data, header=src["STAT"].header, name="STAT"))
        dst[0].header.add_history(history)
        dst.flush()


def run_zap_process(
    input_cube: str | Path,
    output_cube: str | Path,
    source_mask: str | Path,
    *,
    zap_module=None,
) -> None:
    """Run ``zap.process`` with the default A2 contract.

    The musevlt ZAP API has changed over time. This wrapper keeps the call in
    one place and intentionally has no parameter search; if this call fails on
    the target machine, A2 stops and the exact exception is reported.
    """

    if zap_module is None:
        zap_module = importlib.import_module("zap")
    process = getattr(zap_module, "process", None)
    if process is None or not callable(process):
        raise SkyZapError("ZAP module does not expose callable process.")
    try:
        process(str(input_cube), outcubefile=str(output_cube), maskfile=str(source_mask))
    except TypeError:
        process(str(input_cube), str(output_cube), str(source_mask))
    copy_stat_to_output(
        input_cube,
        output_cube,
        history=f"A2 ZAP correction with source mask {source_mask}",
    )


def stage00s_qc_skeleton(
    input_info: InputCubeInfo,
    *,
    run_id: str,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Esqueleto del QC de A2.

    `run_id` es obligatorio: antes se tomaba de una constante de módulo fijada a
    ROXs12b_raw, así que el QC de CUALQUIER run se etiquetaba con ese run_id.
    """
    return {
        "stage": STAGE_NAME,
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "environment": dict(environment or {}),
        "input": {
            "cube": str(input_info.cube),
            "sha256": input_info.sha256,
            "provenance": input_info.provenance,
            "a1_gates_ok": input_info.a1_gates_ok,
            "a1_not_applicable": input_info.a1_not_applicable,
            "a1_scalar_verdicts": list(input_info.a1_scalar_verdicts),
        },
        "decision": {
            "R_skyline_over_continuum": None,
            "threshold_used": "1.5/2.0",
            "sky_fraction": None,
            "zap_applied": False,
            "user_checkpoint": "",
        },
        "mask": {
            "file": "",
            "threshold_sigma": 3,
            "dilation_px": 2,
            "forced_regions": [],
        },
        "zap_params": {"defaults": True, "overrides": {}},
        "products": {"cube_zap": "", "diagnostics": []},
        "verification": {
            "v1_rms_reduction_skylines": None,
            "v2_source_continuum_change_pct": None,
            "v3_halpha_window_change_sigma": None,
            "v4_sky_residual_symmetry": None,
            "v5_eigen_vs_star_corr_max": None,
            "v6_stat_untouched": None,
        },
        "open_issues": [],
    }


def _source_regions_from_json(values: Sequence[str] | None) -> list[SourceRegion]:
    regions: list[SourceRegion] = []
    for raw in values or ():
        payload = json.loads(raw)
        regions.append(
            SourceRegion(
                name=str(payload["name"]),
                yx=(float(payload["y"]), float(payload["x"])),
                radius_px=float(payload["radius_px"]),
                auto_halo=bool(payload.get("auto_halo", False)),
            )
        )
    return regions


def _companion_yx_from_regions(source_regions):
    """Posicion del companero entre las regiones forzadas, si esta declarada.

    Solo se usa para enmascararlo en las medianas azimutales; es un refinamiento,
    no un requisito (la mediana por anillo ya lo absorbe casi entero).
    """

    for region in source_regions or ():
        if "compan" in str(region.name).lower() or str(region.name).lower() in {"b", "secondary"}:
            return (float(region.yx[0]), float(region.yx[1]))
    return None


def _pixel_scale_arcsec(header):
    for key in ("CD1_1", "CDELT1"):
        value = header.get(key)
        if value:
            return abs(float(value)) * 3600.0
    return None


def decision_phase(args: argparse.Namespace) -> int:
    input_info = resolve_input_cube(
        args.input_cube,
        provenance=args.provenance,
        a1_qc_path=args.a1_qc,
        checksum=not args.skip_checksum,
    )
    hdul, data_hdu = read_cube_hdu(input_info.cube)
    try:
        cube = data_hdu.data
        wave = wavelength_axis_from_header(data_hdu.header, cube.shape[0])
        image = whitelight_image(cube)
        source_regions = _source_regions_from_json(args.source)
        source_mask, radii = build_source_mask(
            image,
            source_regions,
            threshold_sigma=args.threshold_sigma,
            dilation_px=args.dilation_px,
        )
        sky_fraction = sky_fraction_from_source_mask(source_mask, np.isfinite(image))
        empty_positions = filter_positions_far_from_sources(
            corner_empty_positions(image.shape, margin_px=args.empty_margin_px),
            source_regions,
        )
        empty_mask = aperture_pixel_mask(image.shape, empty_positions, radius_px=args.empty_radius_px)
        empty_mask &= ~source_mask
        metrics = compute_sky_residual_metrics(cube, wave, empty_mask)
        decision = decide_zap_from_metrics(metrics, sky_fraction)
        # Curva de crecimiento de la primaria. A2 es la UNICA etapa que tiene
        # abierto el cubo sin recortar: B1 recorta y se lleva por delante la
        # mitad exterior del halo, que es justo donde se puede separar cielo de
        # halo. Va aqui por eso, no por comodidad.
        #
        # Es un producto APARTE y NO BLOQUEANTE: si falla, la decision ZAP sigue
        # su curso. Y A2 mide y publica; NO aplica nada -- quien decide usarlo es
        # C2 (`x01_flux_convention`), para que el cambio de convencion de flujo
        # no quede enterrado en una etapa de reduccion.
        growth = None
        growth_error = None
        if not getattr(args, "skip_growth_curve", False):
            try:
                from ..growth_curve import measure_growth_curve

                growth = measure_growth_curve(
                    cube,
                    wave,
                    n_bands=int(getattr(args, "growth_bands", 8)),
                    companion_yx=_companion_yx_from_regions(source_regions),
                    pixel_scale_arcsec=_pixel_scale_arcsec(data_hdu.header),
                )
            except Exception as exc:  # noqa: BLE001 - diagnostico, nunca bloqueante
                growth_error = f"{exc.__class__.__name__}: {exc}"
    finally:
        hdul.close()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "zap_source_mask.fits"
    preview_path = output_dir / "stage00s_zap_source_mask.png"
    write_mask_fits(source_mask, mask_path)
    write_mask_preview(image, source_mask, preview_path, positions=source_regions)

    qc = stage00s_qc_skeleton(input_info, run_id=args.run_id)
    qc["decision"].update(
        {
            "R_skyline_over_continuum": decision.r_skyline_over_continuum,
            "skyline_rms_median": decision.skyline_rms_median,
            "continuum_rms_median": decision.continuum_rms_median,
            "sky_fraction": decision.sky_fraction,
            "zap_applied": decision.zap_applied,
            "checkpoint_required": decision.checkpoint_required,
            "verdict": decision.decision,
            "user_checkpoint": "not_needed" if decision.decision == "not_needed" else "required",
        }
    )
    qc["mask"].update(
        {
            "file": str(mask_path),
            "threshold_sigma": args.threshold_sigma,
            "dilation_px": args.dilation_px,
            "forced_regions": [region.name for region in source_regions],
            "effective_radii_px": radii,
            "preview": str(preview_path),
        }
    )
    qc["verification"]["v1_rms_reduction_skylines"] = (
        None if decision.zap_applied else decision.r_skyline_over_continuum
    )
    if growth is not None:
        qc["growth_curve"] = growth
        # El cielo que A2 usa para ZAP y el suelo que sale de extrapolar el halo
        # NO son el mismo numero, y la diferencia importa: la mascara de cielo
        # todavia contiene halo. Se publican los dos, con el aviso, en vez de
        # elegir uno en silencio.
        floors = [band["sky_floor"] for band in growth.get("bands", [])]
        if floors:
            qc["growth_curve"]["sky_floor_vs_zap_mask"] = {
                "extrapolated_floor_min": float(min(floors)),
                "extrapolated_floor_max": float(max(floors)),
                "zap_sky_fraction": decision.sky_fraction,
                "note": (
                    "The ZAP sky mask still contains AO halo at these radii; the extrapolated "
                    "floor is the halo-free level. They are not interchangeable."
                ),
            }
    elif growth_error is not None:
        qc["growth_curve"] = {"status": "failed", "error": growth_error}
    qc_path = Path(args.qc_output)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qc["decision"], indent=2))
    if decision.checkpoint_required:
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A2 ZAP decision and execution helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    decision_parser = subparsers.add_parser("decision", help="Measure whether ZAP is needed.")
    decision_parser.add_argument("--run-id", required=True,
                                 help="Run al que pertenece este QC (se escribe en stage00s_qc.json).")
    decision_parser.add_argument("--input-cube", required=True)
    decision_parser.add_argument("--provenance", choices=["raw_reduction", "adp"], required=True)
    decision_parser.add_argument("--a1-qc")
    decision_parser.add_argument("--output-dir", required=True)
    decision_parser.add_argument("--qc-output", required=True)
    decision_parser.add_argument("--source", action="append", help='JSON source: {"name":"primary","y":0,"x":0,"radius_px":8}')
    decision_parser.add_argument("--threshold-sigma", type=float, default=3.0)
    decision_parser.add_argument("--dilation-px", type=int, default=2)
    decision_parser.add_argument("--empty-margin-px", type=int, default=8)
    decision_parser.add_argument("--empty-radius-px", type=float, default=2.0)
    decision_parser.add_argument("--skip-checksum", action="store_true")
    decision_parser.add_argument(
        "--skip-growth-curve", action="store_true",
        help="no medir la curva de crecimiento de la primaria (producto aparte, no bloqueante)")
    decision_parser.add_argument(
        "--growth-bands", type=int, default=8,
        help="numero de bandas espectrales de la curva de crecimiento (default 8)")
    decision_parser.set_defaults(func=decision_phase)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SkyZapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
