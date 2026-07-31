"""A3 telluric-correction decision and application helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Mapping, Sequence

import numpy as np
from astropy.io import fits

from ..config import load_run_config
from ..stats import robust_sigma
from .esorex_driver import parse_esorex_recipes
from .verify import circular_aperture_mask, extract_aperture_spectrum


STAGE_NAME = "00t_telluric"

HALPHA_PROTECTED = (6540.0, 6590.0)
NALGS_PROTECTED = (5780.0, 6050.0)
PROTECTED_WINDOWS = (HALPHA_PROTECTED, NALGS_PROTECTED)
TELLURIC_BANDS = {
    "O2_B": (6864.0, 6960.0),
    "H2O_7200": (7160.0, 7340.0),
    "H2O_8200": (8130.0, 8350.0),
}
DEFAULT_FIT_REGIONS = tuple(TELLURIC_BANDS.values())


def stage00t_config_from_run(
    run_id: str | None = None,
    *,
    project_root: str | Path | None = None,
    overrides: Mapping[str, object] | None = None,
    allow_run_id_mismatch: bool = False,
) -> dict:
    """Resolve A3's knobs for a run, the way `stage_xNN_config_from_run` does.

    A3 predates the `musepipe.stages` convention: it has no stage module, and
    its knobs live half in the argparse defaults of `main` and half as literals
    inside the numeric functions (the 3 % threshold, the continuum sidebands,
    the band edges). Anything that wants to *reproduce* A3 — the debug notebook
    above all — would otherwise have to copy those literals by hand, which is
    exactly what once made the C3 notebook fail to reproduce the chain.

    Two deliberate choices:

    - ``a3_science_needs_red_continuum`` defaults to ``True``, unlike the
      ``store_true`` flag in `main`. All three A3 QCs on disk record ``true``,
      and `decide_telluric` short-circuits to ``not_needed_science`` when it is
      false, contradicting every verdict on record: the flag was passed. This
      resolver reproduces what ran. The argparse default is left alone — moving
      it would move frozen numbers.
    - Nothing calls this from `decision_phase`/`main`. It is read-only; wiring
      it in would be a behaviour change on a stage with three frozen QCs.

    ``a3_primary_yx`` has no default: A3 requires it on the command line, and
    the multi-night QC does not record it (a schema gap the notebook surfaces).
    """

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
    cfg.setdefault("a3_primary_yx", None)
    cfg.setdefault("a3_radius_px", 8.0)
    cfg.setdefault("a3_science_needs_red_continuum", True)
    cfg.setdefault("a3_threshold_pct", 3.0)
    cfg.setdefault("a3_bands_A", {name: list(band) for name, band in TELLURIC_BANDS.items()})
    cfg.setdefault("a3_protected_windows_A", [list(win) for win in PROTECTED_WINDOWS])
    cfg.setdefault("a3_side_width_A", 40.0)
    cfg.setdefault("a3_gap_A", 10.0)
    cfg.setdefault("a3_min_transmission", 0.05)
    return cfg


class TelluricError(RuntimeError):
    """Raised when A3 must stop at a gate or checkpoint."""


@dataclass(frozen=True)
class TelluricInputInfo:
    cube: Path
    upstream: str
    sha256: str
    has_data: bool
    has_stat: bool


@dataclass(frozen=True)
class TelluricDecision:
    depth_pct_by_band: dict[str, float]
    telluric_applied: bool
    science_needs_red_continuum: bool
    decision: str
    checkpoint_required: bool


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, text=True, capture_output=True)


def check_molecfit_environment(
    *,
    esorex: str = "esorex",
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run_command,
) -> dict[str, object]:
    """Verify that molecfit EsoRex recipes are available."""

    executable = shutil.which(esorex) if "/" not in esorex else esorex
    if executable is None:
        raise TelluricError("EsoRex executable not found; cannot access molecfit recipes.")
    result = runner([executable, "--recipes"])
    if result.returncode != 0:
        raise TelluricError(f"Could not list EsoRex recipes: {result.stderr.strip()}")
    recipes = set(parse_esorex_recipes(result.stdout))
    required = {"molecfit_model", "molecfit_calctrans", "molecfit_correct"}
    missing = sorted(required - recipes)
    if missing:
        raise TelluricError(f"Missing molecfit recipes: {', '.join(missing)}")
    version = "recipes_present_version_not_reported"
    for line in result.stdout.splitlines():
        if "molecfit" in line.lower() and "version" in line.lower():
            version = line.strip()
            break
    return {
        "molecfit_version": version,
        "molecfit_recipes": sorted(required),
        "esorex_path": str(executable),
        "gdas_profile": "unknown",
    }


def resolve_input_cube(
    cube_path: str | Path,
    *,
    upstream: str,
    qc_path: str | Path | None = None,
    checksum: bool = True,
) -> TelluricInputInfo:
    cube = Path(cube_path).expanduser()
    if not cube.exists():
        raise TelluricError(f"Input cube does not exist: {cube}")
    upstream_norm = str(upstream).upper()
    if upstream_norm not in {"A2", "A1", "ADP"}:
        raise TelluricError("upstream must be A2, A1, or ADP.")

    with fits.open(cube, memmap=True) as hdul:
        has_data = "DATA" in hdul or any(getattr(hdu.data, "ndim", 0) == 3 for hdu in hdul)
        has_stat = "STAT" in hdul
    if not has_data or not has_stat:
        raise TelluricError(f"Input cube must contain DATA and STAT; got DATA={has_data}, STAT={has_stat}.")

    if upstream_norm in {"A1", "A2"}:
        if qc_path is None:
            raise TelluricError(f"QC path is required for upstream={upstream_norm}.")
        with Path(qc_path).open("r", encoding="utf-8") as handle:
            qc = json.load(handle)
        if qc.get("open_issues"):
            raise TelluricError(f"{upstream_norm} QC has open issues; stop before A3.")
        if upstream_norm == "A1" and not set(("fase0", "fase1", "fase2", "fase3")).issubset(
            set(qc.get("gates_passed", []))
        ):
            raise TelluricError("A1 gates are incomplete.")
        if upstream_norm == "A2" and "decision" not in qc:
            raise TelluricError("A2 QC does not contain a decision block.")

    return TelluricInputInfo(
        cube=cube,
        upstream=upstream_norm,
        sha256=sha256_file(cube) if checksum else "",
        has_data=has_data,
        has_stat=has_stat,
    )


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
    raise TelluricError("Could not recover wavelength axis from DATA header.")


def window_mask(wave: Sequence[float], window: tuple[float, float]) -> np.ndarray:
    wave_arr = np.asarray(wave, dtype=np.float64)
    lo, hi = window
    return (wave_arr >= lo) & (wave_arr <= hi) & np.isfinite(wave_arr)


def protected_mask(wave: Sequence[float], windows: Sequence[tuple[float, float]] = PROTECTED_WINDOWS) -> np.ndarray:
    mask = np.zeros(np.asarray(wave).shape, dtype=bool)
    for window in windows:
        mask |= window_mask(wave, window)
    return mask


def local_continuum_linear(
    wave: Sequence[float],
    spectrum: Sequence[float],
    band: tuple[float, float],
    *,
    side_width_A: float = 40.0,
    gap_A: float = 10.0,
) -> np.ndarray:
    """Interpolate a local continuum across one telluric band."""

    wave_arr = np.asarray(wave, dtype=np.float64)
    spec = np.asarray(spectrum, dtype=np.float64)
    lo, hi = band
    left = (wave_arr >= lo - gap_A - side_width_A) & (wave_arr <= lo - gap_A)
    right = (wave_arr >= hi + gap_A) & (wave_arr <= hi + gap_A + side_width_A)
    left &= np.isfinite(spec)
    right &= np.isfinite(spec)
    if not left.any() or not right.any():
        finite = np.isfinite(spec)
        fallback = float(np.nanmedian(spec[finite])) if finite.any() else 1.0
        return np.full(wave_arr.shape, fallback, dtype=np.float64)
    x = np.array([np.nanmedian(wave_arr[left]), np.nanmedian(wave_arr[right])], dtype=np.float64)
    y = np.array([np.nanmedian(spec[left]), np.nanmedian(spec[right])], dtype=np.float64)
    if not np.all(np.isfinite(y)) or x[0] == x[1]:
        return np.full(wave_arr.shape, float(np.nanmedian(spec[np.isfinite(spec)])), dtype=np.float64)
    return np.interp(wave_arr, x, y)


def measure_telluric_depths(
    wave: Sequence[float],
    spectrum: Sequence[float],
    *,
    bands: Mapping[str, tuple[float, float]] = TELLURIC_BANDS,
) -> dict[str, float]:
    """Measure telluric depth as percent drop relative to local continuum."""

    wave_arr = np.asarray(wave, dtype=np.float64)
    spec = np.asarray(spectrum, dtype=np.float64)
    depths: dict[str, float] = {}
    for name, band in bands.items():
        mask = window_mask(wave_arr, band)
        if not mask.any():
            depths[name] = float("nan")
            continue
        continuum = local_continuum_linear(wave_arr, spec, band)
        valid = mask & np.isfinite(spec) & np.isfinite(continuum) & (continuum != 0)
        if not valid.any():
            depths[name] = float("nan")
            continue
        ratio = spec[valid] / continuum[valid]
        depth = 100.0 * (1.0 - float(np.nanmedian(ratio)))
        depths[name] = max(0.0, depth)
    return depths


def decide_telluric(
    depth_pct_by_band: Mapping[str, float],
    *,
    science_needs_red_continuum: bool,
    threshold_pct: float = 3.0,
) -> TelluricDecision:
    depths = {str(key): float(value) for key, value in depth_pct_by_band.items()}
    finite_depths = [value for value in depths.values() if np.isfinite(value)]
    max_depth = max(finite_depths) if finite_depths else float("nan")
    if not science_needs_red_continuum:
        decision = "not_needed_science"
        applied = False
        checkpoint = False
    elif np.isfinite(max_depth) and max_depth < threshold_pct:
        decision = "not_needed_shallow"
        applied = False
        checkpoint = False
    elif np.isfinite(max_depth):
        decision = "needed"
        applied = True
        checkpoint = True
    else:
        decision = "depth_unknown_checkpoint"
        applied = False
        checkpoint = True
    return TelluricDecision(
        depth_pct_by_band=depths,
        telluric_applied=applied,
        science_needs_red_continuum=bool(science_needs_red_continuum),
        decision=decision,
        checkpoint_required=checkpoint,
    )


def enforce_protected_transmission(
    wave: Sequence[float],
    transmission: Sequence[float],
    *,
    windows: Sequence[tuple[float, float]] = PROTECTED_WINDOWS,
) -> np.ndarray:
    trans = np.asarray(transmission, dtype=np.float64).copy()
    if trans.ndim != 1:
        raise TelluricError("Transmission must be a 1D array.")
    mask = protected_mask(wave, windows)
    if mask.shape != trans.shape:
        raise TelluricError("Transmission and wavelength arrays must have the same shape.")
    trans[mask] = 1.0
    return trans


def validate_transmission_physical(transmission: Sequence[float]) -> bool:
    trans = np.asarray(transmission, dtype=np.float64)
    return bool(np.all(np.isfinite(trans)) and np.nanmin(trans) > 0.0 and np.nanmax(trans) <= 1.0)


def apply_transmission_to_arrays(
    data: np.ndarray,
    stat: np.ndarray,
    wave: Sequence[float],
    transmission: Sequence[float],
    *,
    min_transmission: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply DATA/T and STAT/T^2, forcing protected windows to T=1."""

    cube = np.asarray(data, dtype=np.float64)
    variance = np.asarray(stat, dtype=np.float64)
    if cube.shape != variance.shape:
        raise TelluricError(f"DATA shape {cube.shape} != STAT shape {variance.shape}.")
    trans = enforce_protected_transmission(wave, transmission)
    if trans.shape[0] != cube.shape[0]:
        raise TelluricError("Transmission length does not match cube wavelength axis.")
    if np.nanmin(trans) < min_transmission:
        raise TelluricError(f"Transmission below safety floor {min_transmission}.")
    scale = trans[:, None, None]
    return cube / scale, variance / (scale**2), trans


def write_transmission_fits(wave: Sequence[float], transmission: Sequence[float], path: str | Path) -> Path:
    cols = [
        fits.Column(name="wave_A", format="D", array=np.asarray(wave, dtype=np.float64)),
        fits.Column(name="transmission", format="D", array=np.asarray(transmission, dtype=np.float64)),
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fits.BinTableHDU.from_columns(cols, name="TELLURIC_TRANS").writeto(output, overwrite=True)
    return output


def apply_transmission_to_cube_file(
    input_cube: str | Path,
    output_cube: str | Path,
    transmission: Sequence[float],
    *,
    transmission_output: str | Path | None = None,
    history: str = "A3 telluric correction applied",
) -> Path:
    """Apply a 1D transmission to a FITS cube and write a new corrected cube."""

    with fits.open(input_cube, memmap=True) as hdul:
        data_hdu = hdul["DATA"] if "DATA" in hdul else next(
            hdu for hdu in hdul if getattr(hdu.data, "ndim", 0) == 3
        )
        if "STAT" not in hdul:
            raise TelluricError("Input cube has no STAT extension.")
        data = np.asarray(data_hdu.data, dtype=np.float64)
        stat = np.asarray(hdul["STAT"].data, dtype=np.float64)
        wave = wavelength_axis_from_header(data_hdu.header, data.shape[0])
        data_corr, stat_corr, trans = apply_transmission_to_arrays(data, stat, wave, transmission)
        out_hdus = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header.copy())])
        data_header = data_hdu.header.copy()
        stat_header = hdul["STAT"].header.copy()
        out_hdus.append(fits.ImageHDU(data=data_corr.astype(np.float32), header=data_header, name="DATA"))
        out_hdus.append(fits.ImageHDU(data=stat_corr.astype(np.float32), header=stat_header, name="STAT"))
        out_hdus[0].header.add_history(history)

    output = Path(output_cube)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_hdus.writeto(output, overwrite=True)
    if transmission_output is not None:
        write_transmission_fits(wave, trans, transmission_output)
    return output


def verify_outside_bands_unchanged(
    pre_spec: Sequence[float],
    post_spec: Sequence[float],
    wave: Sequence[float],
    *,
    max_change_pct: float = 0.2,
    corrected_bands: Sequence[tuple[float, float]] = DEFAULT_FIT_REGIONS,
) -> bool:
    pre = np.asarray(pre_spec, dtype=np.float64)
    post = np.asarray(post_spec, dtype=np.float64)
    wave_arr = np.asarray(wave, dtype=np.float64)
    mask = np.isfinite(pre) & np.isfinite(post) & (pre != 0)
    for band in corrected_bands:
        mask &= ~window_mask(wave_arr, band)
    mask &= ~protected_mask(wave_arr)
    if not mask.any():
        raise TelluricError("No outside-band channels available for verification.")
    change_pct = 100.0 * np.nanmedian(np.abs(post[mask] / pre[mask] - 1.0))
    return bool(change_pct <= max_change_pct)


def verify_halpha_untouched(
    pre_cube: np.ndarray,
    post_cube: np.ndarray,
    wave: Sequence[float],
    companion_yx: tuple[float, float],
    *,
    aperture_radius_px: float,
) -> bool:
    mask = window_mask(wave, HALPHA_PROTECTED)
    if not mask.any():
        raise TelluricError("No Halpha channels in wavelength axis.")
    pre_spec = extract_aperture_spectrum(np.asarray(pre_cube), companion_yx, aperture_radius_px)
    post_spec = extract_aperture_spectrum(np.asarray(post_cube), companion_yx, aperture_radius_px)
    return bool(np.allclose(pre_spec[mask], post_spec[mask], rtol=0.0, atol=1e-6))


def verify_stat_scaled(
    stat_pre: np.ndarray,
    stat_post: np.ndarray,
    wave: Sequence[float],
    transmission: Sequence[float],
    *,
    rtol: float = 1e-6,
) -> bool:
    _, expected, trans = apply_transmission_to_arrays(
        np.ones_like(stat_pre, dtype=np.float64),
        np.asarray(stat_pre, dtype=np.float64),
        wave,
        transmission,
    )
    return bool(np.allclose(stat_post, expected, rtol=rtol, atol=0.0) and validate_transmission_physical(trans))


def stage00t_qc_skeleton(
    input_info: TelluricInputInfo,
    *,
    run_id: str,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Esqueleto del QC de A3. `run_id` obligatorio (ver `stage00s_qc_skeleton`)."""
    return {
        "stage": STAGE_NAME,
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "environment": dict(environment or {}),
        "input": {"cube": str(input_info.cube), "sha256": input_info.sha256, "upstream": input_info.upstream},
        "decision": {
            "depth_pct_by_band": {},
            "telluric_applied": False,
            "science_needs_red_continuum": True,
            "user_checkpoint": "",
        },
        "fit": {
            "molecules": ["O2", "H2O"],
            "regions_A": [list(region) for region in DEFAULT_FIT_REGIONS],
            "excluded_subregions_A": [],
            "chi2_by_region": {},
            "kernel": 0.0,
        },
        "protected_windows_A": [list(window) for window in PROTECTED_WINDOWS],
        "products": {"cube_telcorr": "", "transmission": ""},
        "verification": {
            "v1_residual_pct_by_band": {},
            "v2_outside_bands_change_pct": 0.0,
            "v3_halpha_untouched": True,
            "v4_transmission_physical": True,
            "v5_stat_scaled": True,
        },
        "open_issues": [],
    }


def _read_primary_spectrum(cube_path: Path, yx: tuple[float, float], radius_px: float):
    with fits.open(cube_path, memmap=True) as hdul:
        data_hdu = hdul["DATA"] if "DATA" in hdul else next(
            hdu for hdu in hdul if getattr(hdu.data, "ndim", 0) == 3
        )
        cube = np.asarray(data_hdu.data, dtype=np.float64)
        wave = wavelength_axis_from_header(data_hdu.header, cube.shape[0])
        spec = extract_aperture_spectrum(cube, yx, radius_px)
    return wave, spec


def decision_phase(args: argparse.Namespace) -> int:
    input_info = resolve_input_cube(
        args.input_cube,
        upstream=args.upstream,
        qc_path=args.qc,
        checksum=not args.skip_checksum,
    )
    wave, spec = _read_primary_spectrum(input_info.cube, (args.primary_y, args.primary_x), args.radius_px)
    depths = measure_telluric_depths(wave, spec)
    decision = decide_telluric(depths, science_needs_red_continuum=args.science_needs_red_continuum)
    qc = stage00t_qc_skeleton(input_info, run_id=args.run_id)
    qc["decision"].update(
        {
            "depth_pct_by_band": decision.depth_pct_by_band,
            "telluric_applied": decision.telluric_applied,
            "science_needs_red_continuum": decision.science_needs_red_continuum,
            "verdict": decision.decision,
            "checkpoint_required": decision.checkpoint_required,
            "user_checkpoint": "required" if decision.checkpoint_required else "not_needed",
        }
    )
    output = Path(args.qc_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qc["decision"], indent=2))
    return 3 if decision.checkpoint_required else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A3 telluric decision and application helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("check-env", help="Verify molecfit recipes.")
    env_parser.add_argument("--esorex", default="esorex")

    def _env(args):
        print(json.dumps(check_molecfit_environment(esorex=args.esorex), indent=2))
        return 0

    env_parser.set_defaults(func=_env)

    decision_parser = subparsers.add_parser("decision", help="Measure whether telluric correction is needed.")
    decision_parser.add_argument("--run-id", required=True,
                                 help="Run al que pertenece este QC (se escribe en stage00t_qc.json).")
    decision_parser.add_argument("--input-cube", required=True)
    decision_parser.add_argument("--upstream", choices=["A2", "A1", "ADP"], required=True)
    decision_parser.add_argument("--qc")
    decision_parser.add_argument("--primary-y", type=float, required=True)
    decision_parser.add_argument("--primary-x", type=float, required=True)
    decision_parser.add_argument("--radius-px", type=float, default=8.0)
    decision_parser.add_argument("--qc-output", required=True)
    decision_parser.add_argument("--science-needs-red-continuum", action="store_true")
    decision_parser.add_argument("--skip-checksum", action="store_true")
    decision_parser.set_defaults(func=decision_phase)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except TelluricError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
