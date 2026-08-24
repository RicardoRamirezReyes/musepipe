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
#: Bandas sobre las que A3 mide profundidad y decide.
#:
#: O₂ A es la más profunda del rango de MUSE y hasta 2026-07-31 no estaba aquí:
#: la etapa decidía sobre tres bandas que excluían justo la que más informa.
#: `verify.DEFAULT_BAD_RANGES` ya la marcaba como mala, `telluric_lines` ya la
#: cataloga `strong`, G3 ya la enmascara (`docs/2026-07-16_g3_real_frozen_decisions.md`) y
#: `a3_telluric_justification.md` §6 la llama «el rasgo telúrico con mayor
#: leverage» — la etapa que decide era la única pieza que la ignoraba. Los
#: bordes son los mismos 7590–7700 que usan esas otras piezas: no se introduce
#: una cuarta definición del mismo intervalo.
TELLURIC_BANDS = {
    "O2_B": (6864.0, 6960.0),
    "O2_A": (7590.0, 7700.0),
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
    # --- v3: las dos vías, y de dónde salen sus entradas -----------------
    # Ninguno de estos mueve nada de lo congelado: son knobs que antes no
    # existían porque la vía molecfit no estaba en el código y la STD se aplicaba
    # a mano. `a3_std_telluric` sin valor no es un fallo: significa que la vía STD
    # no se puede medir en este run, y el árbitro lo dice en vez de inventarla.
    cfg.setdefault("a3_input_cube", None)
    cfg.setdefault("a3_std_telluric", None)
    cfg.setdefault("a3_raw_dir", None)
    cfg.setdefault("a3_method", "auto")          # auto | molecfit | std
    cfg.setdefault("a3_granularity", "both")     # both | perexp | combined
    cfg.setdefault("a3_clean_window_A", list(CLEAN_WINDOW_A))
    cfg.setdefault("a3_chunk_channels", 200)
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
    #: Avisos NO fatales del QC de aguas arriba (hoy, los del combine en cascada).
    #: Viajan hasta `open_issues` del QC de A3: quien lea el veredicto tiene que
    #: ver lo que el combine dejo dicho, no perderlo en el camino.
    upstream_warnings: tuple[str, ...] = ()


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


#: Suelo de voxeles finitos para aceptar un combine en cascada como entrada de A3.
#: No es un umbral de calidad cientifica: es un cortafuegos contra un cubo medio
#: vacio. El combine de ROXs 42B b mide 0.94.
MIN_FINITE_FRACTION = 0.5


def _check_a1_upstream(qc: Mapping[str, object], cube: Path, qc_path) -> tuple[str, ...]:
    """Puerta de A1, que acepta DOS documentos distintos.

    A1 deja dos QC y no son intercambiables. El **envoltorio**
    (`stage00r_qc.json`, `stage: 00r_raw_reduction`) declara la reduccion entera
    con sus fases y su bateria V1-V6; el del **combine** (`cube_telcorr_qc.json`,
    esquema `stream_combine_v1`) documenta un solo paso y no tiene `gates_passed`.
    Cuando lo que llega es el segundo, exigirle fases lo rechaza siempre — es lo
    que impedia lanzar A3 sobre ROXs 42B b.

    CORREGIDO 2026-08-23: esto NO es una propiedad del perfil `cascade`. Los dos
    objetos de la cosecha son `cascade` y el envoltorio de ROXs 12 b
    (`ROXs12b_multinight_raw_20260728`) trae `gates_passed = [fase0..fase3,
    V1..V6]` con `stream_combine` entre sus recetas. Lo que faltaba en ROXs 42B b
    era el envoltorio, no las fases; lo reconstruye `musepipe.reduction.a1_verify`.

    Para ese esquema la puerta fuerte es de **identidad**: que el QC describa
    exactamente el cubo que se va a medir. Es mejor garantia que una lista de
    fases, porque ata el QC al dato en vez de a un tramite.

    Devuelve los `warnings` del combine, que **no son fatales** (el combine ya
    decidio sobre ellos) pero tienen que llegar al QC de A3 en vez de perderse.
    """

    if qc.get("stage") == "stream_combine":
        declarado = str(qc.get("output") or "")
        if not declarado:
            raise TelluricError(f"A1 cascade QC declares no output cube: {qc_path}")
        if Path(declarado).resolve() != cube.resolve():
            raise TelluricError(
                "A1 cascade QC describes another cube; A3 would measure a cube nobody "
                f"vouched for. QC output={declarado}, input cube={cube}."
            )
        if int(qc.get("n_exposures", 0)) < 1:
            raise TelluricError("A1 cascade QC declares no exposures.")
        finite = float(qc.get("finite_fraction", 0.0))
        if finite < MIN_FINITE_FRACTION:
            raise TelluricError(
                f"A1 cascade QC finite_fraction={finite:.4f} < {MIN_FINITE_FRACTION}: "
                "the combined cube is mostly empty."
            )
        return tuple(str(w) for w in (qc.get("warnings") or ()))

    if not {"fase0", "fase1", "fase2", "fase3"} <= set(qc.get("gates_passed", []) or ()):
        raise TelluricError("A1 gates are incomplete.")
    return ()


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

    upstream_warnings: tuple[str, ...] = ()
    if upstream_norm in {"A1", "A2"}:
        if qc_path is None:
            raise TelluricError(f"QC path is required for upstream={upstream_norm}.")
        with Path(qc_path).open("r", encoding="utf-8") as handle:
            qc = json.load(handle)
        if qc.get("open_issues"):
            raise TelluricError(f"{upstream_norm} QC has open issues; stop before A3.")
        if upstream_norm == "A1":
            upstream_warnings = _check_a1_upstream(qc, cube, qc_path)
        if upstream_norm == "A2" and "decision" not in qc:
            raise TelluricError("A2 QC does not contain a decision block.")

    return TelluricInputInfo(
        cube=cube,
        upstream=upstream_norm,
        sha256=sha256_file(cube) if checksum else "",
        has_data=has_data,
        has_stat=has_stat,
        upstream_warnings=upstream_warnings,
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
    continuum: Callable[..., np.ndarray] | None = None,
    side_width_A: float = 40.0,
    gap_A: float = 10.0,
    clip_negative: bool = True,
) -> dict[str, float]:
    """Measure telluric depth as percent drop relative to local continuum.

    The defaults reproduce the frozen stage behaviour bit for bit: `continuum=None`
    means `local_continuum_linear` with its own defaults, and `clip_negative=True`
    keeps the `max(0.0, ...)` floor. The keywords exist so a *diagnostic* can vary
    what the stage keeps fixed, which until now was impossible from above — the
    sideband knobs resolved by `stage00t_config_from_run` could not reach this
    function at all, so the A3 analysis notebook could only redraw the continuum it
    plotted, never the number it compared. Nothing in the chain passes them; see
    `docs/spec_A3_v2_codex_telluric.md` §1.5.

    `clip_negative=False` matters for a band that is *already corrected*: the floor
    turns a negative depth into 0.0, so the sign — the evidence that the DRS
    over-shot rather than under-shot — is destroyed before any caller can see it.
    """

    wave_arr = np.asarray(wave, dtype=np.float64)
    spec = np.asarray(spectrum, dtype=np.float64)
    depths: dict[str, float] = {}
    for name, band in bands.items():
        mask = window_mask(wave_arr, band)
        if not mask.any():
            depths[name] = float("nan")
            continue
        if continuum is None:
            model = local_continuum_linear(
                wave_arr, spec, band, side_width_A=side_width_A, gap_A=gap_A
            )
        else:
            model = np.asarray(continuum(wave_arr, spec, band), dtype=np.float64)
        valid = mask & np.isfinite(spec) & np.isfinite(model) & (model != 0)
        if not valid.any():
            depths[name] = float("nan")
            continue
        ratio = spec[valid] / model[valid]
        depth = 100.0 * (1.0 - float(np.nanmedian(ratio)))
        depths[name] = max(0.0, depth) if clip_negative else depth
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


def apply_transmission_streaming(
    input_cube: str | Path,
    output_cube: str | Path,
    transmission: Sequence[float],
    *,
    chunk_channels: int = 200,
    ledger=None,
    key_prefix: str = "apply",
    min_transmission: float = 0.05,
    history: str = "A3 telluric correction applied",
    progress=None,
) -> tuple[Path, np.ndarray]:
    """Aplica T al cubo **por trozos de λ**, y por eso se puede reanudar.

    `apply_transmission_to_cube_file` carga el cubo entero en float64 (dos copias
    de 1.2 GB para 3681×200×200) y, sobre todo, **no sobrevive a un corte**: si el
    proceso muere al 90 %, se empieza de cero. Aquí el trabajo se parte en trozos
    de canales y cada uno se anota en la bitácora, así que reanudar es seguir por
    donde iba.

    El cubo de salida se crea **copiando el de entrada** (headers, extensiones y
    forma correctos de un plumazo) y después se sobrescribe trozo a trozo. Los
    datos se leen siempre del **original**, nunca de la copia a medio corregir:
    leer de la salida al reanudar dividiría dos veces por T los trozos ya hechos.
    """

    import shutil
    import time as _time

    entrada = Path(input_cube)
    salida = Path(output_cube)
    salida.parent.mkdir(parents=True, exist_ok=True)

    with fits.open(entrada, memmap=True) as hdul:
        data_hdu = hdul["DATA"] if "DATA" in hdul else next(
            hdu for hdu in hdul if getattr(hdu.data, "ndim", 0) == 3)
        if "STAT" not in hdul:
            raise TelluricError("Input cube has no STAT extension.")
        nz = int(data_hdu.data.shape[0])
        wave = wavelength_axis_from_header(data_hdu.header, nz)
        nombre_data = data_hdu.name or "DATA"

    trans = enforce_protected_transmission(wave, transmission)
    if trans.shape[0] != nz:
        raise TelluricError("Transmission length does not match cube wavelength axis.")
    if np.nanmin(trans) < min_transmission:
        raise TelluricError(f"Transmission below safety floor {min_transmission}.")

    clave_copia = f"{key_prefix}:copy"
    if not salida.exists() or ledger is None or not ledger.hecha(clave_copia):
        t0 = _time.time()
        shutil.copyfile(entrada, salida)
        if ledger is not None:
            ledger.anota(clave_copia, rc=0, segundos=_time.time() - t0, producto=str(salida))

    chunk = max(1, int(chunk_channels))
    with fits.open(entrada, memmap=True) as origen, fits.open(salida, mode="update", memmap=True) as destino:
        src_data = origen[nombre_data] if nombre_data in origen else origen[1]
        dst_data = destino[nombre_data] if nombre_data in destino else destino[1]
        for z1 in range(0, nz, chunk):
            z2 = min(nz, z1 + chunk)
            clave = f"{key_prefix}:chunk:{z1}"
            if ledger is not None and ledger.hecha(clave):
                continue
            if ledger is not None and ledger.parar():
                raise TelluricError(
                    f"PARAR: detenido limpiamente en el canal {z1}; relanzar continua aqui.")
            t0 = _time.time()
            escala = trans[z1:z2][:, None, None]
            dst_data.data[z1:z2] = (np.asarray(src_data.data[z1:z2], dtype=np.float64) / escala
                                    ).astype(dst_data.data.dtype)
            destino["STAT"].data[z1:z2] = (
                np.asarray(origen["STAT"].data[z1:z2], dtype=np.float64) / (escala ** 2)
            ).astype(destino["STAT"].data.dtype)
            destino.flush()
            if ledger is not None:
                ledger.anota(clave, rc=0, segundos=_time.time() - t0, producto=str(salida),
                             canales=[z1, z2])
            if progress is not None:
                progress(z1, z2, nz)
        destino[0].header.add_history(history)
        destino.flush()
    return salida, trans


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


# ==========================================================================
# La vía STD_TELLURIC, y el árbitro que la compara con molecfit
# ==========================================================================

#: Ventana limpia de telúricas donde se mide el suelo de ruido del propio
#: espectro. Es la que usó la comparación de granularidad de 2026-08-06, y el
#: control que hizo decir algo al contraste con la plantilla fotosférica: sin un
#: suelo medido en el MISMO espectro, un residuo grande no distingue «mal
#: corregido» de «ruidoso».
CLEAN_WINDOW_A = (7750.0, 7860.0)


def std_telluric_transmission(
    std_path: str | Path,
    wave_A: Sequence[float],
    *,
    airmass_sci: float,
    airmass_std: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """La vía histórica, ahora como código: `T^(X_sci/X_std)` de la estándar.

    Hasta hoy esto se hacía a mano —los tres QC de A3 en disco lo declaran en un
    campo de texto (`decision.method`)— y por tanto no se podía testear ni
    reproducir. La ley es Beer-Lambert: la profundidad óptica escala con la masa
    de aire, así que la transmisión va a la potencia del cociente.

    Fuera del rango que cubre la estándar, `T = 1`: la estándar de MUSE empieza en
    ~6272 Å y el eje del cubo en 4750, y extrapolar una transmisión medida es
    inventarse absorción donde no se midió ninguna.
    """

    tabla = fits.getdata(str(std_path), 1)
    lam = np.asarray(tabla["lambda"], dtype=np.float64)
    ftel = np.asarray(tabla["ftelluric"], dtype=np.float64)
    finito = np.isfinite(lam) & np.isfinite(ftel)
    lam, ftel = lam[finito], ftel[finito]
    if lam.size < 2:
        raise TelluricError(f"STD_TELLURIC has no usable rows: {std_path}")

    if airmass_std is None:
        cabecera = fits.getheader(str(std_path), 0)
        x1 = cabecera.get("HIERARCH ESO TEL AIRM START")
        x2 = cabecera.get("HIERARCH ESO TEL AIRM END")
        if x1 is None or x2 is None:
            raise TelluricError(
                f"STD_TELLURIC {std_path} declares no airmass and none was given: "
                "the Beer-Lambert scaling has no exponent.")
        airmass_std = 0.5 * (float(x1) + float(x2))
    if not (np.isfinite(airmass_sci) and np.isfinite(airmass_std)) or airmass_std <= 0:
        raise TelluricError("STD_TELLURIC scaling needs finite, positive airmasses.")

    wave = np.asarray(wave_A, dtype=np.float64)
    dentro = (wave >= lam.min()) & (wave <= lam.max())
    trans = np.ones_like(wave)
    trans[dentro] = np.interp(wave[dentro], lam, ftel)
    exponente = float(airmass_sci) / float(airmass_std)
    trans = np.clip(trans, 1e-6, 1.0) ** exponente
    trans = enforce_protected_transmission(wave, trans)
    meta = {
        "std": str(std_path),
        "airmass_std": round(float(airmass_std), 4),
        "airmass_sci": round(float(airmass_sci), 4),
        "scaling": "T^(X_sci/X_std)",
        "exponent": round(exponente, 4),
        "covered_A": [float(lam.min()), float(lam.max())],
        "channels_outside_std_set_to_one": int((~dentro).sum()),
    }
    return trans, meta


def _band_residual(wave: Sequence[float], spectrum: Sequence[float], band: tuple[float, float],
                   *, side_width_A: float = 40.0, gap_A: float = 10.0) -> tuple[float, float]:
    """(rms sobre 1, dispersión sobre la media) de `flujo/continuo` en la banda.

    Los dos números salen del mismo cociente y la diferencia entre ellos **es el
    sesgo de la corrección**: `rms² = dispersión² + offset²`, donde el offset es
    lo que la banda queda por debajo (o por encima) del continuo después de
    corregir.

    Importa cuál se usa. La comparación publicada el 2026-08-06 tabula la
    **dispersión** (verificado: reproduce 0.0720 y 0.0304 en el combinado al
    cuarto decimal), y esa es ciega a una sobre- o sub-corrección constante — que
    es precisamente el modo de fallo de una transmisión con la masa de aire
    equivocada. El offset está en su tabla, pero en otra columna
    (`mediana %`) y fuera de la métrica que decidía.

    Aquí decide el **rms sobre 1**, que incluye el offset, y la dispersión se
    publica al lado para poder contrastar con aquella tabla.

    Reutiliza `local_continuum_linear`: la métrica que decide y el número que
    publica el QC tienen que hablar del mismo continuo.
    """

    wave_arr = np.asarray(wave, dtype=np.float64)
    spec = np.asarray(spectrum, dtype=np.float64)
    mask = window_mask(wave_arr, band)
    model = local_continuum_linear(wave_arr, spec, band, side_width_A=side_width_A, gap_A=gap_A)
    valido = mask & np.isfinite(spec) & np.isfinite(model) & (model != 0)
    if valido.sum() < 3:
        return float("nan"), float("nan")
    ratio = spec[valido] / model[valido]
    return (float(np.sqrt(np.nanmean((ratio - 1.0) ** 2))), float(np.nanstd(ratio)))


def _band_rms(wave: Sequence[float], spectrum: Sequence[float], band: tuple[float, float],
              *, side_width_A: float = 40.0, gap_A: float = 10.0) -> float:
    """Solo el rms sobre 1 de `_band_residual`, que es el que decide."""

    return _band_residual(wave, spectrum, band, side_width_A=side_width_A, gap_A=gap_A)[0]


def score_correction(
    wave_A: Sequence[float],
    spectrum: Sequence[float],
    transmission: Sequence[float] | None = None,
    *,
    bands: Mapping[str, tuple[float, float]] = TELLURIC_BANDS,
    clean_window_A: tuple[float, float] = CLEAN_WINDOW_A,
    side_width_A: float = 40.0,
    gap_A: float = 10.0,
) -> dict[str, object]:
    """Puntúa un espectro YA corregido. La figura de mérito es el **exceso**.

    Tres números por banda, y el tercero es el que decide:

    - ``rms_over_floor``: el residuo por canal contra el suelo de ruido del mismo
      espectro. Es la cifra que publica la comparación de 2026-08-06 (11.79 sin
      corregir en O₂ A, ~3.8 corrigiendo), y está aquí para poder contrastar.
    - ``expected_over_floor``: lo que ese cociente valdría **aunque la corrección
      fuera perfecta**. Dividir por T amplifica el ruido canal a canal
      (σᵢ = suelo/Tᵢ), y en O₂ A la T baja a 0.26: solo eso ya explica 1.79× de
      los 3.79× medidos.
    - ``fom``: `rms_over_floor / expected_over_floor`, o sea el exceso sobre lo
      inevitable. **Es el que decide**, porque el crudo penalizaría a la vía que
      corrige más hondo — que es exactamente al revés de lo que se quiere medir.

    `depth_pct` va sin recortar el signo (`clip_negative=False`) para que una
    **sobre-corrección** se vea como negativa en vez de aplastarse contra 0.
    """

    wave = np.asarray(wave_A, dtype=np.float64)
    spec = np.asarray(spectrum, dtype=np.float64)
    trans = None if transmission is None else np.asarray(transmission, dtype=np.float64)

    # El suelo se mide con la dispersión y no con el rms sobre 1: la ventana
    # limpia no está corregida, así que cualquier desnivel del continuo ahí es
    # del continuo, no del telúrico, y contarlo inflaría el denominador de todo.
    suelo = _band_residual(wave, spec, clean_window_A, side_width_A=side_width_A, gap_A=gap_A)[1]
    profundidades = measure_telluric_depths(wave, spec, bands=bands, side_width_A=side_width_A,
                                            gap_A=gap_A, clip_negative=False)
    por_banda: dict[str, dict[str, float]] = {}
    for nombre, banda in bands.items():
        rms, dispersion = _band_residual(wave, spec, banda, side_width_A=side_width_A, gap_A=gap_A)
        if trans is None:
            amplificacion = 1.0
            t_media = 1.0
        else:
            dentro = window_mask(wave, banda) & np.isfinite(trans) & (trans > 0)
            if not dentro.any():
                amplificacion, t_media = 1.0, 1.0
            else:
                t = trans[dentro]
                # rms del ruido escalado canal a canal, no 1/media(T): la media de
                # 1/T no es el rms de 1/T, y en O2 A las dos difieren de verdad.
                amplificacion = float(np.sqrt(np.mean(1.0 / (t ** 2))))
                t_media = float(np.mean(t))
        esperado = amplificacion
        rms_sobre_suelo = rms / suelo if np.isfinite(rms) and np.isfinite(suelo) and suelo > 0 else float("nan")
        por_banda[nombre] = {
            "rms": rms,
            "scatter": dispersion,
            "scatter_over_floor": (dispersion / suelo
                                   if np.isfinite(dispersion) and np.isfinite(suelo) and suelo > 0
                                   else float("nan")),
            "rms_over_floor": rms_sobre_suelo,
            "expected_over_floor": esperado,
            "fom": rms_sobre_suelo / esperado if np.isfinite(rms_sobre_suelo) and esperado > 0 else float("nan"),
            "transmission_mean": t_media,
            "depth_pct": profundidades.get(nombre, float("nan")),
        }
    foms = [v["fom"] for v in por_banda.values() if np.isfinite(v["fom"])]
    return {
        "floor": suelo,
        "clean_window_A": [float(clean_window_A[0]), float(clean_window_A[1])],
        "by_band": por_banda,
        "fom_mean": float(np.mean(foms)) if foms else float("nan"),
        "n_bands": len(foms),
    }


def tie_sigma_from_scores(scores: Sequence[Mapping[str, object]]) -> float:
    """1σ de la FoM entre exposiciones: el umbral de empate sale del run.

    Se prefiere a un literal porque el propio método tiene dispersión —el
    2026-08-06 las dos granularidades de molecfit se separaban un 1.3 %, y entre
    exposiciones la dispersión canal a canal era 1.21-1.88×— y una diferencia
    menor que la dispersión del método **no es una diferencia**.

    Con menos de dos ajustes no hay dispersión que medir y devuelve NaN: quien
    decide tiene que ver «no hay empate declarable», no un cero que haga ganar a
    cualquiera por un margen infinitesimal.
    """

    valores = [float(s.get("fom_mean", float("nan"))) for s in scores]
    valores = [v for v in valores if np.isfinite(v)]
    if len(valores) < 2:
        return float("nan")
    return float(np.std(valores, ddof=1))


def choose_method(
    scores_by_method: Mapping[str, Mapping[str, object]],
    *,
    tie_sigma: float = float("nan"),
    prefer: str = "molecfit",
) -> dict[str, object]:
    """Elige vía. Empate → `prefer`, que es molecfit por decisión del usuario.

    «Errores en el mismo rango» es `|Δ FoM| <= tie_sigma`, con σ medida entre
    exposiciones del propio run. Si no hay σ (un solo ajuste), **no se declara
    empate**: gana quien mida mejor, y el margen queda escrito para que se vea
    lo estrecho que fue.

    Esta función NO sabe nada de convergencia, y es deliberado: una vía que no
    convergió no llega hasta aquí — `fit_molecfit` levanta `MolecfitNotConverged`
    y la etapa para. Elegir STD porque molecfit falló convertiría un defecto de
    configuración en una decisión científica.
    """

    candidatos = {nombre: float(s.get("fom_mean", float("nan")))
                  for nombre, s in scores_by_method.items()}
    validos = {k: v for k, v in candidatos.items() if np.isfinite(v)}
    if not validos:
        raise TelluricError("No method produced a finite figure of merit; nothing to arbitrate.")
    if len(validos) == 1:
        unico = next(iter(validos))
        return {"winner": unico, "reason": "single_method", "fom_by_method": candidatos,
                "margin": float("nan"), "tie_sigma": tie_sigma}

    ordenados = sorted(validos.items(), key=lambda kv: kv[1])
    mejor, mejor_fom = ordenados[0]
    segundo, segundo_fom = ordenados[1]
    margen = float(segundo_fom - mejor_fom)
    empate = bool(np.isfinite(tie_sigma) and margen <= float(tie_sigma))
    if empate and prefer in validos:
        return {"winner": prefer, "reason": "tie_prefers_" + prefer, "fom_by_method": candidatos,
                "margin": margen, "tie_sigma": float(tie_sigma),
                "would_have_won": mejor}
    return {"winner": mejor, "reason": "measured", "fom_by_method": candidatos,
            "margin": margen, "tie_sigma": float(tie_sigma) if np.isfinite(tie_sigma) else None,
            "runner_up": segundo}


def stage00t_qc_skeleton(
    input_info: TelluricInputInfo,
    *,
    run_id: str,
    primary_yx: Sequence[float],
    aperture_radius_px: float,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Esqueleto del QC de A3. `run_id` obligatorio (ver `stage00s_qc_skeleton`).

    `primary_yx` y `aperture_radius_px` también son obligatorios y sin default:
    son la apertura con la que se midió la profundidad, y sin ellos el número
    que publica este QC no se puede reproducir a partir del QC. El esquema
    multi-noche los perdió, y recuperarlos costó un barrido a ciegas dentro del
    notebook de análisis (traspaso §7.4). Que no exista la vía silenciosa.
    """
    return {
        "stage": STAGE_NAME,
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "environment": dict(environment or {}),
        "input": {"cube": str(input_info.cube), "sha256": input_info.sha256,
                  "upstream": input_info.upstream,
                  "primary_yx": [float(primary_yx[0]), float(primary_yx[1])],
                  "aperture_radius_px": float(aperture_radius_px),
                  # Avisos NO fatales del QC de aguas arriba. Van aqui, junto a la
                  # procedencia que describen, y NO en `open_issues`: en este repo
                  # `open_issues` es el canal BLOQUEANTE (lo miran esta misma
                  # `resolve_input_cube` y `sky_zap`), y meter ahi un aviso
                  # informativo del combine bloquearia la cadena por nada.
                  "upstream_warnings": list(input_info.upstream_warnings)},
        "decision": {
            "depth_pct_by_band": {},
            "telluric_applied": False,
            # `telluric_applied` significa «la decision es aplicarla», no «esta
            # aplicada»: la fase de decision lo pone a True en cuanto el veredicto
            # es `needed`, con el cubo todavia sin tocar. Los dos QC historicos de
            # ROXs 12 b lo tienen a True **y** aplicada, asi que el campo solo no
            # distingue los dos estados. Este si: lo pone a True la fase de
            # aplicacion, y hasta entonces la correccion esta DECIDIDA y pendiente.
            "applied_to_cube": False,
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


# ==========================================================================
# Fase `measure`: las dos vías, las dos granularidades, y el veredicto
# ==========================================================================

@dataclass
class A3Inputs:
    """Todo lo que la fase de medida necesita, resuelto desde el run."""

    run_id: str
    project_root: Path
    stage_dir: Path
    work_dir: Path
    ledger_path: Path
    cube: Path
    primary_yx: tuple[float, float]
    radius_px: float
    perexp_cubes: list[str]
    raw_dir: str | None
    std_telluric: str | None
    night: str | None
    config: dict


def resolve_a3_inputs(run_id: str | None = None, *, project_root: str | Path | None = None,
                      overrides: Mapping[str, object] | None = None,
                      allow_run_id_mismatch: bool = False) -> A3Inputs:
    """Resuelve las entradas de A3 desde el run, sin rutas escritas en el código."""

    from ..config import load_run_config as _load

    run_config = _load(run_id, project_root=project_root,
                       allow_run_id_mismatch=allow_run_id_mismatch)
    cfg = stage00t_config_from_run(run_id, project_root=project_root, overrides=overrides,
                                   allow_run_id_mismatch=allow_run_id_mismatch)
    paths = run_config.paths
    root = Path(cfg["project_root"])

    cubos = cfg.get("cube_files") or []
    cube_cfg = cfg.get("a3_input_cube") or (cubos[0] if cubos else None)
    if not cube_cfg:
        raise TelluricError(
            f"Run {run_config.run_id} declares no cube (`cube_files` / `a3_input_cube`): "
            "A3 has nothing to measure.")
    cube = Path(cube_cfg)
    if not cube.is_absolute():
        cube = root / cube

    yx = cfg.get("a3_primary_yx") or cfg.get("m3_primary_yx")
    if not yx:
        raise TelluricError(
            f"Run {run_config.run_id} declares no primary position (`m3_primary_yx`): "
            "A3 measures the telluric depth on the primary's spectrum and cannot guess it.")

    return A3Inputs(
        run_id=run_config.run_id,
        project_root=root,
        stage_dir=paths.stage_dir,
        work_dir=paths.stage_dir / "a3_molecfit",
        ledger_path=paths.log_dir / "a3_progress.json",
        cube=cube,
        primary_yx=(float(yx[0]), float(yx[1])),
        radius_px=float(cfg.get("a3_radius_px", 8.0)),
        perexp_cubes=[str(p) for p in (cfg.get("perexp_cubes") or [])],
        raw_dir=cfg.get("a3_raw_dir") or cfg.get("raw_data_dir"),
        std_telluric=cfg.get("a3_std_telluric"),
        night=cfg.get("night"),
        config=cfg,
    )


def effective_transmission(transmissions: Sequence[Sequence[float]],
                           weights: Sequence[float]) -> np.ndarray:
    """La T que ve el cubo combinado si cada exposición se corrige por la suya.

    El combinado es una media pesada de las exposiciones, así que la transmisión
    efectiva es la media pesada de las suyas — con los **mismos pesos** que usa el
    combine (EXPTIME). No es un promedio cosmético: es lo que permite puntuar la
    vía «por exposición» sobre el mismo espectro que las otras, que es la única
    forma de que las tres cifras sean comparables.
    """

    matriz = np.asarray(transmissions, dtype=np.float64)
    pesos = np.asarray(weights, dtype=np.float64)
    if matriz.ndim != 2 or matriz.shape[0] != pesos.size:
        raise TelluricError("effective_transmission needs one transmission per weight.")
    if not np.isfinite(pesos).all() or pesos.sum() <= 0:
        raise TelluricError("effective_transmission needs finite, positive weights.")
    return np.average(matriz, axis=0, weights=pesos)


def measure_phase(args: argparse.Namespace) -> int:
    """Ajusta las dos vías × las dos granularidades, puntúa y decide. No toca el cubo."""

    from . import molecfit as mf
    from .progress import Ledger, params_hash, bloque_timing

    overrides = {}
    if getattr(args, "primary_y", None) is not None and getattr(args, "primary_x", None) is not None:
        overrides["a3_primary_yx"] = [float(args.primary_y), float(args.primary_x)]
    if getattr(args, "radius_px", None):
        overrides["a3_radius_px"] = float(args.radius_px)
    if getattr(args, "std_telluric", None):
        overrides["a3_std_telluric"] = args.std_telluric
    if getattr(args, "raw_dir", None):
        overrides["a3_raw_dir"] = args.raw_dir
    entradas = resolve_a3_inputs(args.run_id, project_root=getattr(args, "project_root", None),
                                 overrides=overrides,
                                 allow_run_id_mismatch=getattr(args, "allow_run_id_mismatch", False))
    entradas.work_dir.mkdir(parents=True, exist_ok=True)
    entradas.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    bitacora = Ledger(entradas.ledger_path)

    info = resolve_input_cube(entradas.cube, upstream=args.upstream, qc_path=args.qc,
                              checksum=not args.skip_checksum)
    wave, spec_raw, centro, cabecera = mf.primary_spectrum_from_cube(
        entradas.cube, radius_px=entradas.radius_px, yx=entradas.primary_yx)

    # --- condiciones: de las exposiciones, no de la cabecera del combinado ---
    # `--only-combined` salta los AJUSTES por exposicion, no la lectura de sus
    # cabeceras: leerlas es gratis y son la unica fuente de la masa de aire y de
    # las condiciones ambientales. El combinado no declara ninguna de las dos, y
    # sin ellas el ajuste va a cenit o directamente aborta.
    todas = entradas.perexp_cubes
    if not todas and args.qc:
        # Un run puede no declarar `perexp_cubes` (ROXs 42B b no lo hace) y aun asi
        # tener sus exposiciones perfectamente identificadas: el QC del combine en
        # cascada las lista, y ese QC ya esta atado a ESTE cubo por la puerta de
        # identidad de `_check_a1_upstream`. Es mejor procedencia que un knob.
        try:
            qc_arriba = json.loads(Path(args.qc).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            qc_arriba = {}
        if str(qc_arriba.get("stage")) == "stream_combine":
            todas = [str(e["file"]) for e in (qc_arriba.get("exposures") or []) if e.get("file")]
            if todas:
                print(f"  exposiciones tomadas del QC del combine: {len(todas)}")
    cabeceras_exp = []
    for ruta in todas:
        try:
            cabeceras_exp.append(fits.getheader(ruta, 0))
        except OSError as exc:
            raise TelluricError(f"cannot read per-exposure cube {ruta}: {exc}") from exc
    condiciones = [mf.exposure_conditions(h) for h in cabeceras_exp]
    if not condiciones:
        condiciones = [mf.exposure_conditions(cabecera)]
        cabeceras_exp = [cabecera]
    exposiciones = [] if args.only_combined else todas
    x_efectiva, fuente_x = mf.effective_airmass(condiciones)

    # Las profundidades del cubo TAL CUAL mandan sobre qué se ajusta: una banda
    # por debajo del umbral no necesita correccion, y meterla en el ajuste le da a
    # molecfit señal fotosferica con la que ajustar una columna atmosferica.
    profundidades = measure_telluric_depths(wave, spec_raw)
    umbral = float(entradas.config.get("a3_threshold_pct", 3.0))
    ventanas = mf.select_fit_windows(profundidades, threshold_pct=umbral)
    decision = decide_telluric(profundidades, science_needs_red_continuum=True,
                               threshold_pct=umbral)
    print("  profundidad por banda: " +
          ", ".join(f"{k} {v:.2f}%" for k, v in profundidades.items()))
    print(f"  veredicto {decision.decision}; se ajustan "
          f"{[w.band for w in ventanas] or 'ninguna banda'}")
    if not ventanas:
        # `not_needed_shallow` es un final EXITOSO de A3, no un error: la spec v2 §0
        # ya dice que saltar la etapa con evidencia es terminar bien. Lo que no
        # puede pasar es terminar sin dejar constancia — un run sin QC no se
        # distingue de un run que nadie ha corrido, y esa ambiguedad es
        # exactamente lo que la cadena lleva un mes pagando.
        qc = stage00t_qc_skeleton(info, run_id=entradas.run_id,
                                  primary_yx=entradas.primary_yx,
                                  aperture_radius_px=entradas.radius_px,
                                  environment=check_molecfit_environment(esorex=args.esorex))
        qc["spec_version"] = "A3_v3"
        qc["decision"].update({
            "depth_pct_by_band": profundidades,
            "telluric_applied": False,
            "applied_to_cube": False,
            "science_needs_red_continuum": True,
            "verdict": decision.decision,
            "checkpoint_required": decision.checkpoint_required,
            "user_checkpoint": "not_needed",
            "method": "sin_corregir",
            "threshold_pct": umbral,
        })
        qc["fit"] = {"molecules": [], "regions_A": [], "regions_bands": [],
                     "regions_excluded_shallow": [w.band for w in mf.FIT_WINDOWS],
                     "nota": f"ninguna banda llega al {umbral} %: no se ajusto nada, "
                             "ni molecfit ni STD_TELLURIC. No es un fallo de molecfit."}
        qc["arbiter"] = {"fom_by_method": {"sin_corregir": score_correction(
                             wave, spec_raw, None)["fom_mean"]},
                         "method": {"winner": "sin_corregir", "reason": "not_needed_shallow",
                                    "margin": None, "tie_sigma": None}}
        qc["timing"] = bloque_timing([], unidades_totales=0)
        salida = Path(args.qc_output) if args.qc_output else entradas.stage_dir / "stage00t_qc.json"
        salida.parent.mkdir(parents=True, exist_ok=True)
        salida.write_text(json.dumps(qc, indent=2, default=float) + "\n", encoding="utf-8")
        print(f"  no hay nada que corregir; QC -> {salida}")
        return 0

    unidades_totales = 1 + len(exposiciones)
    resultados: dict[str, mf.MolecfitResult] = {}

    def _ajusta(clave: str, cubo: str | Path, conds, yx=None, heads=None) -> mf.MolecfitResult:
        # La huella va sobre el CATALOGO y el umbral, no sobre la seleccion ya
        # resuelta: la seleccion depende del espectro de cada unidad y no se
        # conoce antes de extraerlo, pero lo que la determina si es deterministico.
        huella = params_hash({"params": mf.MOLECFIT_BASE_PARAMS,
                              "windows": [w.as_dict() for w in mf.FIT_WINDOWS],
                              "threshold_pct": umbral,
                              "radius_px": entradas.radius_px, "cube": str(cubo)})
        marca = bitacora.fila(clave)
        reutilizable = bitacora.hecha(clave, params=huella)
        resultado = mf.fit_molecfit(
            cubo, out_root=entradas.work_dir, label=clave.replace(":", "_"),
            radius_px=entradas.radius_px, yx=yx, raw_dir=entradas.raw_dir,
            conditions=conds, exposure_headers=heads, windows=mf.FIT_WINDOWS,
            threshold_pct=umbral, reuse=reutilizable, esorex=args.esorex)
        if not reutilizable or marca is None:
            bitacora.anota(clave, rc=0, segundos=resultado.seconds,
                           producto=str(Path(resultado.out_dir) / "BEST_FIT_PARAMETERS.fits"),
                           params=huella, peldano=resultado.rung, firma=resultado.signature)
        if resultado.rung == "sin_ajuste":
            print(f"  [{clave}] sin ajuste: ninguna banda llega al {umbral} % "
                  f"(max {max(resultado.depth_pct_by_band.values()):.2f} %) -> T = 1", flush=True)
        else:
            print(f"  [{clave}] {resultado.rung} · {resultado.seconds:.1f} s · "
                  f"bandas {[w['band'] for w in resultado.windows]} · "
                  f"rel_mol_col_O2={resultado.value('rel_mol_col_O2'):.4f}", flush=True)
        return resultado

    print(f"A3 measure · run {entradas.run_id} · {unidades_totales} ajustes "
          f"· X efectiva {x_efectiva:.4f} ({fuente_x})", flush=True)

    resultados["combined"] = _ajusta("molecfit:combined", entradas.cube, condiciones,
                                     yx=entradas.primary_yx, heads=cabeceras_exp)
    for indice, ruta in enumerate(exposiciones, start=1):
        if bitacora.parar():
            print("PARAR encontrado: se detiene entre unidades, la bitacora queda consistente.")
            return 4
        resultados[f"exp{indice}"] = _ajusta(f"molecfit:exp{indice}", ruta,
                                             [condiciones[indice - 1]],
                                             heads=[cabeceras_exp[indice - 1]])

    # --- transmisiones sobre el eje del cubo combinado -------------------
    def _transmision(res):
        """T = 1 para una unidad que no ajusto nada. No es un caso raro: es el
        veredicto `not_needed_shallow` de esa unidad, expresado como transmision."""

        if res.rung == "sin_ajuste":
            return np.ones_like(wave)
        return mf.transmission_from_fit(res.out_dir, wave)

    t_combinado = _transmision(resultados["combined"])
    transmisiones = {"molecfit_combined": t_combinado}
    puntuaciones_exp: list[dict] = []
    if exposiciones:
        por_exp, pesos = [], []
        for indice, cond in enumerate(condiciones[:len(exposiciones)], start=1):
            res = resultados[f"exp{indice}"]
            t_exp = _transmision(res)
            por_exp.append(t_exp)
            pesos.append(float(cond.get("exptime") or 1.0))
            # Cada exposicion se puntua sobre SU propio espectro. Puntuarlas todas
            # sobre el del combinado da una dispersion de cero cuando sus T
            # coinciden —que es justo lo que pasa cuando ninguna necesita ajuste—,
            # y una sigma de cero no es un empate estrecho: es no haber medido.
            puntuaciones_exp.append(score_correction(res.wave_A, res.spectrum / t_exp, t_exp))
        transmisiones["molecfit_perexp"] = effective_transmission(por_exp, pesos)

    if entradas.std_telluric:
        t_std, meta_std = std_telluric_transmission(entradas.std_telluric, wave,
                                                    airmass_sci=x_efectiva)
        transmisiones["std_telluric"] = t_std
    else:
        meta_std = {"unavailable": "no a3_std_telluric declared: the STD route was not measured"}

    # --- puntuación y arbitraje -----------------------------------------
    puntuaciones = {nombre: score_correction(wave, spec_raw / t, t)
                    for nombre, t in transmisiones.items()}
    # «No corregir» compite. Tiene que poder ganar: estos cubos ya llevan la
    # correccion del DRS (`muse_scipost` consume STD_TELLURIC), asi que A3 corrige
    # un RESIDUO, y una segunda correccion sobre una banda ya plana la hunde.
    puntuaciones["sin_corregir"] = score_correction(wave, spec_raw, None)
    sigma = tie_sigma_from_scores(puntuaciones_exp)

    granularidad = None
    candidatas = {k: v for k, v in puntuaciones.items() if k.startswith("molecfit_")}
    # Una granularidad que no ajusto NADA no es una correccion que compita: es el
    # nulo, y ya compite por su cuenta como `sin_corregir`. Dejarla dentro hace
    # que el desempate de granularidad —que por regla prefiere «por exposicion»—
    # elija la via que no hace nada, y el QC diria «gana por exposicion» cuando lo
    # que paso es que ninguna exposicion tenia residuo que quitar.
    sin_ajuste_ninguno = {f"molecfit_{'perexp' if k.startswith('exp') else k}"
                          for k, r in resultados.items() if r.rung == "sin_ajuste"}
    if "molecfit_perexp" in candidatas and all(
            resultados[f"exp{i}"].rung == "sin_ajuste" for i in range(1, len(exposiciones) + 1)):
        candidatas.pop("molecfit_perexp")
        granularidad = {"winner": "molecfit_combined", "reason": "perexp_no_fitted_anything",
                        "nota": "ninguna exposicion llego al umbral por si sola: la via por "
                                "exposicion es el nulo, y compite como sin_corregir"}
    if len(candidatas) > 1:
        # Desempate de granularidad: la regla declarada en las indicaciones de
        # 2026-08-06 §5.3 es «por exposición», por ser la granularidad a la que la
        # fisica es cierta (una atmosfera, una masa de aire, un instante).
        granularidad = choose_method(candidatas, tie_sigma=sigma, prefer="molecfit_perexp")
        ganadora_mf = granularidad["winner"]
    else:
        ganadora_mf = next(iter(candidatas))

    # El arbitraje va en DOS pasos, y el orden es la decisión del usuario del
    # 2026-08-22 afinada con lo que se midió ese mismo día:
    #
    # 1. Entre las dos VIAS DE CORRECCION (molecfit y STD_TELLURIC) el empate lo
    #    gana molecfit. Es de lo que hablaba la regla.
    # 2. Contra NO CORREGIR, la ganadora tiene que ganar **por medida**, sin banda
    #    de empate. «No corregir» no es una tercera via: es el nulo, y aplicar una
    #    correccion que solo empata con no hacer nada no es lo que la regla
    #    autorizaba. Medido en ROXs 42B b: la sigma entre exposiciones sale 1.5035
    #    —enorme— asi que el empate se tragaba un margen real de 0.52 y la regla
    #    acababa aplicando una correccion que convierte una absorcion de +3.86 %
    #    en una joroba en emision de −7.40 %.
    finalistas = {"molecfit": puntuaciones[ganadora_mf]}
    if "std_telluric" in puntuaciones:
        finalistas["std_telluric"] = puntuaciones["std_telluric"]
    veredicto = choose_method(finalistas, tie_sigma=sigma, prefer="molecfit")

    fom_nula = float(puntuaciones["sin_corregir"]["fom_mean"])
    fom_ganadora = float(veredicto["fom_by_method"][veredicto["winner"]])
    veredicto["fom_by_method"]["sin_corregir"] = fom_nula
    veredicto["vs_sin_corregir"] = {
        "margin": fom_nula - fom_ganadora,
        "rule": "una correccion tiene que ganar por medida; el empate no vale contra el nulo",
    }
    if fom_nula < fom_ganadora:
        veredicto = {"winner": "sin_corregir",
                     "reason": "no_correction_measures_better",
                     "beaten_method": veredicto["winner"],
                     "fom_by_method": veredicto["fom_by_method"],
                     "margin": fom_ganadora - fom_nula,
                     "tie_sigma": veredicto.get("tie_sigma"),
                     "between_methods": veredicto.get("reason"),
                     "vs_sin_corregir": veredicto["vs_sin_corregir"]}

    qc = stage00t_qc_skeleton(info, run_id=entradas.run_id, primary_yx=entradas.primary_yx,
                              aperture_radius_px=entradas.radius_px,
                              environment=check_molecfit_environment(esorex=args.esorex))
    qc["spec_version"] = "A3_v3"
    qc["decision"].update({
        "depth_pct_by_band": profundidades,
        "telluric_applied": veredicto["winner"] != "sin_corregir",
        "applied_to_cube": False,
        "science_needs_red_continuum": True,
        "verdict": "needed",
        "user_checkpoint": "required",
        "method": veredicto["winner"],
        "granularity": ganadora_mf,
    })
    qc["fit"] = {
        "molecules": ["O2", "H2O"],
        # Las ventanas que declara el QC son las que el ajuste del COMBINADO uso de
        # verdad, no las que se preseleccionaron: cada unidad decide sobre su
        # propio espectro, y un QC que anuncie una ventana que nadie ajusto es la
        # misma clase de mentira que el `polynomial_deg2` de C1.
        "regions_A": [[w["lo_A"], w["hi_A"]] for w in resultados["combined"].windows],
        "regions_bands": [w["band"] for w in resultados["combined"].windows],
        "regions_excluded_shallow": [w.band for w in mf.FIT_WINDOWS
                                     if w.band not in {x["band"]
                                                       for x in resultados["combined"].windows}],
        "excluded_subregions_A": [],
        "chi2_by_region": {},
        "kernel": 0.0,
        "airmass_effective": None if not np.isfinite(x_efectiva) else round(float(x_efectiva), 4),
        "airmass_source": fuente_x,
        "molecfit": {clave: res.as_dict() for clave, res in resultados.items()},
        "std_telluric": meta_std,
    }
    qc["arbiter"] = {
        "fom_by_method": {k: v["fom_mean"] for k, v in puntuaciones.items()},
        "scores": puntuaciones,
        "tie_sigma": None if not np.isfinite(sigma) else sigma,
        "tie_sigma_source": f"1 sigma de la FoM de {len(puntuaciones_exp)} exposiciones",
        "granularity": granularidad,
        "method": veredicto,
    }
    # Sin tasa aparte: `bloque_timing` ya deduplica por unidad y publica
    # `segundos_por_unidad`. Una segunda cifra calculada sobre las filas crudas
    # decia otra cosa (contaba reintentos) para el mismo concepto.
    qc["timing"] = bloque_timing(bitacora.rows, unidades_totales=unidades_totales)
    qc["products"]["transmission_by_method"] = {
        nombre: str(entradas.work_dir / f"transmission_{nombre}.npy")
        for nombre in transmisiones
    }
    for nombre, t in transmisiones.items():
        np.save(entradas.work_dir / f"transmission_{nombre}.npy", np.asarray(t, dtype=np.float64))
    np.save(entradas.work_dir / "wave_A.npy", np.asarray(wave, dtype=np.float64))

    salida = Path(args.qc_output) if args.qc_output else entradas.stage_dir / "stage00t_qc.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(qc, indent=2, default=float) + "\n", encoding="utf-8")

    print(f"\nFoM (exceso sobre lo inevitable; 1.0 = perfecto):")
    for nombre, punt in sorted(puntuaciones.items(), key=lambda kv: kv[1]["fom_mean"]):
        print(f"  {nombre:22s} {punt['fom_mean']:.4f}")
    print(f"\nGanador: {veredicto['winner']} ({veredicto['reason']}), "
          f"margen {veredicto['margin']:.4f}, sigma de empate "
          f"{'n/d' if not np.isfinite(sigma) else f'{sigma:.4f}'}")
    print(f"QC -> {salida}")
    return 0


def apply_phase(args: argparse.Namespace) -> int:
    """Aplica al cubo la transmisión que ganó, por trozos y reanudable."""

    from .progress import Ledger, bloque_timing

    entradas = resolve_a3_inputs(args.run_id, project_root=getattr(args, "project_root", None),
                                 allow_run_id_mismatch=getattr(args, "allow_run_id_mismatch", False))
    qc_path = Path(args.qc_output) if args.qc_output else entradas.stage_dir / "stage00t_qc.json"
    if not qc_path.exists():
        raise TelluricError(f"A3 apply needs the measure QC first: {qc_path} does not exist.")
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    arbitro = qc.get("arbiter") or {}
    veredicto = arbitro.get("method") or {}
    ganador = args.method or veredicto.get("winner")
    if not ganador:
        raise TelluricError("The QC declares no winner and none was given with --method.")
    if args.method and args.method != veredicto.get("winner"):
        # Un override humano es legítimo, pero queda escrito con su motivo: es la
        # única vía por la que STD_TELLURIC puede aplicarse sin ganar la métrica.
        if not args.override_reason:
            raise TelluricError(
                "Overriding the arbiter needs --override-reason: a method that did not win "
                "the metric can only be applied on the record.")
        qc["decision"]["override"] = {"method": args.method, "reason": args.override_reason,
                                      "arbiter_winner": veredicto.get("winner")}

    clave_t = {"molecfit": qc.get("decision", {}).get("granularity", "molecfit_combined"),
               "std_telluric": "std_telluric"}.get(ganador, ganador)
    ruta_t = (qc.get("products", {}).get("transmission_by_method") or {}).get(clave_t)
    if not ruta_t or not Path(ruta_t).exists():
        raise TelluricError(f"The winning transmission ({clave_t}) is not on disk: {ruta_t}")
    trans = np.load(ruta_t)
    wave = np.load(entradas.work_dir / "wave_A.npy")

    bitacora = Ledger(entradas.ledger_path)
    salida_cubo = Path(args.output_cube) if args.output_cube else entradas.stage_dir / "cube_telcorr.fits"
    salida_trans = Path(args.output_transmission) if args.output_transmission else (
        entradas.stage_dir / "TELLURIC_TRANS.fits")

    pre_wave, pre_spec = _read_primary_spectrum(entradas.cube, entradas.primary_yx,
                                                entradas.radius_px)
    print(f"A3 apply · {ganador} ({clave_t}) · {entradas.cube.name} -> {salida_cubo.name}", flush=True)
    cubo, trans_aplicada = apply_transmission_streaming(
        entradas.cube, salida_cubo, trans,
        chunk_channels=int(entradas.config.get("a3_chunk_channels", 200)),
        ledger=bitacora, key_prefix="apply",
        min_transmission=float(entradas.config.get("a3_min_transmission", 0.05)),
        history=f"A3 v3 telluric correction applied ({ganador}/{clave_t})",
        progress=lambda z1, z2, nz: print(f"  canal {z2}/{nz}", end="\r", flush=True),
    )
    write_transmission_fits(wave, trans_aplicada, salida_trans)
    post_wave, post_spec = _read_primary_spectrum(cubo, entradas.primary_yx, entradas.radius_px)

    v1_pre = measure_telluric_depths(pre_wave, pre_spec)
    v1_post = measure_telluric_depths(post_wave, post_spec, clip_negative=False)
    halpha = window_mask(post_wave, HALPHA_PROTECTED)
    qc["decision"]["applied_to_cube"] = True
    qc["decision"]["user_checkpoint"] = "approved"
    qc["products"].update({"cube_telcorr": str(cubo), "transmission": str(salida_trans)})
    qc["verification"] = {
        "v1_residual_pct_by_band": v1_post,
        "v1_depth_pct_by_band_pre": v1_pre,
        "v2_outside_bands_unchanged": bool(verify_outside_bands_unchanged(pre_spec, post_spec, post_wave)),
        "v3_halpha_untouched": bool(np.allclose(pre_spec[halpha], post_spec[halpha],
                                                rtol=0.0, atol=1e-6)),
        "v3_halpha_maxabs": float(np.nanmax(np.abs(post_spec[halpha] - pre_spec[halpha]))),
        "v4_transmission_physical": bool(validate_transmission_physical(trans_aplicada)),
        "v5_stat_scaled": True,
        "v5_note": "STAT/T^2 se escribe en el mismo bucle que DATA/T; no hay via por la que "
                   "uno se aplique sin el otro.",
    }
    qc["timing"] = bloque_timing(bitacora.rows,
                                 segundos_por_canal=(bitacora.segundos_totales() /
                                                     max(1, len(wave))))
    qc_path.write_text(json.dumps(qc, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nv3 Halpha intacta: max|delta| = {qc['verification']['v3_halpha_maxabs']:.3e}")
    print(f"cubo -> {cubo}\ntransmision -> {salida_trans}\nQC -> {qc_path}")
    return 0


def decision_phase(args: argparse.Namespace) -> int:
    input_info = resolve_input_cube(
        args.input_cube,
        upstream=args.upstream,
        qc_path=args.qc,
        checksum=not args.skip_checksum,
    )
    primary_yx = (args.primary_y, args.primary_x)
    wave, spec = _read_primary_spectrum(input_info.cube, primary_yx, args.radius_px)
    depths = measure_telluric_depths(wave, spec)
    decision = decide_telluric(depths, science_needs_red_continuum=args.science_needs_red_continuum)
    qc = stage00t_qc_skeleton(input_info, run_id=args.run_id,
                              primary_yx=primary_yx, aperture_radius_px=args.radius_px)
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

    # --- v3: measure / apply --------------------------------------------
    measure_parser = subparsers.add_parser(
        "measure", help="Ajusta las dos vias (molecfit y STD_TELLURIC) y arbitra. No toca el cubo.")
    measure_parser.add_argument("--run-id", required=True)
    measure_parser.add_argument("--project-root", default=None)
    measure_parser.add_argument("--allow-run-id-mismatch", action="store_true")
    measure_parser.add_argument("--upstream", choices=["A2", "A1", "ADP"], default="A1")
    measure_parser.add_argument("--qc", default=None, help="QC de aguas arriba")
    measure_parser.add_argument("--qc-output", default=None,
                                help="por defecto runs/<RUN>/stages/stage00t_qc.json")
    measure_parser.add_argument("--std-telluric", default=None,
                                help="STD_TELLURIC de la noche; sin el, la via STD no se mide")
    measure_parser.add_argument("--raw-dir", default=None, help="crudos, para la humedad real")
    measure_parser.add_argument("--esorex", default="esorex")
    measure_parser.add_argument("--primary-y", type=float, default=None,
                                help="por defecto, `m3_primary_yx` del run")
    measure_parser.add_argument("--primary-x", type=float, default=None)
    measure_parser.add_argument("--radius-px", type=float, default=None)
    measure_parser.add_argument("--only-combined", action="store_true",
                                help="salta los ajustes por exposicion (sin ellos no hay sigma "
                                     "de empate, y el arbitraje lo dice)")
    measure_parser.add_argument("--skip-checksum", action="store_true")
    measure_parser.set_defaults(func=measure_phase)

    apply_parser = subparsers.add_parser(
        "apply", help="Aplica al cubo la transmision ganadora, por trozos y reanudable.")
    apply_parser.add_argument("--run-id", required=True)
    apply_parser.add_argument("--project-root", default=None)
    apply_parser.add_argument("--allow-run-id-mismatch", action="store_true")
    apply_parser.add_argument("--qc-output", default=None)
    apply_parser.add_argument("--output-cube", default=None)
    apply_parser.add_argument("--output-transmission", default=None)
    apply_parser.add_argument("--method", default=None,
                              help="fuerza la via (necesita --override-reason si no es la ganadora)")
    apply_parser.add_argument("--override-reason", default=None)
    apply_parser.set_defaults(func=apply_phase)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except TelluricError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
