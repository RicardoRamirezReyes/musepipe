"""Frozen internal cache format for external libraries (WP-G3R-2).

The fetch script converts every published source (spectra or tracks) into this
one on-disk format; the WP-G3R-3/4/5 adapters read it back. Keeping the reader
and writer here means the format is defined in exactly one place (rule §0.5.2).

Layout (frozen):

* Spectra (BT-Settl grid nodes, empirical templates): one ``.npz`` per spectrum
  with ``wave_A`` (float64, Å), ``flux`` (F_lambda, float64) and ``meta_json``
  (a JSON string carrying identity + provenance). Wavelengths are trimmed to
  ``[wave_lo, wave_hi]`` (default 4000–10000 Å) to bound size/IO.
* Tracks: one ``.npz`` per family holding the irregular published grid as the
  1-D arrays in :data:`TRACK_ARRAYS` plus ``meta_json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import TemplateSpectrum

TRIM_WAVE_LO_A = 4000.0
TRIM_WAVE_HI_A = 10000.0

# Published track grids come as irregular samples of these quantities.
TRACK_ARRAYS = (
    "mass_msun",
    "age_gyr",
    "teff_k",
    "l_bol_lsun",
    "radius_rsun",
    "logg",
)


def _meta_to_json(meta) -> str:
    if not isinstance(meta, dict):
        raise RuntimeError(f"meta must be a dict, got {type(meta).__name__}")
    try:
        return json.dumps(meta, sort_keys=True)
    except TypeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"meta is not JSON-serializable: {exc}") from exc


def write_spectrum_npz(
    out_path: str | Path,
    wave_A,
    flux,
    meta,
    *,
    wave_lo: float = TRIM_WAVE_LO_A,
    wave_hi: float = TRIM_WAVE_HI_A,
) -> Path:
    """Write one cached spectrum, trimmed to ``[wave_lo, wave_hi]`` and sorted
    ascending in wavelength. Returns the output path.

    ``RuntimeError`` if wave/flux shapes disagree or nothing survives the trim.
    """
    wave = np.asarray(wave_A, dtype=np.float64)
    fl = np.asarray(flux, dtype=np.float64)
    if wave.shape != fl.shape or wave.ndim != 1:
        raise RuntimeError(
            f"wave/flux must be matching 1-D arrays, got {wave.shape} vs {fl.shape}")
    order = np.argsort(wave)
    wave, fl = wave[order], fl[order]
    sel = (wave >= wave_lo) & (wave <= wave_hi)
    if not sel.any():
        raise RuntimeError(
            f"spectrum has no samples in [{wave_lo}, {wave_hi}] Å "
            f"(covers {wave.min():.1f}–{wave.max():.1f} Å) — coverage failure")
    meta_json = _meta_to_json(meta)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, wave_A=wave[sel], flux=fl[sel], meta_json=meta_json)
    # np.savez appends .npz if absent; normalise the returned path.
    return out_path if out_path.suffix == ".npz" else out_path.with_suffix(".npz")


def load_spectrum_npz(path: str | Path) -> TemplateSpectrum:
    """Read a cached spectrum back into a :class:`TemplateSpectrum`."""
    with np.load(path, allow_pickle=False) as z:
        wave = np.asarray(z["wave_A"], dtype=np.float64)
        flux = np.asarray(z["flux"], dtype=np.float64)
        meta = json.loads(str(z["meta_json"]))
    return TemplateSpectrum(wave, flux, meta=meta)


def write_tracks_npz(out_path: str | Path, arrays: dict, meta) -> Path:
    """Write one cached track family. ``arrays`` must contain every key in
    :data:`TRACK_ARRAYS` as equal-length 1-D arrays. Returns the output path."""
    missing = [k for k in TRACK_ARRAYS if k not in arrays]
    if missing:
        raise RuntimeError(f"tracks missing required arrays: {missing}")
    cols = {k: np.asarray(arrays[k], dtype=np.float64) for k in TRACK_ARRAYS}
    lengths = {k: v.shape for k, v in cols.items()}
    n = cols["mass_msun"].shape
    if any(v.ndim != 1 for v in cols.values()) or any(s != n for s in lengths.values()):
        raise RuntimeError(f"track arrays must be equal-length 1-D, got {lengths}")
    meta_json = _meta_to_json(meta)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, meta_json=meta_json, **cols)
    return out_path if out_path.suffix == ".npz" else out_path.with_suffix(".npz")


def load_tracks_npz(path: str | Path) -> dict:
    """Read a cached track family into ``{**arrays, "meta": dict}``."""
    with np.load(path, allow_pickle=False) as z:
        out = {k: np.asarray(z[k], dtype=np.float64) for k in TRACK_ARRAYS}
        out["meta"] = json.loads(str(z["meta_json"]))
    return out


__all__ = [
    "TRACK_ARRAYS",
    "TRIM_WAVE_HI_A",
    "TRIM_WAVE_LO_A",
    "load_spectrum_npz",
    "load_tracks_npz",
    "write_spectrum_npz",
    "write_tracks_npz",
]
