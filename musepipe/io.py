"""FITS, JSON, CSV, and NumPy IO helpers shared by pipeline stages."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from astropy.io import fits


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path, rows, fieldnames=None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_cube_data(hdul, cube_index=0):
    """Return a 3D cube from a FITS HDUList with common pipeline layouts."""

    if hdul[0].data is not None:
        data = hdul[0].data
    elif "CUBES" in hdul:
        data = hdul["CUBES"].data
    else:
        raise RuntimeError("No cube data found in FITS file.")

    if data.ndim == 4:
        return data[int(cube_index)]
    if data.ndim == 3:
        return data
    raise RuntimeError(f"Expected a 3D or 4D cube, got {data.shape}.")


def read_wavelength_axis(hdul, data_shape=None):
    """Return a wavelength axis from an extension or standard FITS keywords."""

    if "WAVELENGTH" in hdul:
        return hdul["WAVELENGTH"].data.astype(np.float64)

    header = hdul[0].header
    if data_shape is None:
        data = get_cube_data(hdul)
        data_shape = data.shape
    nz = int(data_shape[0])

    if all(key in header for key in ("WMIN", "DW")):
        return float(header["WMIN"]) + np.arange(nz, dtype=np.float64) * float(header["DW"])
    if all(key in header for key in ("CRVAL3", "CDELT3")):
        crpix = float(header.get("CRPIX3", 1.0))
        return float(header["CRVAL3"]) + (
            np.arange(nz, dtype=np.float64) + 1.0 - crpix
        ) * float(header["CDELT3"])
    raise RuntimeError("Could not recover wavelength axis from FITS file.")


def read_wavelengths_and_masks(stack_path, good_mask_path=None, bad_mask_path=None):
    """Read wavelength, good-mask, and bad-mask arrays from a stage FITS stack."""

    with fits.open(stack_path, memmap=True) as hdul:
        waves = hdul["WAVELENGTH"].data.astype(np.float64)
        if "GOOD_WAVE_MASK" in hdul:
            good = hdul["GOOD_WAVE_MASK"].data.astype(bool)
        else:
            good = np.ones(waves.size, dtype=bool)
        if "BAD_WAVE_MASK" in hdul:
            bad = hdul["BAD_WAVE_MASK"].data.astype(bool)
        else:
            bad = ~good

    if good_mask_path is not None and Path(good_mask_path).exists():
        good = np.load(good_mask_path).astype(bool)
    if bad_mask_path is not None and Path(bad_mask_path).exists():
        bad = np.load(bad_mask_path).astype(bool)
        good = good & ~bad
    return waves, good, bad


#: Unidad nativa de los cubos MUSE flux-calibrados por scipost.
MUSE_NATIVE_BUNIT = "10**(-20)*erg/s/cm**2/Angstrom"


def cube_bunit(path, ext=None) -> str:
    """`BUNIT` de un cubo, buscando en el HDU indicado y luego en el resto.

    Devuelve "" si ningun HDU lo declara: la unidad se desconoce y quien llama
    debe decidir (no se inventa la nativa de MUSE).
    """
    with fits.open(path, memmap=True) as hdul:
        order = [ext] if ext is not None else []
        order += [i for i in range(len(hdul)) if i != ext]
        for idx in order:
            try:
                value = hdul[idx].header.get("BUNIT")
            except (IndexError, KeyError):
                continue
            if value:
                return str(value)
    return ""


def resolve_bunit(cfg, *, stack_bunit=None, override_key=None) -> str:
    """Unidad de flujo de un producto espectral, en orden de confianza.

    1. la que trae el cubo de trabajo (`stack_bunit`, si el stack la declara);
    2. la del cubo de entrada del run (`cfg["cube_files"][0]`) — los stacks
       escritos antes de que B1/B2 propagaran `BUNIT` no la llevan, y re-correr
       B1/B2 solo por la cabecera cuesta ~20 min;
    3. el override explicito de config (`override_key`), si existe;
    4. "" — desconocida; el producto lo dira en vez de fingir una unidad.
    """
    if stack_bunit:
        return str(stack_bunit)
    cubes = (cfg or {}).get("cube_files") or []
    if cubes:
        try:
            found = cube_bunit(cubes[0], ext=(cfg or {}).get("data_ext"))
        except (OSError, ValueError):
            found = ""
        if found:
            return found
    if override_key:
        return str((cfg or {}).get(override_key) or "")
    return ""


def bunit_to_cgs_scale(bunit) -> float | None:
    """Factor que lleva `bunit` a erg/s/cm2/A, o None si no es esa dimension.

    `'10**(-20)*erg/s/cm**2/Angstrom'` -> `1e-20`. Sirve para que una etapa lea
    la escala fisica del propio producto en vez de depender de un knob de
    config (ver `h03_flux_unit_cgs`).
    """
    text = str(bunit or "").strip()
    if not text:
        return None
    import warnings

    import astropy.units as u

    try:
        with warnings.catch_warnings():
            # El BUNIT de MUSE lleva varias barras: la FITS lo desaconseja y
            # astropy avisa, pero lo interpreta bien.
            warnings.simplefilter("ignore", u.UnitsWarning)
            unit = u.Unit(text)
        return float((1.0 * unit).to(u.erg / u.s / u.cm**2 / u.AA).value)
    except (ValueError, TypeError, u.UnitConversionError):
        return None


def flux_unit_cgs(cfg, *, bunit=None, key="h03_flux_unit_cgs") -> float:
    """Escala que lleva el flujo del producto a erg/s/cm2/A.

    Orden: el knob explicito de config (manda, para no mover resultados ya
    congelados) -> el `BUNIT` del propio producto -> error.

    NO hay default silencioso: los defaults divergentes que habia (`1.0` en
    H03/G3, `1e-20` en `models.derived`) significaban que un run sin el knob
    producia L/Mdot con un factor 1e20 de diferencia segun quien lo leyera.
    """
    declared = (cfg or {}).get(key)
    if declared is not None:
        return float(declared)
    scale = bunit_to_cgs_scale(bunit)
    if scale is not None:
        return scale
    raise ValueError(
        f"No se puede fijar la escala de flujo: el config no declara {key!r} y el "
        f"producto no trae un BUNIT interpretable (BUNIT={bunit!r}). Declara "
        f"{key} en runs/<RUN>/config/config.json (los cubos MUSE de scipost son "
        f"{MUSE_NATIVE_BUNIT} -> 1e-20) o re-genera el producto con BUNIT."
    )


__all__ = [
    "MUSE_NATIVE_BUNIT",
    "bunit_to_cgs_scale",
    "flux_unit_cgs",
    "cube_bunit",
    "get_cube_data",
    "read_json",
    "read_wavelength_axis",
    "read_wavelengths_and_masks",
    "resolve_bunit",
    "write_csv",
    "write_json",
]
