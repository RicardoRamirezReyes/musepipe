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


def write_csv(path, rows, fieldnames) -> None:
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


__all__ = [
    "get_cube_data",
    "read_json",
    "read_wavelength_axis",
    "read_wavelengths_and_masks",
    "write_csv",
    "write_json",
]
