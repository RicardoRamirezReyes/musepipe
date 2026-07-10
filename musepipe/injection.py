"""Reusable synthetic-source injection helpers for MUSE cubes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import shutil

import numpy as np

from .psf import evaluate_psf_model


@dataclass(frozen=True)
class InjectionSource:
    """One synthetic source to inject into a cube."""

    y: float
    x: float
    total_line_flux: float
    line_center_A: float
    line_fwhm_A: float
    label: str = "source"
    continuum_flux_density: float = 0.0
    cube_index: int | None = None
    psf_model: dict | None = None
    psf_image: np.ndarray | None = None
    psf_fwhm_scale: float = 1.0
    template_radius_nsigma: float = 5.0


def channel_widths(wavelengths_A):
    wave = np.asarray(wavelengths_A, dtype=np.float64)
    if wave.ndim != 1 or wave.size == 0:
        raise ValueError("wavelengths_A must be a non-empty 1D array.")
    if wave.size == 1:
        return np.ones(1, dtype=np.float64)
    edges = np.empty(wave.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
    edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
    widths = np.diff(edges)
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("wavelengths_A must be finite and increasing.")
    return widths


def gaussian_line_profile(wavelengths_A, center_A, fwhm_A):
    """Return a line profile normalized to unit integrated flux."""

    wave = np.asarray(wavelengths_A, dtype=np.float64)
    sigma = float(fwhm_A) / 2.354820045
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("line_fwhm_A must be positive.")
    profile = np.exp(-0.5 * ((wave - float(center_A)) / sigma) ** 2)
    widths = channel_widths(wave)
    norm = float(np.nansum(profile * widths))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid spectral line normalization.")
    return profile / norm


def _scaled_psf_model(model_doc, fwhm_scale):
    scale = float(fwhm_scale)
    if np.isclose(scale, 1.0):
        return model_doc
    model = deepcopy(model_doc)
    for key in ("fwhm_maj", "fwhm_min"):
        if key in model.get("coefficients", {}):
            coeff = list(model["coefficients"][key].get("coefficients", []))
            model["coefficients"][key]["coefficients"] = [float(value) * scale for value in coeff]
    return model


def normalized_spatial_psf(shape, y, x, *, wavelength_A, psf_model=None, psf_image=None, fwhm_scale=1.0):
    """Return a finite spatial PSF image normalized to unit sum."""

    ny, nx = map(int, shape)
    if psf_image is not None:
        psf = np.asarray(psf_image, dtype=np.float64)
        if psf.shape != (ny, nx):
            raise ValueError(f"psf_image shape {psf.shape} does not match {(ny, nx)}.")
    else:
        if psf_model is None:
            raise ValueError("psf_model is required when psf_image is not supplied.")
        yy, xx = np.indices((ny, nx), dtype=np.float64)
        model = _scaled_psf_model(psf_model, fwhm_scale)
        psf = evaluate_psf_model(model, float(wavelength_A), yy - float(y), xx - float(x))
    psf = np.asarray(psf, dtype=np.float64)
    psf[~np.isfinite(psf)] = 0.0
    psf = np.clip(psf, 0.0, None)
    norm = float(np.nansum(psf))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Spatial PSF has invalid normalization.")
    return psf / norm


def _source_from_mapping(value):
    if isinstance(value, InjectionSource):
        return value
    return InjectionSource(
        y=float(value["y"]),
        x=float(value["x"]),
        total_line_flux=float(value["total_line_flux"]),
        line_center_A=float(value["line_center_A"]),
        line_fwhm_A=float(value["line_fwhm_A"]),
        label=str(value.get("label", "source")),
        continuum_flux_density=float(value.get("continuum_flux_density", 0.0)),
        cube_index=None if value.get("cube_index") is None else int(value["cube_index"]),
        psf_model=value.get("psf_model"),
        psf_image=value.get("psf_image"),
        psf_fwhm_scale=float(value.get("psf_fwhm_scale", 1.0)),
        template_radius_nsigma=float(value.get("template_radius_nsigma", 5.0)),
    )


def source_template(cube_shape, wavelengths_A, source, *, psf_model=None):
    """Return the full-cube flux-density template for a source."""

    src = _source_from_mapping(source)
    if len(cube_shape) != 3:
        raise ValueError("cube_shape must be (nz, ny, nx).")
    nz, ny, nx = map(int, cube_shape)
    wave = np.asarray(wavelengths_A, dtype=np.float64)
    if wave.shape != (nz,):
        raise ValueError(f"wavelengths_A shape {wave.shape} does not match nz={nz}.")
    spectral = gaussian_line_profile(wave, src.line_center_A, src.line_fwhm_A)
    spatial = normalized_spatial_psf(
        (ny, nx),
        src.y,
        src.x,
        wavelength_A=src.line_center_A,
        psf_model=src.psf_model or psf_model,
        psf_image=src.psf_image,
        fwhm_scale=src.psf_fwhm_scale,
    )
    line = float(src.total_line_flux) * spectral[:, None, None] * spatial[None, :, :]
    if src.continuum_flux_density:
        line = line + float(src.continuum_flux_density) * spatial[None, :, :]
    return line.astype(np.float64)


def inject(cube, catalog, *, wavelengths_A, psf_model=None, copy=True):
    """Inject catalog sources into a 3D cube or 4D cube stack.

    The cube values are assumed to be flux density per wavelength channel. Each
    source line profile is normalized so that summing the injected delta over
    spatial pixels and wavelength channel widths recovers ``total_line_flux``.
    """

    arr = np.array(cube, dtype=np.float64, copy=bool(copy))
    if arr.ndim not in {3, 4}:
        raise ValueError(f"Expected cube with 3 or 4 dimensions, got {arr.shape}.")
    wave = np.asarray(wavelengths_A, dtype=np.float64)
    base_shape = arr.shape if arr.ndim == 3 else arr.shape[1:]
    for item in catalog:
        src = _source_from_mapping(item)
        template = source_template(base_shape, wave, src, psf_model=psf_model)
        if arr.ndim == 3:
            arr += template
        else:
            if src.cube_index is None:
                arr += template[None, :, :, :]
            else:
                arr[int(src.cube_index)] += template
    return arr


def integrated_delta_flux(delta_cube, wavelengths_A):
    """Return integrated flux over all spatial pixels and wavelengths."""

    delta = np.asarray(delta_cube, dtype=np.float64)
    wave = np.asarray(wavelengths_A, dtype=np.float64)
    widths = channel_widths(wave)
    if delta.ndim == 4:
        return float(np.nansum(delta * widths[None, :, None, None]))
    if delta.ndim == 3:
        return float(np.nansum(delta * widths[:, None, None]))
    raise ValueError("delta_cube must be 3D or 4D.")


def file_sha256(path, *, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path):
    """Hash file names and contents under a directory deterministically."""

    root = Path(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(file_sha256(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def create_run_clone(base_run_dir, clone_run_dir, *, dirs=("config", "stages", "tables"), extra_files=()):
    """Copy a run into a fresh clone directory without modifying the base run."""

    base = Path(base_run_dir)
    clone = Path(clone_run_dir)
    if not base.exists():
        raise FileNotFoundError(base)
    if clone.exists():
        raise FileExistsError(clone)
    clone.mkdir(parents=True)
    for name in dirs:
        src = base / name
        if src.exists():
            shutil.copytree(src, clone / name)
    for name in extra_files:
        src = base / name
        if src.exists():
            dst = clone / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return clone


__all__ = [
    "InjectionSource",
    "channel_widths",
    "create_run_clone",
    "file_sha256",
    "gaussian_line_profile",
    "inject",
    "integrated_delta_flux",
    "normalized_spatial_psf",
    "source_template",
    "tree_sha256",
]
