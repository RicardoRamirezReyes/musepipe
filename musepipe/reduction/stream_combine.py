"""Streaming exposure combine for per-exposure MUSE cubes.

``muse_exp_combine`` resamples every PIXTABLE_REDUCED at once, which needs far
more memory than a workstation has for 30 NFM exposures. This module combines
the already-resampled per-exposure cubes instead, one wavelength chunk at a
time, so peak memory stays bounded by the chunk size instead of by the number
of exposures.

The alignment convention follows ``stages/stage01_align``: the primary star is
located on a white-light image, the cube is cropped around the rounded centroid
so the star lands on pixel index ``crop_npix // 2``, and the sub-pixel residual
is removed with a cubic shift for DATA and a squared-bilinear kernel for STAT.
Cubes are combined as a weighted mean, so STAT stays an exact variance of the
mean under the usual assumption that exposures are independent.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import spline_filter1d

from ..stages.stage01_align import (
    apply_wavelength_indices,
    build_wavelength_axis,
    find_centroid_maoppy_moffat,
    find_centroid_peak,
    make_white_light_image,
    sanitize_wavelength_indices,
)


class StreamCombineError(RuntimeError):
    """Raised when the per-exposure cubes cannot be combined safely."""


# The combined cube is a working product, not the final science crop: B1
# (stage01) re-centres and re-crops it to crop_npix and needs margin, so this
# is deliberately wider than the 170 px science crop.
DEFAULT_CROP_NPIX = 200
DEFAULT_PAD = 12
DEFAULT_CHUNK_CHANNELS = 128
DEFAULT_COARSE_STRIDE = 25
DEFAULT_SIGCLIP_K = 3.0
DEFAULT_SIGCLIP_MIN_N = 5
MAX_CRVAL3_SPREAD_CHANNELS = 0.05
CD_MATRIX_RTOL = 1e-9


@dataclass(frozen=True)
class ExposureAlignment:
    """Per-exposure geometry needed to place one cube on the output grid."""

    index: int
    file: str
    exposure_id: str
    shape: tuple[int, int, int]
    y_center: float
    x_center: float
    centroid_fallback: bool
    window: tuple[int, int, int, int]
    shift_y: float
    shift_x: float
    exptime: float
    weight: float
    ra_deg: float
    dec_deg: float
    crval3: float
    in_bounds: bool
    mjd_obs: float = float("nan")

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["shape"] = [int(v) for v in self.shape]
        payload["window"] = [int(v) for v in self.window]
        return payload


@dataclass(frozen=True)
class StreamCombinePlan:
    """Validated recipe for one streaming combine."""

    run_id: str
    target_name: str
    output: str
    crop_npix: int
    pad: int
    data_ext: int
    stat_ext: str
    chunk_channels: int
    method: str
    sigclip_k: float
    sigclip_min_n: int
    weight_mode: str
    drop_wave_min_A: float
    drop_wave_max_A: float
    centering_method: str
    reference: dict
    wavelength: dict
    exposures: tuple[ExposureAlignment, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    #: {ruta del cubo de la exposición: ruta a un .npy con T(λ) sobre el eje de
    #: salida}. La clave es el **fichero** y no `exposure_id` porque ese ID no es
    #: único: sale del **directorio padre** (los cubos por exposición se llaman
    #: todos `DATACUBE_FINAL.fits`), así que dos exposiciones que compartan carpeta
    #: comparten ID — y la T acabaría en la exposición equivocada sin que nada
    #: fallara. Es la vía
    #: «molecfit por exposición» de A3: cada exposición se corrige **con la suya**
    #: antes de entrar al combinado, que es lo único que distingue esa granularidad
    #: de aplicar una T única al cubo ya combinado.
    #:
    #: Va en el plan y no en un gancho a propósito. El hook `transform` existe,
    #: pero devuelve **solo DATA y deliberadamente no toca STAT** —correcto para
    #: restar un modelo determinista, falso para dividir por T, que exige
    #: `STAT/T²`—, así que usarlo dejaría el error sin escalar en todo el cubo.
    #: Y estando en el plan es procedencia: queda escrito qué T entró en cada
    #: exposición.
    transmission_by_exposure: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "target_name": self.target_name,
            "output": self.output,
            "crop_npix": int(self.crop_npix),
            "pad": int(self.pad),
            "data_ext": int(self.data_ext),
            "stat_ext": str(self.stat_ext),
            "chunk_channels": int(self.chunk_channels),
            "method": self.method,
            "sigclip_k": float(self.sigclip_k),
            "sigclip_min_n": int(self.sigclip_min_n),
            "weight_mode": self.weight_mode,
            "drop_wave_min_A": float(self.drop_wave_min_A),
            "drop_wave_max_A": float(self.drop_wave_max_A),
            "centering_method": self.centering_method,
            "reference": self.reference,
            "wavelength": self.wavelength,
            "n_exposures": len(self.exposures),
            "exposures": [exp.as_dict() for exp in self.exposures],
            "warnings": list(self.warnings),
            "transmission_by_exposure": dict(self.transmission_by_exposure),
        }


# --------------------------------------------------------------------------
# centroids
# --------------------------------------------------------------------------


def measure_primary_center(
    path,
    *,
    data_ext: int = 1,
    drop_wave_min_A: float = 5780.0,
    drop_wave_max_A: float = 6050.0,
    coarse_stride: int = DEFAULT_COARSE_STRIDE,
    stamp_half_size: int = 10,
    maoppy_max_nfev: int = 120,
    centering_method: str = "maoppy",
) -> dict:
    """Locate the brightest source without loading a full cube into memory.

    A strided white-light image over the full field gives an integer peak, then
    the centroid is refined on a small stamp that uses every channel.

    ``centering_method="maoppy"`` inherits ``stage01_align``'s behaviour, which
    silently falls back to the flux-weighted box centroid when the Moffat fit
    cannot run. The plan records every fallback so the provenance stays honest.
    """

    with fits.open(path, memmap=True) as hdul:
        data = hdul[data_ext].data
        nz, ny, nx = data.shape
        wave = build_wavelength_axis(hdul[data_ext].header, hdul[0].header, nz)
        channel_index, wave = sanitize_wavelength_indices(wave, nz)

        # Keep at least ~20 channels in the coarse pass so short cubes still work.
        stride = max(1, min(int(coarse_stride), max(1, wave.size // 20)))
        coarse_cube = np.asarray(data[channel_index[::stride]], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            coarse_white = make_white_light_image(
                coarse_cube, wave[::stride], drop_wave_min_A, drop_wave_max_A
            )
        del coarse_cube
        y_coarse, x_coarse = find_centroid_peak(coarse_white, box_half_size=4)

        stamp_reach = max(2 * int(stamp_half_size), 12)
        yi = int(np.round(y_coarse))
        xi = int(np.round(x_coarse))
        sy1 = max(0, yi - stamp_reach)
        sy2 = min(ny, yi + stamp_reach + 1)
        sx1 = max(0, xi - stamp_reach)
        sx2 = min(nx, xi + stamp_reach + 1)
        stamp_cube = apply_wavelength_indices(data[:, sy1:sy2, sx1:sx2], channel_index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            white = make_white_light_image(stamp_cube, wave, drop_wave_min_A, drop_wave_max_A)
        del stamp_cube

        y_local, x_local = find_centroid_peak(white, box_half_size=4)
        if centering_method == "peak":
            y_fit, x_fit, fallback = y_local, x_local, False
        elif centering_method == "maoppy":
            y_fit, x_fit, fallback = find_centroid_maoppy_moffat(
                white,
                y_init=y_local,
                x_init=x_local,
                stamp_half_size=int(stamp_half_size),
                max_nfev=int(maoppy_max_nfev),
            )
        else:
            raise ValueError(f"Unknown centering_method: {centering_method}")

        y_center = float(y_fit + sy1)
        x_center = float(x_fit + sx1)
        celestial = WCS(hdul[data_ext].header).celestial
        ra_deg, dec_deg = celestial.all_pix2world([[x_center, y_center]], 0)[0]
        header_cube = hdul[data_ext].header
        primary_header = hdul[0].header
        return {
            "file": str(path),
            "shape": (int(nz), int(ny), int(nx)),
            "y_center": y_center,
            "x_center": x_center,
            "y_coarse": float(y_coarse),
            "x_coarse": float(x_coarse),
            "centroid_fallback": bool(fallback),
            "ra_deg": float(ra_deg),
            "dec_deg": float(dec_deg),
            "peak_value": float(np.nanmax(white)),
            "exptime": float(primary_header.get("EXPTIME", np.nan)),
            "mjd_obs": float(primary_header.get("MJD-OBS", np.nan)),
            "crval3": float(header_cube["CRVAL3"]),
            "cd3_3": float(header_cube.get("CD3_3", header_cube.get("CDELT3"))),
            "crpix3": float(header_cube.get("CRPIX3", 1.0)),
            "ctype3": str(header_cube.get("CTYPE3", "")),
            "cd_matrix": [
                float(header_cube["CD1_1"]),
                float(header_cube.get("CD1_2", 0.0)),
                float(header_cube.get("CD2_1", 0.0)),
                float(header_cube["CD2_2"]),
            ],
            "wave_min_A": float(wave[0]),
            "wave_max_A": float(wave[-1]),
            "n_wave": int(wave.size),
        }


# --------------------------------------------------------------------------
# chunk-safe spatial shifts
# --------------------------------------------------------------------------


def shift_data_chunk(chunk_zyx: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """Cubic sub-pixel shift applied plane by plane over a wavelength chunk.

    ``stage01_align.apply_spatial_alignment`` shifts a whole cube at once, which
    lets ``scipy.ndimage.shift`` spline-filter the wavelength axis as well. That
    filter is recursive, so applying it to a chunk would make the result depend
    on where the chunk starts. Pre-filtering only the two spatial axes and then
    interpolating each plane on its own reproduces the whole-cube result to
    float32 precision while staying chunk-invariant.
    """

    chunk = np.asarray(chunk_zyx, dtype=np.float32)
    if chunk.ndim != 3:
        raise ValueError("expected a (nz, ny, nx) chunk")
    if shift_y == 0.0 and shift_x == 0.0:
        return chunk.copy()
    valid = np.isfinite(chunk)
    filled = np.nan_to_num(chunk, nan=0.0)
    spline = spline_filter1d(filled, order=3, axis=1, output=np.float32)
    spline = spline_filter1d(spline, order=3, axis=2, output=np.float32)
    shifted = np.empty_like(spline)
    for z in range(spline.shape[0]):
        shifted[z] = ndi_shift(
            spline[z],
            shift=(shift_y, shift_x),
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    valid_shifted = ndi_shift(
        valid.astype(np.float32),
        shift=(0.0, shift_y, shift_x),
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out = shifted / np.maximum(valid_shifted, 1e-6)
    out[valid_shifted < 0.5] = np.nan
    return out.astype(np.float32)


def shift_variance_chunk(chunk_zyx: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """Propagate variance through a sub-pixel shift with squared bilinear weights.

    Same kernel as ``stage01_align.bilinear_shift_variance``, vectorised over
    wavelength because the interpolation weights do not depend on the channel.
    """

    arr = np.nan_to_num(np.asarray(chunk_zyx, dtype=np.float64), nan=0.0)
    if arr.ndim != 3:
        raise ValueError("expected a (nz, ny, nx) chunk")
    nz, ny, nx = arr.shape
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    src_y = yy - float(shift_y)
    src_x = xx - float(shift_x)
    y0 = np.floor(src_y).astype(int)
    x0 = np.floor(src_x).astype(int)
    fy = src_y - y0
    fx = src_x - x0
    out = np.zeros_like(arr)
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            yi = y0 + dy
            xi = x0 + dx
            valid = (yi >= 0) & (yi < ny) & (xi >= 0) & (xi < nx)
            weight2 = (wy * wx) ** 2
            out[:, valid] += weight2[valid] * arr[:, yi[valid], xi[valid]]
    out[out == 0] = np.nan
    return out


def read_window(hdu_data, z1: int, z2: int, window: Sequence[int]) -> np.ndarray:
    """Read one wavelength chunk of a spatial window, NaN-padding outside the cube."""

    wy1, wy2, wx1, wx2 = (int(v) for v in window)
    _, ny, nx = hdu_data.shape
    out = np.full((int(z2) - int(z1), wy2 - wy1, wx2 - wx1), np.nan, dtype=np.float32)
    sy1, sy2 = max(0, wy1), min(ny, wy2)
    sx1, sx2 = max(0, wx1), min(nx, wx2)
    if sy1 >= sy2 or sx1 >= sx2:
        return out
    out[:, sy1 - wy1 : sy2 - wy1, sx1 - wx1 : sx2 - wx1] = np.asarray(
        hdu_data[int(z1) : int(z2), sy1:sy2, sx1:sx2], dtype=np.float32
    )
    return out


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


def _exposure_id(path) -> str:
    return Path(path).parent.name or Path(path).stem


def exposure_from_measurement(
    measurement: dict, *, index: int, weight: float, crop_npix: int, pad: int
) -> ExposureAlignment:
    """Geometría de una exposición a partir de su centroide medido.

    Extraído de ``build_stream_combine_plan`` sin cambiar una línea de la
    aritmética: la ventana se toma alrededor del píxel REDONDEADO y el resto
    subpíxel viaja en ``shift``, que es lo que ``_aligned_chunk`` aplica luego.
    Vive aquí, y no en el constructor del plan, porque ``musepipe.observations``
    tiene que rehacer esta misma geometría cuando un cubo se re-resuelve y hay
    que volver a medirlo — y hacerlo con una copia de la fórmula sería tener
    dos convenciones de alineado con un solo nombre.
    """

    half = int(crop_npix) // 2
    reach = half + int(pad)
    y_center = float(measurement["y_center"])
    x_center = float(measurement["x_center"])
    yi = int(np.round(y_center))
    xi = int(np.round(x_center))
    window = (yi - reach, yi - reach + int(crop_npix) + 2 * int(pad),
              xi - reach, xi - reach + int(crop_npix) + 2 * int(pad))
    _, ny, nx = measurement["shape"]
    in_bounds = window[0] >= 0 and window[2] >= 0 and window[1] <= ny and window[3] <= nx
    return ExposureAlignment(
        index=int(index),
        file=measurement["file"],
        exposure_id=_exposure_id(measurement["file"]),
        shape=tuple(measurement["shape"]),
        y_center=y_center,
        x_center=x_center,
        centroid_fallback=bool(measurement["centroid_fallback"]),
        window=window,
        shift_y=float(yi - y_center),
        shift_x=float(xi - x_center),
        exptime=measurement["exptime"],
        weight=float(weight),
        ra_deg=measurement["ra_deg"],
        dec_deg=measurement["dec_deg"],
        crval3=measurement["crval3"],
        in_bounds=bool(in_bounds),
        mjd_obs=measurement["mjd_obs"],
    )


def _weights(exptimes: Sequence[float], weight_mode: str) -> list[float]:
    if weight_mode == "none":
        return [1.0] * len(exptimes)
    if weight_mode == "exptime":
        values = np.asarray(exptimes, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise StreamCombineError("exptime weighting needs a positive EXPTIME in every cube")
        return [float(v) for v in values]
    raise ValueError(f"Unknown weight_mode: {weight_mode}")


def _alignment_repeatability(
    exposures: Sequence[ExposureAlignment], *, max_gap_days: float = 0.5
) -> dict:
    """Estimate centroid precision from exposures that repeat a dither position.

    Exposures taken back-to-back at the same nominal offset should give the same
    centroid, so their scatter bounds the alignment error that reaches the
    combined cube. It is an upper bound: real pointing jitter is included.

    Different nights repeat the same dither pattern and can land on the same
    rounded pixel while sitting elsewhere at the sub-pixel level, so a group is
    only kept together while consecutive exposures stay within ``max_gap_days``.
    """

    groups: dict[tuple, list[ExposureAlignment]] = {}
    for exposure in exposures:
        key = (exposure.shape, int(round(exposure.y_center)), int(round(exposure.x_center)))
        groups.setdefault(key, []).append(exposure)
    repeated = []
    for members in groups.values():
        ordered = sorted(members, key=lambda m: m.mjd_obs)
        run = [ordered[0]]
        for exposure in ordered[1:]:
            gap = abs(exposure.mjd_obs - run[-1].mjd_obs)
            if not np.isfinite(gap) or gap <= max_gap_days:
                run.append(exposure)
            else:
                repeated.append(run)
                run = [exposure]
        repeated.append(run)
    repeated = [run for run in repeated if len(run) > 1]
    deviations = []
    for members in repeated:
        mean_y = float(np.mean([m.y_center for m in members]))
        mean_x = float(np.mean([m.x_center for m in members]))
        deviations.extend((m.y_center - mean_y, m.x_center - mean_x) for m in members)
    if not deviations:
        return {"n_groups": 0, "n_exposures": 0, "centroid_repeatability_px": None}
    array = np.asarray(deviations, dtype=float)
    return {
        "n_groups": len(repeated),
        "n_exposures": int(array.shape[0]),
        "centroid_repeatability_px": float(np.sqrt((array**2).sum(axis=1).mean())),
        "centroid_repeatability_y_px": float(np.sqrt((array[:, 0] ** 2).mean())),
        "centroid_repeatability_x_px": float(np.sqrt((array[:, 1] ** 2).mean())),
    }


def _astrometry_groups(exposures: Sequence[ExposureAlignment], cos_dec: float) -> dict:
    """Cluster the WCS-implied primary positions to expose pointing systematics."""

    ra = np.asarray([exp.ra_deg for exp in exposures], dtype=float)
    dec = np.asarray([exp.dec_deg for exp in exposures], dtype=float)
    ra_arcsec = (ra - np.median(ra)) * 3600.0 * cos_dec
    dec_arcsec = (dec - np.median(dec)) * 3600.0
    order = np.argsort(dec_arcsec)
    clusters: list[list[int]] = [[int(order[0])]]
    for index in order[1:]:
        if dec_arcsec[index] - dec_arcsec[clusters[-1][-1]] > 1.0:
            clusters.append([int(index)])
        else:
            clusters[-1].append(int(index))
    summary = [
        {
            "n_exposures": len(members),
            "median_dra_arcsec": float(np.median(ra_arcsec[members])),
            "median_ddec_arcsec": float(np.median(dec_arcsec[members])),
            "exposure_ids": [exposures[i].exposure_id for i in sorted(members)],
        }
        for members in clusters
    ]
    centres = np.asarray([[g["median_dra_arcsec"], g["median_ddec_arcsec"]] for g in summary])
    spread = 0.0
    if len(centres) > 1:
        spread = float(
            max(np.hypot(*(centres[i] - centres[j]))
                for i in range(len(centres)) for j in range(i + 1, len(centres)))
        )
    return {
        "n_groups": len(summary),
        "max_group_offset_arcsec": spread,
        "groups": summary,
    }


def build_stream_combine_plan(
    cube_files: Sequence[str],
    *,
    run_id: str,
    output: str,
    target_name: str | None = None,
    crop_npix: int = DEFAULT_CROP_NPIX,
    pad: int = DEFAULT_PAD,
    data_ext: int = 1,
    stat_ext: str = "STAT",
    chunk_channels: int = DEFAULT_CHUNK_CHANNELS,
    method: str = "mean",
    sigclip_k: float = DEFAULT_SIGCLIP_K,
    sigclip_min_n: int = DEFAULT_SIGCLIP_MIN_N,
    weight_mode: str = "exptime",
    drop_wave_min_A: float = 5780.0,
    drop_wave_max_A: float = 6050.0,
    centering_method: str = "maoppy",
    coarse_stride: int = DEFAULT_COARSE_STRIDE,
    progress=None,
) -> StreamCombinePlan:
    """Measure every centroid and validate that the cubes share one grid."""

    files = [str(f) for f in cube_files]
    if len(files) < 2:
        raise StreamCombineError("need at least two cubes to combine")
    if method not in {"mean", "sigclip"}:
        raise StreamCombineError(f"unknown method: {method}")
    if int(crop_npix) <= 0:
        raise StreamCombineError("crop_npix must be positive")

    measurements = []
    for index, path in enumerate(files):
        if progress is not None:
            progress(index, len(files), path)
        measurements.append(
            measure_primary_center(
                path,
                data_ext=data_ext,
                drop_wave_min_A=drop_wave_min_A,
                drop_wave_max_A=drop_wave_max_A,
                coarse_stride=coarse_stride,
                centering_method=centering_method,
            )
        )

    n_wave = {m["n_wave"] for m in measurements}
    if len(n_wave) != 1:
        raise StreamCombineError(f"cubes have different channel counts: {sorted(n_wave)}")
    for key in ("cd3_3", "crpix3", "ctype3"):
        values = {m[key] for m in measurements}
        if len(values) != 1:
            raise StreamCombineError(f"cubes disagree on {key}: {sorted(values)}")
    cd_matrices = {tuple(m["cd_matrix"]) for m in measurements}
    reference_cd = np.asarray(measurements[0]["cd_matrix"], dtype=float)
    for candidate in cd_matrices:
        if not np.allclose(np.asarray(candidate, dtype=float), reference_cd, rtol=CD_MATRIX_RTOL, atol=0.0):
            raise StreamCombineError(
                "cubes are not on a common sky orientation; translation-only alignment is invalid"
            )

    cd3_3 = float(measurements[0]["cd3_3"])
    crval3_values = np.asarray([m["crval3"] for m in measurements], dtype=float)
    crval3_spread_channels = float((crval3_values.max() - crval3_values.min()) / abs(cd3_3))
    warnings: list[str] = []
    if crval3_spread_channels > MAX_CRVAL3_SPREAD_CHANNELS:
        raise StreamCombineError(
            f"CRVAL3 spread is {crval3_spread_channels:.4f} channels, above the "
            f"{MAX_CRVAL3_SPREAD_CHANNELS} channel tolerance; spectral resampling is required"
        )
    if crval3_spread_channels > 0:
        warnings.append(
            f"CRVAL3 differs by up to {crval3_spread_channels:.4f} channels "
            f"({(crval3_values.max() - crval3_values.min()):.5f} A); combined without spectral resampling"
        )

    half = int(crop_npix) // 2
    weights = _weights([m["exptime"] for m in measurements], weight_mode)

    exposures = []
    for index, (measurement, weight) in enumerate(zip(measurements, weights)):
        exposure = exposure_from_measurement(
            measurement, index=index, weight=weight, crop_npix=crop_npix, pad=pad
        )
        if not exposure.in_bounds:
            warnings.append(
                f"{exposure.exposure_id}: crop window falls off the cube; "
                "missing pixels are NaN-filled"
            )
        exposures.append(exposure)

    if any(exp.centroid_fallback for exp in exposures):
        failed = [exp.exposure_id for exp in exposures if exp.centroid_fallback]
        warnings.append(
            f"Moffat centroid fell back to the flux peak for {len(failed)}/{len(exposures)} "
            f"exposures: {', '.join(failed)}"
        )

    repeatability = _alignment_repeatability(exposures)
    if repeatability["n_groups"] > 0 and repeatability["centroid_repeatability_px"] > 0.2:
        warnings.append(
            f"centroid repeatability is {repeatability['centroid_repeatability_px']:.3f} px "
            "within repeated dither positions; the combined PSF will be smeared"
        )

    ra = np.asarray([exp.ra_deg for exp in exposures], dtype=float)
    dec = np.asarray([exp.dec_deg for exp in exposures], dtype=float)
    cos_dec = float(np.cos(np.radians(np.median(dec))))
    pixel_scale_deg = float(np.hypot(reference_cd[0], reference_cd[2]))
    astrometry = _astrometry_groups(exposures, cos_dec)
    if astrometry["max_group_offset_arcsec"] > 1.0:
        warnings.append(
            f"per-exposure WCS places the primary up to "
            f"{astrometry['max_group_offset_arcsec']:.2f}\" apart between "
            f"{astrometry['n_groups']} pointing groups; alignment is by centroid so this only "
            "affects the absolute CRVAL written to the combined cube"
        )
    reference = {
        "ra_deg": float(np.median(ra)),
        "dec_deg": float(np.median(dec)),
        "ra_scatter_arcsec": float(np.std(ra) * cos_dec * 3600.0),
        "dec_scatter_arcsec": float(np.std(dec) * 3600.0),
        "pixel_scale_arcsec": float(pixel_scale_deg * 3600.0),
        "center_pixel_yx": [int(half), int(half)],
        "cd_matrix": [float(v) for v in reference_cd],
        "wcs_source": "median of the per-exposure primary positions",
        "astrometry_groups": astrometry,
        "alignment_repeatability": repeatability,
    }
    wavelength = {
        "n_channels": int(measurements[0]["n_wave"]),
        "crval3": float(np.median(crval3_values)),
        "crval3_source_index": int(np.argmin(np.abs(crval3_values - np.median(crval3_values)))),
        "cd3_3": cd3_3,
        "crpix3": float(measurements[0]["crpix3"]),
        "ctype3": str(measurements[0]["ctype3"]),
        "crval3_spread_channels": crval3_spread_channels,
        "wave_min_A": float(measurements[0]["wave_min_A"]),
        "wave_max_A": float(measurements[0]["wave_max_A"]),
    }
    return StreamCombinePlan(
        run_id=str(run_id),
        target_name=str(target_name or run_id),
        output=str(output),
        crop_npix=int(crop_npix),
        pad=int(pad),
        data_ext=int(data_ext),
        stat_ext=str(stat_ext),
        chunk_channels=int(chunk_channels),
        method=str(method),
        sigclip_k=float(sigclip_k),
        sigclip_min_n=int(sigclip_min_n),
        weight_mode=str(weight_mode),
        drop_wave_min_A=float(drop_wave_min_A),
        drop_wave_max_A=float(drop_wave_max_A),
        centering_method=str(centering_method),
        reference=reference,
        wavelength=wavelength,
        exposures=tuple(exposures),
        warnings=tuple(warnings),
    )


def plan_from_dict(payload: dict) -> StreamCombinePlan:
    """Rebuild a plan from its JSON form."""

    exposures = tuple(
        ExposureAlignment(
            index=int(item["index"]),
            file=str(item["file"]),
            exposure_id=str(item["exposure_id"]),
            shape=tuple(int(v) for v in item["shape"]),
            y_center=float(item["y_center"]),
            x_center=float(item["x_center"]),
            centroid_fallback=bool(item["centroid_fallback"]),
            window=tuple(int(v) for v in item["window"]),
            shift_y=float(item["shift_y"]),
            shift_x=float(item["shift_x"]),
            exptime=float(item["exptime"]),
            weight=float(item["weight"]),
            ra_deg=float(item["ra_deg"]),
            dec_deg=float(item["dec_deg"]),
            crval3=float(item["crval3"]),
            in_bounds=bool(item["in_bounds"]),
            mjd_obs=float(item.get("mjd_obs", float("nan"))),
        )
        for item in payload["exposures"]
    )
    return StreamCombinePlan(
        run_id=str(payload["run_id"]),
        target_name=str(payload["target_name"]),
        output=str(payload["output"]),
        crop_npix=int(payload["crop_npix"]),
        pad=int(payload["pad"]),
        data_ext=int(payload["data_ext"]),
        stat_ext=str(payload["stat_ext"]),
        chunk_channels=int(payload["chunk_channels"]),
        method=str(payload["method"]),
        sigclip_k=float(payload["sigclip_k"]),
        sigclip_min_n=int(payload["sigclip_min_n"]),
        weight_mode=str(payload["weight_mode"]),
        drop_wave_min_A=float(payload["drop_wave_min_A"]),
        drop_wave_max_A=float(payload["drop_wave_max_A"]),
        centering_method=str(payload["centering_method"]),
        reference=dict(payload["reference"]),
        wavelength=dict(payload["wavelength"]),
        exposures=exposures,
        warnings=tuple(payload.get("warnings", [])),
        transmission_by_exposure=dict(payload.get("transmission_by_exposure", {})),
    )


# --------------------------------------------------------------------------
# execute
# --------------------------------------------------------------------------


def _aligned_chunk(exposure: ExposureAlignment, plan: StreamCombinePlan, hdul, z1: int, z2: int):
    """Return the cropped, aligned DATA and STAT chunk for one exposure."""

    pad = int(plan.pad)
    npix = int(plan.crop_npix)
    data_window = read_window(hdul[plan.data_ext].data, z1, z2, exposure.window)
    stat_window = read_window(hdul[plan.stat_ext].data, z1, z2, exposure.window)
    data_shifted = shift_data_chunk(data_window, exposure.shift_y, exposure.shift_x)
    stat_shifted = shift_variance_chunk(stat_window, exposure.shift_y, exposure.shift_x)
    data = data_shifted[:, pad : pad + npix, pad : pad + npix]
    stat = stat_shifted[:, pad : pad + npix, pad : pad + npix].astype(np.float32)
    valid = np.isfinite(data) & np.isfinite(stat) & (stat > 0)
    data = np.where(valid, data, np.nan)
    stat = np.where(valid, stat, np.nan)
    return data, stat, valid


def _sigclip_mask(stack: np.ndarray, variance: np.ndarray, k: float, min_n: int) -> np.ndarray:
    """Flag voxels further than ``k`` sigma from the per-voxel median.

    The scale is the larger of the robust scatter across exposures and the
    formal noise from STAT. Taking the maximum keeps the clip conservative and
    keeps it working where the MAD collapses to zero, which happens whenever at
    least half of the exposures share an identical value.
    """

    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
        mad = np.nanmedian(np.abs(stack - median), axis=0) * 1.4826
        formal = np.sqrt(np.nanmedian(variance, axis=0))
    scale = np.fmax(np.nan_to_num(mad, nan=0.0), np.nan_to_num(formal, nan=0.0))
    usable = (counts >= int(min_n)) & np.isfinite(scale) & (scale > 0)
    deviation = np.abs(stack - median)
    rejected = np.zeros(stack.shape, dtype=bool)
    np.greater(deviation, float(k) * scale, out=rejected, where=usable & finite)
    return rejected & finite


def wavelength_axis(plan: StreamCombinePlan) -> np.ndarray:
    """The output wavelength axis of a plan, from the header keywords it froze."""

    meta = plan.wavelength
    nz = int(meta["n_channels"])
    crpix = float(meta.get("crpix3", 1.0))
    return float(meta["crval3"]) + float(meta["cd3_3"]) * (np.arange(nz, dtype=np.float64) + 1.0 - crpix)


def _load_transmissions(plan: StreamCombinePlan, nz: int) -> dict:
    """{indice de exposicion: T(λ)} desde las rutas que declara el plan.

    Se carga UNA vez y se valida la longitud aquí: un desajuste de eje descubierto
    a mitad del combine deja un cubo a medias, y este combine dura horas.
    """

    if not plan.transmission_by_exposure:
        return {}
    por_indice: dict[int, np.ndarray] = {}
    for exposure in plan.exposures:
        ruta = (plan.transmission_by_exposure.get(exposure.file)
                or plan.transmission_by_exposure.get(str(Path(exposure.file).resolve())))
        if ruta is None:
            raise StreamCombineError(
                f"transmission_by_exposure is declared but {exposure.file} has none: "
                "correcting only some exposures would combine two different calibrations "
                "into one cube.")
        trans = np.asarray(np.load(str(ruta)), dtype=np.float64)
        if trans.shape != (nz,):
            raise StreamCombineError(
                f"transmission for {exposure.exposure_id} has {trans.shape} rows, "
                f"the output axis has {nz}.")
        if not np.all(np.isfinite(trans)) or np.nanmin(trans) <= 0:
            raise StreamCombineError(
                f"transmission for {exposure.exposure_id} is not positive and finite.")
        por_indice[int(exposure.index)] = trans
    return por_indice


def _apply_transmission_chunk(trans: np.ndarray, z1: int, z2: int, data, stat):
    """DATA/T y STAT/T^2, los dos en el mismo sitio.

    Que estén juntos no es estilo: es lo que hace imposible aplicar uno sin el
    otro. Dividir el dato por T sin escalar la varianza deja un cubo cuyo error
    miente justo donde más se corrigió.
    """

    escala = trans[z1:z2][:, None, None]
    return data / escala, stat / (escala ** 2)


def combine_streaming(plan: StreamCombinePlan, *, progress=None, transform=None) -> dict:
    """Combine the planned exposures one wavelength chunk at a time.

    ``transform(exposure, wave_chunk, data, stat) -> data`` runs on every
    exposure's chunk **after** it has been cropped and aligned and **before** it
    is accumulated. That is the hook C1b uses to subtract each exposure's own PSF
    model before combining — the whole point of modelling the PSF per
    observation, since the combined cube mixes exposures whose PSF differs.
    ``stat`` viaja con el dato porque el ajuste de amplitud pesa por varianza,
    igual que el de C3.

    STAT is deliberately NOT transformed: subtracting a deterministic model does
    not change the variance of the pixel. Without ``transform`` the behaviour is
    bit-for-bit the one that produced the cubes on disk.
    """

    npix = int(plan.crop_npix)
    nz = int(plan.wavelength["n_channels"])
    n_exp = len(plan.exposures)
    weights = np.asarray([exp.weight for exp in plan.exposures], dtype=np.float64)

    weighted_sum = np.zeros((nz, npix, npix), dtype=np.float64)
    weight_sum = np.zeros((nz, npix, npix), dtype=np.float64)
    variance_sum = np.zeros((nz, npix, npix), dtype=np.float64)
    count = np.zeros((nz, npix, npix), dtype=np.int16)
    rejected_total = 0
    contributed_total = 0

    chunk = max(1, int(plan.chunk_channels))
    transmissions = _load_transmissions(plan, nz)
    wave = wavelength_axis(plan) if (transform is not None or transmissions) else None
    handles = [fits.open(exp.file, memmap=True) for exp in plan.exposures]
    try:
        for z1 in range(0, nz, chunk):
            z2 = min(nz, z1 + chunk)
            if progress is not None:
                progress(z1, z2, nz)
            if plan.method == "mean":
                for exposure, hdul in zip(plan.exposures, handles):
                    data, stat, valid = _aligned_chunk(exposure, plan, hdul, z1, z2)
                    trans = transmissions.get(int(exposure.index))
                    if trans is not None:
                        data, stat = _apply_transmission_chunk(trans, z1, z2, data, stat)
                    if transform is not None:
                        data = transform(exposure, wave[z1:z2], data, stat)
                    w = float(exposure.weight)
                    weighted_sum[z1:z2] += np.where(valid, data * w, 0.0)
                    weight_sum[z1:z2] += np.where(valid, w, 0.0)
                    variance_sum[z1:z2] += np.where(valid, stat * (w * w), 0.0)
                    count[z1:z2] += valid
                    contributed_total += int(valid.sum())
                continue

            data_stack = np.empty((n_exp, z2 - z1, npix, npix), dtype=np.float32)
            stat_stack = np.empty_like(data_stack)
            for slot, (exposure, hdul) in enumerate(zip(plan.exposures, handles)):
                data, stat, _ = _aligned_chunk(exposure, plan, hdul, z1, z2)
                trans = transmissions.get(int(exposure.index))
                if trans is not None:
                    data, stat = _apply_transmission_chunk(trans, z1, z2, data, stat)
                if transform is not None:
                    # Los MISMOS cuatro argumentos que la rama `mean` y que el
                    # contrato del docstring. Esta rama pasaba tres, asi que
                    # cualquier gancho reventaba en cuanto el plan no combinaba
                    # por media — que es como combina ROXs 12 b (2026-08-19).
                    data = transform(exposure, wave[z1:z2], data, stat)
                data_stack[slot] = data
                stat_stack[slot] = stat
            rejected = _sigclip_mask(data_stack, stat_stack, plan.sigclip_k, plan.sigclip_min_n)
            keep = np.isfinite(data_stack) & np.isfinite(stat_stack) & ~rejected
            rejected_total += int(rejected.sum())
            contributed_total += int(keep.sum())
            del rejected
            # Accumulate one exposure at a time: a 4-D weighted temporary would
            # cost more than the stack itself.
            for slot in range(n_exp):
                kept = keep[slot]
                w = float(weights[slot])
                weighted_sum[z1:z2] += np.where(kept, data_stack[slot] * w, 0.0)
                weight_sum[z1:z2] += np.where(kept, w, 0.0)
                variance_sum[z1:z2] += np.where(kept, stat_stack[slot] * (w * w), 0.0)
            count[z1:z2] += keep.sum(axis=0).astype(np.int16)
            del data_stack, stat_stack, keep
    finally:
        for hdul in handles:
            hdul.close()

    empty = count == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        data = weighted_sum / weight_sum
        stat = variance_sum / (weight_sum * weight_sum)
    data[empty] = np.nan
    stat[empty] = np.nan

    qc = {
        "n_exposures": n_exp,
        "method": plan.method,
        "weight_mode": plan.weight_mode,
        "sigclip_k": float(plan.sigclip_k) if plan.method == "sigclip" else None,
        "cube_shape": [int(nz), int(npix), int(npix)],
        "voxels_total": int(nz) * int(npix) * int(npix),
        "contributions_total": int(contributed_total),
        "contributions_rejected": int(rejected_total),
        "rejected_fraction": (
            float(rejected_total) / float(rejected_total + contributed_total)
            if (rejected_total + contributed_total) > 0
            else 0.0
        ),
        "count_min": int(count.min()),
        "count_median": float(np.median(count)),
        "count_max": int(count.max()),
        "empty_voxel_fraction": float(np.mean(empty)),
        "finite_fraction": float(np.isfinite(data).mean()),
        "interp_kernel": {"data": "cubic_spline_per_plane", "stat": "bilinear_kernel_squared"},
        "transmission_by_exposure": {exp.file: str(plan.transmission_by_exposure[exp.file])
                                     for exp in plan.exposures
                                     if exp.file in plan.transmission_by_exposure},
    }
    return {
        "data": data.astype(np.float32),
        "stat": stat.astype(np.float32),
        "count": count,
        "qc": qc,
    }


def build_output_header(plan: StreamCombinePlan, *, bunit: str | None = None) -> fits.Header:
    """Build the WCS of the combined cube: the primary sits on the crop centre."""

    half = int(plan.crop_npix) // 2
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRPIX1"] = half + 1.0
    header["CRPIX2"] = half + 1.0
    header["CRVAL1"] = float(plan.reference["ra_deg"])
    header["CRVAL2"] = float(plan.reference["dec_deg"])
    cd = plan.reference["cd_matrix"]
    header["CD1_1"] = float(cd[0])
    header["CD1_2"] = float(cd[1])
    header["CD2_1"] = float(cd[2])
    header["CD2_2"] = float(cd[3])
    header["CTYPE3"] = str(plan.wavelength["ctype3"])
    header["CUNIT3"] = "Angstrom"
    header["CRPIX3"] = float(plan.wavelength["crpix3"])
    header["CRVAL3"] = float(plan.wavelength["crval3"])
    header["CD3_3"] = float(plan.wavelength["cd3_3"])
    header["CD1_3"] = 0.0
    header["CD2_3"] = 0.0
    header["CD3_1"] = 0.0
    header["CD3_2"] = 0.0
    if bunit:
        header["BUNIT"] = bunit
    return header


def write_combined_cube(result: dict, plan: StreamCombinePlan, output, *, bunit=None, stat_bunit=None):
    """Write DATA/STAT/NEXP with the same layout as a DRS DATACUBE_FINAL."""

    output = Path(output)
    if output.exists():
        raise StreamCombineError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    primary = fits.Header()
    primary["RUNID"] = str(plan.run_id)
    primary["TARGET"] = str(plan.target_name)
    primary["OBJECT"] = str(plan.target_name)
    primary["ORIGIN"] = "musepipe.stream_combine"
    primary["COMBMETH"] = str(plan.method)
    primary["COMBWGT"] = str(plan.weight_mode)
    primary["NEXP"] = len(plan.exposures)
    primary["CROPPIX"] = int(plan.crop_npix)
    primary["CENTER"] = str(plan.centering_method)
    primary["SPATIAL"] = "subpixel"
    primary["EXPTOT"] = float(sum(exp.exptime for exp in plan.exposures))
    if plan.method == "sigclip":
        primary["SIGCLIPK"] = float(plan.sigclip_k)
    primary.add_comment("Streaming replacement for muse_exp_combine (memory-bound cube stack).")
    for exposure in plan.exposures:
        primary.add_history(
            f"exp {exposure.index:02d} {exposure.exposure_id} "
            f"t={exposure.exptime:.0f}s shift=({exposure.shift_y:+.3f},{exposure.shift_x:+.3f})"
        )

    wcs_header = build_output_header(plan, bunit=bunit)
    data_hdu = fits.ImageHDU(data=result["data"], header=wcs_header.copy(), name="DATA")
    stat_header = build_output_header(plan, bunit=stat_bunit)
    stat_hdu = fits.ImageHDU(data=result["stat"], header=stat_header, name="STAT")
    count_hdu = fits.ImageHDU(data=result["count"], name="NEXP")
    count_hdu.header["BUNIT"] = "count"
    count_hdu.header.add_comment("Number of exposures contributing to each voxel.")
    fits.HDUList([fits.PrimaryHDU(header=primary), data_hdu, stat_hdu, count_hdu]).writeto(output)
    return output


__all__ = [
    "ExposureAlignment",
    "StreamCombineError",
    "StreamCombinePlan",
    "build_output_header",
    "build_stream_combine_plan",
    "combine_streaming",
    "exposure_from_measurement",
    "measure_primary_center",
    "plan_from_dict",
    "read_window",
    "shift_data_chunk",
    "shift_variance_chunk",
    "write_combined_cube",
]
