"""Stripe geometry, range, mask and spectral-shift primitives for Stage 02.

Extracted verbatim from ``02_xcorr_stripes.ipynb`` (the canonical, full-memory
cross-correlation stripe notebook) as the first increment of refactor task O2a.

Scope of this module: pure, side-effect-free helpers only — geometry/orientation
parsing, stripe-range construction and validation, mask building, and 1D
spectral-shift primitives. No cube IO, FITS access, plotting, or reference-frame
orchestration lives here; those remain in the notebooks pending later increments.

The low-memory variant (``02_low_memory_xcorr_stripes.ipynb``) carries byte-identical
copies of every function below, so this module is a safe common source for both.
The diagnostic/template notebooks (``02b``, ``02c``) hold drifted copies that
should be replaced by imports from here.
"""

import warnings

import numpy as np
from scipy.ndimage import shift as ndi_shift

from .stats import finite_percentile, median_finite, robust_sigma

__all__ = [
    "safe_float",
    "_optional_float",
    "_expand_per_cube",
    "_stripe_group_key",
    "_stripe_group_sort_value",
    "_parse_xcorr_group_keys",
    "_normalize_stripe_orientation",
    "_orientation_to_stripe_angle_deg",
    "_normalize_stripe_angle_deg",
    "_axis_from_stripe_angle",
    "_parse_stripe_geometry_config",
    "_range_validation_hint",
    "_validate_manual_stripe_ranges",
    "_validate_manual_range_list_for_stage02",
    "_build_stripe_ranges",
    "_build_stripe_ranges_flexible",
    "_axis_aligned_masks",
    "_stripe_coordinate_map",
    "_validate_projected_stripe_ranges",
    "_projected_masks_from_ranges",
    "_build_projected_stripe_geometry",
    "_build_stripe_geometry",
    "_normalize_for_xcorr",
    "_xcorr_shift_pixels",
    "_shift_spectral_nan_safe",
    "_shift_spectral_integer",
    "_normalize_columns_for_xcorr",
    "xcorr_shift_map",
    "stripe_channel_amplitudes",
    "stripe_metric_exclusion_mask",
    "stripe_metric_summary",
    "shift_spectral_variance",
    "apply_stripe_spectral_shifts",
]


def safe_float(x):
    try:
        val = float(x)
        return val if np.isfinite(val) else None
    except (ValueError, TypeError):
        return None


def _optional_float(value, default=None):
    if value is None:
        return default
    return float(value)


def _expand_per_cube(value, n_cubes, name, default=None):
    if value is None:
        return [default for _ in range(n_cubes)]
    if isinstance(value, str) or np.isscalar(value):
        return [value for _ in range(n_cubes)]

    out = list(value)
    if len(out) == n_cubes:
        return out
    if len(out) == 1:
        warnings.warn(f"{name} has length 1 but N={n_cubes}; expanding the single value.")
        return out * n_cubes

    raise ValueError(f"{name} must have length N={n_cubes}; got {len(out)}.")


def _stripe_group_key(angle_deg, ndigits=6):
    return f"angle_{round(float(angle_deg) % 180.0, ndigits):g}"


def _stripe_group_sort_value(key):
    text = str(key)
    if text.startswith("angle_"):
        try:
            return (0, float(text.replace("angle_", "")), 0.0)
        except Exception:
            return (0, np.inf, 0.0)
    if text.startswith("drot_"):
        parts = text.split("_")
        try:
            drot = float(parts[1])
        except Exception:
            drot = np.inf
        try:
            split = float(parts[3]) if len(parts) >= 4 and parts[2] == "split" else 0.0
        except Exception:
            split = 0.0
        return (1, drot, split)
    return (9, np.inf, text)


def _parse_xcorr_group_keys(stripe_angles, stripe_group_key=None, n_cubes=None):
    if n_cubes is None:
        n_cubes = len(stripe_angles)
    if stripe_group_key is None:
        return [_stripe_group_key(a) for a in stripe_angles]
    keys = _expand_per_cube(
        stripe_group_key,
        n_cubes,
        "xcorr_stripe_group_key",
        default=None,
    )
    if any(k is None for k in keys):
        raise ValueError("xcorr_stripe_group_key contains None entries.")
    return [str(k) for k in keys]


def _normalize_stripe_orientation(value: str) -> str:
    s = str(value).strip().lower()
    aliases = {
        "h": "horizontal",
        "horizontal": "horizontal",
        "y": "horizontal",
        "v": "vertical",
        "vertical": "vertical",
        "x": "vertical",
        "a": "angled",
        "angle": "angled",
        "angled": "angled",
        "diagonal": "angled",
        "oblique": "angled",
    }
    if s not in aliases:
        raise ValueError(f"Invalid stripe orientation: {value}")
    return aliases[s]


def _orientation_to_stripe_angle_deg(orientation: str) -> float:
    orientation = _normalize_stripe_orientation(orientation)
    if orientation == "horizontal":
        return 90.0
    if orientation == "vertical":
        return 0.0
    return 90.0


def _normalize_stripe_angle_deg(angle, orientation="vertical") -> float:
    if angle is None:
        return _orientation_to_stripe_angle_deg(orientation)
    try:
        angle = float(angle)
    except Exception:
        return _orientation_to_stripe_angle_deg(orientation)
    if not np.isfinite(angle):
        return _orientation_to_stripe_angle_deg(orientation)
    return angle % 180.0


def _axis_from_stripe_angle(angle_deg, tol=1e-6):
    angle = float(angle_deg) % 180.0
    if np.isclose(angle, 0.0, atol=tol) or np.isclose(angle, 180.0, atol=tol):
        return "x", "vertical"
    if np.isclose(angle, 90.0, atol=tol):
        return "y", "horizontal"
    return "angle", "angled"


def _parse_stripe_geometry_config(stripe_orientation, stripe_angle_deg, n_cubes):
    orientations = _expand_per_cube(
        stripe_orientation,
        n_cubes,
        "stripe_orientation",
        default="horizontal",
    )
    orientations = [_normalize_stripe_orientation(o) for o in orientations]

    angles_raw = _expand_per_cube(
        stripe_angle_deg,
        n_cubes,
        "stripe_angle_deg",
        default=None,
    )
    angles = [
        _normalize_stripe_angle_deg(a, orientation=o)
        for a, o in zip(angles_raw, orientations)
    ]

    return orientations, angles


def _range_validation_hint(length, nstripes):
    if length > nstripes and nstripes > 0 and length % nstripes == 0:
        return (
            " It looks like several per-cube templates were pasted into one "
            "flat list. Use xcorr_stripe_ranges_by_cube / XCORR_STRIPE_RANGES_BY_CUBE "
            "for one template per cube."
        )
    return ""


def _validate_manual_stripe_ranges(manual_ranges, ny: int, nx: int, orientation: str):
    orientation = _normalize_stripe_orientation(orientation)
    lim = ny if orientation == "horizontal" else nx

    out = []
    for pair in manual_ranges:
        if len(pair) != 2:
            raise ValueError(f"Invalid stripe range: {pair}")
        a1, a2 = int(pair[0]), int(pair[1])
        if not (0 <= a1 < a2 <= lim):
            raise ValueError(
                f"Invalid manual stripe range {pair} for orientation={orientation}, limit={lim}"
            )
        out.append((a1, a2))

    return out


def _validate_manual_range_list_for_stage02(manual_ranges, nstripes, label):
    if manual_ranges is None:
        return None
    if not isinstance(manual_ranges, (list, tuple)):
        raise TypeError(f"{label} must be None or a list of [start, stop] pairs.")
    if len(manual_ranges) != int(nstripes):
        raise ValueError(
            f"{label} has {len(manual_ranges)} ranges, but xcorr_nstripes={int(nstripes)}."
            + _range_validation_hint(len(manual_ranges), int(nstripes))
        )

    out = []
    prev_stop = None
    for s, pair in enumerate(manual_ranges):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{label} stripe {s} must be a [start, stop] pair; got {pair!r}")
        try:
            a1 = float(pair[0])
            a2 = float(pair[1])
        except Exception as exc:
            raise ValueError(f"{label} stripe {s} has non-numeric bounds: {pair!r}") from exc
        if not (np.isfinite(a1) and np.isfinite(a2)):
            raise ValueError(f"{label} stripe {s} has non-finite bounds: {pair!r}")
        if not (a1 < a2):
            raise ValueError(f"{label} stripe {s} must satisfy start < stop; got {pair!r}")
        if prev_stop is not None:
            if a1 < prev_stop:
                raise ValueError(
                    f"{label} stripe {s} starts at {a1:g}, before the previous stop {prev_stop:g}. "
                    "Overlapping ranges would apply multiple spectral shifts to the same pixels."
                )
            if a1 > prev_stop:
                warnings.warn(
                    f"{label} has a gap between {prev_stop:g} and {a1:g}. "
                    "That may be intentional, but pixels in the gap will not be assigned to a manual stripe."
                )
        prev_stop = a2

        if a1.is_integer() and a2.is_integer():
            out.append([int(a1), int(a2)])
        else:
            out.append([a1, a2])
    return out


def _build_stripe_ranges(ny: int, nx: int, nstripes: int, orientation: str):
    orientation = _normalize_stripe_orientation(orientation)

    if orientation == "horizontal":
        edges = np.linspace(0, ny, nstripes + 1, dtype=int)
        ranges = [(int(edges[i]), int(edges[i + 1])) for i in range(nstripes)]
        axis = "y"
    else:
        edges = np.linspace(0, nx, nstripes + 1, dtype=int)
        ranges = [(int(edges[i]), int(edges[i + 1])) for i in range(nstripes)]
        axis = "x"

    return ranges, axis


def _build_stripe_ranges_flexible(ny: int, nx: int, nstripes: int, orientation: str,
                                  manual_stripe_ranges=None):
    orientation = _normalize_stripe_orientation(orientation)

    if orientation == "angled":
        raise ValueError("Angled stripes require _build_stripe_geometry with stripe_angle_deg.")

    if manual_stripe_ranges is not None:
        stripe_ranges = _validate_manual_stripe_ranges(
            manual_stripe_ranges, ny, nx, orientation
        )
        axis = "y" if orientation == "horizontal" else "x"
        return stripe_ranges, axis

    return _build_stripe_ranges(ny, nx, nstripes, orientation)


def _axis_aligned_masks(ny, nx, stripe_ranges, axis):
    masks = []
    for a1, a2 in stripe_ranges:
        mask = np.zeros((ny, nx), dtype=bool)
        if axis == "y":
            mask[a1:a2, :] = True
        elif axis == "x":
            mask[:, a1:a2] = True
        else:
            raise ValueError(f"Invalid axis: {axis}")
        masks.append(mask)
    return masks


def _stripe_coordinate_map(ny, nx, stripe_angle_deg):
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    theta = np.deg2rad(float(stripe_angle_deg) % 180.0)
    coord = xx * np.cos(theta) + yy * np.sin(theta)
    coord -= np.nanmin(coord)
    return coord


def _validate_projected_stripe_ranges(manual_ranges, coord_span):
    out = []
    for pair in manual_ranges:
        if len(pair) != 2:
            raise ValueError(f"Invalid projected stripe range: {pair}")
        a1, a2 = float(pair[0]), float(pair[1])
        if not (0.0 <= a1 < a2 <= coord_span + 1e-6):
            raise ValueError(
                f"Invalid projected stripe range {pair}; projection span is {coord_span:.3f} pixels"
            )
        out.append((a1, a2))
    return out


def _projected_masks_from_ranges(coord, stripe_ranges):
    masks = []
    for s, (a1, a2) in enumerate(stripe_ranges):
        if s == len(stripe_ranges) - 1:
            mask = (coord >= a1) & (coord <= a2)
        else:
            mask = (coord >= a1) & (coord < a2)
        if np.count_nonzero(mask) == 0:
            warnings.warn(f"Projected stripe {s} contains no pixels.")
        masks.append(mask)
    return masks


def _build_projected_stripe_geometry(ny, nx, nstripes, stripe_angle_deg, manual_stripe_ranges=None):
    coord = _stripe_coordinate_map(ny, nx, stripe_angle_deg)
    coord_span = float(np.nanmax(coord))

    if manual_stripe_ranges is None:
        edges = np.linspace(0.0, coord_span, int(nstripes) + 1)
        stripe_ranges = [(float(edges[i]), float(edges[i + 1])) for i in range(int(nstripes))]
    else:
        stripe_ranges = _validate_projected_stripe_ranges(manual_stripe_ranges, coord_span)

    masks = _projected_masks_from_ranges(coord, stripe_ranges)
    return stripe_ranges, masks


def _build_stripe_geometry(ny: int, nx: int, nstripes: int, orientation: str,
                           manual_stripe_ranges=None, stripe_angle_deg=None):
    angle = _normalize_stripe_angle_deg(stripe_angle_deg, orientation)
    axis, orientation_from_angle = _axis_from_stripe_angle(angle)

    if axis in ("x", "y"):
        stripe_ranges, axis = _build_stripe_ranges_flexible(
            ny,
            nx,
            nstripes,
            orientation_from_angle,
            manual_stripe_ranges=manual_stripe_ranges,
        )
        masks = _axis_aligned_masks(ny, nx, stripe_ranges, axis)
    else:
        stripe_ranges, masks = _build_projected_stripe_geometry(
            ny,
            nx,
            nstripes,
            angle,
            manual_stripe_ranges=manual_stripe_ranges,
        )

    return stripe_ranges, axis, angle, masks


def _normalize_for_xcorr(spec: np.ndarray):
    s = np.asarray(spec, dtype=np.float64)
    good = np.isfinite(s)
    if np.count_nonzero(good) < 3:
        return None

    x = s[good]
    med = np.nanmedian(x)
    x = x - med
    std = np.nanstd(x)

    if not np.isfinite(std) or std <= 0:
        return None

    out = np.full_like(s, np.nan, dtype=np.float64)
    out[good] = (s[good] - med) / std
    return out


def _xcorr_shift_pixels(ref_spec: np.ndarray, tgt_spec: np.ndarray, max_lag: int = 6):
    """
    Returns dz in spectral channels.
    Positive dz means tgt appears shifted to larger channel index relative to ref.
    """
    r = _normalize_for_xcorr(ref_spec)
    t = _normalize_for_xcorr(tgt_spec)

    if r is None or t is None:
        return np.nan

    good = np.isfinite(r) & np.isfinite(t)
    if np.count_nonzero(good) < 10:
        return np.nan

    r = r[good]
    t = t[good]

    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    corr = np.full(lags.shape, np.nan, dtype=np.float64)

    for i, lag in enumerate(lags):
        if lag < 0:
            a = r[-lag:]
            b = t[:len(t) + lag]
        elif lag > 0:
            a = r[:-lag]
            b = t[lag:]
        else:
            a = r
            b = t

        if len(a) < 5:
            continue

        corr[i] = np.nansum(a * b)

    if not np.isfinite(corr).any():
        return np.nan

    i0 = int(np.nanargmax(corr))
    lag0 = float(lags[i0])

    # Optional subpixel refinement with 3-point parabola
    if 0 < i0 < len(corr) - 1:
        y1, y2, y3 = corr[i0 - 1], corr[i0], corr[i0 + 1]
        denom = (y1 - 2 * y2 + y3)
        if np.isfinite(denom) and np.abs(denom) > 1e-12:
            delta = 0.5 * (y1 - y3) / denom
            if np.abs(delta) <= 1.0:
                lag0 = lag0 + float(delta)

    return lag0


def _shift_spectral_nan_safe(block: np.ndarray, dz: float):
    """
    Spectral shift along the first axis using interpolation, NaN-safe.
    Works for 3D rectangular stripes and 2D angled-stripe pixel stacks.
    """
    data0 = np.nan_to_num(block, nan=0.0).astype(np.float32, copy=False)
    valid = np.isfinite(block).astype(np.float32)
    shift_vec = (float(dz),) + (0.0,) * (data0.ndim - 1)

    shifted = ndi_shift(
        data0,
        shift=shift_vec,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True
    )

    valid_s = ndi_shift(
        valid,
        shift=shift_vec,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        out = shifted / np.maximum(valid_s, 1e-6)

    out[valid_s < 0.5] = np.nan
    return out.astype(np.float32)


def _shift_spectral_integer(block: np.ndarray, dz: float):
    """
    Spectral shift by integer channels only, no interpolation.
    """
    k = int(np.round(dz))
    out = np.full_like(block, np.nan, dtype=np.float32)

    if k == 0:
        return block.astype(np.float32, copy=True)

    if k > 0:
        out[k:, ...] = block[:-k, ...]
    else:
        out[:k, ...] = block[-k:, ...]

    return out


def _normalize_columns_for_xcorr(flat):
    """Vectorized, column-wise equivalent of ``_normalize_for_xcorr``.

    ``flat`` has shape ``(nz, P)``. Returns ``(T, col_ok)`` where ``T`` is the
    normalized array (NaN wherever the input was non-finite, and entire columns
    set to NaN where ``_normalize_for_xcorr`` would have returned ``None``), and
    ``col_ok`` is a boolean mask of columns that normalized successfully.
    Each column reproduces ``_normalize_for_xcorr`` bit-for-bit.
    """
    s = np.asarray(flat, dtype=np.float64)
    good = np.isfinite(s)
    cnt = good.sum(axis=0)
    masked = np.where(good, s, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(masked, axis=0)
        std = np.nanstd(masked, axis=0)
        with np.errstate(all="ignore"):
            out = (s - med) / std
    out[~good] = np.nan
    col_ok = (cnt >= 3) & np.isfinite(std) & (std > 0)
    out[:, ~col_ok] = np.nan
    return out, col_ok


def xcorr_shift_map(cube_use, ref_spec=None, max_lag=6):
    """Per-spaxel spectral cross-correlation shift map (channels), vectorized.

    ``cube_use`` is ``(nz, ny, nx)`` already restricted to the xcorr channel
    window. ``ref_spec`` is an optional raw 1D reference (``nz``); if ``None``
    the robust median over all spaxels is used. Returns an ``(ny, nx)`` float32
    map of subpixel shifts.

    This reproduces the scalar loop
    ``_xcorr_shift_pixels(_normalize_for_xcorr(ref), _normalize_for_xcorr(spec))``
    exactly. Spaxels whose finite channels cover the reference support are solved
    with a fully vectorized correlation; any spaxel with extra masked channels
    (or that fails normalization) falls back to the scalar ``_xcorr_shift_pixels``
    so the result is identical to the per-spaxel implementation.
    """
    cube_use = np.asarray(cube_use)
    nz, ny, nx = cube_use.shape
    if ref_spec is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ref_spec = np.nanmedian(cube_use.reshape(nz, -1), axis=1)

    R = _normalize_for_xcorr(ref_spec)
    P = ny * nx
    out = np.full(P, np.nan, dtype=np.float64)
    if R is None:
        return out.reshape(ny, nx).astype(np.float32)

    flat = cube_use.reshape(nz, P)
    T, col_ok = _normalize_columns_for_xcorr(flat)
    ref_good = np.isfinite(R)
    L = int(ref_good.sum())

    fast = col_ok & np.isfinite(T[ref_good, :]).all(axis=0)

    if L >= 10 and fast.any():
        fidx = np.where(fast)[0]
        Rc = R[ref_good]
        Tc = T[np.ix_(ref_good, fidx)]
        lags = np.arange(-int(max_lag), int(max_lag) + 1, dtype=int)
        nl = lags.size
        Nf = fidx.size
        corr = np.full((nl, Nf), np.nan)
        for li, lag in enumerate(lags):
            if lag < 0:
                a = Rc[-lag:]
                B = Tc[:L + lag, :]
            elif lag > 0:
                a = Rc[:-lag]
                B = Tc[lag:, :]
            else:
                a = Rc
                B = Tc
            if a.shape[0] < 5:
                continue
            corr[li, :] = (a[:, None] * B).sum(axis=0)

        anyf = np.isfinite(corr).any(axis=0)
        i0 = np.zeros(Nf, dtype=int)
        if anyf.any():
            filled = np.where(np.isfinite(corr[:, anyf]), corr[:, anyf], -np.inf)
            i0[anyf] = np.argmax(filled, axis=0)
        lag0 = lags[i0].astype(np.float64)

        cols = np.arange(Nf)
        ii = np.clip(i0, 1, nl - 2)
        y1 = corr[ii - 1, cols]
        y2 = corr[ii, cols]
        y3 = corr[ii + 1, cols]
        denom = y1 - 2.0 * y2 + y3
        interior = anyf & (i0 > 0) & (i0 < nl - 1)
        cond = interior & np.isfinite(denom) & (np.abs(denom) > 1e-12)
        with np.errstate(all="ignore"):
            delta = 0.5 * (y1 - y3) / denom
        apply_delta = cond & (np.abs(delta) <= 1.0)
        lag0[apply_delta] = lag0[apply_delta] + delta[apply_delta]
        lag0[~anyf] = np.nan
        out[fidx] = lag0

    for p in np.where(~fast)[0]:
        out[p] = _xcorr_shift_pixels(R, T[:, p], max_lag=max_lag)

    return out.reshape(ny, nx).astype(np.float32)


def _spatial_metric_mask(ny, nx, source_mask=None, edge_trim_pix=0):
    mask = np.ones((int(ny), int(nx)), dtype=bool)
    if source_mask is not None:
        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.shape != mask.shape:
            raise ValueError(
                f"source_mask shape {source_mask.shape} does not match cube plane {(ny, nx)}"
            )
        mask &= ~source_mask

    trim = int(edge_trim_pix or 0)
    if trim > 0:
        if 2 * trim >= ny or 2 * trim >= nx:
            raise ValueError("edge_trim_pix removes the full spatial frame.")
        mask[:trim, :] = False
        mask[-trim:, :] = False
        mask[:, :trim] = False
        mask[:, -trim:] = False
    return mask


def _profile_scale(profile, normalization):
    if normalization in (None, "none"):
        return 1.0
    if normalization == "median_abs":
        med = np.nanmedian(profile)
        scale = abs(float(med)) if np.isfinite(med) else np.nan
        return scale if np.isfinite(scale) and scale > 0 else 1.0
    if normalization == "robust_sigma":
        scale = robust_sigma(profile)
        return scale if np.isfinite(scale) and scale > 0 else 1.0
    raise ValueError("normalization must be 'median_abs', 'robust_sigma', 'none', or None.")


def _transverse_profile_for_metric(plane, axis, stripe_masks, spatial_mask, min_pixels):
    plane = np.asarray(plane, dtype=np.float64)
    finite = np.isfinite(plane) & spatial_mask
    ny, nx = plane.shape

    if axis == "x":
        in_stripes = np.any(np.asarray(stripe_masks, dtype=bool), axis=0)
        profile = []
        for x in range(nx):
            good = finite[:, x] & in_stripes[:, x]
            if np.count_nonzero(good) >= min_pixels:
                profile.append(float(np.nanmedian(plane[good, x])))
            else:
                profile.append(np.nan)
        return np.asarray(profile, dtype=np.float64)

    if axis == "y":
        in_stripes = np.any(np.asarray(stripe_masks, dtype=bool), axis=0)
        profile = []
        for y in range(ny):
            good = finite[y, :] & in_stripes[y, :]
            if np.count_nonzero(good) >= min_pixels:
                profile.append(float(np.nanmedian(plane[y, good])))
            else:
                profile.append(np.nan)
        return np.asarray(profile, dtype=np.float64)

    if axis == "angle":
        profile = []
        for stripe_mask in stripe_masks:
            good = finite & np.asarray(stripe_mask, dtype=bool)
            if np.count_nonzero(good) >= min_pixels:
                profile.append(float(np.nanmedian(plane[good])))
            else:
                profile.append(np.nan)
        return np.asarray(profile, dtype=np.float64)

    raise ValueError(f"Invalid axis: {axis}")


def stripe_channel_amplitudes(
    cube_zyx,
    *,
    nstripes,
    stripe_orientation="vertical",
    manual_stripe_ranges=None,
    stripe_angle_deg=None,
    source_mask=None,
    edge_trim_pix=0,
    normalization="median_abs",
    min_pixels=3,
):
    """Return one robust stripe-amplitude metric per spectral channel.

    The metric collapses each channel along the stripe direction, subtracts the
    transverse-profile median, and reports ``robust_sigma`` of the normalized
    residual profile. ``source_mask=True`` pixels are excluded from the collapse.
    """

    cube = np.asarray(cube_zyx, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (nz, ny, nx), got {cube.shape}")
    nz, ny, nx = cube.shape
    _, axis, _, stripe_masks = _build_stripe_geometry(
        ny,
        nx,
        int(nstripes),
        stripe_orientation,
        manual_stripe_ranges=manual_stripe_ranges,
        stripe_angle_deg=stripe_angle_deg,
    )
    spatial_mask = _spatial_metric_mask(
        ny,
        nx,
        source_mask=source_mask,
        edge_trim_pix=edge_trim_pix,
    )
    amp = np.full(nz, np.nan, dtype=np.float64)
    for z in range(nz):
        profile = _transverse_profile_for_metric(
            cube[z],
            axis,
            stripe_masks,
            spatial_mask,
            min_pixels=int(min_pixels),
        )
        finite = np.isfinite(profile)
        if np.count_nonzero(finite) < 3:
            continue
        scale = _profile_scale(profile[finite], normalization)
        norm_profile = profile / scale
        norm_profile = norm_profile - np.nanmedian(norm_profile)
        amp[z] = robust_sigma(norm_profile)
    return amp.astype(np.float64)


def stripe_metric_exclusion_mask(
    wavelengths_A,
    *,
    excluded_windows_A=((5780.0, 6050.0),),
    spectral_edge_channels=0,
):
    wave = np.asarray(wavelengths_A, dtype=np.float64)
    excluded = np.zeros(wave.shape, dtype=bool)
    for wmin, wmax in excluded_windows_A or ():
        excluded |= (wave >= float(wmin)) & (wave <= float(wmax))
    edge = int(spectral_edge_channels or 0)
    if edge > 0:
        if 2 * edge >= wave.size:
            raise ValueError("spectral_edge_channels removes the full wavelength axis.")
        excluded[:edge] = True
        excluded[-edge:] = True
    return excluded


def stripe_metric_summary(
    amplitude_pre,
    amplitude_post,
    wavelengths_A,
    *,
    excluded_windows_A=((5780.0, 6050.0),),
    spectral_edge_channels=0,
    dirty_threshold_factor=3.0,
    table_path="tables/stage02_stripe_metric.csv",
):
    pre = np.asarray(amplitude_pre, dtype=np.float64)
    post = np.asarray(amplitude_post, dtype=np.float64)
    wave = np.asarray(wavelengths_A, dtype=np.float64)
    if pre.shape != post.shape or pre.shape != wave.shape:
        raise ValueError("pre, post, and wavelength arrays must have matching 1D shapes.")

    excluded = stripe_metric_exclusion_mask(
        wave,
        excluded_windows_A=excluded_windows_A,
        spectral_edge_channels=spectral_edge_channels,
    )
    usable = (~excluded) & np.isfinite(pre) & np.isfinite(post)
    pre_med = median_finite(pre[usable])
    post_med = median_finite(post[usable])
    post_p95 = finite_percentile(post[usable], 95.0)

    reduction_factor = np.nan
    if np.isfinite(pre_med) and np.isfinite(post_med):
        if post_med > 0:
            reduction_factor = float(pre_med / post_med)
        elif pre_med <= 0:
            reduction_factor = 1.0

    threshold = float(dirty_threshold_factor) * post_med if np.isfinite(post_med) else np.nan
    dirty_mask = usable & np.isfinite(threshold) & (post > threshold)
    return {
        "amp_pre_median": None if not np.isfinite(pre_med) else float(pre_med),
        "amp_post_median": None if not np.isfinite(post_med) else float(post_med),
        "amp_post_p95": None if not np.isfinite(post_p95) else float(post_p95),
        "reduction_factor": (
            None if not np.isfinite(reduction_factor) else float(reduction_factor)
        ),
        "dirty_channels": [int(i) for i in np.where(dirty_mask)[0]],
        "table": str(table_path),
        "mask_excluded_windows_A": [
            [float(wmin), float(wmax)] for wmin, wmax in (excluded_windows_A or ())
        ],
    }


def shift_spectral_variance(block, dz, apply_mode="subpixel"):
    """Propagate variance through the same 1D spectral shift used for DATA."""

    if apply_mode == "none" or not np.isfinite(dz) or float(dz) == 0.0:
        return np.asarray(block, dtype=np.float32).copy(), "none"
    if apply_mode == "integer" or np.isclose(float(dz), np.round(float(dz)), atol=1e-8):
        return _shift_spectral_integer(block, dz), "integer"
    if apply_mode != "subpixel":
        raise ValueError("apply_mode must be 'none', 'integer', or 'subpixel'.")

    arr = np.asarray(block, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    nz = arr.shape[0]
    for z in range(nz):
        src = z - float(dz)
        z0 = int(np.floor(src))
        frac = src - z0
        acc = np.zeros(arr.shape[1:], dtype=np.float64)
        weight = np.zeros(arr.shape[1:], dtype=np.float64)
        for zi, wi in ((z0, 1.0 - frac), (z0 + 1, frac)):
            if zi < 0 or zi >= nz or wi == 0.0:
                continue
            finite = np.isfinite(arr[zi])
            acc[finite] += (wi * wi) * arr[zi][finite]
            weight[finite] += wi * wi
        good = weight > 0
        out[z][good] = acc[good]
    return out.astype(np.float32), "linear_kernel_squared"


def apply_stripe_spectral_shifts(
    cube_zyx,
    shifts,
    *,
    nstripes,
    stripe_orientation="vertical",
    manual_stripe_ranges=None,
    stripe_angle_deg=None,
    apply_mode="subpixel",
    is_variance=False,
):
    """Apply one spectral shift per stripe and return ``(cube, kernel_label)``."""

    cube = np.asarray(cube_zyx, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (nz, ny, nx), got {cube.shape}")
    shifts = np.asarray(shifts, dtype=np.float64).ravel()
    nz, ny, nx = cube.shape
    stripe_ranges, axis, _, stripe_masks = _build_stripe_geometry(
        ny,
        nx,
        int(nstripes),
        stripe_orientation,
        manual_stripe_ranges=manual_stripe_ranges,
        stripe_angle_deg=stripe_angle_deg,
    )
    if shifts.size != len(stripe_ranges):
        raise ValueError(f"Expected {len(stripe_ranges)} shifts, got {shifts.size}.")

    out = cube.copy()
    kernels = set()
    for s, (a1, a2) in enumerate(stripe_ranges):
        dz = float(shifts[s])
        if not np.isfinite(dz):
            kernels.add("skipped_nan")
            continue

        if axis == "y":
            view = out[:, int(a1):int(a2), :]
        elif axis == "x":
            view = out[:, :, int(a1):int(a2)]
        elif axis == "angle":
            view = out[:, stripe_masks[s]]
        else:
            raise ValueError(f"Invalid axis: {axis}")

        if is_variance:
            shifted, kernel = shift_spectral_variance(view, dz, apply_mode=apply_mode)
        elif apply_mode == "none" or dz == 0.0:
            shifted, kernel = view.astype(np.float32, copy=True), "none"
        elif apply_mode == "integer":
            shifted, kernel = _shift_spectral_integer(view, dz), "integer"
        elif apply_mode == "subpixel":
            shifted, kernel = _shift_spectral_nan_safe(view, dz), "cubic_nan_safe"
        else:
            raise ValueError("apply_mode must be 'none', 'integer', or 'subpixel'.")

        if axis == "y":
            out[:, int(a1):int(a2), :] = shifted
        elif axis == "x":
            out[:, :, int(a1):int(a2)] = shifted
        else:
            out[:, stripe_masks[s]] = shifted
        kernels.add(kernel)

    return out.astype(np.float32, copy=False), ",".join(sorted(kernels))
