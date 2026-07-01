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
