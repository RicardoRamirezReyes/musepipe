"""Small helpers for warning about stale crop-size products in notebooks."""

from __future__ import annotations

from collections.abc import Sequence


QC_SHAPE_KEYS = (
    "spatial_shape",
    "cube_shape",
    "input_shape",
    "output_shape",
    "input_stage01_shape",
    "input_stage02_shape",
    "output_stage03_shape",
    "base_shape",
    "variance_stack_shape",
)


def expected_crop_shape(cfg):
    """Return the configured square crop shape as (ny, nx), or None."""
    try:
        npix = int(cfg.get("crop_npix"))
    except Exception:
        return None
    if npix <= 0:
        return None
    return (npix, npix)


def _is_scalar(value):
    return isinstance(value, (str, bytes)) or not isinstance(value, Sequence)


def spatial_shape_from_shape(shape):
    """Extract (ny, nx) from a normal array shape or a per-cube shape list."""
    if shape is None or _is_scalar(shape):
        return None
    vals = list(shape)
    if not vals:
        return None

    # Stage01 native products can store [[nz, ny, nx], ...].
    if not _is_scalar(vals[0]):
        return spatial_shape_from_shape(vals[0])

    if len(vals) < 2:
        return None
    try:
        return (int(vals[-2]), int(vals[-1]))
    except Exception:
        return None


def spatial_shape_from_qc(qc):
    """Return the first spatial shape advertised by a QC payload."""
    if not isinstance(qc, dict):
        return None
    for key in QC_SHAPE_KEYS:
        shape = spatial_shape_from_shape(qc.get(key))
        if shape is not None:
            return shape
    try:
        npix = int(qc.get("crop_npix"))
    except Exception:
        return None
    if npix > 0:
        return (npix, npix)
    return None


def warn_if_crop_mismatch(label, cfg, shape_yx=None, *, strict=False):
    """Print a warning when a product shape does not match CFG['crop_npix']."""
    expected = expected_crop_shape(cfg)
    found = spatial_shape_from_shape(shape_yx)
    if expected is None or found is None or found == expected:
        return False

    msg = (
        f"WARNING: {label} spatial shape is {found}, but current "
        f"CFG['crop_npix'] expects {expected}. This product may come from an "
        "older crop. Rerun affected upstream stages before interpreting these results."
    )
    print(msg)
    if strict:
        raise RuntimeError(msg)
    return True


def warn_qc_crop_mismatch(label, cfg, qc, *, strict=False):
    """Print a warning when a saved QC payload does not match CFG['crop_npix']."""
    expected = expected_crop_shape(cfg)
    found = spatial_shape_from_qc(qc)
    mismatch = warn_if_crop_mismatch(label, cfg, found, strict=strict)

    if isinstance(qc, dict) and expected is not None and qc.get("crop_npix") is not None:
        try:
            stored = int(qc.get("crop_npix"))
        except Exception:
            stored = None
        if stored is not None and (stored, stored) != expected:
            msg = (
                f"WARNING: {label} stored crop_npix={stored}, but current "
                f"CFG['crop_npix']={expected[0]}. Rerun affected upstream stages."
            )
            print(msg)
            if strict:
                raise RuntimeError(msg)
            mismatch = True

    return mismatch
