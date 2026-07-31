"""Aperture and same-radius control-position helpers."""

from __future__ import annotations

import math

import numpy as np


def angular_separation_deg(a, b) -> float:
    """Smallest angular separation between two angles in radians, in degrees."""

    return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))


def same_radius_control_positions(
    object_yx,
    star_yx,
    ny,
    nx,
    n_positions=8,
    exclude_angle_deg=25.0,
    margin_px=4,
):
    """Return integer control positions at the same star-object radius."""

    oy, ox = map(float, object_yx)
    sy, sx = map(float, star_yx)
    dy = oy - sy
    dx = ox - sx
    radius = math.hypot(dy, dx)
    theta0 = math.atan2(dy, dx)

    controls = []
    for k in range(int(n_positions)):
        theta = theta0 + 2.0 * math.pi * k / float(n_positions)
        if angular_separation_deg(theta, theta0) < exclude_angle_deg:
            continue
        y = int(round(sy + radius * math.sin(theta)))
        x = int(round(sx + radius * math.cos(theta)))
        if margin_px <= y < ny - margin_px and margin_px <= x < nx - margin_px:
            controls.append((y, x))
    return controls


def aperture_weights(ny, nx, center_yx, aperture) -> np.ndarray:
    """Build a 2D aperture-weight image."""

    y0, x0 = map(float, center_yx)
    yy, xx = np.mgrid[:ny, :nx]
    rr2 = (yy - y0) ** 2 + (xx - x0) ** 2
    weights = np.zeros((ny, nx), dtype=np.float64)
    kind = aperture["kind"]

    if kind == "pixel":
        y = int(round(y0))
        x = int(round(x0))
        if 0 <= y < ny and 0 <= x < nx:
            weights[y, x] = 1.0
    elif kind == "box":
        size = int(aperture.get("size", 3))
        if size < 1 or size % 2 == 0:
            raise ValueError(
                "box aperture size must be an odd positive integer, got "
                f"size={size}. An even size has no integer-centred box: "
                f"`half = size // 2` would silently return the {2 * (size // 2) + 1}x"
                f"{2 * (size // 2) + 1} one under the wrong label."
            )
        half = size // 2
        y = int(round(y0))
        x = int(round(x0))
        y1 = max(0, y - half)
        y2 = min(ny, y + half + 1)
        x1 = max(0, x - half)
        x2 = min(nx, x + half + 1)
        weights[y1:y2, x1:x2] = 1.0
    elif kind == "circle":
        radius = float(aperture["radius_px"])
        weights[rr2 <= radius**2] = 1.0
    elif kind == "gaussian":
        sigma = float(aperture["sigma_px"])
        radius = float(aperture.get("radius_px", 3.0 * sigma))
        mask = rr2 <= radius**2
        weights[mask] = np.exp(-0.5 * rr2[mask] / sigma**2)
    else:
        raise ValueError(f"Unknown aperture kind: {kind}")
    return weights


def _box_bounds(ny, nx, yc, xc, box_size):
    box_size = int(box_size)
    if box_size < 1 or box_size % 2 == 0:
        raise ValueError(
            f"box_size must be an odd positive integer, got box_size={box_size}. "
            "The run knob is `box_aperture_size_px`."
        )
    half = box_size // 2
    yc = int(yc)
    xc = int(xc)
    return (
        max(0, yc - half),
        min(ny, yc + half + 1),
        max(0, xc - half),
        min(nx, xc + half + 1),
    )


def box_spectrum_sum(cube_zyx, yc, xc, box_size=3) -> np.ndarray:
    """Sum spectrum inside a clipped square aperture."""

    cube = np.asarray(cube_zyx)
    _, ny, nx = cube.shape
    y1, y2, x1, x2 = _box_bounds(ny, nx, yc, xc, box_size)
    sub = cube[:, y1:y2, x1:x2]
    return np.nansum(sub, axis=(1, 2)).astype(np.float64)


def box_spectrum_mean(cube_zyx, yc, xc, box_size=3) -> np.ndarray:
    """Mean spectrum inside a clipped square aperture."""

    cube = np.asarray(cube_zyx)
    _, ny, nx = cube.shape
    y1, y2, x1, x2 = _box_bounds(ny, nx, yc, xc, box_size)
    sub = cube[:, y1:y2, x1:x2]
    return np.nanmean(sub, axis=(1, 2)).astype(np.float64)


__all__ = [
    "angular_separation_deg",
    "aperture_weights",
    "box_spectrum_mean",
    "box_spectrum_sum",
    "same_radius_control_positions",
]
