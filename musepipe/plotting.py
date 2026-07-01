"""Plotting helpers shared by stage diagnostics.

Later phases will move robust image scaling and common annotations here.
"""


def smoothing_label(sigma_px):
    """Human-readable label describing whether/how a panel was smoothed."""

    if sigma_px is not None and float(sigma_px) > 0:
        return f"smoothed: Gaussian sigma={float(sigma_px):g} px"
    return "raw pixels (no smoothing)"


__all__ = ["smoothing_label"]
