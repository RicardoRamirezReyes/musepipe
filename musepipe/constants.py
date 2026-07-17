"""Physical and conversion constants (single source of truth).

Named, with units. Scientific/astrophysical target values live in run config,
never here (plan §6 / index §7.4).
"""

from __future__ import annotations

import math
import re

# Speed of light
C_KMS = 299792.458  # km/s

# Numeric spectral-type code (plan WP-G3R-4, decision D-frozen encoding):
# M0=0.0 ... M9=9.0, L0=10.0 ...; earlier classes are negative (K5=-5.0),
# half subtype = 0.5. Continuous so nearest-type lookups are simple.
_SPT_CLASSES = "OBAFGKMLTY"
_SPT_M_INDEX = _SPT_CLASSES.index("M")  # 6
_SPT_RE = re.compile(r"^([OBAFGKMLTY])(\d(?:\.\d)?)$")


def spt_code(label) -> float:
    """Spectral-type label (e.g. 'M4.5', 'K7', 'O5') -> numeric code."""
    m = _SPT_RE.match(str(label).strip())
    if not m:
        raise ValueError(f"unrecognized spectral type: {label!r}")
    cls, sub = m.group(1), float(m.group(2))
    return 10.0 * (_SPT_CLASSES.index(cls) - _SPT_M_INDEX) + sub


def spt_label(code) -> str:
    """Numeric code -> spectral-type label, rounded to the nearest half subtype."""
    code = round(float(code) * 2.0) / 2.0  # nearest 0.5
    k = math.floor(code / 10.0)
    sub = code - 10.0 * k
    idx = _SPT_M_INDEX + k
    if idx < 0 or idx >= len(_SPT_CLASSES):
        raise ValueError(f"spectral-type code out of range: {code}")
    sub_str = str(int(sub)) if sub == int(sub) else f"{sub:.1f}"
    return f"{_SPT_CLASSES[idx]}{sub_str}"

# Gaussian FWHM <-> sigma
FWHM_OVER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))  # ~2.35482
SIGMA_OVER_FWHM = 1.0 / FWHM_OVER_SIGMA


def fwhm_to_sigma(fwhm):
    return float(fwhm) * SIGMA_OVER_FWHM


def sigma_to_fwhm(sigma):
    return float(sigma) * FWHM_OVER_SIGMA


__all__ = ["C_KMS", "FWHM_OVER_SIGMA", "SIGMA_OVER_FWHM", "fwhm_to_sigma",
           "sigma_to_fwhm", "spt_code", "spt_label"]
