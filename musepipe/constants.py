"""Physical and conversion constants (single source of truth).

Named, with units. Scientific/astrophysical target values live in run config,
never here (plan §6 / index §7.4).
"""

from __future__ import annotations

import math
import re

# Speed of light
C_KMS = 299792.458  # km/s

# Physical constants for the derived chain (WP-G3R-9), cgs, with sources.
PC_CM = 3.0856775814913673e18       # parsec [cm] (IAU 2015 B2)
RSUN_CM = 6.957e10                   # nominal solar radius [cm] (IAU 2015 B3)
RJUP_CM = 7.1492e9                   # nominal Jupiter equatorial radius [cm] (IAU 2015 B3)
SIGMA_SB_CGS = 5.670374419e-5        # Stefan-Boltzmann [erg cm^-2 s^-1 K^-4] (CODATA 2018)
LSUN_ERG_S = 3.828e33               # nominal solar luminosity [erg/s] (IAU 2015 B3)
MSUN_OVER_MJUP = 1047.57            # M_sun / M_Jup (IAU nominal GM ratio)

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


__all__ = ["C_KMS", "FWHM_OVER_SIGMA", "LSUN_ERG_S", "MSUN_OVER_MJUP", "PC_CM",
           "RJUP_CM", "RSUN_CM", "SIGMA_OVER_FWHM", "SIGMA_SB_CGS",
           "fwhm_to_sigma", "sigma_to_fwhm", "spt_code", "spt_label"]
