"""Physical and conversion constants (single source of truth).

Named, with units. Scientific/astrophysical target values live in run config,
never here (plan §6 / index §7.4).
"""

from __future__ import annotations

import math

# Speed of light
C_KMS = 299792.458  # km/s

# Gaussian FWHM <-> sigma
FWHM_OVER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))  # ~2.35482
SIGMA_OVER_FWHM = 1.0 / FWHM_OVER_SIGMA


def fwhm_to_sigma(fwhm):
    return float(fwhm) * SIGMA_OVER_FWHM


def sigma_to_fwhm(sigma):
    return float(sigma) * FWHM_OVER_SIGMA


__all__ = ["C_KMS", "FWHM_OVER_SIGMA", "SIGMA_OVER_FWHM", "fwhm_to_sigma", "sigma_to_fwhm"]
