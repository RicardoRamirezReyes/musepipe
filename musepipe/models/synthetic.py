"""Synthetic mini-library and mini-tracks for tests/demos (spec G3 §1.3).

Real libraries (BT-Settl, BHAC15/ATMO2020, Luhman/Bonnefoy) live outside the
repo; these deterministic stand-ins let the adapter contract, fit and label
machinery be validated without any external data.
"""

from __future__ import annotations

import numpy as np

from . import TemplateSpectrum


class SyntheticGridLibrary:
    """A smooth pseudo-photosphere: Rayleigh-Jeans-ish slope + a Teff-dependent
    absorption band. Deterministic; ``grid()`` exposes the axes."""

    def __init__(self, teff_axis=(2600.0, 2800.0, 3000.0), logg_axis=(3.5, 4.0, 4.5),
                 wave_A=None, band_center_A=8180.0):
        self._teff = np.asarray(teff_axis, float)
        self._logg = np.asarray(logg_axis, float)
        self.wave_A = np.linspace(6000.0, 9000.0, 600) if wave_A is None else np.asarray(wave_A, float)
        self.band_center_A = float(band_center_A)

    def grid(self):
        return {"teff": self._teff.copy(), "logg": self._logg.copy()}

    def get(self, *, teff, logg=4.0, **_):
        w = self.wave_A
        cont = (w / 6000.0) ** (-2.0) * (float(teff) / 3000.0)  # warmer -> brighter continuum
        depth = 0.35 * (3000.0 - float(teff)) / 400.0            # cooler -> deeper band
        band = 1.0 - np.clip(depth, 0.0, 0.9) * np.exp(-0.5 * ((w - self.band_center_A) / 60.0) ** 2)
        grav = 1.0 + 0.02 * (float(logg) - 4.0)
        return TemplateSpectrum(w, cont * band * grav, meta={"teff": float(teff), "logg": float(logg)})


class SyntheticTracks:
    """Monotone toy evolutionary model: mass ∝ L_bol^0.4, radius ∝ L_bol^0.3,
    with an age-dependent scaling. Interpolation error = half a step term."""

    def __init__(self, name="synthetic_tracks", version="0.1"):
        self.name = name
        self.version = version

    def lookup(self, l_bol, age, *, teff=None):
        l = float(l_bol); a = float(age)
        age_fac = (a / 5.0) ** -0.15
        mass = 0.02 * (l / 1e-3) ** 0.4 * age_fac
        radius = 0.15 * (l / 1e-3) ** 0.3 * age_fac
        return {
            "mass_msun": float(mass), "mass_err": float(0.1 * mass),
            "radius_rsun": float(radius), "radius_err": float(0.1 * radius),
            "logg": float(4.0 + np.log10(mass / max(radius, 1e-6) ** 2)),
            "source": f"{self.name}:{self.version}",
        }


__all__ = ["SyntheticGridLibrary", "SyntheticTracks"]
