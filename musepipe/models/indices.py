"""Spectral indices as an independent SpT estimator (spec G3 §3.2, plan WP-G3R-7).

An index is a ratio of mean flux over numerator vs denominator wavelength
windows. Window definitions and the SpT calibrations come ONLY from config
(decision D7, transcribed from the cited paper — never from memory); a window
outside coverage or >50% masked is a hard error, not a silent guess. Errors are
propagated by a seeded Gaussian Monte Carlo on the per-channel flux.
"""

from __future__ import annotations

import numpy as np


def _window_mean(wave, flux, good, windows, name):
    sel = np.zeros(wave.size, dtype=bool)
    for lo, hi in windows:
        lo, hi = float(lo), float(hi)
        if lo < wave.min() or hi > wave.max():
            raise RuntimeError(
                f"index {name}: window {lo}-{hi} Å outside coverage "
                f"[{wave.min():.1f}, {wave.max():.1f}]")
        w = (wave >= lo) & (wave <= hi)
        if w.sum() == 0:
            raise RuntimeError(f"index {name}: window {lo}-{hi} Å has no channels")
        if (w & ~good).sum() / w.sum() > 0.5:
            raise RuntimeError(f"index {name}: window {lo}-{hi} Å is >50% masked")
        sel |= w
    usable = sel & good
    if not usable.any():
        raise RuntimeError(f"index {name}: no usable channels in windows")
    return usable


def measure_indices(wave, flux, err, definitions, *, mask=None, seed=0, n_mc=500) -> dict:
    """Measure each index and its MC error.

    ``definitions``: ``{name: {numerator:[[lo,hi],...], denominator:[[lo,hi],...],
    citation: str}}``. Returns ``{name: {value, err, citation}}``.
    """
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    err = np.asarray(err, float)
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if mask is not None:
        good &= ~np.asarray(mask, dtype=bool)
    rng = np.random.default_rng(int(seed))

    out = {}
    for name, d in definitions.items():
        num = _window_mean(wave, flux, good, d["numerator"], name)
        den = _window_mean(wave, flux, good, d["denominator"], name)

        def ratio(f):
            return float(np.mean(f[num]) / np.mean(f[den]))

        value = ratio(flux)
        pert = flux[None, :] + rng.normal(0.0, np.where(good, err, 0.0), size=(int(n_mc), flux.size))
        samples = np.array([ratio(pert[k]) for k in range(int(n_mc))])
        out[name] = {"value": value, "err": float(np.std(samples)),
                     "citation": d.get("citation")}
    return out


def indices_to_spt(index_values, calibration) -> tuple[float, float]:
    """Combine per-index SpT estimates into (spt_code, err).

    ``calibration``: ``{name: {spt_poly:[c0,c1,...], center: x0, citation: str}}``
    with SpT = c0 + c1·(x−x0) + … (x = index value; ``center`` defaults to 0).
    The centered form matches the published relations (e.g. Riddick et al. 2007)
    verbatim, avoiding transcription error from expanding into powers of x.
    Combined SpT = mean over indices; err combines the per-index MC propagation
    and the between-index scatter.
    """
    ests, errs = [], []
    for name, cal in calibration.items():
        if name not in index_values:
            continue
        x = index_values[name]["value"] - float(cal.get("center", 0.0))
        xe = index_values[name]["err"]
        coeffs = np.asarray(cal["spt_poly"], float)[::-1]  # np.polyval wants high->low
        ests.append(float(np.polyval(coeffs, x)))
        deriv = np.polyval(np.polyder(coeffs), x)
        errs.append(abs(float(deriv)) * xe)
    if not ests:
        return float("nan"), float("nan")
    ests = np.asarray(ests)
    errs = np.asarray(errs)
    spt = float(np.mean(ests))
    err = float(np.sqrt(np.var(ests) + np.mean(errs ** 2)))
    return spt, err


__all__ = ["indices_to_spt", "measure_indices"]
