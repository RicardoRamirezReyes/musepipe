"""Spectral indices as an independent SpT estimator (spec G3 §3.2, plan WP-G3R-7).

An index is a ratio of mean flux over numerator vs denominator wavelength
windows. Window definitions and the SpT calibrations come ONLY from config
(decision D7, transcribed from the cited paper — never from memory); a window
outside coverage or >50% masked is a hard error, not a silent guess. Errors are
propagated by a seeded Gaussian Monte Carlo on the per-channel flux.
"""

from __future__ import annotations

import numpy as np

from ..constants import spt_code


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


def calibration_index_range(cal):
    """Rango de VALORES del índice en que vale su calibración, o ``None``.

    Decisión del autor 2026-09-24 (``docs/2026-09-24_residuo_telurico_h2o_y_mascara_d9.md``
    §5.2): un índice cuyo valor medido cae fuera del rango de su calibración no
    se usa. La calibración publicada declara el rango en TIPO (``range``:
    ``"M3-M8"``, Riddick+2007); aquí se convierte a rango de índice resolviendo
    SpT(x) = extremos sobre la rama monótona del polinomio que contiene el
    centro (x = índice − center). ``index_range: [lo, hi]`` explícito manda.
    Devuelve ``{"index": [lo, hi], "spt": [lo, hi] | None, "source": str}``.
    """
    if cal.get("index_range") is not None:
        lo, hi = (float(v) for v in cal["index_range"])
        return {"index": [min(lo, hi), max(lo, hi)], "spt": None, "source": "index_range"}
    rng = cal.get("range")
    if not rng:
        return None
    parts = [t.strip() for t in str(rng).replace("–", "-").split("-") if t.strip()]
    if len(parts) != 2:
        raise RuntimeError(f"rango de calibración ilegible: {rng!r}")
    s_lo, s_hi = sorted((spt_code(parts[0]), spt_code(parts[1])))
    center = float(cal.get("center", 0.0))
    coeffs = np.asarray(cal["spt_poly"], float)[::-1]
    deriv = np.polyder(coeffs)
    # rama monótona que contiene x = 0: entre las raíces reales de la derivada
    roots = np.sort([r.real for r in np.roots(deriv) if abs(r.imag) < 1e-12]) \
        if deriv.size > 1 else np.array([])
    left = max([r for r in roots if r < 0.0], default=-np.inf)
    right = min([r for r in roots if r > 0.0], default=np.inf)
    xs = []
    for target in (s_lo, s_hi):
        c = coeffs.copy()
        c[-1] -= target
        sol = [r.real for r in np.roots(c) if abs(r.imag) < 1e-9
               and left - 1e-12 <= r.real <= right + 1e-12]
        if not sol:
            return {"index": None, "spt": [s_lo, s_hi],
                    "source": f"range {rng!r}: no solution on the monotonic branch"}
        xs.append(min(sol, key=abs))
    lo, hi = sorted(x + center for x in xs)
    return {"index": [float(lo), float(hi)], "spt": [s_lo, s_hi],
            "source": f"range {rng!r} mapped through spt_poly (monotonic branch)"}


def select_in_range(index_values, calibration):
    """Separa los índices calibrados en USADOS (valor dentro del rango de su
    calibración) y EXCLUIDOS (fuera, o rango no determinable), con el motivo.

    Solo mira los índices que tienen calibración (los de gravedad, como Na I,
    no entran en el tipo). Devuelve ``(usados, excluidos)``; ``excluidos`` es
    ``{name: {value, index_range, spt_range, reason}}``.
    """
    used, excluded = {}, {}
    for name, cal in calibration.items():
        if name not in index_values:
            continue
        v = float(index_values[name]["value"])
        rng = calibration_index_range(cal)
        if rng is None or rng.get("index") is None:
            excluded[name] = {"value": v, "index_range": None,
                              "spt_range": (rng or {}).get("spt"),
                              "reason": ("calibration range not declared" if rng is None
                                         else rng["source"])}
            continue
        lo, hi = rng["index"]
        if lo <= v <= hi:
            used[name] = index_values[name]
        else:
            excluded[name] = {"value": v, "index_range": [lo, hi], "spt_range": rng["spt"],
                              "reason": f"value {v:.4f} outside calibration range "
                                        f"[{lo:.4f}, {hi:.4f}] ({cal.get('range', 'index_range')})"}
    return used, excluded


__all__ = ["calibration_index_range", "indices_to_spt", "measure_indices", "select_in_range"]
