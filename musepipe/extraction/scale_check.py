"""Empirical common-flux-scale check between extraction methods (D1 v2 §3.1).

Structural gate, not a statistical test: a real convention offset (e.g. the
x20 pedestal the optimal-LS controls inherited from the stage04b residual)
exceeds the frozen threshold by orders of magnitude, while reconciled methods
sit well below it. A failed check degrades the PAIR in D1, never the global
verdict.
"""

from __future__ import annotations

import numpy as np


def pair_scale_check(controls_i, controls_j, *, gate_sigma: float = 5.0) -> dict:
    """Check that two methods' control spectra share a common flux scale.

    Uses the broad-band mean diff per control: ``mu = mean_k(mean_lambda(d_k))``
    with ``s = std_k(ddof=1)`` and the frozen gate ``|mu| / (s/sqrt(n)) < 5.0``
    (D1 v2 §3.1). Also reports the ratio of median absolute control levels,
    used by D1 to distinguish a pair-level degradation from an extraction-stage
    bug (all primary pairs failing with a >x10 level ratio).
    """

    ci = np.asarray(controls_i, dtype=np.float64)
    cj = np.asarray(controls_j, dtype=np.float64)
    if ci.ndim != 2 or cj.ndim != 2:
        raise ValueError("control spectra must be 2D (n_controls, n_channels).")
    if ci.shape != cj.shape:
        raise ValueError(f"control spectra shapes differ: {ci.shape} != {cj.shape}.")

    n = int(ci.shape[0])
    gate = float(gate_sigma)
    level_i = float(np.nanmedian(np.abs(ci))) if ci.size else np.nan
    level_j = float(np.nanmedian(np.abs(cj))) if cj.size else np.nan
    if np.isfinite(level_i) and np.isfinite(level_j) and min(level_i, level_j) > 0:
        level_ratio = float(max(level_i, level_j) / min(level_i, level_j))
    else:
        level_ratio = None

    out = {
        "n_controls": n,
        "gate_sigma": gate,
        "mu": None,
        "s": None,
        "stat": None,
        "ok": None,
        "level_ratio": level_ratio,
    }
    if n < 2:
        out["reason"] = "insufficient_controls"
        return out

    with np.errstate(all="ignore"):
        per_control = np.nanmean(ci - cj, axis=1)
    per_control = per_control[np.isfinite(per_control)]
    if per_control.size < 2:
        out["reason"] = "no_finite_control_diffs"
        return out

    mu = float(np.mean(per_control))
    s = float(np.std(per_control, ddof=1))
    out["mu"] = mu
    out["s"] = s
    if s <= 0:
        stat = 0.0 if mu == 0.0 else float("inf")
    else:
        stat = abs(mu) / (s / np.sqrt(per_control.size))
    out["stat"] = float(stat)
    out["ok"] = bool(stat < gate)
    return out


__all__ = ["pair_scale_check"]
