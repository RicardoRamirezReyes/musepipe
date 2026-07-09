"""Empirical common-flux-scale check between extraction methods (D1 v2 §3.1).

Structural gate, not a statistical test: a real convention offset (e.g. the
x20 pedestal the optimal-LS controls inherited from the stage04b residual)
exceeds the frozen threshold by orders of magnitude, while reconciled methods
sit well below it. A failed check degrades the PAIR in D1, never the global
verdict.
"""

from __future__ import annotations

import numpy as np


def pair_scale_check(controls_i, controls_j, *, gate_sigma: float = 5.0, level_ratio_max: float = 3.0) -> dict:
    """Check that two methods' control spectra share a common flux scale.

    Uses the broad-band mean diff per control: ``mu = mean_k(mean_lambda(d_k))``
    with ``s = std_k(ddof=1)`` and stat ``|mu| / (s/sqrt(n))``, plus the ratio
    of median absolute control levels. The pair FAILS only when the offset is
    significant (stat >= gate_sigma) AND the levels are grossly mismatched
    (level_ratio > level_ratio_max): a significant but level-consistent offset
    is a method's additive bias at the control ring, which the centred t of
    D1 v2 §3.4 absorbs by design (spec v2 §3.1, revision v2.1 — the x20
    convention break showed level_ratio ~20 vs ~1.1-1.6 once reconciled).
    D1 additionally treats all primary pairs failing with level_ratio > x10
    as an extraction-stage bug.
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
        "level_ratio_max": float(level_ratio_max),
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
    significant = bool(stat >= gate)
    if level_ratio is None:
        # No usable level reference: fall back to significance alone.
        out["ok"] = not significant
    else:
        out["ok"] = not (significant and level_ratio > float(level_ratio_max))
    return out


__all__ = ["pair_scale_check"]
