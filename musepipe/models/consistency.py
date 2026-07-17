"""Quantified consistency between independent G3 estimates (spec G3 §1.3, plan
WP-G3R-11). Each pair yields a σ discrepancy and a FROZEN verdict:
``consistent`` (<2σ), ``tension`` (2–3σ), ``discrepant`` (>3σ)."""

from __future__ import annotations

import numpy as np

CONSISTENCY_FIELDS = ["quantity_a", "value_a", "err_a", "quantity_b", "value_b",
                      "err_b", "sigma_discrepancy", "verdict", "notes"]


def sigma_discrepancy(va, ea, vb, eb) -> float:
    va, vb = float(va), float(vb)
    if not (np.isfinite(va) and np.isfinite(vb)):
        return float("nan")
    denom = float(np.hypot(float(ea), float(eb)))
    if denom <= 0:
        return 0.0 if va == vb else float("inf")
    return abs(va - vb) / denom


def verdict(sigma) -> str:
    if not np.isfinite(sigma):
        return "not_computed"
    if sigma < 2.0:
        return "consistent"
    if sigma <= 3.0:
        return "tension"
    return "discrepant"


def consistency_row(quantity_a, va, ea, quantity_b, vb, eb, notes="") -> dict:
    s = sigma_discrepancy(va, ea, vb, eb)
    return {"quantity_a": quantity_a, "value_a": va, "err_a": ea,
            "quantity_b": quantity_b, "value_b": vb, "err_b": eb,
            "sigma_discrepancy": s, "verdict": verdict(s), "notes": notes}


def not_computed_row(quantity_a, quantity_b, notes="") -> dict:
    return {"quantity_a": quantity_a, "value_a": "", "err_a": "",
            "quantity_b": quantity_b, "value_b": "", "err_b": "",
            "sigma_discrepancy": float("nan"), "verdict": "not_computed", "notes": notes}


__all__ = ["CONSISTENCY_FIELDS", "consistency_row", "not_computed_row",
           "sigma_discrepancy", "verdict"]
