"""Multiline accretion (spec G3 §3.5): L_acc per line from G2 measurements and
published L_acc–L_line relations, combined result, and Mdot with inherited M,R.

Reuses the H03 conversions so the Halpha result is identical to H03 (V5)."""

from __future__ import annotations

import math

import numpy as np

from ..stages.stage_h03_limits import (MAGNETOSPHERIC_FACTOR, lacc_lsun_from_lha,
                                       luminosity_erg_s, mdot_msun_yr_from_lacc)

L_SUN_ERG_S = 3.828e33


def line_lacc(line_flux, *, distance_pc, av, a_lambda_over_av, relation, flux_unit_cgs=1.0):
    """Dereddened L_line (Lsun) → L_acc (Lsun) via a cited relation.

    ``relation`` = {"a": slope, "b": intercept, "scatter_dex": s, "citation": ...,
    "validity_range": [...]}; scatter enters the L_acc error in dex.
    """
    for key in ("a", "b", "scatter_dex", "citation"):
        if relation.get(key) is None:
            raise RuntimeError(f"L_acc–L_line relation requires '{key}' (spec G3 §1.2 mandatory citation).")
    f_obs = float(line_flux) * float(flux_unit_cgs)
    f_dered = f_obs * (10.0 ** (0.4 * float(a_lambda_over_av) * float(av)))
    l_line_erg = luminosity_erg_s(f_dered, distance_pc)
    l_line_lsun = l_line_erg / L_SUN_ERG_S
    l_acc_lsun = lacc_lsun_from_lha(l_line_lsun, float(relation["a"]), float(relation["b"]))
    return {"l_line_lsun": float(l_line_lsun), "l_acc_lsun": float(l_acc_lsun),
            "scatter_dex": float(relation["scatter_dex"]), "citation": str(relation["citation"])}


def combine_accretion(per_line):
    """Combine per-line L_acc. Detections combined only if compatible; limits →
    the MOST RESTRICTIVE individual (never a statistical combination of limits)."""
    detections = [p for p in per_line if p.get("status") == "detected" and np.isfinite(p.get("l_acc_lsun", np.nan))]
    limits = [p for p in per_line if p.get("status") in ("upper_limit", "marginal") and np.isfinite(p.get("l_acc_lsun", np.nan))]
    if detections:
        vals = np.array([p["l_acc_lsun"] for p in detections])
        errs = np.array([max(p.get("l_acc_err_lsun", np.nan), 1e-30) for p in detections])
        logs = np.log10(vals)
        wmean = float(np.average(logs, weights=1.0 / (np.array([p["scatter_dex"] for p in detections]) ** 2)))
        chi2 = float(np.sum(((logs - wmean) / np.array([p["scatter_dex"] for p in detections])) ** 2))
        compatible = chi2 < 3.0 * max(1, len(detections) - 1)
        return {"kind": "detection", "l_acc_lsun": float(10.0 ** wmean) if compatible else None,
                "compatible": bool(compatible), "chi2": chi2,
                "status": "combined" if compatible else "discrepant"}
    if limits:
        best = min(limits, key=lambda p: p["l_acc_lsun"])
        return {"kind": "upper_limit", "l_acc_lsun": float(best["l_acc_lsun"]),
                "from_line": best.get("name"), "rule": "most_restrictive_individual"}
    return {"kind": "none", "l_acc_lsun": None}


def mdot_from_lacc(l_acc_lsun, mass_msun, radius_rsun, *, factor=MAGNETOSPHERIC_FACTOR):
    """Mdot = factor · L_acc · R / (G M); factor 1.25 for R_in = 5 R_star (spec §3.5)."""
    base = mdot_msun_yr_from_lacc(l_acc_lsun, mass_msun, radius_rsun)
    return float(factor) * base if np.isfinite(base) else np.nan


def mdot_mc(l_acc_lsun, mass_msun, mass_err, radius_rsun, radius_err, scatter_dex,
            *, factor=MAGNETOSPHERIC_FACTOR, n_mc=2000, seed=0):
    """Monte-Carlo Mdot sampling L_acc (log-normal via scatter), M and R."""
    rng = np.random.default_rng(int(seed))
    lacc = 10.0 ** (np.log10(l_acc_lsun) + rng.normal(0.0, scatter_dex, n_mc))
    m = np.clip(rng.normal(mass_msun, mass_err, n_mc), 1e-6, None)
    r = np.clip(rng.normal(radius_rsun, radius_err, n_mc), 1e-6, None)
    vals = np.array([mdot_from_lacc(l, mi, ri, factor=factor) for l, mi, ri in zip(lacc, m, r)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"p16": np.nan, "p50": np.nan, "p84": np.nan}
    return {"p16": float(np.percentile(vals, 16)), "p50": float(np.percentile(vals, 50)),
            "p84": float(np.percentile(vals, 84)), "n_mc": int(n_mc), "seed": int(seed)}


__all__ = ["combine_accretion", "line_lacc", "mdot_from_lacc", "mdot_mc"]
