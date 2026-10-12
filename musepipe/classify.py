"""Phase G4 transparent source classification: a hypothesis × test evidence
matrix (spec G4). The product is the MATRIX, never an opaque label (§1.1).

Each test contributes, per hypothesis, a discrete verdict
(supports|neutral|disfavors|excludes) mapped to a log-likelihood weight from
FROZEN config thresholds (§1.2 anti-bias). Correlated tests are grouped so
shared evidence is not counted as independent (§4.1).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

HYPOTHESES = (
    "planet_forming", "substellar_companion", "brown_dwarf",
    "m_star_associated", "m_star_background", "contaminant", "artifact",
)
VERDICTS = ("supports", "neutral", "disfavors", "excludes", "not_available")

DEFAULT_WEIGHTS = {"supports": 1.0, "neutral": 0.0, "disfavors": -1.0,
                   "excludes": -1000.0, "not_available": 0.0}


def frozen_thresholds_hash(thresholds):
    """Stable sha256 of the frozen threshold dict (spec §6 / V5 anti-bias)."""
    blob = json.dumps(thresholds, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _cell_weight(verdict, weights):
    return float(weights.get(str(verdict), 0.0))


def combine_evidence(matrix, *, groups=(), weights=None):
    """Sum per-hypothesis log-likelihood over tests; correlated groups count once.

    ``matrix`` = {test: {hypothesis: {"verdict": ...}}}. ``groups`` is a list of
    test-name tuples that share evidence (e.g. ("T3","T4","T6")); within a group
    only the strongest (most negative-or-positive by |weight|) verdict per
    hypothesis contributes, so shared spectrum/A_V evidence is not multiplied.
    """
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    grouped = {t for g in groups for t in g}
    log_l = {h: 0.0 for h in HYPOTHESES}

    # ungrouped tests: additive
    for test, per_h in matrix.items():
        if test in grouped:
            continue
        for h in HYPOTHESES:
            v = (per_h.get(h) or {}).get("verdict", "not_available")
            log_l[h] += _cell_weight(v, weights)
    # grouped tests: one joint contribution per group (strongest |weight|)
    for g in groups:
        present = [t for t in g if t in matrix]
        if not present:
            continue
        for h in HYPOTHESES:
            ws = [_cell_weight((matrix[t].get(h) or {}).get("verdict", "not_available"), weights) for t in present]
            log_l[h] += ws[int(np.argmax(np.abs(ws)))] if ws else 0.0
    top = max(log_l.values()) if log_l else 0.0
    ranking = sorted(({"hypothesis": h, "log_l_rel": log_l[h] - top} for h in HYPOTHESES),
                     key=lambda d: d["log_l_rel"], reverse=True)
    return log_l, ranking


def dominant_tests(matrix, hypothesis, *, weights=None, top_n=3):
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    scored = []
    for test, per_h in matrix.items():
        v = (per_h.get(hypothesis) or {}).get("verdict", "not_available")
        w = _cell_weight(v, weights)
        if w != 0:
            scored.append((test, w))
    scored.sort(key=lambda x: abs(x[1]), reverse=True)
    return [t for t, _ in scored[:top_n]]


def n_independent_supports(matrix, hypothesis, *, groups=()):
    """Count independent tests that 'support' the hypothesis (a group counts once)."""
    grouped = {t for g in groups for t in g}
    n = 0
    for test, per_h in matrix.items():
        if test in grouped:
            continue
        if (per_h.get(hypothesis) or {}).get("verdict") == "supports":
            n += 1
    for g in groups:
        if any((matrix.get(t, {}).get(hypothesis) or {}).get("verdict") == "supports" for t in g):
            n += 1
    return n


def has_exclusion_against(matrix, hypothesis):
    return any((per_h.get(hypothesis) or {}).get("verdict") == "excludes" for per_h in matrix.values())


def background_probability(density_per_arcsec2, area_arcsec2):
    """Poisson P(>=1 contaminant) = 1 - exp(-density*area)."""
    mu = float(density_per_arcsec2) * float(area_arcsec2)
    return float(1.0 - np.exp(-mu))


def leave_one_out_leader(matrix, *, groups=(), weights=None):
    """Return (full_leader, {omitted_test: leader}) — leader hypothesis per omission."""
    _ll, rank = combine_evidence(matrix, groups=groups, weights=weights)
    full = rank[0]["hypothesis"]
    loo = {}
    for omit in list(matrix.keys()):
        sub = {t: v for t, v in matrix.items() if t != omit}
        _l, r = combine_evidence(sub, groups=groups, weights=weights)
        loo[omit] = r[0]["hypothesis"]
    return full, loo


def tied_at_top(matrix, *, groups=(), weights=None, margin=1e-9):
    """Hypotheses within ``margin`` of the top log-likelihood, in HYPOTHESES order.

    One element = a leader. More than one = a tie, and in a tie the first one of
    ``combine_evidence``'s ranking is only the first one of the HYPOTHESES tuple
    (the sort is stable): it is not a leader."""
    log_l, _ = combine_evidence(matrix, groups=groups, weights=weights)
    top = max(log_l.values())
    return tuple(h for h in HYPOTHESES if top - log_l[h] < margin)


def classify(matrix, *, background_prob, thresholds, groups=(), weights=None, tie_margin=1e-9):
    """Final class + robustness (spec §6 frozen rules).

    A tie at the top (``tied_at_top`` with ``tie_margin``) has no leader: the
    label is None and the robustness ``ambiguous`` (spec §8: report all the tied
    hypotheses, do not force one). Before 2026-10-10 the tuple order broke the
    tie, and leave-one-out could not see it because every omission broke it the
    same way: in ROXs 12 b, without a usable mass, four bound hypotheses tied and
    the first of the tuple came out ``secure``. Leave-one-out now compares the
    tied SET: an omission that splits or widens it changes the leader."""
    tied = tied_at_top(matrix, groups=groups, weights=weights, margin=tie_margin)
    loo = {omit: tied_at_top({t: v for t, v in matrix.items() if t != omit},
                             groups=groups, weights=weights, margin=tie_margin)
           for omit in matrix}
    loo_stable = all(v == tied for v in loo.values())
    leader = tied[0] if len(tied) == 1 else None
    n_supp = min(n_independent_supports(matrix, h, groups=groups) for h in tied)
    excluded = any(has_exclusion_against(matrix, h) for h in tied)
    p_bg = float(background_prob)
    secure_p = float(thresholds.get("secure_bg_prob_max", 0.01))
    probable_p = float(thresholds.get("probable_bg_prob_max", 0.10))
    min_supp = int(thresholds.get("secure_min_supports", 3))

    if leader is None:
        robustness = "ambiguous"
    elif n_supp >= min_supp and not excluded and loo_stable and p_bg < secure_p:
        robustness = "secure"
    elif loo_stable and not excluded and p_bg < probable_p:
        robustness = "probable"
    else:
        robustness = "ambiguous"
    return {"label": leader, "tied_at_top": list(tied), "robustness": robustness,
            "leave_one_out_stable": bool(loo_stable),
            "n_independent_supports": int(n_supp), "exclusion_against_leader": bool(excluded),
            "background_prob": p_bg}


__all__ = [
    "DEFAULT_WEIGHTS", "HYPOTHESES", "VERDICTS", "background_probability",
    "classify", "combine_evidence", "dominant_tests", "frozen_thresholds_hash",
    "has_exclusion_against", "leave_one_out_leader", "n_independent_supports", "tied_at_top",
]
