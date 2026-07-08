"""Phase G0 closure helpers: hash-chain verification and legacy comparison.

G0 (`docs/spec_G0_codex_real_cube_execution.md`) executes the full A→F chain on
the real ADP cube and leaves an auditable record. These helpers build the two
G0-specific artifacts the spec adds (§5.2 phase QC, §5.3 legacy comparison) and
back the two G0 tests. They are read-only over an existing run's products.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def verify_hash_chain(links):
    """Verify a declared input/output sha256 chain across stages.

    ``links`` is an ordered list of dicts with ``stage``, ``input_sha`` and
    ``output_sha`` (any may be ``None`` when a stage does not declare it). A
    link is a mismatch when its ``input_sha`` is set but differs from the
    previous stage's ``output_sha`` — the signature of mixed-run products
    (risk #4 of the master plan). Returns {status, checks}.
    """
    checks = []
    prev_out = None
    ok = True
    for link in links:
        stage = link.get("stage")
        in_sha = link.get("input_sha")
        out_sha = link.get("output_sha")
        status = "pass"
        if in_sha is not None and prev_out is not None and in_sha != prev_out:
            status = "fail"
            ok = False
        checks.append({"stage": stage, "input_sha": in_sha,
                       "expected_prev_output": prev_out, "status": status})
        if out_sha is not None:
            prev_out = out_sha
    return {"status": "pass" if ok else "fail", "checks": checks}


def _band_flux(wave, flux, band):
    lo, hi = band
    sel = np.isfinite(wave) & np.isfinite(flux) & (wave >= lo) & (wave <= hi)
    if not np.any(sel):
        return np.nan
    return float(np.nansum(flux[sel]))


def legacy_comparison(new_wave, new_flux, legacy_wave, legacy_flux, bands, *, sigma_threshold=2.0):
    """Compare band-integrated flux of a new spectrum vs a legacy one.

    Both spectra are resampled to nothing; each band is integrated on its own
    grid. The ratio (and its deviation from 1) is reported per band; a band is
    flagged when |log10(ratio)| exceeds ``log10`` of a factor implied by the
    channel noise — here we simply flag ratios beyond a coarse factor derived
    from ``sigma_threshold`` (documentary comparison across different cubes).
    """
    new_wave = np.asarray(new_wave, float); new_flux = np.asarray(new_flux, float)
    legacy_wave = np.asarray(legacy_wave, float); legacy_flux = np.asarray(legacy_flux, float)
    rows = []
    for band in bands:
        f_new = _band_flux(new_wave, new_flux, band)
        f_leg = _band_flux(legacy_wave, legacy_flux, band)
        ratio = f_new / f_leg if (np.isfinite(f_new) and np.isfinite(f_leg) and f_leg != 0) else np.nan
        flagged = bool(np.isfinite(ratio) and (ratio > float(sigma_threshold) or ratio < 1.0 / float(sigma_threshold)))
        rows.append({
            "band_lo_A": float(band[0]), "band_hi_A": float(band[1]),
            "flux_new": f_new, "flux_legacy": f_leg, "ratio_new_over_legacy": ratio,
            "flagged": flagged,
        })
    return rows


def build_g0_qc(run_id, *, input_cube, stages_executed, stat_verdict, hash_chain_ok,
                pytest_before, pytest_after, frozen_criteria_untouched, open_issues):
    """Assemble the G0 phase QC (spec §5.2)."""
    return {
        "stage": "g0_real_cube_execution",
        "run_id": str(run_id),
        "input_cube": input_cube,
        "stages_executed": stages_executed,
        "stat_verdict": stat_verdict,
        "hash_chain_ok": bool(hash_chain_ok),
        "pytest_before": pytest_before,
        "pytest_after": pytest_after,
        "frozen_criteria_untouched": bool(frozen_criteria_untouched),
        "open_issues": list(open_issues),
    }


def read_json(path):
    return json.loads(Path(path).read_text())


__all__ = ["build_g0_qc", "legacy_comparison", "verify_hash_chain", "read_json"]
