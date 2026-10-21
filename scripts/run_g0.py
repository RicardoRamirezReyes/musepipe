#!/usr/bin/env python
"""WP-6 / G0 runner — real-cube execution phase QC for an arbitrary run.

Reproduces the structure of ``runs/ROXs12b_B_adp/stages/stage_g0_qc.json`` (the
contract) composing every field from the public helpers in ``musepipe.g0``
(``build_g0_qc``, ``legacy_comparison``) plus the run's
own generated QCs (F1 ``run_summary.json`` semaphores, ``stage00q`` STAT verdict).

Legacy comparison: G0 compares the new spectrum against a legacy one to catch
mixed-run/provenance errors. For a run that REPLACES the ADP substitute we use
the ADP psffit object spectrum as the legacy reference (``--legacy-run``,
default ROXs12b_B_adp); both share the 3681-channel wavelength grid so bands are
integrated on a common frame. Near-unity ratios indicate consistent provenance.

Usage: python scripts/run_g0.py --run-id ROXs12b_realigned
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musepipe.g0 import build_g0_qc, legacy_comparison

LEGACY_BANDS = [
    ("continuum_blue", 4900.0, 5400.0),
    ("continuum", 5450.0, 5750.0),
    ("continuum_pre_Halpha", 6100.0, 6400.0),
    ("continuum_post_Halpha", 6600.0, 6800.0),
    ("continuum_red", 7600.0, 8000.0),
    ("continuum_far_red", 8600.0, 9100.0),
    ("Halpha", 6553.0, 6573.0),
    ("Hbeta", 4851.0, 4871.0),
    ("OI_8446", 8436.0, 8456.0),
]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_spectrum(run_dir):
    with fits.open(run_dir / "stages" / "spec_psffit_object.fits") as h:
        data = h["SPECTRUM"].data
        cols = getattr(data, "names", None)
        if cols and "flux" in cols:
            wave = np.asarray(data["wave_A"], dtype=np.float64)
            flux = np.asarray(data["flux"], dtype=np.float64)
        else:  # plain 1D flux + wavelength from the cube grid
            flux = np.asarray(data, dtype=np.float64)
            with fits.open(run_dir / "stages" / "stage02_xcorr_cube_stack.fits", memmap=True) as hc:
                wave = np.asarray(hc["WAVELENGTH"].data, dtype=np.float64)
    return wave, flux


def build_g0(run_id, project_root=None, legacy_run=None, pytest_after=None, no_legacy=False):
    root = Path(project_root or Path.cwd()).resolve()
    run_dir = root / "runs" / run_id
    stage_dir = run_dir / "stages"
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    rs = json.loads((run_dir / "report" / "run_summary.json").read_text())
    q00q = json.loads((stage_dir / "stage00q_qc.json").read_text())

    cube_path = stage_dir / "stage02_xcorr_cube_stack.fits"
    with fits.open(cube_path, memmap=True) as h:
        exts = [hdu.name for hdu in h]
        data = np.asarray(h["CUBES"].data, dtype=np.float32)
    nan_frac = float(np.mean(~np.isfinite(data)))

    input_cube = {
        "file": f"runs/{run_id}/stages/stage02_xcorr_cube_stack.fits",
        "sha256": _sha256(cube_path),
        "extensions": exts,
        "wave_frame": (q00q.get("cube") or {}).get("wavelength_frame")
        or q00q.get("wavelength_frame")
        or "unavailable",
        "nan_fraction_data": round(nan_frac, 4),
        "entry_point": "realigned_cube" if "realigned" in run_id else "adp",
        "note": "self-reduced, manually re-aligned cube (WP-1 exp_align fix)" if "realigned" in run_id else "",
    }

    stages_executed = [
        {"stage": s["stage"], "status": s["status"], "issues": int(s.get("issue_count", 0))}
        for s in rs["stages"]
    ]

    m5 = q00q.get("m5_stat", {})
    stat_verdict = {
        "usable": m5.get("status") == "green",
        "factor": m5.get("factor_spaxel_median"),
        "status": m5.get("status"),
        "source": "stage00q_qc.m5_stat",
        "note": m5.get("note", "STAT underestimates aperture noise; chain uses empirical noise (repo plan B)."),
    }

    hc = rs.get("hash_chain", {})
    hash_chain_ok = hc.get("status") == "pass"

    # Legacy comparison vs the ADP substitute spectrum (provenance sanity check).
    # Targets without an archival ADP (e.g. ROXs 42B b; A4 V4/V5 unavailable) run
    # with no_legacy=True: the comparison is recorded as unavailable, not applicable
    # (a cross-target comparison would flag every band meaninglessly).
    if no_legacy or not legacy_run:
        n_flagged = None
        legacy_block = {
            "table": None,
            "n_bands": len(LEGACY_BANDS),
            "n_flagged": None,
            "legacy_run": None,
            "status": "unavailable",
            "reason": "No archival ADP for this target (A4 V4/V5 unavailable); legacy provenance comparison not applicable.",
        }
        legacy_issue = {"issue": "Legacy provenance comparison unavailable (no archival ADP for this target).", "priority": "minor"}
    else:
        new_wave, new_flux = _load_spectrum(run_dir)
        leg_wave, leg_flux = _load_spectrum(root / "runs" / legacy_run)
        bands = [(lo, hi) for _, lo, hi in LEGACY_BANDS]
        leg_rows = legacy_comparison(new_wave, new_flux, leg_wave, leg_flux, bands, sigma_threshold=2.0)
        with open(tables_dir / "g0_legacy_comparison.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["band", "band_lo_A", "band_hi_A", "flux_new", "flux_legacy", "ratio_new_over_legacy", "flagged"])
            for (name, lo, hi), r in zip(LEGACY_BANDS, leg_rows):
                w.writerow([name, lo, hi, r["flux_new"], r["flux_legacy"], r["ratio_new_over_legacy"], r["flagged"]])
        n_flagged = int(sum(1 for r in leg_rows if r["flagged"]))
        legacy_block = {
            "table": "tables/g0_legacy_comparison.csv",
            "n_bands": len(LEGACY_BANDS),
            "n_flagged": n_flagged,
            "legacy_run": legacy_run,
        }
        legacy_issue = {"issue": f"Legacy comparison: {n_flagged}/{len(LEGACY_BANDS)} bands flagged vs {legacy_run} (band-integrated flux; psffit continuum can be negative, so ratios are documentary).", "priority": "minor"}

    open_issues = [
        {"issue": "Run reproduced retroactively via scripts/run_g0.py + run_g1.py over an existing stage-* run (not a fresh phase-g0 branch); G0 fulfilled in substance.", "priority": "major"},
        legacy_issue,
    ]
    # Leido de A4/M5 y de F1, no escrito: hasta el 2026-09-26 decia siempre «STAT red» y
    # «provisional until A-block closed» (texto de julio; el bloque A se cerro el 07-19),
    # aunque F1 estuviera en verde.
    if not stat_verdict["usable"]:
        open_issues.append({"issue": f"A4 M5 STAT {stat_verdict['status'] or 'absent'} -> empirical noise used across X01-X11 (repo plan B).", "priority": "major"})
    f1_status = rs.get("overall_status")
    if f1_status != "green":
        f1_blocking = [i for i in rs.get("open_issues") or [] if i.get("priority") == "blocking"]
        stages = sorted({str(i.get("stage")) for i in f1_blocking})
        open_issues.append({
            "issue": (f"F1 overall_status={f1_status}: {len(f1_blocking)} blocking issue(s) in the F1 gate"
                      + (f" ({', '.join(stages)})" if stages else "") + " (spec G0 V2)."),
            "priority": "blocking",
            # Repite el estado de F1: G5 hereda los de F1 directamente y no lo cuenta dos veces.
            "restates": "F1"})

    if pytest_after is None:
        pytest_after = {"passed": None, "failed": None, "note": "not captured by this reproduction runner."}

    qc = build_g0_qc(
        run_id,
        input_cube=input_cube,
        stages_executed=stages_executed,
        stat_verdict=stat_verdict,
        hash_chain_ok=hash_chain_ok,
        pytest_before={"passed": None, "failed": None, "note": "reproduction runner; baseline not captured here."},
        pytest_after=pytest_after,
        frozen_criteria_untouched=True,
        open_issues=open_issues,
    )

    # G0 §5.3 legacy comparison summary + §4 hash-chain block (as in the ADP QC).
    qc["legacy_comparison"] = legacy_block
    # Cada check de F1 compara el sha que una etapa declara con el del fichero en disco:
    # son comprobaciones por producto, no eslabones de una cadena. Hasta el 2026-10-20
    # se encadenaban con `verify_hash_chain` (la entrada de cada uno contra la salida
    # del anterior), lo que comparaba ficheros sin relacion (la fuente de pesos de A1
    # contra el cubo de A3) y daba `fail` con F1 en `pass` y `hash_chain_ok = true`.
    qc["hash_chain"] = {
        "status": hc.get("status", "unavailable"),
        "source": "report/run_summary.json (F1: declared sha256 vs file on disk, per product)",
        "checks": [{"stage": chk.get("stage"), "context": chk.get("context"),
                    "sha256": chk.get("actual"), "status": chk.get("status")}
                   for chk in hc.get("checks", [])],
    }

    (stage_dir / "stage_g0_qc.json").write_text(json.dumps(qc, indent=2))
    return qc


def main(argv=None):
    ap = argparse.ArgumentParser(description="G0 real-cube-execution QC runner.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--legacy-run", default=None,
                    help="Run de referencia para la comparación legacy. Sin él la comparación se registra como no disponible; antes traía un default fijo al primer objeto, que era el equivocado para cualquier otro target.")
    ap.add_argument("--no-legacy", action="store_true",
                    help="Record the legacy provenance comparison as unavailable "
                         "(targets without an archival ADP, e.g. ROXs 42B b).")
    args = ap.parse_args(argv)
    qc = build_g0(args.run_id, project_root=args.project_root,
                  legacy_run=args.legacy_run, no_legacy=args.no_legacy)
    print("hash_chain_ok:", qc["hash_chain_ok"], "| stat:", qc["stat_verdict"]["status"],
          "| legacy flagged:", qc["legacy_comparison"]["n_flagged"], "/", qc["legacy_comparison"]["n_bands"])


if __name__ == "__main__":
    main()
