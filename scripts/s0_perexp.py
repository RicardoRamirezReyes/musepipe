#!/usr/bin/env python3
"""S0 per-exposure: run the wavesol offset map on each of the 7 per-exposure
cubes to CONFIRM the G1 closure (docs/decision_g1_wavesol_2026-07-17.md).

The combined cube is blind to slicer stripes because the field rotates ~5.9 deg
between the 7 exposures (S0 combined showed no structure). Each single-exposure
cube has ONE fixed slicer geometry, so if per-slice wavelength stripes exist
they should now appear as slicer-aligned structure in the per-exposure offset
map. Since ABSROT is only -16..+3 deg from vertical, one `--orientation vertical`
run per cube reports both the vertical (structure) and horizontal (transverse)
significances, covering both cardinal axes.

Runs `musepipe.qc.wavesol_map` on exp{1..7}/DATACUBE_FINAL.fits (products next to
each cube), then aggregates a comparison table vs the combined-cube S0 and prints
a verdict. Idempotent (skips a cube whose QC already exists unless --force).

    python scripts/s0_perexp.py [--force]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
PEREXP = Path("/mnt/2TB/MUSE_work/ROXs12b_perexp")
COMBINED_QC = ROOT / "runs" / "ROXs12b_realigned" / "stages" / "stageS0_qc.json"
SUMMARY_CSV = ROOT / "runs" / "ROXs12b_realigned" / "tables" / "s0_perexp_summary.csv"
N_EXP = 7
SIG_THRESHOLD = 3.0  # same as evaluate_gate_g1


def _absrot(cube: Path) -> float:
    with fits.open(cube) as h:
        hdr = h[0].header
        for k in ("HIERARCH ESO ADA ABSROT START", "HIERARCH ESO ADA ABSROT END",
                  "ESO ADA ABSROT START"):
            if k in hdr:
                return float(hdr[k])
    return float("nan")


def _run_one(i: int, force: bool) -> None:
    d = PEREXP / f"exp{i}"
    cube = d / "DATACUBE_FINAL.fits"
    qc = d / "stageS0_qc.json"
    if qc.exists() and not force:
        print(f"exp{i}: S0 QC exists, skipping (use --force to redo)", flush=True)
        return
    cmd = [sys.executable, "-m", "musepipe.qc.wavesol_map",
           "--cube", str(cube), "--qc-output", str(qc),
           "--map-output", str(d / "stageS0_offset_map.fits"),
           "--plot-output", str(d / "stageS0.png"),
           "--orientation", "vertical", "--skip-checksum"]
    print(f"exp{i}: running S0 ...", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _verdict(sig: float, transv: float) -> str:
    """Slicer structure in EITHER cardinal axis (>3x and >2x the other)."""
    if sig > SIG_THRESHOLD and sig > 2.0 * transv:
        return "vertical-stripes"
    if transv > SIG_THRESHOLD and transv > 2.0 * sig:
        return "horizontal-stripes"
    return "none"


def aggregate() -> None:
    rows = []
    for i in range(1, N_EXP + 1):
        qc = PEREXP / f"exp{i}" / "stageS0_qc.json"
        if not qc.exists():
            print(f"exp{i}: no QC, skipped in aggregate", flush=True)
            continue
        q = json.loads(qc.read_text())
        m = q["metrics"]
        absrot = _absrot(PEREXP / f"exp{i}" / "DATACUBE_FINAL.fits")
        rows.append({
            "exposure": f"exp{i}",
            "absrot_deg": round(absrot, 2),
            "p95_abs_offset_A": round(m["p95_abs_offset_A"], 4),
            "stripe_sig_vertical": round(m["structure_significance"], 2),
            "transverse_sig_horizontal": round(m["transverse_significance"], 2),
            "n_low_err": m["n_selected_low_err"],
            "verdict": _verdict(m["structure_significance"], m["transverse_significance"]),
        })

    # combined-cube reference
    if COMBINED_QC.exists():
        cq = json.loads(COMBINED_QC.read_text())["metrics"]
        rows.append({
            "exposure": "combined",
            "absrot_deg": "",
            "p95_abs_offset_A": round(cq["p95_abs_offset_A"], 4),
            "stripe_sig_vertical": round(cq["structure_significance"], 2),
            "transverse_sig_horizontal": round(cq["transverse_significance"], 2),
            "n_low_err": cq["n_selected_low_err"],
            "verdict": _verdict(cq["structure_significance"], cq["transverse_significance"]),
        })

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== S0 per-exposure summary ===")
    hdr = ("exposure", "ABSROT", "p95[A]", "stripe_v", "transv_h", "n_low_err", "verdict")
    print("{:>9} {:>7} {:>7} {:>9} {:>9} {:>9}  {}".format(*hdr))
    for r in rows:
        print("{:>9} {:>7} {:>7} {:>9} {:>9} {:>9}  {}".format(
            r["exposure"], str(r["absrot_deg"]), r["p95_abs_offset_A"],
            r["stripe_sig_vertical"], r["transverse_sig_horizontal"],
            r["n_low_err"], r["verdict"]))
    per = [r for r in rows if r["exposure"] != "combined"]
    n_stripes = sum(1 for r in per if r["verdict"] != "none")
    print(f"\nExposures with slicer-aligned structure: {n_stripes}/{len(per)}")
    print(f"table -> {SUMMARY_CSV}")
    _write_summary_figure(rows)


def _write_summary_figure(rows) -> None:
    import os

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [r["exposure"] for r in rows]
    v = [r["stripe_sig_vertical"] for r in rows]
    h = [r["transverse_sig_horizontal"] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.2, v, 0.4, label="stripe sig (vertical)", color="C0")
    ax.bar(x + 0.2, h, 0.4, label="transverse control (horizontal)", color="0.7")
    ax.axhline(SIG_THRESHOLD, color="tab:red", ls="--", label=f"stripe threshold {SIG_THRESHOLD:g}x")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("structure significance (x noise)")
    ax.set_title("S0 per-exposure: no cube exceeds the stripe threshold -> slicer stripes ruled out")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = PEREXP / "s0_perexp_summary.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"figure -> {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S0 per-exposure confirmation of the G1 closure.")
    ap.add_argument("--force", action="store_true", help="Re-run S0 even if a cube's QC exists.")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args(argv)

    if not args.aggregate_only:
        for i in range(1, N_EXP + 1):
            _run_one(i, args.force)
    aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
