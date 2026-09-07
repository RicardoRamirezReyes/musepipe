#!/usr/bin/env python3
"""A2a: master BIAS by night vs the global master (evidence for A2b).

The realigned reduction combined all 44 raw BIAS (4 nights) into ONE global
master (open_issue in stage00r_qc.json: "Calibrations not grouped by night/mode").
This checks whether that biases the level/RON: it groups the BIAS by night (from
the global muse_bias.sof), runs `esorex muse_bias` per night into a scratch dir
(NEVER touching the master in use), then compares -- per IFU (24) and quadrant
(4) -- the pipeline QC keywords LEVEL{q} MEAN (ADU) and MASTER{q} RON against the
global master.

Criterion (for the human A2b decision): |Delta level| < 1 ADU and |Delta RON| <
5% => global grouping acceptable.

Outputs: runs/ROXs12b_raw/tables/bias_by_night_comparison.csv and a
`bias_grouping_check` block appended to stage00r_qc.json (realigned).

    python scripts/a2a_bias_by_night.py [--work-dir DIR] [--compare-only]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
# Rutas por objeto: se resuelven en main() desde --run-id y el config del run.
# Antes eran constantes de módulo fijadas al primer objeto reducido, de modo que
# cualquier otro target las heredaba en silencio (plan multi-objeto, WP-P3w).
GLOBAL_SOF = None
GLOBAL_MASTER_DIR = None
# El directorio de trabajo externo se declara con `MUSE_WORK`; sin el, se
# escribe bajo `work/` en la raiz del repo. No hay ruta de nadie cableada.
DEFAULT_WORK = Path(os.environ.get("MUSE_WORK", "work")) / "a2a_bias_by_night"
TABLE_OUT = None
STAGE00R = None
ESOREX = "esorex"
N_IFU = 24
QUADRANTS = (1, 2, 3, 4)


def _night_of(path: str) -> str:
    # filename: .../MUSE.2022-08-28T10:02:49.179.fits -> 2022-08-28
    name = Path(path).name
    return name.split("T")[0].replace("MUSE.", "")


def group_sof_by_night() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for line in GLOBAL_SOF.read_text().splitlines():
        line = line.strip()
        if not line or " " not in line:
            continue
        path, tag = line.rsplit(" ", 1)
        if tag != "BIAS":
            continue
        groups.setdefault(_night_of(path), []).append(line)
    return dict(sorted(groups.items()))


def _run_esorex_bias(sof: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "muse_bias.log"
    cmd = [ESOREX, f"--log-dir={outdir}", "muse_bias", str(sof)]
    print(f"[{time.strftime('%H:%M:%S')}] muse_bias: cwd={outdir}", flush=True)
    t0 = time.time()
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=str(outdir), stdout=fh, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] muse_bias exit={proc.returncode} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"muse_bias failed for {sof} (exit {proc.returncode}); see {log}")
    n = len(list(outdir.glob("MASTER_BIAS-*.fits")))
    if n != N_IFU:
        raise SystemExit(f"GATE: expected {N_IFU} MASTER_BIAS, got {n} in {outdir}")


def read_qc(master_dir: Path) -> dict[tuple[int, int], dict[str, float]]:
    """{(ifu, quadrant): {'ron':.., 'level':..}} from the QC header keywords."""
    out: dict[tuple[int, int], dict[str, float]] = {}
    for ifu in range(1, N_IFU + 1):
        f = master_dir / f"MASTER_BIAS-{ifu:02d}.fits"
        if not f.exists():
            raise FileNotFoundError(f)
        h = fits.getheader(f)
        for q in QUADRANTS:
            out[(ifu, q)] = {
                "ron": float(h[f"ESO QC BIAS MASTER{q} RON"]),
                "level": float(h[f"ESO QC BIAS LEVEL{q} MEAN"]),
            }
    return out


def compare(work: Path, nights: list[str]) -> tuple[list[dict], dict]:
    glob_qc = read_qc(GLOBAL_MASTER_DIR)
    rows = []
    max_level = 0.0
    max_ron_pct = 0.0
    for night in nights:
        night_qc = read_qc(work / f"night_{night}")
        for key in sorted(glob_qc):
            ifu, q = key
            g, n = glob_qc[key], night_qc[key]
            dlevel = n["level"] - g["level"]
            dron_pct = 100.0 * (n["ron"] - g["ron"]) / g["ron"] if g["ron"] else float("nan")
            rows.append({"night": night, "ifu": ifu, "quadrant": q,
                         "level_global_adu": round(g["level"], 4), "level_night_adu": round(n["level"], 4),
                         "delta_level_adu": round(dlevel, 4),
                         "ron_global": round(g["ron"], 4), "ron_night": round(n["ron"], 4),
                         "delta_ron_pct": round(dron_pct, 4)})
            max_level = max(max_level, abs(dlevel))
            max_ron_pct = max(max_ron_pct, abs(dron_pct))
    summary = {
        "n_nights": len(nights), "nights": nights, "n_ifu": N_IFU, "n_quadrants": len(QUADRANTS),
        "max_level_diff_adu": round(max_level, 4), "max_ron_diff_pct": round(max_ron_pct, 4),
        "criterion": "|delta level| < 1 ADU AND |delta RON| < 5%",
        "verdict": "global_grouping_acceptable" if (max_level < 1.0 and max_ron_pct < 5.0)
        else "review_needed",
    }
    return rows, summary


def _resolve_paths(raw_run_id: str) -> None:
    """Fija las rutas del árbol de reducción del run raw indicado."""
    global GLOBAL_SOF, GLOBAL_MASTER_DIR, TABLE_OUT
    rr = ROOT / "runs" / raw_run_id / "raw_reduction"
    global STAGE00R
    GLOBAL_SOF = rr / "sof" / "muse_bias.sof"
    GLOBAL_MASTER_DIR = rr / "products" / "muse_bias"
    TABLE_OUT = ROOT / "runs" / raw_run_id / "tables" / "bias_by_night_comparison.csv"
    STAGE00R = ROOT / "runs" / raw_run_id / "stages" / "stage00r_qc.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A2a master BIAS by night vs global.")
    ap.add_argument("--raw-run-id", required=True,
                    help="Run con el árbol raw_reduction del que salen SOF y master BIAS global.")
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK))
    ap.add_argument("--compare-only", action="store_true",
                    help="skip esorex, just compare existing per-night masters")
    args = ap.parse_args(argv)
    _resolve_paths(args.raw_run_id)
    work = Path(args.work_dir)

    groups = group_sof_by_night()
    nights = list(groups)
    print(f"BIAS by night: {[(n, len(v)) for n, v in groups.items()]}")

    if not args.compare_only:
        work.mkdir(parents=True, exist_ok=True)
        for night, lines in groups.items():
            sof = work / f"muse_bias_{night}.sof"
            sof.write_text("\n".join(lines) + "\n")
            _run_esorex_bias(sof, work / f"night_{night}")

    rows, summary = compare(work, nights)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if STAGE00R.exists():
        qc = json.loads(STAGE00R.read_text())
        qc["bias_grouping_check"] = {**summary, "table": str(TABLE_OUT.relative_to(ROOT)),
                                     "work_dir": str(work),
                                     "note": "A2a evidence for the A2b human decision; per-night "
                                             "masters computed in a scratch dir, the master in use "
                                             "was NOT replaced."}
        STAGE00R.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")

    print(f"A2a: max |delta level| = {summary['max_level_diff_adu']} ADU, "
          f"max |delta RON| = {summary['max_ron_diff_pct']}% -> {summary['verdict']}")
    print(f"table -> {TABLE_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
