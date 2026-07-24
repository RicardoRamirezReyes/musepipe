#!/usr/bin/env python3
"""Execute a streaming exposure combine from a plan produced by plan_stream_combine."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astropy.io import fits  # noqa: E402

from musepipe.reduction.esorex_driver import sha256_file  # noqa: E402
from musepipe.reduction.stream_combine import (  # noqa: E402
    StreamCombineError,
    combine_streaming,
    plan_from_dict,
    write_combined_cube,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combine per-exposure cubes one wavelength chunk at a time.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", help="Override the output cube path from the plan")
    parser.add_argument("--qc-json", help="Default: <output dir>/cube_telcorr_qc.json")
    parser.add_argument("--chunk-channels", type=int, help="Override the plan chunk size")
    parser.add_argument("--execute", action="store_true", help="Required to write the cube")
    args = parser.parse_args(argv)

    plan_path = Path(args.plan)
    plan = plan_from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
    if plan.run_id != args.run_id:
        raise StreamCombineError(f"plan is for run {plan.run_id}, not {args.run_id}")
    if args.chunk_channels:
        plan = replace(plan, chunk_channels=int(args.chunk_channels))

    output = Path(args.output or plan.output)
    if output.exists():
        raise StreamCombineError(f"refusing to overwrite {output}")
    qc_json = Path(args.qc_json) if args.qc_json else output.with_name(f"{output.stem}_qc.json")

    nz = int(plan.wavelength["n_channels"])
    npix = int(plan.crop_npix)
    stack_gb = (len(plan.exposures) * plan.chunk_channels * npix * npix * 4 * 2) / 1024**3
    print(f"plan: {len(plan.exposures)} exposures, method={plan.method}, weight={plan.weight_mode}")
    print(f"output cube: {nz} x {npix} x {npix} -> {output}")
    print(f"chunk={plan.chunk_channels} channels, peak stack ~{stack_gb:.2f} GB "
          f"({'stacked' if plan.method == 'sigclip' else 'accumulated'})")
    if not args.execute:
        print("dry run: pass --execute to write the cube")
        return 0

    with fits.open(plan.exposures[0].file, memmap=True) as hdul:
        bunit = hdul[plan.data_ext].header.get("BUNIT")
        stat_bunit = hdul[plan.stat_ext].header.get("BUNIT")

    started = time.time()

    def progress(z1: int, z2: int, total: int) -> None:
        elapsed = time.time() - started
        print(f"  channels {z1:5d}-{z2:5d} / {total} ({100.0 * z1 / total:5.1f}%) "
              f"{elapsed / 60.0:6.1f} min", flush=True)

    result = combine_streaming(plan, progress=progress)
    path = write_combined_cube(result, plan, output, bunit=bunit, stat_bunit=stat_bunit)

    qc = dict(result["qc"])
    qc.update({
        "run_id": plan.run_id,
        "target_name": plan.target_name,
        "stage": "stream_combine",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "output": str(path),
        "output_sha256": sha256_file(path),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "reference": plan.reference,
        "wavelength": plan.wavelength,
        "warnings": list(plan.warnings),
        "exposures": [
            {
                "index": exp.index,
                "exposure_id": exp.exposure_id,
                "file": exp.file,
                "exptime": exp.exptime,
                "weight": exp.weight,
                "y_center": exp.y_center,
                "x_center": exp.x_center,
                "shift_y": exp.shift_y,
                "shift_x": exp.shift_x,
                "in_bounds": exp.in_bounds,
                "centroid_fallback": exp.centroid_fallback,
            }
            for exp in plan.exposures
        ],
    })
    qc_json.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nCombined cube written: {path}")
    print(f"  finite fraction {qc['finite_fraction']:.4f}, "
          f"exposures per voxel min/median/max {qc['count_min']}/{qc['count_median']:.0f}/{qc['count_max']}")
    if plan.method == "sigclip":
        print(f"  sigma-clip rejected {qc['rejected_fraction'] * 100:.3f}% of contributions")
    print(f"  QC -> {qc_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, StreamCombineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
