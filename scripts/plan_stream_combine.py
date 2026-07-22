#!/usr/bin/env python3
"""Plan a streaming exposure combine without writing the science cube.

Measures the primary centroid in every per-exposure cube, checks that the cubes
share one spectral grid and sky orientation, and writes an auditable plan JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musepipe.config import load_run_config  # noqa: E402
from musepipe.reduction.stream_combine import (  # noqa: E402
    DEFAULT_CHUNK_CHANNELS,
    DEFAULT_PAD,
    DEFAULT_SIGCLIP_K,
    DEFAULT_SIGCLIP_MIN_N,
    StreamCombineError,
    build_stream_combine_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan a streaming muse_exp_combine replacement.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--output", help="Combined cube path (default: runs/<RUN_ID>/cube_telcorr.fits)")
    parser.add_argument("--plan-json", help="Plan path (default: runs/<RUN_ID>/stages/stream_combine_plan.json)")
    parser.add_argument("--crop-npix", type=int, help="Default: crop_npix from the run config")
    parser.add_argument("--pad", type=int, default=DEFAULT_PAD)
    parser.add_argument("--chunk-channels", type=int, default=DEFAULT_CHUNK_CHANNELS)
    parser.add_argument("--method", choices=("mean", "sigclip"), default="mean")
    parser.add_argument("--sigclip-k", type=float, default=DEFAULT_SIGCLIP_K)
    parser.add_argument("--sigclip-min-n", type=int, default=DEFAULT_SIGCLIP_MIN_N)
    parser.add_argument("--weight", choices=("exptime", "none"), default="exptime")
    parser.add_argument(
        "--centering",
        choices=("peak", "maoppy"),
        help="Centroid estimator (default: centering_method from the run config)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing plan JSON")
    args = parser.parse_args(argv)

    run_config = load_run_config(args.run_id, project_root=args.project_root)
    config = run_config.config
    if config.get("run_id") != args.run_id:
        raise StreamCombineError("run_id does not match its config")
    cube_files = config.get("cube_files")
    if not isinstance(cube_files, list) or len(cube_files) < 2:
        raise StreamCombineError("config has no cube_files list with at least two cubes")
    missing = [f for f in cube_files if not Path(f).exists()]
    if missing:
        raise StreamCombineError(f"{len(missing)} cube_files do not exist, first: {missing[0]}")

    paths = run_config.paths
    output = Path(args.output) if args.output else paths.run_dir / "cube_telcorr.fits"
    plan_json = Path(args.plan_json) if args.plan_json else paths.stage_dir / "stream_combine_plan.json"
    if plan_json.exists() and not args.overwrite:
        raise StreamCombineError(f"refusing to overwrite {plan_json} (use --overwrite)")
    if output.exists():
        raise StreamCombineError(f"combined cube already exists: {output}")

    def progress(index: int, total: int, path: str) -> None:
        print(f"[{index + 1:2d}/{total}] centroid {Path(path).parent.name}", flush=True)

    plan = build_stream_combine_plan(
        cube_files,
        run_id=args.run_id,
        target_name=config.get("target_name", args.run_id),
        output=str(output),
        crop_npix=int(args.crop_npix or config.get("crop_npix", 170)),
        pad=int(args.pad),
        data_ext=int(config.get("data_ext", 1)),
        stat_ext=str(config.get("stat_ext", "STAT")),
        chunk_channels=int(args.chunk_channels),
        method=args.method,
        sigclip_k=float(args.sigclip_k),
        sigclip_min_n=int(args.sigclip_min_n),
        weight_mode=args.weight,
        drop_wave_min_A=float(config.get("drop_wave_min_A", 5780.0)),
        drop_wave_max_A=float(config.get("drop_wave_max_A", 6050.0)),
        centering_method=str(args.centering or config.get("centering_method", "maoppy")),
        progress=progress,
    )

    plan_json.parent.mkdir(parents=True, exist_ok=True)
    plan_json.write_text(json.dumps(plan.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    shifts = [(exp.shift_y, exp.shift_x) for exp in plan.exposures]
    print(f"\nPlanned {len(plan.exposures)} exposures -> {output}")
    print(f"  crop {plan.crop_npix} px, primary at pixel index {plan.crop_npix // 2}")
    print(f"  method={plan.method} weight={plan.weight_mode}")
    print(f"  sub-pixel shifts |dy|<={max(abs(s[0]) for s in shifts):.3f} "
          f"|dx|<={max(abs(s[1]) for s in shifts):.3f}")
    repeatability = plan.reference["alignment_repeatability"]
    if repeatability["n_groups"]:
        print(f"  centroid repeatability: {repeatability['centroid_repeatability_px']:.4f} px "
              f"over {repeatability['n_groups']} repeated dither positions")
    print(f"  primary sky scatter: {plan.reference['ra_scatter_arcsec']:.3f}\" RA, "
          f"{plan.reference['dec_scatter_arcsec']:.3f}\" Dec "
          f"({plan.reference['astrometry_groups']['n_groups']} pointing groups)")
    for warning in plan.warnings:
        print(f"  WARNING: {warning}")
    print(f"  plan -> {plan_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, StreamCombineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
