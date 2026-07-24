#!/usr/bin/env python3
"""Write an alignment-review checkpoint from completed per-exposure scipost."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musepipe.reduction.perexp_plan import PerExposurePlanError, build_alignment_plan, write_alignment_review  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan exp_align without executing EsoRex.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config_path = ROOT / "runs" / args.run_id / "config" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config", {}).get("run_id") != args.run_id:
        raise PerExposurePlanError(f"config run_id mismatch: {config_path}")
    execution = json.loads(Path(args.execution).read_text(encoding="utf-8"))
    if execution.get("run_id") != args.run_id or execution.get("status") not in {"complete", "complete_with_exclusions"}:
        raise PerExposurePlanError("execution is not a completed P2 run for this run_id")
    raw_exposures = execution.get("exposures")
    if not isinstance(raw_exposures, dict):
        raise PerExposurePlanError("execution has no exposure records")
    images: dict[str, str] = {}
    for exposure_id, result in raw_exposures.items():
        if not isinstance(exposure_id, str) or not isinstance(result, dict) or result.get("status") != "complete":
            continue
        products = result.get("products")
        paths = products.get("IMAGE_FOV") if isinstance(products, dict) else None
        if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str):
            raise PerExposurePlanError(f"{exposure_id}: expected exactly one IMAGE_FOV")
        images[exposure_id] = paths[0]
    output_dir = Path(args.output_dir)
    if (output_dir / "alignment_review.json").exists() or (output_dir / "sof" / "muse_exp_align.sof").exists():
        raise PerExposurePlanError(f"refusing to overwrite existing alignment review in {output_dir}")
    review = write_alignment_review(build_alignment_plan(images), output_dir)
    print(f"Alignment review planned: {len(images)} IMAGE_FOV -> {review}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, PerExposurePlanError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
