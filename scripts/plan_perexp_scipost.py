#!/usr/bin/env python3
"""Create P2 per-exposure scipost SOFs and an alignment-review checkpoint.

This command is intentionally planning-only. It never invokes EsoRex.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musepipe.reduction.perexp_plan import (  # noqa: E402
    PerExposurePlanError,
    build_scipost_plans,
    load_exposures,
    validate_scipost_plan,
    write_scipost_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan P2 per-exposure scipost without executing EsoRex.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--exposures-json", required=True)
    parser.add_argument("--calibrations-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save", choices=["cube", "cube,skymodel"], default="cube,skymodel")
    args = parser.parse_args(argv)
    config_path = ROOT / "runs" / args.run_id / "config" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config", {}).get("run_id") != args.run_id:
        raise PerExposurePlanError(f"config run_id mismatch: {config_path}")
    expected_ifus, exposures = load_exposures(args.exposures_json, run_id=args.run_id)
    calibrations = json.loads(Path(args.calibrations_json).read_text(encoding="utf-8"))
    plan = build_scipost_plans(
        run_id=args.run_id,
        exposures=exposures,
        expected_ifus=expected_ifus,
        calibration_payload=calibrations,
        save=args.save,
    )
    validate_scipost_plan(plan)
    manifest = write_scipost_plan(plan, args.output_dir)
    print(f"P2 plan complete: {len(exposures)} exposures -> {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, PerExposurePlanError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
