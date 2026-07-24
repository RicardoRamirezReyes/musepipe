#!/usr/bin/env python3
"""Create explicit P2 JSON inputs from completed A1 per-night manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musepipe.reduction.perexp_inputs import PerExposureInputError, build_perexp_inputs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build explicit P2 inputs without invoking EsoRex.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", action="append", required=True, help="Completed A1 products_manifest.json; repeat per night.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config_path = ROOT / "runs" / args.run_id / "config" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config", {}).get("run_id") != args.run_id:
        raise PerExposureInputError(f"config run_id mismatch: {config_path}")
    root = Path(args.output_dir)
    exposures_path = root / "inputs" / "exposures.json"
    calibrations_path = root / "inputs" / "night_calibrations.json"
    if exposures_path.exists() or calibrations_path.exists():
        raise PerExposureInputError(f"refusing to overwrite existing P2 inputs in {root}")
    exposures, calibrations = build_perexp_inputs(args.manifest, run_id=args.run_id)
    exposures_path.parent.mkdir(parents=True, exist_ok=True)
    exposures_path.write_text(json.dumps(exposures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    calibrations_path.write_text(json.dumps(calibrations, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"P2 inputs complete: {len(exposures['exposures'])} exposures -> {root / 'inputs'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, PerExposureInputError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
