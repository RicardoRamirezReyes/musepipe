#!/usr/bin/env python3
"""Create an auditable muse_exp_combine SOF from completed P2 products."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musepipe.reduction.esorex_driver import RecipePlan, SofEntry, classify_fits, sha256_file, write_sof  # noqa: E402


class CombinePlanError(RuntimeError):
    """Raised when P2 products cannot safely form a combine SOF."""


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CombinePlanError(f"expected JSON object: {path}")
    return payload


def _validated(path: Path, tag: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise CombinePlanError(f"missing or empty {tag}: {path}")
    if classify_fits(path, checksum=False).tag != tag:
        raise CombinePlanError(f"{path}: FITS tag is not {tag}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan muse_exp_combine without executing EsoRex.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--calibrations-json", required=True)
    parser.add_argument("--offset-list", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = _json(ROOT / "runs" / args.run_id / "config" / "config.json")
    if config.get("config", {}).get("run_id") != args.run_id:
        raise CombinePlanError("run_id does not match its config")
    execution = _json(Path(args.execution))
    if execution.get("run_id") != args.run_id or execution.get("status") not in {"complete", "complete_with_exclusions"}:
        raise CombinePlanError("execution is not complete for this run")
    raw_exposures = execution.get("exposures")
    if not isinstance(raw_exposures, dict):
        raise CombinePlanError("execution has no exposure records")
    reduced: list[Path] = []
    for exposure_id, result in sorted(raw_exposures.items()):
        if not isinstance(exposure_id, str) or not isinstance(result, dict) or result.get("status") != "complete":
            continue
        products = result.get("products")
        paths = products.get("PIXTABLE_REDUCED") if isinstance(products, dict) else None
        if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str):
            raise CombinePlanError(f"{exposure_id}: expected one PIXTABLE_REDUCED")
        path = Path(paths[0])
        _validated(path, "PIXTABLE_REDUCED")
        reduced.append(path)
    if len(reduced) < 2:
        raise CombinePlanError("need at least two PIXTABLE_REDUCED products")
    offset_list = Path(args.offset_list)
    _validated(offset_list, "OFFSET_LIST")
    calibrations = _json(Path(args.calibrations_json)).get("night_calibrations")
    if not isinstance(calibrations, dict):
        raise CombinePlanError("calibrations JSON has no night_calibrations")
    filter_paths = {Path(path) for group in calibrations.values() if isinstance(group, dict) for path in group.get("FILTER_LIST", [])}
    if len(filter_paths) != 1:
        raise CombinePlanError("expected one shared FILTER_LIST")
    filter_list = next(iter(filter_paths))
    _validated(filter_list, "FILTER_LIST")
    output = Path(args.output_dir)
    sof = output / "sof" / "muse_exp_combine_manual.sof"
    manifest = output / "exp_combine_plan.json"
    if sof.exists() or manifest.exists():
        raise CombinePlanError(f"refusing to overwrite combine plan in {output}")
    plan = RecipePlan("muse_exp_combine", tuple([*(SofEntry(path, "PIXTABLE_REDUCED") for path in reduced), SofEntry(offset_list, "OFFSET_LIST"), SofEntry(filter_list, "FILTER_LIST")]))
    write_sof(plan, sof)
    payload = {
        "run_id": args.run_id,
        "status": "awaiting_exp_combine",
        "n_exposures": len(reduced),
        "excluded_exposures": execution.get("excluded_exposures", []),
        "sof": str(sof),
        "sof_sha256": sha256_file(sof),
        "offset_list": str(offset_list),
        "offset_list_sha256": sha256_file(offset_list),
        "filter_list": str(filter_list),
        "pixtables": [str(path) for path in reduced],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Combine plan complete: {len(reduced)} PIXTABLE_REDUCED -> {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, CombinePlanError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
