#!/usr/bin/env python3
"""Execute a reviewed per-exposure scipost plan with product gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musepipe.reduction.esorex_driver import classify_fits, parse_log_warnings, sha256_file  # noqa: E402


class PerExposureExecutionError(RuntimeError):
    """Raised when a per-exposure scipost run cannot safely continue."""


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PerExposureExecutionError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _products(output_dir: Path) -> dict[str, list[str]]:
    products: dict[str, list[str]] = {}
    for path in sorted(output_dir.glob("*.fits")):
        if path.stat().st_size == 0:
            continue
        tag = classify_fits(path, checksum=False).tag
        products.setdefault(tag, []).append(str(path))
    return products


def _gate(products: dict[str, list[str]], exposure_id: str, save: str) -> None:
    expected = {"PIXTABLE_REDUCED": 1} if save == "individual" else {"DATACUBE_FINAL": 1, "IMAGE_FOV": 1, "SKY_SPECTRUM": 1}
    failures = [f"{tag}={len(products.get(tag, []))}" for tag, count in expected.items() if len(products.get(tag, [])) != count]
    if failures:
        raise PerExposureExecutionError(f"{exposure_id}: product gate failed ({', '.join(failures)})")


def _lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise PerExposureExecutionError(f"another scipost execution owns {path}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute validated P2 scipost SOFs one exposure at a time.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", help="Execution root; defaults to the plan directory.")
    parser.add_argument("--esorex", default="esorex")
    parser.add_argument("--save", choices=["cube,skymodel", "individual"], default="cube,skymodel")
    parser.add_argument("--execute", action="store_true", help="Required to invoke EsoRex.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exclude", action="append", default=[], help="Exposure id approved for exclusion; repeat as needed.")
    args = parser.parse_args(argv)
    if not args.execute:
        raise PerExposureExecutionError("pass --execute to invoke EsoRex")
    if not shutil.which(args.esorex):
        raise PerExposureExecutionError(f"esorex not found: {args.esorex}")
    config = _load_json(ROOT / "runs" / args.run_id / "config" / "config.json")
    if config.get("config", {}).get("run_id") != args.run_id:
        raise PerExposureExecutionError("run_id does not match its config")
    plan_path = Path(args.plan)
    plan = _load_json(plan_path)
    if plan.get("run_id") != args.run_id or plan.get("status") != "awaiting_perexp_scipost":
        raise PerExposureExecutionError("plan is not a pending P2 scipost plan for this run")
    exposures = plan.get("exposures")
    if not isinstance(exposures, list) or not exposures:
        raise PerExposureExecutionError("plan has no exposures")
    planned_ids = {exposure.get("id") for exposure in exposures if isinstance(exposure, dict)}
    unknown_exclusions = sorted(set(args.exclude) - planned_ids)
    if unknown_exclusions:
        raise PerExposureExecutionError(f"unknown exposure exclusions: {', '.join(unknown_exclusions)}")
    root = Path(args.output_dir) if args.output_dir else plan_path.parent
    root.mkdir(parents=True, exist_ok=True)
    execution_path = root / "perexp_scipost_execution.json"
    if execution_path.exists() and not args.resume:
        raise PerExposureExecutionError(f"execution manifest exists: {execution_path}; pass --resume after review")
    execution = _load_json(execution_path) if execution_path.exists() else {"run_id": args.run_id, "plan": str(plan_path), "save": args.save, "exposures": {}}
    if execution.get("save") != args.save:
        raise PerExposureExecutionError("execution save mode does not match the requested mode")
    completed = execution.get("exposures")
    if not isinstance(completed, dict):
        raise PerExposureExecutionError("execution manifest has invalid exposures")
    lock_path = root / ".run_perexp_scipost.lock"
    descriptor = _lock(lock_path)
    try:
        os.write(descriptor, json.dumps({"pid": os.getpid(), "created_utc": datetime.now(timezone.utc).isoformat()}).encode("utf-8"))
        execution["status"] = "running"
        _write_json(execution_path, execution)
        for exposure in exposures:
            if not isinstance(exposure, dict):
                raise PerExposureExecutionError("plan contains an invalid exposure")
            exposure_id = exposure.get("id")
            sof = exposure.get("sof")
            expected_hash = exposure.get("sof_sha256")
            if not isinstance(exposure_id, str) or not isinstance(sof, str) or not isinstance(expected_hash, str):
                raise PerExposureExecutionError("plan exposure lacks id, SOF, or SOF hash")
            if exposure_id in args.exclude:
                completed[exposure_id] = {
                    "status": "excluded",
                    "reason": "User-approved exclusion after muse_sky_lines_fit did not converge with skymethod=model.",
                    "partial_output_dir": str(root / "products" / exposure_id),
                }
                _write_json(execution_path, execution)
                continue
            sof_path = Path(sof)
            if not sof_path.exists() or sha256_file(sof_path) != expected_hash:
                raise PerExposureExecutionError(f"{exposure_id}: SOF changed since planning")
            output_dir = root / "products" / exposure_id
            previous = completed.get(exposure_id)
            if isinstance(previous, dict) and previous.get("status") == "complete":
                products = _products(output_dir)
                _gate(products, exposure_id, args.save)
                continue
            if output_dir.exists() and any(output_dir.iterdir()):
                if args.resume:
                    log_path = output_dir / "muse_scipost.log"
                    if not log_path.exists():
                        raise PerExposureExecutionError(f"{exposure_id}: existing output has no log for recovery")
                    parsed = parse_log_warnings(log_path.read_text(encoding="utf-8", errors="replace"))
                    if parsed["errors"]:
                        raise PerExposureExecutionError(f"{exposure_id}: existing output log contains errors")
                    products = _products(output_dir)
                    _gate(products, exposure_id, args.save)
                    completed[exposure_id] = {
                        "status": "complete",
                        "recovered_after_gate_update": True,
                        "log": str(log_path),
                        "warnings": parsed["warnings"],
                        "elevated_warnings": parsed["elevated"],
                        "products": products,
                    }
                    _write_json(execution_path, execution)
                    continue
                raise PerExposureExecutionError(f"{exposure_id}: output directory is not empty; inspect before retrying")
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "muse_scipost.log"
            command = [args.esorex, f"--log-dir={output_dir}", "muse_scipost", f"--save={args.save}", str(sof_path)]
            started = time.monotonic()
            with log_path.open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=output_dir, stdout=handle, stderr=subprocess.STDOUT, check=False)
            duration_s = round(time.monotonic() - started, 3)
            esorex_log = output_dir / ".logfile"
            if esorex_log.exists():
                shutil.copyfile(esorex_log, log_path)
            parsed = parse_log_warnings(log_path.read_text(encoding="utf-8", errors="replace"))
            if result.returncode != 0 or parsed["errors"]:
                raise PerExposureExecutionError(f"{exposure_id}: muse_scipost failed; inspect {log_path}")
            products = _products(output_dir)
            _gate(products, exposure_id, args.save)
            completed[exposure_id] = {
                "status": "complete",
                "command": command,
                "duration_s": duration_s,
                "log": str(log_path),
                "warnings": parsed["warnings"],
                "elevated_warnings": parsed["elevated"],
                "products": products,
            }
            _write_json(execution_path, execution)
        excluded = [exposure_id for exposure_id, result in completed.items() if isinstance(result, dict) and result.get("status") == "excluded"]
        execution["status"] = "complete_with_exclusions" if excluded else "complete"
        execution["excluded_exposures"] = sorted(excluded)
        execution["next_checkpoint"] = (
            {"status": "awaiting_exp_combine", "reason": "All PIXTABLE_REDUCED products require a reviewed OFFSET_LIST before combination."}
            if args.save == "individual"
            else {"status": "awaiting_image_fov_review", "reason": "Validate and review all IMAGE_FOV products before exp_align, offsets, or combination."}
        )
        _write_json(execution_path, execution)
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
    print(f"P2 scipost complete: {len(exposures)} exposures -> {execution_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, PerExposureExecutionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
