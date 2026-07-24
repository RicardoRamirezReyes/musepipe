#!/usr/bin/env python3
"""Auditable A1-prefix reduction grouped by observing night.

The historical global ROXs42Bb prefix remains read-only. This command builds
new per-night products from raw flats/arcs/science data while using the ESO
archive MASTER_BIAS selected by the approved exception for this dataset.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musepipe.reduction.esorex_driver import (  # noqa: E402
    RawRecord,
    RecipePlan,
    build_recipe_plan,
    classify_fits,
    parse_log_warnings,
    read_inventory_csv,
    sha256_file,
    validate_sof,
    write_sof,
)

STEP_ORDER = ["flat", "wavecal", "lsf", "scibasic_object", "scibasic_std", "standard"]
STATIC_TAGS = {
    "GEOMETRY_TABLE",
    "ASTROMETRY_WCS",
    "ASTROMETRY_REFERENCE",
    "BADPIX_TABLE",
    "VIGNETTING_MASK",
    "STD_FLUX_TABLE",
    "EXTINCT_TABLE",
    "FILTER_LIST",
    "SKY_LINES",
    "LINE_CATALOG",
}


class CascadeError(RuntimeError):
    """Raised when a provenance or product gate prevents A1 from continuing."""


def _parse_time(record: RawRecord) -> datetime | None:
    try:
        return datetime.fromisoformat(record.date_obs.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def observing_night(record: RawRecord) -> str:
    """Return the observing-night label using the approved 12:00 UTC boundary."""
    observed = _parse_time(record)
    if observed is None:
        raise CascadeError(f"Cannot derive observing night from DATE-OBS for {record.path}")
    return (observed - timedelta(hours=12)).date().isoformat()


def _median_time(records: Iterable[RawRecord]) -> datetime:
    times = sorted(time for record in records if (time := _parse_time(record)) is not None)
    if not times:
        raise CascadeError("Cannot associate calibrations without valid DATE-OBS values.")
    return times[len(times) // 2]


def _same_setup(candidate: RawRecord, science: RawRecord) -> bool:
    return candidate.ins_mode == science.ins_mode and candidate.binning == science.binning


def _is_raw(record: RawRecord, tag: str) -> bool:
    if record.tag != tag:
        return False
    if tag == "OBJECT":
        return record.dpr_catg.upper() == "SCIENCE"
    return record.dpr_catg.upper() == "CALIB"


def _nearest_date_group(records: list[RawRecord], reference: datetime) -> list[RawRecord]:
    grouped: dict[str, list[RawRecord]] = defaultdict(list)
    for record in records:
        observed = _parse_time(record)
        if observed is not None:
            grouped[observed.date().isoformat()].append(record)
    if not grouped:
        raise CascadeError("No dated calibration candidates are available.")
    return min(grouped.values(), key=lambda group: abs((_median_time(group) - reference).total_seconds()))


def _single_nearest(records: list[RawRecord], reference: datetime) -> RawRecord:
    dated = [(record, _parse_time(record)) for record in records]
    dated = [(record, observed) for record, observed in dated if observed is not None]
    if not dated:
        raise CascadeError("No dated calibration candidates are available.")
    return min(dated, key=lambda item: abs((item[1] - reference).total_seconds()))[0]


def _selected_records(records: list[RawRecord], science: list[RawRecord]) -> tuple[list[RawRecord], dict[str, object]]:
    reference = _median_time(science)
    setup = science[0]

    def raw_candidates(tag: str) -> list[RawRecord]:
        return [record for record in records if _is_raw(record, tag) and _same_setup(record, setup)]

    flats = _nearest_date_group(raw_candidates("FLAT"), reference)
    arcs = _nearest_date_group(raw_candidates("ARC"), reference)
    stds = _nearest_date_group(raw_candidates("STD"), reference)
    illum = _single_nearest(raw_candidates("ILLUM"), reference)

    # Older inventories predate the pro_catg column. The logical tag itself was
    # classified from that header, so it remains valid for those persisted CSVs.
    archive_bias = [record for record in records if record.tag == "MASTER_BIAS"]
    bias = _single_nearest(archive_bias, reference)
    static = [record for record in records if record.tag in STATIC_TAGS]

    selected = [*science, *flats, *arcs, *stds, illum, *static]
    association = {
        "observing_night": observing_night(science[0]),
        "reference_utc": reference.isoformat().replace("+00:00", "Z"),
        "science": [str(record.path) for record in science],
        "calibrations": {
            "MASTER_BIAS": [str(bias.path)],
            "FLAT": [str(record.path) for record in flats],
            "ARC": [str(record.path) for record in arcs],
            "ILLUM": [str(illum.path)],
            "STD": [str(record.path) for record in stds],
            "static": [str(record.path) for record in static],
        },
        "bias_exception": {
            "decision": "ESO archive MASTER_BIAS approved by user",
            "file": str(bias.path),
            "ins_mode": bias.ins_mode,
            "science_ins_mode": setup.ins_mode,
        },
    }
    return selected, association


def build_night_associations(records: list[RawRecord]) -> dict[str, tuple[list[RawRecord], dict[str, object]]]:
    science = [record for record in records if _is_raw(record, "OBJECT")]
    groups: dict[str, list[RawRecord]] = defaultdict(list)
    for record in science:
        groups[observing_night(record)].append(record)
    if not groups:
        raise CascadeError("The inventory has no raw OBJECT science exposures.")
    return {night: _selected_records(records, group) for night, group in sorted(groups.items())}


def _trim_scibasic(plan: RecipePlan, primary_tag: str) -> RecipePlan:
    kept: list = []
    for entry in plan.entries:
        if entry.tag in {"OBJECT", "STD"} and entry.tag != primary_tag:
            continue
        kept.append(entry)
    return RecipePlan(recipe=plan.recipe, entries=tuple(kept), params=plan.params)


def _products(outdir: Path, tag: str) -> list[str]:
    return sorted(str(path) for path in outdir.glob(f"{tag}*.fits") if path.stat().st_size > 0)


def _product_fingerprint(paths: list[str]) -> list[dict[str, object]]:
    fingerprint = []
    for path in paths:
        product = Path(path)
        if product.exists():
            fingerprint.append({"file": path, "size_bytes": product.stat().st_size, "mtime_ns": product.stat().st_mtime_ns})
        else:
            fingerprint.append({"file": path, "status": "planned"})
    return fingerprint


def _validate_available_entries(plan: RecipePlan) -> None:
    """Validate raw/static inputs during dry-run while allowing planned products."""
    available = tuple(entry for entry in plan.entries if entry.path.exists())
    if available:
        validate_sof(RecipePlan(recipe=plan.recipe, entries=available, params=plan.params))


def _fingerprint(inventory: Path, plan: RecipePlan, products: dict[str, list[str]]) -> str:
    payload = {
        "inventory_sha256": sha256_file(inventory),
        "recipe": plan.recipe,
        "sof": "\n".join(entry.render() for entry in plan.entries),
        "params": dict(plan.params or {}),
        "input_products": {tag: _product_fingerprint(paths) for tag, paths in sorted(products.items())},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_product_headers(products: dict[str, list[str]]) -> None:
    for expected_tag, paths in products.items():
        for path in paths:
            actual = classify_fits(path, checksum=False).tag
            if actual != expected_tag:
                raise CascadeError(f"{path}: expected {expected_tag}, FITS header reports {actual}.")


def _run_recipe(esorex: str, plan: RecipePlan, sof: Path, outdir: Path, log: Path) -> dict[str, object]:
    validate_sof(plan)
    write_sof(plan, sof)
    outdir.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [esorex, f"--log-dir={outdir}", plan.recipe]
    command.extend(f"--{key}={value}" for key, value in (plan.params or {}).items())
    command.append(str(sof))
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=outdir, stdout=handle, stderr=subprocess.STDOUT, check=False)
    duration_s = round(time.monotonic() - started, 3)
    esorex_log = outdir / ".logfile"
    if esorex_log.exists():
        shutil.copyfile(esorex_log, log)
    text = log.read_text(encoding="utf-8", errors="replace")
    parsed = parse_log_warnings(text)
    if result.returncode != 0 or parsed["errors"]:
        raise CascadeError(f"{plan.recipe} failed; inspect {log}")
    return {
        "name": plan.recipe,
        "status": "ok",
        "command": command,
        "sof": str(sof),
        "log": str(log),
        "esorex_log": str(esorex_log) if esorex_log.exists() else "",
        "duration_s": duration_s,
        "warnings": parsed["warnings"],
        "elevated_warnings": parsed["elevated"],
    }


def _work_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise CascadeError(f"Another cascade owns {path}; remove it only after confirming it is stale.") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "created_utc": datetime.now(timezone.utc).isoformat()}, handle)


def _verify_identity(run_id: str, raw_data_dir: Path, records: list[RawRecord]) -> None:
    if run_id == "ROXs42Bb":
        raise CascadeError("runs/ROXs42Bb is legacy and cannot be used by this cascade.")
    config_path = ROOT / "runs" / run_id / "config" / "config.json"
    config = _load_json(config_path).get("config", {})
    if config.get("run_id") != run_id:
        raise CascadeError(f"Config run_id mismatch in {config_path}")
    configured = Path(str(config.get("raw_data_dir", ""))).resolve()
    if configured != raw_data_dir.resolve():
        raise CascadeError(f"--raw-data-dir {raw_data_dir} does not match config {configured}")
    for record in records:
        try:
            record.path.resolve().relative_to(raw_data_dir.resolve())
        except ValueError as exc:
            raise CascadeError(f"Inventory path outside raw-data-dir: {record.path}") from exc


def _gates(n_object: int, n_std: int, step: str) -> dict[str, int]:
    if step == "flat":
        return {"MASTER_FLAT": 24, "TRACE_TABLE": 24}
    if step == "wavecal":
        return {"WAVECAL_TABLE": 24}
    if step == "lsf":
        return {"LSF_PROFILE": 24}
    if step == "scibasic_object":
        return {"PIXTABLE_OBJECT": n_object * 24}
    if step == "scibasic_std":
        return {"PIXTABLE_STD": n_std * 24}
    if step == "standard":
        return {"STD_RESPONSE": n_std, "STD_TELLURIC": n_std}
    raise CascadeError(f"Unknown gate step {step}")


def _write_qc(run_id: str, payload: dict[str, object]) -> None:
    path = ROOT / "runs" / run_id / "stages" / "stage00r_p0_calibration_audit.json"
    existing = _load_json(path)
    groups = existing.setdefault("groups", {})
    groups[payload["night"]] = payload
    existing.update({"stage": "00r_raw_reduction", "run_id": run_id, "policy": "per_observing_night"})
    _write_json(path, existing)


def run_night(args: argparse.Namespace, night: str, records: list[RawRecord], association: dict[str, object]) -> None:
    selected = records
    science = [record for record in selected if _is_raw(record, "OBJECT")]
    stds = [record for record in selected if _is_raw(record, "STD")]
    if not science or not stds:
        raise CascadeError(f"{night}: selected records need science and one raw standard.")
    work = Path(args.work_dir) / f"night_{night}"
    manifest_path = work / "products_manifest.json"
    lock_path = work / ".reduce_cascade.lock"
    _work_lock(lock_path)
    try:
        manifest = _load_json(manifest_path)
        if manifest and not args.resume:
            raise CascadeError(f"{manifest_path} exists; pass --resume only after checking its checkpoints.")
        products: dict[str, list[str]] = {"MASTER_BIAS": list(association["calibrations"]["MASTER_BIAS"])}
        _validate_product_headers(products)
        manifest = {
            "schema_version": 2,
            "run_id": args.run_id,
            "night": night,
            "raw_data_dir": str(Path(args.raw_data_dir).resolve()),
            "baseline": {"work_dir": "/mnt/2TB/MUSE_work/ROXs42Bb_raw_reduction", "status": "reference_only"},
            "association": association,
            "products": products,
            "recipes": [],
            "checkpoints": {},
            "status": "dry_run" if not args.execute else "running",
        }
        inventory = ROOT / "runs" / args.run_id / "raw_reduction" / "raw_inventory.csv"
        log_dir = ROOT / "runs" / args.run_id / "logs"
        product_root = work / "products"
        sof_root = work / "sof"
        for step in STEP_ORDER:
            if STEP_ORDER.index(step) > STEP_ORDER.index(args.stop_after):
                break
            recipe = "muse_scibasic" if step.startswith("scibasic") else f"muse_{step}"
            plan = build_recipe_plan(recipe, selected, products=products)
            if step == "scibasic_object":
                plan = _trim_scibasic(plan, "OBJECT")
            elif step == "scibasic_std":
                plan = _trim_scibasic(plan, "STD")
            _validate_available_entries(plan)
            sof = sof_root / f"{step}.sof"
            write_sof(plan, sof)
            fingerprint = _fingerprint(inventory, plan, products)
            gates = _gates(len(science), len(stds), step)
            outdir = product_root / step
            previous = _load_json(manifest_path).get("checkpoints", {}).get(step, {}) if args.resume else {}
            existing = {tag: _products(outdir, tag) for tag in gates}
            can_resume = previous.get("fingerprint") == fingerprint and all(len(existing[tag]) == count for tag, count in gates.items())
            if can_resume:
                _validate_product_headers(existing)
                products.update(existing)
                manifest["checkpoints"][step] = previous
                continue
            if args.resume and any(existing.values()):
                raise CascadeError(f"{night}:{step} has products that do not match its checkpoint.")
            if not args.execute:
                manifest["checkpoints"][step] = {"fingerprint": fingerprint, "status": "planned", "gates": gates, "sof": str(sof)}
                for tag, count in gates.items():
                    products[tag] = [str(outdir / "expected" / f"{tag}_{index:04d}.fits") for index in range(1, count + 1)]
                continue
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            summary = _run_recipe(args.esorex, plan, sof, outdir, log_dir / f"reduce_{night}_{step}_{timestamp}.log")
            produced = {tag: _products(outdir, tag) for tag in gates}
            for tag, count in gates.items():
                if len(produced[tag]) != count:
                    raise CascadeError(f"{night}:{step} expected {count} {tag}, found {len(produced[tag])}.")
            _validate_product_headers(produced)
            products.update(produced)
            summary["products"] = {tag: _product_fingerprint(paths) for tag, paths in produced.items()}
            manifest["recipes"].append(summary)
            manifest["checkpoints"][step] = {"fingerprint": fingerprint, "status": "complete", "gates": gates, "sof": str(sof)}
            manifest["products"] = products
            _write_json(manifest_path, manifest)
        manifest["products"] = products
        manifest["status"] = "complete" if args.execute else "dry_run"
        _write_json(manifest_path, manifest)
        _write_qc(args.run_id, {"night": night, "status": manifest["status"], "association": association, "manifest": str(manifest_path)})
    finally:
        lock_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-observing-night MUSE A1 prefix.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--night", action="append", help="Observing night label; default: all groups.")
    parser.add_argument("--stop-after", choices=STEP_ORDER, default="standard")
    parser.add_argument("--esorex", default="esorex")
    parser.add_argument("--execute", action="store_true", help="Run esorex. Without this flag only SOFs/QC are built.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and not shutil.which(args.esorex):
        raise CascadeError(f"esorex not found: {args.esorex}")
    inventory = ROOT / "runs" / args.run_id / "raw_reduction" / "raw_inventory.csv"
    if not inventory.exists():
        raise CascadeError(f"Inventory missing: {inventory}")
    records = read_inventory_csv(inventory)
    _verify_identity(args.run_id, Path(args.raw_data_dir), records)
    associations = build_night_associations(records)
    requested = args.night or list(associations)
    unknown = sorted(set(requested) - set(associations))
    if unknown:
        raise CascadeError(f"Unknown observing-night labels: {', '.join(unknown)}")
    for night in requested:
        selected, association = associations[night]
        run_night(args, night, selected, association)
        print(f"{night}: {'executed' if args.execute else 'dry-run'} complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CascadeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
