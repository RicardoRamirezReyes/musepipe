"""Pure planning helpers for per-exposure MUSE scipost and alignment review.

P2 deliberately stops before executing EsoRex. A review of per-exposure
products is required before any automatic or manual OFFSET_LIST is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .esorex_driver import RecipePlan, SofEntry, classify_fits, sha256_file, write_sof


class PerExposurePlanError(RuntimeError):
    """Raised when the P2 input contract is incomplete or inconsistent."""


@dataclass(frozen=True)
class ExposureInput:
    """The 24 pixtables belonging to one raw science exposure."""

    exposure_id: str
    night: str
    pixtables: tuple[Path, ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class PerExposureScipostPlan:
    """Validated scipost SOF plans grouped by exposure."""

    run_id: str
    expected_ifus: int
    save: str
    exposures: tuple[ExposureInput, ...]
    recipes: Mapping[str, RecipePlan]


def _load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PerExposurePlanError(f"Expected a JSON object: {path}")
    return payload


def load_exposures(path: str | Path, *, run_id: str) -> tuple[int, tuple[ExposureInput, ...]]:
    """Load an explicit exposure-to-pixtable mapping without filename inference."""
    payload = _load_json(path)
    if payload.get("schema_version") != 1:
        raise PerExposurePlanError("exposures JSON must use schema_version=1")
    if payload.get("run_id") != run_id:
        raise PerExposurePlanError("exposures JSON run_id does not match the requested run")
    expected_ifus = payload.get("expected_ifus")
    if not isinstance(expected_ifus, int) or expected_ifus < 1:
        raise PerExposurePlanError("expected_ifus must be a positive integer")
    raw_exposures = payload.get("exposures")
    if not isinstance(raw_exposures, list) or not raw_exposures:
        raise PerExposurePlanError("exposures must be a non-empty list")
    exposures: list[ExposureInput] = []
    seen_ids: set[str] = set()
    for raw in raw_exposures:
        if not isinstance(raw, dict):
            raise PerExposurePlanError("every exposure must be an object")
        exposure_id = raw.get("id")
        night = raw.get("night")
        pixtables = raw.get("pixtables")
        if not isinstance(exposure_id, str) or not exposure_id:
            raise PerExposurePlanError("every exposure needs a non-empty id")
        if exposure_id in seen_ids:
            raise PerExposurePlanError(f"duplicate exposure id: {exposure_id}")
        if not isinstance(night, str) or not night:
            raise PerExposurePlanError(f"{exposure_id}: missing night")
        if not isinstance(pixtables, list) or not all(isinstance(item, str) for item in pixtables):
            raise PerExposurePlanError(f"{exposure_id}: pixtables must be a string list")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise PerExposurePlanError(f"{exposure_id}: metadata must be an object")
        seen_ids.add(exposure_id)
        exposures.append(ExposureInput(exposure_id, night, tuple(Path(item) for item in pixtables), metadata))
    return expected_ifus, tuple(exposures)


def _night_calibrations(payload: Mapping[str, object], night: str) -> Mapping[str, Sequence[str]]:
    groups = payload.get("night_calibrations")
    if not isinstance(groups, dict) or night not in groups or not isinstance(groups[night], dict):
        raise PerExposurePlanError(f"missing calibrations for night {night}")
    return groups[night]


def _require_paths(calibrations: Mapping[str, Sequence[str]], tag: str, count: int | None = None) -> list[Path]:
    paths = calibrations.get(tag)
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise PerExposurePlanError(f"missing {tag} calibration paths")
    if count is not None and len(paths) != count:
        raise PerExposurePlanError(f"expected {count} {tag} paths, got {len(paths)}")
    return [Path(path) for path in paths]


def build_scipost_plans(
    *,
    run_id: str,
    exposures: Sequence[ExposureInput],
    expected_ifus: int,
    calibration_payload: Mapping[str, object],
    save: str,
) -> PerExposureScipostPlan:
    """Build one scipost RecipePlan per explicitly declared exposure."""
    if save not in {"cube", "cube,skymodel"}:
        raise PerExposurePlanError("save must be 'cube' or 'cube,skymodel'")
    recipes: dict[str, RecipePlan] = {}
    for exposure in exposures:
        if len(exposure.pixtables) != expected_ifus:
            raise PerExposurePlanError(
                f"{exposure.exposure_id}: expected {expected_ifus} PIXTABLE_OBJECT, got {len(exposure.pixtables)}"
            )
        calibrations = _night_calibrations(calibration_payload, exposure.night)
        entries = [SofEntry(path, "PIXTABLE_OBJECT") for path in exposure.pixtables]
        entries.extend(SofEntry(path, "STD_RESPONSE") for path in _require_paths(calibrations, "STD_RESPONSE", 1))
        entries.extend(SofEntry(path, "STD_TELLURIC") for path in _require_paths(calibrations, "STD_TELLURIC", 1))
        entries.extend(SofEntry(path, "EXTINCT_TABLE") for path in _require_paths(calibrations, "EXTINCT_TABLE", 1))
        entries.extend(SofEntry(path, "FILTER_LIST") for path in _require_paths(calibrations, "FILTER_LIST", 1))
        lsf_paths = _require_paths(calibrations, "LSF_PROFILE")
        if len(lsf_paths) not in {1, expected_ifus}:
            raise PerExposurePlanError(
                f"expected 1 merged or {expected_ifus} per-IFU LSF_PROFILE paths, got {len(lsf_paths)}"
            )
        entries.extend(SofEntry(path, "LSF_PROFILE") for path in lsf_paths)
        for optional_tag in ("ASTROMETRY_WCS", "SKY_LINES"):
            for path in _require_paths(calibrations, optional_tag) if optional_tag in calibrations else []:
                entries.append(SofEntry(path, optional_tag))
        recipes[exposure.exposure_id] = RecipePlan("muse_scipost", tuple(entries), params={"save": save})
    return PerExposureScipostPlan(run_id, expected_ifus, save, tuple(exposures), recipes)


def validate_scipost_plan(plan: PerExposureScipostPlan) -> None:
    """Check current FITS products against their declared SOF tags."""
    for exposure_id, recipe in plan.recipes.items():
        counts: dict[str, int] = {}
        for entry in recipe.entries:
            if not entry.path.exists() or entry.path.stat().st_size == 0:
                raise PerExposurePlanError(f"{exposure_id}: missing or empty {entry.path}")
            actual = classify_fits(entry.path, checksum=False).tag
            if actual != entry.tag:
                raise PerExposurePlanError(
                    f"{exposure_id}: {entry.path} has tag {actual}, expected {entry.tag}"
                )
            counts[entry.tag] = counts.get(entry.tag, 0) + 1
        if counts.get("PIXTABLE_OBJECT") != plan.expected_ifus:
            raise PerExposurePlanError(f"{exposure_id}: invalid pixtable count")
        if counts.get("STD_RESPONSE") != 1 or counts.get("STD_TELLURIC") != 1:
            raise PerExposurePlanError(f"{exposure_id}: response/telluric gate failed")
        if counts.get("EXTINCT_TABLE") != 1 or counts.get("FILTER_LIST") != 1:
            raise PerExposurePlanError(f"{exposure_id}: extinction/filter gate failed")
        if counts.get("LSF_PROFILE") not in {1, plan.expected_ifus}:
            raise PerExposurePlanError(f"{exposure_id}: invalid LSF count")


def write_scipost_plan(plan: PerExposureScipostPlan, output_dir: str | Path) -> Path:
    """Write SOFs plus an awaiting-review manifest. This function never executes EsoRex."""
    root = Path(output_dir)
    sof_dir = root / "sof"
    sof_hashes: dict[str, str] = {}
    for exposure_id, recipe in plan.recipes.items():
        sof = write_sof(recipe, sof_dir / f"muse_scipost_{exposure_id}.sof")
        sof_hashes[exposure_id] = sha256_file(sof)
    payload = {
        "schema_version": 1,
        "run_id": plan.run_id,
        "expected_ifus": plan.expected_ifus,
        "save": plan.save,
        "status": "awaiting_perexp_scipost",
        "exposures": [
            {
                "id": exposure.exposure_id,
                "night": exposure.night,
                "pixtables": [str(path) for path in exposure.pixtables],
                "metadata": dict(exposure.metadata),
                "sof": str(sof_dir / f"muse_scipost_{exposure.exposure_id}.sof"),
                "sof_sha256": sof_hashes[exposure.exposure_id],
            }
            for exposure in plan.exposures
        ],
        "alignment_checkpoint": {
            "status": "blocked",
            "reason": "Run scipost, validate all IMAGE_FOV products, then review alignment before offsets or combination.",
        },
    }
    manifest = root / "perexp_scipost_plan.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_alignment_plan(image_fov_by_exposure: Mapping[str, str | Path]) -> RecipePlan:
    """Create, but do not execute, the align SOF after all per-exposure images exist."""
    if len(image_fov_by_exposure) < 2:
        raise PerExposurePlanError("muse_exp_align requires at least two IMAGE_FOV products")
    entries = tuple(SofEntry(Path(path), "IMAGE_FOV") for _, path in sorted(image_fov_by_exposure.items()))
    for entry in entries:
        if not entry.path.exists() or entry.path.stat().st_size == 0:
            raise PerExposurePlanError(f"missing or empty IMAGE_FOV: {entry.path}")
        actual = classify_fits(entry.path, checksum=False).tag
        if actual != entry.tag:
            raise PerExposurePlanError(f"{entry.path} has tag {actual}, expected {entry.tag}")
    return RecipePlan("muse_exp_align", entries)


def write_alignment_review(plan: RecipePlan, output_dir: str | Path) -> Path:
    """Write an alignment SOF and an explicit human-review checkpoint."""
    root = Path(output_dir)
    sof = write_sof(plan, root / "sof" / "muse_exp_align.sof")
    payload = {
        "status": "awaiting_alignment_review",
        "align_sof": str(sof),
        "align_sof_sha256": sha256_file(sof),
        "required_decision": ["approved_auto_offsets", "approved_manual_offsets", "rejected"],
        "required_review": ["PREVIEW_FOV", "SOURCE_LIST", "OFFSET_LIST", "spurious-source check"],
    }
    output = root / "alignment_review.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
