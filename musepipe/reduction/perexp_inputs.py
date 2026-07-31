"""Build explicit P2 inputs from completed per-night A1 manifests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from astropy.io import fits

from .esorex_driver import classify_fits


class PerExposureInputError(RuntimeError):
    """Raised when completed A1 products cannot form an unambiguous P2 input."""


def _load_manifest(path: Path, run_id: str) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_id") != run_id:
        raise PerExposureInputError(f"{path}: run_id does not match {run_id}")
    if payload.get("status") != "complete":
        raise PerExposureInputError(f"{path}: A1 manifest is not complete")
    return payload


def _paths(payload: Mapping[str, object], tag: str, count: int) -> list[Path]:
    products = payload.get("products")
    values = products.get(tag) if isinstance(products, dict) else None
    if not isinstance(values, list) or len(values) != count or not all(isinstance(value, str) for value in values):
        raise PerExposureInputError(f"missing {count} {tag} products")
    paths = [Path(value) for value in values]
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise PerExposureInputError(f"missing or empty {tag}: {path}")
        if classify_fits(path, checksum=False).tag != tag:
            raise PerExposureInputError(f"{path}: FITS tag is not {tag}")
    return paths


def _paths_with_allowed_counts(
    payload: Mapping[str, object], tag: str, counts: set[int]
) -> list[Path]:
    products = payload.get("products")
    values = products.get(tag) if isinstance(products, dict) else None
    if not isinstance(values, list) or len(values) not in counts or not all(isinstance(value, str) for value in values):
        expected = " or ".join(str(count) for count in sorted(counts))
        raise PerExposureInputError(f"missing {expected} {tag} products")
    paths = [Path(value) for value in values]
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise PerExposureInputError(f"missing or empty {tag}: {path}")
        if classify_fits(path, checksum=False).tag != tag:
            raise PerExposureInputError(f"{path}: FITS tag is not {tag}")
    return paths


def _static_paths(payload: Mapping[str, object], tags: set[str]) -> dict[str, list[str]]:
    association = payload.get("association")
    calibrations = association.get("calibrations") if isinstance(association, dict) else None
    static = calibrations.get("static") if isinstance(calibrations, dict) else None
    if not isinstance(static, list) or not all(isinstance(value, str) for value in static):
        raise PerExposureInputError("manifest has no static calibration paths")
    selected: dict[str, list[str]] = {tag: [] for tag in tags}
    for value in static:
        path = Path(value)
        if not path.exists() or path.stat().st_size == 0:
            raise PerExposureInputError(f"missing or empty static calibration: {path}")
        tag = classify_fits(path, checksum=False).tag
        if tag in selected:
            selected[tag].append(str(path))
    for tag in ("EXTINCT_TABLE", "FILTER_LIST"):
        if len(selected[tag]) != 1:
            raise PerExposureInputError(f"expected exactly one {tag}")
    return {tag: values for tag, values in selected.items() if values}


def build_perexp_inputs(manifest_paths: Sequence[str | Path], *, run_id: str) -> tuple[dict[str, object], dict[str, object]]:
    """Return explicit exposure and nightly-calibration JSON payloads for P2.

    Pixtables are assigned through their FITS provenance header rather than by
    parsing output filenames. The resulting JSON is the only input consumed by
    the P2 SOF planner.
    """
    exposure_rows: list[dict[str, object]] = []
    night_calibrations: dict[str, object] = {}
    expected_ifus: int | None = None
    for manifest_path in sorted(Path(value) for value in manifest_paths):
        payload = _load_manifest(manifest_path, run_id)
        night = payload.get("night")
        association = payload.get("association")
        science = association.get("science") if isinstance(association, dict) else None
        if not isinstance(night, str) or not isinstance(science, list) or not all(isinstance(value, str) for value in science):
            raise PerExposureInputError(f"{manifest_path}: missing night or science association")
        if night in night_calibrations:
            raise PerExposureInputError(f"duplicate manifest for night {night}")
        lsf_paths = _paths_with_allowed_counts(payload, "LSF_PROFILE", {1, 24})
        response_paths = _paths(payload, "STD_RESPONSE", 1)
        telluric_paths = _paths(payload, "STD_TELLURIC", 1)
        pixtable_paths = _paths(payload, "PIXTABLE_OBJECT", len(science) * 24)
        if expected_ifus is None:
            expected_ifus = 24
        elif expected_ifus != 24:
            raise PerExposureInputError("nights disagree on the number of IFUs")
        by_raw_name: dict[str, list[Path]] = {Path(value).name: [] for value in science}
        for pixtable in pixtable_paths:
            raw_name = fits.getheader(pixtable).get("HIERARCH ESO PRO REC1 RAW1 NAME", "")
            if raw_name not in by_raw_name:
                raise PerExposureInputError(f"{pixtable}: unknown RAW1 provenance {raw_name!r}")
            by_raw_name[raw_name].append(pixtable)
        for raw in science:
            raw_path = Path(raw)
            group = sorted(by_raw_name[raw_path.name])
            if len(group) != expected_ifus:
                raise PerExposureInputError(f"{raw_path}: expected {expected_ifus} pixtables, found {len(group)}")
            date_obs = fits.getheader(raw_path).get("DATE-OBS", "")
            exposure_rows.append(
                {
                    "id": f"{night}_{raw_path.stem}",
                    "night": night,
                    "pixtables": [str(path) for path in group],
                    "metadata": {"date_obs": str(date_obs), "raw_object": str(raw_path)},
                }
            )
        calibrations = {
            "STD_RESPONSE": [str(path) for path in response_paths],
            "STD_TELLURIC": [str(path) for path in telluric_paths],
            "LSF_PROFILE": [str(path) for path in lsf_paths],
            **_static_paths(payload, {"EXTINCT_TABLE", "FILTER_LIST", "ASTROMETRY_WCS", "SKY_LINES"}),
        }
        night_calibrations[night] = calibrations
    if expected_ifus is None or not exposure_rows:
        raise PerExposureInputError("no completed night manifests were supplied")
    return (
        {"schema_version": 1, "run_id": run_id, "expected_ifus": expected_ifus, "exposures": exposure_rows},
        {"schema_version": 1, "run_id": run_id, "night_calibrations": night_calibrations},
    )
