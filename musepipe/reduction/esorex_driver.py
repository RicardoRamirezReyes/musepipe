"""Driver utilities for A1 raw reduction with the ESO MUSE pipeline.

This module deliberately separates pure planning/validation helpers from the
side-effecting EsoRex execution. The raw data reduction is expensive and has
human checkpoints, so tests exercise the pieces that can be verified with
synthetic FITS files.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Callable, Iterable, Mapping, Sequence

from astropy.io import fits


RUN_ID = "ROXs12b_raw"
STAGE_NAME = "00r_raw_reduction"
MUSE_RECIPE_PREFIX = "muse_"


class ReductionError(RuntimeError):
    """Raised when an A1 gate fails and the cascade must stop."""


@dataclass(frozen=True)
class RawRecord:
    """One FITS file classified from headers, not from its filename."""

    path: Path
    tag: str
    dpr_type: str
    dpr_catg: str
    date_obs: str
    ins_mode: str
    binning: str
    exptime: float | None
    sha256: str

    def as_csv_row(self) -> dict[str, str]:
        return {
            "file": str(self.path),
            "tag": self.tag,
            "dpr_type": self.dpr_type,
            "dpr_catg": self.dpr_catg,
            "date_obs": self.date_obs,
            "ins_mode": self.ins_mode,
            "binning": self.binning,
            "exptime": "" if self.exptime is None else f"{self.exptime:.9g}",
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SofEntry:
    """One line of an EsoRex SOF file."""

    path: Path
    tag: str

    def render(self) -> str:
        return f"{self.path} {self.tag}"


@dataclass(frozen=True)
class RecipePlan:
    """A concrete recipe invocation plan."""

    recipe: str
    entries: tuple[SofEntry, ...]
    params: Mapping[str, str] | None = None


@dataclass(frozen=True)
class RecipeRequirement:
    """Input requirement for building a SOF from inventory and prior products."""

    logical_tag: str
    sof_tag: str
    source: str = "inventory"
    min_count: int = 1
    optional: bool = False


DEFAULT_RECIPE_REQUIREMENTS: dict[str, tuple[RecipeRequirement, ...]] = {
    "muse_bias": (
        RecipeRequirement("BIAS", "BIAS", min_count=5),
    ),
    "muse_flat": (
        RecipeRequirement("FLAT", "FLAT", min_count=3),
        RecipeRequirement("MASTER_BIAS", "MASTER_BIAS", source="products"),
    ),
    "muse_wavecal": (
        RecipeRequirement("ARC", "ARC", min_count=1),
        RecipeRequirement("MASTER_BIAS", "MASTER_BIAS", source="products"),
        RecipeRequirement("MASTER_FLAT", "MASTER_FLAT", source="products", optional=True),
        RecipeRequirement("TRACE_TABLE", "TRACE_TABLE", source="products"),
        RecipeRequirement("LINE_CATALOG", "LINE_CATALOG"),
    ),
    "muse_lsf": (
        RecipeRequirement("ARC", "ARC", min_count=1, optional=True),
        RecipeRequirement("MASTER_BIAS", "MASTER_BIAS", source="products"),
        RecipeRequirement("MASTER_FLAT", "MASTER_FLAT", source="products", optional=True),
        RecipeRequirement("TRACE_TABLE", "TRACE_TABLE", source="products"),
        RecipeRequirement("WAVECAL_TABLE", "WAVECAL_TABLE", source="products"),
        RecipeRequirement("LINE_CATALOG", "LINE_CATALOG"),
    ),
    "muse_twilight": (
        RecipeRequirement("TWILIGHT", "TWILIGHT", min_count=1, optional=True),
        RecipeRequirement("MASTER_BIAS", "MASTER_BIAS", source="products"),
        RecipeRequirement("MASTER_FLAT", "MASTER_FLAT", source="products"),
        RecipeRequirement("WAVECAL_TABLE", "WAVECAL_TABLE", source="products"),
        RecipeRequirement("LSF_PROFILE", "LSF_PROFILE", source="products", optional=True),
    ),
    "muse_scibasic": (
        RecipeRequirement("OBJECT", "OBJECT", min_count=1),
        RecipeRequirement("STD", "STD", min_count=1, optional=True),
        RecipeRequirement("ILLUM", "ILLUM", min_count=1, optional=True),
        RecipeRequirement("MASTER_BIAS", "MASTER_BIAS", source="products"),
        RecipeRequirement("MASTER_FLAT", "MASTER_FLAT", source="products"),
        RecipeRequirement("TRACE_TABLE", "TRACE_TABLE", source="products"),
        RecipeRequirement("WAVECAL_TABLE", "WAVECAL_TABLE", source="products"),
        RecipeRequirement("LSF_PROFILE", "LSF_PROFILE", source="products", optional=True),
        RecipeRequirement("GEOMETRY_TABLE", "GEOMETRY_TABLE"),
    ),
    "muse_standard": (
        RecipeRequirement("PIXTABLE_STD", "PIXTABLE_STD", source="products"),
        RecipeRequirement("STD_FLUX_TABLE", "STD_FLUX_TABLE"),
        RecipeRequirement("EXTINCT_TABLE", "EXTINCT_TABLE"),
    ),
    "muse_scipost": (
        RecipeRequirement("PIXTABLE_OBJECT", "PIXTABLE_OBJECT", source="products"),
        RecipeRequirement("STD_RESPONSE", "STD_RESPONSE", source="products"),
        RecipeRequirement("EXTINCT_TABLE", "EXTINCT_TABLE"),
        RecipeRequirement("LSF_PROFILE", "LSF_PROFILE", source="products", optional=True),
        RecipeRequirement("ASTROMETRY_WCS", "ASTROMETRY_WCS", optional=True),
        RecipeRequirement("SKY_LINES", "SKY_LINES", optional=True),
    ),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header_value(header: fits.Header, *keys: str, default: str = "") -> str:
    for key in keys:
        if key in header and header[key] is not None:
            return str(header[key]).strip()
    return default


def _header_float(header: fits.Header, *keys: str) -> float | None:
    for key in keys:
        if key in header:
            try:
                return float(header[key])
            except (TypeError, ValueError):
                return None
    return None


def _binning_from_header(header: fits.Header) -> str:
    xbin = _header_value(header, "HIERARCH ESO DET WIN1 BINX", "DET WIN1 BINX")
    ybin = _header_value(header, "HIERARCH ESO DET WIN1 BINY", "DET WIN1 BINY")
    if xbin and ybin:
        return f"{xbin}x{ybin}"
    return _header_value(header, "CCDSUM", default="")


def normalize_muse_tag(header: fits.Header) -> str:
    """Return a conservative logical MUSE tag from FITS classification keys."""

    dpr_type = _header_value(header, "HIERARCH ESO DPR TYPE", "DPR TYPE").upper()
    dpr_catg = _header_value(header, "HIERARCH ESO DPR CATG", "DPR CATG").upper()
    pro_catg = _header_value(header, "HIERARCH ESO PRO CATG", "PRO CATG").upper()
    combined = " ".join(value for value in (dpr_catg, dpr_type, pro_catg) if value)

    static_tags = (
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
    )
    for tag in static_tags:
        if tag in combined:
            return tag

    product_tags = (
        "MASTER_BIAS",
        "MASTER_DARK",
        "MASTER_FLAT",
        "TRACE_TABLE",
        "WAVECAL_TABLE",
        "LSF_PROFILE",
        "PIXTABLE_OBJECT",
        "PIXTABLE_STD",
        "STD_RESPONSE",
    )
    for tag in product_tags:
        if tag in combined:
            return tag

    if "BIAS" in combined:
        return "BIAS"
    if "DARK" in combined:
        return "DARK"
    if "ILLUM" in combined:
        return "ILLUM"
    if "TWILIGHT" in combined or "SKYFLAT" in combined or "SKY FLAT" in combined:
        return "TWILIGHT"
    if "FLAT" in combined:
        return "FLAT"
    if "WAVE" in combined or "ARC" in combined or "LAMP,WAVE" in combined:
        return "ARC"
    if "STD" in combined or "STANDARD" in combined:
        return "STD"
    if "OBJECT" in combined or "SCIENCE" in combined:
        return "OBJECT"
    return "UNKNOWN"


def classify_fits(path: str | Path, *, checksum: bool = True) -> RawRecord:
    """Classify one FITS file using DPR/PRO headers."""

    fits_path = Path(path)
    with fits.open(fits_path, memmap=True) as hdul:
        header = hdul[0].header
        tag = normalize_muse_tag(header)
        dpr_type = _header_value(header, "HIERARCH ESO DPR TYPE", "DPR TYPE")
        dpr_catg = _header_value(header, "HIERARCH ESO DPR CATG", "DPR CATG")
        date_obs = _header_value(header, "DATE-OBS", "MJD-OBS")
        ins_mode = _header_value(header, "HIERARCH ESO INS MODE", "INS MODE")
        exptime = _header_float(header, "EXPTIME", "HIERARCH ESO DET SEQ1 DIT")
        binning = _binning_from_header(header)

    return RawRecord(
        path=fits_path,
        tag=tag,
        dpr_type=dpr_type,
        dpr_catg=dpr_catg,
        date_obs=date_obs,
        ins_mode=ins_mode,
        binning=binning,
        exptime=exptime,
        sha256=sha256_file(fits_path) if checksum else "",
    )


def build_inventory(paths: Iterable[str | Path], *, checksum: bool = True) -> list[RawRecord]:
    """Build a sorted raw inventory from FITS headers."""

    records = [classify_fits(path, checksum=checksum) for path in paths]
    return sorted(records, key=lambda record: (record.tag, str(record.path)))


def write_inventory_csv(records: Sequence[RawRecord], path: str | Path) -> None:
    """Write the A1 raw inventory CSV with stable columns."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "tag",
        "dpr_type",
        "dpr_catg",
        "date_obs",
        "ins_mode",
        "binning",
        "exptime",
        "sha256",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_csv_row())


def read_inventory_csv(path: str | Path) -> list[RawRecord]:
    records: list[RawRecord] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            exptime = float(row["exptime"]) if row.get("exptime") else None
            records.append(
                RawRecord(
                    path=Path(row["file"]),
                    tag=row["tag"],
                    dpr_type=row.get("dpr_type", ""),
                    dpr_catg=row.get("dpr_catg", ""),
                    date_obs=row.get("date_obs", ""),
                    ins_mode=row.get("ins_mode", ""),
                    binning=row.get("binning", ""),
                    exptime=exptime,
                    sha256=row.get("sha256", ""),
                )
            )
    return records


def read_products_json(path: str | Path | None) -> dict[str, list[str]]:
    """Read a product map JSON used to build SOFs after earlier recipes."""

    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ReductionError(f"Products JSON must contain an object: {path}")
    products: dict[str, list[str]] = {}
    for tag, values in payload.items():
        if isinstance(values, str):
            products[str(tag)] = [values]
        elif isinstance(values, list) and all(isinstance(value, str) for value in values):
            products[str(tag)] = list(values)
        else:
            raise ReductionError(
                f"Products JSON value for {tag!r} must be a string or list of strings."
            )
    return products


def group_records_by_tag(records: Iterable[RawRecord]) -> dict[str, list[RawRecord]]:
    grouped: dict[str, list[RawRecord]] = {}
    for record in records:
        grouped.setdefault(record.tag, []).append(record)
    return grouped


def _product_entries(products: Mapping[str, Sequence[str | Path]]) -> dict[str, list[Path]]:
    normalized: dict[str, list[Path]] = {}
    for tag, paths in products.items():
        normalized[str(tag)] = [Path(path) for path in paths]
    return normalized


def build_recipe_plan(
    recipe: str,
    records: Sequence[RawRecord],
    products: Mapping[str, Sequence[str | Path]] | None = None,
    *,
    requirements: Mapping[str, Sequence[RecipeRequirement]] = DEFAULT_RECIPE_REQUIREMENTS,
    params: Mapping[str, str] | None = None,
) -> RecipePlan:
    """Build and validate one recipe SOF plan from inventory and products."""

    if recipe not in requirements:
        raise ReductionError(f"No SOF requirements registered for recipe {recipe!r}.")

    by_tag = group_records_by_tag(records)
    product_map = _product_entries(products or {})
    entries: list[SofEntry] = []
    missing: list[str] = []

    for requirement in requirements[recipe]:
        if requirement.source == "inventory":
            paths = [record.path for record in by_tag.get(requirement.logical_tag, [])]
        elif requirement.source == "products":
            paths = list(product_map.get(requirement.logical_tag, []))
        else:
            raise ReductionError(
                f"Unsupported source {requirement.source!r} for {recipe}:{requirement.logical_tag}."
            )

        if len(paths) < requirement.min_count:
            if requirement.optional:
                continue
            missing.append(
                f"{requirement.logical_tag} ({len(paths)}/{requirement.min_count})"
            )
            continue

        for path in paths:
            entries.append(SofEntry(Path(path), requirement.sof_tag))

    if missing:
        raise ReductionError(f"{recipe} missing required SOF inputs: {', '.join(missing)}")
    return RecipePlan(recipe=recipe, entries=tuple(entries), params=params)


def write_sof(plan: RecipePlan, path: str | Path) -> Path:
    """Write one SOF file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(entry.render() for entry in plan.entries) + "\n"
    output.write_text(text, encoding="utf-8")
    return output


def validate_sof(plan: RecipePlan, *, check_header_tags: bool = True) -> None:
    """Validate that a SOF refers to existing inputs and matching DPR tags."""

    errors: list[str] = []
    for entry in plan.entries:
        if not entry.path.exists():
            errors.append(f"missing file: {entry.path}")
            continue
        if not check_header_tags:
            continue
        try:
            record = classify_fits(entry.path, checksum=False)
        except Exception as exc:  # pragma: no cover - defensive report path
            errors.append(f"cannot read FITS header for {entry.path}: {exc}")
            continue
        if record.tag != entry.tag:
            errors.append(f"{entry.path}: SOF tag {entry.tag} != header tag {record.tag}")
    if errors:
        raise ReductionError("; ".join(errors))


def parse_esorex_recipes(output: str) -> list[str]:
    """Parse recipe names from ``esorex --recipes`` output."""

    recipes: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        token = stripped.split(":", 1)[0].split()[0] if stripped.split() else ""
        if token:
            recipes.append(token)
    return recipes


def parse_version(text: str) -> str:
    match = re.search(r"version\s+([0-9]+(?:\.[0-9]+)+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, text=True, capture_output=True)


def detect_muse_pipeline_version(
    recipe: str,
    executable: str,
    *,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run_command,
) -> str:
    """Return the MUSE plugin version from a recipe man-page.

    `esorex --man-page <recipe>` prints a line such as
    ``muse_scipost -- version 2.10.16``; that plugin version (not the esorex
    executor version) is what belongs in the reproducibility record.
    """

    result = runner([executable, "--man-page", recipe])
    text = f"{result.stdout}\n{result.stderr}"
    match = re.search(
        rf"{re.escape(recipe)}\s*--\s*version\s+([0-9]+(?:\.[0-9]+)+)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else ""


def check_esorex_environment(
    *,
    esorex: str = "esorex",
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run_command,
    min_muse_version: str = "2.8",
) -> dict[str, object]:
    """Check that EsoRex and MUSE recipes are available.

    The installed EsoRex binary alone is not enough. A1 must stop if the MUSE
    plugin recipes are not listed, because the reduction would otherwise fail
    later with much less context.
    """

    executable = shutil.which(esorex) if os.sep not in esorex else esorex
    if executable is None:
        raise ReductionError("EsoRex executable not found in PATH.")

    version_result = runner([executable, "--version"])
    if version_result.returncode != 0:
        raise ReductionError(f"EsoRex version check failed: {version_result.stderr.strip()}")

    recipes_result = runner([executable, "--recipes"])
    if recipes_result.returncode != 0:
        raise ReductionError(f"EsoRex recipe listing failed: {recipes_result.stderr.strip()}")

    recipes = parse_esorex_recipes(recipes_result.stdout)
    muse_recipes = sorted(recipe for recipe in recipes if recipe.startswith(MUSE_RECIPE_PREFIX))
    if not muse_recipes:
        raise ReductionError(
            "EsoRex is installed, but no MUSE recipes (muse_*) are available. "
            "Install the ESO MUSE pipeline plugin or run inside a container that includes it."
        )

    # The MUSE plugin version is NOT the esorex banner version: `esorex --recipes`
    # opens with "ESO Recipe Execution Tool, version 3.x", so parsing its stdout
    # would record the executor version. Query a recipe man-page instead, which
    # prints "<recipe> -- version <plugin_version>".
    muse_version = detect_muse_pipeline_version(muse_recipes[0], executable, runner=runner)
    if muse_version:
        if _version_tuple(muse_version) < _version_tuple(min_muse_version):
            raise ReductionError(
                f"MUSE pipeline version {muse_version} is older than required {min_muse_version}."
            )
    else:
        muse_version = "recipes_present_version_not_reported"

    return {
        "esorex": parse_version(version_result.stdout) or version_result.stdout.strip(),
        "esorex_path": str(executable),
        "muse_pipeline": muse_version,
        "muse_recipes": muse_recipes,
        "platform": platform.platform(),
        "container_or_native": "native",
    }


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def disk_free_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def total_ram_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    return pages * page_size / (1024**3)


def ensure_a1_run_tree(
    project_root: str | Path,
    *,
    run_id: str = RUN_ID,
    raw_data_dir: str | Path,
    adp_reference: str | Path | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Create the A1 run directory skeleton and config/QC placeholders."""

    root = Path(project_root).resolve()
    run_dir = root / "runs" / run_id
    paths = {
        "run_dir": run_dir,
        "config_dir": run_dir / "config",
        "raw_reduction_dir": run_dir / "raw_reduction",
        "sof_dir": run_dir / "raw_reduction" / "sof",
        "stage_dir": run_dir / "stages",
        "plot_dir": run_dir / "plots",
        "log_dir": run_dir / "logs",
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "meta": {
            "created_utc": utc_now_iso(),
            "stage": STAGE_NAME,
        },
        "config": {
            "run_id": run_id,
            "entry_point": "raw",
            "raw_data_dir": str(Path(raw_data_dir).expanduser()),
            "adp_reference": "" if adp_reference is None else str(Path(adp_reference).expanduser()),
        },
    }
    config_path = paths["config_dir"] / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    qc_path = paths["stage_dir"] / "stage00r_qc.json"
    if not qc_path.exists():
        qc_payload = stage00r_qc_skeleton(
            run_id=run_id,
            environment=environment or {},
            adp_reference=adp_reference,
            raw_inventory_csv=paths["raw_reduction_dir"] / "raw_inventory.csv",
        )
        qc_path.write_text(json.dumps(qc_payload, indent=2) + "\n", encoding="utf-8")

    paths["config_json"] = config_path
    paths["stage_qc"] = qc_path
    return paths


def stage00r_qc_skeleton(
    *,
    run_id: str = RUN_ID,
    environment: Mapping[str, object],
    adp_reference: str | Path | None,
    raw_inventory_csv: str | Path,
) -> dict[str, object]:
    adp_payload: dict[str, object] = {"file": "" if adp_reference is None else str(adp_reference)}
    if adp_reference is not None and Path(adp_reference).exists():
        adp_payload["sha256"] = sha256_file(adp_reference)
    else:
        adp_payload["sha256"] = ""

    return {
        "stage": STAGE_NAME,
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "environment": dict(environment),
        "inputs": {
            "adp_reference": adp_payload,
            "prog_id": "",
            "obs_id": "",
            "ins_mode": "",
            "n_science_exposures": 0,
            "raw_inventory_csv": str(raw_inventory_csv),
        },
        "recipes": [],
        "scipost_params": {"skymethod": "", "save": "cube,individual", "filter": ""},
        "warnings_by_recipe": {},
        "products": {"datacube": "", "pixtables_reduced": [], "whitelight": ""},
        "verification": {
            "v1_stat_present": False,
            "v2_std_residual_rms": None,
            "v3_wcs_ok": False,
            "v4_adp_whitelight_corr": None,
            "v5_adp_star_spec_ratio_rms": None,
            "v6_sky_mask_clean": False,
        },
        "gates_passed": [],
        "open_issues": [],
    }


def parse_log_warnings(log_text: str) -> dict[str, list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    elevated: list[str] = []
    elevated_pattern = re.compile(r"saturat|missing|extrapolat|bad pixels\s*>\s*\d+%", re.I)
    for line in log_text.splitlines():
        if "[ WARNING ]" in line or "WARNING" in line:
            warnings.append(line)
            if elevated_pattern.search(line):
                elevated.append(line)
        if "[ ERROR ]" in line or "ERROR" in line:
            errors.append(line)
    return {"warnings": warnings, "errors": errors, "elevated": elevated}


def build_esorex_command(
    plan: RecipePlan,
    sof_path: str | Path,
    *,
    esorex: str = "esorex",
    output_dir: str | Path | None = None,
    log_file: str | Path | None = None,
) -> list[str]:
    command = [esorex]
    if output_dir is not None:
        command.append(f"--output-dir={Path(output_dir)}")
    if log_file is not None:
        command.append(f"--log-file={Path(log_file)}")
    # Recipe parameters must follow the recipe name: esorex parses anything
    # before it as a global option and aborts with "not recognized".
    command.append(plan.recipe)
    for key, value in (plan.params or {}).items():
        command.append(f"--{key}={value}")
    command.append(str(sof_path))
    return command


def parse_params(values: Sequence[str] | None) -> dict[str, str]:
    params: dict[str, str] = {}
    for value in values or ():
        if "=" not in value:
            raise ReductionError(f"Recipe parameter must be KEY=VALUE, got {value!r}.")
        key, param_value = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ReductionError(f"Recipe parameter key is empty in {value!r}.")
        params[key] = param_value.strip()
    return params


def run_recipe_plan(
    plan: RecipePlan,
    *,
    sof_path: str | Path,
    output_dir: str | Path,
    log_file: str | Path,
    esorex: str = "esorex",
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run_command,
) -> dict[str, object]:
    """Run one EsoRex recipe and return a QC-ready summary."""

    validate_sof(plan)
    write_sof(plan, sof_path)
    command = build_esorex_command(
        plan,
        sof_path,
        esorex=esorex,
        output_dir=output_dir,
        log_file=log_file,
    )
    start = time.monotonic()
    result = runner(command)
    duration = time.monotonic() - start
    log_path = Path(log_file)
    log_text = ""
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    else:
        log_text = (result.stdout or "") + "\n" + (result.stderr or "")
    parsed = parse_log_warnings(log_text)
    status = "ok" if result.returncode == 0 and not parsed["errors"] else "failed"
    if status != "ok":
        raise ReductionError(
            f"{plan.recipe} failed with return code {result.returncode}; see {log_file}"
        )
    return {
        "name": plan.recipe,
        "sof": str(sof_path),
        "status": status,
        "duration_s": round(duration, 3),
        "n_warnings": len(parsed["warnings"]),
        "warnings": parsed["warnings"],
        "elevated_warnings": parsed["elevated"],
        "command": command,
    }


def build_sof_phase(args: argparse.Namespace) -> int:
    records = read_inventory_csv(args.inventory)
    products = read_products_json(args.products_json)
    params = parse_params(args.param)
    if args.recipe == "muse_scipost":
        params.setdefault("save", "cube,individual")
    plan = build_recipe_plan(args.recipe, records, products=products, params=params)
    validate_sof(plan, check_header_tags=not args.skip_header_check)
    sof_path = Path(args.sof_dir) / f"{args.recipe}.sof"
    write_sof(plan, sof_path)
    print(f"Wrote {sof_path} with {len(plan.entries)} inputs")
    for entry in plan.entries:
        print(f"{entry.tag}: {entry.path}")
    if params:
        print("params: " + ", ".join(f"{key}={value}" for key, value in sorted(params.items())))
    return 0


def run_recipe_phase(args: argparse.Namespace) -> int:
    records = read_inventory_csv(args.inventory)
    products = read_products_json(args.products_json)
    params = parse_params(args.param)
    if args.recipe == "muse_scipost":
        params.setdefault("save", "cube,individual")
    plan = build_recipe_plan(args.recipe, records, products=products, params=params)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sof_path = Path(args.sof_dir) / f"{args.recipe}.sof"
    log_file = Path(args.log_dir) / f"{args.recipe}_{timestamp}.log"
    summary = run_recipe_plan(
        plan,
        sof_path=sof_path,
        output_dir=args.output_dir,
        log_file=log_file,
        esorex=args.esorex,
    )
    print(json.dumps(summary, indent=2))
    return 0


def phase0(args: argparse.Namespace) -> int:
    environment = check_esorex_environment(esorex=args.esorex)
    free_gb = disk_free_gb(args.project_root)
    ram_gb = total_ram_gb()
    if free_gb < args.min_free_gb:
        raise ReductionError(f"Only {free_gb:.1f} GB free; required {args.min_free_gb:.1f} GB.")
    if ram_gb is not None and ram_gb < args.min_ram_gb:
        raise ReductionError(f"Only {ram_gb:.1f} GB RAM; expected at least {args.min_ram_gb:.1f} GB.")
    environment = dict(environment)
    environment["disk_free_gb"] = round(free_gb, 3)
    environment["ram_total_gb"] = None if ram_gb is None else round(ram_gb, 3)
    paths = ensure_a1_run_tree(
        args.project_root,
        run_id=args.run_id,
        raw_data_dir=args.raw_data_dir,
        adp_reference=args.adp_reference,
        environment=environment,
    )
    print(f"A1 phase0 OK. QC skeleton: {paths['stage_qc']}")
    return 0


def inventory_phase(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_data_dir)
    files = sorted(raw_dir.glob("*.fits")) + sorted(raw_dir.glob("*.fits.fz"))
    if not files:
        raise ReductionError(f"No FITS files found in raw data dir: {raw_dir}")
    records = build_inventory(files, checksum=not args.skip_checksum)
    output = Path(args.output)
    write_inventory_csv(records, output)
    counts = group_records_by_tag(records)
    print(f"Wrote {len(records)} rows to {output}")
    for tag in sorted(counts):
        print(f"{tag}: {len(counts[tag])}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A1 raw-reduction driver helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase0_parser = subparsers.add_parser("phase0", help="Check environment and create run tree.")
    phase0_parser.add_argument("--project-root", default=".")
    phase0_parser.add_argument("--run-id", default=RUN_ID)
    phase0_parser.add_argument("--raw-data-dir", required=True)
    phase0_parser.add_argument("--adp-reference")
    phase0_parser.add_argument("--esorex", default="esorex")
    phase0_parser.add_argument("--min-free-gb", type=float, default=120.0)
    phase0_parser.add_argument("--min-ram-gb", type=float, default=16.0)
    phase0_parser.set_defaults(func=phase0)

    inventory_parser = subparsers.add_parser("inventory", help="Build raw_inventory.csv.")
    inventory_parser.add_argument("--raw-data-dir", required=True)
    inventory_parser.add_argument("--output", required=True)
    inventory_parser.add_argument("--skip-checksum", action="store_true")
    inventory_parser.set_defaults(func=inventory_phase)

    build_sof_parser = subparsers.add_parser("build-sof", help="Build and dry-run validate one SOF.")
    build_sof_parser.add_argument("--inventory", required=True)
    build_sof_parser.add_argument("--recipe", required=True)
    build_sof_parser.add_argument("--sof-dir", required=True)
    build_sof_parser.add_argument("--products-json")
    build_sof_parser.add_argument("--param", action="append", help="Recipe parameter as KEY=VALUE.")
    build_sof_parser.add_argument("--skip-header-check", action="store_true")
    build_sof_parser.set_defaults(func=build_sof_phase)

    run_recipe_parser = subparsers.add_parser("run-recipe", help="Build, validate, and run one recipe.")
    run_recipe_parser.add_argument("--inventory", required=True)
    run_recipe_parser.add_argument("--recipe", required=True)
    run_recipe_parser.add_argument("--sof-dir", required=True)
    run_recipe_parser.add_argument("--products-json")
    run_recipe_parser.add_argument("--param", action="append", help="Recipe parameter as KEY=VALUE.")
    run_recipe_parser.add_argument("--output-dir", required=True)
    run_recipe_parser.add_argument("--log-dir", required=True)
    run_recipe_parser.add_argument("--esorex", default="esorex")
    run_recipe_parser.set_defaults(func=run_recipe_phase)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ReductionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
