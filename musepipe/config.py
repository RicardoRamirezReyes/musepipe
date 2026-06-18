"""Configuration loading and run selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from .paths import RunPaths


RUN_ID_ENV_VAR = "MUSE_RUN_ID"
ACTIVE_RUN_FILENAME = "active_run.txt"


class ConfigError(RuntimeError):
    """Raised when run selection or config loading is inconsistent."""


@dataclass(frozen=True)
class RunConfig:
    """Loaded config payload plus resolved run paths."""

    run_id: str
    config: dict[str, Any]
    meta: dict[str, Any]
    paths: RunPaths
    payload: dict[str, Any]


def project_root_path(project_root: str | Path | None = None) -> Path:
    """Return an absolute project root path."""

    return Path("." if project_root is None else project_root).resolve()


def _first_active_run_line(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            return value
    return None


def get_run_id(
    project_root: str | Path | None = None,
    *,
    env_var: str = RUN_ID_ENV_VAR,
    active_run_file: str | Path = ACTIVE_RUN_FILENAME,
    default: str | None = None,
) -> str:
    """Resolve the active run id.

    Priority is:

    1. Environment variable, by default ``MUSE_RUN_ID``.
    2. ``active_run.txt`` in the project root, using the first non-comment line.
    3. Explicit ``default`` argument.

    A ``ConfigError`` is raised if none of those sources is available.
    """

    env_value = os.environ.get(env_var)
    if env_value is not None and env_value.strip():
        return env_value.strip()

    root = project_root_path(project_root)
    active_path = Path(active_run_file)
    if not active_path.is_absolute():
        active_path = root / active_path
    if active_path.exists():
        value = _first_active_run_line(active_path)
        if value:
            return value

    if default is not None and str(default).strip():
        return str(default).strip()

    raise ConfigError(
        f"No active run id found. Set {env_var}, create {active_path}, "
        "or pass run_id/default explicitly."
    )


def load_config_payload(config_json: str | Path) -> dict[str, Any]:
    """Read and minimally validate a config JSON payload."""

    path = Path(config_json)
    if not path.exists():
        raise ConfigError(f"Config JSON does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ConfigError(f"Config JSON must contain an object: {path}")
    if not isinstance(payload.get("config"), dict):
        raise ConfigError(f"Config JSON missing object key 'config': {path}")
    if payload.get("meta") is not None and not isinstance(payload.get("meta"), dict):
        raise ConfigError(f"Config JSON key 'meta' must be an object when present: {path}")
    return payload


def validate_run_config(
    config: dict[str, Any],
    expected_run_id: str,
    *,
    config_json: str | Path | None = None,
    allow_run_id_mismatch: bool = False,
) -> None:
    """Validate that a loaded config belongs to the selected run."""

    if not isinstance(config, dict):
        raise ConfigError("Config must be a dictionary.")

    cfg_run_id = config.get("run_id")
    if cfg_run_id is None:
        raise ConfigError("Config is missing required key 'run_id'.")

    if str(cfg_run_id) != str(expected_run_id) and not allow_run_id_mismatch:
        location = f" in {config_json}" if config_json is not None else ""
        raise ConfigError(
            f"Config run_id={cfg_run_id!r} does not match selected "
            f"run_id={expected_run_id!r}{location}."
        )


def load_run_config(
    run_id: str | None = None,
    *,
    project_root: str | Path | None = None,
    validate: bool = True,
    allow_run_id_mismatch: bool = False,
) -> RunConfig:
    """Load ``runs/<RUN_ID>/config/config.json`` with resolved paths."""

    root = project_root_path(project_root)
    selected_run_id = get_run_id(root) if run_id is None else str(run_id)
    paths = RunPaths.from_project_root(selected_run_id, root)
    payload = load_config_payload(paths.config_json)
    config = payload["config"]
    meta = payload.get("meta") or {}

    if validate:
        validate_run_config(
            config,
            selected_run_id,
            config_json=paths.config_json,
            allow_run_id_mismatch=allow_run_id_mismatch,
        )

    return RunConfig(
        run_id=selected_run_id,
        config=config,
        meta=meta,
        paths=paths,
        payload=payload,
    )


def available_run_ids(project_root: str | Path | None = None) -> list[str]:
    """Return run directory names that contain ``config/config.json``."""

    root = project_root_path(project_root)
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return []
    return sorted(
        path.name
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "config" / "config.json").exists()
    )


__all__ = [
    "ACTIVE_RUN_FILENAME",
    "RUN_ID_ENV_VAR",
    "ConfigError",
    "RunConfig",
    "available_run_ids",
    "get_run_id",
    "load_config_payload",
    "load_run_config",
    "project_root_path",
    "validate_run_config",
]
