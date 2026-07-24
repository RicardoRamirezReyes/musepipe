"""Identidad de los objetos observados (perfiles bajo ``targets/``).

Separa *quién* es el objeto (nombre para mostrar, alias, referencias) de *cómo*
se procesa (parámetros de etapa, que viven en ``runs/<run>/config/config.json``).

Motivación (problema P4c del plan `docs/plan_multiobjeto_notebooks_2026-07-24.md`):
la prosa de los informes tenía el nombre del primer objeto incrustado en el
código, así que `runs/ROXs42Bb_realigned/report/report.md` afirmaba hablar de
"ROXs 12 B". El nombre debe venir del perfil del objeto, no de un literal.

Un perfil es un JSON en ``targets/<slug>.json``::

    {"slug": "ROXs42Bb", "display_name": "ROXs 42B b",
     "aliases": ["ROXs42Bb", "ROX42Bb"], "references": {...}}

Si un objeto no tiene perfil, `display_name` devuelve el slug tal cual: nunca el
nombre de otro objeto.

**Solo stdlib**: se importa desde notebooks y desde el generador de informes.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def targets_dir(project_root: str | Path | None = None) -> Path:
    root = Path("." if project_root is None else project_root).resolve()
    return root / "targets"


def _normalize(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


@lru_cache(maxsize=None)
def _load_all(targets_path: str) -> tuple[dict[str, Any], ...]:
    directory = Path(targets_path)
    if not directory.is_dir():
        return ()
    profiles = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("slug"):
            profiles.append(payload)
    return tuple(profiles)


def load_target(name: str, *, project_root: str | Path | None = None) -> dict[str, Any] | None:
    """Perfil del objeto por slug o alias (None si no hay perfil)."""
    if not name:
        return None
    wanted = _normalize(name)
    for profile in _load_all(str(targets_dir(project_root))):
        candidates = [profile["slug"], *profile.get("aliases", []),
                      profile.get("display_name", "")]
        if any(_normalize(c) == wanted for c in candidates if c):
            return profile
    return None


def display_name(name: str, *, project_root: str | Path | None = None) -> str:
    """Nombre legible del objeto; si no hay perfil, el propio identificador.

    Nunca inventa ni sustituye por otro objeto: ante la duda, devuelve lo que le
    dieron.
    """
    profile = load_target(name, project_root=project_root)
    if profile and profile.get("display_name"):
        return str(profile["display_name"])
    return str(name)


def target_of_run(run_id: str, *, project_root: str | Path | None = None) -> str | None:
    """Objeto de un run: `chain.target`, `config.target_name`, o el prefijo del run."""
    root = Path("." if project_root is None else project_root).resolve()
    config_json = root / "runs" / str(run_id) / "config" / "config.json"
    if config_json.exists():
        try:
            payload = json.loads(config_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        chain = payload.get("chain")
        if isinstance(chain, dict) and chain.get("target"):
            return str(chain["target"])
        target = payload.get("config", {}).get("target_name")
        if target:
            return str(target)
    return str(run_id).split("_", 1)[0] or None


def run_display_name(run_id: str, *, project_root: str | Path | None = None) -> str:
    """Nombre legible del objeto al que pertenece un run."""
    return display_name(target_of_run(run_id, project_root=project_root) or run_id,
                        project_root=project_root)


__all__ = [
    "targets_dir",
    "load_target",
    "display_name",
    "target_of_run",
    "run_display_name",
]
