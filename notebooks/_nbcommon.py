"""Utilidades compartidas por los notebooks de revisión A1..G5.

Deliberadamente ligero (solo stdlib): cada notebook debe poder abrirse y
auditar el QC de un run sin importar toda la pila científica. Las etapas
*ejecutables* sí importan `musepipe` en su propia celda (guardada por RUN).

Resolución del run:
  1. argumento explícito a `resolve_run_id`
  2. variable de entorno `MUSE_RUN_ID`
  3. `DEFAULT_RUN_ID` (abajo)

OJO: a diferencia de `musepipe.config.get_run_id`, aquí NO se lee
`active_run.txt` — los notebooks de revisión auditan por defecto el run
científico realineado, mientras que `active_run.txt` suele apuntar a un run
de smoke-test. Si ambos difieren, `resolve_run_id` lo avisa por stdout.
Todos los comandos canónicos de los notebooks pasan `--run-id` explícito,
así que la ejecución (RUN=True) nunca depende de `active_run.txt`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Run por defecto para la revisión: el cubo auto-reducido y re-alineado (plan B).
DEFAULT_RUN_ID = "ROXs12b_realigned"


def project_root() -> Path:
    """Raíz del repo = carpeta padre de `notebooks/`."""
    return Path(__file__).resolve().parent.parent


def _active_run_txt() -> str | None:
    """Primera línea no comentada de active_run.txt (o None si no existe)."""
    path = project_root() / "active_run.txt"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            return value
    return None


def resolve_run_id(run_id: str | None = None) -> str:
    resolved = run_id or os.environ.get("MUSE_RUN_ID") or DEFAULT_RUN_ID
    active = _active_run_txt()
    if active and active != resolved:
        print(
            f"AVISO: active_run.txt apunta a {active!r} pero este notebook audita "
            f"{resolved!r}. Los CLI de musepipe SIN --run-id usarían {active!r}."
        )
    return resolved


def run_dir(run_id: str | None = None) -> Path:
    return project_root() / "runs" / resolve_run_id(run_id)


def load_qc(relpath: str, run_id: str | None = None) -> dict:
    """Carga un JSON de QC bajo `runs/<run>/<relpath>`.

    `relpath` es relativo al directorio del run, p.ej. `stages/stage_e01_qc.json`
    o `report/run_summary.json`.
    """
    path = run_dir(run_id) / relpath
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}.\n"
            f"¿Corriste la etapa en el run '{resolve_run_id(run_id)}'? "
            f"Ver la celda 'Cómo ejecutar de forma independiente'."
        )
    with open(path) as fh:
        return json.load(fh)


def qc_path(relpath: str, run_id: str | None = None) -> Path:
    return run_dir(run_id) / relpath


def flatten(d: dict | list, prefix: str = "") -> dict:
    """Aplana un dict/list anidado a rutas 'a.b.c' -> valor escalar.

    Útil para volcar un QC como tabla plana en una celda.
    """
    out: dict = {}
    if isinstance(d, dict):
        items = d.items()
    elif isinstance(d, list):
        items = ((str(i), v) for i, v in enumerate(d))
    else:
        return {prefix: d}
    for k, v in items:
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, (dict, list)):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def show(d: dict, keys=None, title: str | None = None) -> None:
    """Imprime un subconjunto de un QC de forma legible."""
    if title:
        print(f"== {title} ==")
    flat = flatten(d)
    if keys is None:
        for k, v in flat.items():
            print(f"{k:55s} {v}")
    else:
        shown: set[str] = set()
        for k in keys:
            # Match exacto de una ruta hoja: imprime solo esa (evita que 'verdict'
            # arrastre todo 'verdict_by_pair.*').
            if k in flat:
                if k not in shown:
                    shown.add(k)
                    print(f"{k:55s} {flat[k]}")
                continue
            hits = {fk: fv for fk, fv in flat.items() if k.lower() in fk.lower()}
            if not hits:
                print(f"{k:55s} <no encontrado>")
            for fk, fv in hits.items():
                if fk in shown:
                    continue
                shown.add(fk)
                print(f"{fk:55s} {fv}")
