"""Tercera capa de generacion de notebooks: lo PROPIO DE UN OBJETO.

Los dos generadores existentes son **generales**: `build_review_notebooks.py`
audita la cadena y `build_debug_notebooks.py` la rehace dentro del notebook. Los
dos producen exactamente lo mismo para todos los objetos, y eso es deliberado —
que ningun objeto sea "el canonico" es lo que hace comparables los resultados.

Pero algunos objetos necesitan pasos que los demas no. ROXs 42B b tiene, segun
la literatura, una fuente de fondo brillante en un costado (a veces citada como
ROXs 42B c); un objeto con una galaxia de fondo, o con una binaria cerca,
necesitaria enmascararla antes de medir. Meter eso en el generador general lo
llenaria de casos particulares y de `if target == ...`, que es justo lo que la
simetria entre objetos pretende evitar.

De ahi esta tercera capa. Sus reglas:

1. **Se aplica DESPUES** de que el generador general haya construido el
   notebook, y trabaja sobre el producto: inserta celdas, sustituye codigo o
   cambia perillas. No reimplementa nada.
2. **Es por objeto**: `scripts/target_overrides/<slug>.py`. Si no existe el
   modulo, no pasa nada y el notebook sale igual que hoy.
3. **Se declara en el propio notebook**: todo notebook modificado lleva una
   celda de aviso al principio diciendo QUE se le aplico y POR QUE. Sin eso
   estariamos recreando el problema de los notebooks editados a mano, pero a
   mayor escala y sin dejar rastro.
4. **Nunca se edita el `.ipynb`**: los cambios van en el modulo del objeto y se
   regenera, igual que con los otros dos generadores.

## Como se escribe un override

    # scripts/target_overrides/MiObjeto.py
    STAGES = {"C2", "C3"}          # etapas que toca; el resto salen intactas
    REASON = "una linea diciendo por que este objeto necesita esto"

    def apply(stage_id, cells, ctx):
        return insert_after_markdown(cells, "## 5 ", md("..."), code("..."))

`ctx` trae `target`, `run_id`, `kind` ("review" o "debug") y `md`/`code` para
construir celdas con el mismo formato que el generador general.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

OVERRIDES_DIR = Path(__file__).resolve().parent
#: Celda de aviso que se antepone a cualquier notebook modificado.
BANNER_PREFIX = "> **Notebook con ajustes propios de este objeto.**"


def load_override(target):
    """Modulo de override de un objeto, o None si no tiene."""

    path = OVERRIDES_DIR / f"{target}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"target_overrides.{target}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in ("STAGES", "REASON", "apply"):
        if not hasattr(module, attr):
            raise RuntimeError(f"{path} must define {attr}.")
    return module


def find_markdown(cells, prefix):
    """Indice de la primera celda markdown que empieza por `prefix`."""

    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        if "".join(cell.get("source", [])).lstrip().startswith(prefix):
            return index
    raise KeyError(f"No markdown cell starting with {prefix!r}; the general builder moved it.")


def insert_after_markdown(cells, prefix, *new_cells):
    """Inserta celdas justo despues de la seccion cuyo titulo empieza por `prefix`.

    Se ancla en el TEXTO del titulo y no en un indice: si el generador general
    reordena secciones, esto falla ruidosamente (KeyError) en vez de insertar en
    el sitio equivocado en silencio.
    """

    index = find_markdown(cells, prefix)
    # Saltar hasta el final de esa seccion (la siguiente celda markdown "## ").
    cursor = index + 1
    while cursor < len(cells) and not (
        cells[cursor].get("cell_type") == "markdown"
        and "".join(cells[cursor].get("source", [])).lstrip().startswith("## ")
    ):
        cursor += 1
    return list(cells[:cursor]) + list(new_cells) + list(cells[cursor:])


def replace_in_code(cells, old, new, *, count=1):
    """Sustituye texto dentro de celdas de codigo. Falla si no encuentra nada."""

    out = []
    done = 0
    for cell in cells:
        if cell.get("cell_type") == "code" and done < count:
            src = "".join(cell.get("source", []))
            if old in src:
                cell = dict(cell, source=src.replace(old, new).splitlines(keepends=True))
                done += 1
        out.append(cell)
    if done == 0:
        raise KeyError(f"{old!r} not found in any code cell; the general builder changed it.")
    return out


def banner_cell(md, target, reason, stage_id):
    return md(
        f"{BANNER_PREFIX}\n\n"
        f"Ademas de lo que genera el constructor general, a **{target}** se le aplica "
        f"`scripts/target_overrides/{target}.py` sobre la etapa **{stage_id}**:\n\n"
        f"> {reason}\n\n"
        "Como todo lo demas, **no se edita este `.ipynb`**: los cambios van en ese modulo "
        "y se regenera."
    )


def apply_overrides(cells, *, target, stage_id, run_id, kind, md, code):
    """Aplica el override del objeto, si lo hay. Devuelve `(cells, applied)`.

    Se llama al final del generador general, con el notebook ya construido.
    """

    module = load_override(target)
    if module is None or stage_id not in set(module.STAGES):
        return cells, False
    ctx = {"target": target, "run_id": run_id, "kind": kind, "md": md, "code": code}
    out = module.apply(stage_id, list(cells), ctx)
    if not out or out is cells:
        raise RuntimeError(f"{target} override for {stage_id} returned nothing.")
    return [banner_cell(md, target, module.REASON, stage_id)] + list(out), True


__all__ = [
    "BANNER_PREFIX",
    "apply_overrides",
    "banner_cell",
    "find_markdown",
    "insert_after_markdown",
    "load_override",
    "replace_in_code",
]
