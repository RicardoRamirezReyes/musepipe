"""Dónde encuentran los tests sus datos y cómo cargan los scripts: un solo sitio (auditoría del 2026-10-29, Fase 0).

Es para los tests NUEVOS: los existentes tienen cada uno su manera (cuatro en total: `ROOT/runs`,
`MUSE_PROJECT_ROOT` antes o después de `ROOT`, el cwd y rutas fijas) y no se cambian sin permiso del autor
(`AGENTS.md` → Tests). Se importa siempre como `from tests._datos import ...`, nunca `from _datos`, que crearía una
segunda copia del módulo con sus propias cachés. Empieza por `_`, así que pytest no lo recolecta (como
`tests/_g5_fixture.py`).

Variables de entorno, con una sola precedencia: la variable si está definida y, si no, el valor por defecto.

- `MUSE_PROJECT_ROOT`: la raíz con `runs/` y `../Data` (por defecto, este clon). Desde un worktree, el checkout
  principal. Los `targets/` y el código se leen siempre de este clon.
- `MUSE_TEFF_CADENAS`: carpetas de cadenas de la T_eff, separadas por `os.pathsep` (`:`); una sola carpeta también
  vale. Por defecto, las del paper (`~/muse_noche_20261024_c4/cadenas`).
- `MUSE_IR_DATA`: los espectros y rejillas del infrarrojo (por defecto `/mnt/2TB/IR_Data`).
- `MUSE_G3_LIBRARIES`: las bibliotecas de G3 (la misma variable que los notebooks de análisis); si no está, la
  `g3_libraries_root` del run resuelta contra `MUSE_PROJECT_ROOT`, no contra el clon del código.

Convención para un test nuevo:

1. Un fichero nuevo por análisis (`test_<script>_<prueba>.py`). No se añaden clases a ficheros existentes.
2. Primero las clases sintéticas, que no leen nada de disco. Las que leen datos reales llevan
   `@pytest.mark.external_data` y `@requiere(...)`, que las salta con un motivo que nombra lo que falta; si tardan
   5 s o más, también `@pytest.mark.slow`.
3. Invariantes y consistencia antes que cifras del run vivo: «con r_K = 0 la verosimilitud es la del modelo sin
   disco, bit a bit», «el texto del paper sale de las cadenas». Si hace falta un literal, en una tabla al principio
   del fichero, con su fecha y su origen.
4. Nunca un id de run escrito a mano: `run_del_paper(slug)` lo lee de `targets/<slug>.json`.
5. Un estado global que se cambie se restaura (`raiz_temporal`), y lo cacheado (`modelo_teff`) no se muta.
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import json
import os
import sys
import unittest
from pathlib import Path

#: la raíz del código (este clon)
REPO = Path(__file__).resolve().parents[1]
#: las cadenas de la T_eff del paper (C4 v2 `decoupled_c0`, 2026-10-24)
CADENAS_DEL_PAPER = Path("~/muse_noche_20261024_c4/cadenas")
IR_DATA = Path("/mnt/2TB/IR_Data")


def raiz_proyecto() -> Path:
    """La raíz con `runs/`: `MUSE_PROJECT_ROOT` si está definida; si no, este clon."""
    v = os.environ.get("MUSE_PROJECT_ROOT")
    return Path(v).expanduser().resolve() if v else REPO


def runs_dir() -> Path:
    return raiz_proyecto() / "runs"


def runs_del_paper() -> dict:
    """{slug: run del paper} de `targets/*.json` (los de este clon), en orden de slug."""
    out = {}
    for f in sorted((REPO / "targets").glob("*.json")):
        d = json.loads(f.read_text())
        if "paper" in d:
            out[d["slug"]] = d["paper"]["run"]
    return out


def run_del_paper(slug: str) -> str:
    return runs_del_paper()[slug]


def config_del_run(run: str) -> dict:
    """La clave `config` de `runs/<run>/config/config.json` (sin pasar por `musepipe.config`, que valida contra el cwd)."""
    return json.loads((runs_dir() / run / "config" / "config.json").read_text())["config"]


def cadenas_teff() -> list:
    """Las carpetas de `MUSE_TEFF_CADENAS` (siempre una lista; `:` las separa) o la del paper."""
    v = os.environ.get("MUSE_TEFF_CADENAS")
    partes = [p for p in v.split(os.pathsep) if p] if v else [str(CADENAS_DEL_PAPER)]
    return [Path(p).expanduser() for p in partes]


def ir_data() -> Path:
    v = os.environ.get("MUSE_IR_DATA")
    return Path(v).expanduser() if v else IR_DATA


def bibliotecas_g3(run: str) -> Path:
    """`MUSE_G3_LIBRARIES` o la `g3_libraries_root` del run contra `raiz_proyecto()`."""
    v = os.environ.get("MUSE_G3_LIBRARIES")
    if v:
        return Path(v).expanduser().resolve()
    return (raiz_proyecto() / config_del_run(run)["g3_libraries_root"]).resolve()


# ------------------------------------------------------------------ saltos
def _hay_runs_del_paper():
    runs = runs_del_paper()
    falta = [r for r in runs.values() if not (runs_dir() / r / "stages" / "spec_final_object.fits").exists()]
    return bool(runs) and not falta, f"sin los runs del paper en {runs_dir()} (define MUSE_PROJECT_ROOT)"


def _hay_cadenas():
    return any(d.is_dir() for d in cadenas_teff()), \
        f"sin las cadenas de la T_eff en {os.pathsep.join(map(str, cadenas_teff()))} (define MUSE_TEFF_CADENAS)"


def _hay_ir():
    return ir_data().is_dir(), f"sin {ir_data()} (define MUSE_IR_DATA)"


def _hay_bibliotecas():
    ok, motivo = _hay_runs_del_paper()
    if not ok:
        return ok, motivo
    falta = [r for r in runs_del_paper().values() if not bibliotecas_g3(r).is_dir()]
    return not falta, f"sin las bibliotecas de G3 de {', '.join(falta)} (define MUSE_G3_LIBRARIES o MUSE_PROJECT_ROOT)"


RECURSOS = {"runs": _hay_runs_del_paper, "cadenas": _hay_cadenas, "ir": _hay_ir, "bibliotecas": _hay_bibliotecas}


def falta(*recursos) -> str | None:
    """El motivo del primer recurso de `RECURSOS` que falta, o None si están todos."""
    for r in recursos:
        ok, motivo = RECURSOS[r]()
        if not ok:
            return motivo
    return None


def requiere(*recursos):
    """`unittest.skipUnless` para una clase o un test: lo salta, nombrando la variable, si falta algún recurso
    («runs», «cadenas», «ir», «bibliotecas»). Se apila con `@pytest.mark.external_data`."""
    motivo = falta(*recursos)
    return unittest.skipUnless(motivo is None, motivo or "")


# ------------------------------------------------------------------ scripts
def carga_script(nombre: str):
    """`scripts.<nombre>`, el MISMO módulo que importan los scripts (`from scripts import teff_bayes`): una sola
    copia por proceso. Los `_carga` de los tests existentes crean una segunda (`teff_bayes`, con su propia raíz)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    return importlib.import_module(f"scripts.{nombre}")


@contextlib.contextmanager
def raiz_temporal(raiz=None):
    """Lee los runs y las bibliotecas de `raiz` (por defecto `raiz_proyecto()`) mientras dura el bloque:
    `teff_bayes.fija_raiz` (que cambia también la de `ajuste_conjunto`, `tipo_ir_plantillas`, `tipo_ir_indices`
    y `suelo_cromosferico`) y `paper_figures.PROYECTO` si está importado. Restaura las dos al salir, también con
    una excepción."""
    tb = carga_script("teff_bayes")
    antes = tb.ROOT
    pf = sys.modules.get("scripts.paper_figures")
    pf_antes = getattr(pf, "PROYECTO", None)
    tb.fija_raiz(raiz or raiz_proyecto())
    if pf is not None:
        pf.PROYECTO = tb.ROOT
    try:
        yield tb.ROOT
    finally:
        tb.fija_raiz(antes)
        if pf is not None:
            pf.PROYECTO = pf_antes


@functools.lru_cache(maxsize=None)
def modelo_teff(slug: str, **kw):
    """`teff_bayes.Modelo(slug, **kw)` construido una vez por proceso (tarda segundos y lee /mnt/2TB). Los
    argumentos van por nombre y deben ser hashables (tuplas, no listas). No se muta."""
    tb = carga_script("teff_bayes")
    with raiz_temporal():
        return tb.Modelo(slug, **kw)
