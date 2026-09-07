"""Material de revisión de A1: coste medido, grafo de fases y paneles de cubos.

El notebook A1 llevaba escritos a mano los números de la primera reducción que
se hizo en esta máquina (7 exposiciones, 289 min, cubo 3681×330×338) y los
servía **idénticos** a cualquier otro objeto: el A1 de ROXs 42B b afirmaba, con
todas sus cifras, la reducción de ROXs 12 B. Aquí vive lo que el notebook
necesita para contar la reducción de SU objeto, leyendo lo que la cadena ya
dejó escrito en disco:

* `measured_runtime` — lo que costó *esta* ejecución, receta a receta, del
  `duration_s` de los manifiestos de noche, del `perexp_scipost_execution.json`
  y del `elapsed_minutes` del combinado.
* `estimate_esorex_runtime` — la extrapolación a datos nuevos. Recalibrada
  contra las **cuatro noches reales** de los dos objetos y, sobre todo, con
  `n_nights`: en perfil `cascade` la calibración se paga *por noche*, que es
  justo lo que la fórmula anterior (escrita para una única noche) ignoraba.
* `a1_phase_graph` — el flujo realmente ejecutado, con entradas y salidas por
  fase, para dibujarlo estilo esoreflex. Las entradas salen de
  `esorex_driver.DEFAULT_RECIPE_REQUIREMENTS` y las salidas de las puertas de
  producto de la cascada: el esquema no es un dibujo aparte que se pueda
  desincronizar, es una vista de la tabla que gobierna los SOF.
* `list_a1_cubes` + `cube_panels` + `halo_scale` — el mosaico N×5 (mediana,
  tres bandas y el canal de Hα) de los cubos que entraron al combinado.

Nota de unidades: los cubos de `muse_scipost` **ya están calibrados en flujo**
(`BUNIT` = 10⁻²⁰ erg/s/cm²/Å), así que los paneles NO se dividen por `EXPTIME`.
Dos exposiciones de 300 s y 720 s del mismo campo deben dar el mismo nivel de
flujo y distinta S/N; dividir por el tiempo rompería precisamente eso.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import load_config_payload, project_root_path, run_workdir_setting
from ..qc.cube_qc import detect_primary_yx, wavelength_axis_from_header
from ..spectral import nearest_channel_index
from .esorex_driver import DEFAULT_RECIPE_REQUIREMENTS
from .verify import DEFAULT_BAD_RANGES


class A1ReviewError(RuntimeError):
    """No se puede construir el material de revisión de A1."""


def _root(project_root=None) -> Path:
    """Raíz del repositorio, sin depender del directorio de trabajo.

    `config.project_root_path(None)` devuelve el **cwd**, que para un stage
    lanzado desde la raíz es correcto pero para un notebook no: Jupyter arranca
    con el cwd en `notebooks/<objeto>/`, y entonces las rutas salían como
    `notebooks/ROXs12b/runs/<run>/config/config.json`. Este módulo lo llaman
    sobre todo los notebooks, así que la raíz se deduce de la posición del
    propio fichero (`<raíz>/musepipe/reduction/a1_review.py`) y solo se cae al
    cwd si esa deducción no encuentra el paquete.
    """

    if project_root is not None:
        return project_root_path(project_root)
    here = Path(__file__).resolve().parents[2]
    # Basta con reconocer el paquete: `runs/` puede no existir todavía en un
    # clon recién hecho, y exigirlo devolvería el cwd justo cuando el mensaje
    # de error necesita apuntar a la raíz de verdad.
    if (here / "musepipe" / "__init__.py").is_file():
        return here
    return project_root_path(None)


# --------------------------------------------------------------------------
# Fases de la cascada
# --------------------------------------------------------------------------
#: Orden de las recetas por noche. Copia declarada de
#: `scripts/reduce_cascade.STEP_ORDER`: `scripts/` no es un paquete importable,
#: así que el valor se repite aquí y `tests/test_a1_review.py` comprueba que no
#: haya derivado del original (mismo guardián que la celda de drift de los
#: notebooks de análisis).
NIGHT_STEP_ORDER = ("flat", "wavecal", "lsf", "scibasic_object", "scibasic_std", "standard")

#: Receta esorex de cada paso (scibasic corre dos veces, recortando el SOF).
STEP_RECIPE = {
    "flat": "muse_flat",
    "wavecal": "muse_wavecal",
    "lsf": "muse_lsf",
    "scibasic_object": "muse_scibasic",
    "scibasic_std": "muse_scibasic",
    "standard": "muse_standard",
}

#: Salidas exigidas a cada paso, en el mismo orden y con los mismos conteos que
#: `reduce_cascade._gates`. `{n_object}` / `{n_std}` se rellenan con los conteos
#: reales de la noche. Verificado contra el original en los tests.
STEP_OUTPUTS = {
    "flat": (("MASTER_FLAT", "24"), ("TRACE_TABLE", "24")),
    "wavecal": (("WAVECAL_TABLE", "24"),),
    "lsf": (("LSF_PROFILE", "24"),),
    "scibasic_object": (("PIXTABLE_OBJECT", "{n_object}×24"),),
    "scibasic_std": (("PIXTABLE_STD", "{n_std}×24"),),
    "standard": (("STD_RESPONSE", "{n_std}"), ("STD_TELLURIC", "{n_std}")),
}

#: Entradas que `_trim_scibasic` quita del SOF en cada pasada de scibasic.
STEP_DROPPED_INPUTS = {"scibasic_object": ("STD",), "scibasic_std": ("OBJECT",)}

#: Tags que no son productos de esta reducción sino calibraciones estáticas del
#: pipeline (mismo conjunto que `reduce_cascade.STATIC_TAGS`).
STATIC_TAGS = frozenset({
    "GEOMETRY_TABLE", "ASTROMETRY_WCS", "ASTROMETRY_REFERENCE", "BADPIX_TABLE",
    "VIGNETTING_MASK", "STD_FLUX_TABLE", "EXTINCT_TABLE", "LINE_CATALOG",
    "SKY_LINES", "FILTER_LIST",
})


@dataclass(frozen=True)
class PhaseIO:
    """Una entrada o una salida de una fase, tal como la declara el SOF."""

    tag: str
    count: str = ""
    optional: bool = False
    source: str = "inventory"  # inventory | products | static


@dataclass
class Phase:
    """Una caja del esquema de reducción."""

    key: str
    title: str
    lane: str
    recipe: str | None = None
    inputs: tuple[PhaseIO, ...] = ()
    outputs: tuple[PhaseIO, ...] = ()
    status: str = "pending"  # ok | reused | resumed | pending | rejected | gate
    duration_s: float | None = None
    detail: str = ""

    @property
    def duration_min(self) -> float | None:
        return None if self.duration_s is None else self.duration_s / 60.0


# --------------------------------------------------------------------------
# Resolución del run que ejecutó A1
# --------------------------------------------------------------------------
def _payload(run_id: str, root: Path) -> dict:
    path = root / "runs" / str(run_id) / "config" / "config.json"
    if not path.exists():
        raise A1ReviewError(f"El run {run_id!r} no tiene config: {path}")
    return load_config_payload(path)


def resolve_a1_run(run_id: str, *, project_root=None) -> str:
    """Run en el que corrió A1 para este objeto (`chain.stage_runs['A1']`)."""

    root = _root(project_root)
    chain = _payload(run_id, root).get("chain") or {}
    return str((chain.get("stage_runs") or {}).get("A1", run_id))


def reduction_profile(run_id: str, *, project_root=None) -> str:
    root = _root(project_root)
    chain = _payload(run_id, root).get("chain") or {}
    return str(chain.get("reduction_profile") or "(no declarado)")


def a1_work_dir(run_id: str, *, project_root=None) -> Path | None:
    """Directorio de trabajo de la reducción (`$MUSE_WORK/...`).

    Se declara en el config del run que ejecutó A1, no en el de la cadena.
    """

    root = _root(project_root)
    for candidate in (resolve_a1_run(run_id, project_root=root), run_id):
        value = run_workdir_setting(candidate, "work_dir", project_root=root)
        if value:
            return Path(str(value))
    return None


def night_manifests(run_id: str, *, project_root=None) -> list[dict]:
    """Manifiestos `night_*/products_manifest.json`, ordenados por noche."""

    work = a1_work_dir(run_id, project_root=project_root)
    if work is None or not work.exists():
        return []
    out = []
    for path in sorted(work.glob("night_*/products_manifest.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(out, key=lambda m: str(m.get("night", "")))


def _perexp_search_roots(work: Path) -> list[Path]:
    """Dónde buscar la P2 de ESTE objeto, sin salirse de él.

    La P2 no siempre vive dentro del work-dir: ROXs 12 B tiene el `_by_night`
    para P1 y un `_p2` hermano para las exposiciones. Se admiten hermanos que
    compartan el prefijo del work-dir, nunca `work.parent` entero —
    el directorio de trabajo contiene los dos objetos y buscar ahí hacía que el
    coste de un objeto se leyera del otro.
    """

    roots = [work]
    stem = work.name
    for suffix in ("_by_night", "_reduction_by_night"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parent = work.parent
    if parent.exists():
        roots += [p for p in sorted(parent.iterdir())
                  if p.is_dir() and p != work and p.name.startswith(stem)]
    return roots


def _perexp_execution(run_id: str, *, project_root=None) -> dict | None:
    """`perexp_scipost_execution.json` de la pasada que produjo los cubos.

    Hay varias pasadas por objeto (`cube,skymodel`, `individual`, `skymodel`):
    se prefiere la que guardó cubos, que es la que alimenta el combinado. Cada
    fichero declara su `run_id` y se descarta el que no sea el del run que
    ejecutó A1: la procedencia se comprueba, no se supone.
    """

    root = _root(project_root)
    work = a1_work_dir(run_id, project_root=root)
    if work is None:
        return None
    a1_run = resolve_a1_run(run_id, project_root=root)
    best = None
    for search_root in _perexp_search_roots(work):
        if not search_root.exists():
            continue
        for path in sorted(search_root.glob("**/perexp_scipost_execution.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(payload.get("run_id") or "") != a1_run:
                continue
            save = str(payload.get("save") or "")
            score = 2 if "cube" in save else (0 if save else 1)
            key = (score, len(payload.get("exposures") or {}))
            if best is None or key > best[0]:
                best = (key, payload)
    return None if best is None else best[1]


# --------------------------------------------------------------------------
# Plan y QC del combinado
# --------------------------------------------------------------------------
def _cube_files(run_id: str, root: Path) -> list[str]:
    return list(_payload(run_id, root).get("config", {}).get("cube_files") or [])


def _combine_plan_candidates(run_id: str, root: Path) -> list[Path]:
    """Dónde puede estar el plan del combinado, en orden de preferencia.

    `runs/<run>/stages/`, `runs/<run>/`, y el directorio del cubo declarado en
    `cube_files` (ROXs 12 B lo tiene junto al cubo final, en el work-dir,
    porque el combinado corrió allí).
    """

    run_dir = root / "runs" / str(run_id)
    candidates = [run_dir / "stages" / "stream_combine_plan.json",
                  run_dir / "stream_combine_plan.json"]
    candidates += [Path(c).parent / "stream_combine_plan.json" for c in _cube_files(run_id, root)]
    return candidates


def combine_plan_path(run_id: str, *, project_root=None) -> Path | None:
    """DÓNDE está el `stream_combine_plan.json` del objeto (el que se puede leer).

    Va separado de `combine_plan` porque quien construye una vista por
    observación (`musepipe.observations`) tiene que **citar** de dónde salió la
    geometría en su procedencia, no sólo usarla.
    """

    for path in _combine_plan_candidates(run_id, _root(project_root)):
        if not path.exists():
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return path
    return None


def combine_plan(run_id: str, *, project_root=None) -> dict | None:
    """`stream_combine_plan.json` del objeto, esté donde esté."""

    for path in _combine_plan_candidates(run_id, _root(project_root)):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return None


def combine_qc(run_id: str, *, project_root=None) -> dict | None:
    """QC del combinado streaming (`stream_combine_v1`)."""

    root = _root(project_root)
    run_dir = root / "runs" / str(run_id)
    candidates = [run_dir / "cube_telcorr_qc.json", run_dir / "stages" / "cube_telcorr_qc.json"]
    for cube in _cube_files(run_id, root):
        cube_path = Path(cube)
        candidates.append(cube_path.with_name(cube_path.stem + "_qc.json"))
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return None


# --------------------------------------------------------------------------
# Coste medido y coste estimado
# --------------------------------------------------------------------------
#: Constantes de la fórmula, en minutos, recalibradas contra las cuatro noches
#: reales (ROXs 12 B 2022-08-29/31, ROXs 42B b 2022-08-28/30) y las dos pasadas
#: de scipost. La versión anterior venía del dataset de 7 exposiciones y daba
#: `t_scipost = 10.5 min/exp`, más del doble de lo que cuesta hoy.
T_CAL_NIGHT_MIN = 62.0        # flat + wavecal + lsf, por noche
T_CAL_NIGHT_NO_LSF_MIN = 34.0  # idem reutilizando LSF_PROFILE de archivo
T_STD_NIGHT_MIN = 6.2         # scibasic(std) + standard, por noche
T_SCIBASIC_EXP_MIN = 3.0      # scibasic(object), por exposición
T_SCIPOST_EXP_MIN = 4.5       # scipost por exposición (P2)
T_COMBINE_EXP_MIN = 1.24      # combinado streaming, por exposición


def estimate_esorex_runtime(
    n_exp: int,
    n_nights: int = 1,
    *,
    reuse_lsf: bool = False,
    reuse_calibrations: bool = False,
    cores_factor: float = 1.0,
) -> float:
    """Wall-time estimado de la reducción completa, en minutos.

    En perfil `cascade` la calibración se paga **por noche**: un objeto de dos
    noches paga dos flats, dos wavecals y dos lsf. `reuse_lsf` refleja el
    `reused_archive_product` que la cascada aplica cuando hay un LSF_PROFILE de
    archivo válido; `reuse_calibrations` anula toda la calibración (masters ya
    hechos). `muse_bias` no entra: los dos objetos reutilizaron el MASTER_BIAS
    de archivo.
    """

    n_exp = max(int(n_exp), 0)
    n_nights = max(int(n_nights), 1)
    if reuse_calibrations:
        per_night = 0.0
    else:
        per_night = (T_CAL_NIGHT_NO_LSF_MIN if reuse_lsf else T_CAL_NIGHT_MIN) + T_STD_NIGHT_MIN
    per_exp = T_SCIBASIC_EXP_MIN + T_SCIPOST_EXP_MIN + T_COMBINE_EXP_MIN
    return (n_nights * per_night + n_exp * per_exp) / max(float(cores_factor), 1e-6)


def _recipe_durations(manifest: dict) -> dict[str, float | None]:
    """Reparte `recipes[].duration_s` del manifiesto entre los pasos de la noche.

    Las recetas se anotan en el orden de `NIGHT_STEP_ORDER`, y `muse_scibasic`
    aparece dos veces (objeto y estándar). Un paso reanudado desde checkpoint no
    deja entrada: se devuelve `None`, y eso se dice, no se rellena con un cero.
    """

    recipes = list(manifest.get("recipes") or [])
    checkpoints = manifest.get("checkpoints") or {}
    out: dict[str, float | None] = {}
    ptr = 0
    for step in NIGHT_STEP_ORDER:
        if step not in checkpoints:
            continue
        want = STEP_RECIPE[step]
        if ptr < len(recipes) and recipes[ptr].get("name") == want:
            out[step] = recipes[ptr].get("duration_s")
            ptr += 1
        else:
            out[step] = None
    return out


def measured_runtime(run_id: str, *, project_root=None) -> dict:
    """Lo que costó de verdad la reducción de ESTE objeto.

    Devuelve `{nights: [...], perexp: {...}, combine: {...}, totals: {...}}` con
    los minutos por receta. `resumed` cuenta los pasos que la cascada dio por
    buenos desde un checkpoint anterior y que por tanto no cuestan nada en esta
    ejecución pero sí en una desde cero: se declaran para que el total medido no
    se confunda con el coste real de reducir.
    """

    root = _root(project_root)
    nights, resumed = [], 0
    for manifest in night_manifests(run_id, project_root=root):
        durations = _recipe_durations(manifest)
        checkpoints = manifest.get("checkpoints") or {}
        steps = []
        for step in NIGHT_STEP_ORDER:
            if step not in checkpoints:
                continue
            status = str(checkpoints[step].get("status") or "")
            seconds = durations.get(step)
            if seconds is None and status != "reused_archive_product":
                resumed += 1
            steps.append({
                "step": step,
                "recipe": STEP_RECIPE[step],
                "status": status,
                "minutes": None if seconds is None else seconds / 60.0,
            })
        science = (manifest.get("association") or {}).get("science") or []
        nights.append({
            "night": manifest.get("night"),
            "n_science": len(science),
            "steps": steps,
            "minutes": sum(s["minutes"] for s in steps if s["minutes"] is not None),
        })

    execution = _perexp_execution(run_id, project_root=root)
    exposures = (execution or {}).get("exposures") or {}
    per_exp = [e.get("duration_s") for e in exposures.values() if isinstance(e, dict)]
    per_exp = [v / 60.0 for v in per_exp if isinstance(v, (int, float))]
    perexp = {
        "n_exposures": len(exposures),
        "minutes": float(sum(per_exp)),
        "median_minutes": float(np.median(per_exp)) if per_exp else None,
        "status": (execution or {}).get("status"),
        "excluded": list((execution or {}).get("excluded_exposures") or []),
    }

    qc = combine_qc(run_id, project_root=root) or {}
    combine = {
        "minutes": qc.get("elapsed_minutes"),
        "n_exposures": qc.get("n_exposures"),
        "method": qc.get("method"),
    }

    total = sum(n["minutes"] for n in nights) + perexp["minutes"] + (combine["minutes"] or 0.0)
    n_exp = perexp["n_exposures"] or combine["n_exposures"] or 0
    reuse_lsf = any(
        s["status"] == "reused_archive_product"
        for night in nights for s in night["steps"] if s["step"] == "lsf"
    )
    return {
        "nights": nights,
        "perexp": perexp,
        "combine": combine,
        "resumed_steps": resumed,
        "totals": {
            "measured_minutes": total,
            "n_nights": len(nights),
            "n_exposures": n_exp,
            "n_combined": combine["n_exposures"],
            "reuse_lsf": reuse_lsf,
            "from_scratch_minutes": estimate_esorex_runtime(
                n_exp, max(len(nights), 1), reuse_lsf=reuse_lsf
            ),
        },
    }


# --------------------------------------------------------------------------
# Grafo de fases
# --------------------------------------------------------------------------
def _inputs_for(step: str) -> tuple[PhaseIO, ...]:
    """Entradas del SOF de un paso, leídas de la tabla que gobierna los SOF."""

    recipe = STEP_RECIPE[step]
    dropped = STEP_DROPPED_INPUTS.get(step, ())
    out = []
    for req in DEFAULT_RECIPE_REQUIREMENTS.get(recipe, ()):
        if req.logical_tag in dropped:
            continue
        source = "static" if req.logical_tag in STATIC_TAGS else req.source
        out.append(PhaseIO(
            tag=req.logical_tag,
            count="" if req.min_count <= 1 else f"×{req.min_count}",
            optional=req.optional,
            source=source,
        ))
    return tuple(out)


def _outputs_for(step: str, n_object: int, n_std: int) -> tuple[PhaseIO, ...]:
    return tuple(
        PhaseIO(tag=tag, count=count.format(n_object=n_object, n_std=n_std), source="products")
        for tag, count in STEP_OUTPUTS[step]
    )


def a1_phase_graph(run_id: str, *, project_root=None) -> list[Phase]:
    """Las fases que A1 ejecutó realmente para este objeto, con I/O y duración.

    Tres carriles: P1 (calibración y scibasic, **por noche**), P2 (scipost por
    exposición, con sus dos puertas humanas) y P3 (el combinado). La rama
    `muse_exp_combine` se incluye marcada como descartada porque forma parte de
    la historia: se planificó, corrió y no produjo cubo.
    """

    root = _root(project_root)
    runtime = measured_runtime(run_id, project_root=root)
    phases: list[Phase] = []

    phases.append(Phase(
        key="inventory", title="Inventario y agrupación por noche", lane="P1 · por noche",
        inputs=(PhaseIO("RAW MUSE", source="inventory"),),
        outputs=(PhaseIO("raw_inventory.csv", source="products"),
                 PhaseIO("night associations", source="products")),
        status="ok",
        detail=f"{runtime['totals']['n_nights']} noche(s), corte a las 12:00 UTC",
    ))
    phases.append(Phase(
        key="bias", title="MASTER_BIAS de archivo", lane="P1 · por noche",
        recipe="(muse_bias no se ejecuta)",
        inputs=(PhaseIO("MASTER_BIAS ESO", source="static"),),
        outputs=(PhaseIO("MASTER_BIAS", "×1", source="products"),),
        status="reused", detail="bias_exception declarada en la asociación",
    ))

    manifests = {str(m.get("night")): m for m in night_manifests(run_id, project_root=root)}
    for night in runtime["nights"]:
        manifest = manifests.get(str(night["night"]), {})
        checkpoints = manifest.get("checkpoints") or {}
        n_object = night["n_science"]
        n_std = len(((manifest.get("association") or {}).get("calibrations") or {}).get("STD") or [])
        n_std = n_std or 1
        for entry in night["steps"]:
            step = entry["step"]
            raw_status = str(checkpoints.get(step, {}).get("status") or "")
            if raw_status == "reused_archive_product":
                status, detail = "reused", "LSF_PROFILE de archivo"
            elif entry["minutes"] is None:
                status, detail = "resumed", "reanudado desde checkpoint (sin duración registrada)"
            else:
                status, detail = "ok", ""
            phases.append(Phase(
                key=f"{night['night']}:{step}", title=f"{STEP_RECIPE[step]} · {step}",
                lane=f"P1 · noche {night['night']}", recipe=STEP_RECIPE[step],
                inputs=_inputs_for(step), outputs=_outputs_for(step, n_object, n_std),
                status=status,
                duration_s=None if entry["minutes"] is None else entry["minutes"] * 60.0,
                detail=detail,
            ))

    perexp = runtime["perexp"]
    phases.append(Phase(
        key="gate_scipost", title="Puerta · plan de scipost", lane="P2 · por exposición",
        inputs=(PhaseIO("perexp_scipost_plan.json", source="products"),),
        outputs=(PhaseIO("plan aprobado", source="products"),),
        status="gate", detail="el plan se escribe bloqueado y se libera a mano",
    ))
    phases.append(Phase(
        key="scipost", title="muse_scipost · por exposición", lane="P2 · por exposición",
        recipe="muse_scipost",
        inputs=_scipost_inputs(), outputs=(
            PhaseIO("DATACUBE_FINAL", f"×{perexp['n_exposures']}", source="products"),
            PhaseIO("IMAGE_FOV", f"×{perexp['n_exposures']}", source="products"),
            PhaseIO("SKY_*", source="products"),
            PhaseIO("PIXTABLE_REDUCED", source="products", optional=True),
        ),
        status="ok" if perexp["n_exposures"] else "pending",
        duration_s=perexp["minutes"] * 60.0 if perexp["minutes"] else None,
        detail=f"{perexp['n_exposures']} exposiciones"
               + (f", {len(perexp['excluded'])} excluida(s)" if perexp["excluded"] else ""),
    ))
    phases.append(Phase(
        key="gate_align", title="Puerta · revisión de alineado", lane="P2 · por exposición",
        inputs=(PhaseIO("IMAGE_FOV", source="products"),),
        outputs=(PhaseIO("offsets aprobados", source="products"),),
        status="gate", detail="approved_auto / approved_manual / rejected",
    ))
    phases.append(Phase(
        key="exp_combine", title="muse_exp_combine", lane="P3 · combinado",
        recipe="muse_exp_combine",
        inputs=(PhaseIO("PIXTABLE_REDUCED", source="products"),),
        outputs=(PhaseIO("(ningún cubo)", source="products"),),
        status="rejected", detail="corrió y no produjo cubo → descartado",
    ))
    combine = runtime["combine"]
    phases.append(Phase(
        key="stream_combine", title="Combinado streaming (propio)", lane="P3 · combinado",
        recipe="stream_combine (propio)",
        inputs=(PhaseIO("DATACUBE_FINAL", f"×{combine['n_exposures'] or '?'}", source="products"),),
        outputs=(PhaseIO("cubo final", source="products"),
                 PhaseIO("stream_combine_v1 QC", source="products")),
        status="ok" if combine["minutes"] else "pending",
        duration_s=(combine["minutes"] or 0.0) * 60.0 if combine["minutes"] else None,
        detail=f"{combine['method'] or 'sigclip'}, recorte 200 px, pesos por exptime",
    ))
    return phases


def _scipost_inputs() -> tuple[PhaseIO, ...]:
    out = []
    for req in DEFAULT_RECIPE_REQUIREMENTS.get("muse_scipost", ()):
        source = "static" if req.logical_tag in STATIC_TAGS else req.source
        out.append(PhaseIO(req.logical_tag, "×24" if req.logical_tag == "PIXTABLE_OBJECT" else "",
                           req.optional, source))
    out.append(PhaseIO("STD_TELLURIC", source="products"))
    return tuple(out)


# --------------------------------------------------------------------------
# Cubos del mosaico
# --------------------------------------------------------------------------
@dataclass
class CubeRow:
    """Un cubo que entró (o es) el producto de A1, con su encuadre ya medido."""

    path: str
    label: str
    kind: str  # "exposure" | "combined" | "legacy"
    exptime: float | None = None
    night: str | None = None
    center_yx: tuple[float, float] | None = None
    window: tuple[int, int, int, int] | None = None
    in_bounds: bool = True
    centroid_fallback: bool = False

    @property
    def exists(self) -> bool:
        return os.path.exists(self.path)


def list_a1_cubes(run_id: str, *, project_root=None, include_legacy: bool = False) -> list[CubeRow]:
    """Los cubos del mosaico, con el encuadre que usó el combinado.

    Prioriza los cubos **por exposición** del plan de combinado, que ya traen la
    posición de la primaria ajustada con MAOPPY (`measure_primary_center`) y la
    ventana de recorte: el mosaico enseña así exactamente lo que entró al
    combinado, en vez de un encuadre recalculado aquí.

    Si ninguno sobrevive en disco —el caso de ROXs 12 B, cuyos 29 cubos por
    exposición se purgaron tras combinar— cae al cubo combinado, con una única
    fila. `include_legacy` añade los cubos de `perexp_cubes` (una reducción
    anterior del mismo objeto); está apagado porque mezcla dos reducciones.
    """

    root = _root(project_root)
    rows: list[CubeRow] = []
    plan = combine_plan(run_id, project_root=root) or combine_qc(run_id, project_root=root) or {}
    for entry in plan.get("exposures") or []:
        path = str(entry.get("file") or "")
        if not path or not os.path.exists(path):
            continue
        exposure_id = str(entry.get("exposure_id") or Path(path).parent.name)
        window = entry.get("window")
        y_c, x_c = entry.get("y_center"), entry.get("x_center")
        rows.append(CubeRow(
            path=path, label=exposure_id, kind="exposure",
            exptime=entry.get("exptime"),
            night=exposure_id.split("_", 1)[0] if "_" in exposure_id else None,
            center_yx=(float(y_c), float(x_c)) if y_c is not None and x_c is not None else None,
            window=tuple(int(v) for v in window) if window and len(window) == 4 else None,
            in_bounds=bool(entry.get("in_bounds", True)),
            centroid_fallback=bool(entry.get("centroid_fallback", False)),
        ))

    if include_legacy:
        for path in (run_workdir_setting(run_id, "perexp_cubes", project_root=root) or []):
            if os.path.exists(str(path)):
                rows.append(CubeRow(path=str(path), label=Path(str(path)).parent.name,
                                    kind="legacy"))

    if rows:
        return rows

    qc = combine_qc(run_id, project_root=root) or {}
    center = ((qc.get("reference") or {}).get("center_pixel_yx")
              or (plan.get("reference") or {}).get("center_pixel_yx"))
    for path in _cube_files(run_id, root):
        if os.path.exists(str(path)):
            rows.append(CubeRow(
                path=str(path), label=Path(str(path)).name, kind="combined",
                center_yx=(float(center[0]), float(center[1])) if center else None,
            ))
    return rows


# --------------------------------------------------------------------------
# Paneles: mediana, tres bandas y el canal de Hα
# --------------------------------------------------------------------------
PANEL_COLUMNS = ("mediana", "banda azul", "banda centro", "banda roja", "canal Hα")


def halpha_observed_A(run_id: str, *, project_root=None) -> tuple[float, float, float]:
    """(λ_Hα observada, λ_reposo, RV) a partir del config del objeto.

    La RV sistémica **no tiene default silencioso**, igual que en E1: un cubo
    con la ventana de Hα puesta en el sitio equivocado es peor que no dibujarla.
    """

    root = _root(project_root)
    cfg = _payload(run_id, root).get("config", {})
    rest = cfg.get("h01_halpha_rest_A", cfg.get("halpha_A"))
    if rest is None:
        raise A1ReviewError(
            f"El config de {run_id!r} no declara `halpha_A` ni `h01_halpha_rest_A`.")
    rv = None
    for key in ("h01_rv_sys_kms", "rv_sys_kms", "systemic_rv_kms"):
        if cfg.get(key) is not None:
            rv = float(cfg[key])
            break
    if rv is None:
        raise A1ReviewError(
            f"El config de {run_id!r} no declara la RV sistémica (`h01_rv_sys_kms`); "
            "sin ella no se puede situar el canal de Hα.")
    c_kms = 299792.458
    return float(rest) * (1.0 + rv / c_kms), float(rest), rv


def _good_wavelength_mask(wave: np.ndarray) -> np.ndarray:
    """Canales fuera del láser AO y de las bandas telúricas fuertes."""

    good = np.isfinite(wave)
    for lo, hi in DEFAULT_BAD_RANGES:
        good &= ~((wave >= lo) & (wave <= hi))
    return good


def band_centers_A(wave: np.ndarray) -> tuple[float, float, float]:
    """Centros de las bandas azul / centro / roja, esquivando los rangos malos."""

    good = wave[_good_wavelength_mask(wave)]
    if good.size == 0:
        good = wave[np.isfinite(wave)]
    return tuple(float(v) for v in np.percentile(good, (5.0, 50.0, 95.0)))


def _nanmedian(block) -> np.ndarray:
    """Mediana por spaxel ignorando el aviso de columnas enteramente NaN.

    Los spaxels de borde no tienen datos en algunas bandas: es una condición
    esperada del recorte, no un síntoma, y su aviso ensucia el notebook.
    """

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                                category=RuntimeWarning)
        return np.nanmedian(block, axis=0)


@dataclass
class PanelSet:
    """Las cinco imágenes de una fila del mosaico."""

    label: str
    images: list[np.ndarray] = field(default_factory=list)
    center_yx: tuple[float, float] | None = None
    wavelengths_A: tuple[float, ...] = ()
    exptime: float | None = None
    bunit: str = ""
    note: str = ""


def _crop_box(row: CubeRow, shape_yx: tuple[int, int], crop_npix: int) -> tuple[int, int, int, int]:
    ny, nx = shape_yx
    if row.window is not None:
        y0, y1, x0, x1 = row.window
        return max(y0, 0), min(y1, ny), max(x0, 0), min(x1, nx)
    if row.center_yx is not None:
        cy, cx = row.center_yx
    else:
        cy, cx = ny / 2.0, nx / 2.0
    half = crop_npix // 2
    y0 = int(np.clip(round(cy) - half, 0, max(ny - 1, 0)))
    x0 = int(np.clip(round(cx) - half, 0, max(nx - 1, 0)))
    return y0, min(y0 + crop_npix, ny), x0, min(x0 + crop_npix, nx)


def cube_panels(
    row: CubeRow,
    *,
    halpha_A: float,
    stride: int = 25,
    band_halfwidth_ch: int = 40,
    halpha_halfwidth_ch: int = 3,
    crop_npix: int = 200,
    data_ext: int = 1,
) -> PanelSet:
    """Las 5 imágenes de un cubo: mediana, tres bandas y el canal de Hα.

    La mediana se calcula sobre canales diezmados (`stride`, el mismo truco de
    `stream_combine.measure_primary_center`): leer los 3681 canales de un cubo
    de 3 GB para una figura de auditoría no aporta nada. Medido: ~3.4 s por
    cubo, ~2 min las 30 filas de ROXs 42B b.
    """

    from astropy.io import fits  # local: el módulo se importa sin el stack a veces

    with fits.open(row.path, memmap=True) as hdul:
        exptime = hdul[0].header.get("EXPTIME", row.exptime)
        hdu = hdul[data_ext]
        data = hdu.data
        if data is None or data.ndim != 3:
            raise A1ReviewError(f"{row.path}: la extensión {data_ext} no es un cubo 3D")
        nz, ny, nx = data.shape
        wave = wavelength_axis_from_header(hdu.header, nz)
        y0, y1, x0, x1 = _crop_box(row, (ny, nx), crop_npix)

        median = _nanmedian(data[::max(int(stride), 1), y0:y1, x0:x1])
        images = [median]
        centers = list(band_centers_A(wave))
        for center in centers:
            k = nearest_channel_index(wave, center)
            lo, hi = max(k - band_halfwidth_ch, 0), min(k + band_halfwidth_ch + 1, nz)
            images.append(_nanmedian(data[lo:hi, y0:y1, x0:x1]))
        k = nearest_channel_index(wave, halpha_A)
        lo, hi = max(k - halpha_halfwidth_ch, 0), min(k + halpha_halfwidth_ch + 1, nz)
        images.append(_nanmedian(data[lo:hi, y0:y1, x0:x1]))
        bunit = str(hdu.header.get("BUNIT", ""))

    if row.center_yx is not None:
        center_yx = (row.center_yx[0] - y0, row.center_yx[1] - x0)
    else:
        # Sin encuadre declarado: se localiza la primaria como lo hace el QC de
        # cubo, sobre la propia mediana ya calculada.
        center_yx = tuple(float(v) for v in detect_primary_yx(median[None, :, :]))

    return PanelSet(
        label=row.label, images=images, center_yx=center_yx,
        wavelengths_A=(float("nan"), *centers, float(halpha_A)),
        exptime=exptime, bunit=bunit,
    )


def _ring_values(images, centers, r_in_px: float, r_out_px: float) -> np.ndarray:
    """Píxeles finitos del anillo `r_in ≤ r < r_out` de todas las imágenes."""

    values = []
    for image, center in zip(images, centers):
        image = np.asarray(image, dtype=np.float64)
        if image.ndim != 2 or center is None:
            continue
        yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
        radius = np.hypot(yy - float(center[0]), xx - float(center[1]))
        ring = (radius >= r_in_px) & (radius < r_out_px) & np.isfinite(image)
        if np.any(ring):
            values.append(image[ring])
    return np.concatenate(values) if values else np.empty(0)


def halo_scale(
    images,
    centers,
    *,
    r_in_px: float = 5.0,
    r_out_px: float = 60.0,
    pct: tuple[float, float] = (1.0, 99.0),
) -> tuple[float, float]:
    """(vmin, vmax) a partir del anillo del halo de todas las imágenes dadas."""

    stacked = _ring_values(images, centers, r_in_px, r_out_px)
    if stacked.size == 0:
        return 0.0, 1.0
    vmin, vmax = (float(v) for v in np.percentile(stacked, pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def halo_norm(
    images,
    centers,
    *,
    r_in_px: float = 5.0,
    r_out_px: float = 60.0,
    high_pct: float = 99.9,
    linear_width_sigma: float = 5.0,
):
    """Norma asinh «centrada en el halo», compartida por un grupo de imágenes.

    El rango dinámico de estos cubos es brutal: en un cubo real de ROXs 42B b el
    pico vale ~70000, el halo a 5–20 px ~260 y a 20–60 px ~14. Con una escala
    lineal, elijas el corte que elijas, o se ve el núcleo o se ve el halo. La
    norma `asinh` es lineal cerca de cero —con anchura fijada por el **ruido de
    fondo medido en el anillo exterior**, no por un número inventado— y
    logarítmica por encima, así que enseña a la vez el halo AO, las aspas de
    difracción y las fuentes puntuales del campo.

    No se usa `LogNorm`: el 24 % de los píxeles de estos paneles no es positivo.
    """

    from matplotlib.colors import AsinhNorm

    ring = _ring_values(images, centers, r_in_px, r_out_px)
    outer = _ring_values(images, centers, r_out_px, 1.5 * r_out_px)
    if outer.size < 32:
        outer = _ring_values(images, centers, 0.5 * (r_in_px + r_out_px), r_out_px)
    if ring.size == 0:
        return AsinhNorm(linear_width=1.0, vmin=0.0, vmax=1.0)

    if outer.size:
        sigma = 0.5 * float(np.percentile(outer, 84) - np.percentile(outer, 16))
    else:
        sigma = 0.5 * float(np.percentile(ring, 84) - np.percentile(ring, 16))
    sigma = sigma if np.isfinite(sigma) and sigma > 0 else 1.0

    vmax = float(np.percentile(ring, high_pct))
    vmin = -3.0 * sigma
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + max(sigma, 1.0)
    return AsinhNorm(linear_width=linear_width_sigma * sigma, vmin=vmin, vmax=vmax)


# --------------------------------------------------------------------------
# Caché de paneles
# --------------------------------------------------------------------------
def panel_cache_path(run_id: str, *, project_root=None) -> Path:
    root = _root(project_root)
    return root / "runs" / str(run_id) / "tables" / "a1_cube_panels.npz"


def _is_legacy(run_id: str, root: Path) -> bool:
    try:
        return bool((_payload(run_id, root).get("meta") or {}).get("legacy"))
    except A1ReviewError:
        return False


def build_panels(
    run_id: str,
    rows,
    *,
    project_root=None,
    use_cache: bool = True,
    progress=None,
    **panel_kwargs,
) -> list[PanelSet]:
    """Paneles de todas las filas, con caché en `runs/<run>/tables/`.

    El cálculo cuesta ~2 min la primera vez; la caché hace que reejecutar el
    notebook sea instantáneo. No se escribe nada en un run marcado `legacy`.
    """

    root = _root(project_root)
    halpha_A, _rest, _rv = halpha_observed_A(run_id, project_root=root)
    cache_file = panel_cache_path(run_id, project_root=root)
    key = "|".join(f"{r.path}:{os.path.getmtime(r.path):.0f}" for r in rows if r.exists)

    if use_cache and cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=False) as blob:
                if str(blob["key"]) == key and float(blob["halpha_A"]) == halpha_A:
                    return _panels_from_cache(blob, rows)
        except (OSError, ValueError, KeyError):
            pass

    panels = []
    for index, row in enumerate(rows, 1):
        if progress is not None:
            progress(index, len(rows), row.label)
        panels.append(cube_panels(row, halpha_A=halpha_A, **panel_kwargs))

    if use_cache and panels and not _is_legacy(run_id, root):
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {"key": np.array(key), "halpha_A": np.array(halpha_A),
                       "labels": np.array([p.label for p in panels])}
            for i, panel in enumerate(panels):
                payload[f"center_{i}"] = np.array(panel.center_yx, dtype=np.float64)
                payload[f"waves_{i}"] = np.array(panel.wavelengths_A, dtype=np.float64)
                for j, image in enumerate(panel.images):
                    payload[f"img_{i}_{j}"] = np.asarray(image, dtype=np.float32)
            np.savez_compressed(cache_file, **payload)
        except OSError:
            pass
    return panels


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
#: Color de cada estado en el esquema. `resumed` y `reused` no son lo mismo:
#: reutilizado = la cascada aceptó un producto de archivo; reanudado = el paso
#: ya estaba hecho en una ejecución anterior y no volvió a cobrarse.
STATUS_STYLE = {
    "ok": ("#d7ecd9", "#2e7d32", "ejecutado"),
    "reused": ("#d9e8f6", "#1565c0", "reutilizado de archivo"),
    "resumed": ("#fdf0d0", "#e08c00", "reanudado (sin coste aquí)"),
    "gate": ("#ece0f3", "#6a1b9a", "puerta humana"),
    "rejected": ("#f6dcdc", "#b71c1c", "descartado"),
    "pending": ("#ebebeb", "#8a8a8a", "no ejecutado"),
}


def save_figure(fig, run_id: str, name: str, *, project_root=None, dpi: int = 200) -> Path | None:
    """Guarda una figura en `runs/<run>/plots/`, nunca en un run `legacy`."""

    root = _root(project_root)
    if _is_legacy(run_id, root):
        return None
    path = root / "runs" / str(run_id) / "plots" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _wrapped(text: str, width: int, max_lines: int) -> str:
    """Texto envuelto a `width` columnas y recortado a `max_lines` líneas."""

    import textwrap

    if not text:
        return ""
    lines = textwrap.wrap(text, width=width, break_long_words=False) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(width - 1, 1)] + "…"
    return "\n".join(lines)


def _io_line(items, arrow: str, *, width: int = 30, max_lines: int = 3) -> str:
    parts = []
    for io in items:
        text = io.tag + (f" {io.count}" if io.count else "")
        parts.append(f"({text})" if io.optional else text)
    return _wrapped(arrow + " " + ", ".join(parts), width, max_lines) if parts else ""


def draw_phase_schematic(phases, *, ax=None, box_w: float = 1.15, box_h: float = 0.62):
    """Esquema de la reducción estilo esoreflex: carriles, cajas y flechas.

    Un carril por etapa del flujo (P1 por noche, P2 por exposición, P3
    combinado). Cada caja lleva su receta, su duración real y, alrededor, las
    **entradas** (arriba) y las **salidas** (abajo) que declara el SOF.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    lanes: list[str] = []
    for phase in phases:
        if phase.lane not in lanes:
            lanes.append(phase.lane)
    by_lane = {lane: [p for p in phases if p.lane == lane] for lane in lanes}
    width = max(len(v) for v in by_lane.values())
    pitch_x, pitch_y = 3.5, 2.9
    x_left = -3.4

    if ax is None:
        span_x = (width - 1) * pitch_x + 2 * box_w - x_left + 1.0
        _fig, ax = plt.subplots(figsize=(0.72 * span_x, 0.72 * (len(lanes) * pitch_y + 1.6)))

    centers: dict[str, tuple[float, float]] = {}
    for row, lane in enumerate(lanes):
        y = -row * pitch_y
        ax.text(x_left, y, lane.replace(" · ", "\n"), ha="left", va="center",
                fontsize=8.5, weight="bold", color="#37474f")
        for col, phase in enumerate(by_lane[lane]):
            x = col * pitch_x
            fill, edge, _label = STATUS_STYLE.get(phase.status, STATUS_STYLE["pending"])
            ax.add_patch(FancyBboxPatch(
                (x - box_w, y - box_h), 2 * box_w, 2 * box_h,
                boxstyle="round,pad=0.06", linewidth=1.4, facecolor=fill, edgecolor=edge))
            ax.text(x, y + 0.22, _wrapped(phase.recipe or phase.title, 22, 2),
                    ha="center", va="center", fontsize=7.2, weight="bold", color="#1b1b1b")
            minutes = phase.duration_min
            sub = f"{minutes:.0f} min" if minutes else (phase.detail or "—")
            ax.text(x, y - 0.26, _wrapped(sub, 26, 2), ha="center", va="center",
                    fontsize=6.4, color=edge, style="italic")
            ax.text(x, y + box_h + 0.12, _io_line(phase.inputs, "↓"), ha="center",
                    va="bottom", fontsize=5.7, color="#455a64", linespacing=1.25)
            ax.text(x, y - box_h - 0.12, _io_line(phase.outputs, "→"), ha="center",
                    va="top", fontsize=5.7, color="#455a64", linespacing=1.25)
            centers[phase.key] = (x, y)
            if col:
                previous = by_lane[lane][col - 1]
                ax.add_patch(FancyArrowPatch(
                    (centers[previous.key][0] + box_w, y), (x - box_w, y),
                    arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color="#78909c"))
        if row:
            first_prev = centers[by_lane[lanes[row - 1]][0].key]
            ax.add_patch(FancyArrowPatch(
                (first_prev[0], first_prev[1] - box_h - 0.80),
                (centers[by_lane[lane][0].key][0], y + box_h + 0.80),
                arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color="#546e7a"))

    used = {p.status for p in phases}
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                          markerfacecolor=STATUS_STYLE[s][0],
                          markeredgecolor=STATUS_STYLE[s][1], label=STATUS_STYLE[s][2])
               for s in STATUS_STYLE if s in used]
    ax.legend(handles=handles, loc="upper center", fontsize=7.5, frameon=False,
              ncol=len(handles), bbox_to_anchor=(0.5, 1.02))
    ax.set_xlim(x_left - 0.2, (width - 1) * pitch_x + box_w + 0.9)
    ax.set_ylim(-(len(lanes) - 1) * pitch_y - 1.7, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax.figure


#: Las seis verificaciones de A1: (etiqueta, clave del **veredicto** booleano si
#: la hay, prefijo de sus claves, descripción). V2/V4/V5 no publican booleano en
#: la variante `cascade`: solo un número. Tratar ese número como veredicto era
#: el error obvio —un residuo de 0.00122 es «verdadero» en Python y no significa
#: que la verificación haya pasado—, así que se informan como *medidas*.
VERIFICATION_LABELS = (
    ("V1", "v1_stat_present", "v1_", "STAT presente, positiva y con pocos NaN"),
    ("V2", None, "v2_", "Residuo del estándar (respuesta de flujo)"),
    ("V3", "v3_wcs_ok", "v3_", "WCS y eje espectral"),
    ("V4", None, "v4_", "Luz blanca vs ADP de ESO"),
    ("V5", None, "v5_", "Espectro estelar vs ADP"),
    ("V6", "v6_sky_mask_clean", "v6_", "Máscara de cielo limpia"),
)

#: Claves que acompañan a una verificación sin ser su resultado.
_VERIFICATION_ANNEX = ("_plot", "_method", "_aperture_radius_px")


def a1_qc_payload(run_id: str, *, project_root=None) -> dict:
    """`stage00r_qc.json` del run que ejecutó A1 (o `{}`)."""

    root = _root(project_root)
    path = root / "runs" / resolve_a1_run(run_id, project_root=root) / "stages" / "stage00r_qc.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def is_qc_skeleton(payload) -> bool:
    """¿El QC de A1 es el esqueleto vacío que escribe el driver?

    `runs/ROXs42Bb_raw/stages/stage00r_qc.json` tiene `v1_stat_present: false`,
    y eso NO quiere decir que V1 fallara: quiere decir que nadie la corrió. Sin
    recetas, sin puertas y sin estado, el fichero es una plantilla.
    """

    if not payload:
        return True
    return (not payload.get("recipes") and not payload.get("gates_passed")
            and not payload.get("status"))


def verification_status(run_id: str, *, project_root=None) -> list[dict]:
    """Estado real de V1–V6 en el QC de A1 de ESTE objeto.

    El QC tiene dos formas según la vía de reducción (`qc_schema_variant`): en
    `monolithic` cada verificación es un dict `{ok, status, message}` y en
    `cascade` un escalar. Se sirven las dos, y lo que no está medido se dice
    «no ejecutada», no «falló».
    """

    payload = a1_qc_payload(run_id, project_root=project_root)
    verification = payload.get("verification") or {}
    skeleton = is_qc_skeleton(payload)

    out = []
    for tag, verdict_key, prefix, label in VERIFICATION_LABELS:
        keys = [k for k in verification if k.startswith(prefix)]
        values = {k: verification[k] for k in keys
                  if not any(k.endswith(sfx) for sfx in _VERIFICATION_ANNEX)}
        entry = {"tag": tag, "label": label, "values": values}
        if not keys:
            entry.update(status="no medido", message="ausente en el QC de esta reducción")
        elif skeleton:
            entry.update(status="no ejecutada",
                         message="el QC de A1 es el esqueleto vacío (sin recetas ni puertas)")
        else:
            primary = verification.get(verdict_key) if verdict_key in verification else None
            if primary is None and values:
                primary = next(iter(values.values()))
            if isinstance(primary, dict):
                entry.update(
                    status="ok" if primary.get("ok") else str(primary.get("status") or "?"),
                    message=str(primary.get("message") or ""))
            elif primary is None:
                entry.update(status="unavailable", message="sin medir en esta reducción")
            elif verdict_key in verification and isinstance(verification[verdict_key], bool):
                entry.update(
                    status="ok" if verification[verdict_key] else "no",
                    message=", ".join(f"{k}={v}" for k, v in values.items() if k != verdict_key))
            else:
                entry.update(status="medido",
                             message=", ".join(f"{k}={v}" for k, v in values.items()))
        out.append(entry)
    return out


def _verification_plot_paths(run_id: str, root: Path) -> dict[str, Path]:
    """Los PNG de evidencia que la reducción ya dejó escritos, si existen."""

    a1_run = resolve_a1_run(run_id, project_root=root)
    plots_dir = root / "runs" / a1_run / "plots"
    wanted = {
        "V1": "stage00r_v1_nan_map.png",
        "V4": "stage00r_v4_adp_whitelight_psf_matched.png",
        "V6": "stage00r_v6_sky_mask_example.png",
    }
    found = {}
    for tag, name in wanted.items():
        path = plots_dir / name
        if path.exists():
            found[tag] = path
    return found


def draw_verification_panel(run_id: str, *, project_root=None, fig=None):
    """Figura 2×2 con la evidencia de las verificaciones de A1.

    Arriba, las que dejaron imagen: el mapa de NaN de V1 y el **ajuste** de V4
    (el PSF-matching contra el ADP de ESO). Abajo, las dos que están medidas en
    los dos objetos porque las produce el combinado: los desplazamientos por
    exposición y la dispersión de `CRVAL3`. Así la figura nunca sale vacía,
    aunque V1–V6 no se hayan corrido —el caso de ROXs 42B b.
    """

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    root = _root(project_root)
    plan = combine_plan(run_id, project_root=root) or {}
    qc = combine_qc(run_id, project_root=root) or {}
    exposures = plan.get("exposures") or qc.get("exposures") or []
    images = _verification_plot_paths(run_id, root)

    if fig is None:
        fig = plt.figure(figsize=(12.4, 8.6))
    axes = fig.subplots(2, 2)

    for ax, tag, title in ((axes[0][0], "V1", "V1 · mapa de NaN del cubo"),
                           (axes[0][1], "V4", "V4 · ajuste PSF-matching vs ADP")):
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if tag in images:
            ax.imshow(mpimg.imread(str(images[tag])))
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, f"{tag} no medida en esta reducción\n"
                              "(el QC de A1 es un esqueleto en perfil cascade)",
                    ha="center", va="center", fontsize=9, color="#8a8a8a",
                    transform=ax.transAxes)

    ax = axes[1][0]
    if exposures:
        nights = sorted({str(e.get("exposure_id", ""))[:10] for e in exposures})
        for night in nights:
            sel = [e for e in exposures if str(e.get("exposure_id", "")).startswith(night)]
            ax.scatter([e.get("shift_x", 0.0) for e in sel],
                       [e.get("shift_y", 0.0) for e in sel], s=26, alpha=0.85, label=night)
        repeat = ((qc.get("reference") or plan.get("reference") or {})
                  .get("alignment_repeatability") or {}).get("centroid_repeatability_px")
        if repeat:
            ax.add_patch(plt.Circle((0, 0), float(repeat), fill=False, linestyle="--",
                                    edgecolor="#b71c1c", linewidth=1.2,
                                    label=f"repetibilidad {float(repeat):.3f} px"))
        ax.axhline(0, color="#cfd8dc", linewidth=0.8, zorder=0)
        ax.axvline(0, color="#cfd8dc", linewidth=0.8, zorder=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7.5, frameon=False)
    else:
        ax.text(0.5, 0.5, "sin plan de combinado", ha="center", va="center",
                transform=ax.transAxes, color="#8a8a8a")
    ax.set_title("Alineado · desplazamiento aplicado por exposición", fontsize=10)
    ax.set_xlabel("shift_x [px]", fontsize=9)
    ax.set_ylabel("shift_y [px]", fontsize=9)

    ax = axes[1][1]
    crval = [e.get("crval3") for e in exposures if e.get("crval3") is not None]
    if crval:
        step = float((qc.get("wavelength") or {}).get("cd3_3") or 1.25)
        offsets = (np.asarray(crval, dtype=float) - float(np.median(crval))) / step
        ax.plot(range(1, len(offsets) + 1), offsets, "o-", markersize=4,
                color="#1565c0", linewidth=0.9)
        ax.axhline(0.0, color="#cfd8dc", linewidth=0.8)
        spread = (qc.get("wavelength") or {}).get("crval3_spread_channels")
        ax.set_title("V3 · CRVAL3 por exposición"
                     + (f" (dispersión {float(spread):.4f} canales)" if spread else ""),
                     fontsize=10)
    else:
        ax.text(0.5, 0.5, "sin CRVAL3 por exposición", ha="center", va="center",
                transform=ax.transAxes, color="#8a8a8a")
        ax.set_title("V3 · CRVAL3 por exposición", fontsize=10)
    ax.set_xlabel("exposición", fontsize=9)
    ax.set_ylabel("CRVAL3 − mediana [canales]", fontsize=9)

    fig.tight_layout()
    return fig


def draw_cube_mosaic(
    panels,
    rows=None,
    *,
    columns=PANEL_COLUMNS,
    r_in_px: float = 5.0,
    r_out_px: float = 60.0,
    row_height_in: float = 1.85,
    cmap: str = "magma",
):
    """El mosaico N×5, con la escala del halo compartida por columna.

    Compartir vmin/vmax entre todas las filas de una columna es lo que convierte
    el mosaico en un diagnóstico: si una exposición entró con mal seeing, con el
    halo desplazado o con menos flujo, se ve al comparar su fila con las demás.
    Autoescalar cada panel lo escondería.
    """

    import matplotlib.pyplot as plt

    n = len(panels)
    if n == 0:
        raise A1ReviewError("No hay cubos que dibujar.")
    n_cols = len(columns)
    fig, axes = plt.subplots(n, n_cols, figsize=(2.35 * n_cols, row_height_in * n),
                             squeeze=False)

    norms = []
    for col in range(n_cols):
        images = [p.images[col] for p in panels if col < len(p.images)]
        centers = [p.center_yx for p in panels if col < len(p.images)]
        norms.append(halo_norm(images, centers, r_in_px=r_in_px, r_out_px=r_out_px))

    for i, panel in enumerate(panels):
        row = rows[i] if rows is not None and i < len(rows) else None
        for col in range(n_cols):
            ax = axes[i][col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col >= len(panel.images):
                ax.axis("off")
                continue
            ax.imshow(panel.images[col], origin="lower", cmap=cmap, norm=norms[col])
            if i == 0:
                wave = panel.wavelengths_A[col] if col < len(panel.wavelengths_A) else float("nan")
                header = columns[col] + ("" if not np.isfinite(wave) else f"\n{wave:.0f} Å")
                ax.set_title(header, fontsize=8.5, pad=5)
            if col == 0:
                flag = ""
                if row is not None and (not row.in_bounds or row.centroid_fallback):
                    flag = "  ⚠"
                label = panel.label
                if len(label) > 26:
                    label = label[:12] + "…" + label[-13:]
                extra = f"\n{panel.exptime:.0f} s" if panel.exptime else ""
                ax.set_ylabel(label + extra + flag, fontsize=6.6, rotation=0,
                              ha="right", va="center", labelpad=6)

    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.14, right=0.99, top=0.97, bottom=0.01)
    return fig


def _panels_from_cache(blob, rows) -> list[PanelSet]:
    labels = [str(v) for v in blob["labels"]]
    panels = []
    for i, label in enumerate(labels):
        images = []
        j = 0
        while f"img_{i}_{j}" in blob:
            images.append(blob[f"img_{i}_{j}"])
            j += 1
        row = rows[i] if i < len(rows) else None
        panels.append(PanelSet(
            label=label, images=images,
            center_yx=tuple(float(v) for v in blob[f"center_{i}"]),
            wavelengths_A=tuple(float(v) for v in blob[f"waves_{i}"]),
            exptime=row.exptime if row is not None else None,
            note="desde caché",
        ))
    return panels
