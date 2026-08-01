"""Utilidades compartidas por los notebooks de revisión A1..G5.

Deliberadamente ligero (solo stdlib): cada notebook debe poder abrirse y
auditar el QC de un run sin importar toda la pila científica. Las etapas
*ejecutables* sí importan `musepipe` en su propia celda (guardada por RUN).

Resolución del run:
  1. argumento explícito a `resolve_run_id` (lo que hace cada notebook)
  2. variable de entorno `MUSE_RUN_ID`
  3. sin más: **no hay default**. Un default apuntando al primer objeto reducido
     hacía que cualquier notebook sin run explícito lo auditara en silencio.

OJO: a diferencia de `musepipe.config.get_run_id`, aquí NO se lee
`active_run.txt` — los notebooks de revisión auditan el run que declaran,
mientras que `active_run.txt` suele apuntar a un run de smoke-test. Si ambos
difieren, `resolve_run_id` lo avisa por stdout.
Todos los comandos canónicos de los notebooks pasan `--run-id` explícito,
así que la ejecución (RUN=True) nunca depende de `active_run.txt`.

Resolución del QC (manifiesto `chain`)
-------------------------------------
La cadena de un objeto puede repartirse entre varios runs: en ROXs 42B b, A1
corrió en `ROXs42Bb_raw` y el resto en `ROXs42Bb_realigned`. El run *activo*
declara ese reparto en la clave `chain` de su `config/config.json`:

    "chain": {"target": "ROXs42Bb", "default_run": "ROXs42Bb_realigned",
              "stage_runs": {"A1": "ROXs42Bb_raw"},
              "reduction_profile": "cascade"}

`load_qc` usa el registro de etapas (`musepipe.stage_registry`) para saber a qué
etapa pertenece una ruta de QC, consulta `chain.stage_runs` y prueba también los
alias del QC (p.ej. `cube_telcorr_qc.json` para A1 en el perfil cascade). Cuando
resuelve en un run distinto del activo lo **dice**; y si ese run pertenece a otro
objeto, avisa en rojo (contaminación cross-object).

Plan y contexto: `docs/plan_multiobjeto_notebooks_2026-07-24.md`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# NO hay run por defecto: un default apuntando al primer objeto reducido hacía
# que cualquier notebook sin run explícito auditara ese objeto en silencio. Cada
# notebook declara el suyo en la celda de setup (`resolve_run_id('<run>')`), o se
# fija con la variable de entorno MUSE_RUN_ID.
DEFAULT_RUN_ID = None

#: run activo del notebook, fijado por `resolve_run_id`. Permite distinguir
#: "el run de este notebook" de un override explícito pasado a `load_qc`.
_ACTIVE_RUN: str | None = None


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
    global _ACTIVE_RUN
    resolved = run_id or os.environ.get("MUSE_RUN_ID") or DEFAULT_RUN_ID
    if not resolved:
        raise ValueError(
            "No hay run que auditar: pasa el run a resolve_run_id('<run>') o "
            "exporta MUSE_RUN_ID. No hay default: apuntaría al objeto equivocado."
        )
    active = _active_run_txt()
    if active and active != resolved:
        print(
            f"AVISO: active_run.txt apunta a {active!r} pero este notebook audita "
            f"{resolved!r}. Los CLI de musepipe SIN --run-id usarían {active!r}."
        )
    _ACTIVE_RUN = resolved
    return resolved


def _effective_run(run_id: str | None = None) -> str:
    """Run a usar, SIN tocar el run activo ni volver a avisar de active_run.txt.

    `run_id=None` significa "el run activo de este notebook", no "re-resolver
    desde cero": si `resolve_run_id` ya corrió en la celda de setup, se respeta.
    Solo `resolve_run_id` fija `_ACTIVE_RUN`; las funciones de lectura nunca lo
    mutan (si lo hicieran, un override explícito se convertiría en el run activo
    y el aviso cross-object nunca dispararía).
    """
    resolved = (run_id or _ACTIVE_RUN or os.environ.get("MUSE_RUN_ID")
                or DEFAULT_RUN_ID)
    if not resolved:
        raise ValueError(
            "No hay run activo: ejecuta antes la celda de setup "
            "(`RUN_ID = nb.resolve_run_id('<run>')`) o exporta MUSE_RUN_ID."
        )
    return resolved


def run_dir(run_id: str | None = None) -> Path:
    return project_root() / "runs" / _effective_run(run_id)


# --------------------------------------------------------------------------
# Registro de etapas y manifiesto de cadena
# --------------------------------------------------------------------------
def _registry():
    """Importa `musepipe.stage_registry` (solo stdlib) de forma perezosa.

    Perezosa porque la raíz del repo puede no estar en sys.path al importar este
    módulo (p.ej. con el cwd en `notebooks/<objeto>/`).
    """
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from musepipe import stage_registry

        return stage_registry
    except ImportError:  # pragma: no cover - fallback si falta musepipe
        return None


def _config_payload(run_id: str) -> dict:
    path = project_root() / "runs" / run_id / "config" / "config.json"
    if not path.exists():
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def chain_of(run_id: str | None = None) -> dict:
    """Manifiesto `chain` del run (dict vacío si el run no lo declara)."""
    rid = run_id or _ACTIVE_RUN or DEFAULT_RUN_ID
    if not rid:
        return {}
    chain = _config_payload(rid).get("chain")
    return chain if isinstance(chain, dict) else {}


def _normalize(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def run_target(run_id: str, *, infer: bool = True) -> str | None:
    """Objeto al que pertenece un run.

    Orden: `chain.target` -> `config.target_name` -> (si `infer`) el prefijo del
    nombre del run según la convención `<objeto>_<variante>`. Los runs `_raw` no
    declaran objeto, así que sin la inferencia se mostrarían como '?'.
    """
    payload = _config_payload(run_id)
    chain = payload.get("chain")
    if isinstance(chain, dict) and chain.get("target"):
        return str(chain["target"])
    target = payload.get("config", {}).get("target_name")
    if target:
        return str(target)
    if infer and "_" in run_id:
        return run_id.split("_", 1)[0]
    return run_id if infer else None


def display_name(run_id: str | None = None) -> str:
    """Nombre legible del objeto de un run (`targets/<slug>.json` -> `display_name`).

    Para títulos y prosa: 'ROXs 42B b' en vez del slug 'ROXs42Bb'. Si el objeto no
    tiene ficha en `targets/`, devuelve el slug tal cual (nunca inventa un nombre,
    y nunca cae al del primer objeto).
    """
    slug = run_target(_effective_run(run_id))
    if not slug:
        return str(run_id or "")
    path = project_root() / "targets" / f"{slug}.json"
    try:
        with open(path) as fh:
            return str(json.load(fh).get("display_name") or slug)
    except (OSError, json.JSONDecodeError):
        return slug


def _run_object_prefix(run_id: str) -> str:
    """Objeto según la convención de nombres de runs: `<objeto>_<variante>`."""
    return _normalize(run_id.split("_", 1)[0])


def _is_cross_object(candidate_run: str, active_run: str) -> bool:
    """True si `candidate_run` pertenece a OTRO objeto que `active_run`.

    Se compara el **prefijo del nombre del run**, no `target_name` del config:
    los runs `_raw` no declaran objeto y los que lo declaran no son consistentes
    (`ROXs12b_B_adp` lleva `target_name='ROX12b'`, sin la 's'), lo que daría
    falsos positivos. Un aviso que salta cuando no debe se acaba ignorando, y
    entonces no sirve para nada.
    """
    return _run_object_prefix(candidate_run) != _run_object_prefix(active_run)


def _candidates(relpath: str, run_id: str) -> list[tuple[str, str, str]]:
    """[(run, ruta, motivo)] a probar, en orden de preferencia."""
    reg = _registry()
    stage = reg.stage_for_qc(relpath) if reg else None
    paths = list(stage.qc_paths) if stage else [relpath]
    if relpath in paths:  # la ruta pedida siempre primero
        paths.remove(relpath)
    paths.insert(0, relpath)

    chain = chain_of(run_id)
    runs: list[tuple[str, str]] = [(run_id, "run activo")]
    if stage is not None:
        override = (chain.get("stage_runs") or {}).get(stage.id)
        if override and override != run_id:
            runs.insert(0, (override, f"chain.stage_runs[{stage.id!r}]"))
    default_run = chain.get("default_run")
    if default_run and default_run not in {r for r, _ in runs}:
        runs.append((default_run, "chain.default_run"))

    # Precedencia: la ruta PEDIDA en todos los runs candidatos, y solo si no
    # está en ninguno se prueban los alias. Al contrario (run primero) un alias
    # podría devolver un QC de otro esquema sin que se haya pedido.
    out = []
    for i, path in enumerate(paths):
        for run, why in runs:
            reason = why if i == 0 else f"{why} + alias del registro (pedido: {relpath})"
            out.append((run, path, reason))
    return out


def _sibling_runs_with(relpath: str, exclude: set[str]) -> list[str]:
    """Runs hermanos donde sí existe `relpath` (para el mensaje de error)."""
    runs_dir = project_root() / "runs"
    if not runs_dir.exists():
        return []
    found = []
    for entry in sorted(runs_dir.iterdir()):
        if entry.name in exclude:
            continue
        try:
            if (entry / relpath).exists():
                found.append(entry.name)
        except OSError:
            continue
    return found


def _stage_label(relpath: str) -> str:
    reg = _registry()
    stage = reg.stage_for_qc(relpath) if reg else None
    return f"{stage.id} ({stage.slug})" if stage else "etapa desconocida"


def resolve_qc(relpath: str, run_id: str | None = None) -> tuple[Path, str, str]:
    """Localiza el QC de una etapa. -> (ruta, run donde está, motivo).

    Lanza `FileNotFoundError` si no está en ningún run de la cadena.
    """
    active = _ACTIVE_RUN or _effective_run(None)
    rid = _effective_run(run_id)
    explicit_override = run_id is not None and run_id != active

    for run, path, why in _candidates(relpath, rid):
        full = project_root() / "runs" / run / path
        if not full.exists():
            continue
        # --- F3b: aviso de contaminación cross-object -------------------
        if _is_cross_object(run, active):
            print(
                f"\n⚠️  CROSS-OBJECT: {_stage_label(relpath)} se está leyendo de "
                f"runs/{run} (objeto {run_target(run) or '?'}), pero la cadena activa es "
                f"{active} (objeto {run_target(active) or chain_of(active).get('target') or '?'}).\n"
                f"    Ruta: {path}\n"
                f"    Los números de abajo NO son de este objeto. Ver H1 en "
                f"docs/plan_multiobjeto_notebooks_2026-07-24.md.\n"
            )
        elif run != active:
            note = "override explícito" if explicit_override else why
            print(f"[procedencia] {_stage_label(relpath)}: runs/{run}/{path}  ({note})")
        return full, run, why

    # --- no está en ningún candidato: mensaje diferenciado --------------
    tried = {run for run, _, _ in _candidates(relpath, rid)}
    reg = _registry()
    stage = reg.stage_for_qc(relpath) if reg else None
    siblings = _sibling_runs_with(relpath, exclude=tried)
    lines = [f"No existe el QC de {_stage_label(relpath)} para la cadena de {rid!r}.",
             f"  ruta pedida : {relpath}",
             f"  runs probados: {', '.join(sorted(tried))}"]
    if stage is not None and stage.qc_aliases:
        lines.append(f"  alias probados: {', '.join(stage.qc_aliases)}")
    if siblings:
        lines += [
            f"  -> SÍ existe en: {', '.join('runs/' + s for s in siblings)}",
            f"     Si pertenece a esta cadena, declara el override en el config del run activo:",
            f'       runs/{rid}/config/config.json -> chain.stage_runs["{stage.id if stage else "<ID>"}"] = "{siblings[0]}"',
            "     Si pertenece a OTRO objeto, no lo uses: contaminarías el resultado.",
        ]
    else:
        lines += [
            "  -> No existe en ningún run del repo: la etapa NO se ha ejecutado para este objeto.",
            "     Ejecútala con la celda 'Ejecutar o auditar' (RUN=True) o con el comando de",
            "     'Cómo ejecutar de forma independiente', y re-corre esta celda.",
        ]
    raise FileNotFoundError("\n".join(lines))


def load_qc(relpath: str, run_id: str | None = None) -> dict:
    """Carga un JSON de QC resolviéndolo por el manifiesto `chain` del run.

    `relpath` es relativo al directorio del run, p.ej. `stages/stage_e01_qc.json`
    o `report/run_summary.json`.
    """
    path, _run, _why = resolve_qc(relpath, run_id)
    with open(path) as fh:
        return json.load(fh)


def load_qc_optional(relpath: str, run_id: str | None = None) -> dict | None:
    """Como `load_qc`, pero devuelve None e informa en vez de lanzar.

    Para etapas que legítimamente pueden no haberse ejecutado en un objeto: el
    notebook se ejecuta de principio a fin y el hueco queda visible.
    """
    try:
        return load_qc(relpath, run_id)
    except FileNotFoundError as exc:
        print(f"[etapa pendiente] {exc}")
        return None


def qc_path(relpath: str, run_id: str | None = None) -> Path:
    """Ruta del QC resuelta por la cadena (sin exigir que exista)."""
    try:
        return resolve_qc(relpath, run_id)[0]
    except FileNotFoundError:
        return run_dir(run_id) / relpath


# --------------------------------------------------------------------------
# Panel de la cadena (se muestra en el primer notebook, A1)
# --------------------------------------------------------------------------
def stage_status(run_id: str | None = None) -> list[dict]:
    """Estado de cada etapa registrada para la cadena del run activo."""
    rid = _effective_run(run_id)
    reg = _registry()
    if reg is None:
        return []
    rows = []
    for stage in reg.STAGES:
        # Si la etapa no declara QC canónico (A2/A3), se usa su primer alias:
        # el panel debe reflejar el estado real, no un hueco administrativo.
        probe = stage.qc or (stage.qc_paths[0] if stage.qc_paths else None)
        row = dict(id=stage.id, block=stage.block, slug=stage.slug,
                   qc=probe, run=None, estado="sin QC registrado", mtime="")
        if probe is None:
            rows.append(row)
            continue
        try:
            path, run, _why = resolve_qc(probe, rid)
        except FileNotFoundError:
            row["estado"] = "no ejecutada"
            rows.append(row)
            continue
        row["run"] = run
        if _is_cross_object(run, rid):
            row["estado"] = "CROSS-OBJECT"
        elif run != rid:
            row["estado"] = "otro run (declarado)"
        else:
            row["estado"] = "ok"
        try:
            import datetime as _dt

            row["mtime"] = _dt.datetime.utcfromtimestamp(
                path.stat().st_mtime).strftime("%Y-%m-%d")
        except OSError:
            pass
        rows.append(row)
    return rows


def show_chain(run_id: str | None = None) -> None:
    """Imprime el mapa completo de la cadena: etapa -> run -> estado -> fecha."""
    rid = _effective_run(run_id)
    chain = chain_of(rid)
    print(f"== Cadena de {chain.get('target') or run_target(rid) or '?'} "
          f"(run activo: {rid}) ==")
    if not chain:
        print("  (este run no declara 'chain' en config/config.json — "
              "todo se resuelve contra el run activo)")
    else:
        print(f"  default_run       : {chain.get('default_run', rid)}")
        print(f"  reduction_profile : {chain.get('reduction_profile', '(no declarado)')}")
        overrides = chain.get("stage_runs") or {}
        print(f"  stage_runs        : {overrides or '(ninguno)'}")
    rows = stage_status(rid)
    if not rows:
        print("  (registro de etapas no disponible)")
        return
    print()
    print(f"  {'et':5s} {'bl':3s} {'run que la contiene':24s} {'estado':22s} QC")
    for r in rows:
        mark = {"ok": " ", "otro run (declarado)": "→", "no ejecutada": "·",
                "CROSS-OBJECT": "!"}.get(r["estado"], " ")
        print(f"{mark} {r['id']:5s} {r['block']:3s} {str(r['run'] or '-'):24s} "
              f"{r['estado']:22s} {r['mtime']}")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["estado"]] = counts.get(r["estado"], 0) + 1
    print("\n  resumen:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    pend = [r["id"] for r in rows if r["estado"] == "no ejecutada"]
    if pend:
        print(f"  etapas pendientes para este objeto: {', '.join(pend)}")


def provenance_line(relpath: str, run_id: str | None = None) -> str:
    """Una línea con la procedencia del QC de esta etapa (celda de setup)."""
    rid = _effective_run(run_id)
    try:
        path, run, why = resolve_qc(relpath, rid)
    except FileNotFoundError:
        return f"{_stage_label(relpath)}: QC no disponible (etapa no ejecutada para esta cadena)"
    suffix = "" if run == rid else f"  [{why}]"
    return f"{_stage_label(relpath)}: lee runs/{run}/{path.relative_to(project_root() / 'runs' / run)}{suffix}"


# --------------------------------------------------------------------------
# Comandos de las etapas de reducción (bloque A)
# --------------------------------------------------------------------------
def _sibling_setting(run_id: str, key: str):
    """Busca `key` en el config de otros runs del MISMO objeto."""
    runs_dir = project_root() / "runs"
    if not runs_dir.exists():
        return None
    for entry in sorted(runs_dir.iterdir()):
        if entry.name == run_id or _is_cross_object(entry.name, run_id):
            continue
        value = _config_payload(entry.name).get("config", {}).get(key)
        if value:
            return value
    return None


def _launch_context(run_id: str) -> dict:
    """Valores con los que se rellenan las plantillas de comando de una etapa."""
    payload = _config_payload(run_id)
    cfg = payload.get("config", {})
    root = project_root()
    ctx = {
        "run_id": run_id,
        "run_dir": str(root / "runs" / run_id),
        "stage_dir": str(root / "runs" / run_id / "stages"),
        # Los insumos de reducción son del OBJETO, no de un run concreto: si el
        # run que ejecuta la etapa no los declara, se buscan en los runs hermanos
        # del mismo objeto (p.ej. el `_raw`). Es resolución, no duplicación.
        "raw_data_dir": cfg.get("raw_data_dir") or _sibling_setting(run_id, "raw_data_dir"),
        "work_dir": cfg.get("work_dir") or _sibling_setting(run_id, "work_dir"),
        "provenance": "raw_reduction" if cfg.get("entry_point") == "raw" else "adp",
    }
    cubes = cfg.get("cube_files") or []
    ctx["cube"] = str(cubes[0]) if cubes else None
    # Posición de la primaria: la mide B3 y queda en su QC, pero **en el marco
    # RECORTADO de B1** (`stage01c_qc.json` lo declara: su `input_cube` es el
    # stack de B2, 170x170). El `{cube}` de estas plantillas es `cube_files[0]`,
    # el cubo SIN recortar de 200 px, así que hay que sumar el desfase del
    # recorte o el comando apunta 15 px fuera de la estrella — que es lo que
    # hacía, y por eso A3 se lanzó a mano en ROXs 12 b.
    try:
        pos = load_qc("stages/stage01c_qc.json", run_id)["primary"]["pos_yx"]
        bounds = (load_qc("stages/stage01_qc.json", run_id)
                  .get("crop_bounds_per_cube") or [{}])[0]
        dy, dx = float(bounds["y1"]), float(bounds["x1"])
        ctx["primary_y"], ctx["primary_x"] = float(pos[0]) + dy, float(pos[1]) + dx
    except (FileNotFoundError, KeyError, TypeError, IndexError):
        # Sin el desfase NO se emite la posición sin corregir: un número
        # plausible y falso es peor que no tener número. (`ValueError` no se
        # captura a propósito: es «no hay run activo», y eso sí hay que verlo.)
        ctx["primary_y"] = ctx["primary_x"] = None
    ctx["upstream"], ctx["upstream_qc"] = _upstream_evidence(run_id)
    return ctx


def _upstream_evidence(run_id: str) -> tuple[str | None, str | None]:
    """La etapa de la que viene el cubo y el QC que la avala, para A3.

    A3 no acepta un cubo sin procedencia: pide el QC de A2, o el de A1. Cuál de
    los dos existe **depende del objeto** —ROXs 42B b nunca corrió A2— así que la
    plantilla no puede llevar `--upstream A2` escrito a mano: para ese objeto no
    hay tal QC y el comando publicado no se puede ejecutar.

    En A1 hay dos ficheros posibles y **los elige el perfil**, que es lo que
    declara `Stage.qc_schema_variant`: en `cascade` la evidencia es el
    `cube_telcorr_qc.json` del combine (esquema `stream_combine_v1`), no el
    `stage00r_qc.json` de fases, que en cascada se queda como esqueleto vacío.
    """

    reg = _registry()
    if reg is None:
        return None, None
    profile = chain_of(run_id).get("reduction_profile", "*")
    for stage_id in ("A2", "A1"):
        stage = reg.by_id(stage_id)
        if stage is None:
            continue
        rutas = list(stage.qc_paths)
        if stage_id == "A1" and profile == "cascade":
            rutas.sort(key=lambda p: p != "cube_telcorr_qc.json")
        for rel in rutas:
            try:
                path, _run, _why = resolve_qc(rel, run_id)
            except FileNotFoundError:
                continue
            return stage_id, str(path)
    return None, None


def launch_command(stage_id: str, run_id: str | None = None) -> tuple[str, str, list[str]]:
    """Comando real de una etapa `launch` para ESTE objeto.

    -> (comando, run que lo ejecuta, marcadores sin resolver)

    El run puede no ser el activo: A1 suele correr en el run de reducción
    (`chain.stage_runs['A1']`). El perfil (`chain.reduction_profile`) elige entre
    las vías de reducción disponibles — p.ej. A1 monolítica (`reduce_raw.sh`) o
    en cascada por noche/exposición (`reduce_cascade.py`).

    Los marcadores que no se pueden resolver se DEVUELVEN, no se inventan: el
    notebook los enseña y se niega a lanzar hasta que estén declarados.
    """
    import string

    reg = _registry()
    stage = reg.by_id(stage_id) if reg else None
    if stage is None or not stage.launch:
        return "", run_id or "", ["etapa sin plantilla de lanzamiento"]

    active = _effective_run(run_id)
    chain = chain_of(active)
    target_run = (chain.get("stage_runs") or {}).get(stage_id, active)
    profile = chain.get("reduction_profile", "*")
    template = stage.launch.get(profile) or stage.launch.get("*")
    if template is None:
        return "", target_run, [
            f"perfil {profile!r} sin plantilla para {stage_id} "
            f"(disponibles: {', '.join(sorted(stage.launch))})"
        ]

    ctx = _launch_context(target_run)
    needed = [f for _, f, _, _ in string.Formatter().parse(template) if f]
    missing = [f for f in needed if ctx.get(f) in (None, "")]
    safe = {k: ("{" + k + "}" if v in (None, "") else v) for k, v in ctx.items()}
    return template.format(**{k: safe.get(k, "{" + k + "}") for k in needed}), target_run, missing


class evidence_guard:
    """Contexto para las celdas de *evidencia*, que leen claves concretas del QC.

    Esas celdas se escribieron mirando el QC de un objeto. En otro objeto la
    misma etapa puede emitir claves distintas (otra vía de reducción, otra
    configuración) y la celda revienta a mitad del notebook.

    Este contexto **no silencia**: nombra la etapa, el QC y la clave que falta,
    y deja el notebook seguir. Un `try/except` mudo daría por bueno un notebook
    que no dice nada; esto deja constancia de que la evidencia no aplica.

    Solo atrapa errores de *forma* del QC (KeyError, TypeError, IndexError,
    AttributeError). Cualquier otro error sigue propagándose.
    """

    SHAPE_ERRORS = (KeyError, TypeError, IndexError, AttributeError)

    def __init__(self, stage_id: str, qc_relpath: str | None = None):
        self.stage_id = stage_id
        self.qc_relpath = qc_relpath

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None or not issubclass(exc_type, self.SHAPE_ERRORS):
            return False
        where = f" ({self.qc_relpath})" if self.qc_relpath else ""
        run = _ACTIVE_RUN or "?"
        print(
            f"\n⚠️  EVIDENCIA NO APLICABLE a esta cadena — etapa {self.stage_id}{where}\n"
            f"    {exc_type.__name__}: {exc}\n"
            f"    El QC de {run} existe pero no tiene la forma que esta celda espera:\n"
            f"    la evidencia se escribió contra el QC de otro objeto o de otra variante\n"
            f"    de la etapa. No es un hueco de ejecución, es una diferencia de esquema.\n"
            f"    Ver H4 en docs/plan_multiobjeto_notebooks_2026-07-24.md.\n"
        )
        return True


# --------------------------------------------------------------------------
# Volcado de QC
# --------------------------------------------------------------------------
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


def show(d: dict | None, keys=None, title: str | None = None) -> None:
    """Imprime un subconjunto de un QC de forma legible."""
    if title:
        print(f"== {title} ==")
    if d is None:
        print("(sin QC: etapa no ejecutada para esta cadena)")
        return
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
