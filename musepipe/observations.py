"""La vista POR OBSERVACIÓN de un run: qué cubos, con qué geometría y qué peso.

La cadena canónica trabaja sobre **un** cubo, el combinado, y por eso hasta
ahora nadie necesitaba una respuesta formal a «¿cuáles son las exposiciones de
este objeto y dónde cae la primaria en cada una?». Modelar la PSF por
observación sí la necesita, y la respuesta **no se inventa aquí**: ya está
escrita en el `stream_combine_plan.json` con el que se construyó el combinado
—ventana, desplazamiento subpíxel, peso, `EXPTIME`, MJD y astrometría por
exposición—, así que este módulo lo lee, no lo recalcula.

Lo único que sí resuelve es el desfase entre lo que el plan dice y lo que hay
en disco. El plan de ROXs 12 b apunta a un árbol de reducción que ya se borró,
y los cubos de esas mismas 29 exposiciones viven hoy en otro sitio
(`perexp_cubes` / `perexp_dir` del config). Cuando eso pasa:

* la exposición se re-resuelve **por `exposure_id`**, que es el nombre del
  directorio de la exposición y no cambia entre reducciones;
* y su centroide se **vuelve a medir**, porque un cubo re-reducido no tiene por
  qué caer en el mismo píxel que el que se midió entonces. Usar la geometría
  vieja sobre datos nuevos es exactamente el error que este módulo existe para
  no cometer: la diferencia se mide y viaja en la procedencia.

Si alguna exposición no se puede resolver, esto **falla con la lista**. No hay
camino silencioso: una PSF por observación con exposiciones que faltan es una
mezcla que no representa al combinado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .config import load_run_config
from .reduction.a1_review import combine_plan, combine_plan_path
from .reduction.stream_combine import (
    StreamCombinePlan,
    exposure_from_measurement,
    measure_primary_center,
    plan_from_dict,
)

#: Un centroide re-medido que se aparte más que esto del que el plan escribió
#: no es ruido de medida: es otro cubo, u otra reducción. Se avisa, no se falla,
#: porque la geometría que se usa es SIEMPRE la re-medida.
REMEASURE_WARN_PX = 0.5


class ObservationInputError(RuntimeError):
    """El run no tiene una vista por observación utilizable."""


class ObservationPlanMissing(ObservationInputError):
    """El objeto no declara exposiciones: no hay `stream_combine_plan.json`.

    Es distinto de «el plan está y algo no cuadra», y por eso tiene su propio
    tipo: un run cuyo cubo no salga de una combinación por exposiciones —una
    reducción monolítica, un run histórico— sencillamente no puede ajustar por
    observación, y quien lo pida tiene que poder distinguir «aquí no aplica» de
    «aquí falta algo que debería estar».
    """


@dataclass(frozen=True)
class ObservationSet:
    """Las exposiciones de un run, con rutas vivas y geometría utilizable."""

    plan: StreamCombinePlan
    source: str
    substituted: tuple[str, ...] = ()
    remeasured: tuple[dict, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def exposures(self):
        return self.plan.exposures

    def __len__(self) -> int:
        return len(self.plan.exposures)

    def provenance(self) -> dict:
        """Lo que el QC de cualquier etapa que use esto tiene que registrar."""

        return {
            "source_plan": self.source,
            "n_exposures": len(self.plan.exposures),
            "crop_npix": int(self.plan.crop_npix),
            "pad": int(self.plan.pad),
            "weight_mode": str(self.plan.weight_mode),
            "combine_method": str(self.plan.method),
            "substituted_paths": list(self.substituted),
            "remeasured": [dict(row) for row in self.remeasured],
            "warnings": list(self.warnings),
            "exposures": [
                {
                    "exposure_id": exp.exposure_id,
                    "file": exp.file,
                    "weight": float(exp.weight),
                    "exptime": float(exp.exptime),
                    "mjd_obs": float(exp.mjd_obs),
                    "y_center": float(exp.y_center),
                    "x_center": float(exp.x_center),
                    "shift_yx": [float(exp.shift_y), float(exp.shift_x)],
                    "window": [int(v) for v in exp.window],
                    "in_bounds": bool(exp.in_bounds),
                }
                for exp in self.plan.exposures
            ],
        }

    def to_json(self) -> dict:
        return {
            "schema_version": 1,
            "provenance": self.provenance(),
            "plan": self.plan.as_dict(),
        }


def _declared_candidates(cfg, project_root) -> dict[str, Path]:
    """`exposure_id -> ruta` de lo que el CONFIG declara, nunca de un glob ciego.

    `perexp_cubes` es una lista explícita; `perexp_dir` es la raíz de un árbol
    `<raíz>/<exposure_id>/DATACUBE_FINAL.fits`. Se admiten los dos porque los
    dos existen en los runs de este repositorio.
    """

    candidates: dict[str, Path] = {}
    listed = cfg.get("perexp_cubes") or []
    root = cfg.get("perexp_dir")
    if root:
        listed = list(listed) + sorted(str(p) for p in Path(root).glob("*/DATACUBE_FINAL.fits"))
    for value in listed:
        path = Path(value)
        if not path.is_absolute():
            path = Path(project_root) / path
        if path.exists():
            candidates.setdefault(path.parent.name, path)
    return candidates


def _remeasure(exposure, plan: StreamCombinePlan, index: int):
    """Vuelve a medir la primaria y devuelve (exposición nueva, fila de la deriva)."""

    measurement = measure_primary_center(
        exposure.file,
        data_ext=plan.data_ext,
        drop_wave_min_A=plan.drop_wave_min_A,
        drop_wave_max_A=plan.drop_wave_max_A,
        centering_method=plan.centering_method,
    )
    fresh = exposure_from_measurement(
        measurement,
        index=index,
        weight=float(exposure.weight),
        crop_npix=int(plan.crop_npix),
        pad=int(plan.pad),
    )
    delta_y = float(fresh.y_center - exposure.y_center)
    delta_x = float(fresh.x_center - exposure.x_center)
    row = {
        "exposure_id": exposure.exposure_id,
        "dy_px": delta_y,
        "dx_px": delta_x,
        "centroid_fallback": bool(fresh.centroid_fallback),
    }
    return fresh, row


def resolve_observation_plan(
    run_id=None,
    *,
    project_root=None,
    allow_run_id_mismatch: bool = False,
    remeasure: str = "substituted",
    plan_json=None,
) -> ObservationSet:
    """Las exposiciones del run, listas para ajustar o para combinar.

    ``remeasure``:

    * ``"substituted"`` (por defecto) — sólo se vuelve a medir el centroide de
      las exposiciones cuya ruta hubo que re-resolver;
    * ``"all"`` — se re-mide todo (comprobación cara: ~2 s por cubo);
    * ``"none"`` — se usa la geometría del plan tal cual. Sólo tiene sentido
      cuando ninguna ruta cambió.

    ``plan_json`` lee una vista ya resuelta (la que escribe C1 en sus
    productos) en vez de rehacer el trabajo, validando que los ficheros siguen
    ahí. Es la caché, expresada como producto y no como estado escondido.
    """

    run = load_run_config(
        run_id, project_root=project_root, allow_run_id_mismatch=allow_run_id_mismatch
    )
    cfg = dict(run.config)
    root = run.paths.project_root

    if plan_json is not None:
        payload = json.loads(Path(plan_json).read_text(encoding="utf-8"))
        plan = plan_from_dict(payload["plan"])
        missing = [exp.exposure_id for exp in plan.exposures if not Path(exp.file).exists()]
        if missing:
            raise ObservationInputError(
                f"{plan_json}: {len(missing)} cubos ya no existen "
                f"({', '.join(missing[:3])}{'…' if len(missing) > 3 else ''}). "
                "Borra el fichero para que se vuelva a resolver."
            )
        prov = payload.get("provenance") or {}
        return ObservationSet(
            plan=plan,
            source=str(prov.get("source_plan", plan_json)),
            substituted=tuple(prov.get("substituted_paths", ())),
            remeasured=tuple(prov.get("remeasured", ())),
            warnings=tuple(prov.get("warnings", ())),
        )

    payload = combine_plan(run.run_id, project_root=root)
    if payload is None:
        raise ObservationPlanMissing(
            f"{run.run_id}: no hay `stream_combine_plan.json`. La PSF por observación "
            "necesita saber qué exposiciones entraron en el combinado y con qué "
            "geometría; sin el plan no se puede reconstruir. Genera uno con "
            "`python scripts/plan_stream_combine.py --run-id <RUN>`."
        )
    plan = plan_from_dict(payload)
    source = str(combine_plan_path(run.run_id, project_root=root) or "")

    candidates = _declared_candidates(cfg, root)
    resolved, substituted, missing = [], [], []
    for exposure in plan.exposures:
        if Path(exposure.file).exists():
            resolved.append((exposure, False))
            continue
        alternative = candidates.get(exposure.exposure_id)
        if alternative is None:
            missing.append(exposure.exposure_id)
            continue
        resolved.append((replace(exposure, file=str(alternative)), True))
        substituted.append(exposure.exposure_id)

    if missing:
        raise ObservationInputError(
            f"{run.run_id}: {len(missing)} de {len(plan.exposures)} exposiciones del plan no "
            f"existen y no están declaradas en `perexp_cubes`/`perexp_dir`: "
            f"{', '.join(missing)}. Declara dónde están o vuelve a generarlas con "
            "`python scripts/regen_perexp_cubes.py --step all`."
        )

    warnings: list[str] = list(plan.warnings)
    remeasured: list[dict] = []
    exposures = []
    for index, (exposure, was_substituted) in enumerate(resolved):
        needs = remeasure == "all" or (remeasure == "substituted" and was_substituted)
        if not needs:
            exposures.append(replace(exposure, index=index))
            continue
        fresh, row = _remeasure(exposure, plan, index)
        exposures.append(fresh)
        remeasured.append(row)
        if max(abs(row["dy_px"]), abs(row["dx_px"])) > REMEASURE_WARN_PX:
            warnings.append(
                f"{row['exposure_id']}: la primaria se movió "
                f"({row['dy_px']:+.2f}, {row['dx_px']:+.2f}) px respecto del plan; "
                "se usa la medida nueva"
            )

    return ObservationSet(
        plan=replace(plan, exposures=tuple(exposures)),
        source=source,
        substituted=tuple(substituted),
        remeasured=tuple(remeasured),
        warnings=tuple(warnings),
    )


__all__ = [
    "ObservationInputError",
    "ObservationPlanMissing",
    "ObservationSet",
    "REMEASURE_WARN_PX",
    "resolve_observation_plan",
]
