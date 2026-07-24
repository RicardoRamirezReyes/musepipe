"""Registro declarativo de las etapas de la cadena canónica A1..G5 (+S0/S1).

Fuente única de verdad para la parte *machine-readable* de cada etapa: qué QC
emite, con qué nombres alternativos según el productor, y por qué punto de
entrada se re-ejecuta.

Consumidores:
  * ``scripts/build_review_notebooks.py`` — valida sus entradas contra este
    registro al arrancar (una divergencia aborta la generación).
  * ``notebooks/_nbcommon.py`` — resuelve ``load_qc(relpath)`` -> etapa -> run,
    combinándolo con el manifiesto ``chain`` del run.
  * ``tests/`` — contrato de resolución.

**Solo stdlib**: los notebooks lo importan sin la pila científica.

Motivación (problema P2 del plan `docs/plan_multiobjeto_notebooks_2026-07-24.md`):
los nombres de QC pertenecían al *productor*, no a la *etapa*. A1 emite
``stages/stage00r_qc.json`` por la vía monolítica (``reduce_raw.sh``) y
``cube_telcorr_qc.json`` por la vía cascade (``reduce_cascade.py``), con
esquemas de claves distintos. ``qc_aliases`` modela eso explícitamente.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Stage:
    """Una etapa de la cadena."""

    id: str
    slug: str
    block: str
    qc: str | None = None
    #: rutas alternativas del mismo QC lógico, por perfil de reducción
    qc_aliases: tuple[str, ...] = ()
    #: True si la etapa puede no haberse ejecutado sin que eso sea un fallo
    qc_optional: bool = False
    #: tipo de punto de entrada: script | module_main | module_run | pyscript | cli | audit
    exec_kind: str = "audit"
    #: variante de esquema del QC, por perfil de reducción (solo cuando difiere)
    qc_schema_variant: dict[str, str] = field(default_factory=dict)
    #: plantillas de comando por perfil de reducción, para `exec_kind="launch"`.
    #: Los `{marcadores}` se resuelven contra el config del run que ejecuta la
    #: etapa (ver `notebooks/_nbcommon.launch_command`). Clave "*" = cualquiera.
    launch: dict[str, str] = field(default_factory=dict)

    @property
    def qc_paths(self) -> tuple[str, ...]:
        """Todas las rutas bajo las que puede aparecer el QC de esta etapa."""
        return tuple(p for p in (self.qc, *self.qc_aliases) if p)

    @property
    def parameterized(self) -> bool:
        """True si la etapa acepta ``--run-id`` (se puede correr para cualquier objeto).

        Las ``audit`` no se ejecutan: son las que bloquean la exigencia
        "cadena completa desde notebooks" (WP Track E).
        """
        return self.exec_kind in {"script", "module_main", "module_run",
                                  "pyscript", "launch"}


STAGES: tuple[Stage, ...] = (
    # ===================== BLOQUE A — reducción =====================
    Stage("A1", "A1_raw_reduction", "A", "stages/stage00r_qc.json",
          qc_aliases=("cube_telcorr_qc.json",),
          qc_schema_variant={"monolithic": "stage00r_v1", "cascade": "stream_combine_v1"},
          exec_kind="launch",
          launch={
              "monolithic": ("bash scripts/reduce_raw.sh phase0 --run-id {run_id} "
                             "--raw-data-dir {raw_data_dir}"),
              "cascade": ("python scripts/reduce_cascade.py --run-id {run_id} "
                          "--raw-data-dir {raw_data_dir} --work-dir {work_dir} --execute"),
          }),
    # A2/A3 no tienen un QC "canónico" declarado en el builder (sus notebooks
    # auditan el QC de A4 y el del run de reducción), pero sus ficheros existen y
    # deben ser atribuibles a su etapa para que los mensajes y `show_chain` no
    # digan "etapa desconocida". Van como alias, no como `qc`.
    Stage("A2", "A2_sky_zap", "A", None,
          qc_aliases=("stages/stage00s_qc.json",), exec_kind="launch",
          launch={"*": ("bash scripts/sky_zap.sh decision --run-id {run_id} "
                        "--input-cube {cube} --provenance {provenance} "
                        "--output-dir {stage_dir} "
                        "--qc-output {stage_dir}/stage00s_qc.json")}),
    Stage("A3", "A3_telluric", "A", None,
          qc_aliases=("stages/stage00t_qc.json", "stages/stage00t_realigned_qc.json"),
          exec_kind="launch",
          launch={"*": ("bash scripts/telluric.sh decision --run-id {run_id} "
                        "--input-cube {cube} --upstream A2 "
                        "--primary-y {primary_y} --primary-x {primary_x} "
                        "--qc-output {stage_dir}/stage00t_qc.json")}),
    Stage("A4", "A4_cube_qc", "A", "stages/stage00q_qc.json", exec_kind="launch",
          launch={"*": ("python -m musepipe.qc.cube_qc m3-flux --run-id {run_id} "
                        "--cube {cube} --aperture-correction growth_curve "
                        "--truncation-correction "
                        "--qc-output {stage_dir}/stage00q_qc.json")}),
    # ===================== BLOQUE B — alineado =====================
    Stage("B1", "B1_load_align_crop", "B", "stages/stage01_qc.json", exec_kind="module_run"),
    Stage("B2", "B2_xcorr_stripes", "B", "stages/stage02_xcorr_qc.json", exec_kind="script"),
    Stage("B3", "B3_localize", "B", "stages/stage01c_qc.json", exec_kind="script"),
    # ===================== BLOQUE C — extracción =====================
    Stage("C1", "C1_chromatic_psf", "C", "stages/stage_e01_qc.json", exec_kind="script"),
    Stage("04b", "C_04b_local_surface", "C", "stages/stage04b_qc.json", exec_kind="module_main"),
    Stage("C2", "C2_aperture", "C", "stages/spec_aperture_qc.json", exec_kind="script"),
    Stage("C3", "C3_optimal", "C", "stages/spec_optimal_qc.json", exec_kind="script"),
    Stage("C4", "C4_psffit", "C", "stages/spec_psffit_qc.json", exec_kind="script"),
    Stage("C5", "C5_sgf", "C", "stages/spec_sgf_qc.json", qc_optional=True, exec_kind="module_main"),
    Stage("C6", "C6_lpm", "C", "stages/spec_lpm_qc.json", qc_optional=True, exec_kind="module_main"),
    # ===================== BLOQUE D — comparación/calibración =====================
    Stage("D1", "D1_method_compare", "D", "stages/stage_x10_qc.json", exec_kind="script"),
    Stage("D2", "D2_calibrate", "D", "stages/stage_x11_qc.json", exec_kind="script"),
    # ===================== BLOQUE E — detección/límites =====================
    Stage("E1", "E1_halpha_detect", "E", "stages/stage_h01_qc.json", exec_kind="script"),
    Stage("E1b", "E1b_fov_detection", "E", "stages/stage_h01b_qc.json",
          qc_optional=True, exec_kind="module_main"),
    Stage("E2", "E2_artifacts", "E", "stages/stage_h02_qc.json", exec_kind="script"),
    Stage("E3", "E3_upper_limits", "E", "stages/stage_h03_qc.json", exec_kind="script"),
    Stage("E4", "E4_injection", "E", "stages/stage_h04_qc.json", exec_kind="script"),
    Stage("E5", "E5_contrast_curves", "E", "stages/stage_h05_qc.json",
          qc_optional=True, exec_kind="module_main"),
    Stage("E6", "E6_roc_curves", "E", "stages/stage_h06_qc.json",
          qc_optional=True, exec_kind="module_main"),
    # ===================== BLOQUE F — informe =====================
    Stage("F1", "F1_final_report", "F", "report/run_summary.json", exec_kind="pyscript"),
    # ===================== BLOQUE G — caracterización =====================
    Stage("G0", "G0_real_cube", "G", "stages/stage_g0_qc.json", exec_kind="pyscript"),
    Stage("G1", "G1_extraction_validation", "G", "stages/stage_g1_qc.json", exec_kind="pyscript"),
    Stage("G2", "G2_measure_lines", "G", "stages/stage_g2_qc.json", exec_kind="module_run"),
    Stage("G3", "G3_accretion", "G", "stages/stage_g3_qc.json", exec_kind="module_run"),
    Stage("G4", "G4_classify", "G", "stages/stage_g4_classification.json", exec_kind="module_run"),
    Stage("G5", "G5_final_synthesis", "G",
          "report/characterization/characterization_summary.json", exec_kind="pyscript"),
    # ===================== BLOQUE S — wavesol/mapas =====================
    Stage("S0", "S0_wavesol_map", "S", "stages/stageS0_qc.json", exec_kind="module_main"),
    Stage("S1", "S1_halpha_map", "S", "stages/stageS1_qc.json", exec_kind="module_main"),
)


_BY_ID = {s.id: s for s in STAGES}
_BY_SLUG = {s.slug: s for s in STAGES}
# ruta de QC -> etapa (incluye alias). Una ruta pertenece a una sola etapa.
_BY_QC: dict[str, Stage] = {}
for _s in STAGES:
    for _p in _s.qc_paths:
        _BY_QC.setdefault(_p, _s)


def by_id(stage_id: str) -> Stage | None:
    return _BY_ID.get(stage_id)


def by_slug(slug: str) -> Stage | None:
    return _BY_SLUG.get(slug)


def stage_for_qc(relpath: str) -> Stage | None:
    """Etapa dueña de una ruta de QC (o None si la ruta no está registrada).

    Sirve para que un ``load_qc('stages/stage00r_qc.json')`` sepa que pertenece
    a A1 y consulte ``chain.stage_runs['A1']``.
    """
    return _BY_QC.get(relpath)


def stage_ids() -> tuple[str, ...]:
    return tuple(s.id for s in STAGES)


__all__ = ["Stage", "STAGES", "by_id", "by_slug", "stage_for_qc", "stage_ids"]
