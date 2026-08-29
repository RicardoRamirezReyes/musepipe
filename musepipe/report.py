"""Deterministic final-report builder for MUSE reduction runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np

from .config import load_run_config
from .stage_registry import STAGES
from .extraction.aperture import sha256_file
from .extraction.product import SpectrumProduct
from .io import read_json
from .paths import RunPaths


REPORT_TEMPLATE = """# MUSE Run Report

## Executive Summary
{executive_summary}

## QC Checklist
{qc_table}

## Standard Figures
{figure_list}

## Master Tables
{table_list}

## Line Diagnostics
{line_diagnostics}

## Accretion Relation
{accretion_relation}

## Variability Caveat
{variability_caveat}

## Accepted Limitations
{accepted_limitations}

## Historical Context
{historical_context}

## Reproduction
{reproduction}
"""


# S7c note (fixed prose): with both Balmer lines as upper limits, the decrement
# cannot pin the extinction, so the accretion limit is reported over an A_V grid.
LINE_DIAGNOSTICS_NOTE = (
    "Both Halpha and Hbeta are upper limits (non-detections), so the Balmer decrement "
    "Halpha/Hbeta does NOT constrain the line-of-sight extinction A_Halpha toward the "
    "companion. The accretion limit is therefore reported over a grid of assumed A_V "
    "(`tables/mdot_limit_vs_extinction.csv`, S7b; `extinction_ladder` in stage_h03_qc.json) "
    "rather than a decrement-derived value. This follows the accreting-companion framework of "
    "Aoyama & Ikoma (2019) and Hashimoto et al. (2020), in which the Halpha-derived accretion "
    "luminosity scales with the assumed extinction: a larger A_Halpha relaxes the Mdot limit "
    "(about x8 from A_V=0 to A_Halpha=2 for this dataset). The adopted A_V (Rizzuto et al. 2015) "
    "is used for the headline limit; the ladder brackets the systematic."
)


# R1 note (fixed prose): the Halpha accretion limit is reported under TWO
# L_acc-L_line calibrations, because in the planetary/BD regime probed here the
# stellar (Alcala+2017) and planetary-shock (Aoyama+2021) relations diverge.
def accretion_relation_note(target_display: str) -> str:
    """Nota R1 parametrizada por objeto.

    Antes era una constante con "ROXs 12 B" incrustado, así que el informe de
    CUALQUIER objeto afirmaba hablar del primero (problema P4c).
    """
    return (
    "The Halpha-derived accretion-luminosity limit is reported under TWO calibrations of the "
    "L_acc-L_Halpha relation. (1) The HEADLINE limit uses the empirical stellar/brown-dwarf "
    "relation of Alcala et al. (2017), consistent with the rest of this pipeline. (2) A parallel "
    "limit uses the theoretical planetary accretion-shock relation of Aoyama et al. (2021, "
    "ApJL 917, L30; extended by Marleau & Aoyama 2023, RNAAS 7, 28), which is more appropriate "
    f"for a planetary/brown-dwarf accretor such as {target_display}. In our regime (L_Halpha well below "
    "1e-6 Lsun) the planetary relation has a shallower slope and gives a LARGER L_acc for the same "
    "L_Halpha, hence a WEAKER (higher) Mdot upper limit. Reporting both is the standard post-"
    "Hashimoto (2020) practice; the values appear as `mdot_msun_yr` (Alcala) and "
    "`mdot_aoyama21_msun_yr` (Aoyama+21) in stage_h03_qc.json and "
    "`tables/mdot_limit_vs_extinction.csv`. The planetary relation is valid for L_acc <= 1e-4 Lsun; "
    "the Alcala relation remains the adopted headline pending a companion-class decision."
    )


# R2 note (fixed prose): accretion is time-variable, and this dataset is a single
# epoch, so the non-detection / Mdot limit applies only to that epoch.
VARIABILITY_CAVEAT_NOTE = (
    "This result is a SINGLE-EPOCH measurement. All science frames were taken on one night "
    "(2022-09-01; MJD 59823.025-59823.086), a set of 7 exposures spanning about 87 minutes. "
    "Accretion onto young sub-stellar and planetary-mass objects is known to be variable on "
    "timescales from hours to years (e.g. Cody & Hillenbrand 2014; multi-epoch monitoring "
    "advocated by Hashimoto et al. 2020, sec 5.4). A non-detection at this epoch therefore does "
    "NOT exclude accretion at other times: the companion could be accreting episodically, or "
    "accreting steadily at a level that this ~87-minute window happened to sample below the "
    "detection threshold. The reported Halpha flux and Mdot upper limits should be read as "
    "constraints on the accretion state DURING THIS EPOCH, not as a time-averaged or permanent "
    "upper bound. Confirming or tightening the limit requires multi-epoch observations."
)


# Gate policy (frozen, auditable): specific red checks that are DOWNGRADED to a
# yellow "accepted limitation" instead of blocking the package. Each entry is a
# stage id -> {exact _walk_statuses path -> justification}. Matching is per-path
# so a NEW or different failure in the same stage still surfaces as red. The
# raw red token is kept in the report, annotated with the reason; nothing is
# hidden. Changing this set is a documented policy revision (hash below).
ACCEPTED_LIMITATIONS = {
    "A4_cube_qc": {
        "m5_stat.status": (
            "M5 STAT variance underestimated by MUSE cube resampling covariance (inherent to "
            "drizzle-style resampling, not a reduction error). MEASURED empirically (R4/STAT_EMP, "
            "inter-exposure scatter of the 7 regenerated per-exposure cubes; "
            "stageR4_stat_emp_qc.json): the DRS STAT underestimates the per-voxel variance by "
            "x2.9 at the typical background voxel (x3.3 bias-corrected), up to x8.6 flux-weighted "
            "(an upper limit, inflated by source seeing/transparency variation) -- consistent with "
            "the ~4-6x expectation. Mitigated by using empirical control-based noise throughout "
            "X01-X11 (plan B). Not a fixable defect."
        ),
    },
    "E2_artifacts": {
        "t2.status": (
            "T2 (PSF-shape test) is repurposed for a NON-DETECTION (H02 spec sec.2): the global "
            "Halpha maximum being NOT PSF-shaped SUPPORTS the non-detection. _overall_status does not "
            "branch on the input detection verdict, so it mislabels this expected outcome as 'fail'; "
            "reinterpreted per spec (overall_raw retained in the QC)."
        ),
    },
    "E4_injection": {
        "checks.v4_hierarchy.status": (
            "Injection-throughput hierarchy/monotonicity break is the documented edge-position "
            "pathology (aperture/optimal_ls are insensitive in the steep star-halo gradient at the "
            "companion position); the CANONICAL psffit throughput is well-behaved (~0.67) and "
            "unaffected, and determinism is test-guaranteed."
        ),
        # Retirada 2026-08-20 con E4 v3. Esta aceptacion existia porque el gate
        # contaba FILAS: el diagnostico de 2026-07-15 ya decia que los aciertos
        # eran "UN unico caso duplicado entre modos de continuo" en control3, o
        # sea el defecto de multiplicidad que v3 arregla contando POSICIONES
        # (docs/spec_E4_v3_codex_injection_recovery.md §0). Con la causa
        # corregida, dejar la llave seria auto-aceptar `v2_nulls_clean` sea cual
        # sea la razon: un fallo legitimo de dos posiciones de tres entraria como
        # nota aceptada sin que nadie lo viera. Ademas su texto ya no describia
        # este run —hablaba de 2/96 filas y de `psffit`, que hoy no es extremo.
    },
    "D2_calibrate": {
        # Accepted ONLY after full diagnosis (docs/2026-07-10_d2_red_continuum_diagnosis.md):
        # the flag is a real, characterized, non-removable inter-method systematic,
        # not a defect and not a mislabel of real signal (v3 now gates on the
        # signal-free inter-method metric, not |runmed-poly|).
        "checks.v3_continuum_stable.ok": (
            "Red-band (8600-9100 A) inter-method continuum LEVEL systematic (psffit vs "
            "optimal_psfsub ~1.76x): residual chromatic halo subtraction of the bright primary at "
            "the faint companion, at the PSF floor (Psfao, ~4.4% red ring residual). The red "
            "spectral SHAPE is real and method-consistent (corr 0.957); only the LEVEL disagrees. "
            "Does NOT affect the Halpha emission-line non-detection or the Mdot limit. Fully "
            "diagnosed in docs/2026-07-10_d2_red_continuum_diagnosis.md."
        ),
    },
}
ACCEPTED_LIMITATIONS_HASH = hashlib.sha256(
    json.dumps(ACCEPTED_LIMITATIONS, sort_keys=True).encode("utf-8")
).hexdigest()[:12]


STAGE_DEFINITIONS = [
    {"id": "A1_raw_reduction", "qc": "stage00r_qc.json", "required": True},
    {"id": "A2_sky_zap", "qc": "stage00s_qc.json", "required": "conditional"},
    {"id": "A3_telluric", "qc": "stage00t_qc.json", "required": "optional"},
    {"id": "A4_cube_qc", "qc": "stage00q_qc.json", "required": True},
    {"id": "B1_align", "qc": "stage01_qc.json", "required": True},
    {"id": "B2_xcorr_stripes", "qc": "stage02_xcorr_qc.json", "required": True},
    {"id": "B3_localize", "qc": "stage01c_qc.json", "required": True},
    {"id": "C1_psf", "qc": "stage_e01_qc.json", "required": True},
    {"id": "C2_aperture", "qc": "spec_aperture_qc.json", "required": True},
    {"id": "C3_optimal", "qc": "spec_optimal_qc.json", "required": True},
    {"id": "C4_psffit", "qc": "spec_psffit_qc.json", "required": True},
    {"id": "D1_compare", "qc": "stage_x10_qc.json", "required": True},
    {"id": "D2_calibrate", "qc": "stage_x11_qc.json", "required": True},
    {"id": "E1_halpha", "qc": "stage_h01_qc.json", "required": True},
    {"id": "E2_artifacts", "qc": "stage_h02_qc.json", "required": True},
    {"id": "E4_injection", "qc": "stage_h04_qc.json", "required": True},
    {"id": "E3_limits", "qc": "stage_h03_qc.json", "required": "conditional"},
]


STANDARD_FIGURES = [
    {"id": "fig01_field", "filename": "fig01_field.png", "title": "Field, sources, apertures, and controls"},
    {"id": "fig02_psf_lambda", "filename": "fig02_psf_lambda.png", "title": "Chromatic PSF model"},
    {"id": "fig03_methods", "filename": "fig03_methods.png", "title": "Spectra by extraction method"},
    {"id": "fig04_final_spectrum", "filename": "fig04_final_spectrum.png", "title": "Final calibrated spectrum and errors"},
    {"id": "fig05_halpha_zoom", "filename": "fig05_halpha_zoom.png", "title": "Halpha zoom by method"},
    {"id": "fig06_throughput", "filename": "fig06_throughput.png", "title": "Injection throughput and completeness"},
    {"id": "fig07_halpha_result", "filename": "fig07_halpha_result.png", "title": "Halpha result or upper-limit context"},
]


SPECTRUM_PRODUCT_CANDIDATES = {
    "aperture": "spec_calibrated_aperture_object.fits",
    "optimal_ls": "spec_calibrated_optimal_ls_object.fits",
    "optimal_psfsub": "spec_calibrated_optimal_psfsub_object.fits",
    "psffit": "spec_calibrated_psffit_object.fits",
    "sgf": "spec_calibrated_sgf_object.fits",
    "lpm": "spec_calibrated_lpm_object.fits",
    "final": "spec_final_object.fits",
}


@dataclass(frozen=True)
class ReportPaths:
    run_paths: RunPaths
    report_dir: Path
    figures_dir: Path
    extra_dir: Path
    run_summary_json: Path
    report_md: Path
    table_lines_csv: Path
    table_methods_csv: Path
    table_qc_csv: Path


def report_paths(run_id, project_root=None):
    run_paths = RunPaths.from_project_root(run_id, project_root)
    report_dir = run_paths.run_dir / "report"
    return ReportPaths(
        run_paths=run_paths,
        report_dir=report_dir,
        figures_dir=report_dir / "figures",
        extra_dir=report_dir / "extra",
        run_summary_json=report_dir / "run_summary.json",
        report_md=report_dir / "report.md",
        table_lines_csv=report_dir / "table_lines.csv",
        table_methods_csv=report_dir / "table_methods.csv",
        table_qc_csv=report_dir / "table_qc.csv",
    )


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_deterministic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv_deterministic(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value):
    if value in (None, ""):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _first_existing(paths):
    for path in paths:
        if path is None:
            continue
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


def resolve_product_path(path_value, run_paths):
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = [
        run_paths.run_dir / path,
        run_paths.project_root / path,
        run_paths.stage_dir / path,
        run_paths.plot_dir / path,
        run_paths.table_dir / path,
    ]
    return _first_existing(candidates) or path


def _registry_stage_by_qc(filename):
    """La etapa del registry que declara ese QC, por cualquiera de sus nombres.

    `STAGE_DEFINITIONS` mantiene su propia lista de ficheros, y ese es justo el
    problema: cuando el registry gana un alias o la cadena reparte una etapa a
    otro run, F1 no se entera. El registry es la fuente de verdad declarada en
    `CLAUDE.md`; aquí solo se usa para AMPLIAR los sitios donde buscar, nunca
    para cambiar qué etapas son obligatorias.
    """

    for stage in STAGES:
        if any(Path(qc).name == filename for qc in stage.qc_paths):
            return stage
    return None


def _qc_candidates(item, run_paths, stage_runs):
    """Rutas donde puede estar el QC de una etapa, en orden de preferencia.

    Dos cosas que la ruta fija `stage_dir / item["qc"]` no ve, y que en
    ROXs 42B b daban dos `required QC missing` FALSOS y bloqueantes:

    * **Los alias del registry.** B2 escribe `stage02_qc.json` o
      `stage02_xcorr_qc.json` segun cuando se corriera; el registry declara los
      dos y F1 solo miraba el nuevo.
    * **`chain.stage_runs`.** A1 puede vivir en otro run (en ROXs 42B b, en
      `ROXs42Bb_raw`), que es el reparto que el propio config declara.
    """

    stage = _registry_stage_by_qc(item["qc"])
    names = [item["qc"]]
    if stage is not None:
        names += [qc for qc in stage.qc_paths if Path(qc).name != item["qc"]]

    # El run que declara `chain.stage_runs` MANDA sobre el fichero que por
    # casualidad esté en este run: `ROXs12b_realigned/stages/stage00r_qc.json`
    # existe pero describe OTRA reducción (7 exposiciones, V2/V5/V6
    # `unavailable`), mientras la que alimenta la cadena vive en
    # `ROXs12b_multinight_raw_20260728`. Leer la de al lado es exactamente la
    # trampa que el reparto de la cadena existe para evitar.
    roots = []
    other = (stage_runs or {}).get(stage.id) if stage is not None else None
    if other and other != run_paths.run_id:
        roots.append(RunPaths.from_project_root(other, run_paths.project_root))
    roots.append(run_paths)

    # El nombre CANONICO se busca en todos los runs antes de bajar a los alias:
    # en ROXs 42B b el alias `cube_telcorr_qc.json` existe en el propio run y el
    # canonico `stage00r_qc.json` en el run que declara `chain.stage_runs`, y el
    # bueno es el segundo — el envoltorio, con sus fases y su bateria V1-V6; el
    # alias es el QC del combine, que documenta un solo paso.
    out = []
    for name in names:
        rel = Path(name)
        for paths in roots:
            out.append(paths.run_dir / rel if rel.parts[:1] == ("stages",) else paths.stage_dir / rel.name)
            out.append(paths.run_dir / rel.name)
    return out


def read_qc_payloads(run_paths, stage_runs=None):
    if stage_runs is None:
        try:
            payload = load_run_config(run_paths.run_id, project_root=run_paths.project_root).payload
            stage_runs = ((payload or {}).get("chain") or {}).get("stage_runs") or {}
        except Exception:  # noqa: BLE001 - sin config, se busca solo en este run
            stage_runs = {}
    payloads = {}
    for item in STAGE_DEFINITIONS:
        candidates = _qc_candidates(item, run_paths, stage_runs)
        path = _first_existing(candidates) or candidates[0]
        payloads[item["id"]] = {"path": path, "exists": path.exists(), "qc": read_json(path) if path.exists() else None}
    return payloads


def _status_token(value):
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if value is None:
        return None
    return str(value).strip().lower()


def _walk_statuses(payload, prefix=""):
    statuses = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in {"status", "verdict", "ok"}:
                token = _status_token(value)
                if token is not None:
                    statuses.append((path, token))
            statuses.extend(_walk_statuses(value, path))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            statuses.extend(_walk_statuses(item, f"{prefix}[{index}]"))
    return statuses


#: Como se traduce la `priority` que un QC DECLARA en su `open_issue` a la del
#: reporte. `accepted`/`info` no bloquean: son diagnosticos, y escalarlos era lo
#: que publicaba la nota de M4 —cuyo texto dice literalmente «retain as a
#: systematic diagnostic»— como bloqueante de la compuerta final.
_DECLARED_ISSUE_PRIORITY = {
    "accepted": "minor",
    "info": "minor",
    "minor": "minor",
    "major": "major",
    "blocking": "blocking",
}


def _issue_text(issue):
    """El texto de un `open_issue`, sea cadena suelta o dict con `priority`."""

    if isinstance(issue, dict):
        return str(issue.get("issue", issue))
    return str(issue)


def _issue_texts(open_issues):
    return [_issue_text(issue) for issue in open_issues]


def _declared_priority(issue):
    if not isinstance(issue, dict):
        return None
    declared = issue.get("priority")
    if declared is None:
        return None
    return _DECLARED_ISSUE_PRIORITY.get(str(declared).strip().lower(), "major")


#: Rutas que NO son un hecho propio sino el rollup de las hojas del mismo QC.
#: Solo se tratan como derivadas cuando el documento declara quien decide su
#: estado (`status_detail.driven_by`), que es lo que permite re-evaluarlas
#: contra las hojas en vez de creerlas a ciegas.
_ROLLUP_PATHS = ("status", "status_detail.status")


def _rollup_paths(qc):
    detail = qc.get("status_detail") if isinstance(qc, dict) else None
    if not isinstance(detail, dict) or "driven_by" not in detail:
        return ()
    return _ROLLUP_PATHS


def stage_status(stage_id, qc, *, required=True, accepted=None):
    """Return (status, issues, accepted_applied) for one stage.

    ``accepted`` maps exact _walk_statuses paths (e.g. "m5_stat.status") to a
    justification; a matching red token is DOWNGRADED to a yellow accepted
    limitation (kept in the issue list, annotated) rather than making the stage
    red. Non-matching red tokens still make the stage red.

    Un rollup NO es un hecho nuevo. A4 publica `status` (y `status_detail.status`)
    como agregado de M1-M5, asi que el rojo de M5 —que la politica YA acepta—
    volvia a entrar por esas dos rutas, que no estan en la lista, y dejaba la
    etapa en rojo contando tres veces el mismo hecho. Aqui se re-evaluan contra
    las hojas: si toda hoja roja esta aceptada, el rollup se anota como derivado
    en vez de bloquear; si queda UNA hoja roja sin aceptar, el rollup sigue rojo.
    Meter "status" en ACCEPTED_LIMITATIONS habria sido lo contrario: auto-aceptar
    cualquier rojo futuro de la etapa.
    """

    if qc is None:
        if required is True:
            return "red", ["required QC missing"], []
        if required == "conditional":
            return "yellow", ["conditional stage QC missing"], []
        return "not_run", [], []
    accepted = accepted or {}
    statuses = _walk_statuses(qc)
    red_tokens = {"fail", "failed", "red", "blocked", "error", "uninterpretable"}
    yellow_tokens = {"mixed", "unknown", "unavailable", "not_checked", "not_run", "skipped"}
    red_items = [(path, token) for path, token in statuses if token in red_tokens]
    rollups = _rollup_paths(qc)
    leaf_red = [(p, t) for p, t in red_items if p not in rollups]
    rollup_red = [(p, t) for p, t in red_items if p in rollups]
    genuine_red = [(p, t) for p, t in leaf_red if p not in accepted]
    accepted_red = [(p, t) for p, t in leaf_red if p in accepted]
    yellow = [f"{path}={token}" for path, token in statuses if token in yellow_tokens]
    open_issues = list(qc.get("open_issues", [])) if isinstance(qc, dict) and isinstance(qc.get("open_issues", []), list) else []
    accepted_applied = [{"path": p, "token": t, "reason": accepted[p]} for p, t in accepted_red]
    accepted_notes = [f"{p}={t} [accepted limitation: {accepted[p]}]" for p, t in accepted_red]
    if genuine_red:
        # Con una hoja roja sin aceptar, el rollup describe un fallo real: se
        # cuenta como rojo igual que antes.
        genuine = [f"{p}={t}" for p, t in genuine_red + rollup_red]
        return "red", genuine + accepted_notes + yellow + _issue_texts(open_issues), accepted_applied
    if rollup_red and accepted_red:
        driver = (qc.get("status_detail") or {}).get("driven_by")
        reason = (
            f"Rollup of this stage's own metrics, driven by {driver}, whose red is already an "
            "accepted limitation; counting it again would report one fact twice."
        )
        for path, token in rollup_red:
            accepted_applied.append({"path": path, "token": token, "reason": reason})
            accepted_notes.append(f"{path}={token} [accepted limitation: {reason}]")
    elif rollup_red:
        # Rollup rojo sin ninguna hoja roja: el documento se contradice a si
        # mismo y eso no se silencia.
        genuine = [f"{p}={t} (rollup with no red metric underneath)" for p, t in rollup_red]
        return "red", genuine + yellow + _issue_texts(open_issues), accepted_applied
    if accepted_red or yellow or open_issues:
        return "yellow", accepted_notes + yellow + _issue_texts(open_issues), accepted_applied
    return "green", [], []


def aggregate_open_issues(stage_rows, qc_payloads):
    out = []
    stage_status_by_id = {row["stage"]: row["status"] for row in stage_rows}
    for stage_id, payload in qc_payloads.items():
        qc = payload["qc"]
        if not isinstance(qc, dict):
            continue
        for issue in qc.get("open_issues", []) or []:
            # Una prioridad DECLARADA manda sobre la heuristica: quien escribe el
            # QC sabe si su nota es un diagnostico o un fallo. La heuristica sigue
            # para los `open_issues` que son cadenas sueltas, que son los de todas
            # las demas etapas.
            priority = _declared_priority(issue)
            if priority is None:
                priority = "minor"
                if stage_status_by_id.get(stage_id) == "red":
                    priority = "blocking"
                elif stage_id.startswith(("E", "D", "C")):
                    priority = "major"
            out.append({"stage": stage_id, "priority": priority, "issue": _issue_text(issue)})
    # El `summary` de una etapa roja CONTIENE ya el texto de sus `open_issues`
    # concatenado (`_stage_status` los une), asi que anadirlo cuando la etapa ya
    # ha contribuido sus issues los cuenta dos veces. Medido el 2026-08-29: los
    # dos objetos declaraban 8 `blocking` de los que 2 eran ese resumen, o sea 6
    # problemas distintos. El veredicto no cambia -sigue rojo con uno solo- pero
    # el numero es el que se cita.
    #
    # El resumen SI hace falta cuando la etapa esta roja y no declaro ningun
    # issue: entonces es el unico registro de por que lo esta.
    con_issues = {item["stage"] for item in out}
    for row in stage_rows:
        if row["status"] == "red" and row["stage"] not in con_issues:
            out.append({"stage": row["stage"], "priority": "blocking", "issue": row["summary"]})
    return sorted(out, key=lambda item: ({"blocking": 0, "major": 1, "minor": 2}.get(item["priority"], 3), item["stage"], item["issue"]))


def _declared_hash_records(payload, run_paths, prefix=""):
    records = []
    if isinstance(payload, dict):
        keys = set(payload)
        file_key = next((key for key in ("file", "path", "input", "input_file", "input_cube", "cube", "product") if key in keys), None)
        hash_key = next((key for key in ("sha256", "hash", "input_sha256", "output_sha256") if key in keys), None)
        if file_key and hash_key and payload.get(file_key) and payload.get(hash_key):
            path = resolve_product_path(payload[file_key], run_paths)
            records.append({"context": prefix or file_key, "path": path, "expected": str(payload[hash_key])})
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            records.extend(_declared_hash_records(value, run_paths, child))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            records.extend(_declared_hash_records(value, run_paths, f"{prefix}[{index}]"))
    return records


def check_declared_hashes(qc_payloads, run_paths):
    rows = []
    for stage_id, payload in qc_payloads.items():
        qc = payload["qc"]
        if not isinstance(qc, dict):
            continue
        for record in _declared_hash_records(qc, run_paths):
            path = Path(record["path"])
            actual = sha256_file(path) if path.exists() else None
            rows.append(
                {
                    "stage": stage_id,
                    "context": record["context"],
                    "path": str(path),
                    "expected": record["expected"],
                    "actual": actual,
                    "status": "pass" if actual == record["expected"] else "fail",
                }
            )
    return rows


def spectrum_product_paths(run_paths):
    return {name: run_paths.stage_dir / filename for name, filename in SPECTRUM_PRODUCT_CANDIDATES.items()}


def check_spectrum_conventions(run_paths):
    products = {}
    product_rows = []
    for name, path in spectrum_product_paths(run_paths).items():
        if not path.exists():
            continue
        try:
            product = SpectrumProduct.read(path)
            products[name] = product
            incube = Path(str(product.header.get("INCUBE", "")))
            actual_hash = None
            hash_status = "unavailable"
            if incube.exists() and product.header.get("INCUBESH"):
                actual_hash = sha256_file(incube)
                hash_status = "pass" if actual_hash == str(product.header.get("INCUBESH")) else "fail"
            product_rows.append(
                {
                    "method": name,
                    "path": str(path),
                    "formatv": int(product.header.get("FORMATV")),
                    "normrad": float(product.header.get("NORMRAD")),
                    "wframe": str(product.header.get("WFRAME")),
                    "incube": str(product.header.get("INCUBE", "")),
                    "incubesh": str(product.header.get("INCUBESH", "")),
                    "actual_incube_sha256": actual_hash,
                    "hash_status": hash_status,
                }
            )
        except Exception as exc:
            product_rows.append(
                {
                    "method": name,
                    "path": str(path),
                    "formatv": "",
                    "normrad": "",
                    "wframe": "",
                    "incube": "",
                    "incubesh": "",
                    "actual_incube_sha256": "",
                    "hash_status": f"fail:{exc}",
                }
            )
    conventions = {
        "formatv": sorted({row["formatv"] for row in product_rows if row["formatv"] != ""}),
        "normrad": sorted({row["normrad"] for row in product_rows if row["normrad"] != ""}),
        "wframe": sorted({row["wframe"] for row in product_rows if row["wframe"] != ""}),
    }
    status = "pass"
    issues = []
    for key, values in conventions.items():
        if len(values) > 1:
            status = "fail"
            issues.append(f"Spectrum products disagree in {key}: {values}")
    for row in product_rows:
        if str(row["hash_status"]).startswith("fail"):
            status = "fail"
            issues.append(f"{row['method']} INCUBESH hash mismatch")
    return {"status": status, "products": product_rows, "conventions": conventions, "issues": issues}


def _ensure_plot_backend():
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "musepipe_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save_placeholder(path, title, message):
    plt = _ensure_plot_backend()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=11)
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=8)
    ax.set_axis_off()
    fig.savefig(path, dpi=120, metadata={"Software": "musepipe-report"})
    plt.close(fig)
    return {"source": None, "status": "missing", "issue": message}


def _copy_or_placeholder(source, dest, title, issue):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source and Path(source).exists():
        shutil.copyfile(source, dest)
        return {"source": str(source), "status": "copied", "issue": ""}
    return _save_placeholder(dest, title, issue)


def _plot_spectra_by_method(dest, run_paths, *, halpha_zoom=False):
    plt = _ensure_plot_backend()
    products = []
    for method in ("aperture", "optimal_ls", "optimal_psfsub", "psffit", "sgf", "lpm"):
        path = run_paths.stage_dir / SPECTRUM_PRODUCT_CANDIDATES[method]
        if path.exists():
            products.append((method, SpectrumProduct.read(path)))
    if not products:
        return _save_placeholder(dest, "Spectra by method", "calibrated method spectra unavailable")
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for method, product in products:
        wave = np.asarray(product.wave_A, dtype=np.float64)
        flux = np.asarray(product.flux, dtype=np.float64)
        err = np.asarray((product.extra_columns or {}).get("flux_err_total", product.flux_err), dtype=np.float64)
        mask = np.isfinite(wave) & np.isfinite(flux)
        if halpha_zoom:
            mask &= (wave >= 6500.0) & (wave <= 6625.0)
        if not np.any(mask):
            continue
        ax.plot(wave[mask], flux[mask], lw=0.8, label=method)
        lo = flux[mask] - err[mask]
        hi = flux[mask] + err[mask]
        ax.fill_between(wave[mask], lo, hi, alpha=0.10)
    if halpha_zoom:
        ax.axvline(6562.8, color="0.35", lw=0.8, ls="--")
        ax.set_title("Halpha zoom by extraction method")
    else:
        ax.set_title("Calibrated spectra by extraction method")
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("Flux")
    ax.legend(fontsize=8)
    fig.savefig(dest, dpi=150, metadata={"Software": "musepipe-report"})
    plt.close(fig)
    return {"source": "calibrated_spectrum_products", "status": "generated", "issue": ""}


def _plot_final_spectrum(dest, run_paths):
    path = run_paths.stage_dir / SPECTRUM_PRODUCT_CANDIDATES["final"]
    if not path.exists():
        return _save_placeholder(dest, "Final calibrated spectrum", "spec_final_object.fits unavailable")
    plt = _ensure_plot_backend()
    product = SpectrumProduct.read(path)
    wave = np.asarray(product.wave_A, dtype=np.float64)
    flux = np.asarray(product.flux, dtype=np.float64)
    err = np.asarray((product.extra_columns or {}).get("flux_err_total", product.flux_err), dtype=np.float64)
    mask = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.plot(wave[mask], flux[mask], lw=0.8, color="black")
    ax.fill_between(wave[mask], flux[mask] - err[mask], flux[mask] + err[mask], color="tab:blue", alpha=0.15)
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("Flux")
    ax.set_title("Final calibrated spectrum with total error")
    fig.savefig(dest, dpi=150, metadata={"Software": "musepipe-report"})
    plt.close(fig)
    return {"source": str(path), "status": "generated", "issue": ""}


def _figure_sources(qc_payloads, run_paths):
    def qc_path(stage, *keys):
        qc = qc_payloads.get(stage, {}).get("qc") or {}
        current = qc
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return resolve_product_path(current, run_paths) if current else None

    return {
        "fig01_field": qc_path("B3_localize", "figures", "field") or qc_path("B3_localize", "figures", "summary"),
        "fig02_psf_lambda": qc_path("C1_psf", "figures", "summary"),
        "fig06_throughput": qc_path("E4_injection", "figures", "throughput"),
        "fig07_halpha_result": qc_path("E3_limits", "figures", "context") or qc_path("E2_artifacts", "figures", "t1"),
    }


def build_standard_figures(paths: ReportPaths, qc_payloads):
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    sources = _figure_sources(qc_payloads, paths.run_paths)
    result = {}
    for fig in STANDARD_FIGURES:
        dest = paths.figures_dir / fig["filename"]
        if fig["id"] == "fig03_methods":
            meta = _plot_spectra_by_method(dest, paths.run_paths)
        elif fig["id"] == "fig04_final_spectrum":
            meta = _plot_final_spectrum(dest, paths.run_paths)
        elif fig["id"] == "fig05_halpha_zoom":
            meta = _plot_spectra_by_method(dest, paths.run_paths, halpha_zoom=True)
        else:
            meta = _copy_or_placeholder(sources.get(fig["id"]), dest, fig["title"], f"{fig['title']} source unavailable")
        result[fig["id"]] = {"path": str(dest), "title": fig["title"], **meta, "sha256": sha256_file(dest)}
    return result


def build_table_lines(run_paths):
    rows = []
    h01_rows = read_csv_rows(run_paths.table_dir / "halpha_detection_by_method.csv")
    for row in h01_rows:
        rows.append(
            {
                "line": "Halpha",
                "result_type": "measurement",
                "method": row.get("method", ""),
                "flux_or_limit": row.get("matched_flux", ""),
                "fap": row.get("global_empirical_fap", ""),
                "throughput": "",
                "l_halpha": "",
                "mdot": "",
                "source": "halpha_detection_by_method.csv",
            }
        )
    h03_rows = read_csv_rows(run_paths.table_dir / "halpha_upper_limits.csv")
    for row in h03_rows:
        if row.get("row_kind") not in {"combined_final", "method"}:
            continue
        rows.append(
            {
                "line": "Halpha",
                "result_type": "upper_limit",
                "method": row.get("method", ""),
                "flux_or_limit": row.get("f_lim_dereddened", row.get("f_lim_observed", "")),
                "fap": "0.01",
                "throughput": row.get("throughput", ""),
                "l_halpha": row.get("l_halpha_erg_s", ""),
                "mdot": row.get("mdot_msun_yr", ""),
                "source": "halpha_upper_limits.csv",
            }
        )
    available = {row["line"] for row in rows}
    for line in ("Hbeta", "OI_8446"):
        if line not in available:
            rows.append(
                {
                    "line": line,
                    "result_type": "unavailable",
                    "method": "",
                    "flux_or_limit": "",
                    "fap": "",
                    "throughput": "",
                    "l_halpha": "",
                    "mdot": "",
                    "source": "not reported by upstream stages",
                }
            )
    return rows


def build_table_methods(run_paths):
    rows = []
    for row in read_csv_rows(run_paths.table_dir / "method_comparison.csv"):
        rows.append(
            {
                "source": "D1_compare",
                "method": row.get("method_i", ""),
                "pair": row.get("pair", ""),
                "band": row.get("band", ""),
                "z": row.get("z", ""),
                "snr": "",
                "verdict": "",
            }
        )
    for row in read_csv_rows(run_paths.table_dir / "halpha_detection_by_method.csv"):
        rows.append(
            {
                "source": "E1_halpha",
                "method": row.get("method", ""),
                "pair": "",
                "band": "Halpha",
                "z": row.get("matched_z", ""),
                "snr": row.get("matched_z", ""),
                "verdict": "",
            }
        )
    return rows


def make_qc_rows(qc_payloads):
    rows = []
    accepted_all = []
    for item in STAGE_DEFINITIONS:
        payload = qc_payloads[item["id"]]
        required = item["required"]
        status, issues, accepted_applied = stage_status(
            item["id"], payload["qc"], required=required, accepted=ACCEPTED_LIMITATIONS.get(item["id"])
        )
        for record in accepted_applied:
            accepted_all.append({"stage": item["id"], **record})
        rows.append(
            {
                "stage": item["id"],
                "required": str(required),
                "status": status,
                "qc_path": str(payload["path"]),
                "issue_count": len(issues),
                "accepted_limitations": len(accepted_applied),
                "summary": "; ".join(issues),
            }
        )
    return rows, accepted_all


def make_run_summary(run_id, project_root=None):
    paths = report_paths(run_id, project_root)
    qc_payloads = read_qc_payloads(paths.run_paths)
    qc_rows, accepted_limitations = make_qc_rows(qc_payloads)
    hash_rows = check_declared_hashes(qc_payloads, paths.run_paths)
    spectrum_check = check_spectrum_conventions(paths.run_paths)
    issues = aggregate_open_issues(qc_rows, qc_payloads)
    for row in hash_rows:
        if row["status"] != "pass":
            issues.append({"stage": row["stage"], "priority": "blocking", "issue": f"Hash mismatch at {row['context']}"})
    for issue in spectrum_check["issues"]:
        issues.append({"stage": "spectral_products", "priority": "blocking", "issue": issue})
    figures = build_standard_figures(paths, qc_payloads)
    for fig_id, meta in figures.items():
        if meta["status"] == "missing":
            issues.append({"stage": "figures", "priority": "major", "issue": f"{fig_id}: {meta['issue']}"})

    table_lines = build_table_lines(paths.run_paths)
    table_methods = build_table_methods(paths.run_paths)
    overall = "green"
    if any(row["status"] == "red" for row in qc_rows) or any(item["priority"] == "blocking" for item in issues):
        overall = "red"
    elif any(row["status"] == "yellow" for row in qc_rows) or issues:
        overall = "yellow"

    summary = {
        "schema_version": 1,
        "run_id": str(run_id),
        "overall_status": overall,
        "stages": qc_rows,
        "hash_chain": {
            "status": "pass" if all(row["status"] == "pass" for row in hash_rows) else ("unavailable" if not hash_rows else "fail"),
            "checks": hash_rows,
        },
        "spectrum_conventions": spectrum_check,
        "open_issues": issues,
        "accepted_limitations": accepted_limitations,
        "gate_policy": {
            "accepted_limitations_hash": ACCEPTED_LIMITATIONS_HASH,
            "note": (
                "Frozen gate policy: the listed specific red checks are downgraded to yellow accepted "
                "limitations (inherent/reinterpreted, documented) and do not block the package; all "
                "other red checks still block."
            ),
        },
        "figures": figures,
        "tables": {
            "table_lines": str(paths.table_lines_csv),
            "table_methods": str(paths.table_methods_csv),
            "table_qc": str(paths.table_qc_csv),
        },
        "traceability": {
            "report_source": "run_summary.json",
            "table_lines_rows": len(table_lines),
            "table_methods_rows": len(table_methods),
            "table_qc_rows": len(qc_rows),
        },
    }
    return summary, table_lines, table_methods, qc_rows


def _markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def _executive_summary(summary):
    hstage = next((row for row in summary["stages"] if row["stage"] == "E1_halpha"), {})
    e3 = next((row for row in summary["stages"] if row["stage"] == "E3_limits"), {})
    issues = summary.get("open_issues", [])
    accepted = summary.get("accepted_limitations", []) or []
    lines = [
        f"- Run: {summary['run_id']}",
        f"- Global package status: {summary['overall_status']}",
        f"- Halpha stage status: {hstage.get('status', 'unknown')}",
        f"- Limit stage status: {e3.get('status', 'unknown')}",
        f"- Open issues: {len(issues)}",
        f"- Accepted limitations (documented, non-blocking): {len(accepted)} "
        "- see the Accepted Limitations section",
    ]
    if any(a.get("stage") == "D2_calibrate" for a in accepted):
        lines.append(
            "- Red-band continuum: diagnosed as real cool-dwarf signal + an inter-method halo "
            "systematic (not a PSF defect; C1 is already Psfao). Full diagnosis: "
            "docs/2026-07-10_d2_red_continuum_diagnosis.md"
        )
    return "\n".join(lines)


def _target_display(summary):
    """Nombre legible del objeto del run que se está reportando."""
    from .targets import run_display_name

    return summary.get("target_display") or run_display_name(summary["run_id"])


def render_report_markdown(summary):
    qc_rows = summary["stages"]
    figure_rows = [
        {"figure": meta["path"], "status": meta["status"], "source": meta.get("source") or meta.get("issue", "")}
        for _key, meta in sorted(summary["figures"].items())
    ]
    table_rows = [{"table": key, "path": value} for key, value in sorted(summary["tables"].items())]
    accepted = summary.get("accepted_limitations") or []
    if accepted:
        accepted_rows = [
            {"stage": a["stage"], "check": f"{a['path']}={a['token']}", "reason": a["reason"]}
            for a in accepted
        ]
        gate = summary.get("gate_policy", {})
        accepted_md = (
            f"Frozen gate policy (hash {gate.get('accepted_limitations_hash', '?')}): these specific red "
            "checks are downgraded to documented yellow limitations and do NOT block the package; every "
            "other red check still blocks.\n\n"
            + _markdown_table(accepted_rows, ["stage", "check", "reason"])
        )
    else:
        accepted_md = "None: no red checks were downgraded by the gate policy."
    historical = "Historic local-surface context is included only when upstream Stage06 products are present in the summary."
    reproduction = "\n".join(
        [
            f"- build command: scripts/build_report.py --run-id {summary['run_id']}",
            "- report.md is generated from run_summary.json and the fixed report template.",
            "- hashes and spectral conventions are recorded in run_summary.json.",
        ]
    )
    return REPORT_TEMPLATE.format(
        executive_summary=_executive_summary(summary),
        qc_table=_markdown_table(qc_rows, ["stage", "required", "status", "issue_count"]),
        figure_list=_markdown_table(figure_rows, ["figure", "status", "source"]),
        table_list=_markdown_table(table_rows, ["table", "path"]),
        line_diagnostics=LINE_DIAGNOSTICS_NOTE,
        accretion_relation=accretion_relation_note(_target_display(summary)),
        variability_caveat=VARIABILITY_CAVEAT_NOTE,
        accepted_limitations=accepted_md,
        historical_context=historical,
        reproduction=reproduction,
    )


def build_report(run_id=None, *, project_root=None, allow_run_id_mismatch=False):
    run_config = load_run_config(
        run_id,
        project_root=project_root,
        allow_run_id_mismatch=allow_run_id_mismatch,
    )
    paths = report_paths(run_config.run_id, run_config.paths.project_root)
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.extra_dir.mkdir(parents=True, exist_ok=True)

    summary, table_lines, table_methods, table_qc = make_run_summary(run_config.run_id, run_config.paths.project_root)
    write_csv_deterministic(
        paths.table_lines_csv,
        table_lines,
        ["line", "result_type", "method", "flux_or_limit", "fap", "throughput", "l_halpha", "mdot", "source"],
    )
    write_csv_deterministic(
        paths.table_methods_csv,
        table_methods,
        ["source", "method", "pair", "band", "z", "snr", "verdict"],
    )
    write_csv_deterministic(
        paths.table_qc_csv,
        table_qc,
        ["stage", "required", "status", "qc_path", "issue_count", "summary"],
    )
    write_json_deterministic(paths.run_summary_json, summary)
    paths.report_md.write_text(render_report_markdown(summary), encoding="utf-8")
    manifest = {
        "run_summary": str(paths.run_summary_json),
        "report_md": str(paths.report_md),
        "tables": summary["tables"],
        "figures": {key: value["path"] for key, value in sorted(summary["figures"].items())},
        "overall_status": summary["overall_status"],
    }
    return {"paths": paths, "summary": summary, "manifest": manifest}


def report_tree_hash(report_dir):
    report = Path(report_dir)
    items = []
    for path in sorted(item for item in report.rglob("*") if item.is_file()):
        items.append((path.relative_to(report).as_posix(), sha256_file(path)))
    return items


def template_contains_hardcoded_numbers(template=REPORT_TEMPLATE):
    return bool(re.search(r"\d", template))


__all__ = [
    "REPORT_TEMPLATE",
    "STANDARD_FIGURES",
    "STAGE_DEFINITIONS",
    "build_report",
    "check_declared_hashes",
    "check_spectrum_conventions",
    "make_run_summary",
    "render_report_markdown",
    "report_paths",
    "report_tree_hash",
    "stage_status",
    "template_contains_hardcoded_numbers",
]
