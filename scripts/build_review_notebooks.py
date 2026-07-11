#!/usr/bin/env python3
"""Genera los notebooks de revisión A1..G5 (uno por spec) en `notebooks/`.

Cada notebook es una interfaz DELGADA sobre `musepipe/` + `scripts/`: no
reimplementa lógica. Por defecto AUDITA el QC existente del run realineado; con
`RUN=True` re-ejecuta la etapa mediante su comando canónico (idéntico al que se
documenta en la celda "Cómo ejecutar de forma independiente").

Uso:
    python scripts/build_review_notebooks.py            # genera las 25
    python scripts/build_review_notebooks.py C1 D2      # solo algunas

Regenerar es idempotente: sobrescribe los .ipynb de `notebooks/`.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
DEFAULT_RUN = "ROXs12b_realigned"


# --------------------------------------------------------------------------
# Construcción de celdas nbformat 4.5 (sin dependencias externas)
# --------------------------------------------------------------------------
def _id() -> str:
    return uuid.uuid4().hex[:12]


def _src(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "id": _id(), "source": _src(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "id": _id(),
        "execution_count": None,
        "outputs": [],
        "source": _src(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "MUSE", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------
# Comando canónico por tipo de etapa
# --------------------------------------------------------------------------
def canonical_cmd(exec_spec: dict) -> str:
    kind = exec_spec["kind"]
    if kind == "script":
        return f"bash scripts/{exec_spec['target']} --run-id $RUN"
    if kind == "module_main":
        return f"python -m {exec_spec['target']} --run-id $RUN"
    if kind == "module_run":
        mod, fn = exec_spec["target"], exec_spec["fn"]
        return f"python -c \"from {mod} import {fn}; {fn}('$RUN')\""
    if kind == "pyscript":
        return f"python scripts/{exec_spec['target']} --run-id $RUN"
    return ""  # audit


# --------------------------------------------------------------------------
# Plantilla común
# --------------------------------------------------------------------------
def build_cells(s: dict) -> list[dict]:
    cells: list[dict] = []
    spec_link = f"[`docs/{s['spec']}`](../docs/{s['spec']})" if s.get("spec") else "—"

    # 1. Encabezado / spec / rol
    cells.append(md(
        f"# {s['id']} · {s['title']}\n\n"
        f"**Spec:** {spec_link}  |  **Bloque:** {s['block']}  |  "
        f"**Run por defecto:** `{DEFAULT_RUN}`\n\n"
        f"{s['what']}\n\n"
        f"| | |\n|---|---|\n"
        f"| **Entrada** | {s['inputs']} |\n"
        f"| **Salida (QC/productos)** | {s['outputs']} |\n"
        f"| **Consume aguas abajo** | {s['downstream']} |\n"
    ))

    # 2. Cómo ejecutar de forma independiente
    cmd = canonical_cmd(s["exec"])
    if s["exec"]["kind"] == "audit":
        howto = (
            "## Cómo ejecutar de forma independiente\n\n"
            "> ⚠️ **Etapa no re-ejecutable desde raw en este repo.** En la poda WP-10 se "
            "borraron los intermedios regenerables (`muse_scibasic`, `muse_scipost`, …). "
            "Se conservaron los productos finales y todo el QC. Este notebook **audita** el "
            "producto/QC existente y documenta el comando histórico.\n\n"
            f"Comando histórico (referencia, requiere los raw + `esorex`):\n\n"
            f"```bash\nconda activate MUSE\n{s['exec'].get('hist_cmd','(ver spec)')}\n```\n"
        )
    else:
        howto = (
            "## Cómo ejecutar de forma independiente\n\n"
            "```bash\n"
            "conda activate MUSE               # kernel/env con astropy + musepipe\n"
            f"export RUN={DEFAULT_RUN}   # o ROXs12b_B_adp para comparar\n"
            f"cd {ROOT.name}                    # raíz del repo\n"
            f"{cmd}\n"
            "```\n\n"
            f"{s['exec'].get('cost','')}\n\n"
            "La celda de abajo hace lo mismo desde el notebook (guardada por `RUN`)."
        )
    cells.append(md(howto))

    # 3. Setup común
    cells.append(code(
        "import os, sys\n"
        "sys.path.insert(0, os.path.dirname(os.path.abspath('_nbcommon.py')))\n"
        "import _nbcommon as nb\n"
        f"RUN_ID = nb.resolve_run_id({s['run_override']!r})\n"
        "print('run =', RUN_ID)\n"
        "print('dir =', nb.run_dir(RUN_ID))"
    ))

    # 4. Ejecutar o auditar (guardada)
    if s["exec"]["kind"] == "audit":
        cells.append(md("## Auditar\n\nEtapa de solo-auditoría: se carga el producto/QC más abajo."))
    else:
        run_line = f"get_ipython().system({cmd.replace('$RUN', '{RUN_ID}')!r})".replace("{RUN_ID}", "' + RUN_ID + '")
        cells.append(md("## Ejecutar o auditar"))
        cells.append(code(
            "RUN = False   # -> True para RE-EJECUTAR esta etapa (regenera su QC)\n\n"
            "if RUN:\n"
            f"    cmd = {cmd!r}.replace('$RUN', RUN_ID)\n"
            "    print('ejecutando:', cmd)\n"
            "    import subprocess\n"
            "    subprocess.run(cmd, shell=True, cwd=str(nb.project_root()), check=True)\n"
            "else:\n"
            "    print('Modo auditoría (RUN=False): se carga el QC existente abajo.')"
        ))

    # 5. QC / resultados
    if s.get("qc"):
        salient = s.get("salient", [])
        cells.append(md("## QC / resultados"))
        cells.append(code(
            f"qc = nb.load_qc({s['qc']!r}, RUN_ID)\n"
            f"nb.show(qc, keys={salient!r}, title={s['id']!r})"
        ))
    else:
        cells.append(md(
            "## QC / resultados\n\n"
            "Esta etapa no escribe un QC propio en este run; su resultado queda "
            "embebido en la reducción / documentado en la nota de decisión de abajo."
        ))

    # 6. Decisiones
    dec_lines = ["## Decisiones y notas"]
    for text, doc in s["decisions"]:
        link = f" · [`docs/{doc}`](../docs/{doc})" if doc else ""
        dec_lines.append(f"- {text}{link}")
    cells.append(md("\n".join(dec_lines)))

    # 7. Checks
    if s.get("checks"):
        cells.append(md("## Checks"))
        cells.append(code(s["checks"]))

    return cells


# --------------------------------------------------------------------------
# Tabla de etapas (spec -> punto de entrada canónico -> QC -> decisiones)
# --------------------------------------------------------------------------
STAGES: list[dict] = [
    # ===================== BLOQUE A — reducción =====================
    dict(
        id="A1", slug="A1_raw_reduction", title="Reducción raw (esorex)", block="A · Reducción",
        spec="spec_A1_codex_raw_reduction.md", run_override=None,
        what="Reduce los raw MUSE con esorex y alinea las exposiciones hasta `cube_telcorr.fits`.",
        inputs="Raw MUSE + calibraciones", outputs="`cube_telcorr.fits`, `stages/stage00r_qc.json`",
        downstream="Todo el bloque B",
        exec=dict(kind="audit", hist_cmd="bash scripts/reduce_raw.sh"),
        qc="stages/stage00r_qc.json",
        salient=["shape", "sha", "offset", "esorex", "muse", "V1", "V3", "V4"],
        decisions=[
            ("**Alineación por plan B (OFFSET_LIST manual)**, no `exp_align` — era espurio; el cubo realineado ≡ ADP a través de B.", None),
            ("Provenance QC en **AMARILLO**: V1/V3/V4 pass; V2/V5/V6 no disponibles (documentado, no fallos).", None),
            ("Nada es paper-válido hasta cerrar el A-block (directiva del usuario 2026-07-07).", None),
        ],
        checks="print('cube shape / provenance:')\nnb.show(qc, keys=['shape','offset','esorex'])",
    ),
    dict(
        id="A2", slug="A2_sky_zap", title="Decisión ZAP (cielo)", block="A · Reducción",
        spec="spec_A2_codex_sky_zap.md", run_override=None,
        what="Decide y aplica (o descarta) la sustracción de cielo con ZAP.",
        inputs="Cubo reducido", outputs="Cubo con cielo tratado (sin QC separado en este run)",
        downstream="A3, A4",
        exec=dict(kind="audit", hist_cmd="bash scripts/sky_zap.sh"),
        qc=None,
        decisions=[
            ("El cubo restado de cielo no tiene skylines usables (residual al nivel de ruido); ver M1/M2 en A4, medidos del SKY_SPECTRUM cacheado.", None),
        ],
        checks=None,
    ),
    dict(
        id="A3", slug="A3_telluric", title="Corrección telúrica", block="A · Reducción",
        spec="spec_A3_codex_telluric.md", run_override=None,
        what="Corrige absorción telúrica para producir `cube_telcorr.fits`.",
        inputs="Cubo (post-cielo)", outputs="`cube_telcorr.fits`",
        downstream="A4, B1",
        exec=dict(kind="audit", hist_cmd="bash scripts/telluric.sh"),
        qc=None,
        decisions=[
            ("**Decisión: STD_TELLURIC + escala por airmass**, NO molecfit (no convergió). Justificado para el paper.", "a3_telluric_justification.md"),
        ],
        checks=None,
    ),
    dict(
        id="A4", slug="A4_cube_qc", title="QC del cubo (M1–M5)", block="A · Reducción",
        spec="spec_A4_codex_cube_qc.md", run_override=None,
        what="Métricas de calidad del cubo: solución en λ (M1/M2), flujo absoluto (M3), STAT (M5).",
        inputs="`cube_telcorr.fits`, SKY_SPECTRUM, Gaia DR3", outputs="`stages/stage00q_qc.json`",
        downstream="D2/E1 (usan σ empírico), E3 (flujo)",
        exec=dict(kind="audit",
                  hist_cmd=("# M1/M2 (LSF) desde el airglow cacheado:\n"
                            "python -m musepipe.qc.cube_qc m1m2-sky --sky-spectrum <SKY_SPECTRUM...> --qc-output <...>\n"
                            "# M3 (flujo absoluto vs Gaia RP, con growth-curve + truncación):\n"
                            "python -m musepipe.qc.cube_qc m3-flux --cube <cube_telcorr.fits> --run-id $RUN \\\n"
                            "    --aperture-correction growth_curve --truncation-correction --qc-output <...>")),
        qc="stages/stage00q_qc.json",
        salient=["m1", "m2", "lsf", "m3", "flux_factor", "m5", "factor_spaxel"],
        decisions=[
            ("**M3 CERRADO (GREEN)**: flujo absoluto validado vs Gaia DR3 RP, factor 0.973 (~3%) tras growth-curve + truncación de cola.", None),
            ("**M5 STAT en ROJO (inherente)**: el STAT subestima el ruido ~4–6× por covarianza del remuestreo → σ SIEMPRE empírico, control=objeto.", "noise_model.md"),
            ("**M2 LSF@Hα = 2.383 Å medido** del airglow (NFM más angosta que el nominal 2.6); usado en E1/E3.", None),
        ],
        checks="nb.show(qc, keys=['m3','flux_factor','m5','lsf'])",
    ),
    # ===================== BLOQUE B — preparación =====================
    dict(
        id="B1", slug="B1_load_align_crop", title="Carga / alineación / crop", block="B · Preparación",
        spec="spec_B1_codex_load_align_crop.md", run_override=None,
        what="Carga el cubo, centra y recorta al campo de interés.",
        inputs="`cube_telcorr.fits`", outputs="`stages/stage01_qc.json`",
        downstream="B2, B3, C1",
        exec=dict(kind="module_run", target="musepipe.stages.stage01_align", fn="run_stage01",
                  cost="Ligero (segundos)."),
        qc="stages/stage01_qc.json",
        salient=["star", "center", "crop", "profile", "fwhm"],
        decisions=[
            ("El shift subpixel (spline orden 3) de la alineación es el ORIGEN físico de la correlación del ruido (§4 del modelo de ruido).", "noise_model.md"),
        ],
        checks="nb.show(qc, keys=['center','crop'])",
    ),
    dict(
        id="B2", slug="B2_xcorr_stripes", title="Xcorr / franjas", block="B · Preparación",
        spec="spec_B2_codex_xcorr_stripes.md", run_override=None,
        what="Correlación cruzada entre exposiciones y detección/corrección de franjas.",
        inputs="Cubo alineado", outputs="`stages/stage02_qc.json` (+ `stage02_xcorr_qc.json`)",
        downstream="C1, 04b",
        exec=dict(kind="script", target="stage02_xcorr.sh", cost="Ligero."),
        qc="stages/stage02_xcorr_qc.json",
        salient=["stripe", "finite", "dirty", "exposure"],
        decisions=[
            ("Exposición única: no requirió corrección de franjas (finite ~94%, sin canales sucios).", None),
        ],
        checks="nb.show(qc, keys=['finite','stripe'])",
    ),
    dict(
        id="B3", slug="B3_localize", title="Localización del compañero", block="B · Preparación",
        spec="spec_B3_codex_target_localization.md", run_override=None,
        what="Localiza el compañero ROXs 12 B en el campo.",
        inputs="Cubo alineado", outputs="`stages/stage01c_qc.json`",
        downstream="C1–C4 (posición de extracción)",
        exec=dict(kind="script", target="stage01c_localize.sh", cost="Ligero."),
        qc="stages/stage01c_qc.json",
        salient=["companion", "position", "separation", "band", "detect"],
        decisions=[
            ("Banda de detección ensanchada al rojo (8800–9350 Å): el compañero es muy rojo (enana fría).", None),
        ],
        checks="nb.show(qc, keys=['position','separation'])",
    ),
    # ===================== BLOQUE C — extracción =====================
    dict(
        id="C1", slug="C1_chromatic_psf", title="PSF cromática (Moffat/Psfao)", block="C · Extracción",
        spec="spec_C1_codex_chromatic_psf.md", run_override=None,
        what="Ajusta la PSF cromática de la primaria por bin de λ, comparando Moffat y Psfao y seleccionando la mejor.",
        inputs="Cubo alineado + posición primaria", outputs="`stages/stage_e01_qc.json`, `stages/psf_model.json`",
        downstream="C3, C4, E4 (modelo de PSF)",
        exec=dict(kind="script", target="stage_e01_psf.sh",
                  cost="Moderado (Psfao con lru_cache; minutos)."),
        qc="stages/stage_e01_qc.json",
        salient=["form", "model_comparison", "ring", "residual", "psfao", "moffat"],
        decisions=[
            ("**Blocker #8 CERRADO**: C1 ajusta Moffat **y** Psfao por bin y selecciona por menor residuo del anillo (empate→Moffat).", "d1_canonical_method_decision.md"),
            ("**Forma elegida = psfao** (residuo de anillo ~4.6% vs ~19% Moffat). `psf_model.json` byte-idéntico a la etapa lateral: consolidación neutra.", None),
            ("Híbrido azimutal **descartado**: con Psfao empeoraba (5%→44%) por la FWHM inflada de Moffat.", None),
        ],
        checks="nb.show(qc, keys=['form','model_comparison','ring'])",
    ),
    dict(
        id="04b", slug="C_04b_local_surface", title="Fondo local (superficie)", block="C · Extracción",
        spec=None, run_override=None,
        what="Sustrae una superficie local al fondo alrededor del compañero.",
        inputs="Cubo alineado (native_stage02)", outputs="`stages/stage04b_qc.json`",
        downstream="C2/C3 (control local)",
        exec=dict(kind="module_run", target="musepipe.stages.stage04b_local_surface", fn="run_stage04b",
                  cost="Ligero–moderado."),
        qc="stages/stage04b_qc.json",
        salient=["surface", "residual", "pedestal", "mode"],
        decisions=[
            ("Uso operativo de la cadena local para ROXs12b documentado en la guía.", "roxs12b_clean_spectrum_pipeline.md"),
        ],
        checks="nb.show(qc, keys=['residual','mode'])",
    ),
    dict(
        id="C2", slug="C2_aperture", title="Extracción por apertura", block="C · Extracción",
        spec="spec_C2_codex_aperture_extraction.md", run_override=None,
        what="Extrae el espectro del compañero por apertura, con controles al mismo radio.",
        inputs="Cubo + posición", outputs="`stages/spec_aperture_qc.json`",
        downstream="D1, E1 (controles)",
        exec=dict(kind="script", target="stage_x01_aperture.sh", cost="Ligero–moderado."),
        qc="stages/spec_aperture_qc.json",
        salient=["aperture", "radius", "control", "background", "apcorr"],
        decisions=[
            ("Principio **control = objeto**: controles con annulus bkg + apcorr, procesados idénticos al objeto.", "noise_model.md"),
        ],
        checks="nb.show(qc, keys=['radius','control'])",
    ),
    dict(
        id="C3", slug="C3_optimal", title="Extracción óptima", block="C · Extracción",
        spec="spec_C3_codex_optimal_extraction.md", run_override=None,
        what="Extracción óptima (Horne) ponderada por la PSF.",
        inputs="Cubo + PSF (C1)", outputs="`stages/spec_optimal_qc.json`",
        downstream="D1, E1",
        exec=dict(kind="script", target="stage_x02_optimal.sh", cost="Moderado."),
        qc="stages/spec_optimal_qc.json",
        salient=["optimal", "psfsub", "local", "control", "throughput"],
        decisions=[
            ("Dos variantes: `optimal_psfsub` (validada por G1) y `optimal_ls` (rechazada, insensible en el borde).", None),
        ],
        checks="nb.show(qc, keys=['optimal','control'])",
    ),
    dict(
        id="C4", slug="C4_psffit", title="Ajuste de PSF (psffit)", block="C · Extracción",
        spec="spec_C4_codex_psf_fitting.md", run_override=None,
        what="Ajusta simultáneamente estrella + compañero sobre el modelo de PSF (método canónico).",
        inputs="Cubo + PSF (C1)", outputs="`stages/spec_psffit_qc.json`",
        downstream="D1, D2, E1, E3 (método canónico)",
        exec=dict(kind="script", target="stage_x03_psffit.sh",
                  cost="Moderado (~3 min con Psfao lru_cache; sin cache era 2h+)."),
        qc="stages/spec_psffit_qc.json",
        salient=["psffit", "companion", "flux", "throughput", "control"],
        decisions=[
            ("Maneja el gradiente de halo AO sin la sobre-sustracción de la rama local-surface; positivo y físico en el borde del compañero.", None),
        ],
        checks="nb.show(qc, keys=['flux','control'])",
    ),
    # ===================== BLOQUE D — método + calibración =====================
    dict(
        id="D1", slug="D1_method_compare", title="Comparación inter-método", block="D · Método",
        spec="spec_D1_v2_codex_method_comparison.md", run_override=None,
        what="Compara los métodos de extracción sobre 33 controles y emite `recommended_method` (nunca fija el canónico).",
        inputs="C2/C3/C4 + G1 verdicts", outputs="`stages/stage_x10_qc.json`",
        downstream="D2 (consume el canónico de config)",
        exec=dict(kind="script", target="stage_x10_compare.sh", cost="Moderado."),
        qc="stages/stage_x10_qc.json",
        salient=["verdict", "action", "reason", "recommended_method", "level_ratio"],
        decisions=[
            ("**Decisión humana: canónico = `psffit`** (validado por G1, físico en el borde), registrado en `config.json` (`x11_canonical_method`).", "d1_canonical_method_decision.md"),
            ("**Veredicto = `divergent_continuum`** (B6 rojo lejano, t=+4.5). Es divergencia REAL, no de convención (level_ratio 1.0015).", None),
            ("**B6 aceptado como sistemática presupuestada**: Psfao no la cierra; `action=iterate_C1` reconocido pero NO accionado.", None),
        ],
        checks="nb.show(qc, keys=['verdict','recommended','level_ratio'])",
    ),
    dict(
        id="D2", slug="D2_calibrate", title="Calibración espectral", block="D · Método",
        spec="spec_D2_codex_spectral_calibration.md", run_override=None,
        what="Calibra el espectro canónico (Δλ, escala de flujo, continuo) y produce `spec_final_object.fits`.",
        inputs="Método canónico (psffit) + M3", outputs="`stages/stage_x11_qc.json`, `spec_final_object.fits`",
        downstream="E1, E3, G2",
        exec=dict(kind="script", target="stage_x11_calibrate.sh", cost="Ligero–moderado."),
        qc="stages/stage_x11_qc.json",
        salient=["v3_continuum", "intermethod", "fraction_channels", "flux_factor", "scale"],
        decisions=[
            ("**Diagnóstico honesto del 'continuo rojo inestable'**: 3 cosas reales (señal de enana fría + sistemático de nivel inter-método 1.76× + rigidez del polinomio). NO es defecto de PSF.", "d2_red_continuum_diagnosis.md"),
            ("`v3_continuum_stable` recableado para gatear sobre la métrica libre de señal (`fraction_channels_methods_agree`), no sobre `|runmed−poly5|`.", None),
            ("El sistemático rojo NO afecta la línea Hα ni el límite de Ṁ → limitación aceptada documentada en F1.", None),
        ],
        checks="nb.show(qc, keys=['intermethod','fraction_channels','flux_factor'])",
    ),
    # ===================== BLOQUE E — resultado =====================
    dict(
        id="E1", slug="E1_halpha_detect", title="Detección Hα", block="E · Resultado",
        spec="spec_E1_codex_halpha_detection.md", run_override=None,
        what="Test de detección de emisión Hα del compañero con matched filter y controles.",
        inputs="`spec_final_object.fits` + controles", outputs="`stages/stage_h01_qc.json`",
        downstream="E2, E3, G2 (V3)",
        exec=dict(kind="script", target="stage_h01_detect.sh", cost="Ligero–moderado."),
        qc="stages/stage_h01_qc.json",
        salient=["verdict", "non_detection", "fap", "rv", "z", "lsf"],
        decisions=[
            ("**VEREDICTO = `non_detection`** — ningún método supera el FAP global; el pico psffit es RV-inconsistente. ENDPOINT CIENTÍFICO.", None),
            ("LSF = 2.383 Å (medida, A4/M2), 33 controles → min_resolvable_fap ≈ 0.029.", None),
        ],
        checks="nb.show(qc, keys=['verdict','fap','rv'])",
    ),
    dict(
        id="E2", slug="E2_artifacts", title="Batería de artefactos", block="E · Resultado",
        spec="spec_E2_codex_artifact_tests.md", run_override=None,
        what="Batería de tests de artefactos (T1–T5) para caracterizar la no-detección.",
        inputs="Productos de extracción", outputs="`stages/stage_h02_qc.json`",
        downstream="E3, F1",
        exec=dict(kind="script", target="stage_h02_artifacts.sh", cost="Ligero."),
        qc="stages/stage_h02_qc.json",
        salient=["overall", "t2", "chi2", "centroid", "interpretation"],
        decisions=[
            ("**T2 reinterpretado para no-detección** (spec §2): el máximo global NO tiene forma de PSF (chi2≈0.99) → *apoya* la no-detección. Es limitación aceptada en F1.", None),
        ],
        checks="nb.show(qc, keys=['overall','t2','interpretation'])",
    ),
    dict(
        id="E3", slug="E3_upper_limits", title="Límites superiores (Ṁ)", block="E · Resultado",
        spec="spec_E3_codex_upper_limits.md", run_override=None,
        what="Límite superior de la tasa de acreción Ṁ a partir de la no-detección (Gumbel 99%).",
        inputs="E1 + throughput (E4) + config físico", outputs="`stages/stage_h03_qc.json`, `tables/halpha_upper_limits.csv`",
        downstream="F1, comparación con G3",
        exec=dict(kind="script", target="stage_h03_limits.sh", cost="Ligero–moderado."),
        qc="stages/stage_h03_qc.json",
        salient=["mdot", "limit", "luminosity", "throughput", "gumbel", "definition_note"],
        decisions=[
            ("**Ṁ(99%) = 8.2×10⁻¹³ M☉/yr** (psffit, L_Hα deredden, throughput 0.80, d=138.6pc, A_V=1.8, Alcalá+2017).", "mdot_limit_definition_note.md"),
            ("Knob `h03_flux_unit_cgs=1e-20` (unidad nativa scipost) — sin él L/Ṁ salían ~10²⁰ altos.", None),
            ("Difiere de G3 (1.3×10⁻¹²) solo por DEFINICIÓN (Gumbel99 sin R_in vs 5σ con R_in); cadena física idéntica. Ninguna elegida canónica aún.", None),
        ],
        checks="nb.show(qc, keys=['mdot','throughput','definition_note'])",
    ),
    dict(
        id="E4", slug="E4_injection", title="Inyección-recuperación", block="E · Resultado",
        spec="spec_E4_codex_injection_recovery.md", run_override=None,
        what="Inyecta señal sintética y mide el throughput de cada método (grid completo 112 casos).",
        inputs="Extractores reales (C2/C3/C4) + PSF", outputs="`stages/stage_h04_qc.json`, `tables/injection_throughput_by_method.csv`",
        downstream="E3 (throughput), G1",
        exec=dict(kind="script", target="stage_h04_injection.sh",
                  cost="Pesado (~22 min grid 112 casos con `h04_process_pool`)."),
        qc="stages/stage_h04_qc.json",
        salient=["throughput", "psffit", "hierarchy", "v3_monotonic", "v4_hierarchy", "nulls"],
        decisions=[
            ("**Throughput psffit ≈ 0.80–0.90** (canónico); optimal_psfsub 0.72; aperture/optimal_ls insensibles en el borde.", None),
            ("V3-monotonic + V4-hierarchy FALLAN por patología de borde (optimal_ls/aperture) — canónico psffit NO afectado → limitación aceptada en F1.", None),
        ],
        checks="nb.show(qc, keys=['throughput','v4_hierarchy'])",
    ),
    # ===================== BLOQUE F — paquete =====================
    dict(
        id="F1", slug="F1_final_report", title="Paquete final + gate", block="F · Paquete",
        spec="spec_F1_codex_final_products.md", run_override=None,
        what="Consolida A→E en el paquete final y aplica el gate (semáforo por etapa + limitaciones aceptadas).",
        inputs="Todos los QC A→E", outputs="`report/run_summary.json`, `report/report.md`, figuras/tablas",
        downstream="Revisión humana / decisión de publicación",
        exec=dict(kind="pyscript", target="build_report.py", cost="Ligero."),
        qc="report/run_summary.json",
        salient=["overall_status", "accepted_limitations", "gate_policy", "open_issues"],
        decisions=[
            ("**Gate `overall_status: yellow`, 0 rojos bloqueantes** en el realineado (era rojo en ADP).", None),
            ("Política de gate congelada: SOLO 4 rojos específicos (A4/M5, D2 continuo, E2/T2, E4 jerarquía) bajan a 'limitación aceptada'; cualquier OTRO rojo bloquea.", "d2_red_continuum_diagnosis.md"),
            ("F1 se niega correctamente a dar luz verde de paper mientras el A-block siga provisional.", None),
        ],
        checks="print('overall_status =', qc.get('overall_status'))\nprint('accepted_limitations =', len(qc.get('accepted_limitations',[])))\nprint('open_issues =', len(qc.get('open_issues',[])))",
    ),
    # ===================== BLOQUE G — caracterización =====================
    dict(
        id="G0", slug="G0_real_cube", title="Ejecución cubo real", block="G · Caracterización",
        spec="spec_G0_codex_real_cube_execution.md", run_override=None,
        what="Verifica la cadena de hash y compara con el legacy sobre el cubo real.",
        inputs="Productos A→F", outputs="`stages/stage_g0_qc.json`, `tables/g0_legacy_comparison.csv`",
        downstream="G1–G5",
        exec=dict(kind="pyscript", target="run_g0.py", cost="Ligero."),
        qc="stages/stage_g0_qc.json",
        salient=["hash_chain", "legacy", "entry_point"],
        decisions=[
            ("G0 cerrado retroactivamente; `hash_chain_ok=True`. Comparación legacy = calibración de flujo distinta (esperado).", "g0_execution_log.md"),
        ],
        checks="nb.show(qc, keys=['hash_chain','entry_point'])",
    ),
    dict(
        id="G1", slug="G1_extraction_validation", title="Validación de extracción", block="G · Caracterización",
        spec="spec_G1_codex_extraction_validation.md", run_override=None,
        what="Valida los métodos de extracción: covarianza espectral, inflación espacial, presupuesto de sesgo.",
        inputs="Controles + E4 grid", outputs="`stages/stage_g1_qc.json`, `g1_channel_covariance.npz`",
        downstream="G2, E3 (throughput/bias)",
        exec=dict(kind="pyscript", target="run_g1.py", cost="Moderado."),
        qc="stages/stage_g1_qc.json",
        salient=["validated", "throughput", "corr_length", "n_eff", "spatial_inflation"],
        decisions=[
            ("psffit & optimal_psfsub = `validated_with_bias`; aperture & optimal_ls = `rejected` (insensibles en el borde).", "noise_model.md"),
            ("Correlación espectral 1.45±0.08 ch, n_eff/n=0.69; inflación espacial box3≈6.5×, box5≈17× → confirma A4/M5.", None),
        ],
        checks="nb.show(qc, keys=['corr_length','n_eff','spatial_inflation'])",
    ),
    dict(
        id="G2", slug="G2_measure_lines", title="Medición de líneas", block="G · Caracterización",
        spec="spec_G2_codex_spectral_measurements.md", run_override=None,
        what="Mide líneas de forma genérica (flujo, EW, centroide, FWHM, RV, límites) sobre el espectro final.",
        inputs="`spec_final_object.fits`", outputs="`stages/stage_g2_qc.json`",
        downstream="G3, G4",
        exec=dict(kind="module_run", target="musepipe.stages.stage_g2_measure_lines", fn="run_stage_g2",
                  cost="Ligero."),
        qc="stages/stage_g2_qc.json",
        salient=["detected", "upper_limit", "halpha", "consistent"],
        decisions=[
            ("0 líneas detectadas / 23 límites — consistente con la no-detección (V3: Hα upper_limit = E1 non_detection).", None),
            ("Cero lógica específica de Hα: catálogo genérico en config (24 líneas de stage07 por defecto).", None),
        ],
        checks="nb.show(qc, keys=['detected','upper_limit'])",
    ),
    dict(
        id="G3", slug="G3_accretion", title="Inferencia física", block="G · Caracterización",
        spec="spec_G3_codex_physical_inference.md", run_override=None,
        what="Infiere L_acc y Ṁ (multilínea, límite = más restrictivo) y ajusta plantillas (diferido).",
        inputs="G2 + relaciones de acreción + modelos", outputs="`stages/stage_g3_qc.json`",
        downstream="G4, G5",
        exec=dict(kind="module_run", target="musepipe.stages.stage_g3_accretion", fn="run_stage_g3_accretion",
                  cost="Moderado (MC n=2000)."),
        qc="stages/stage_g3_qc.json",
        salient=["mdot", "l_acc", "not_constrained", "template", "definition_note"],
        decisions=[
            ("**Ṁ ≈ 1.3×10⁻¹² M☉/yr** (5σ de G2 + factor R_in 1.25); difiere de E3 solo por definición.", "mdot_limit_definition_note.md"),
            ("Plantilla/atmósfera/tracks = `not_constrained` (pending_libraries: BT-Settl/BHAC15/Luhman-Bonnefoy diferidas).", None),
        ],
        checks="nb.show(qc, keys=['mdot','l_acc','not_constrained'])",
    ),
    dict(
        id="G4", slug="G4_classify", title="Clasificación de fuente", block="G · Caracterización",
        spec="spec_G4_codex_source_classification.md", run_override=None,
        what="Clasifica la fuente con una matriz hipótesis×test (umbrales congelados anti-sesgo).",
        inputs="G2/G3 + astrometría + densidad de fondo", outputs="`stages/stage_g4_classification.json`",
        downstream="G5",
        exec=dict(kind="module_run", target="musepipe.stages.stage_g4_classify", fn="run_stage_g4",
                  cost="Ligero."),
        qc="stages/stage_g4_classification.json",
        salient=["final_class.label", "robustness", "background_probability", "exclusion", "leave_one_out"],
        decisions=[
            ("**Clase = `substellar_companion`, robustez = `ambiguous`**: es compañero REAL ligado (no fondo P=6.3e-4, no artefacto), pero planeta/BD/M no se resuelve sin tipado espectral.", None),
            ("Resolver la ambigüedad requiere G3 real (SpT/Teff) + 2ª época astrométrica + densidad de fondo final.", None),
        ],
        checks="nb.show(qc, keys=['final_class.label','robustness','background_probability'])",
    ),
    dict(
        id="G5", slug="G5_final_synthesis", title="Síntesis final", block="G · Caracterización",
        spec="spec_G5_codex_final_synthesis.md", run_override=None,
        what="Consolida G0–G4 en el paquete de caracterización (tablas + supuestos + síntesis).",
        inputs="G0–G4", outputs="`report/characterization/` (6 tablas + md + summary.json)",
        downstream="Revisión humana",
        exec=dict(kind="pyscript", target="build_characterization.py", cost="Ligero."),
        qc="report/characterization/characterization_summary.json",
        salient=["class", "mdot", "traceability", "consistency", "provisional"],
        decisions=[
            ("G-block CERRADO (provisional): paquete completo pero hereda el bloqueo del A-block + G3 diferido → clasificación ambigua.", None),
            ("Determinismo verificado (dos builds → hash idéntico); F1 intacto (V6).", None),
        ],
        checks="nb.show(qc, keys=['class','mdot'])",
    ),
]


def main(argv: list[str]) -> None:
    want = {a.upper() for a in argv}
    NB_DIR.mkdir(exist_ok=True)
    written = []
    for s in STAGES:
        if want and s["id"].upper() not in want:
            continue
        nb = notebook(build_cells(s))
        out = NB_DIR / f"{s['slug']}.ipynb"
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
        written.append(out.name)
    print(f"Generados {len(written)} notebooks en {NB_DIR}/:")
    for name in written:
        print("  ", name)


if __name__ == "__main__":
    main(sys.argv[1:])
