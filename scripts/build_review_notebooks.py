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

    # 1b. Narrativa opcional (qué es / por qué), antes del cómo-ejecutar
    if s.get("narrative_md"):
        cells.append(md(s["narrative_md"]))

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

    # 2b. Coste de ejecución (opcional, p.ej. esorex en A1)
    if s.get("runtime_md"):
        cells.append(md(s["runtime_md"]))
        if s.get("runtime_code"):
            cells.append(code(s["runtime_code"]))

    # 3. Setup común
    cells.append(code(
        "import os, sys\n"
        "# Añade notebooks/ (para _nbcommon) y la RAÍZ del repo (para importar musepipe),\n"
        "# funcione el cwd en notebooks/ o en la raíz del repo.\n"
        "_here = os.getcwd()\n"
        "if os.path.basename(_here) != 'notebooks' and os.path.isdir(os.path.join(_here, 'notebooks')):\n"
        "    _here = os.path.join(_here, 'notebooks')\n"
        "for _p in (_here, os.path.dirname(_here)):\n"
        "    if _p not in sys.path:\n"
        "        sys.path.insert(0, _p)\n"
        "import _nbcommon as nb\n"
        "_root = str(nb.project_root())\n"
        "if _root not in sys.path:\n"
        "    sys.path.insert(0, _root)   # asegura 'import musepipe'\n"
        f"RUN_ID = nb.resolve_run_id({s['run_override']!r})\n"
        "print('run  =', RUN_ID)\n"
        "print('root =', _root)\n"
        "print('dir  =', nb.run_dir(RUN_ID))"
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
    elif not (s.get("evidence_md") or s.get("evidence_code")):
        cells.append(md(
            "## QC / resultados\n\n"
            "Esta etapa no escribe un QC propio en este run; su resultado queda "
            "embebido en la reducción / documentado en la nota de decisión de abajo."
        ))

    # 5b. Evidencia / explicación a medida (adicional; puede coexistir con el QC)
    if s.get("evidence_md"):
        cells.append(md(s["evidence_md"]))
    if s.get("evidence_code"):
        cells.append(code(s["evidence_code"]))

    # 5c. Plot opcional único (requiere kernel MUSE)
    if s.get("plot_md"):
        cells.append(md(s["plot_md"]))
    if s.get("plot_code"):
        cells.append(code(s["plot_code"]))

    # 5d. Varios plots (lista de {md, code})
    for _p in s.get("plots", []):
        if _p.get("md"):
            cells.append(md(_p["md"]))
        if _p.get("code"):
            cells.append(code(_p["code"]))

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

    # 8. Conclusión fechada (opcional)
    if s.get("conclusion_md"):
        cells.append(md(s["conclusion_md"]))

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
        runtime_md=(
            "## Coste de ejecución (esorex)\n\n"
            "> ⏱️ **Referencia real** medida en esta máquina (esorex 3.13.10 / MUSE 2.10.16, "
            "dataset NFM-AO de ROXs 12: **7 exposiciones × 24 IFUs = 168 pixtables**).\n\n"
            "| Receta | Tiempo | Escala con |\n|---|---:|---|\n"
            "| bias | 33 min | calibración (~fijo) |\n"
            "| flat | 52 min | calibración |\n"
            "| wavecal | 51 min | calibración |\n"
            "| lsf | 50 min | calibración |\n"
            "| scibasic (std) | 4.5 min | 1× |\n"
            "| standard | 1.6 min | 1× |\n"
            "| scibasic (object) | 24 min | **N_exp** |\n"
            "| scipost | 73 min | **N_exp** |\n"
            "| **Total (7 exp)** | **≈ 289 min (~4.8 h)** | |\n\n"
            "**Fórmula para datos nuevos** (mismo instrumento/máquina):\n\n"
            "```\n"
            "T(min) ≈ T_cal + T_std + N_exp·(t_scibasic + t_scipost) + T_combine\n"
            "```\n\n"
            "con constantes medidas aquí:\n\n"
            "- `T_cal ≈ 186 min` = bias+flat+wavecal+lsf. **Una vez por noche/modo**; "
            "`0` si reutilizas los master calibrations.\n"
            "- `T_std ≈ 6 min` = scibasic_std + standard (una vez).\n"
            "- `t_scibasic ≈ 3.4 min/exp`, `t_scipost ≈ 10.5 min/exp` (24 IFUs c/u; plan B = "
            "scipost por exposición).\n"
            "- `T_combine ≈ 5 min` = muse_exp_align + muse_exp_combine (plan B, offsets manuales).\n\n"
            "**Nota:** scibasic/scipost paralelizan sobre los 24 IFUs (OpenMP) → el tiempo escala "
            "aprox. inverso al nº de núcleos; `cores_factor` ajusta ese factor respecto a esta máquina "
            "base (=1.0). La calibración domina: reutilizar masters recorta ~3 h."
        ),
        runtime_code=(
            "def estimate_esorex_runtime(n_exp, reuse_calibrations=False, cores_factor=1.0):\n"
            "    \"\"\"Estima el wall-time de la reducción esorex (min), calibrada en la\n"
            "    máquina de referencia (7 exp NFM-AO ~= 289 min). Ver tabla de arriba.\"\"\"\n"
            "    T_cal = 0.0 if reuse_calibrations else 186.0  # bias+flat+wavecal+lsf\n"
            "    T_std = 6.0                                    # scibasic_std + standard\n"
            "    t_scibasic, t_scipost = 3.4, 10.5             # min por exposición (24 IFU)\n"
            "    T_combine = 5.0                                # exp_align + exp_combine\n"
            "    return (T_cal + T_std + n_exp * (t_scibasic + t_scipost) + T_combine) / cores_factor\n\n"
            "for n in (1, 3, 7, 10):\n"
            "    m = estimate_esorex_runtime(n)\n"
            "    print(f'{n:2d} exp  ->  {m:5.0f} min  (~{m/60:.1f} h)')\n"
            "print('7 exp reutilizando masters ->',\n"
            "      f'{estimate_esorex_runtime(7, reuse_calibrations=True):.0f} min')"
        ),
        qc="stages/stage00r_qc.json",
        salient=["shape", "sha", "offset", "esorex", "muse"],
        evidence_md=(
            "## Verificaciones (V1–V6): qué comprueban y qué respondieron\n\n"
            "El QC de A1 (`stage00r_qc.json`) no re-reduce: **verifica** que el cubo entregado es "
            "sano y trazable. Cada check tiene un significado concreto:\n\n"
            "| Check | Qué comprueba | Resultado | Significado |\n|---|---|---|---|\n"
            "| **V1** STAT | La extensión STAT (varianza) existe, es positiva y con pocos NaN "
            "(excluyendo canales láser AO y spaxels de borde) | **ok** (NaN 0.24%, 216 canales láser "
            "y 6972 spaxels de borde excluidos) | El cubo trae su mapa de varianza y no está corrupto "
            "→ base para toda la propagación de error aguas abajo |\n"
            "| **V2** estándar | Continuo del estándar vs su curva de respuesta | **unavailable** | "
            "No hay curva de respuesta del estándar para esta reducción → no se pudo cerrar la "
            "validación de flujo relativa aquí (se cierra por otra vía en A4/M3 vs Gaia) |\n"
            "| **V3** WCS | `CRVAL3` y paso espectral correctos | **ok** | Solución de longitud de "
            "onda y WCS sanos → los λ del cubo son fiables |\n"
            "| **V4** vs ADP | Correlación de la imagen luz-blanca del cubo propio con la del ADP de "
            "ESO | **ok** (corr = 0.9994, shift entero (0,0)) | El cubo auto-reducido reproduce la "
            "morfología del ADP oficial → validación cruzada independiente de la reducción |\n"
            "| **V5** espectro estelar | Razón del espectro de la estrella entre cubo y ADP | "
            "**unavailable** | Los dos cubos ponen la estrella en píxeles distintos; hace falta "
            "registro por-cubo → no comparable con la API punto-único |\n"
            "| **V6** máscara de cielo | La máscara de cielo de scipost es limpia | **unavailable** | "
            "scipost no exporta la máscara como producto 2D verificable → no auditable aquí |\n\n"
            "V1/V3/V4 pasan; V2/V5/V6 son **lagunas de proveniencia documentadas** (no fallos "
            "físicos). Por eso el semáforo A1 = **yellow**. La celda de abajo los imprime en vivo."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage00r_qc.json', RUN_ID)\n"
            "labels = {\n"
            "    'v1_stat_present':      'V1 · STAT presente y sano',\n"
            "    'v2_std_residual_rms':  'V2 · Residuo del estándar (respuesta de flujo)',\n"
            "    'v3_wcs_ok':            'V3 · WCS / eje espectral',\n"
            "    'v4_adp_whitelight_corr':'V4 · Correlación luz-blanca vs ADP',\n"
            "    'v5_adp_star_spec_ratio':'V5 · Razón de espectro estelar vs ADP',\n"
            "    'v6_sky_mask_clean':    'V6 · Máscara de cielo limpia',\n"
            "}\n"
            "ver = q.get('verification', {})\n"
            "for k, lab in labels.items():\n"
            "    v = ver.get(k, {})\n"
            "    res = 'ok' if v.get('ok') else v.get('status', '?')\n"
            "    print(f'{lab}\\n   -> {res}\\n   {v.get(\"message\", \"\")}\\n')"
        ),
        decisions=[
            ("**Alineación por plan B (OFFSET_LIST manual)**, no `exp_align` — daba offsets espurios de hasta 3.305\" (cross-match de speckles NFM); el manual desde el centroide de la primaria da máx 0.62\". El cubo realineado ≡ ADP a través del bloque B.", None),
            ("Provenance QC = **AMARILLO**: V1/V3/V4 pasan; V2/V5/V6 = `unavailable` (lagunas documentadas, no fallos). Ver tabla de verificaciones arriba.", None),
            ("**Estado A-block (actualizado 2026-07-10):** los 6 blockers duros están **CERRADOS** → F1 realineado = `yellow`, **0 bloqueantes**. Quedan 4 `open_issues` NO bloqueantes (V2/V5/V6, agrupación de calibraciones BIAS, molecfit no convergió→STD_TELLURIC). El paquete ya **no bloquea por A**; la validez para paper es juicio científico con esos caveats declarados. **Supera la directiva absoluta del 2026-07-07.**", None),
        ],
        checks="print('open_issues A1 (no bloqueantes):')\nfor i, s in enumerate(qc.get('open_issues', []), 1):\n    print(f'  {i}. {s}')",
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**A1: `cube_telcorr.fits` reducido y alineado; semáforo A1 = `yellow` (no bloqueante).**\n\n"
            "- **Fecha:** reducción 2026-07-08 (esorex 3.13.10 / MUSE 2.10.16); provenance QC escrito "
            "2026-07-09.\n"
            "- **Datos:** 7 exposiciones NFM-AO (OB 3444577, Prog 109.23B7.002, "
            "MUSE.2022-09-01T00:36–02:03).\n"
            "- **Alineación:** plan B, OFFSET_LIST manual desde el centroide de la primaria (máx "
            "0.62\"), porque `muse_exp_align` dio offsets espurios de hasta 3.305\".\n"
            "- **Verificaciones:** V1/V3/V4 pass; V2/V5/V6 `unavailable` (documentadas).\n"
            "- **Cubo:** 3681×330×338, sha256 `9fff16b7…`; corr luz-blanca vs ADP = 0.9994.\n"
            "- **A-block:** 6 blockers duros cerrados (F1 yellow, 0 bloqueantes); 4 caveats no "
            "bloqueantes documentados. Nada es aún paper-final sin declarar esos caveats."
        ),
    ),
    dict(
        id="A2", slug="A2_sky_zap", title="Decisión ZAP (cielo)", block="A · Reducción",
        spec="spec_A2_codex_sky_zap.md", run_override=None,
        what="Decide y aplica (o descarta) la sustracción de cielo con ZAP.",
        inputs="Cubo reducido", outputs="Cubo con cielo tratado (sin QC separado en este run)",
        downstream="A3, A4",
        exec=dict(kind="audit", hist_cmd="bash scripts/sky_zap.sh"),
        qc=None,
        narrative_md=(
            "## Qué es ZAP y por qué se necesita\n\n"
            "**ZAP** (*Zurich Atmosphere Purge*, Soto et al. 2016) es una sustracción de "
            "**residuos de cielo** para MUSE basada en PCA. El pipeline "
            "(`muse_scipost subtract_sky`) ya resta un modelo de cielo, pero el **airglow** "
            "(líneas de OH y [O I] atmosféricas) es intenso y **varía en el tiempo** entre la "
            "exposición de ciencia y el modelo → suelen quedar **residuos de skylines**. ZAP "
            "construye una base PCA con los spaxels de **cielo** (con las fuentes enmascaradas) "
            "y elimina las componentes que describen esos residuos, dejando la señal astrofísica.\n\n"
            "**Por qué importa aquí:** un residuo de skyline mal restado puede **imitar o "
            "contaminar** una línea espectral. Buscamos una línea débil de Hα en el compañero, "
            "así que el cielo residual es un contaminante de primer orden.\n\n"
            "**El peligro (por qué NO se aplica a ciegas):** ZAP necesita suficientes spaxels de "
            "cielo *reales*. En el **campo diminuto de NFM**, con una estrella brillante y su "
            "compañero, la fracción de cielo es baja y las eigencomponentes pueden **absorber "
            "señal del compañero** — incluso *fabricar o borrar* una línea en Hα. Regla del "
            "proyecto: ante la duda, **no tocar la señal**.\n\n"
            "**Decisión pre-registrada** (`musepipe.reduction.sky_zap.classify_zap_decision`), "
            "por métrica, no por juicio. `R` = RMS mediano en ventanas de skyline ÷ RMS mediano "
            "en continuo, medido en aperturas de cielo vacías:\n\n"
            "| Condición | Decisión |\n|---|---|\n"
            "| `R ≤ 1.5` | **no necesario** → `zap_applied = False` |\n"
            "| `R > 2.0` | **necesario** → `zap_applied = True` |\n"
            "| `1.5 < R ≤ 2.0` | zona gris → checkpoint (no aplicar, preguntar) |\n"
            "| fracción de cielo `< 0.25` | cielo insuficiente → checkpoint (no aplicar) |\n\n"
            "Los parámetros de ZAP quedan en *default*; no se itera buscando 'el mejor resultado'."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Métrica **M4 de cielo** del QC del cubo (`stages/stage00q_qc.json`) aplicada a la "
            "regla de decisión pre-registrada."
        ),
        evidence_code=(
            "qc = nb.load_qc('stages/stage00q_qc.json', RUN_ID)\n"
            "m4 = qc.get('m4_sky', {})\n"
            "R = m4.get('R')\n"
            "print(f'm4_sky.R = {R}   (RMS skyline / RMS continuo en aperturas vacías)   "
            "[status {m4.get(\"status\")}]')\n\n"
            "# Regla pre-registrada (sky_zap.classify_zap_decision): low=1.5, high=2.0\n"
            "LOW, HIGH = 1.5, 2.0\n"
            "if R is None:\n"
            "    decision = 'sin dato'\n"
            "elif R <= LOW:\n"
            "    decision = 'not_needed  ->  zap_applied = False'\n"
            "elif R > HIGH:\n"
            "    decision = 'needed      ->  zap_applied = True'\n"
            "else:\n"
            "    decision = 'gray_zone   ->  checkpoint (no aplicar)'\n"
            "print(f'Regla:  R <= {LOW} no necesario | R > {HIGH} necesario     =>     {decision}')\n\n"
            "print()\n"
            "print('Corroboración (nota del QC):')\n"
            "print('  ', qc.get('note'))\n"
            "print('M1/M2 se midieron del SKY_SPECTRUM cacheado (airglow, 32 exp,',\n"
            "      qc.get('m1_wavelength', {}).get('n_measurements'), 'medidas) porque el')\n"
            "print('cubo restado de cielo tiene <8 skylines usables.')"
        ),
        plot_md=(
            "## De dónde sale R: los datos y la zona\n\n"
            "**Un solo FITS:** `cube_telcorr.fits` (el producto de A1), extensión **DATA** "
            "(la STAT no interviene en R). R se mide sobre los **spaxels de cielo vacíos** de ese "
            "cubo — no hay varios archivos. *(Los 32 `SKY_SPECTRUM` cacheados son de M1/M2, no de R.)*\n\n"
            "El gráfico reproduce M4 con las funciones canónicas (`compute_sky_residual_metrics`, "
            "`SKYLINE_WINDOWS`, `CONTINUUM_WINDOWS`):\n\n"
            "- **Izquierda:** RMS por canal en las aperturas de cielo vs λ. En rojo las ventanas de "
            "skyline, en verde las de continuo; las punteadas son las medianas cuyo cociente es `R`. "
            "Se ven los residuos de OH en el rojo (>7200 Å), pero su mediana queda **por debajo** del "
            "continuo → R < 1.\n"
            "- **Derecha:** la zona de cielo usada (azul) sobre la luz-blanca; el halo AO de la "
            "primaria queda excluido.\n\n"
            "> Necesita el kernel **MUSE** (astropy) y el cubo en disco. Usa una máscara de cielo "
            "aproximada (percentil 30 de flujo), así que el R reproducido (~0.49) difiere levemente "
            "del oficial 0.547 (máscara de A4); la conclusión `R ≤ 1.5` es idéntica."
        ),
        plot_code=(
            "MAKE_PLOT = True   # carga el cubo (~3.3 GB) vía astropy; requiere kernel MUSE\n"
            "if MAKE_PLOT:\n"
            "    try:\n"
            "        import numpy as np\n"
            "        import matplotlib.pyplot as plt\n"
            "        from astropy.io import fits\n"
            "        from musepipe.reduction.sky_zap import (\n"
            "            compute_sky_residual_metrics, SKYLINE_WINDOWS, CONTINUUM_WINDOWS)\n\n"
            "        qc = nb.load_qc('stages/stage00q_qc.json', RUN_ID)\n"
            "        cube_path = qc.get('input_cube') or qc.get('cube', {}).get('file')\n"
            "        print('FITS usado:', cube_path)\n\n"
            "        h = fits.open(cube_path, memmap=True)\n"
            "        data = np.asarray(h[1].data, dtype=np.float32)\n"
            "        hd = h[1].header\n"
            "        n3 = hd['NAXIS3']\n"
            "        wave = hd['CRVAL3'] + (np.arange(n3) - (hd['CRPIX3'] - 1)) * hd['CD3_3']\n\n"
            "        wl = np.nanmedian(data, axis=0)\n"
            "        finite = np.isfinite(wl)\n"
            "        thr = np.nanpercentile(wl[finite], 30)\n"
            "        sky_mask = finite & (wl < thr)   # cielo vacio ~ spaxels mas debiles\n\n"
            "        m = compute_sky_residual_metrics(data.astype(np.float64), wave, sky_mask)\n"
            "        R = m['R_skyline_over_continuum']\n"
            "        rms = m['channel_rms']\n"
            "        print(f'R reproducido = {R:.3f}   (oficial m4_sky.R = {qc.get(\"m4_sky\", {}).get(\"R\")})')\n\n"
            "        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2),\n"
            "                                       gridspec_kw={'width_ratios': [2, 1]})\n"
            "        ax1.plot(wave, rms, lw=0.5, color='0.35')\n"
            "        for i, (a, b) in enumerate(SKYLINE_WINDOWS):\n"
            "            ax1.axvspan(a, b, color='tab:red', alpha=0.18,\n"
            "                        label='ventana skyline' if i == 0 else None)\n"
            "        for i, (a, b) in enumerate(CONTINUUM_WINDOWS):\n"
            "            ax1.axvspan(a, b, color='tab:green', alpha=0.25,\n"
            "                        label='ventana continuo' if i == 0 else None)\n"
            "        ax1.axhline(m['skyline_rms_median'], color='tab:red', ls='--', lw=1)\n"
            "        ax1.axhline(m['continuum_rms_median'], color='tab:green', ls='--', lw=1)\n"
            "        ax1.set_xlabel('λ [Å]'); ax1.set_ylabel('RMS por canal (aperturas de cielo)')\n"
            "        ax1.set_title(f'M4: RMS vs λ  →  R = med(skyline)/med(continuo) = {R:.3f}')\n"
            "        ax1.set_ylim(0, np.nanpercentile(rms, 99)); ax1.legend(fontsize=8)\n\n"
            "        ax2.imshow(np.log10(np.clip(wl, 1, None)), origin='lower', cmap='gray')\n"
            "        ov = np.zeros((*wl.shape, 4)); ov[sky_mask] = [0.1, 0.5, 1.0, 0.5]\n"
            "        ax2.imshow(ov, origin='lower')\n"
            "        ax2.set_title('Zona de cielo (azul) sobre luz-blanca'); ax2.axis('off')\n"
            "        fig.tight_layout()\n\n"
            "        outdir = nb.run_dir(RUN_ID) / 'plots' / 'a2_m4'\n"
            "        outdir.mkdir(parents=True, exist_ok=True)\n"
            "        fig.savefig(outdir / 'm4_R.png', dpi=110)\n"
            "        print('figura ->', outdir / 'm4_R.png')\n"
            "        plt.show()\n"
            "        h.close()\n"
            "    except Exception as e:\n"
            "        print('No se pudo generar el plot:', type(e).__name__, e)\n"
            "        print('Necesita el kernel MUSE (astropy) y el cubo en disco (campo input_cube del QC).')"
        ),
        decisions=[
            ("La métrica M4 (`R`) y la caracterización del airglow viven en A4/`stage00q_qc.json`; la regla de decisión, en `musepipe/reduction/sky_zap.py`.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**Decisión: ZAP NO aplicado — `zap_applied = False` (`not_needed`).**\n\n"
            "- **Fecha del análisis:** 2026-07-09 (QC A4/M4 sobre el cubo realineado, commit "
            "`700f009`); el cubo se redujo el 2026-07-08.\n"
            "- **Datos:** cubo NFM-AO auto-reducido `cube_telcorr.fits` (OB 3444577, "
            "Prog 109.23B7.002, **7 exposiciones** MUSE.2022-09-01T00:36–02:03) + `SKY_SPECTRUM` "
            "cacheado (32 exposiciones) para caracterizar el airglow.\n"
            "- **Evidencia:** `R = 0.547 ≤ 1.5` (umbral *no necesario*); el cubo restado de cielo "
            "tiene **<8 skylines usables** (residual al nivel de ruido). En el campo diminuto NFM, "
            "ZAP aportaría ~0 y arriesgaría absorber señal del compañero.\n"
            "- **Estado M4 = yellow:** el residuo es bajo, pero la escasez de skylines hace la "
            "métrica menos robusta que en WFM. No bloqueante.\n"
            "- **Para el paper:** registrar como decisión con su métrica (R=0.547), **no** como "
            "omisión. Nada es paper-válido hasta cerrar el A-block."
        ),
    ),
    dict(
        id="A3", slug="A3_telluric", title="Corrección telúrica", block="A · Reducción",
        spec="spec_A3_codex_telluric.md", run_override=None,
        what="Corrige absorción telúrica para producir `cube_telcorr.fits`.",
        inputs="Cubo (post-cielo)", outputs="`cube_telcorr.fits`",
        downstream="A4, B1",
        exec=dict(kind="audit", hist_cmd="bash scripts/telluric.sh"),
        qc=None,
        narrative_md=(
            "## Qué es la corrección telúrica y por qué STD_TELLURIC (no molecfit)\n\n"
            "La atmósfera terrestre imprime **bandas de absorción** (O₂, H₂O) sobre el espectro, "
            "sobre todo en el rojo (>6800 Å): O₂ B ~6870 Å, la fuerte O₂ A ~7600 Å, y H₂O en "
            "~7200/8200/9300 Å. **No son astrofísicas** — para recuperar la forma real del continuo "
            "(del compañero, muy rojo) hay que **dividir por la transmisión atmosférica**.\n\n"
            "Dos caminos:\n"
            "- **molecfit** — ajuste de un modelo físico de la atmósfera (lo preferido).\n"
            "- **STD_TELLURIC** — la transmisión *observada* en la estrella estándar, escalada a la "
            "masa de aire de la ciencia por Beer–Lambert: `T_sci(λ) = T_std(λ)^(X_sci/X_std)`.\n\n"
            "**Aquí molecfit FALLÓ:** el perfil atmosférico **GDAS** para la fecha/coordenada no "
            "estaba disponible → el χ² quedó **congelado** (no mejora entre iteraciones) → "
            "transmisión→0, inutilizable. Es un problema de **configuración** (GDAS ausente en el "
            "`telluriccorr` instalado), no contaminación estelar. Por eso se usó STD_TELLURIC "
            "escalado — el respaldo habitual en MUSE.\n\n"
            "**Ventanas protegidas (`T ≡ 1`, la corrección NO se aplica):**\n"
            "- **6540–6590 Å** = Hα del compañero (diagnóstico de acreción).\n"
            "- **5780–6050 Å** = láser AO de NFM.\n\n"
            "**Consecuencia clave:** Hα está esencialmente libre de telúricas y su ventana está "
            "protegida → **A3 NUNCA afecta el resultado científico** (el límite de Hα). A3 solo "
            "importa para la fidelidad del **continuo rojo** que usan D1 y el modelado."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "El run realineado **no** emitió `stage00t_qc.json`; reutiliza el mismo método y "
            "transmisión. Las métricas de verificación provienen del QC del run crudo "
            "(`ROXs12b_raw`), que documenta la reducción telúrica de referencia."
        ),
        evidence_code=(
            "qt = nb.load_qc('stages/stage00t_qc.json', 'ROXs12b_raw')   # QC telúrico de referencia\n"
            "d, fit, ver = qt['decision'], qt['fit'], qt['verification']\n"
            "print('Decisión:', d['verdict'], '| aplicado:', d['telluric_applied'],\n"
            "      '| checkpoint:', d['user_checkpoint'], '| umbral:', d['threshold_pct'], '%')\n"
            "print('Método:', d['method'])\n"
            "print()\n"
            "print(f\"Escala airmass: X_std={fit['airmass_std']} -> X_sci={fit['airmass_sci']}\"\n"
            "      f\"  (exponente {fit['airmass_sci']/fit['airmass_std']:.3f})\")\n"
            "print()\n"
            "print('Profundidad de banda pre -> post:')\n"
            "pp = ver['v1_o2_depth_pre_post_pct']\n"
            "print(f'  O2 B (~6870 A): {pp[0]}% -> {pp[1]}%')\n"
            "print(f\"  fuera de bandas sin cambio: {ver['v2_outside_bands_unchanged']}\")\n"
            "print(f\"  Halpha intacta: {ver['v3_halpha_untouched']}   transmisión física [0,1]: {ver['v4_transmission_physical']}\")\n"
            "print()\n"
            "print('molecfit (por qué no):')\n"
            "print(' ', qt['open_issues'][0])"
        ),
        plot_md=(
            "## De dónde sale la corrección: la transmisión aplicada\n\n"
            "**FITS usado:** `TELLURIC_TRANS.fits` (transmisión 1D derivada de "
            "`STD_TELLURIC_0001.fits` de `muse_standard`, escalada a la masa de aire de la ciencia). "
            "Archivo pequeño — no hace falta el cubo.\n\n"
            "El gráfico muestra la transmisión vs λ: se ven las bandas telúricas (O₂ B ~6870, la "
            "fuerte O₂ A ~7600, H₂O ~7200/8200/9300) y las **ventanas protegidas** (gris, `T ≡ 1`) — "
            "la de Hα (6540–6590) queda plana justo antes de O₂ B."
        ),
        plot_code=(
            "MAKE_PLOT = True   # archivo pequeño; requiere kernel MUSE (astropy)\n"
            "if MAKE_PLOT:\n"
            "    try:\n"
            "        import os\n"
            "        import numpy as np\n"
            "        import matplotlib.pyplot as plt\n"
            "        from astropy.io import fits\n\n"
            "        qc = nb.load_qc('stages/stage00q_qc.json', RUN_ID)\n"
            "        cube_path = qc.get('input_cube') or qc.get('cube', {}).get('file')\n"
            "        trans_path = os.path.join(os.path.dirname(cube_path), 'TELLURIC_TRANS.fits')\n"
            "        if not os.path.exists(trans_path):\n"
            "            trans_path = str(nb.project_root() / 'runs' / 'ROXs12b_raw' / 'raw_reduction' / 'TELLURIC_TRANS.fits')\n"
            "        print('FITS usado:', trans_path)\n\n"
            "        h = fits.open(trans_path); t = h[1].data\n"
            "        wave = np.asarray(t['wave_A'], dtype=float)\n"
            "        trans = np.asarray(t['transmission'], dtype=float)\n\n"
            "        fig, ax = plt.subplots(figsize=(11, 4))\n"
            "        ax.plot(wave, trans, lw=1.1, color='tab:blue')\n"
            "        for i, (a, b) in enumerate([(6540, 6590), (5780, 6050)]):\n"
            "            ax.axvspan(a, b, color='0.6', alpha=0.35,\n"
            "                       label='ventana protegida (T≡1)' if i == 0 else None)\n"
            "        for x, lab in [(6870, 'O₂ B'), (7200, 'H₂O'), (8200, 'H₂O'), (9300, 'H₂O')]:\n"
            "            ax.annotate(lab, (x, np.interp(x, wave, trans)), textcoords='offset points',\n"
            "                        xytext=(0, -14), ha='center', fontsize=8, color='tab:red')\n"
            "        ax.set_xlabel('λ [Å]'); ax.set_ylabel('Transmisión telúrica aplicada')\n"
            "        ax.set_title('A3 · TELLURIC_TRANS.fits (STD_TELLURIC escalado a airmass sci)')\n"
            "        ax.set_ylim(0, 1.05); ax.legend(fontsize=8); fig.tight_layout()\n\n"
            "        outdir = nb.run_dir(RUN_ID) / 'plots' / 'a3_telluric'\n"
            "        outdir.mkdir(parents=True, exist_ok=True)\n"
            "        fig.savefig(outdir / 'transmission.png', dpi=110)\n"
            "        print('figura ->', outdir / 'transmission.png')\n"
            "        plt.show(); h.close()\n"
            "    except Exception as e:\n"
            "        print('No se pudo generar el plot:', type(e).__name__, e)\n"
            "        print('Necesita el kernel MUSE (astropy) y TELLURIC_TRANS.fits en disco.')"
        ),
        decisions=[
            ("**STD_TELLURIC + escala por airmass**, NO molecfit (no convergió por perfil GDAS ausente). Método de respaldo habitual en MUSE.", "a3_telluric_justification.md"),
            ("Etapa **condicional**: `verdict=needed` por umbral (profundidad máx 6.76% > 3%) + checkpoint humano **aprobado**.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**A3: corrección telúrica APLICADA por STD_TELLURIC escalado a airmass "
            "(`telluric_applied = True`, `needed`, checkpoint aprobado).**\n\n"
            "- **Fecha:** QC telúrico de referencia 2026-07-06 (`ROXs12b_raw`); justificación para "
            "paper 2026-07-10 (`docs/a3_telluric_justification.md`).\n"
            "- **Datos:** `STD_TELLURIC_0001.fits` (muse_standard), escala airmass 1.087 → 1.158 "
            "(exp. 1.065); aplicado → `cube_telcorr.fits`; transmisión en `TELLURIC_TRANS.fits`.\n"
            "- **Evidencia:** O₂ B 6.76% → 0.6%; continuo fuera de bandas sin cambio; Hα intacta "
            "(ventana protegida); STAT escala como T².\n"
            "- **molecfit no convergió** (perfil GDAS ausente → χ² congelado): problema de "
            "configuración documentado; residuo O₂ ~0.6% presupuestado para D2.\n"
            "- **El realineado no emitió su propio `stage00t_qc.json`** (usa el método/transmisión "
            "del crudo). Pendiente: regenerar un QC telúrico propio si se quiere métrica pre/post.\n"
            "- **Impacto en Hα = nulo** (ventana protegida) → el límite de acreción no depende de "
            "esta corrección."
        ),
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
        salient=["m1_wavelength.status", "m2_lsf.status", "m3_flux.status", "m4_sky.status", "m5_stat.status"],
        narrative_md=(
            "## Qué mide A4: las 5 métricas de calidad del cubo (M1–M5)\n\n"
            "A4 no re-reduce: **verifica la calibración** del cubo con 5 métricas, cada una un "
            "aspecto distinto.\n\n"
            "| Métrica | Qué mide | Resultado | Significado |\n|---|---|---|---|\n"
            "| **M1** | Exactitud de la solución de λ (offset vs airglow) | **green** (0.074 Å) | los "
            "λ del cubo están bien (~+3.4 km/s en Hα) |\n"
            "| **M2** | LSF (ancho de la función de dispersión) | **yellow** (2.383 Å @Hα) | "
            "resolución espectral real; el NFM es ~10% más angosto que el nominal → se usa la medida "
            "en E1/E3/G2 |\n"
            "| **M3** | Calibración de flujo absoluto (vs Gaia RP) | **green** (factor 0.973) | la "
            "escala de flujo casa con Gaia a ~3% |\n"
            "| **M4** | Residuo de cielo (la `R` de A2) | **yellow** (R 0.547) | cielo bien restado "
            "(ver A2) |\n"
            "| **M5** | Fiabilidad del STAT (varianza del cubo) | **red** (4.26×) | el STAT subestima "
            "el ruido ~4× → σ **siempre** empírico |\n\n"
            "Las dos decisiones grandes de A4: **M3 cerrado** (flujo validado vs Gaia) y **M5 rojo** "
            "(STAT no sirve → regla *control = objeto*, [`docs/noise_model.md`](../docs/noise_model.md))."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Resumen M1–M5 del `stage00q_qc.json` (estado + cifra de cabecera)."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage00q_qc.json', RUN_ID)\n"
            "m1, m2, m3, m4, m5 = (q['m1_wavelength'], q['m2_lsf'], q['m3_flux'], q['m4_sky'], q['m5_stat'])\n"
            "rows = [\n"
            "    ('M1 λ-solution', m1.get('status'), f\"offset {m1.get('offset_median_A'):.3f} Å (±{m1.get('offset_err_A'):.3f}), {m1.get('n_lines')} líneas\"),\n"
            "    ('M2 LSF',        m2.get('status'), f\"{m2.get('lsf_fwhm_at_halpha_A'):.3f} Å @Hα, dev vs nominal {m2.get('max_dev_vs_nominal_pct'):.1f}%\"),\n"
            "    ('M3 flujo abs',  m3.get('status'), f\"factor {m3.get('flux_factor'):.3f} vs Gaia {m3.get('band')} (growth-curve r={m3.get('plateau_radius_px'):.0f})\"),\n"
            "    ('M4 cielo',      m4.get('status'), f\"R = {m4.get('R')}\"),\n"
            "    ('M5 STAT',       m5.get('status'), f\"factor spaxel {m5.get('factor_spaxel_median')}× (STAT subestima el ruido)\"),\n"
            "]\n"
            "for name, st, detail in rows:\n"
            "    print(f'{name:15s} [{str(st):9s}] {detail}')"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — M2 LSF (medida vs nominal) y M3 growth-curve\n\n"
                    "Ambos desde el QC (baratos, sin cubo). **Izq:** la LSF medida del airglow (azul) "
                    "cae bajo el nominal (gris) → el NFM es más angosto; línea en Hα = 2.383 Å. "
                    "**Der:** el flujo en banda RP crece con el radio hasta el *plateau* (halo AO "
                    "capturado) → `flux_factor = 0.973`."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage00q_qc.json', RUN_ID)\n"
                    "    m2 = q['m2_lsf']; tab = m2['table_A_fwhm']\n"
                    "    w = np.array([r['wave_A'] for r in tab]); f = np.array([r['fwhm_A'] for r in tab])\n"
                    "    nomv = np.array([r['nominal_fwhm_A'] for r in tab])\n"
                    "    m3 = q['m3_flux']; gc = m3['growth_curve']\n"
                    "    gr = np.array([p['radius_px'] for p in gc]); gf = np.array([p['band_flux'] for p in gc])\n\n"
                    "    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.2))\n"
                    "    axL.scatter(w, f, s=10, color='tab:blue', label='LSF medida (airglow)')\n"
                    "    axL.scatter(w, nomv, s=8, color='0.6', label='nominal (código)')\n"
                    "    axL.axvline(6563, color='tab:red', ls=':', label='Hα')\n"
                    "    hal = m2.get('lsf_fwhm_at_halpha_A')\n"
                    "    if hal: axL.axhline(hal, color='tab:red', ls='--', lw=1)\n"
                    "    axL.set_xlabel('λ [Å]'); axL.set_ylabel('FWHM LSF [Å]')\n"
                    "    axL.set_title(f\"M2 · LSF medida vs nominal ({m2['status']}) · @Hα={hal:.3f} Å\")\n"
                    "    axL.legend(fontsize=8)\n"
                    "    axR.plot(gr, gf, 'o-', color='tab:green')\n"
                    "    axR.axvline(m3['plateau_radius_px'], color='0.5', ls='--',\n"
                    "                label=f\"plateau r={m3['plateau_radius_px']:.0f}px\")\n"
                    "    axR.set_xlabel('radio de apertura [px]'); axR.set_ylabel('flujo en banda RP')\n"
                    "    axR.set_title(f\"M3 · growth-curve → factor flujo={m3['flux_factor']:.3f} ({m3['status']})\")\n"
                    "    axR.legend(fontsize=8); fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'a4_qc'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'm2_lsf_m3_growth.png', dpi=110)\n"
                    "    print('figura ->', outdir / 'm2_lsf_m3_growth.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — M5: por qué el STAT no sirve (covarianza del remuestreo)\n\n"
                    "M5 mide que el STAT subestima el ruido **~4.26× por spaxel**. El QC solo guarda "
                    "esa mediana, así que ilustro el **mecanismo** con la inflación espacial de G1 "
                    "(almacenada): al sumar en cajas N×N la varianza real se infla frente a la suma "
                    "ingenua de STAT (que asume píxeles independientes, =1) hasta ~19× en 5×5. Es la "
                    "correlación introducida por el remuestreo del cubo → **σ siempre empírico, "
                    "control = objeto**."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage00q_qc.json', RUN_ID)\n"
                    "    m5 = q['m5_stat']\n"
                    "    g1 = nb.load_qc('stages/stage_g1_qc.json', RUN_ID)['covariance']\n"
                    "    infl = g1['spatial_inflation_by_box']\n"
                    "    boxes = sorted(infl, key=lambda k: int(k))\n"
                    "    xs = [f'{int(b)}×{int(b)}' for b in boxes]; vals = [infl[b] for b in boxes]\n\n"
                    "    fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
                    "    ax.bar(xs, vals, color='tab:orange', alpha=0.85)\n"
                    "    for i, v in enumerate(vals):\n"
                    "        ax.text(i, v + 0.3, f'{v:.1f}×', ha='center', fontsize=9)\n"
                    "    ax.axhline(1.0, color='tab:green', ls='--',\n"
                    "               label='STAT asume =1 (píxeles independientes)')\n"
                    "    ax.set_xlabel('caja de integración (N×N spaxels)')\n"
                    "    ax.set_ylabel('inflación varianza real / suma ingenua')\n"
                    "    ax.set_title(f\"M5 [{m5['status']}]: STAT ~{m5['factor_spaxel_median']}× bajo por spaxel \"\n"
                    "                 f\"(+ covarianza del remuestreo en apertura)\")\n"
                    "    ax.legend(fontsize=8); fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'a4_qc'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'm5_stat_inflation.png', dpi=110)\n"
                    "    print('figura ->', outdir / 'm5_stat_inflation.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**M3 CERRADO (GREEN)**: flujo absoluto validado vs Gaia DR3 RP, factor 0.973 (~3%) tras growth-curve + truncación de cola.", None),
            ("**M5 STAT en ROJO (inherente)**: el STAT subestima el ruido ~4.26× por covarianza del remuestreo → σ SIEMPRE empírico, control=objeto. Limitación aceptada en F1.", "noise_model.md"),
            ("**M2 LSF@Hα = 2.383 Å medido** del airglow (NFM más angosta que el nominal 2.6); usada en E1/E3/G2.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**A4: cubo caracterizado; 2 verdes (M1, M3), 2 amarillos (M2, M4), 1 rojo inherente (M5).**\n\n"
            "- **Fecha:** M1/M2 del airglow SKY_SPECTRUM y M3 vs Gaia, 2026-07-09.\n"
            "- **M1 green:** offset 0.074 Å (solución de λ sana).\n"
            "- **M2 yellow:** LSF 2.383 Å @Hα (NFM más angosto que nominal 2.6); es la LSF usada en "
            "E1/E3/G2.\n"
            "- **M3 green:** flujo absoluto validado vs Gaia DR3 RP, factor 0.973 (~3%), con "
            "growth-curve (halo AO) + truncación de cola.\n"
            "- **M4 yellow:** cielo bien restado (R=0.547, ver A2).\n"
            "- **M5 red (inherente, no defecto):** STAT subestima ~4.26× por la covarianza del "
            "remuestreo → σ SIEMPRE empírico (control=objeto). Limitación aceptada en F1.\n"
            "- **Impacto:** M5 fija la regla de ruido de toda la cadena (D2/E1/E3); M3 sostiene el "
            "flujo absoluto de E3."
        ),
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
        salient=["centering_method", "spatial_shift_mode", "crop", "n_cubes", "covariance_factor_box3", "finite_fraction"],
        narrative_md=(
            "## Qué hace B1 y por qué importa\n\n"
            "B1 **carga** el cubo reducido, lo **centra** en la estrella primaria y lo **recorta** a "
            "una ventana de 170 px → el cubo de trabajo para toda la extracción aguas abajo.\n\n"
            "Es una **migración por equivalencia**: la lógica venía del notebook histórico "
            "`01_load_align_crop.ipynb` y se movió a `musepipe/stages/stage01_align.py` reproduciendo "
            "los productos numéricamente, y solo después se extendió (propagar STAT, registrar "
            "shifts, `entry_point`).\n\n"
            "**En este run:** `n_cubes = 1` — la alineación entre exposiciones ya se hizo en A1 "
            "(plan B, OFFSET_LIST manual), así que el **shift de B1 es 0**; B1 solo centra y recorta "
            "el cubo combinado.\n\n"
            "**El nexo con M5 (ruido):** B1 propaga la extensión STAT por las mismas "
            "transformaciones (kernel `bilinear_kernel_squared`) y registra el **factor de "
            "covarianza espacial box3 ≈ 6.4×**. Ese es el remuestreo del cubo que correlaciona el "
            "ruido — la raíz física de que el STAT subestime (M5) y de la inflación de apertura (G1). "
            "El shift subpixel (spline orden 3) de la alineación es el origen cuantificado en "
            "[`docs/noise_model.md`](../docs/noise_model.md) §4."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Campos clave del `stage01_qc.json`: centrado, shift, crop, STAT propagado y covarianza."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage01_qc.json', RUN_ID)\n"
            "c = q['crop']\n"
            "print('entrada:', q['inputs'][0]['file'])\n"
            "print(f\"n_cubes = {q['n_cubes']}  (1 = cubo ya combinado en A1; alineación inter-exp = plan B)\")\n"
            "print(f\"centrado: {q['centering_method']} (perfil {q['profile']}, fallback={q['shifts'][0].get('fallback_used')})\")\n"
            "print(f\"centro (y,x) = ({c['center_yx'][0]:.2f}, {c['center_yx'][1]:.2f})  [{c['center_source']}]\")\n"
            "sh = q['spatial_shifts'][0]\n"
            "print(f\"shift subpixel (y,x) = ({sh['shift_y']}, {sh['shift_x']})  modo={q['spatial_shift_mode']}\")\n"
            "print(f\"crop = {c['npix']}px  ->  cube_shape {q['cube_shape']}\")\n"
            "print(f\"fracción finita = {q['finite_fraction_per_cube'][0]:.3f}\")\n"
            "st = q['stat']\n"
            "print(f\"STAT propagado: {st['propagated']} (kernel {st['interp_kernel']}); \"\n"
            "      f\"covarianza box3 = {st['covariance_factor_box3']:.2f}×  <-- nexo con M5/G1\")"
        ),
        plot_md=(
            "## Plot — centrado + crop sobre el campo completo\n\n"
            "**FITS usado:** `cube_telcorr.fits` (entrada de B1), luz-blanca. Marca el **centro de la "
            "estrella** (rojo) y las dos cajas: el **crop final de 170 px** (cyan) y el crop inicial "
            "de 80 px (naranja). El halo AO de la primaria domina el campo — por eso el crop de 170 "
            "px conserva el halo hasta la posición del compañero."
        ),
        plot_code=(
            "MAKE_PLOT = True   # carga el cubo (~3.3 GB) vía astropy; requiere kernel MUSE\n"
            "if MAKE_PLOT:\n"
            "    try:\n"
            "        import numpy as np\n"
            "        import matplotlib.pyplot as plt\n"
            "        from matplotlib.patches import Rectangle\n"
            "        from astropy.io import fits\n\n"
            "        q = nb.load_qc('stages/stage01_qc.json', RUN_ID)\n"
            "        cy, cx = q['crop']['center_yx']\n"
            "        cb = q['crop_bounds_per_cube'][0]; ib = q['initial_crop_bounds_per_cube'][0]\n"
            "        cube_path = q['inputs'][0]['file']\n"
            "        print('FITS usado:', cube_path)\n\n"
            "        h = fits.open(cube_path, memmap=True)\n"
            "        wl = np.nanmedian(np.asarray(h[1].data, dtype=np.float32), axis=0); h.close()\n\n"
            "        fig, ax = plt.subplots(figsize=(6.2, 6))\n"
            "        ax.imshow(np.log10(np.clip(wl, 1, None)), origin='lower', cmap='gray')\n"
            "        ax.plot(cx, cy, '+', color='tab:red', ms=14, mew=2,\n"
            "                label=f'centro estrella ({cx:.1f},{cy:.1f})')\n"
            "        ax.add_patch(Rectangle((cb['x1'], cb['y1']), cb['x2'] - cb['x1'], cb['y2'] - cb['y1'],\n"
            "                     fill=False, ec='tab:cyan', lw=2, label=f\"crop {q['crop_npix']}px\"))\n"
            "        ax.add_patch(Rectangle((ib['x1'], ib['y1']), ib['x2'] - ib['x1'], ib['y2'] - ib['y1'],\n"
            "                     fill=False, ec='tab:orange', lw=1.2, ls='--', label='crop inicial 80px'))\n"
            "        ax.set_title('B1 · centrado + crop sobre luz-blanca (log)')\n"
            "        ax.legend(fontsize=8, loc='upper right'); ax.axis('off'); fig.tight_layout()\n"
            "        outdir = nb.run_dir(RUN_ID) / 'plots' / 'b1_align'; outdir.mkdir(parents=True, exist_ok=True)\n"
            "        fig.savefig(outdir / 'crop.png', dpi=110); print('figura ->', outdir / 'crop.png'); plt.show()\n"
            "    except Exception as e:\n"
            "        print('No se pudo generar el plot:', type(e).__name__, e)\n"
            "        print('Necesita el kernel MUSE (astropy) y el cubo en disco.')"
        ),
        decisions=[
            ("**Migración por equivalencia (B1a):** `run_stage01()` reproduce numéricamente el baseline del notebook histórico; STAT propagado y shifts registrados en QC (B1b).", None),
            ("**Centrado `maoppy_refined`** (centroide de la PSF física AO, fallback usado); crop de 170 px alrededor del centro medido para conservar el halo hasta el compañero.", None),
            ("**Covarianza box3 ≈ 6.4×** registrada al propagar STAT: es el remuestreo que correlaciona el ruido → raíz física de M5 (STAT subestima) y de la inflación de apertura (G1).", "noise_model.md"),
            ("**`n_cubes = 1`:** la alineación entre exposiciones ya se hizo en A1 (plan B); el shift de B1 es 0 → B1 no añade remuestreo nuevo, solo centra/recorta.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**B1: cubo centrado en (166.0, 167.8) y recortado a 170 px; STAT propagado; sin "
            "open_issues.**\n\n"
            "- **Fecha:** run realineado (cubo `cube_telcorr.fits`, 2026-07-08).\n"
            "- **Entrada:** `cube_telcorr.fits` (realineado); **salida:** `stage01` stack 170×170 + "
            "`stage01_qc.json`.\n"
            "- **Centrado:** `maoppy_refined`, centro medido (166.0, 167.8); **shift = 0** "
            "(`n_cubes = 1`, alineación ya en A1).\n"
            "- **Fracción finita:** 0.94; **crop:** 170 px (crop inicial 80 px con guarda de 12 px).\n"
            "- **STAT:** propagado con kernel `bilinear_kernel_squared`; **covarianza box3 = 6.38×** "
            "→ el nexo cuantificado con M5/G1 y la regla de σ empírico.\n"
            "- **Downstream:** el cubo de B1 alimenta B2, B3 y C1."
        ),
    ),
    dict(
        id="B2", slug="B2_xcorr_stripes", title="Xcorr / franjas", block="B · Preparación",
        spec="spec_B2_codex_xcorr_stripes.md", run_override=None,
        what="Correlación cruzada entre exposiciones y detección/corrección de franjas.",
        inputs="Cubo alineado", outputs="`stages/stage02_qc.json` (+ `stage02_xcorr_qc.json`)",
        downstream="C1, 04b",
        exec=dict(kind="script", target="stage02_xcorr.sh", cost="Ligero."),
        qc="stages/stage02_xcorr_qc.json",
        salient=["status", "reduction_factor", "dirty_channels", "finite_fraction", "mean_shift"],
        narrative_md=(
            "## Qué hace B2 y por qué\n\n"
            "B2 hace dos cosas sobre el cubo alineado: **correlación cruzada** entre exposiciones "
            "(para un shift espectral relativo) y **detección/corrección de franjas** (*stripes*): un "
            "patrón aditivo coherente del slicer/detector IFU que puede dejar una modulación en el "
            "campo. La maquinaria vive en `musepipe/stripes.py` (diagnosticada en 02b/02c); B2 añade "
            "una **métrica escalar de franjas al QC**, propaga STAT por la corrección, y migra el "
            "driver — no reimplementa la corrección.\n\n"
            "**En este run:** cubo de **exposición única** (`cube_shape[0]=1`) → no hay shift "
            "inter-exposición que corregir, y la métrica de franjas sale **plana** "
            "(amplitud mediana ~0.977, `reduction_factor = 1.0`, **0 canales sucios**) → **no se "
            "aplica corrección de franjas**. Status **green**.\n\n"
            "**Para qué sirve aguas abajo:** E2 usa la métrica de franjas para comprobar que "
            "cualquier 'señal' de Hα no coincida con un canal de franja residual; D1 compara métodos "
            "en canales limpios vs sucios. Aquí, 0 canales sucios → todo limpio. La ventana del "
            "láser AO (5780–6050 Å) se excluye del resumen."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Métrica de franjas + shifts del `stage02_xcorr_qc.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage02_xcorr_qc.json', RUN_ID)\n"
            "sm = q['stripe_metric']\n"
            "print('status:', q['status'])\n"
            "print(f\"n_cubes (cube_shape[0]) = {q['cube_shape'][0]}   apply_mode = {q['apply_mode']}\")\n"
            "print(f\"shift medio/std por canal = {q['mean_shift_per_cube_ch'][0]} / {q['std_shift_per_cube_ch'][0]}\")\n"
            "print(f\"franjas: amp mediana pre={sm['amp_pre_median']:.3f} post={sm['amp_post_median']:.3f}  \"\n"
            "      f\"reduction_factor={sm['reduction_factor']:.2f}\")\n"
            "print(f\"canales sucios = {len(sm['dirty_channels'])}   fracción finita = {q['finite_fraction_per_cube'][0]:.3f}\")\n"
            "st = q['stat']\n"
            "print(f\"STAT: present={st['present']} shift_applied={st['shift_applied']} kernel={st['interp_kernel']}\")\n"
            "print()\n"
            "print('provenance:', q['provenance_note'])"
        ),
        plot_md=(
            "## Plot — amplitud del patrón de franjas vs λ\n\n"
            "**Tabla usada:** `tables/stage02_stripe_metric.csv` (amplitud por canal, pre y post; "
            "aquí **pre ≡ post** porque no se corrigió nada). Archivo pequeño — no hace falta el "
            "cubo. La amplitud se mantiene **plana ~0.977** en todo λ (sin patrón coherente de "
            "franjas); la ventana del láser AO queda excluida (gris)."
        ),
        plot_code=(
            "MAKE_PLOT = True   # tabla pequeña; requiere kernel MUSE (pandas/matplotlib)\n"
            "if MAKE_PLOT:\n"
            "    try:\n"
            "        import pandas as pd\n"
            "        import matplotlib.pyplot as plt\n"
            "        q = nb.load_qc('stages/stage02_xcorr_qc.json', RUN_ID)\n"
            "        sm = q['stripe_metric']\n"
            "        d = pd.read_csv(nb.run_dir(RUN_ID) / 'tables' / 'stage02_stripe_metric.csv')\n"
            "        w = d['wavelength_A'].values; a = d['amplitude_pre'].values\n\n"
            "        fig, ax = plt.subplots(figsize=(11, 4))\n"
            "        ax.plot(w, a, lw=0.4, color='0.4', label='amplitud de franjas (pre ≡ post)')\n"
            "        ax.axhline(sm['amp_pre_median'], color='tab:blue', ls='--', lw=1,\n"
            "                   label=f\"mediana={sm['amp_pre_median']:.3f}\")\n"
            "        for i, (x0, x1) in enumerate(sm.get('mask_excluded_windows_A', [])):\n"
            "            ax.axvspan(x0, x1, color='0.7', alpha=0.4, label='excluido (láser AO)' if i == 0 else None)\n"
            "        ax.set_xlabel('λ [Å]'); ax.set_ylabel('amplitud del patrón de franjas'); ax.set_ylim(0, 1.6)\n"
            "        ax.set_title(f\"B2 · sin franjas: reduction_factor={sm['reduction_factor']:.2f}, \"\n"
            "                     f\"dirty_channels={len(sm['dirty_channels'])} (status {q['status']})\")\n"
            "        ax.legend(fontsize=8); fig.tight_layout()\n"
            "        outdir = nb.run_dir(RUN_ID) / 'plots' / 'b2_stripes'; outdir.mkdir(parents=True, exist_ok=True)\n"
            "        fig.savefig(outdir / 'stripe_amplitude.png', dpi=110)\n"
            "        print('figura ->', outdir / 'stripe_amplitude.png'); plt.show()\n"
            "    except Exception as e:\n"
            "        print('No se pudo generar el plot:', type(e).__name__, e)"
        ),
        decisions=[
            ("**Exposición única → sin corrección de franjas:** `reduction_factor = 1.0`, 0 canales sucios, shifts = 0; los chequeos de franjas/equivalencia se satisfacen trivialmente.", None),
            ("Maquinaria de franjas centralizada en `musepipe/stripes.py` (diagnóstico en 02b/02c); B2 solo añade la **métrica escalar** (para E2/D1), propaga STAT y migra el driver.", None),
            ("Ventana del **láser AO (5780–6050 Å) excluida** del resumen de franjas.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**B2: sin franjas que corregir; status `green`.**\n\n"
            "- **Fecha:** run realineado (cubo 2026-07-08; QC 2026-07-08).\n"
            "- **Datos:** cubo de exposición única; amplitud de franjas mediana **0.977**, "
            "`reduction_factor = 1.0`, **0 canales sucios**, fracción finita 0.94.\n"
            "- **Shifts:** medio/std por canal = 0/0 (nada que alinear entre exposiciones).\n"
            "- **STAT:** presente, `shift_applied = True`, kernel `none` (sin shift fraccional → "
            "sin cambio de varianza).\n"
            "- **Downstream:** E2 y D1 consumen la métrica de franjas (0 sucios → limpio); el cubo "
            "alimenta C1 y 04b.\n"
            "- **Caveat:** al ser exposición única, los chequeos de franjas/equivalencia son "
            "trivialmente satisfechos (open_issue documentado)."
        ),
    ),
    dict(
        id="B3", slug="B3_localize", title="Localización del compañero", block="B · Preparación",
        spec="spec_B3_codex_target_localization.md", run_override=None,
        what="Localiza el compañero ROXs 12 B en el campo.",
        inputs="Cubo alineado", outputs="`stages/stage01c_qc.json`",
        downstream="C1–C4 (posición de extracción)",
        exec=dict(kind="script", target="stage01c_localize.sh", cost="Ligero."),
        qc="stages/stage01c_qc.json",
        salient=["companion.snr_detection", "sep_arcsec", "pa_deg", "band_used_A", "chromatic_centroid_needed"],
        narrative_md=(
            "## Qué hace B3 y por qué\n\n"
            "B3 **mide** las posiciones de la primaria y del compañero en el cubo de trabajo y las "
            "valida contra la astrometría publicada (Bowler+2017: sep ~1.78″, PA ~240°), sustituyendo "
            "las coordenadas *hardcodeadas* por coordenadas medidas. Escribe las coordenadas "
            "**canónicas** que TODAS las etapas siguientes (C1–C4, E4) leerán del QC — nunca más un "
            "número a mano.\n\n"
            "**Decisiones clave:**\n"
            "- **Banda de detección al ROJO (8800–9350 Å):** el compañero es un objeto subestelar "
            "**frío** — solo es detectable en el rojo. Colapsar ahí lo hace visible (SNR ~12).\n"
            "- **Deriva cromática del centroide:** el centroide del compañero se desplaza ~3.56 px "
            "pico-a-pico con λ (>0.5 px umbral) → `chromatic_centroid_needed = True`; la extracción "
            "aguas abajo debe seguir el centroide dependiente de λ.\n"
            "- La posición medida está a **5.23 px** de la vieja coordenada hardcodeada (72,152) → "
            "productos históricos pudieron usar la posición equivocada (open_issue).\n\n"
            "El resultado (sep/PA que casan con Bowler+2017) confirma un **compañero real ligado**, "
            "insumo directo de la clasificación G4."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Posiciones, astrometría vs literatura y deriva cromática del `stage01c_qc.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage01c_qc.json', RUN_ID)\n"
            "pr, cp, ast = q['primary'], q['companion'], q['astrometry']\n"
            "print(f\"primaria  (y,x) = ({pr['pos_yx'][0]:.2f}, {pr['pos_yx'][1]:.2f})  ± {pr['err_px']:.2f} px\")\n"
            "print(f\"compañero (y,x) = ({cp['pos_yx'][0]:.2f}, {cp['pos_yx'][1]:.2f})  ± {cp['err_px']:.2f} px  |  SNR = {cp['snr_detection']:.1f}\")\n"
            "print(f\"banda de detección = {cp['band_used_A']} Å (rojo: compañero frío)\")\n"
            "print()\n"
            "print(f\"separación = {ast['sep_arcsec']:.3f}\\\" (esperado {ast['expected_sep_arcsec']}\\\", {ast['sep_deviation_sigma']:+.2f}σ)\")\n"
            "print(f\"PA         = {ast['pa_deg']:.2f}° (esperado {ast['expected_pa_deg']}°, {ast['pa_deviation_sigma']:+.2f}σ)\")\n"
            "print()\n"
            "ch = q['chromatic']\n"
            "print(f\"deriva cromática (compañero) = {ch['companion_drift_px_peak_to_peak']:.2f} px pico-a-pico \"\n"
            "      f\"(umbral {ch['threshold_px']}) -> chromatic_centroid_needed = {ch['chromatic_centroid_needed']}\")\n"
            "print(f\"legacy hardcoded (x,y) = {q['legacy_check']['hardcoded_companion_xy']}  ->  medido a {q['legacy_check']['distance_px']:.2f} px\")"
        ),
        plot_md=(
            "## Plot — detección en la banda roja (primaria + compañero)\n\n"
            "**FITS usado:** `stage02_xcorr_cube_stack.fits` (cubo de trabajo 170×170), colapsado en "
            "la banda de detección (8800–9350 Å). Marca la **primaria** (cyan), el **compañero** "
            "(círculo verde) y la **posición predicha** de Bowler+2017 (× blanca) — que cae sobre el "
            "compañero medido. El título lleva sep/PA medidos vs esperados."
        ),
        plot_code=(
            "MAKE_PLOT = True   # cubo de trabajo (~0.4 GB); requiere kernel MUSE\n"
            "if MAKE_PLOT:\n"
            "    try:\n"
            "        import numpy as np\n"
            "        import matplotlib.pyplot as plt\n"
            "        from astropy.io import fits\n\n"
            "        q = nb.load_qc('stages/stage01c_qc.json', RUN_ID)\n"
            "        cp, pr, ast = q['companion'], q['primary'], q['astrometry']\n"
            "        cube_path = q['input_cube']['file']\n"
            "        print('FITS usado:', cube_path)\n"
            "        h = fits.open(cube_path, memmap=True); data = h[1].data\n"
            "        data = np.asarray(data[0] if data.ndim == 4 else data, dtype=np.float32)\n"
            "        hd = h[1].header; n3 = data.shape[0]\n"
            "        wave = hd.get('CRVAL3', 4749.533) + (np.arange(n3) - (hd.get('CRPIX3', 1.0) - 1)) * hd.get('CD3_3', 1.25)\n"
            "        lo, hi = cp['band_used_A']; sel = (wave >= lo) & (wave <= hi)\n"
            "        img = np.nanmedian(data[sel], axis=0); h.close()\n\n"
            "        py, px = pr['pos_yx']; cy, cx = cp['pos_yx']; ppy, ppx = cp['predicted_pos_yx']\n"
            "        v = np.nanpercentile(img, [5, 99.5])\n"
            "        fig, ax = plt.subplots(figsize=(6.4, 6))\n"
            "        ax.imshow(img, origin='lower', cmap='magma', vmin=v[0], vmax=v[1])\n"
            "        ax.plot(px, py, '+', color='cyan', ms=14, mew=2, label=f'primaria ({px:.1f},{py:.1f})')\n"
            "        ax.plot(cx, cy, 'o', mfc='none', mec='lime', ms=16, mew=2, label=f\"compañero SNR={cp['snr_detection']:.1f}\")\n"
            "        ax.plot(ppx, ppy, 'x', color='white', ms=9, mew=1.5, label='predicho (Bowler+2017)')\n"
            "        ax.set_title(f\"B3 · banda {lo:.0f}-{hi:.0f} Å · sep={ast['sep_arcsec']:.3f}\\\" \"\n"
            "                     f\"(esp {ast['expected_sep_arcsec']}) PA={ast['pa_deg']:.1f}° ({ast['pa_deviation_sigma']:+.2f}σ)\")\n"
            "        ax.legend(fontsize=8, loc='upper right'); ax.axis('off'); fig.tight_layout()\n"
            "        outdir = nb.run_dir(RUN_ID) / 'plots' / 'b3_localize'; outdir.mkdir(parents=True, exist_ok=True)\n"
            "        fig.savefig(outdir / 'detection.png', dpi=110); print('figura ->', outdir / 'detection.png'); plt.show()\n"
            "    except Exception as e:\n"
            "        print('No se pudo generar el plot:', type(e).__name__, e)"
        ),
        decisions=[
            ("**Banda de detección al rojo (8800–9350 Å):** el compañero es una enana fría, solo detectable en el rojo (SNR ~12).", None),
            ("**Astrometría casa con Bowler+2017 a <0.1σ** (sep 1.802″ vs 1.81, PA 240.1° vs 240) → compañero real ligado (insumo de G4).", None),
            ("**Deriva cromática del centroide ~3.56 px** (>0.5 umbral) → `chromatic_centroid_needed`; C1–C4 deben seguir el centroide λ-dependiente.", None),
            ("Coordenadas **canónicas desde QC** reemplazan la vieja hardcoded (72,152), que estaba a 5.23 px (productos históricos pudieron usar la posición equivocada).", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**B3: compañero localizado en (y,x)=(155.6, 75.8), sep 1.802″ / PA 240.1°, casando con "
            "Bowler+2017 a <0.1σ; SNR 12.3.**\n\n"
            "- **Fecha:** run realineado (cubo 2026-07-08).\n"
            "- **Entrada:** `stage02_xcorr_cube_stack.fits`; **salida:** `stage01c_qc.json` + coords "
            "canónicas para C1–C4/E4.\n"
            "- **Banda de detección:** 8800–9350 Å (rojo, compañero frío); primaria en (84.9, 84.6).\n"
            "- **Deriva cromática:** 3.56 px pico-a-pico → `chromatic_centroid_needed = True`.\n"
            "- **Legacy:** posición medida a 5.23 px de la hardcoded (72,152) — de ahí en adelante se "
            "usan las coords medidas.\n"
            "- **Downstream:** C1–C4 y E4 leen la posición del compañero desde este QC."
        ),
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
        salient=["form_chosen", "reason", "ring_residual_pct_median", "bins_above_5pct", "n_ok_bins"],
        narrative_md=(
            "## Qué hace C1 y la decisión Moffat vs Psfao\n\n"
            "C1 ajusta la **PSF cromática de la primaria** (por bin de λ) para modelar el halo "
            "estelar y poder tenerlo en cuenta en la posición del compañero. Ajusta **dos formas** "
            "por bin — un doble **Moffat** y un **Psfao** físico (`maoppy.Psfao`, PSF de óptica "
            "adaptiva NFM) — y **selecciona** por una métrica canónica: el residuo en el **anillo "
            "del compañero** (radio ≈ 71 px, ancho 3 px).\n\n"
            "**Por qué Psfao:** el halo AO de NFM es ancho y un Moffat no lo ajusta bien en el rojo; "
            "Psfao es el modelo físico del halo AO.\n\n"
            "**Decisión (blocker #8 cerrado):** selección automática por residuo mediano del anillo → "
            "**Psfao 4.44 % < Moffat 15.65 %** → se elige **Psfao**. El **híbrido** azimutal se probó "
            "y se **descartó** (con Psfao empeoraba: la escala venía de la FWHM inflada del Moffat).\n\n"
            "**El nexo con B6/D1 (importante):** aun con Psfao, el **p90** del residuo del anillo es "
            "~31 % (17 bins por encima del 5 %) — los bins del **rojo lejano** no cierran. Ese es el "
            "sistemático **B6** que reaparece en D1 como `divergent_continuum` y se acepta como "
            "presupuestado ([`docs/d1_canonical_method_decision.md`](../docs/d1_canonical_method_decision.md)). "
            "La PSF física **no** lo elimina — es el piso a esta geometría.\n\n"
            "**Producto:** `psf_model.json` con los parámetros Psfao (`r0`, `beta`, …) por bin, "
            "suavizados con un polinomio; lo consumen C3/C4/E4."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Comparación de modelos, métrica del anillo y ajuste del `stage_e01_qc.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage_e01_qc.json', RUN_ID)\n"
            "mc = q['model_comparison']; rm = q['companion_ring_metric']; fit = q['fit']\n"
            "print('forma elegida:', mc['form_chosen'], '|', mc['reason'])\n"
            "print(f\"  Moffat: mediana {mc['moffat']['ring_residual_pct_median']:.2f}%  p90 {mc['moffat']['ring_residual_pct_p90']:.1f}%  ({mc['moffat']['n_bins']} bins)\")\n"
            "print(f\"  Psfao : mediana {mc['psfao']['ring_residual_pct_median']:.2f}%  p90 {mc['psfao']['ring_residual_pct_p90']:.1f}%  ({mc['psfao']['n_bins']} bins)\")\n"
            "print()\n"
            "print(f\"anillo del compañero: radio {rm['radius_px']:.1f} px, ancho {rm['width_px']:.0f} px; \"\n"
            "      f\"residuo mediano {rm['residual_pct_median']:.2f}% (p90 {rm['residual_pct_p90']:.1f}%), \"\n"
            "      f\"bins >5% = {rm['bins_above_5pct']}  <-- B6 rojo lejano\")\n"
            "print(f\"ajuste: {fit['model']} (form={fit['form_chosen']}), fit_radius {fit['fit_radius_px']:.0f} px, \"\n"
            "      f\"{fit['n_ok_bins']} bins OK / {fit['n_bins_rejected']} rechazados\")\n"
            "print(f\"híbrido aplicado: {q['hybrid']['applied']}\")"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — la decisión: Moffat vs Psfao (residuo del anillo)\n\n"
                    "Del `stage_e01_qc.json` (barato). Psfao baja la **mediana** del residuo del anillo "
                    "por debajo del objetivo 5 %; Moffat no. El **p90** de ambos sigue alto (~31–34 %) "
                    "= el residuo B6 del rojo lejano que ni Psfao cierra."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_e01_qc.json', RUN_ID); mc = q['model_comparison']\n"
                    "    med = [mc['moffat']['ring_residual_pct_median'], mc['psfao']['ring_residual_pct_median']]\n"
                    "    p90 = [mc['moffat']['ring_residual_pct_p90'], mc['psfao']['ring_residual_pct_p90']]\n"
                    "    x = np.arange(2); cols = ['tab:red', 'tab:green']\n"
                    "    fig, ax = plt.subplots(figsize=(6.4, 4.2))\n"
                    "    ax.bar(x - 0.18, med, 0.36, color=cols, label='mediana')\n"
                    "    ax.bar(x + 0.18, p90, 0.36, color=cols, alpha=0.45, label='p90')\n"
                    "    ax.axhline(5, color='k', ls='--', lw=1, label='objetivo <5%')\n"
                    "    for i, v in enumerate(med): ax.text(i - 0.18, v + 0.6, f'{v:.1f}%', ha='center', fontsize=9)\n"
                    "    ax.set_xticks(x); ax.set_xticklabels(['Moffat', 'Psfao'])\n"
                    "    ax.set_ylabel('residuo del anillo del compañero [%]')\n"
                    "    ax.set_title(f\"C1 · selección: {mc['form_chosen'].upper()} (mediana {med[1]:.1f}% < {med[0]:.1f}%)\")\n"
                    "    ax.legend(fontsize=8); fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'c1_psf'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'moffat_vs_psfao.png', dpi=110)\n"
                    "    print('figura ->', outdir / 'moffat_vs_psfao.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — la PSF cromática: r0 (Fried) vs λ\n\n"
                    "Del `psf_model.json` `param_table` (barato). El parámetro de Fried `r0` **sube "
                    "con λ** (mejor seeing en el rojo, ~r0 ∝ λ^1.2) — por eso la PSF es *cromática* y "
                    "se ajusta por bin. Es el modelo que C3/C4/E4 consumen."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    p = nb.load_qc('stages/psf_model.json', RUN_ID); pt = p['param_table']\n"
                    "    lam = np.array(pt['lambda_A']); r0 = np.array(pt['r0'], dtype=float)\n"
                    "    fig, ax = plt.subplots(figsize=(9, 4.2))\n"
                    "    ax.plot(lam, r0, 'o-', color='tab:blue', ms=4)\n"
                    "    ax.set_xlabel('λ [Å]'); ax.set_ylabel('r0 (parámetro de Fried) [m]')\n"
                    "    ax.set_title(f\"C1 · PSF cromática {p['form'].upper()} ({p['system']}): r0 sube al rojo (mejor seeing)\")\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'c1_psf'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'chromatic_r0.png', dpi=110)\n"
                    "    print('figura ->', outdir / 'chromatic_r0.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**Blocker #8 CERRADO**: C1 ajusta Moffat **y** Psfao por bin y selecciona por menor residuo del anillo (empate→Moffat).", "d1_canonical_method_decision.md"),
            ("**Forma elegida = psfao** (residuo de anillo mediano 4.44% vs 15.65% Moffat). `psf_model.json` byte-idéntico a la etapa lateral: consolidación neutra.", None),
            ("Híbrido azimutal **descartado**: con Psfao empeoraba (la escala venía de la FWHM inflada del Moffat).", None),
            ("**Residuo B6 del rojo lejano NO se cierra** (p90 ~31%, 17 bins >5%): es el piso a esta geometría → sistemática presupuestada que reaparece en D1 (`divergent_continuum`).", "d1_canonical_method_decision.md"),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**C1: PSF cromática ajustada; forma = Psfao (auto); residuo del anillo mediano 4.44% "
            "(<5% objetivo).**\n\n"
            "- **Fecha:** consolidación WP-5, 2026-07-10 (commit `c3704b8`).\n"
            "- **Ajuste:** `maoppy.Psfao` (muse_nfm), fit_radius 78 px, 43 bins OK / 3 rechazados; "
            "anillo del compañero radio 71 px.\n"
            "- **Selección:** Psfao 4.44% < Moffat 15.65% (mediana del residuo del anillo); híbrido "
            "descartado.\n"
            "- **B6 abierto/aceptado:** p90 ~31% (17 bins >5%) en el rojo lejano — la PSF física no "
            "lo cierra; sistemática presupuestada (D1 `divergent_continuum`).\n"
            "- **Consolidación neutra:** `psf_model.json` byte-idéntico al de la etapa lateral previa.\n"
            "- **Downstream:** `psf_model.json` (params Psfao por bin) lo consumen C3, C4 y E4."
        ),
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
        salient=["method", "local_model_kind", "fit_radius_px", "target_yx", "bad_channel_count"],
        narrative_md=(
            "## Qué hace 04b y por qué\n\n"
            "04b resta una **superficie local** (un **plano**) ajustada al fondo en un **anillo** "
            "alrededor del compañero (máscara 3 px en el núcleo, ajuste hasta 12 px), **canal por "
            "canal**. Así elimina el **gradiente del halo estelar** en la posición del compañero → "
            "un fondo local limpio para extraer su espectro sin el pedestal del halo.\n\n"
            "Trabaja en modo `native_stage02`: directamente sobre el cubo alineado de B2 (la rama de "
            "**sustracción local**, no la de PCA/Stage04). Ajusta el plano con **sigma-clip** "
            "(3σ, 3 iteraciones, mínimo 30 px) enmascarando el núcleo del compañero (3 px) y otros "
            "objetos (la estrella) dentro de 3 px.\n\n"
            "También calcula la **máscara de longitud de onda** buena/mala (ventana del láser AO "
            "5780–6050 Å, 216 canales) y marca los canales de Hα/Hβ y las ventanas de continuo para "
            "aguas abajo.\n\n"
            "**Rol en la cadena:** produce `cube_residual_local_object.fits`, el cubo del objeto con "
            "el fondo local restado; es la base de la rama de extracción **local-surface** "
            "(`optimal_ls`) y de la cadena de objetos lejanos (06/07/08). *(Ojo: D1 documenta que "
            "`optimal_ls` arrastraba este pedestal — por eso el canónico es psffit, no optimal_ls.)*"
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Parámetros del ajuste local y productos del `stage04b_qc.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage04b_qc.json', RUN_ID)\n"
            "print('método:', q['method'], '| input_mode:', q['input_mode'], '| modelo:', q['local_model_kind'])\n"
            "print(f\"objetivo: '{q['target_object']}' en (y,x)={q['target_yx']}\")\n"
            "print(f\"ajuste: anillo máscara {q['mask_radius_px']:.0f}px .. fit {q['fit_radius_px']:.0f}px; \"\n"
            "      f\"sigma-clip {q['local_fit_sigma_clip']} x{q['local_fit_max_iter']}, min {q['local_fit_min_pixels']}px; \"\n"
            "      f\"nfit mediana {q['nfit_median_per_cube'][0]:.0f}px\")\n"
            "print(f\"máscara otros objetos: {q['mask_other_objects']} (r={q['other_mask_radius_px']:.0f}px)\")\n"
            "print(f\"canales malos (láser AO {q['bad_wavelength_ranges_A']}): {q['bad_channel_count']}\")\n"
            "print(f\"Hα en {q['halpha_channels_A']} Å  |  Hβ en {q['hbeta_channels_A']} Å\")\n"
            "print(f\"salida: {q['object_cube_fits'].split('/')[-1]}  (elapsed {q['elapsed_s']:.1f}s)\")"
        ),
        plot_md=(
            "## Plot — fondo local antes/después (stamp del compañero)\n\n"
            "**FITS usados:** el cubo de entrada (`stage02_xcorr_cube_stack.fits`) y el de salida "
            "(`cube_residual_local_object.fits`), colapsados en la banda de continuo de Hα "
            "(6570–6700 Å) en un recorte alrededor del compañero. **Izq:** el gradiente del halo "
            "estelar. **Der:** dentro del anillo de ajuste (cyan, 12 px) el fondo queda **plano** "
            "(plano local restado); el núcleo del compañero (rojo, 3 px) se enmascara del ajuste."
        ),
        plot_code=(
            "MAKE_PLOT = True   # dos cubos (~0.4 GB c/u); requiere kernel MUSE\n"
            "if MAKE_PLOT:\n"
            "    try:\n"
            "        import numpy as np\n"
            "        import matplotlib.pyplot as plt\n"
            "        from matplotlib.patches import Circle\n"
            "        from astropy.io import fits\n\n"
            "        q = nb.load_qc('stages/stage04b_qc.json', RUN_ID)\n"
            "        ty, tx = q['target_yx']; fr = q['fit_radius_px']; mr = q['mask_radius_px']\n"
            "        wave = 4749.533203125 + 1.25 * np.arange(3681); sel = (wave >= 6570) & (wave <= 6700)\n"
            "        def band_img(path):\n"
            "            h = fits.open(path, memmap=True)\n"
            "            hd = next(x for x in h if x.data is not None and np.asarray(x.data).ndim >= 3)\n"
            "            d = np.asarray(hd.data, dtype=np.float32); d = d[0] if d.ndim == 4 else d\n"
            "            img = np.nanmedian(d[sel], axis=0); h.close(); return img\n"
            "        S = 22\n"
            "        imgi = band_img(q['input_cube_fits']); imgo = band_img(q['object_cube_fits'])\n"
            "        sub = lambda im: im[ty - S:ty + S, tx - S:tx + S]\n"
            "        ci, co = sub(imgi), sub(imgo)\n"
            "        v = np.nanpercentile(ci, [5, 99])\n"
            "        fig, axes = plt.subplots(1, 2, figsize=(11, 5))\n"
            "        for ax, im, title in [(axes[0], ci, 'entrada (stage02): gradiente del halo'),\n"
            "                              (axes[1], co, 'salida 04b: fondo local restado')]:\n"
            "            ax.imshow(im, origin='lower', cmap='viridis', vmin=v[0], vmax=v[1])\n"
            "            ax.add_patch(Circle((S, S), mr, fill=False, ec='red', lw=1.5))\n"
            "            ax.add_patch(Circle((S, S), fr, fill=False, ec='cyan', lw=1.5, ls='--'))\n"
            "            ax.set_title(title); ax.axis('off')\n"
            "        axes[0].plot([], [], color='red', label=f'máscara {mr:.0f}px')\n"
            "        axes[0].plot([], [], color='cyan', ls='--', label=f'anillo ajuste {fr:.0f}px')\n"
            "        axes[0].legend(fontsize=8, loc='upper right')\n"
            "        fig.suptitle(f'04b · plano local (anillo {mr:.0f}-{fr:.0f}px) alrededor del compañero · continuo Hα')\n"
            "        fig.tight_layout()\n"
            "        outdir = nb.run_dir(RUN_ID) / 'plots' / 'c_04b'; outdir.mkdir(parents=True, exist_ok=True)\n"
            "        fig.savefig(outdir / 'local_surface.png', dpi=110)\n"
            "        print('figura ->', outdir / 'local_surface.png'); plt.show()\n"
            "    except Exception as e:\n"
            "        print('No se pudo generar el plot:', type(e).__name__, e)"
        ),
        decisions=[
            ("**Sustracción de plano local en anillo (3–12 px)** alrededor del compañero, canal por canal, en modo `native_stage02`: quita el gradiente del halo estelar localmente.", "roxs12b_clean_spectrum_pipeline.md"),
            ("Enmascara el núcleo del compañero (3 px) y otros objetos (estrella); plano con sigma-clip 3σ.", None),
            ("Es la rama **local-surface** (`optimal_ls` / cadena de objetos lejanos); D1 mostró que `optimal_ls` arrastraba este pedestal → el canónico es **psffit**, no optimal_ls.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**04b: plano local restado en un anillo 3–12 px alrededor del compañero (156, 76); "
            "salida `cube_residual_local_object.fits`.**\n\n"
            "- **Fecha:** run realineado (2026-07-09).\n"
            "- **Entrada:** `stage02_xcorr_cube_stack.fits` (native_stage02); **modelo:** plano con "
            "sigma-clip 3σ, mediana 410 px por ajuste.\n"
            "- **Efecto:** elimina el gradiente del halo estelar en la vecindad del compañero → fondo "
            "local plano para la extracción.\n"
            "- **Máscaras:** núcleo del compañero 3 px, otros objetos 3 px; láser AO (5780–6050 Å, "
            "216 canales) marcado como malo.\n"
            "- **Rol:** alimenta la rama de extracción local-surface (`optimal_ls`) y la cadena de "
            "objetos lejanos (06/07/08); no es el método canónico (psffit lo es)."
        ),
    ),
    dict(
        id="C2", slug="C2_aperture", title="Extracción por apertura", block="C · Extracción",
        spec="spec_C2_codex_aperture_extraction.md", run_override=None,
        what="Extrae el espectro del compañero por apertura, con controles al mismo radio.",
        inputs="Cubo + posición", outputs="`stages/spec_aperture_qc.json`",
        downstream="D1, E1 (controles)",
        exec=dict(kind="script", target="stage_x01_aperture.sh", cost="Ligero–moderado."),
        qc="stages/spec_aperture_qc.json",
        salient=["apertures", "errors.mode", "aperture_correction.median", "v4_apcorr_range_ok", "bad_window_channels"],
        narrative_md=(
            "## Qué hace C2 y por qué\n\n"
            "C2 extrae el espectro del compañero por **apertura** (`box3` por defecto, `box5`) y define "
            "el **contrato de producto espectral estándar** (`wave, flux, flux_err, apcorr, npix, "
            "flags`) sobre el que se construyen **todos** los extractores (C3, C4) y D1/E1. Es "
            "infraestructura: cero ciencia nueva, la extracción ya estaba validada.\n\n"
            "**Decisiones clave:**\n"
            "- **`control = objeto`:** 33 aperturas de control se procesan **idénticas** al objeto "
            "(mismo radio, annulus de fondo, apcorr) → el σ es **empírico** (M5 STAT rojo, así que no "
            "se usa el STAT directo). [`docs/noise_model.md`](../docs/noise_model.md)\n"
            "- **Corrección de apertura (growth-curve de la PSF):** box3 capta solo una fracción de "
            "la **PSF AO ancha** → una corrección grande y **cromática** (mediana ~44.7×, ~118× en el "
            "azul → ~21× en el rojo). El chequeo `v4_apcorr_range` **falla** por ese rango enorme — "
            "se marca, no se esconde.\n"
            "- **Apcorr con extracción *wings-intact*** (cubo crudo + annulus), **no** el residual de "
            "04b: 04b sobre-sustrae las alas del compañero (daría box5 < box3, no físico), así que la "
            "curva de crecimiento se mantiene auto-consistente sobre el cubo crudo.\n\n"
            "La apertura NO es el método canónico (G1 la **rechaza** por insensible en el borde del "
            "compañero); C2 aporta el contrato de producto y el control 'A: apertura' para D1."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Aperturas, errores empíricos, corrección de apertura y chequeos del `spec_aperture_qc.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/spec_aperture_qc.json', RUN_ID)\n"
            "err = q['errors']; ac = q['aperture_correction']; ck = q['checks']\n"
            "print('apertures:', q['apertures'], '| posiciones de:', q['positions_from'].split('/')[-1])\n"
            "print(f\"errores: modo={err['mode']}, covarianza box3={err['covariance_factor_box3']:.2f}×, \"\n"
            "      f\"stat/empírico={err['stat_vs_empirical_median_ratio']:.2f}\")\n"
            "print(f\"apcorr ({ac['mode']}): mediana {ac['median']:.1f}× , máx {ac['max']:.1f}× , norm r={ac['norm_radius_px']:.0f}px\")\n"
            "print(f\"flags: bad-window {q['flags']['bad_window_channels']}, skyline {q['flags']['skyline_channels']}, clipped {q['flags']['clipped_channels']}\")\n"
            "print(f\"checks: v2_error_ratio={ck['v2_error_ratio_ok']} v3_roundtrip={ck['v3_roundtrip_ok']} v4_apcorr_range={ck['v4_apcorr_range_ok']}  (v4 falla: apcorr enorme)\")\n"
            "print()\n"
            "for i, s in enumerate(q['open_issues'], 1):\n"
            "    print(f'  open_issue {i}: {s}')"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — el espectro por apertura (box3) + ruido empírico\n\n"
                    "Del producto `spec_aperture_object.fits` y los 33 controles "
                    "(`spec_aperture_controls.npz`). El flujo es **muy ruidoso** (banda ±1σ empírica); "
                    "el continuo suavizado **sube al rojo** (SED real de enana fría) y **no hay nada "
                    "en Hα** (contexto de la no-detección). El hueco es la ventana del láser AO."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    h = fits.open(rd / 'stages' / 'spec_aperture_object.fits'); d = h[1].data\n"
                    "    w = np.asarray(d['wave_A'], float); flux = np.asarray(d['flux'], float); h.close()\n"
                    "    ctrl = np.load(rd / 'stages' / 'spec_aperture_controls.npz')['control_spectra']\n"
                    "    sig = np.nanstd(ctrl, axis=0)\n"
                    "    k = np.ones(41) / 41; sm = np.convolve(np.nan_to_num(flux), k, mode='same')\n"
                    "    fig, ax = plt.subplots(figsize=(11, 4))\n"
                    "    ax.fill_between(w, -sig, sig, color='0.8', label='±1σ empírico (33 controles)')\n"
                    "    ax.plot(w, flux, lw=0.3, color='0.5', alpha=0.6)\n"
                    "    ax.plot(w, sm, lw=1.2, color='tab:blue', label='flujo compañero (suavizado 41ch)')\n"
                    "    ax.axvline(6563, color='tab:red', ls=':', label='Hα')\n"
                    "    ax.set_ylim(np.nanpercentile(flux, 2), np.nanpercentile(flux, 98))\n"
                    "    ax.set_xlabel('λ [Å]'); ax.set_ylabel('flujo (apcorr aplicada)')\n"
                    "    ax.set_title('C2 · espectro por apertura box3 del compañero (errores empíricos)')\n"
                    "    ax.legend(fontsize=8); fig.tight_layout()\n"
                    "    outdir = rd / 'plots' / 'c2_aperture'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'spectrum.png', dpi=110); print('figura ->', outdir / 'spectrum.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — la corrección de apertura cromática\n\n"
                    "La corrección `box3 → flujo total` baja de ~118× en el azul (PSF AO peor) a ~21× "
                    "en el rojo (PSF más apretada); mediana ~44.7×. Su rango enorme es lo que hace "
                    "fallar `v4_apcorr_range` — es real (PSF AO ancha), no un defecto."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    h = fits.open(rd / 'stages' / 'spec_aperture_object.fits'); d = h[1].data\n"
                    "    w = np.asarray(d['wave_A'], float); apc = np.asarray(d['apcorr'], float); h.close()\n"
                    "    fig, ax = plt.subplots(figsize=(9, 3.6))\n"
                    "    ax.plot(w, apc, lw=0.8, color='tab:purple')\n"
                    "    ax.axhline(np.nanmedian(apc), color='k', ls='--', lw=1, label=f'mediana {np.nanmedian(apc):.1f}×')\n"
                    "    ax.set_xlabel('λ [Å]'); ax.set_ylabel('corrección de apertura ×')\n"
                    "    ax.set_title('C2 · corrección de apertura (box3 capta poco de la PSF AO ancha)')\n"
                    "    ax.legend(fontsize=8); fig.tight_layout()\n"
                    "    outdir = rd / 'plots' / 'c2_aperture'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'apcorr.png', dpi=110); print('figura ->', outdir / 'apcorr.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("Principio **control = objeto**: 33 controles con annulus bkg + apcorr, procesados idénticos al objeto → σ **empírico** (M5 rojo).", "noise_model.md"),
            ("**Corrección de apertura cromática grande** (mediana 44.7×): box3 capta poco de la PSF AO ancha; `v4_apcorr_range` falla por el rango (marcado, no oculto).", None),
            ("**Apcorr wings-intact** (cubo crudo + annulus), no el residual 04b, que sobre-sustrae las alas (box5<box3).", None),
            ("Apertura **no canónica**: G1 la rechaza por insensible en el borde; C2 aporta el contrato de producto y el control 'A' para D1.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**C2: espectro por apertura (box3/box5) en el formato de producto estándar; errores "
            "empíricos; apcorr cromática mediana 44.7×.**\n\n"
            "- **Fecha:** cadena D1 v2 sobre el run realineado (2026-07-09).\n"
            "- **Entrada:** cubo stage02; posiciones de B3; 33 controles.\n"
            "- **Espectro:** muy ruidoso, continuo real que sube al rojo (enana fría), **nada en Hα** "
            "(no-detección).\n"
            "- **Errores empíricos** (M5 STAT rojo); covarianza box3 = 6.38×.\n"
            "- **Apcorr:** growth-curve, 118× (azul) → 21× (rojo); `v4_apcorr_range` falla por el "
            "rango (real, PSF AO ancha); apcorr wings-intact para no sobre-sustraer alas.\n"
            "- **Rol:** contrato de producto para C3/C4/D1/E1; método de apertura no canónico "
            "(psffit lo es)."
        ),
    ),
    dict(
        id="C3", slug="C3_optimal", title="Extracción óptima", block="C · Extracción",
        spec="spec_C3_codex_optimal_extraction.md", run_override=None,
        what="Extracción óptima (Horne) ponderada por la PSF.",
        inputs="Cubo + PSF (C1)", outputs="`stages/spec_optimal_qc.json`",
        downstream="D1, E1",
        exec=dict(kind="script", target="stage_x02_optimal.sh", cost="Moderado."),
        qc="stages/spec_optimal_qc.json",
        salient=["variants", "snr_gain_vs_aperture.median", "continuum_bias_vs_aperture_pct", "v3_continuum_bias_ok", "fwhm_pm10pct"],
        narrative_md=(
            "## Qué hace C3 y las dos variantes\n\n"
            "C3 es **extracción óptima de Horne (1986)**: por canal, pondera cada píxel por el "
            "**perfil de PSF esperado** (de C1) y la varianza inversa "
            "(`f = Σ M·P·D/V / Σ M·P²/V`). Al bajar el peso de los píxeles ruidosos, **gana S/N** "
            "frente a la apertura (aquí ~**6.9× mediana**). La fórmula es cerrada; el valor está en "
            "implementarla exacta (tests analíticos de flujo y varianza).\n\n"
            "**Dos variantes del fondo:**\n"
            "- **`optimal_ls`** — usa el residual de superficie local (04b) como fondo.\n"
            "- **`optimal_psfsub`** — ajusta y **resta la PSF de la primaria** primero, y luego "
            "extrae ópticamente el compañero.\n\n"
            "**Decisión:** G1 **valida `psfsub`** (`validated_with_bias`) y la usa como una de las dos "
            "citables (con psffit). **`ls` sobre-sustrae el continuo** (el pedestal de 04b) → sesgo de "
            "continuo **−373 % vs apertura**, `v3_continuum_bias` **falla**. Ambas comparten la forma "
            "roja real (SED de enana fría) pero `ls` queda con un gran desplazamiento negativo.\n\n"
            "Errores **empíricos** (M5 rojo); **robusta a errores de PSF** (±10 % FWHM → 0 % de sesgo "
            "de flujo).\n\n"
            "> **Nota (revisión 2026-07-11):** el continuo de psfsub sale **negativo** — sobre-"
            "sustracción. La sección *Diagnóstico* de abajo muestra que es un **residuo del halo AO "
            "cromático** (peor en el azul), **emparejado en los controles**, correctable con un "
            "*annulus background* que el código soporta pero que psfsub/D2 no aplican. No afecta el "
            "límite de Hα; sí el nivel absoluto para caracterización."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Ganancia de S/N, sesgo de continuo, modelo psfsub y chequeos del `spec_optimal_qc.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/spec_optimal_qc.json', RUN_ID)\n"
            "sg = q['snr_gain_vs_aperture']; ck = q['checks']; pm = q['psfsub_model']\n"
            "print('variantes:', q['variants'], '| ventana:', q['window_px'], 'px | PSF:', q['psf_model'].split('/')[-1])\n"
            "print(f\"ganancia S/N vs apertura: mediana {sg['median']:.2f}× (p10 {sg['p10']:.2f}, p90 {sg['p90']:.1f})\")\n"
            "print(f\"sesgo de continuo vs apertura: {q['continuum_bias_vs_aperture_pct']:.0f}%   <-- ls sobre-sustrae\")\n"
            "print(f\"sensibilidad a PSF (±10% FWHM): {q['psf_sensitivity']['fwhm_pm10pct_flux_bias_pct']:.1f}% de sesgo (robusto)\")\n"
            "print(f\"psfsub: amplitud mediana {pm['amplitude_median']:.3g}, n_fit {pm['n_fit_median']:.0f}, \"\n"
            "      f\"fit_radius {pm['fit_radius_px']:.0f}px, exclude {pm['exclude_radius_px']:.0f}px\")\n"
            "print(f\"checks: v1_snr_gain={ck['v1_snr_gain_ok']} v2_error={ck['v2_error_ratio_ok']} \"\n"
            "      f\"v3_continuum_bias={ck['v3_continuum_bias_ok']} (falla: sesgo ls) v4_clip={ck['v4_clip_concentration_ok']}\")\n"
            "print('open_issue:', q['open_issues'][0])"
        ),
        plot_md=(
            "## Plot — las dos variantes: ls vs psfsub\n\n"
            "Los espectros suavizados de `spec_optimal_object.fits` (ls) y "
            "`spec_optimal_psfsub_object.fits` (psfsub). **`ls` (naranja)** queda muy por debajo de "
            "cero = sobre-sustracción del continuo (el sesgo −373 %); **`psfsub` (verde)** queda cerca "
            "de cero y **sube al rojo** (SED real de enana fría). Comparten la forma, difieren en "
            "nivel — por eso G1 valida psfsub y rechaza ls."
        ),
        plot_code=(
            "try:\n"
            "    import numpy as np\n"
            "    import matplotlib.pyplot as plt\n"
            "    from astropy.io import fits\n"
            "    rd = nb.run_dir(RUN_ID)\n"
            "    def spec(path):\n"
            "        h = fits.open(rd / 'stages' / path); d = h[1].data\n"
            "        w = np.asarray(d['wave_A'], float); f = np.asarray(d['flux'], float); h.close(); return w, f\n"
            "    sm = lambda x, n=41: np.convolve(np.nan_to_num(x), np.ones(n) / n, mode='same')\n"
            "    w, fls = spec('spec_optimal_object.fits')\n"
            "    _, fps = spec('spec_optimal_psfsub_object.fits')\n"
            "    fig, ax = plt.subplots(figsize=(11, 4))\n"
            "    ax.plot(w, sm(fls), lw=1.2, color='tab:orange', label='optimal_ls (superficie local)')\n"
            "    ax.plot(w, sm(fps), lw=1.2, color='tab:green', label='optimal_psfsub (resta de PSF) — validada G1')\n"
            "    ax.axhline(0, color='0.6', lw=0.7); ax.axvline(6563, color='tab:red', ls=':', label='Hα')\n"
            "    allv = np.concatenate([sm(fls), sm(fps)])\n"
            "    ax.set_ylim(np.nanpercentile(allv, 2), np.nanpercentile(allv, 98))\n"
            "    ax.set_xlabel('λ [Å]'); ax.set_ylabel('flujo (suavizado 41ch)')\n"
            "    ax.set_title('C3 · dos variantes: ls sobre-sustrae el continuo (−373% vs apertura), psfsub no')\n"
            "    ax.legend(fontsize=8); fig.tight_layout()\n"
            "    outdir = rd / 'plots' / 'c3_optimal'; outdir.mkdir(parents=True, exist_ok=True)\n"
            "    fig.savefig(outdir / 'ls_vs_psfsub.png', dpi=110); print('figura ->', outdir / 'ls_vs_psfsub.png'); plt.show()\n"
            "except Exception as e:\n"
            "    print('No se pudo generar el plot:', type(e).__name__, e)"
        ),
        plots=[
            dict(
                md=(
                    "## Diagnóstico — ¿sobre-sustracción del continuo? (objeto vs controles)\n\n"
                    "El continuo de psfsub sale **negativo**. La prueba clave: comparar el objeto con "
                    "la **media de los 33 controles** (fondo/ruido puro, debería ~0). Hallazgo:\n\n"
                    "- La media de controles **no** está en 0 y es **cromática**: ~−1315 (azul) → −210 "
                    "(rojo). Es el **residuo del halo AO** que la resta de PSF deja (peor en el azul, "
                    "donde el halo es más ancho — encaja con C1/r0).\n"
                    "- **`objeto − controles`** recupera el **continuo físico** del compañero: sube al "
                    "rojo (SED de enana fría), cerca de 0 en el azul, **nada en Hα**.\n"
                    "- El código soporta este *annulus background* (`optimal.py: local_bkg_annulus_px`) "
                    "y **C2 sí lo aplica** (sus controles quedan en ~0); psfsub y el calibrado D2 **no** "
                    "→ el continuo entregado queda sesgado. **No afecta el límite de Hα** (fondo suave, "
                    "y la detección usa control=objeto); **sí** sesga el nivel absoluto para "
                    "caracterización (G3)."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    rd = nb.run_dir(RUN_ID); wave = 4749.533203125 + 1.25 * np.arange(3681)\n"
                    "    h = fits.open(rd / 'stages' / 'spec_optimal_psfsub_object.fits')\n"
                    "    fo = np.asarray(h[1].data['flux'], float); h.close()\n"
                    "    C = np.load(rd / 'stages' / 'spec_optimal_psfsub_controls.npz')['control_spectra']\n"
                    "    cm = np.nanmean(C, axis=0)\n"
                    "    print('banda            media_ctrl     objeto   obj-ctrl   (unidades nativas)')\n"
                    "    for lo, hi in [(5100, 5500), (6600, 7200), (8000, 8800)]:\n"
                    "        b = (wave >= lo) & (wave <= hi)\n"
                    "        print(f'  {lo}-{hi} Å   {np.nanmedian(cm[b]):9.0f}  {np.nanmedian(fo[b]):9.0f}  {np.nanmedian((fo - cm)[b]):9.0f}')\n"
                    "    sm = lambda x, n=81: np.convolve(np.nan_to_num(x), np.ones(n) / n, mode='same')\n"
                    "    fig, ax = plt.subplots(figsize=(11, 4.2))\n"
                    "    ax.plot(wave, sm(fo), lw=1, color='tab:green', label='objeto psfsub (crudo, sobre-sustraído)')\n"
                    "    ax.plot(wave, sm(cm), lw=1, color='tab:red', ls='--', label='media de 33 controles = fondo residual del halo')\n"
                    "    ax.plot(wave, sm(fo - cm), lw=1.6, color='k', label='objeto − controles = continuo físico')\n"
                    "    ax.axhline(0, color='0.6', lw=0.7); ax.axvline(6563, color='tab:red', ls=':', label='Hα')\n"
                    "    allv = np.concatenate([sm(fo), sm(cm), sm(fo - cm)])\n"
                    "    ax.set_ylim(np.nanpercentile(allv, 2), np.nanpercentile(allv, 98))\n"
                    "    ax.set_xlabel('λ [Å]'); ax.set_ylabel('flujo (suavizado 81ch)')\n"
                    "    ax.set_title('C3 diagnóstico: la media de controles = sobre-sustracción cromática del halo; '\n"
                    "                 'restarla recupera el continuo físico')\n"
                    "    ax.legend(fontsize=8, loc='lower right'); fig.tight_layout()\n"
                    "    outdir = rd / 'plots' / 'c3_optimal'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'oversubtraction_diag.png', dpi=110)\n"
                    "    print('figura ->', outdir / 'oversubtraction_diag.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("Extracción óptima de Horne ponderada por la PSF de C1 → **~6.9× ganancia de S/N** vs apertura (`v1` pasa).", None),
            ("Dos variantes: **`optimal_psfsub` validada por G1** (`validated_with_bias`) y **`optimal_ls` rechazada** — sobre-sustrae el continuo (sesgo −373%, `v3` falla).", None),
            ("Errores empíricos (M5 rojo); robusta a errores de PSF (±10% FWHM → 0% de sesgo de flujo).", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**C3: extracción óptima de Horne; ~6.9× ganancia de S/N vs apertura; dos variantes "
            "(ls, psfsub).**\n\n"
            "- **Fecha:** cadena D1 v2 sobre el run realineado (2026-07-09).\n"
            "- **Entrada:** cubo stage02 + PSF de C1 (`psf_model.json`); ventana 8 px.\n"
            "- **psfsub** (resta de PSF de la primaria): validada por G1, una de las dos citables.\n"
            "- **ls** (superficie local): **sobre-sustrae** el continuo (−373 % vs apertura, "
            "`v3_continuum_bias` falla) — rechazada.\n"
            "- **Robusta a PSF** (±10 % FWHM → 0 % de sesgo); errores empíricos (M5 rojo).\n"
            "- **Hallazgo (2026-07-11):** el continuo de psfsub es negativo por un **residuo de halo "
            "AO cromático** (media de controles −1315 azul → −210 rojo), emparejado en controles. "
            "`objeto − controles` recupera el continuo físico (sube al rojo). Correctable con annulus "
            "background (soportado por el código, aplicado en C2, NO en psfsub/D2). No afecta Hα; sí "
            "el nivel absoluto para G3. Ver [[c3-continuum-oversubtraction]].\n"
            "- **Downstream:** psfsub entra en D1 (par primario psffit vs optimal_psfsub); ls no."
        ),
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
        salient=["chi2r.median", "condition_number_median", "rho_ab_median", "vs_large_aperture_median_ratio", "v3_star_scale_ok"],
        narrative_md=(
            "## Qué hace C4 y por qué es el canónico\n\n"
            "C4 (`psffit`) es el **método primario** recomendado por la literatura para un compañero a "
            "~1.75″. Por canal ajusta un modelo **lineal** por mínimos cuadrados:\n\n"
            "```\nD(y,x) = a·P_estrella + b·P_compañero + (c₀ + c₁·y + c₂·x)\n```\n\n"
            "Las incógnitas por canal son solo `(a, b, c₀, c₁, c₂)`: `a` amplitud de la estrella, `b` "
            "la del compañero, y `(c₀,c₁,c₂)` un **plano de fondo local**. Toda la no-linealidad "
            "(forma de PSF, posiciones) quedó resuelta aguas arriba (C1/B3) → el problema es lineal, "
            "exacto y rápido. **Esa es la decisión de diseño central.**\n\n"
            "**Por qué es el canónico:** ajusta estrella + compañero + fondo **simultáneamente**, "
            "atacando la decontaminación del halo de frente, **sin sustracción agresiva**. El plano "
            "`(c₀,c₁,c₂)` absorbe el gradiente del halo → es el método **menos sesgado** (por eso su "
            "media de controles es la más pequeña; ver el trabajo de referenciación en C3/D2).\n\n"
            "**Calidad del ajuste:** χ²ᵣ ≈ 1.03 (excelente), número de condición 10.7 (bien "
            "condicionado), `rho_ab` 0.17 (estrella y compañero **separables**), crosstalk 0.03 "
            "(las líneas de la estrella no contaminan al compañero), y la estrella se recupera "
            "(flujo psffit / apertura grande = 1.014).\n\n"
            "**Salvedad:** `rho_bc` mediana 0.45 (1303 canales >0.5) — el compañero es débil y queda "
            "parcialmente degenerado con el plano de fondo. Errores empíricos (M5 rojo). El producto "
            "final (`spec_final_object`) lleva además `cont_runmed_biasref` (referenciado a controles, "
            "del trabajo de D2) para G3."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Región de ajuste, condicionamiento, χ²ᵣ, crosstalk y validación de la estrella del "
            "`spec_psffit_qc.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/spec_psffit_qc.json', RUN_ID)\n"
            "fr = q['fit_region']; cond = q['conditioning']; ck = q['checks']\n"
            "print(f\"región: estrella r={fr['star_radius_px']:.0f}px, compañero r={fr['comp_radius_px']:.0f}px, {fr['n_pixels_median']:.0f} px/ajuste\")\n"
            "print(f\"χ²ᵣ: mediana {q['chi2r']['median']:.3f} (p90 {q['chi2r']['p90']:.2f})\")\n"
            "print(f\"condicionamiento: cond={cond['condition_number_median']:.1f}, \"\n"
            "      f\"rho_ab(estrella-compañero)={cond['rho_ab_median']:.2f}, \"\n"
            "      f\"rho_bc(compañero-fondo)={cond['rho_bc_median']:.2f} ({cond['channels_rho_bc_gt_0p5']} canales >0.5)\")\n"
            "print(f\"crosstalk (líneas estrella->compañero) = {q['crosstalk']['metric_corr_b_vs_a_lines']:.3f}\")\n"
            "print(f\"estrella recuperada (psffit/apertura grande) = {q['star_product_check']['vs_large_aperture_median_ratio']:.3f}\")\n"
            "print(f\"checks: v3_star_scale={ck['v3_star_scale_ok']} v4_rho_ab={ck['v4_rho_ab_ok']} rho_bc_warning={ck['rho_bc_warning']}\")\n"
            "print(f\"errores: {q['errors']['mode']} ({q['errors']['n_controls']} controles); open_issue: {q['open_issues'][0]}\")"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — el cubo residual: estrella + compañero removidos\n\n"
                    "Entrada (`stage02`) vs residual del ajuste (`cube_psffit_residual.fits`), colapsados "
                    "en 7000–8500 Å. El halo estelar casi desaparece (queda solo el residuo del anillo "
                    "~4–5% de C1 en el núcleo) y el compañero también → demuestra la decontaminación "
                    "**simultánea** y el χ²ᵣ≈1. `+` estrella, `○` compañero."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    rd = nb.run_dir(RUN_ID); wave = 4749.533203125 + 1.25 * np.arange(3681)\n"
                    "    q = nb.load_qc('stages/spec_psffit_qc.json', RUN_ID)\n"
                    "    def cube(path):\n"
                    "        h = fits.open(path); hd = next(x for x in h if x.data is not None)\n"
                    "        d = np.asarray(hd.data, float); h.close(); return d[0] if d.ndim == 4 else d\n"
                    "    res = cube(q['products']['residual_cube'])\n"
                    "    inp = cube(rd / 'stages' / 'stage02_xcorr_cube_stack.fits')\n"
                    "    sel = (wave >= 7000) & (wave <= 8500)\n"
                    "    imgi = np.nanmedian(inp[sel], axis=0); imgr = np.nanmedian(res[sel], axis=0)\n"
                    "    v = np.nanpercentile(imgi, [30, 99.5])\n"
                    "    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))\n"
                    "    for ax, im, t in [(axes[0], imgi, 'entrada: estrella + compañero'),\n"
                    "                      (axes[1], imgr, 'residual psffit: ambos removidos (χ²ᵣ≈1)')]:\n"
                    "        ax.imshow(im, origin='lower', cmap='magma', vmin=v[0], vmax=v[1])\n"
                    "        ax.plot(85, 85, '+', color='cyan', ms=10); ax.plot(76, 156, 'o', mfc='none', mec='lime', ms=12)\n"
                    "        ax.set_title(t); ax.axis('off')\n"
                    "    fig.tight_layout()\n"
                    "    outdir = rd / 'plots' / 'c4_psffit'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'residual.png', dpi=110); print('figura ->', outdir / 'residual.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — el espectro canónico del compañero (no-detección)\n\n"
                    "`spec_psffit_object.fits` con la banda ±1σ empírica (33 controles) y Hα. El continuo "
                    "sube al rojo (SED real de enana fría) y **no hay nada en Hα** — la no-detección con "
                    "el método canónico. El azul (λ<7000) está dominado por ruido (SNR<1)."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    rd = nb.run_dir(RUN_ID); wave = 4749.533203125 + 1.25 * np.arange(3681)\n"
                    "    h = fits.open(rd / 'stages' / 'spec_psffit_object.fits'); flux = np.asarray(h[1].data['flux'], float); h.close()\n"
                    "    sig = np.nanstd(np.load(rd / 'stages' / 'spec_psffit_controls.npz')['control_spectra'], axis=0)\n"
                    "    sm = np.convolve(np.nan_to_num(flux), np.ones(41) / 41, mode='same')\n"
                    "    fig, ax = plt.subplots(figsize=(11, 4))\n"
                    "    ax.fill_between(wave, -sig, sig, color='0.85', label='±1σ empírico (33 controles)')\n"
                    "    ax.plot(wave, flux, lw=0.3, color='0.55', alpha=0.6)\n"
                    "    ax.plot(wave, sm, lw=1.3, color='tab:blue', label='flujo compañero psffit (suavizado)')\n"
                    "    ax.axvline(6563, color='tab:red', ls=':', label='Hα'); ax.axhline(0, color='0.6', lw=0.6)\n"
                    "    ax.set_ylim(np.nanpercentile(flux, 2), np.nanpercentile(flux, 98))\n"
                    "    ax.set_xlabel('λ [Å]'); ax.set_ylabel('flujo (canónico)')\n"
                    "    ax.set_title('C4 · espectro canónico psffit del compañero — nada en Hα (no-detección)')\n"
                    "    ax.legend(fontsize=8); fig.tight_layout()\n"
                    "    outdir = rd / 'plots' / 'c4_psffit'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'spectrum.png', dpi=110); print('figura ->', outdir / 'spectrum.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**Ajuste lineal por canal** `a·P_estrella + b·P_compañero + plano`: toda la no-linealidad se resuelve aguas arriba (C1/B3). Es el método primario de la literatura.", None),
            ("**Simultáneo estrella+compañero+plano**: maneja el gradiente del halo de frente, sin sobre-sustracción; positivo y físico en el borde → **método canónico**.", "d1_canonical_method_decision.md"),
            ("Ajuste sano: χ²ᵣ≈1.03, cond 10.7, rho_ab 0.17 (separables), estrella recuperada 1.014, crosstalk 0.03.", None),
            ("Salvedad: `rho_bc`≈0.45 (1303 canales >0.5), compañero débil parcialmente degenerado con el plano; errores empíricos (M5 rojo).", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**C4: método canónico `psffit`; ajuste lineal simultáneo estrella+compañero+plano; "
            "χ²ᵣ≈1.03; residual limpio; no-detección.**\n\n"
            "- **Fecha:** cadena D1 v2 sobre el run realineado (2026-07-09).\n"
            "- **Ajuste:** por canal `a·P★ + b·P_c + (c₀+c₁y+c₂x)`; región estrella 20px / compañero "
            "12px; 1705 px/ajuste.\n"
            "- **Validación:** estrella recuperada (1.014 vs apertura grande), crosstalk 0.03, "
            "rho_ab 0.17 (separables).\n"
            "- **Residual:** halo + compañero removidos (queda el anillo ~4–5% de C1).\n"
            "- **Salvedad:** rho_bc≈0.45 (degeneración compañero-fondo en canales débiles); errores "
            "empíricos.\n"
            "- **Downstream:** es el canónico de D1/D2/E1/E3; el par primario de D1 es "
            "psffit vs optimal_psfsub. El producto final lleva `cont_runmed_biasref` para G3."
        ),
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
        salient=["verdict", "action", "recommended_method", "kind", "n_controls"],
        narrative_md=(
            "## Qué hace D1 y cómo decide\n\n"
            "D1 compara los métodos de extracción **por pares y por banda** para decidir cuáles son "
            "consistentes, y emite un `recommended_method` — **nunca fija el canónico** (esa es la "
            "decisión humana).\n\n"
            "**El estadístico es un t control-centrado** (`statistics.kind = t_control_centred`): para "
            "cada banda y par, compara la diferencia de continuo del objeto contra la **distribución "
            "de las diferencias de los 33 controles** (df = 32). Al restar `mu_ctrl` (la diferencia "
            "media de controles = sesgo_i − sesgo_j), **D1 ya está referenciado a controles** — por "
            "eso el pedestal de sobre-sustracción NO driva su veredicto (el trabajo de referenciación "
            "de C3/D2 solo puso a D2 al nivel de lo que D1 ya hacía).\n\n"
            "**Bandas:** B1–B6 (continuo), LHa/LHb/LOI (líneas). **Umbrales:** `|t|>2.08` divergente "
            "(p<0.0455), `|t|>3.25` fuerte (p<0.0027).\n\n"
            "**Veredicto = `divergent_continuum`** en el par primario (psffit vs optimal_psfsub, los "
            "dos validados por G1): **B6 t=+4.1 (fuerte)** y B2 t=−3.0 (marginal). Como es un t "
            "control-centrado, ese +4.1 es el **sistemático cromático genuino** (~1.35× en nivel), no "
            "el pedestal. `optimal_ls` es el **outlier**: todos sus pares divergen 18–34 → G1 lo "
            "rechaza.\n\n"
            "`recommended_method = None` (D1 se niega a auto-elegir con divergencia); el humano eligió "
            "**psffit** ([`docs/d1_canonical_method_decision.md`](../docs/d1_canonical_method_decision.md)). "
            "La `action = iterate_C1_refine_PSF_before_PCA` queda **reconocida pero no accionada** "
            "(Psfao ya está; B6 aceptado como sistemática presupuestada)."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Veredicto, t control-centrado del par primario por banda, y el outlier `optimal_ls`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage_x10_qc.json', RUN_ID)\n"
            "st = q['statistics']\n"
            "print('veredicto:', q['verdict'], '| recommended:', q['recommended_method'], '| action:', q['action'])\n"
            "print(f\"estadístico: {st['kind']} (n_controles={st['n_controls']}, df={st['n_controls']-1}); \"\n"
            "      f\"umbral divergente p<{st['p_divergent']}, fuerte p<{st['p_strong']}\")\n"
            "bands = ['B1','B2','B3','B4','B5','B6','LHa','LHb','LOI']\n"
            "pp = 'psffit_vs_optimal_psfsub'\n"
            "print(f'\\nt control-centrado del par primario ({pp}):')\n"
            "for b in bands:\n"
            "    t = q['t_matrix'][pp][b]\n"
            "    flag = '  <-- FUERTE' if abs(t) > 3.25 else ('  <- marginal' if abs(t) > 2.08 else '')\n"
            "    print(f'   {b:4s}: t = {t:+.2f}{flag}')\n"
            "print('\\noptimal_ls es el outlier (|t| máx por par):')\n"
            "for pair, row in q['t_matrix'].items():\n"
            "    if 'optimal_ls' in pair:\n"
            "        tmax = max(abs(v) for v in row.values())\n"
            "        print(f'   {pair:34s} |t|max = {tmax:.1f}')"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — el t control-centrado por par × banda\n\n"
                    "Del `t_matrix` del QC. Rojo/azul = divergencia (|t| grande). **psffit vs "
                    "optimal_psfsub** (fila primaria) es consistente salvo **B6 (+4.1)** y B2 (−3.0); "
                    "**todos los pares con `optimal_ls`** divergen 18–34 (sobre-sustracción) → ls "
                    "rechazado. psffit vs aperture es consistente en todo."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_x10_qc.json', RUN_ID)\n"
                    "    tm = q['t_matrix']\n"
                    "    bands = ['B1','B2','B3','B4','B5','B6','LHa','LHb','LOI']\n"
                    "    pairs = list(tm.keys())\n"
                    "    M = np.array([[tm[p].get(b, np.nan) for b in bands] for p in pairs])\n"
                    "    fig, ax = plt.subplots(figsize=(9, 4.2))\n"
                    "    im = ax.imshow(M, cmap='RdBu_r', vmin=-5, vmax=5, aspect='auto')\n"
                    "    ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands)\n"
                    "    ax.set_yticks(range(len(pairs))); ax.set_yticklabels([p.replace('_vs_', ' vs ') for p in pairs], fontsize=8)\n"
                    "    for i in range(len(pairs)):\n"
                    "        for j in range(len(bands)):\n"
                    "            v = M[i, j]\n"
                    "            if np.isfinite(v):\n"
                    "                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7, color='k' if abs(v) < 3 else 'w')\n"
                    "    ax.set_title('D1 · t control-centrado por par × banda (|t|>3.25 fuerte)')\n"
                    "    fig.colorbar(im, label='t'); fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'd1_compare'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'tmatrix.png', dpi=110); print('figura ->', outdir / 'tmatrix.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — el par primario: qué driva `divergent_continuum`\n\n"
                    "El t del par primario (psffit vs optimal_psfsub) por banda, con los umbrales "
                    "divergente (2.08) y fuerte (3.25). **B6 (+4.1) cruza el umbral fuerte** y B2 "
                    "(−3.0) el divergente → veredicto `divergent_continuum`. Es el sistemático "
                    "cromático genuino (~1.35×), ya libre del pedestal (t control-centrado)."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_x10_qc.json', RUN_ID)\n"
                    "    bands = ['B1','B2','B3','B4','B5','B6','LHa','LHb','LOI']\n"
                    "    tv = [q['t_matrix']['psffit_vs_optimal_psfsub'][b] for b in bands]\n"
                    "    t_div, t_str = 2.08, 3.25\n"
                    "    cols = ['tab:red' if abs(v) > t_str else ('tab:orange' if abs(v) > t_div else '0.6') for v in tv]\n"
                    "    fig, ax = plt.subplots(figsize=(9, 4))\n"
                    "    ax.bar(bands, tv, color=cols)\n"
                    "    for s in (t_div, t_str):\n"
                    "        ax.axhline(s, color='k', ls=':', lw=0.8); ax.axhline(-s, color='k', ls=':', lw=0.8)\n"
                    "    ax.axhline(0, color='k', lw=0.6)\n"
                    "    ax.set_ylabel('t control-centrado')\n"
                    "    ax.set_title('D1 · par primario psffit vs optimal_psfsub → divergent_continuum (B6 fuerte, B2 marginal)')\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'd1_compare'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'primary_pair.png', dpi=110); print('figura ->', outdir / 'primary_pair.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**Decisión humana: canónico = `psffit`** (validado por G1, físico en el borde); D1 solo recomienda (`recommended_method=None` con divergencia).", "d1_canonical_method_decision.md"),
            ("**Veredicto = `divergent_continuum`** en el par primario: B6 t=+4.1 (fuerte), B2 t=−3.0 (marginal). t **control-centrado** → es el sistemático genuino (~1.35×), no el pedestal.", None),
            ("**`optimal_ls` rechazado**: outlier en todos sus pares (|t| 18–34) por sobre-sustracción.", None),
            ("**B6 aceptado como sistemática presupuestada**: `action=iterate_C1` reconocida pero NO accionada (Psfao ya está; ver D2 y la referenciación de continuo).", "d2_red_continuum_diagnosis.md"),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**D1: veredicto `divergent_continuum` (par primario psffit vs optimal_psfsub); t "
            "control-centrado; recommended_method=None.**\n\n"
            "- **Fecha:** D1 v2 sobre el run realineado (2026-07-09).\n"
            "- **Estadístico:** t control-centrado (33 controles, df=32) → ya libre del pedestal de "
            "sobre-sustracción.\n"
            "- **Driver:** B6 t=+4.1 (fuerte), B2 t=−3.0 (marginal) = sistemático cromático genuino "
            "(~1.35× en nivel).\n"
            "- **ls rechazado:** outlier en todos sus pares (|t| 18–34).\n"
            "- **Canónico:** el humano eligió **psffit** (D1 no auto-elige con divergencia).\n"
            "- **Acción:** `iterate_C1` reconocida pero no accionada (B6 aceptado como sistemática "
            "presupuestada; ver D2)."
        ),
    ),
    dict(
        id="D2", slug="D2_calibrate", title="Calibración espectral", block="D · Método",
        spec="spec_D2_codex_spectral_calibration.md", run_override=None,
        what="Calibra el espectro canónico (Δλ, escala de flujo, continuo) y produce `spec_final_object.fits`.",
        inputs="Método canónico (psffit) + M3", outputs="`stages/stage_x11_qc.json`, `spec_final_object.fits`",
        downstream="E1, E3, G2",
        exec=dict(kind="script", target="stage_x11_calibrate.sh", cost="Ligero–moderado."),
        qc="stages/stage_x11_qc.json",
        salient=["canonical_method", "scale_factor", "v3_continuum_stable.ok", "fraction_channels_methods_agree", "control_referenced"],
        narrative_md=(
            "## Qué hace D2 y qué integramos\n\n"
            "D2 convierte el espectro canónico (psffit) en el **producto científico final**: λ "
            "corregida y en marco declarado, flujo en escala validada, continuo por dos vías, y un "
            "**error total con presupuesto de sistemáticos explícito**. **Aplica factores medidos "
            "aguas arriba** (trazables al QC que los midió) — no mide nada nuevo.\n\n"
            "- **λ:** Δλ = −0.074 Å (de A4/M1), marco final **baricéntrico**.\n"
            "- **Flujo:** escala = 1.0 (M3 factor 0.973 validado vs Gaia DR3, consistente con 1).\n"
            "- **Continuo:** running-median y polinomio; su diferencia es el término `sys_continuum`.\n"
            "- **Error:** `stat` (empírico, M5 rojo) + sistemáticos (flujo-cal, psf, cielo, telúrico, "
            "continuo). El total está **dominado por el stat**.\n\n"
            "**Diagnóstico del 'continuo rojo inestable'** ([`docs/d2_red_continuum_diagnosis.md`]"
            "(../docs/d2_red_continuum_diagnosis.md)): son 3 cosas reales (señal de enana fría + "
            "sistemático de nivel inter-método + rigidez del polinomio), **no** un defecto de PSF.\n\n"
            "**La referenciación a controles que integramos (2026-07-11):** `v3_continuum_stable` "
            "gatea ahora sobre la concordancia inter-método **control-referenciada** "
            "(**0.867**, consistente con el t control-centrado de D1), guardando la cruda (0.317) como "
            "diagnóstico; el sistemático rojo baja de **1.76× → 1.35×**; y el producto final lleva la "
            "columna `cont_runmed_biasref` para G3. v3 sigue <0.90 → falla por el sistemático "
            "cromático **genuino**, no por el pedestal. **No afecta la línea Hα ni el límite de Ṁ.**"
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Calibración aplicada + la referenciación de continuo (antes/después) del `stage_x11_qc.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage_x11_qc.json', RUN_ID)\n"
            "print('canónico:', q['canonical_method'])\n"
            "print(f\"λ: Δλ={q['wavelength']['dlambda_A']:.3f} Å, marco={q['wavelength']['frame_final']}, {q['wavelength']['status']}\")\n"
            "print(f\"flujo: escala={q['flux']['scale_factor']:.3f} ({q['flux']['source'][:60]}...)\")\n"
            "im = q['continuum']['intermethod_systematic']; ar = im['after_control_reference']\n"
            "print()\n"
            "print('continuo inter-método (psffit vs optimal_psfsub):')\n"
            "print(f\"  ANTES  (crudo)       fraction_agree={im['fraction_channels_methods_agree']:.3f}  red_ratio={im['red_band_median_ratio']:.2f}×\")\n"
            "print(f\"  DESPUÉS (referenciado) fraction_agree={ar['fraction_channels_methods_agree']:.3f}  red_ratio={ar['red_band_median_ratio']:.2f}×\")\n"
            "print(f\"  sesgo rojo psffit/psfsub = {ar['canonical_control_bias_red']:+.0f} / {ar['other_control_bias_red']:+.0f}\")\n"
            "v3 = q['checks']['v3_continuum_stable']\n"
            "print(f\"\\nv3_continuum_stable: ok={v3['ok']} (métrica={v3['metric']}, umbral {v3['threshold']}) -> falla por el sistemático genuino\")"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — la referenciación a controles (antes/después)\n\n"
                    "Continuos de los dos métodos G1-validados (psffit, optimal_psfsub), crudos vs "
                    "referenciados a sus controles. En el **rojo** (>7500 Å, donde el compañero se "
                    "detecta) los referenciados **concuerdan**; la concordancia global sube de 0.317 a "
                    "0.867. *(El azul, λ<7000, tiene SNR<1: su discrepancia está dentro del error "
                    "combinado — no es señal.)*"
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    rd = nb.run_dir(RUN_ID); wave = 4749.533203125 + 1.25 * np.arange(3681)\n"
                    "    def col(fn, c):\n"
                    "        h = fits.open(rd / 'stages' / fn); v = np.asarray(h[1].data[c], float); h.close(); return v\n"
                    "    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)\n"
                    "    axl.plot(wave, col('spec_calibrated_psffit_object.fits', 'cont_runmed'), lw=1.1, color='tab:blue', label='psffit')\n"
                    "    axl.plot(wave, col('spec_calibrated_optimal_psfsub_object.fits', 'cont_runmed'), lw=1.1, color='tab:orange', label='optimal_psfsub')\n"
                    "    axl.set_title('ANTES: continuos crudos (concuerdan 0.317)')\n"
                    "    axr.plot(wave, col('spec_calibrated_psffit_object.fits', 'cont_runmed_biasref'), lw=1.1, color='tab:blue', label='psffit ref')\n"
                    "    axr.plot(wave, col('spec_calibrated_optimal_psfsub_object.fits', 'cont_runmed_biasref'), lw=1.1, color='tab:orange', label='optimal_psfsub ref')\n"
                    "    axr.set_title('DESPUÉS: referenciados a controles (0.867)')\n"
                    "    for ax in (axl, axr):\n"
                    "        ax.set_xlabel('λ [Å]'); ax.axvline(6563, color='tab:red', ls=':'); ax.axhline(0, color='0.7', lw=0.6); ax.legend(fontsize=8)\n"
                    "    axl.set_ylabel('continuo'); axl.set_ylim(-1500, 1500)\n"
                    "    axr.text(5000, -1300, 'azul: SNR<1\\n(dentro del error)', fontsize=7, color='0.4')\n"
                    "    fig.suptitle('D2 · referenciación a controles: acerca los dos métodos G1-validados')\n"
                    "    fig.tight_layout()\n"
                    "    outdir = rd / 'plots' / 'd2_calibrate'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'referencing.png', dpi=110); print('figura ->', outdir / 'referencing.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — el presupuesto de error\n\n"
                    "Cada componente del error del `spec_final_object.fits` vs λ (escala log). El total "
                    "está **dominado por el `stat`** (empírico, M5 rojo); el sistemático de continuo "
                    "(runmed vs poly) es el segundo; flujo-cal/cielo/telúrico son ~0."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    rd = nb.run_dir(RUN_ID); wave = 4749.533203125 + 1.25 * np.arange(3681)\n"
                    "    h = fits.open(rd / 'stages' / 'spec_final_object.fits'); d = h[1].data\n"
                    "    comp = {'stat': 'flux_err_stat', 'flujo-cal': 'sys_fluxcal', 'psf': 'sys_psf',\n"
                    "            'cielo': 'sys_sky', 'telúrico': 'sys_telluric', 'continuo': 'sys_continuum'}\n"
                    "    sm = lambda x, n=51: np.convolve(np.nan_to_num(np.abs(x)), np.ones(n) / n, mode='same')\n"
                    "    fig, ax = plt.subplots(figsize=(11, 4))\n"
                    "    for lab, c in comp.items():\n"
                    "        if c in d.columns.names:\n"
                    "            ax.plot(wave, sm(np.asarray(d[c], float)), lw=1, label=lab)\n"
                    "    ax.plot(wave, sm(np.asarray(d['flux_err_total'], float)), lw=2, color='k', label='TOTAL')\n"
                    "    h.close()\n"
                    "    ax.set_yscale('log'); ax.set_xlabel('λ [Å]'); ax.set_ylabel('error (|componente|, suavizado)')\n"
                    "    ax.set_title('D2 · presupuesto de error: stat + sistemáticos'); ax.legend(fontsize=8, ncol=4)\n"
                    "    outdir = rd / 'plots' / 'd2_calibrate'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.tight_layout(); fig.savefig(outdir / 'error_budget.png', dpi=110)\n"
                    "    print('figura ->', outdir / 'error_budget.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**Diagnóstico honesto del 'continuo rojo inestable'**: 3 cosas reales (señal de enana fría + sistemático de nivel inter-método + rigidez del polinomio). NO es defecto de PSF.", "d2_red_continuum_diagnosis.md"),
            ("**Referenciación a controles integrada** (2026-07-11): v3 gatea sobre la métrica referenciada (0.867 vs cruda 0.317); sistemático rojo 1.76×→1.35×; columna `cont_runmed_biasref` entregada para G3.", None),
            ("D1 **ya era control-centrado** (su veredicto refleja el sistemático genuino); esto solo puso a D2 al mismo nivel. v3 sigue <0.90 → sistemático cromático genuino, limitación aceptada en F1.", None),
            ("El sistemático rojo NO afecta la línea Hα ni el límite de Ṁ; escala de flujo 1.0 validada vs Gaia; error total dominado por el stat.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**D2: producto final `spec_final_object` (psffit); Δλ −0.074 Å baricéntrico; flujo escala "
            "1.0 (Gaia); error dominado por el stat.**\n\n"
            "- **Fecha:** calibración D1 v2 realineado (2026-07-09); referenciación integrada "
            "2026-07-11.\n"
            "- **λ/flujo:** todos los factores trazables a A4 (M1 offset, M3 Gaia).\n"
            "- **Continuo:** referenciación a controles integrada — v3 sobre 0.867 (cruda 0.317), "
            "rojo 1.76×→1.35×, columna `cont_runmed_biasref` para G3.\n"
            "- **v3 sigue fallando** (0.867<0.90) por el sistemático cromático genuino → limitación "
            "aceptada; F1 sigue yellow.\n"
            "- **Endpoint intacto:** la línea Hα (E1) y el límite de Ṁ (E3) no dependen de esto.\n"
            "- **Downstream:** `spec_final_object` alimenta E1, E3 y G2."
        ),
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
        salient=["verdict", "reason", "global_fap_lt", "significant_methods", "rv_consistent"],
        narrative_md=(
            "## Qué hace E1 y el resultado\n\n"
            "E1 busca **emisión de Hα** del compañero con un **matched filter** (plantilla de la línea "
            "esperada) y calibra la significancia con **controles** (FAP empírico). Es el **endpoint "
            "científico**.\n\n"
            "Por método: busca en ±500 km/s alrededor de Hα (6562.8 Å, rv_sys −7) → un `z` del matched "
            "filter. La **FAP** = fracción de los 33 máximos nulos (posiciones de control) que superan "
            "el pico del objeto. **Criterio de detección:** `global_fap < 0.01` **Y** un par admisible "
            "(psffit+aperture) **Y** rv dentro de la LSF.\n\n"
            "**VEREDICTO = `non_detection`** (`no_method_passes_global_fap`): ningún método pasa. Los "
            "picos del objeto (z 1.0–3.4) caen **dentro de sus distribuciones nulas** (FAP 0.62–0.97 "
            "≫ 0.01). El pico de **psffit z=3.41** parece 'algo', pero sus nulos llegan a 8 (borde "
            "ruidoso, ~10× ruido) → FAP 0.88; además `rv_consistent=False` (v=+128 vs esperado ~−7) y "
            "FWHM 10 Å (demasiado ancho para Hα) → **ruido, no línea**.\n\n"
            "**Inputs:** LSF 2.383 Å (medida en A4/M2), rv_sys −7 (estimación de literatura), 33 "
            "controles → `min_resolvable_fap` ≈ 0.029 (aún >0.01; un FAP<1% estricto necesitaría ~99 "
            "controles — salvedad).\n\n"
            "**Resultado robusto: no hay señal de acreción en Hα de ROXs 12 B.**"
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Veredicto, criterio y el pico/FAP/rv de cada método del `stage_h01_qc.json` + la tabla."
        ),
        evidence_code=(
            "import pandas as pd\n"
            "q = nb.load_qc('stages/stage_h01_qc.json', RUN_ID)\n"
            "v = q['verdict']; cr = q['criterion']\n"
            "print('VEREDICTO:', v['verdict'], '|', v['reason'], '| métodos significativos:', v['significant_methods'])\n"
            "print(f\"criterio: global_fap<{cr['global_fap_lt']}, par admisible {cr['admissible_pairs']}, rv dentro de LSF\")\n"
            "print(f\"línea: Hα {q['line']['rest_A']} Å, rv_sys {q['line']['rv_sys_kms']} km/s, búsqueda ±{q['line']['search_half_width_kms']:.0f} km/s\")\n"
            "print()\n"
            "d = pd.read_csv(nb.run_dir(RUN_ID) / 'tables' / 'halpha_detection_by_method.csv')\n"
            "for _, r in d.iterrows():\n"
            "    print(f\"  {r['method']:15s} z={r['matched_z']:.2f}  FAP={r['global_empirical_fap']:.2f}  \"\n"
            "          f\"v={r['peak_velocity_kms']:+.0f} km/s  rv_ok={r['rv_consistent']}  fwhm={r['fwhm_A']:.1f} Å\")\n"
            "print(f\"\\nmin_resolvable_fap = {d['minimum_resolvable_fap'].iloc[0]:.3f} (33 controles; <0.01 necesita ~99)\")"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — la no-detección: pico del objeto vs distribución nula\n\n"
                    "Por método, los 33 **máximos nulos** (matched filter en posiciones de control, "
                    "gris) y el **pico del objeto** (estrella). En todos, el pico del objeto queda "
                    "**dentro de la nube nula** → FAP ≫ 0.01. psffit tiene z alto (3.41) pero sus "
                    "nulos llegan a 8 (borde ruidoso) → FAP 0.88; rv inconsistente."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import pandas as pd\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    z = np.load(rd / 'stages' / 'stage_h01_null_maxima.npz')\n"
                    "    d = pd.read_csv(rd / 'tables' / 'halpha_detection_by_method.csv').set_index('method')\n"
                    "    methods = ['aperture', 'optimal_psfsub', 'psffit', 'optimal_ls']\n"
                    "    rng = np.random.default_rng(1)\n"
                    "    fig, ax = plt.subplots(figsize=(9, 4.5))\n"
                    "    for i, m in enumerate(methods):\n"
                    "        nulls = z[f'{m}_null_maxima']; x = i + rng.uniform(-0.12, 0.12, nulls.size)\n"
                    "        ax.scatter(x, nulls, s=14, color='0.6', alpha=0.7, label='máximos nulos (33 controles)' if i == 0 else None)\n"
                    "        obj = d.loc[m, 'matched_z']; fap = d.loc[m, 'global_empirical_fap']; rv = d.loc[m, 'rv_consistent']\n"
                    "        ax.scatter(i, obj, s=170, marker='*', color='tab:orange' if rv else 'tab:red', zorder=5,\n"
                    "                   edgecolor='k', label='pico del objeto' if i == 0 else None)\n"
                    "        ax.text(i, obj + 0.35, f'z={obj:.2f}\\nFAP={fap:.2f}\\nrv_ok={rv}', ha='center', fontsize=7)\n"
                    "    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=9)\n"
                    "    ax.set_ylabel('z del matched filter (máximo en la ventana Hα)')\n"
                    "    ax.set_title('E1 · no-detección: el pico del objeto queda dentro de la nube nula (FAP >> 0.01)')\n"
                    "    ax.legend(fontsize=8, loc='upper left'); fig.tight_layout()\n"
                    "    outdir = rd / 'plots' / 'e1_halpha'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'detection.png', dpi=110); print('figura ->', outdir / 'detection.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — la región de Hα en el espectro canónico\n\n"
                    "El espectro psffit (`spec_final_object.fits`) alrededor de Hα con la banda ±1σ "
                    "empírica y la posición esperada de Hα (6562.8 Å a rv=−7). **No hay línea** por "
                    "encima del ruido en la posición esperada."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    rd = nb.run_dir(RUN_ID); wave = 4749.533203125 + 1.25 * np.arange(3681)\n"
                    "    h = fits.open(rd / 'stages' / 'spec_final_object.fits'); flux = np.asarray(h[1].data['flux'], float); h.close()\n"
                    "    sig = np.nanstd(np.load(rd / 'stages' / 'spec_calibrated_psffit_controls.npz')['control_spectra'], axis=0)\n"
                    "    ha = 6562.8 * (1 + (-7.0) / 299792.458)\n"
                    "    w = (wave >= 6400) & (wave <= 6750)\n"
                    "    fig, ax = plt.subplots(figsize=(9, 4))\n"
                    "    ax.fill_between(wave[w], -sig[w], sig[w], color='0.85', label='±1σ empírico')\n"
                    "    ax.plot(wave[w], flux[w], lw=0.9, color='tab:blue', label='flujo psffit')\n"
                    "    ax.axvline(ha, color='tab:red', ls=':', label=f'Hα esperado ({ha:.1f} Å)')\n"
                    "    ax.axhline(0, color='0.6', lw=0.6)\n"
                    "    ax.set_xlabel('λ [Å]'); ax.set_ylabel('flujo'); ax.legend(fontsize=8)\n"
                    "    ax.set_title('E1 · región de Hα: sin línea sobre el ruido en la posición esperada')\n"
                    "    fig.tight_layout()\n"
                    "    outdir = rd / 'plots' / 'e1_halpha'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'halpha_region.png', dpi=110); print('figura ->', outdir / 'halpha_region.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**VEREDICTO = `non_detection`** — ningún método supera el FAP global (0.62–0.97 ≫ 0.01); los picos caen dentro de la nube nula. **ENDPOINT CIENTÍFICO: no hay señal de acreción en Hα.**", None),
            ("El pico psffit (z=3.41) es RV-inconsistente (v=+128) y demasiado ancho (10 Å) → ruido, no línea; su nube nula llega a z=8 (borde ~10× ruido).", None),
            ("LSF = 2.383 Å (medida, A4/M2), 33 controles → min_resolvable_fap ≈ 0.029 (un FAP<1% estricto necesitaría ~99 controles).", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**E1: veredicto `non_detection` — no hay señal de acreción en Hα de ROXs 12 B.**\n\n"
            "- **Fecha:** cadena D1 v2 realineado (2026-07-09), con LSF medida.\n"
            "- **Todos los métodos:** FAP 0.62–0.97 ≫ 0.01; picos del objeto dentro de sus nubes "
            "nulas.\n"
            "- **psffit z=3.41** (el mayor) es rv-inconsistente (+128 km/s) y demasiado ancho (10 Å) "
            "→ ruido.\n"
            "- **Inputs:** LSF 2.383 Å medida, 33 controles (min_fap 0.029; ~99 para 1% estricto), "
            "rv_sys −7 (literatura).\n"
            "- **Downstream:** alimenta E3 (límite superior de Ṁ) y G2. Es el endpoint científico "
            "del proyecto."
        ),
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
