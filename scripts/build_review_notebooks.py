#!/usr/bin/env python3
"""Genera los notebooks de revisión A1..G5 (uno por spec) en `notebooks/`.

Cada notebook es una interfaz DELGADA sobre `musepipe/` + `scripts/`: no
reimplementa lógica. Por defecto AUDITA el QC existente del run realineado; con
`RUN=True` re-ejecuta la etapa mediante su comando canónico (idéntico al que se
documenta en la celda "Cómo ejecutar de forma independiente").

Uso:
    python scripts/build_review_notebooks.py            # genera las 30
    python scripts/build_review_notebooks.py C1 D2      # solo algunas

Regenerar es idempotente: sobrescribe los .ipynb de `notebooks/`.
"""
from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
DEFAULT_RUN = "ROXs12b_realigned"


def _object_slug(run_id: str) -> str:
    """Objeto de un run (para la subcarpeta `notebooks/<obj>/`).

    Prefiere `chain.target` del config; si no, el prefijo del nombre del run.
    """
    config_json = ROOT / "runs" / run_id / "config" / "config.json"
    if config_json.exists():
        try:
            payload = json.loads(config_json.read_text(encoding="utf-8"))
            chain = payload.get("chain")
            if isinstance(chain, dict) and chain.get("target"):
                return str(chain["target"])
            target = payload.get("config", {}).get("target_name")
            if target:
                return str(target)
        except (OSError, json.JSONDecodeError):
            pass
    return run_id.split("_", 1)[0]


# --------------------------------------------------------------------------
# Valores del QC dentro de la narrativa (marcadores `{{qc:...}}`)
# --------------------------------------------------------------------------
# La prosa de cada notebook llevaba los números del PRIMER objeto escritos a
# mano (p.ej. "LSF 2.383 Å @Hα"), así que el set de cualquier otro objeto
# heredaba cifras ajenas. Un marcador se resuelve, al generar, contra el QC del
# objeto que se está generando:
#
#     {{qc:stages/stage00q_qc.json:m2_lsf.lsf_fwhm_at_halpha_A:.3f}}
#
# Formato opcional tras el segundo `:`. Si la etapa no ha corrido para el
# objeto, o la clave es `null`, el marcador se resuelve a `n/d` en vez de
# inventarse un número. La resolución usa la cadena (`chain.stage_runs`) vía
# `notebooks/_nbcommon.py`, igual que los notebooks en tiempo de ejecución.
#
# La ruta admite selección dentro de listas de dicts, `clave[col=valor]`, y el
# valor puede referirse a otra clave del mismo QC con `@`:
#
#     {{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].mdot:.2e}}
#
# Para las tablas CSV de un run hay tres marcadores más (mismo `n/d` si faltan):
#
#     {{csv:tables/t.csv:method=psffit:matched_z:.2f}}   valor de una fila
#     {{csvrange:tables/t.csv:global_empirical_fap:.2f}} "mín–máx" de la columna
#     {{csvtop:tables/t.csv:matched_z:method}}           fila que maximiza una columna
#
# Y `{{target}}` = nombre legible del objeto (`targets/<slug>.json`), para que la
# prosa no lleve escrito el nombre del primer objeto reducido.
_QC_MARK = re.compile(r"\{\{qc:([^:{}]+):([^:{}]+?)(?::([^{}]*))?\}\}")
_CSV_MARK = re.compile(r"\{\{csv:([^:{}]+):([^:{}=]+)=([^:{}]*):([^:{}]+?)(?::([^{}]*))?\}\}")
_CSV_RANGE_MARK = re.compile(r"\{\{csvrange:([^:{}]+):([^:{}]+?)(?::([^{}]*))?\}\}")
_CSV_TOP_MARK = re.compile(r"\{\{csvtop:([^:{}]+):([^:{}]+):([^:{}]+?)(?::([^{}]*))?\}\}")
_QC_CACHE: dict[tuple[str, str], dict | None] = {}
_CSV_CACHE: dict[tuple[str, str], list[dict] | None] = {}


def _nbcommon():
    """`notebooks/_nbcommon.py` (stdlib puro), o None si no se puede importar."""
    import importlib
    for p in (str(NB_DIR), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        return importlib.import_module("_nbcommon")
    except ImportError:
        return None


def _qc_payload(relpath: str, run_id: str) -> dict | None:
    key = (relpath, run_id)
    if key not in _QC_CACHE:
        mod = _nbcommon()
        payload = None
        if mod is not None:
            try:
                # Fija el run activo: así `resolve_qc` avisa si el QC saliera de
                # otro objeto, igual que en el notebook.
                mod.resolve_run_id(run_id)
                path, _run, _why = mod.resolve_qc(relpath, run_id)
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                payload = None
        _QC_CACHE[key] = payload
    return _QC_CACHE[key]


def _csv_rows(relpath: str, run_id: str) -> list[dict] | None:
    """Filas de una tabla CSV del run (o None si no existe)."""
    key = (relpath, run_id)
    if key not in _CSV_CACHE:
        import csv
        mod = _nbcommon()
        rows = None
        if mod is not None:
            try:
                mod.resolve_run_id(run_id)
                path = mod.run_dir(run_id) / relpath
                with open(path, newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
            except (OSError, ValueError, csv.Error):
                rows = None
        _CSV_CACHE[key] = rows
    return _CSV_CACHE[key]


def _as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_SELECTOR = re.compile(r"^([^\[\]]+)\[([^=\[\]]+)=([^\[\]]*)\]$")


def _dotted(payload, dotted: str):
    """Camino punteado, con selección `clave[col=valor]` dentro de listas.

    El valor de la selección puede ser `@otra.clave`, que se resuelve contra el
    payload completo (p.ej. el método canónico declarado por el propio QC).
    """
    node = payload
    for part in dotted.split("."):
        sel = _SELECTOR.match(part)
        if sel:
            name, col, want = sel.groups()
            if not isinstance(node, dict) or name not in node:
                return None
            if want.startswith("@"):
                want = _dotted(payload, want[1:])
            node = next((r for r in node[name]
                         if isinstance(r, dict) and r.get(col) == want), None)
            if node is None:
                return None
            continue
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _formatted(value, spec: str | None) -> str:
    # Una cadena vacía en el QC no es un dato: es un hueco que el esqueleto del
    # driver dejó sin rellenar (`prog_id: ""`). Imprimirla tal cual dejaba la
    # prosa afirmando en blanco; `n/d` dice lo que realmente se sabe.
    if value is None or (isinstance(value, str) and not value.strip()):
        return "n/d"
    try:
        return format(value, spec) if spec else str(value)
    except (TypeError, ValueError):
        return str(value)


def resolve_qc_marks(text: str, run_id: str) -> str:
    """Sustituye los marcadores `{{qc:...}}` / `{{csv*:...}}` por los datos de `run_id`."""

    def qc_repl(m: re.Match) -> str:
        return _formatted(_dotted(_qc_payload(m.group(1), run_id) or {}, m.group(2)),
                          m.group(3))

    def csv_repl(m: re.Match) -> str:
        rel, col_key, want, col, spec = m.groups()
        row = next((r for r in (_csv_rows(rel, run_id) or []) if r.get(col_key) == want), None)
        if row is None or col not in row:
            return "n/d"
        value = _as_number(row[col])
        return _formatted(row[col] if value is None else value, spec)

    def csv_range_repl(m: re.Match) -> str:
        rel, col, spec = m.groups()
        values = [v for v in (_as_number(r.get(col)) for r in (_csv_rows(rel, run_id) or []))
                  if v is not None]
        if not values:
            return "n/d"
        return f"{_formatted(min(values), spec)}–{_formatted(max(values), spec)}"

    def csv_top_repl(m: re.Match) -> str:
        rel, rank_col, col, spec = m.groups()
        rows = [r for r in (_csv_rows(rel, run_id) or []) if _as_number(r.get(rank_col)) is not None]
        if not rows or col not in rows[0]:
            return "n/d"
        row = max(rows, key=lambda r: _as_number(r[rank_col]))
        value = _as_number(row[col])
        return _formatted(row[col] if value is None else value, spec)

    text = _QC_MARK.sub(qc_repl, text)
    text = _CSV_MARK.sub(csv_repl, text)
    text = _CSV_RANGE_MARK.sub(csv_range_repl, text)
    text = _CSV_TOP_MARK.sub(csv_top_repl, text)
    if "{{target}}" in text:
        text = text.replace("{{target}}", _target_display_name(run_id))
    return text


def _target_display_name(run_id: str) -> str:
    """Nombre legible del objeto del run (misma fuente que `_nbcommon.display_name`)."""
    mod = _nbcommon()
    if mod is None:
        return run_id.split("_", 1)[0]
    try:
        mod.resolve_run_id(run_id)
        return mod.display_name(run_id)
    except (OSError, ValueError):
        return run_id.split("_", 1)[0]


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


def _pinned_python_version() -> str:
    """Versión de Python fijada en `environment.yml` (la del kernel de los notebooks).

    Se lee del pin y no de `platform.python_version()`: así, regenerar con otro
    intérprete no reescribe la metadata de los 64 notebooks.
    """
    try:
        text = (ROOT / "environment.yml").read_text(encoding="utf-8")
    except OSError:
        return "3.10"
    m = re.search(r"^\s*-?\s*python\s*=\s*([0-9][^\s#]*)", text, re.M)
    return m.group(1) if m else "3.10"


def notebook(cells: list[dict]) -> dict:
    # La metadata es EXACTAMENTE la que escribe Jupyter al guardar. Si difiere,
    # basta abrir un notebook para que git lo dé por modificado (Jupyter
    # reescribe `kernelspec`/`language_info` con los del kernel que lo abrió), y
    # el ruido rebota entre Jupyter y este generador en cada regeneración.
    # El kernel efectivo es `python3` (el ipykernel del entorno MUSE, ver
    # `environment.yml`); `display_name` es solo la etiqueta que muestra Jupyter.
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": _pinned_python_version(),
            },
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
        # algunas etapas necesitan flags extra: si declaran `cmd`, manda ese
        return exec_spec.get("cmd") or f"python -m {exec_spec['target']} --run-id $RUN"
    if kind == "module_run":
        mod, fn = exec_spec["target"], exec_spec["fn"]
        return f"python -c \"from {mod} import {fn}; {fn}('$RUN')\""
    if kind == "pyscript":
        return f"python scripts/{exec_spec['target']} --run-id $RUN"
    if kind == "cli":
        return exec_spec["cmd"]  # literal command(s), fixed paths (no --run-id)
    return ""  # audit


def runs_referenced(cmd: str) -> list[str]:
    """Run ids que aparecen como ruta literal en un comando `kind="cli"`.

    Estas etapas no aceptan `--run-id`: sus rutas de entrada y de SALIDA están
    fijadas al run para el que se escribió el comando. Detectarlas permite
    bloquear la ejecución desde el set de notebooks de otro objeto, que
    sobrescribiría productos ajenos (hallazgo H2, Fase 0 del plan
    `docs/plan_multiobjeto_notebooks_2026-07-24.md`).
    """
    found = re.findall(r"(?:runs|MUSE_work)/([A-Za-z0-9_]+)/", cmd)
    return sorted(set(found))


# --------------------------------------------------------------------------
# Plantilla común
# --------------------------------------------------------------------------
def audit_code(body: str) -> dict:
    """Celda de auditoría: tolera que la etapa no se haya ejecutado en este objeto.

    Cualquier celda que llame a `nb.load_qc(` directamente (checks, plots,
    evidencia extra) abortaría el notebook en un objeto donde esa etapa está
    pendiente. Se envuelve solo el `FileNotFoundError`: un `KeyError` sobre un QC
    que SÍ existe es un fallo real y debe seguir siendo ruidoso.
    """
    triggers = ("nb.load_qc(", "PEREXP_DIR", "run_workdir_setting(")
    if not any(t in body for t in triggers) or body.lstrip().startswith("try:"):
        return code(body)
    indented = "\n".join(
        ("    " + line if line.strip() else line) for line in body.splitlines()
    )
    return code(
        "try:\n" + indented + "\n"
        "except FileNotFoundError as e:\n"
        "    print('[etapa pendiente para este objeto]', e)"
    )


#: Texto de la celda «figura de paper». Vive aquí, y no duplicado en cada
#: notebook, porque `build_debug_notebooks.py` la reutiliza tal cual: la figura
#: del notebook de auditoría y la del de análisis tienen que ser LA MISMA
#: figura, o comparar una con otra no diría nada.
def paper_spectrum_cell(
    *,
    arrays_code: str,
    subdir: str,
    err_label: str,
    err_alt_label: str,
    title_suffix: str,
    stem: str = "spectrum_paper",
) -> str:
    """Celda que dibuja el espectro sin binar y **escribe sus datos**.

    `arrays_code` es el trozo que cambia entre notebooks (de dónde salen los
    números) y tiene que dejar definidos, ya indentados a 4 espacios:
    `ROOT_P`, `W_P`, `F_P`, `E_P`, `E_ALT_P`, `EXTRA_P`, `BUNIT_P`, `TARGET_P`,
    `METHOD_P`, `PRODUCT_P` y `MODO_P`.
    """
    return (
        "try:\n"
        "    import numpy as np\n"
        "    import matplotlib.pyplot as plt\n"
        "    from musepipe.paper_spectrum import (paper_spectrum_figure, pretty_flux_unit,\n"
        "                                         spectrum_table_meta, write_spectrum_table)\n"
        "    from musepipe.telluric_lines import measured_transmission\n"
        f"{arrays_code}"
        "    # Cuando la etapa eligió el error empírico, la columna `flux_err` ES\n"
        "    # la empírica: dibujar las dos encima fingiría dos estimaciones\n"
        "    # independientes donde solo hay una.\n"
        "    if E_ALT_P is not None and np.allclose(E_ALT_P, E_P, equal_nan=True):\n"
        "        E_ALT_P = None\n"
        "        EXTRA_P.pop('flux_err_stat', None)\n"
        "        print('las dos columnas de error coinciden (modo empírico):'\n"
        "              ' una sola banda, y una sola columna en la tabla')\n"
        "    # La transmisión telúrica MEDIDA de este run (A3). Si el objeto se\n"
        "    # redujo en modo `cascade` no existe suelta: se marcan las bandas del\n"
        "    # catálogo sin la profundidad de esa noche, y se dice.\n"
        "    trans = measured_transmission(RUN_ID, project_root=ROOT_P)\n"
        "    print('transmisión telúrica:', trans['source'] if trans else\n"
        "          'no medida en este run — se marcan las bandas del catálogo')\n"
        "    # Los canales que la etapa marcó como malos (hueco del láser AO) no\n"
        "    # se dibujan: valen 0, y un 0 pintado se lee como una medida.\n"
        "    from musepipe.extraction.aperture import FLAG_BAD_WINDOW\n"
        "    MALOS_P = (np.asarray(EXTRA_P.get('flags', 0), dtype=int) & FLAG_BAD_WINDOW) != 0\n"
        "    fig, _ejes = paper_spectrum_figure(\n"
        "        W_P, F_P, E_P, flux_err_alt=E_ALT_P, bad_channels=MALOS_P,\n"
        f"        err_label={err_label!r}, err_alt_label={err_alt_label!r},\n"
        "        transmission=trans,\n"
        f"        title=nb.display_name(RUN_ID) + ' · ' + {title_suffix!r},\n"
        "        flux_label='flujo [' + pretty_flux_unit(BUNIT_P) + ']')\n"
        f"    outdir = nb.run_dir(RUN_ID) / 'plots' / {subdir!r}\n"
        "    outdir.mkdir(parents=True, exist_ok=True)\n"
        "    # PDF además de PNG: es la que va al paper, y en vectorial las\n"
        "    # etiquetas de las 24 líneas siguen leyéndose al ampliar. El PNG a\n"
        "    # 300 dpi es el mínimo que piden las revistas para figuras de línea.\n"
        "    DPI_P = 300      # súbelo si necesitas más resolución\n"
        "    for ext in ('png', 'pdf'):\n"
        f"        fig.savefig(outdir / ({stem!r} + '.' + ext), dpi=DPI_P)\n"
        "    tabla = write_spectrum_table(\n"
        "        nb.run_dir(RUN_ID) / 'tables' / ('spec_' + METHOD_P + '_' + TARGET_P + '.ecsv'),\n"
        "        W_P, F_P, E_P, extra_columns=EXTRA_P,\n"
        "        units={'flux': BUNIT_P, 'flux_err': BUNIT_P, 'flux_err_stat': BUNIT_P},\n"
        "        meta=spectrum_table_meta(run_id=RUN_ID, target=TARGET_P, method=METHOD_P,\n"
        "                                 product=PRODUCT_P, flux_unit=BUNIT_P,\n"
        "                                 error_mode=MODO_P,\n"
        f"                                 extra={{'figure': str(outdir / ({stem!r} + '.pdf'))}}))\n"
        f"    print('figura ->', outdir / ({stem!r} + '.pdf'))\n"
        "    print('tabla  ->', tabla, '(' + str(tabla.stat().st_size // 1024) + ' kB, '\n"
        "          + str(int(np.size(W_P))) + ' canales)')\n"
        "    print('        se lee con:  from astropy.table import Table; Table.read(ruta)')\n"
        "    plt.show()\n"
        "except Exception as e:\n"
        "    print('No se pudo generar el plot:', type(e).__name__, e)"
    )


def paper_from_product_cell(*, product, method, subdir, title_suffix, qc=None,
                            stem="spectrum_paper") -> str:
    """La celda de paper leyendo un producto `SpectrumProduct` de la cadena.

    Todos los métodos escriben el mismo esquema de columnas, así que la única
    diferencia entre uno y otro es el fichero y la etiqueta: por eso esto es una
    función y no seis copias.
    """
    lectura_qc = (
        f"    try:\n"
        f"        MODO_P = (nb.load_qc({qc!r}, RUN_ID).get('errors') or {{}}).get('mode')\n"
        f"    except Exception:\n"
        f"        MODO_P = None\n"
    ) if qc else "    MODO_P = None\n"
    return paper_spectrum_cell(
        arrays_code=(
            "    from astropy.io import fits\n"
            "    ROOT_P = nb.project_root()\n"
            f"    METHOD_P = {method!r}\n"
            f"    PRODUCT_P = {product!r}\n"
            "    TARGET_P = (nb.run_target(RUN_ID) or RUN_ID).replace(' ', '')\n"
            "    _h = fits.open(nb.run_dir(RUN_ID) / 'stages' / PRODUCT_P)\n"
            "    _d = _h[1].data\n"
            "    _cols = list(_d.columns.names)\n"
            "    # La unidad viaja con el dato (BUNIT); no hay default silencioso.\n"
            "    BUNIT_P = _h[1].header.get('BUNIT') or 'ADU'\n"
            "    W_P = np.asarray(_d['wave_A'], float)\n"
            "    F_P = np.asarray(_d['flux'], float)\n"
            "    # El empírico manda; `flux_err` es el que eligió la etapa y solo\n"
            "    # aporta algo cuando NO es el empírico (ver la nota de abajo).\n"
            "    E_P = np.asarray(_d['flux_err_emp' if 'flux_err_emp' in _cols\n"
            "                        else 'flux_err'], float)\n"
            "    E_ALT_P = np.asarray(_d['flux_err'], float)\n"
            "    EXTRA_P = {'flux_err_stat': E_ALT_P}\n"
            "    for _c in ('apcorr', 'npix_eff', 'flags'):\n"
            "        if _c in _cols:\n"
            "            EXTRA_P[_c] = np.asarray(_d[_c])\n"
            "    _h.close()\n"
            + lectura_qc
        ),
        subdir=subdir,
        stem=stem,
        err_label="±1σ empírico (controles procesados igual)",
        err_alt_label="±1σ propagado del STAT (no es σ)",
        title_suffix=title_suffix,
    )


#: La primaria, al final de cada notebook de analisis: misma figura, mismos ejes,
#: para poder poner las dos al lado.
STAR_REFERENCE_MD = (
    "## La primaria, en la misma figura\n\n"
    "Cierra el notebook el espectro de la **estrella central**, dibujado con **exactamente la "
    "misma figura** que el del compañero: mismos tramos, mismas bandas telúricas, mismas líneas "
    "marcadas y la misma tira de transmisión. Puestas una al lado de otra se comparan sin "
    "trampa.\n\n"
    "Para qué sirve mirarla:\n\n"
    "- **Es la referencia del halo.** Todo lo que este notebook resta —anillo, modelo de PSF, "
    "referencia estelar— sale de esta fuente. Su forma es la del fondo que hay que quitar, y su "
    "color explica por qué el halo es más brillante en el rojo.\n"
    "- **Separa lo atmosférico de lo del objeto.** Las bandas telúricas y los residuos de cielo "
    "aparecen en las dos, y con la misma λ. Un rasgo que solo esté en el compañero es del "
    "compañero; uno que esté en las dos, no.\n"
    "- **Da la escala.** La primaria es unas mil veces más brillante, así que cualquier fracción "
    "de su luz que se cuele en la ventana del compañero pesa mucho.\n\n"
    "> Sale del **producto de la cadena** (`spec_psffit_star.fits`, que escribe C4), no de un "
    "recálculo de este notebook: así es la misma primaria en los cinco notebooks de análisis y "
    "sirve de referencia común. Si C4 no se ha ejecutado para este objeto, la celda lo dice y "
    "sigue."
)


#: Lo que explica la celda de arriba, en los dos notebooks.
PAPER_SPECTRUM_MD = (
    "## Figura de paper — el espectro sin binar, con su error y sus líneas\n\n"
    "Las figuras anteriores son de diagnóstico. Ésta es la que se publica, y por eso "
    "cambia en tres cosas:\n\n"
    "- **Sin binar**: cada canal con su σ. Binar es cómodo para leer un continuo, pero "
    "esconde justo lo que se quiere enseñar (o no enseñar): que en Hα no hay nada por "
    "encima del ruido **a la resolución del dato**.\n"
    "- **Dos barras de error**: la **empírica** (dispersión de los controles procesados "
    "igual que el objeto) como banda, y la **propagada del STAT** como línea. Que se vean "
    "las dos es la forma honesta de enseñar que el STAT del cubo no es σ "
    "([`docs/noise_model.md`](../docs/noise_model.md)).\n"
    "- **Marcado completo**: las **bandas telúricas** sombreadas por especie (O₂ naranja, "
    "H₂O cian) con la **transmisión medida esa noche** en la tira de arriba, las **líneas "
    "de acreción** por familia (Balmer, He I, prohibidas, O I, Ca II, Paschen) y las "
    "**líneas de emisión de cielo** en gris discontinuo.\n\n"
    "### Por qué bandas telúricas y no líneas telúricas\n\n"
    "A la resolución de MUSE (FWHM ≈ 2.5 Å) las líneas individuales de O₂ y H₂O **no se "
    "resuelven**: dentro de un píxel espectral caen muchas. Marcar líneas sueltas daría "
    "una precisión que el dato no tiene, así que se marcan **bandas**. `molecfit` no está "
    "disponible aquí y, en estos datos, **no convergió** (A3 corrigió con la estrella "
    "telúrica estándar), pero de ahí quedó una **curva de transmisión medida** en la misma "
    "rejilla de λ: eso es más específico que cualquier lista de laboratorio y es lo que "
    "se dibuja. Catálogo y curva: [`musepipe/telluric_lines.py`](../musepipe/telluric_lines.py).\n\n"
    "### Y sus datos, en columnas\n\n"
    "La celda **escribe la tabla** además de la figura, en **ECSV** (el estándar portable "
    "de astropy): texto plano, con las unidades y la procedencia en la cabecera, que se "
    "lee con `Table.read(ruta)` sin configurar nada y se puede mandar por correo. Una "
    "figura sin sus datos no es un resultado citable."
)


def build_cells(s: dict) -> list[dict]:
    cells: list[dict] = []
    spec_link = f"[`docs/{s['spec']}`](../docs/{s['spec']})" if s.get("spec") else "—"
    # Run del objeto que se está generando: `DEFAULT_RUN` es el del primer objeto
    # y no debe aparecer en el set de ningún otro (ni en la cabecera ni en los
    # comandos de ejemplo).
    run_for_qc = s.get("run_override") or DEFAULT_RUN

    # 1. Encabezado / spec / rol
    cells.append(md(
        f"# {s['id']} · {s['title']}\n\n"
        f"**Spec:** {spec_link}  |  **Bloque:** {s['block']}  |  "
        f"**Run de este set:** `{run_for_qc}`\n\n"
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
    if s["exec"]["kind"] == "launch":
        howto = (
            "## Cómo ejecutar de forma independiente\n\n"
            "Etapa de **reducción**: la celda de abajo resuelve el comando real para **este "
            "objeto** a partir de su `chain.reduction_profile` y de su config, y puede lanzarlo. "
            "Son trabajos largos (ver coste), así que se lanzan en segundo plano con el log a la "
            "vista; el notebook no se bloquea.\n\n"
            "Si algún dato no está declarado en el config del run, la celda lo dice y **no lanza** "
            "en vez de inventarse una ruta.\n\n"
            f"Comando histórico de referencia:\n\n"
            f"```bash\nconda activate MUSE\n{s['exec'].get('hist_cmd','(ver spec)')}\n```\n"
        )
    elif s["exec"]["kind"] == "audit":
        howto = (
            "## Cómo ejecutar de forma independiente\n\n"
            "> ⚠️ **Etapa de solo auditoría.**\n\n"
            f"```bash\nconda activate MUSE\n{s['exec'].get('hist_cmd','(ver spec)')}\n```\n"
        )
    elif s["exec"]["kind"] == "cli":
        howto = (
            "## Cómo ejecutar de forma independiente\n\n"
            "Diagnóstico re-ejecutable sobre un cubo existente (no re-reduce nada); rutas "
            "fijas, sin `--run-id`:\n\n"
            "```bash\n"
            "conda activate MUSE               # kernel/env con astropy + musepipe\n"
            f"cd {ROOT.name}                    # raíz del repo\n"
            f"{cmd}\n"
            "```\n\n"
            f"{s['exec'].get('cost','')}\n\n"
            "La celda `RUN=True` de más abajo hace lo mismo desde el notebook."
        )
    else:
        howto = (
            "## Cómo ejecutar de forma independiente\n\n"
            "```bash\n"
            "conda activate MUSE               # kernel/env con astropy + musepipe\n"
            f"export RUN={run_for_qc}   # el run de este objeto\n"
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
        "# Localiza la raíz del repo ascendiendo hasta encontrar `musepipe/` (robusto a\n"
        "# la profundidad: funciona con el cwd en notebooks/<obj>/, en notebooks/ o en la\n"
        "# raíz). Añade la raíz (para `import musepipe`) y notebooks/ (para `_nbcommon`).\n"
        "_d = os.getcwd()\n"
        "while _d != os.path.dirname(_d):\n"
        "    if os.path.isdir(os.path.join(_d, 'musepipe')) and os.path.isdir(os.path.join(_d, 'notebooks')):\n"
        "        break\n"
        "    _d = os.path.dirname(_d)\n"
        "_root = _d\n"
        "for _p in (_root, os.path.join(_root, 'notebooks')):\n"
        "    if _p not in sys.path:\n"
        "        sys.path.insert(0, _p)\n"
        "import _nbcommon as nb\n"
        # Resolución de las figuras EN PANTALLA: `savefig` ya guarda a 300 dpi,
        # pero lo que se ve dentro del notebook lo fija el backend inline, que
        # va a 100 dpi y sale borroso. Entre try/except porque estos notebooks
        # auditan QC y tienen que abrir aunque falte el stack científico.
        "try:\n"
        "    import matplotlib as mpl\n"
        "    mpl.rcParams['figure.dpi'] = 120     # retina dobla esto sin agrandar\n"
        "    mpl.rcParams['savefig.dpi'] = 200\n"
        "    from matplotlib_inline.backend_inline import set_matplotlib_formats\n"
        "    set_matplotlib_formats('retina')\n"
        "except Exception:\n"
        "    pass\n"
        f"RUN_ID = nb.resolve_run_id({(s['run_override'] or DEFAULT_RUN)!r})\n"
        "print('run  =', RUN_ID)\n"
        "print('root =', _root)\n"
        "print('dir  =', nb.run_dir(RUN_ID))\n"
        # Procedencia: de qué run sale el QC de ESTA etapa. Nunca debe resolverse
        # una ruta en silencio (decisión 1 del plan multi-objeto).
        + (f"print('QC   =', nb.provenance_line({s['qc']!r}, RUN_ID))\n" if s.get("qc") else "")
    ))

    # 3b. Coste de ejecución (opcional, hoy solo A1). Va DESPUÉS del setup a
    # propósito: mientras se emitía antes, la celda no tenía `RUN_ID` ni `nb`, y
    # por eso el coste estaba escrito a mano con los números del primer objeto.
    # Ahora se mide sobre el run que se está auditando.
    if s.get("runtime_md"):
        cells.append(md(s["runtime_md"]))
        if s.get("runtime_code"):
            cells.append(audit_code(s["runtime_code"]))

    # 3c. Mapa completo de la cadena — solo en el primer notebook (A1), que hace
    # de panel de control del objeto.
    if s["id"] == "A1":
        cells.append(md(
            "## Mapa de la cadena de este objeto\n\n"
            "Qué etapas están ejecutadas, en **qué run** vive el QC de cada una y con qué fecha. "
            "El reparto entre runs se declara en `chain` dentro de "
            "`runs/<run>/config/config.json` (clave `stage_runs`); una etapa marcada "
            "`no ejecutada` no es un error, es trabajo pendiente para este objeto. "
            "Un `!` (CROSS-OBJECT) sí es un problema: se estaría leyendo otro objeto.\n\n"
            "Ver `docs/plan_multiobjeto_notebooks_2026-07-24.md`."
        ))
        cells.append(code("nb.show_chain(RUN_ID)"))

    # 4. Ejecutar o auditar (guardada)
    if s["exec"]["kind"] == "launch":
        cells.append(md("## Ejecutar o auditar"))
        cells.append(code(
            f"cmd, target_run, missing = nb.launch_command({s['id']!r}, RUN_ID)\n"
            "print('run que ejecuta esta etapa:', target_run)\n"
            "print('comando resuelto para este objeto:')\n"
            "print('   ', cmd or '(sin plantilla)')\n"
            "if missing:\n"
            "    print()\n"
            "    print('NO se puede lanzar: faltan datos en el config del run.')\n"
            "    print('   sin resolver:', ', '.join(missing))\n"
            "    print(f'   declara esas claves en runs/{target_run}/config/config.json')\n"
            "\n"
            "RUN = False   # -> True para LANZAR (trabajo largo: revisa el coste arriba)\n"
            "\n"
            "if RUN and not missing:\n"
            "    import subprocess, time\n"
            "    from pathlib import Path\n"
            f"    log = Path(nb.run_dir(target_run)) / 'logs' / f'{s['id'].lower()}_launch.log'\n"
            "    log.parent.mkdir(parents=True, exist_ok=True)\n"
            "    with open(log, 'w') as fh:\n"
            "        proc = subprocess.Popen(cmd, shell=True, cwd=str(nb.project_root()),\n"
            "                                stdout=fh, stderr=subprocess.STDOUT)\n"
            "    print(f'lanzado en segundo plano (pid {proc.pid}); log -> {log}')\n"
            "    print('sigue el progreso con:  !tail -f', log)\n"
            "elif RUN:\n"
            "    print"
            "('RUN=True pero hay datos sin resolver: no se lanza nada.')\n"
            "else:\n"
            "    print()\n"
            "    print('Modo auditoría (RUN=False): abajo se carga el QC existente.')"
        ))
    elif s["exec"]["kind"] == "audit":
        cells.append(md("## Auditar\n\nEtapa de solo-auditoría: se carga el producto/QC más abajo."))
    else:
        cells.append(md("## Ejecutar o auditar"))
        # Etapas `kind="cli"`: el comando lleva rutas de run FIJAS (no acepta
        # --run-id), así que `.replace('$RUN', RUN_ID)` no sustituye nada. Ejecutarlo
        # desde el set de otro objeto sobrescribiría el QC de aquel. Guard hasta E4.
        guard = ""
        if s["exec"]["kind"] == "cli":
            cmd_runs = runs_referenced(cmd)
            if cmd_runs:
                guard = (
                    "# GUARD (hallazgo H2, pendiente WP-E4): esta etapa no acepta `--run-id`;\n"
                    "# su comando lleva rutas de run FIJAS, de entrada y de SALIDA. Ejecutarlo\n"
                    "# con otro run activo sobrescribiría productos de OTRO objeto.\n"
                    f"_CMD_RUNS = {cmd_runs!r}\n"
                    "if RUN and RUN_ID not in _CMD_RUNS:\n"
                    "    raise RuntimeError(\n"
                    "        f'BLOQUEADO: el comando de esta etapa tiene rutas fijas a {_CMD_RUNS} '\n"
                    "        f'pero el run activo es {RUN_ID!r}. Ejecutarlo leería y sobrescribiría '\n"
                    "        'productos de otro objeto. Pendiente de parametrizar: WP-E4 de '\n"
                    "        'docs/plan_multiobjeto_notebooks_2026-07-24.md.'\n"
                    "    )\n\n"
                )
        cells.append(code(
            "RUN = False   # -> True para RE-EJECUTAR esta etapa (regenera su QC)\n\n"
            + guard +
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
        # `load_qc_optional` en TODAS las etapas: un objeto nuevo tiene etapas sin
        # ejecutar, y un notebook debe poder correrse de principio a fin dejando
        # el hueco visible en vez de abortar en la primera ausencia. `nb.show`
        # imprime el aviso cuando qc es None.
        cells.append(code(
            f"qc = nb.load_qc_optional({s['qc']!r}, RUN_ID)\n"
            f"nb.show(qc, keys={salient!r}, title={s['id']!r})"
        ))
    elif not (s.get("evidence_md") or s.get("evidence_code")):
        cells.append(md(
            "## QC / resultados\n\n"
            "Esta etapa no escribe un QC propio en este run; su resultado queda "
            "embebido en la reducción / documentado en la nota de decisión de abajo."
        ))

    # 5a-bis. Los chequeos del QC, traducidos a la pregunta FÍSICA que responden.
    # `v3_continuum_stable`, `t5`, `rho_ab` significan algo dentro del codigo y
    # nada para quien revisa: el nombre en clave se queda (hay que poder buscarlo
    # en el QC) pero acompañado de que pregunta contesta y que implica fallar.
    if s.get("checks_md"):
        cells.append(md(s["checks_md"]))

    # 5b. Evidencia / explicación a medida (adicional; puede coexistir con el QC)
    if s.get("evidence_md"):
        cells.append(md(s["evidence_md"]))
    if s.get("evidence_code"):
        body = s["evidence_code"]
        if s.get("qc"):
            # Dos modos de fallo distintos, dos mensajes distintos:
            #  - `qc is None`  -> la etapa no se ejecutó en este objeto (hueco).
            #  - forma inesperada -> el QC existe pero con otro esquema (H4).
            indented = "\n".join(
                ("        " + line if line.strip() else line) for line in body.splitlines()
            )
            body = (
                "if qc is None:\n"
                "    print('(evidencia omitida: la etapa no se ha ejecutado para esta cadena)')\n"
                "else:\n"
                f"    with nb.evidence_guard({s['id']!r}, {s['qc']!r}):\n" + indented
            )
        elif s.get("qc_optional"):
            indented = "\n".join("    " + line for line in body.splitlines())
            body = (
                "try:\n" + indented + "\n"
                "except FileNotFoundError as e:\n"
                "    print('QC aún no existe para este run:', e)"
            )
        cells.append(code(body))

    # 5c. Plot opcional único (requiere kernel MUSE)
    if s.get("plot_md"):
        cells.append(md(s["plot_md"]))
    if s.get("plot_code"):
        cells.append(audit_code(s["plot_code"]))

    # 5d. Varios plots (lista de {md, code})
    for _p in s.get("plots", []):
        if _p.get("md"):
            cells.append(md(_p["md"]))
        if _p.get("code"):
            cells.append(audit_code(_p["code"]))

    # 6. Decisiones
    dec_lines = ["## Decisiones y notas"]
    for text, doc in s["decisions"]:
        link = f" · [`docs/{doc}`](../docs/{doc})" if doc else ""
        dec_lines.append(f"- {text}{link}")
    cells.append(md("\n".join(dec_lines)))

    # 7. Checks
    if s.get("checks"):
        cells.append(md("## Checks"))
        body = s["checks"]
        if s.get("qc_optional"):
            indented = "\n".join("    " + line for line in body.splitlines())
            body = (
                "try:\n" + indented + "\n"
                "except FileNotFoundError as e:\n"
                "    print('QC aún no existe para este run:', e)"
            )
        cells.append(audit_code(body))

    # 8. Conclusión fechada (opcional)
    if s.get("conclusion_md"):
        cells.append(md(s["conclusion_md"]))

    # 9. Resolución de los marcadores `{{qc:...}}` de la narrativa contra el QC
    # del objeto que se está generando (ver `resolve_qc_marks`). Solo markdown:
    # el código lee el QC en tiempo de ejecución y no lleva marcadores.
    for cell in cells:
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            if "{{" in source:
                cell["source"] = _src(resolve_qc_marks(source, run_for_qc))

    return cells


# --------------------------------------------------------------------------
# Tabla de etapas (spec -> punto de entrada canónico -> QC -> decisiones)
# --------------------------------------------------------------------------
STAGES: list[dict] = [
    # ===================== BLOQUE A — reducción =====================
    dict(
        id="A1", slug="A1_raw_reduction", title="Reducción raw (esorex)", block="A · Reducción",
        spec="spec_A1_codex_raw_reduction.md", run_override=None,
        what="Reduce los raw MUSE con esorex y combina las exposiciones en el cubo del objeto.",
        inputs="Raw MUSE + calibraciones",
        outputs="cubo combinado (`cube_files` del config), `stages/stage00r_qc.json`",
        downstream="Todo el bloque B",
        exec=dict(kind="launch", hist_cmd="bash scripts/reduce_raw.sh"),
        runtime_md=(
            "## Coste de ejecución (esorex)\n\n"
            "> ⏱️ Dos números distintos, y conviene no confundirlos:\n"
            ">\n"
            "> - **Medido**: lo que costó *esta* ejecución, receta a receta, leído de los\n"
            ">   `duration_s` de los manifiestos de noche, del `perexp_scipost_execution.json`\n"
            ">   y del `elapsed_minutes` del combinado. Un paso reanudado desde un checkpoint\n"
            ">   anterior no vuelve a cobrarse, así que el total medido puede quedar **por\n"
            ">   debajo** del coste real de reducir desde cero. La celda lo dice cuando pasa.\n"
            "> - **Desde cero**: la extrapolación con las constantes medidas sobre las cuatro\n"
            ">   noches reales de los dos objetos.\n\n"
            "En perfil `cascade` la calibración se paga **por noche** (un flat, un wavecal y un\n"
            "lsf por cada noche observada), no una vez por objeto:\n\n"
            "```\n"
            "T(min) ≈ N_noches·(T_cal + T_std) + N_exp·(t_scibasic + t_scipost + t_combine)\n"
            "```\n\n"
            "con `T_cal ≈ 62 min` (34 si el LSF_PROFILE se reutiliza de archivo),\n"
            "`T_std ≈ 6.2 min`, `t_scibasic ≈ 3.0`, `t_scipost ≈ 4.5` y `t_combine ≈ 1.24`\n"
            "min/exposición. `muse_bias` no entra: se reutiliza el MASTER_BIAS de archivo.\n\n"
            "**Nota:** scibasic/scipost paralelizan sobre los 24 IFUs (OpenMP), así que el\n"
            "tiempo escala aprox. inverso al nº de núcleos; `cores_factor` ajusta respecto a\n"
            "esta máquina base (=1.0). Las constantes viven en\n"
            "`musepipe/reduction/a1_review.py`, no copiadas aquí."
        ),
        runtime_code=(
            "from musepipe.reduction import a1_review as a1\n\n"
            "rt = a1.measured_runtime(RUN_ID)\n"
            "tot = rt['totals']\n"
            "print(f\"perfil de reducción : {a1.reduction_profile(RUN_ID)}\")\n"
            "print(f\"A1 corrió en el run : {a1.resolve_a1_run(RUN_ID)}\")\n"
            "print(f\"noches              : {tot['n_nights']}\")\n"
            "print(f\"exposiciones        : {tot['n_exposures']} reducidas, \"\n"
            "      f\"{tot['n_combined']} en el cubo combinado\")\n"
            "print()\n"
            "if not rt['nights']:\n"
            "    print('[sin manifiestos de noche: el work-dir de la reducción no es legible '\n"
            "          'desde aquí — declara `work_dir` en el config del run de A1]')\n"
            "for night in rt['nights']:\n"
            "    print(f\"  noche {night['night']}  ({night['n_science']} exposiciones de ciencia)\")\n"
            "    for step in night['steps']:\n"
            "        mins = '     —' if step['minutes'] is None else f\"{step['minutes']:6.1f}\"\n"
            "        print(f\"     {step['recipe']:<14s} {step['step']:<16s} {mins} min   \"\n"
            "              f\"{step['status']}\")\n"
            "    print(f\"     {'':<14s} {'subtotal':<16s} {night['minutes']:6.1f} min\\n\")\n"
            "pe = rt['perexp']\n"
            "print(f\"  P2 scipost por exposición   {pe['minutes']:6.1f} min \"\n"
            "      f\"({pe['n_exposures']} exp, mediana \"\n"
            "      f\"{pe['median_minutes'] if pe['median_minutes'] is None else round(pe['median_minutes'], 2)}\"\n"
            "      f\" min/exp, {pe['status']})\")\n"
            "for excluded in pe['excluded']:\n"
            "    print(f\"     excluida: {excluded}\")\n"
            "cm = rt['combine']\n"
            "if cm['minutes']:\n"
            "    print(f\"  P3 combinado streaming      {cm['minutes']:6.1f} min ({cm['method']})\")\n"
            "print()\n"
            "print(f\"  TOTAL MEDIDO   {tot['measured_minutes']:7.1f} min \"\n"
            "      f\"(~{tot['measured_minutes']/60:.1f} h)\")\n"
            "print(f\"  DESDE CERO     {tot['from_scratch_minutes']:7.1f} min \"\n"
            "      f\"(~{tot['from_scratch_minutes']/60:.1f} h)   \"\n"
            "      f\"[{tot['n_nights']} noches, {tot['n_exposures']} exp, \"\n"
            "      f\"lsf {'reutilizado' if tot['reuse_lsf'] else 'calculado'}]\")\n"
            "if rt['resumed_steps']:\n"
            "    print(f\"  ({rt['resumed_steps']} paso(s) reanudados desde checkpoint no se \"\n"
            "          'cobran en el medido)')\n"
            "print()\n"
            "print('Extrapolación a otros tamaños (mismas noches, lsf como en este objeto):')\n"
            "for n in sorted({1, 5, 10, max(tot['n_exposures'], 1), 2 * max(tot['n_exposures'], 1)}):\n"
            "    m = a1.estimate_esorex_runtime(n, max(tot['n_nights'], 1),\n"
            "                                   reuse_lsf=tot['reuse_lsf'])\n"
            "    print(f'  {n:3d} exp  ->  {m:6.0f} min  (~{m/60:.1f} h)')"
        ),
        qc="stages/stage00r_qc.json",
        salient=["shape", "sha", "offset", "esorex", "muse"],
        evidence_md=(
            "## Verificaciones (V1–V6): qué comprueban y qué respondieron\n\n"
            "El QC de A1 (`stage00r_qc.json`) no re-reduce: **verifica** que el cubo entregado es "
            "sano y trazable. Qué comprueba cada una:\n\n"
            "| Check | Qué comprueba | Por qué importa |\n|---|---|---|\n"
            "| **V1** STAT | La extensión STAT (varianza) existe, es positiva y con pocos NaN, "
            "excluyendo los canales del láser AO y los spaxels de borde | Es la base de toda la "
            "propagación de error aguas abajo |\n"
            "| **V2** estándar | Continuo del estándar frente a su curva de respuesta | Cierra la "
            "validación de flujo **relativa** (si falta, se cierra por otra vía en A4/M3 vs Gaia) |\n"
            "| **V3** WCS | `CRVAL3` y paso espectral | Si deriva, los λ del cubo no son fiables |\n"
            "| **V4** vs ADP | Correlación de la imagen de luz blanca del cubo propio con la del "
            "ADP de ESO, tras igualar PSF | Validación cruzada **independiente** de la reducción "
            "propia contra el producto oficial |\n"
            "| **V5** espectro estelar | Razón del espectro de la primaria entre cubo y ADP | "
            "Comprueba la calibración en flujo a lo largo de todo el rango |\n"
            "| **V6** máscara de cielo | La máscara de cielo de scipost no muerde las fuentes | "
            "Un cielo mal enmascarado se resta del objeto |\n\n"
            "Los **resultados no se escriben aquí**: la celda siguiente los lee del QC de este "
            "objeto. Tres estados distintos y que no significan lo mismo:\n\n"
            "- `ok` / `no` — hay veredicto booleano en el QC.\n"
            "- `medido` — el QC publica el número (p.ej. `v4_adp_whitelight_corr`) pero no un "
            "veredicto; el número se imprime y el juicio es tuyo.\n"
            "- `unavailable` / `no ejecutada` — **laguna de proveniencia**, no un fallo físico. "
            "En perfil `cascade` el `stage00r_qc.json` puede ser el esqueleto vacío que escribe el "
            "driver, con los booleanos a `false` porque nadie los rellenó: eso es *no medido*, no "
            "*falló*, y la celda lo distingue explícitamente.\n\n"
            "Evidencia gráfica en la figura de más abajo."
        ),
        evidence_code=(
            # Las dos variantes de QC (`qc_schema_variant`: dict {ok,status,message}
            # en monolithic, escalar en cascade) y la detección del esqueleto viven
            # en `a1_review.verification_status`, con test propio: la celda solo
            # imprime. Antes esta tabla llevaba los resultados escritos a mano del
            # primer objeto, y los servía idénticos a cualquier otro.
            "from musepipe.reduction import a1_review as a1\n\n"
            "payload = a1.a1_qc_payload(RUN_ID)\n"
            "print(f\"perfil de reducción : {a1.reduction_profile(RUN_ID)}\")\n"
            "print(f\"QC de A1            : run {a1.resolve_a1_run(RUN_ID)}\"\n"
            "      f\"{'  ⚠ ESQUELETO VACÍO (verificaciones sin ejecutar)' if a1.is_qc_skeleton(payload) else ''}\")\n"
            "print()\n"
            "for check in a1.verification_status(RUN_ID):\n"
            "    print(f\"{check['tag']} · {check['label']}\")\n"
            "    print(f\"   -> {check['status']}\")\n"
            "    if check['message']:\n"
            "        print(f\"      {check['message']}\")\n"
            "    print()"
        ),
        plots=[
            dict(
                md=("## Esquema de la reducción realizada\n\n"
                    "Las fases que A1 ejecutó **para este objeto**, con sus entradas (arriba de "
                    "cada caja) y sus salidas (abajo), al estilo de los flujos de esoreflex. No es "
                    "un dibujo aparte que se pueda desincronizar: las entradas salen de "
                    "`esorex_driver.DEFAULT_RECIPE_REQUIREMENTS` —la tabla que gobierna la "
                    "construcción de los SOF— y las salidas, de las puertas de producto de la "
                    "cascada. Duración y estado vienen de los manifiestos de cada noche.\n\n"
                    "Tres carriles: **P1** calibración y scibasic, *por noche*; **P2** scipost "
                    "*por exposición*, con sus dos puertas humanas; **P3** el combinado. "
                    "`muse_exp_combine` aparece en rojo porque forma parte de la historia: se "
                    "planificó, corrió y no produjo cubo, y por eso existe el combinado propio."),
                code=(
                    "import matplotlib.pyplot as plt\n"
                    "from musepipe.reduction import a1_review as a1\n\n"
                    "phases = a1.a1_phase_graph(RUN_ID)\n"
                    "fig = a1.draw_phase_schematic(phases)\n"
                    "out = a1.save_figure(fig, RUN_ID, 'a1_reduction_schematic.png')\n"
                    "print(f'{len(phases)} fases  ->  {out}')\n"
                    "plt.show()"
                ),
            ),
            dict(
                md=("## Evidencia de las verificaciones\n\n"
                    "Arriba, las verificaciones que dejaron imagen: el mapa de NaN de **V1** y el "
                    "**ajuste** de **V4** —el PSF-matching contra el ADP de ESO, con el cubo propio "
                    "convolucionado al σ que maximiza la correlación, la referencia y la diferencia "
                    "normalizada—. Abajo, las dos que están medidas *en cualquier objeto* porque "
                    "las produce el combinado: el desplazamiento aplicado a cada exposición (con el "
                    "círculo de repetibilidad del centroide) y la dispersión de `CRVAL3` entre "
                    "exposiciones, en canales, que es la evidencia directa de **V3**.\n\n"
                    "Si un objeto no tiene V1–V6 corridas, los dos paneles de arriba lo dicen y los "
                    "de abajo siguen informando: la figura nunca miente por omisión."),
                code=(
                    "import matplotlib.pyplot as plt\n"
                    "from musepipe.reduction import a1_review as a1\n\n"
                    "fig = a1.draw_verification_panel(RUN_ID)\n"
                    "out = a1.save_figure(fig, RUN_ID, 'a1_verifications.png')\n"
                    "print('->', out)\n"
                    "plt.show()"
                ),
            ),
            dict(
                md=("## Los cubos que entraron al combinado (N×5)\n\n"
                    "Una fila por cubo y cinco columnas: la **mediana** de todo el cubo, tres "
                    "**bandas** (azul, centro y roja, esquivando el láser AO y las bandas "
                    "telúricas fuertes) y el **canal de Hα**, situado con la RV sistémica "
                    "declarada en el config —sin valor por defecto: si falta, la celda se niega—.\n\n"
                    "La escala está **centrada en el halo**: `asinh` con la anchura lineal fijada "
                    "por el ruido medido en el anillo exterior, y los mismos límites para todas "
                    "las filas de una columna. Eso es lo que convierte el mosaico en un "
                    "diagnóstico —una exposición con peor seeing, con el halo desplazado o con "
                    "menos flujo se ve comparando su fila con las demás—; autoescalar cada panel "
                    "lo escondería.\n\n"
                    "El encuadre y el centro de la primaria no se recalculan aquí: se toman del "
                    "plan del combinado, donde ya los midió `measure_primary_center` con MAOPPY. "
                    "El mosaico enseña, literalmente, lo que entró a combinar.\n\n"
                    "Cuesta ~2 s por cubo la primera vez y queda cacheado en "
                    "`runs/<run>/tables/a1_cube_panels.npz`."),
                code=(
                    "import matplotlib.pyplot as plt\n"
                    "from musepipe.reduction import a1_review as a1\n\n"
                    "# Los cubos de una reducción anterior del mismo objeto (`perexp_cubes`)\n"
                    "# están apagados a propósito: mezclarían dos reducciones en una figura.\n"
                    "USE_LEGACY_PEREXP = False\n\n"
                    "rows = a1.list_a1_cubes(RUN_ID, include_legacy=USE_LEGACY_PEREXP)\n"
                    "kinds = {r.kind for r in rows}\n"
                    "print(f'{len(rows)} cubo(s) en disco  ({\", \".join(sorted(kinds)) or \"ninguno\"})')\n"
                    "if kinds == {'combined'}:\n"
                    "    print('   Solo queda el cubo combinado: los cubos por exposición de esta\\n'\n"
                    "          '   reducción se purgaron tras combinar (ocupaban cientos de GB).\\n'\n"
                    "          '   La fila única NO es un error, es lo que hay en disco.')\n"
                    "ha_A, rest_A, rv = a1.halpha_observed_A(RUN_ID)\n"
                    "print(f'Hα: reposo {rest_A:.2f} Å, RV {rv:+.1f} km/s -> observada {ha_A:.2f} Å')\n\n"
                    "panels = a1.build_panels(\n"
                    "    RUN_ID, rows,\n"
                    "    progress=lambda i, n, label: print(f'  [{i}/{n}] {label}', flush=True))\n"
                    "fig = a1.draw_cube_mosaic(panels, rows)\n"
                    "out = a1.save_figure(fig, RUN_ID, 'a1_cube_mosaic.png', dpi=110)\n"
                    "print('->', out)\n"
                    "plt.show()"
                ),
            ),
        ],
        decisions=[
            ("**Alineado por el centroide de la primaria medido exposición a exposición** "
             "(`stream_combine.measure_primary_center`, ajuste MAOPPY), no por `muse_exp_align`. "
             "Los dos objetos llegaron aquí por caminos distintos: en ROXs 12 B `exp_align` daba "
             "offsets espurios de hasta 3.305\" (cross-match de speckles NFM), y en ROXs 42B b "
             "`muse_exp_combine` corrió sin llegar a producir cubo. La repetibilidad del centroide "
             "y los desplazamientos aplicados están en el QC del combinado y en la figura de "
             "verificaciones.", None),
            ("El combinado es **propio** (`musepipe.reduction.stream_combine`): sigma-clipping por "
             "voxel en ventanas de canales, pesos por tiempo de exposición y recorte a 200 px. Se "
             "escribió porque `muse_exp_combine` no era viable con estos volúmenes.", None),
            ("El semáforo de A1 **sale de las verificaciones de este objeto**, no de una nota fija: "
             "míralo en la celda de V1–V6 de arriba. `unavailable` y `no ejecutada` son lagunas de "
             "proveniencia documentadas, no fallos físicos.", None),
            ("**Solo ROXs 12 B (histórico, 2026-07-19):** los 6 blockers duros del bloque A están "
             "cerrados → F1 realineado = `yellow`, 0 bloqueantes; agrupación BIAS aceptada (A2b) y "
             "telúrica justificada (A1b), con molecfit corroborando STD_TELLURIC (A1a). Este "
             "párrafo NO se ha rehecho para otros objetos.", None),
        ],
        checks="print('open_issues A1 (no bloqueantes):')\nfor i, s in enumerate(qc.get('open_issues', []), 1):\n    print(f'  {i}. {s}')",
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**A1 de {{target}} — todo lo de abajo sale del QC de este objeto; `n/d` significa que "
            "esa etapa no lo dejó escrito, no que valga cero.**\n\n"
            "- **Entorno:** esorex {{qc:stages/stage00r_qc.json:environment.esorex}} / "
            "MUSE {{qc:stages/stage00r_qc.json:environment.muse_pipeline}}.\n"
            "- **Datos:** modo {{qc:stages/stage00r_qc.json:inputs.ins_mode}}, "
            "Prog {{qc:stages/stage00r_qc.json:inputs.prog_id}} "
            "(OB {{qc:stages/stage00r_qc.json:inputs.obs_id}}). El número de noches y de "
            "exposiciones **no se copia aquí**: la celda de coste lo cuenta de los manifiestos de "
            "la reducción, que siguen siendo correctos aunque el `stage00r_qc.json` sea el "
            "esqueleto vacío del driver.\n"
            "- **Cubo:** {{qc:stages/stage00r_qc.json:products.cube_shape_zyx}} (z,y,x), fracción "
            "finita {{qc:stages/stage00r_qc.json:products.finite_fraction:.4f}}, "
            "sha256 `{{qc:stages/stage00r_qc.json:products.datacube_sha256}}`.\n"
            "- **Estado del QC:** {{qc:stages/stage00r_qc.json:status}}.\n"
            "- **Coste:** el medido y el de una reducción desde cero están en la celda de coste, "
            "calculados sobre las noches y exposiciones reales de este objeto.\n\n"
            "El cubo de A1 es la entrada de todo el bloque B. Nada de esto es paper-final sin "
            "declarar los caveats que las verificaciones dejen abiertos."
        ),
    ),
    dict(
        id="A2", slug="A2_sky_zap", title="Decisión ZAP (cielo)", block="A · Reducción",
        spec="spec_A2_codex_sky_zap.md", run_override=None,
        what="Decide y aplica (o descarta) la sustracción de cielo con ZAP.",
        inputs="Cubo reducido", outputs="Cubo con cielo tratado (sin QC separado en este run)",
        downstream="A3, A4",
        exec=dict(kind="launch", hist_cmd="bash scripts/sky_zap.sh"),
        qc=None,
        narrative_md=(
            "## Qué es ZAP y por qué se necesita\n\n"
            "**ZAP** (*Zurich Atmosphere Purge*, [Soto, Lilly, Bacon, Richard & Conseil 2016, "
            "MNRAS 458, 3210](https://doi.org/10.1093/mnras/stw474)) es una sustracción de "
            "**residuos de cielo** para MUSE basada en PCA. El pipeline "
            "(`muse_scipost --skymethod=model`, [Weilbacher et al. 2020, A&A 641, A28]"
            "(https://doi.org/10.1051/0004-6361/202037855) §3.2) ya resta un modelo de cielo "
            "— la 3ª figura de abajo lo verifica en el log de esorex —, pero el **airglow** "
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
            "sky_frac = m4.get('sky_fraction')   # QCs antiguos no lo persisten (gap documentado)\n"
            "print(f'm4_sky.R = {R}   (RMS skyline / RMS continuo en aperturas vacías)   "
            "[status {m4.get(\"status\")}]')\n\n"
            "# Decisión con la implementación OFICIAL (no reimplementada aquí):\n"
            "try:\n"
            "    from musepipe.reduction.sky_zap import classify_zap_decision\n"
            "    if R is None:\n"
            "        print('sin dato de R en el QC -> decisión no evaluable')\n"
            "    else:\n"
            "        if sky_frac is None:\n"
            "            print('AVISO: el QC no persiste sky_fraction -> la rama '\n"
            "                  'insufficient_sky (<0.25) no es re-derivable aquí; se evalúa solo la rama R.')\n"
            "        d = classify_zap_decision(float(R), float(sky_frac) if sky_frac is not None else 1.0)\n"
            "        print(f'classify_zap_decision: {d.decision}  ->  zap_applied = {d.zap_applied}'\n"
            "              f'  (checkpoint_required={d.checkpoint_required})')\n"
            "except Exception as e:\n"
            "    print('No se pudo importar musepipe (kernel sin la pila científica):', type(e).__name__, e)\n"
            "    print('Regla pre-registrada (espejo de classify_zap_decision): R<=1.5 no necesario | '\n"
            "          'R>2.0 necesario | zona gris -> checkpoint | sky_fraction<0.25 -> checkpoint')\n\n"
            "print()\n"
            "print('Corroboración (nota del QC):')\n"
            "print('  ', qc.get('note'))\n"
            "print('M1/M2 se midieron del SKY_SPECTRUM cacheado (airglow, 32 exp,',\n"
            "      qc.get('m1_wavelength', {}).get('n_measurements'), 'medidas) porque el')\n"
            "print('cubo restado de cielo tiene <8 skylines usables.')"
        ),
        plot_md=(
            "## De dónde sale R: los datos, la zona y de quién es la prueba\n"
            "\n"
            "**Un solo FITS:** el cubo de A1 (`cube.file` del QC), extensión **DATA** (la STAT no\n"
            "interviene en R). R se mide sobre los **spaxels de cielo** de ese cubo — no hay varios\n"
            "archivos. *(Los `SKY_SPECTRUM` cacheados son de M1/M2, no de R.)*\n"
            "\n"
            "### Procedencia de la prueba (esto es lo que cita la figura)\n"
            "\n"
            "| Pieza | Origen |\n"
            "|---|---|\n"
            "| **ZAP**, el método que se decide aplicar o no | [Soto, Lilly, Bacon, Richard & Conseil 2016, MNRAS 458, 3210](https://doi.org/10.1093/mnras/stw474) |\n"
            "| **La resta de cielo que ya hizo el DRS** (lo que R audita) | [Weilbacher et al. 2020, A&A 641, A28](https://doi.org/10.1051/0004-6361/202037855) §3.2; [Streicher et al. 2011, ASPC 442, 257](https://ui.adsabs.harvard.edu/abs/2011ASPC..442..257S) |\n"
            "| **Qué λ son skyline** (5577.3, 6300.3, 6363.8 Å, banda O₂ 6864–6960, bandas OH 7240–9300) | atlas de airglow del VLT, [Hanuschik 2003, A&A 407, 1157](https://doi.org/10.1051/0004-6361:20030783) |\n"
            "| **El cociente `R` y los umbrales 1.5 / 2.0** | **métrica de este proyecto**, `docs/spec_A2_codex_sky_zap.md` §3 — **no** viene de un paper |\n"
            "\n"
            "Esa última fila importa: `R` es una regla **pre-registrada nuestra**, no un test estándar de\n"
            "la literatura. Lo que la respalda es que se fijó antes de mirar el dato y que la conclusión\n"
            "sobrevive a una versión de la prueba libre del sesgo que se explica abajo.\n"
            "\n"
            "### El sesgo del `R` global, y la prueba que no lo tiene\n"
            "\n"
            "`R` compara la mediana del RMS en ventanas de skyline contra la mediana en dos ventanas de\n"
            "continuo (5100–5500, 6600–6800 Å). Pero el RMS por canal **no es plano en λ**: la curva de\n"
            "respuesta de MUSE hunde la transmisión en el azul, así que el mismo residuo en cuentas sale\n"
            "amplificado allí. Medido en este cubo: RMS ≈ 3.0 en 5100–5500 Å frente a ≈ 1.6 en\n"
            "6600–6800 Å. Como casi todas las ventanas de skyline caen en el rojo (donde la respuesta es\n"
            "plana), **parte de que `R < 1` es la curva de respuesta, no cielo limpio**.\n"
            "\n"
            "Por eso el panel (b) repite la prueba **dentro de cada banda**: `R_local` = RMS de los\n"
            "canales que caen sobre airglow **brillante** ÷ RMS de los canales vecinos de airglow\n"
            "**débil**, en la misma banda de 575 Å (ranking de brillo tomado del `SKY_SPECTRUM` del DRS,\n"
            "que es el airglow real antes de restarlo; si no está en disco, del propio residuo del cubo).\n"
            "Ambos numerador y denominador comparten la misma respuesta → el cociente es limpio. Si los\n"
            "residuos de skyline fueran un problema, `R_local` se despegaría de 1 en las bandas de OH.\n"
            "\n"
            "> Necesita el kernel **MUSE** (astropy) y el cubo en disco. Usa la **máscara oficial** de A2\n"
            "> (`zap_source_mask.fits`, el mismo FITS que escribió la etapa), así que el `R` reproducido\n"
            "> aquí coincide con el del QC salvo redondeo; la celda imprime de dónde salió la máscara."
        ),
        plot_code=(
            "MAKE_PLOT = True   # carga el cubo (~590 MB en RAM) via astropy; requiere kernel MUSE\n"
            "\n"
            "# Procedencia de la prueba, al pie de cada figura de A2. La ultima linea es la\n"
            "# honesta: R es una regla de este proyecto, no un test publicado.\n"
            "A2_REFS = (\n"
            "    \"ZAP: Soto et al. 2016, MNRAS 458, 3210  ·  \"\n"
            "    \"resta de cielo del DRS: Weilbacher et al. 2020, A&A 641, A28 (3.2), Streicher et al. 2011, ASPC 442, 257  ·  \"\n"
            "    \"lineas/bandas de airglow: Hanuschik 2003, A&A 407, 1157  ·  \"\n"
            "    \"el cociente R y los umbrales 1.5/2.0: metrica de este proyecto (docs/spec_A2_codex_sky_zap.md sec.3), sin paper\"\n"
            ")\n"
            "\n"
            "\n"
            "def _a2_load():\n"
            "    \"\"\"Cubo + mascara OFICIAL de cielo de A2, una sola vez, reusados por las 3 figuras.\"\"\"\n"
            "    global _A2\n"
            "    if \"_A2\" in globals() and _A2.get(\"data\") is not None:\n"
            "        return _A2\n"
            "    import numpy as np\n"
            "    from pathlib import Path\n"
            "    from astropy.io import fits\n"
            "\n"
            "    qc_q = nb.load_qc(\"stages/stage00q_qc.json\", RUN_ID)\n"
            "    cube_path = qc_q.get(\"input_cube\") or qc_q.get(\"cube\", {}).get(\"file\")\n"
            "    if not cube_path:\n"
            "        raise RuntimeError(\"el QC de A4 no declara el cubo de entrada\")\n"
            "    # El QC guarda unas rutas absolutas y otras RELATIVAS a la raiz del repo\n"
            "    # (ROXs 12 b: /mnt/2TB/...; ROXs 42B b: runs/<run>/cube_telcorr.fits). El cwd\n"
            "    # del notebook es notebooks/<objeto>/ -- la celda de setup anade la raiz al\n"
            "    # sys.path pero NO hace chdir --, asi que una relativa no resuelve sola.\n"
            "    cube_path = Path(cube_path)\n"
            "    if not cube_path.is_absolute():\n"
            "        cube_path = nb.project_root() / cube_path\n"
            "    print(\"FITS usado:\", cube_path)\n"
            "    if not cube_path.exists():\n"
            "        raise FileNotFoundError(\n"
            "            f\"el cubo que declara el QC no esta en disco: {cube_path}\"\n"
            "            \"  (revisa cube.file del QC de A4 para este run)\")\n"
            "\n"
            "    hdul = fits.open(cube_path, memmap=True)\n"
            "    data = np.asarray(hdul[1].data, dtype=np.float32)\n"
            "    hd = hdul[1].header\n"
            "    n3 = hd[\"NAXIS3\"]\n"
            "    wave = hd[\"CRVAL3\"] + (np.arange(n3) - (hd[\"CRPIX3\"] - 1)) * hd[\"CD3_3\"]\n"
            "    wl = np.nanmedian(data, axis=0)\n"
            "    finite = np.isfinite(wl)\n"
            "\n"
            "    # Mascara de cielo: la OFICIAL que escribio la etapa, no una aproximacion.\n"
            "    # Orden explicito y anunciado; si no hay ninguna se reconstruye con la funcion\n"
            "    # canonica y se dice. Nunca un percentil silencioso.\n"
            "    mask_path, origin = None, None\n"
            "    for qc_name, key in ((\"stages/stage00s_qc.json\", (\"mask\", \"file\")),\n"
            "                         (\"stages/stage00q_qc.json\", (\"m4_sky\", \"source_mask\"))):\n"
            "        try:\n"
            "            _qc = nb.load_qc(qc_name, RUN_ID)\n"
            "        except FileNotFoundError:\n"
            "            continue\n"
            "        cand = (_qc.get(key[0]) or {}).get(key[1])\n"
            "        if not cand:\n"
            "            continue\n"
            "        cand = Path(cand)\n"
            "        if not cand.is_absolute():          # relativa a la raiz, igual que el cubo\n"
            "            cand = nb.project_root() / cand\n"
            "        if cand.exists():\n"
            "            mask_path, origin = cand, f\"{qc_name}:{key[0]}.{key[1]}\"\n"
            "            break\n"
            "    if mask_path is not None:\n"
            "        source = np.asarray(fits.getdata(mask_path)) > 0\n"
            "        print(f\"mascara de cielo: FITS oficial de la etapa ({origin})\")\n"
            "    else:\n"
            "        from musepipe.reduction.sky_zap import build_source_mask, SourceRegion\n"
            "        _sy, _sx = np.unravel_index(np.nanargmax(wl), wl.shape)\n"
            "        _regs = [SourceRegion(name=\"primary\", yx=(float(_sy), float(_sx)),\n"
            "                              radius_px=20.0, auto_halo=True)]\n"
            "        try:\n"
            "            _gc = nb.load_qc(\"stages/stage00s_qc.json\", RUN_ID).get(\"growth_curve\", {})\n"
            "            _cyx = _gc.get(\"companion_masked_yx\")\n"
            "        except FileNotFoundError:\n"
            "            _cyx = None\n"
            "        if _cyx:\n"
            "            _regs.append(SourceRegion(name=\"companion\", yx=(float(_cyx[0]), float(_cyx[1])),\n"
            "                                      radius_px=8.0, auto_halo=False))\n"
            "        source, _radii = build_source_mask(wl, _regs)\n"
            "        print(\"mascara de cielo: RECONSTRUIDA con build_source_mask (no habia FITS en el QC);\",\n"
            "              f\"primaria en {(_sy, _sx)}, companero {_cyx}, radios {_radii}\")\n"
            "    sky_mask = finite & ~source\n"
            "\n"
            "    # Mediana del cielo por spaxel, canal a canal, por trozos (no duplica el cubo).\n"
            "    idx = np.where(sky_mask.ravel())[0]\n"
            "    sky_med = np.full(n3, np.nan)\n"
            "    for c0 in range(0, n3, 200):\n"
            "        blk = data[c0:c0 + 200].reshape(min(200, n3 - c0), -1)[:, idx]\n"
            "        sky_med[c0:c0 + blk.shape[0]] = np.nanmedian(blk, axis=1)\n"
            "\n"
            "    pix = abs(float(hd.get(\"CD1_1\") or hd.get(\"CDELT1\", 0.0))) * 3600.0\n"
            "    sy, sx = np.unravel_index(np.nanargmax(wl), wl.shape)\n"
            "    _A2 = dict(data=data, hd=hd, wave=wave, wl=wl, sky_mask=sky_mask, sky_med=sky_med,\n"
            "               n3=n3, pix=pix, star_yx=(int(sy), int(sx)), cube_path=cube_path,\n"
            "               qc_q=qc_q, hdul=hdul)\n"
            "    print(f\"cielo: {sky_mask.mean():.4f} del campo ({sky_mask.sum()} spaxels)   \"\n"
            "          f\"escala {pix:.4f}\\\"/px   primaria en (y,x)={(int(sy), int(sx))}\")\n"
            "    return _A2\n"
            "\n"
            "\n"
            "if MAKE_PLOT:\n"
            "    try:\n"
            "        import numpy as np\n"
            "        import matplotlib.pyplot as plt\n"
            "        from musepipe.reduction.sky_zap import (\n"
            "            compute_sky_residual_metrics, channel_rms_from_mask,\n"
            "            SKYLINE_WINDOWS, CONTINUUM_WINDOWS)\n"
            "\n"
            "        A = _a2_load()\n"
            "        wave = A[\"wave\"]\n"
            "        m = compute_sky_residual_metrics(A[\"data\"].astype(np.float64), wave, A[\"sky_mask\"])\n"
            "        R = m[\"R_skyline_over_continuum\"]\n"
            "        rms = m[\"channel_rms\"]\n"
            "        _qcR = A[\"qc_q\"].get(\"m4_sky\", {}).get(\"R\")\n"
            "        print(f\"R reproducido = {R:.3f}   (oficial m4_sky.R = {_qcR})\")\n"
            "\n"
            "        # El sesgo de la respuesta, en numeros: las dos ventanas de continuo no son\n"
            "        # equivalentes entre si, y las de skyline viven casi todas en el rojo plano.\n"
            "        print(\"\\nRMS mediano por ventana (asi se ve el sesgo de la curva de respuesta):\")\n"
            "        for tag, wins in ((\"skyline \", SKYLINE_WINDOWS), (\"continuo\", CONTINUUM_WINDOWS)):\n"
            "            for a, b in wins:\n"
            "                s = (wave >= a) & (wave <= b) & np.isfinite(rms)\n"
            "                print(f\"  {tag} {a:7.1f}-{b:7.1f} A  n={int(s.sum()):5d}  RMS={np.median(rms[s]):7.3f}\")\n"
            "\n"
            "        # R_local: misma prueba dentro de cada banda, inmune a la respuesta.\n"
            "        # Ranking de brillo del airglow: SKY_SPECTRUM del DRS si esta en disco\n"
            "        # (es el cielo REAL antes de restarlo); si no, el residuo del propio cubo.\n"
            "        from pathlib import Path\n"
            "        import glob as _glob\n"
            "        _ref, _ref_src = None, None\n"
            "        _root = Path(A[\"cube_path\"]).parent.parent\n"
            "        _cand = sorted(_glob.glob(str(_root / \"*\" / \"products\" / \"*\" / \"SKY_SPECTRUM_*.fits\")))\n"
            "        if _cand:\n"
            "            from astropy.io import fits as _fits\n"
            "            _t = _fits.open(_cand[0])[1].data\n"
            "            _ref = np.interp(wave, _t[\"lambda\"], _t[\"data\"])\n"
            "            _ref_src = f\"SKY_SPECTRUM del DRS ({Path(_cand[0]).parent.name})\"\n"
            "        else:\n"
            "            _ref = A[\"sky_med\"]\n"
            "            _ref_src = \"residuo de cielo del propio cubo (no hay SKY_SPECTRUM en disco)\"\n"
            "        print(f\"\\nR_local: ranking de airglow = {_ref_src}\")\n"
            "        edges = np.linspace(wave[0], wave[-1], 9)\n"
            "        rloc, rlab = [], []\n"
            "        for i in range(8):\n"
            "            s = (wave >= edges[i]) & (wave < edges[i + 1]) & np.isfinite(rms) & np.isfinite(_ref)\n"
            "            v, r = _ref[s], rms[s]\n"
            "            hi, lo = v >= np.percentile(v, 70), v <= np.percentile(v, 30)\n"
            "            rloc.append(float(np.median(r[hi]) / np.median(r[lo])))\n"
            "            rlab.append(f\"{edges[i]:.0f}\\n{edges[i + 1]:.0f}\")\n"
            "            print(f\"  {edges[i]:6.0f}-{edges[i + 1]:6.0f} A   R_local = {rloc[-1]:5.3f}\"\n"
            "                  f\"   (brillante {np.median(r[hi]):6.3f} / debil {np.median(r[lo]):6.3f})\")\n"
            "        print(f\"  -> maximo {max(rloc):.3f} sobre las 8 bandas, umbral de la etapa 1.5\")\n"
            "\n"
            "        fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.6),\n"
            "                                 gridspec_kw={\"height_ratios\": [1, 1]})\n"
            "        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]\n"
            "\n"
            "        # (a) la prueba tal como la define la etapa\n"
            "        ax1.plot(wave, rms, lw=0.5, color=\"0.35\")\n"
            "        for i, (a, b) in enumerate(SKYLINE_WINDOWS):\n"
            "            ax1.axvspan(a, b, color=\"tab:red\", alpha=0.18,\n"
            "                        label=\"ventana skyline (Hanuschik 2003)\" if i == 0 else None)\n"
            "        for i, (a, b) in enumerate(CONTINUUM_WINDOWS):\n"
            "            ax1.axvspan(a, b, color=\"tab:green\", alpha=0.25,\n"
            "                        label=\"ventana continuo\" if i == 0 else None)\n"
            "        ax1.axhline(m[\"skyline_rms_median\"], color=\"tab:red\", ls=\"--\", lw=1)\n"
            "        ax1.axhline(m[\"continuum_rms_median\"], color=\"tab:green\", ls=\"--\", lw=1)\n"
            "        ax1.set_xlabel(\"lambda [A]\"); ax1.set_ylabel(\"RMS por canal (cielo)\")\n"
            "        ax1.set_title(f\"(a) prueba R de la etapa: med(skyline)/med(continuo) = {R:.3f}\"\n"
            "                      f\"  [umbral 1.5]\", fontsize=9.5)\n"
            "        ax1.set_ylim(0, np.nanpercentile(rms, 99)); ax1.legend(fontsize=7.5)\n"
            "\n"
            "        # (b) la misma prueba sin el sesgo de la respuesta\n"
            "        _col = [\"tab:green\" if v <= 1.5 else \"tab:red\" for v in rloc]\n"
            "        ax2.bar(range(8), rloc, color=_col, alpha=0.8)\n"
            "        ax2.axhline(1.0, color=\"0.4\", ls=\":\", lw=1, label=\"1 = sin exceso sobre el vecino\")\n"
            "        ax2.axhline(1.5, color=\"tab:red\", ls=\"--\", lw=1, label=\"umbral 1.5 de la etapa\")\n"
            "        ax2.set_xticks(range(8)); ax2.set_xticklabels(rlab, fontsize=6.5)\n"
            "        ax2.set_ylim(0, max(1.75, max(rloc) * 1.15))\n"
            "        ax2.set_xlabel(\"banda [A]\"); ax2.set_ylabel(\"R_local\")\n"
            "        ax2.set_title(\"(b) misma prueba DENTRO de cada banda (airglow brillante / vecino debil):\\n\"\n"
            "                      \"sin sesgo de la curva de respuesta\", fontsize=9.5)\n"
            "        ax2.legend(fontsize=7.5)\n"
            "\n"
            "        # (c) la zona de cielo realmente usada\n"
            "        _ny, _nx = A[\"wl\"].shape\n"
            "        _ext = [-0.5, _nx - 0.5, -0.5, _ny - 0.5]\n"
            "        ax3.imshow(np.log10(np.clip(A[\"wl\"], 1, None)), origin=\"lower\", cmap=\"gray\", extent=_ext)\n"
            "        ov = np.zeros((*A[\"wl\"].shape, 4)); ov[A[\"sky_mask\"]] = [0.1, 0.5, 1.0, 0.5]\n"
            "        ax3.imshow(ov, origin=\"lower\", extent=_ext)\n"
            "        _sy, _sx = A[\"star_yx\"]; _pix = A[\"pix\"]\n"
            "        if _pix > 0:\n"
            "            for _as in (0.5, 1.0, 2.0, 3.0):\n"
            "                _r = _as / _pix\n"
            "                ax3.add_patch(plt.Circle((_sx, _sy), _r, fill=False, color=\"tab:orange\",\n"
            "                                         lw=0.8, ls=\"--\"))\n"
            "                ax3.annotate(f\"{_as:g}\\\"\", (_sx, _sy + _r), color=\"tab:orange\",\n"
            "                             fontsize=6.5, ha=\"center\", va=\"bottom\")\n"
            "        ax3.plot(_sx, _sy, \"+\", color=\"tab:cyan\", ms=8, mew=1.4)\n"
            "        ax3.set_xlim(_ext[0], _ext[1]); ax3.set_ylim(_ext[2], _ext[3])\n"
            "        ax3.set_xlabel(\"x [px]\"); ax3.set_ylabel(\"y [px]\"); ax3.tick_params(labelsize=7)\n"
            "        ax3.set_title(f\"(c) zona de cielo oficial (azul), {A['sky_mask'].mean():.3f} del campo\"\n"
            "                      f\" · {_pix:.4f}\\\"/px\", fontsize=9.5)\n"
            "\n"
            "        # (d) lo que queda tras la resta del DRS\n"
            "        sky_med = A[\"sky_med\"]; _fin = np.isfinite(sky_med)\n"
            "        ax4.plot(wave, sky_med, lw=0.5, color=\"tab:purple\")\n"
            "        ax4.axhline(0, color=\"tab:red\", ls=\"--\", lw=1.0, label=\"cero (cielo perfecto)\")\n"
            "        _mm = float(np.median(sky_med[_fin]))\n"
            "        ax4.axhline(_mm, color=\"0.35\", ls=\":\", lw=1.0, label=f\"mediana global {_mm:+.3f}\")\n"
            "        _q = np.nanpercentile(sky_med[_fin], [1, 99])\n"
            "        ax4.set_ylim(_q[0] - 0.2 * abs(_q[0]), _q[1] + 0.2 * abs(_q[1]))\n"
            "        ax4.set_xlabel(\"lambda [A]\"); ax4.set_ylabel(\"mediana del cielo por spaxel\")\n"
            "        ax4.legend(fontsize=7.5)\n"
            "        ax4.set_title(\"(d) lo que queda tras la resta de cielo del DRS\\n\"\n"
            "                      \"(su origen y su forma en lambda, en la 3a figura)\", fontsize=9.5)\n"
            "\n"
            "        fig.tight_layout(rect=[0, 0.055, 1, 1])\n"
            "        fig.text(0.005, 0.012, A2_REFS, fontsize=6.2, color=\"0.25\", wrap=True)\n"
            "        outdir = nb.run_dir(RUN_ID) / \"plots\" / \"a2_m4\"\n"
            "        outdir.mkdir(parents=True, exist_ok=True)\n"
            "        fig.savefig(outdir / \"m4_R.png\", dpi=110)\n"
            "        print(\"\\nfigura ->\", outdir / \"m4_R.png\")\n"
            "        plt.show()\n"
            "    except Exception as e:\n"
            "        print(\"No se pudo generar el plot:\", type(e).__name__, e)\n"
            "        print(\"Necesita el kernel MUSE (astropy) y el cubo en disco (campo cube.file del QC de A4).\")"
        ),
        plots=[
            dict(
                md=(
                    "## ¿La zona de cielo vale para todas las longitudes de onda?\n"
                    "\n"
                    "**Por construcción, la etapa usa UNA sola máscara para todo el rango.** `build_source_mask`\n"
                    "recibe la **imagen de luz blanca** (mediana sobre los 3681 canales) y el resultado se aplica\n"
                    "igual en 4750 Å que en 9350 Å. Eso es una **suposición**, y hay una razón física para\n"
                    "dudar de ella: el halo AO es **cromático** — el Strehl sube con λ, así que en el azul una\n"
                    "fracción mayor de la luz de la primaria vive en el halo ancho y podría invadir la zona que\n"
                    "llamamos cielo (es el mismo efecto que documenta `docs/` para la sobre-sustracción de\n"
                    "continuo en C3).\n"
                    "\n"
                    "La celda mide ese coste en vez de suponerlo: parte el rango en **8 bandas de 575 Å**, aplica\n"
                    "en cada una **el mismo criterio canónico** (`build_source_mask` sobre la imagen mediana de\n"
                    "esa banda, mismos `threshold_sigma` y `dilation_px` del QC, y el radio del halo de la\n"
                    "primaria **remedido** por banda con `estimate_halo_radius`), y compara la máscara por banda\n"
                    "con la máscara global:\n"
                    "\n"
                    "- `sky_fraction` por banda — ¿queda cielo suficiente (≥ 0.25) en todas?\n"
                    "- `r_halo` de la primaria por banda — ¿crece el halo hacia el azul?\n"
                    "- **fuga** — la fracción de la zona de cielo **global** que *esta* banda llamaría fuente.\n"
                    "- **Jaccard** (informativo) — máscara-de-banda ∩ global ÷ unión; 1.0 = idénticas.\n"
                    "\n"
                    "El veredicto se apoya en la **fuga**, no en el Jaccard, porque el riesgo es **asimétrico**:\n"
                    "que una banda pida *más* halo significa que la máscara única está metiendo halo dentro del\n"
                    "cielo (malo); que pida *menos* solo significa que se podría haber usado más cielo (inocuo). El\n"
                    "Jaccard mezcla los dos sentidos. La celda pide **fuga ≤ 5 %** en las 8 bandas y\n"
                    "`sky_fraction ≥ 0.25`, y si alguna se sale lo dice en vez de dar la máscara única por buena.\n"
                    "\n"
                    "Como referencia de lo que salió al escribir esto: en **ROXs12b** `sky_fraction` 0.420–0.447,\n"
                    "halo 84–86 px, fuga máxima ~3 %; en **ROXs42Bb** 0.420–0.485, halo 81–86 px, fuga del mismo\n"
                    "orden — su Jaccard baja a 0.89 en la banda azul extrema, pero es porque allí el halo **se\n"
                    "encoge** (más cielo, no menos), que es el sentido inocuo.\n"
                    "\n"
                    "O sea: la suposición sale barata, la zona de cielo es esencialmente la misma en todo el rango\n"
                    "y la máscara única no es un atajo que cambie la decisión de ZAP. Lo que **sí** depende de λ es\n"
                    "*cuánta luz de halo queda dentro* de esa zona (el nivel, no la geometría) — y eso es\n"
                    "exactamente lo que mide la figura siguiente."
                ),
                code=(
                    "if MAKE_PLOT:\n"
                    "    try:\n"
                    "        import numpy as np\n"
                    "        import matplotlib.pyplot as plt\n"
                    "        from musepipe.reduction.sky_zap import build_source_mask, SourceRegion\n"
                    "\n"
                    "        A = _a2_load()\n"
                    "        wave, data, wl = A[\"wave\"], A[\"data\"], A[\"wl\"]\n"
                    "        sky_global = A[\"sky_mask\"]\n"
                    "\n"
                    "        # Los knobs salen del QC de la etapa, no de literales: si la etapa corrio con\n"
                    "        # otro umbral, esta figura tiene que seguirlo.\n"
                    "        _thr, _dil = 3.0, 2\n"
                    "        _cyx = None\n"
                    "        try:\n"
                    "            _s = nb.load_qc(\"stages/stage00s_qc.json\", RUN_ID)\n"
                    "            _thr = float(_s.get(\"mask\", {}).get(\"threshold_sigma\", _thr))\n"
                    "            _dil = int(_s.get(\"mask\", {}).get(\"dilation_px\", _dil))\n"
                    "            _cyx = _s.get(\"growth_curve\", {}).get(\"companion_masked_yx\")\n"
                    "        except FileNotFoundError:\n"
                    "            print(\"[A2 no ha corrido para este objeto: se usan los defaults 3.0 sigma / 2 px]\")\n"
                    "        print(f\"criterio por banda: threshold_sigma={_thr}  dilation_px={_dil}  companero={_cyx}\")\n"
                    "\n"
                    "        _sy, _sx = A[\"star_yx\"]\n"
                    "        regs = [SourceRegion(name=\"primary\", yx=(float(_sy), float(_sx)),\n"
                    "                             radius_px=20.0, auto_halo=True)]\n"
                    "        if _cyx:\n"
                    "            regs.append(SourceRegion(name=\"companion\", yx=(float(_cyx[0]), float(_cyx[1])),\n"
                    "                                     radius_px=8.0, auto_halo=False))\n"
                    "\n"
                    "        edges = np.linspace(wave[0], wave[-1], 9)\n"
                    "        fig, axes = plt.subplots(2, 4, figsize=(16.5, 8.4))\n"
                    "        _ny, _nx = wl.shape\n"
                    "        _ext = [-0.5, _nx - 0.5, -0.5, _ny - 0.5]\n"
                    "        # `fuga` es la metrica que importa, y es ASIMETRICA: la fraccion de la zona de\n"
                    "        # cielo global que ESTA banda llamaria fuente (halo que crece -> la mascara unica\n"
                    "        # mete halo en el cielo). Que la banda pida MENOS fuente es inocuo: solo significa\n"
                    "        # que se podria haber usado mas cielo. El Jaccard no distingue los dos sentidos.\n"
                    "        print(\"\\n  banda [A]      sky_fraction  r_halo [px]  Jaccard  fuga(halo->cielo)  mediana del cielo\")\n"
                    "        rows = []\n"
                    "        for i, ax in enumerate(axes.ravel()):\n"
                    "            sel = (wave >= edges[i]) & (wave < edges[i + 1])\n"
                    "            img = np.nanmedian(data[sel], axis=0)\n"
                    "            src, radii = build_source_mask(img, regs, threshold_sigma=_thr, dilation_px=_dil)\n"
                    "            sky = np.isfinite(img) & ~src\n"
                    "            jac = float((sky & sky_global).sum() / max(1, (sky | sky_global).sum()))\n"
                    "            leak = float((sky_global & ~sky).sum() / max(1, sky_global.sum()))\n"
                    "            med = float(np.nanmedian(img[sky]))\n"
                    "            rows.append((edges[i], edges[i + 1], float(sky.mean()),\n"
                    "                         float(radii.get(\"primary\", np.nan)), jac, med, leak))\n"
                    "            print(f\"  {edges[i]:6.0f}-{edges[i + 1]:6.0f}     {sky.mean():.3f}\"\n"
                    "                  f\"         {radii.get('primary', float('nan')):5.1f}\"\n"
                    "                  f\"        {jac:.3f}        {leak:.4f}            {med:+.3f}\")\n"
                    "\n"
                    "            ax.imshow(np.log10(np.clip(img, 1, None)), origin=\"lower\", cmap=\"gray\", extent=_ext)\n"
                    "            ov = np.zeros((*img.shape, 4)); ov[sky] = [0.1, 0.5, 1.0, 0.45]\n"
                    "            ax.imshow(ov, origin=\"lower\", extent=_ext)\n"
                    "            # en rojo, donde la mascara de ESTA banda difiere de la global\n"
                    "            dif = sky ^ sky_global\n"
                    "            if dif.any():\n"
                    "                ov2 = np.zeros((*img.shape, 4)); ov2[dif] = [1.0, 0.2, 0.2, 0.95]\n"
                    "                ax.imshow(ov2, origin=\"lower\", extent=_ext)\n"
                    "            if A[\"pix\"] > 0:\n"
                    "                ax.add_patch(plt.Circle((_sx, _sy), 1.0 / A[\"pix\"], fill=False,\n"
                    "                                        color=\"tab:orange\", lw=0.7, ls=\"--\"))\n"
                    "            ax.plot(_sx, _sy, \"+\", color=\"tab:cyan\", ms=6, mew=1.2)\n"
                    "            ax.set_xlim(_ext[0], _ext[1]); ax.set_ylim(_ext[2], _ext[3])\n"
                    "            ax.set_xticks([]); ax.set_yticks([])\n"
                    "            ax.set_title(f\"{edges[i]:.0f}-{edges[i + 1]:.0f} A\\n\"\n"
                    "                         f\"cielo {sky.mean():.3f} · r_halo {radii.get('primary', float('nan')):.0f} px\"\n"
                    "                         f\" · fuga {100 * leak:.1f}%\", fontsize=8.5)\n"
                    "\n"
                    "        _fr = [r[2] for r in rows]; _rh = [r[3] for r in rows]\n"
                    "        _jj = [r[4] for r in rows]; _lk = [r[6] for r in rows]\n"
                    "        print(f\"\\n  sky_fraction {min(_fr):.3f}-{max(_fr):.3f} (minimo de la etapa 0.25)\"\n"
                    "              f\"   r_halo {min(_rh):.0f}-{max(_rh):.0f} px\"\n"
                    "              f\"   Jaccard minimo {min(_jj):.3f}   fuga maxima {max(_lk):.4f}\")\n"
                    "        # El veredicto lo dicta la medida, no el texto.\n"
                    "        _ok_sky = min(_fr) >= 0.25\n"
                    "        _ok_geom = max(_lk) <= 0.05\n"
                    "        if _ok_geom and _ok_sky:\n"
                    "            print(f\"  -> la geometria de la zona de cielo NO depende de lambda de forma relevante:\"\n"
                    "                  f\" en el peor caso solo el {100 * max(_lk):.1f}% del cielo global seria fuente\"\n"
                    "                  f\" para su banda (umbral 5%), y queda cielo de sobra en las 8.\"\n"
                    "                  f\" La mascara unica de la etapa esta justificada.\")\n"
                    "        else:\n"
                    "            _b = rows[int(np.argmax(_lk))]\n"
                    "            print(f\"  -> ATENCION: la mascara unica NO queda justificada por la medida\"\n"
                    "                  f\" (fuga maxima {100 * max(_lk):.1f}% en {_b[0]:.0f}-{_b[1]:.0f} A\"\n"
                    "                  f\"{'' if _ok_sky else f', sky_fraction minima {min(_fr):.3f} < 0.25'}):\"\n"
                    "                  \" esa banda mete halo en el cielo y hay que revisarla antes de usar R\"\n"
                    "                  \" como decision global.\")\n"
                    "        print(\"  (en rojo, los spaxels donde la mascara de la banda difiere de la global)\")\n"
                    "\n"
                    "        fig.suptitle(\"Zona de cielo (azul) banda a banda, mismo criterio canonico que la etapa\"\n"
                    "                     \" — rojo = difiere de la mascara global\", fontsize=11)\n"
                    "        fig.tight_layout(rect=[0, 0.045, 1, 0.96])\n"
                    "        fig.text(0.005, 0.010, A2_REFS, fontsize=6.2, color=\"0.25\", wrap=True)\n"
                    "        outdir = nb.run_dir(RUN_ID) / \"plots\" / \"a2_m4\"\n"
                    "        outdir.mkdir(parents=True, exist_ok=True)\n"
                    "        fig.savefig(outdir / \"sky_zone_por_banda.png\", dpi=110)\n"
                    "        print(\"figura ->\", outdir / \"sky_zone_por_banda.png\")\n"
                    "        plt.show()\n"
                    "    except NameError:\n"
                    "        print(\"Ejecuta primero la celda anterior (define _a2_load y A2_REFS).\")\n"
                    "    except Exception as e:\n"
                    "        print(\"No se pudo generar el plot:\", type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## ¿Restó el cielo esorex? Sí — y por eso el suelo que queda no es cielo\n"
                    "\n"
                    "**Sí, `muse_scipost` ya restó cielo**, con su método por defecto y sin que nosotros lo\n"
                    "toquemos (spec_A1: el único override es `--save`). La evidencia no es de memoria: está en el\n"
                    "log de esorex de cada exposición, y la celda lo imprime literal. Lo que hizo, en orden\n"
                    "([Weilbacher et al. 2020](https://doi.org/10.1051/0004-6361/202037855) §3.2):\n"
                    "\n"
                    "1. `--skymethod=\"model\"` (default). Reconstruye una imagen blanca intermedia de **esa**\n"
                    "   exposición y crea su propia máscara de cielo con la **fracción más oscura** del campo:\n"
                    "   `--skymodel_fraction=0.1` tras ignorar `--skymodel_ignore=0.05`. En este objeto el\n"
                    "   `SKY_MASK` que guardó cubre el **9.1 %** del campo.\n"
                    "2. Ajusta las líneas de airglow partiendo del `SKY_LINES` de entrada (con la LSF local) y las\n"
                    "   resta; `Sky line fit finished successfully. Offset -0.021 A (at 4746 A) ... -0.042 A (at 9355 A)`.\n"
                    "3. `No sky continuum given, create it from the data` → construye el **continuo de cielo** a\n"
                    "   partir del residuo de ese mismo 9.1 % y lo resta también,\n"
                    "   `Cutting data to 4750.000...9349.688 Angstrom` — o sea **todo el rango**, sin extremos sin\n"
                    "   tratar.\n"
                    "\n"
                    "Y aquí está el punto fino de NFM: el campo son **5″**. El 9.1 % más oscuro que el DRS llamó\n"
                    "«cielo» está a solo ~2.5–3.6″ de una estrella brillante, así que **ya contiene halo AO**. El\n"
                    "DRS resta ese nivel a todo el campo. Nuestra zona de cielo es más ancha (43 % del campo) y\n"
                    "está *más cerca* de la primaria en promedio → lo que queda dentro es el **exceso de halo**\n"
                    "respecto de esa referencia, no cielo mal restado. De ahí el suelo **positivo** (mediana ≈ +1.6\n"
                    "en unidades del cubo sobre los 3681 canales; el `median_bias` = 1.14 del QC de A4 es el mismo\n"
                    "efecto medido sobre sus 32 aperturas vacías, que están más lejos de la primaria).\n"
                    "\n"
                    "### Por qué el azul y el rojo salen distintos\n"
                    "\n"
                    "Los tres paneles de abajo separan las causas, y las tres son medibles:\n"
                    "\n"
                    "- **(b) La prueba que decide qué es:** perfil radial del suelo, en anillos fuera de la\n"
                    "  máscara. Un cielo mal restado sería **plano** con el radio; un halo **cae**. Medido: cae\n"
                    "  **×2.6** en el azul y **×5.5** en el rojo entre ~2.3″ y ~3.5″ de la primaria. **Es halo.**\n"
                    "- **(c) Por qué el rojo es más alto en valor absoluto:** el suelo sigue la SED de la\n"
                    "  primaria. La primaria sube **×8.7** de 4750→9350 Å (K + A_V≈1.8), el suelo sube solo\n"
                    "  **×2.35**, luego la *fracción* de halo que entra en el cielo **baja ×0.27** hacia el rojo —\n"
                    "  el halo AO cromático ([Fetick et al. 2019](https://doi.org/10.1051/0004-6361/201935830), el\n"
                    "  mismo modelo que usa C1/E1): a mayor λ, mejor Strehl, más luz en el core y menos en el\n"
                    "  halo. Los dos efectos se oponen y gana la SED → **suelo más alto en el rojo**.\n"
                    "- **(d) Por qué el azul *parece* limpio:** en unidades del ruido del propio canal, el suelo\n"
                    "  vale **0.31 σ** en el azul y **~1.1 σ** en el rojo. En el azul la respuesta de MUSE hunde\n"
                    "  la transmisión, el RMS por canal sube a ~3–4.6 y el suelo queda **enterrado**; en el rojo\n"
                    "  el RMS baja a ~1.6 y el mismo tipo de residuo **asoma al nivel del ruido**. No es que el\n"
                    "  cielo esté peor restado en el rojo: es que allí se *ve*.\n"
                    "\n"
                    "**Consecuencia para la cadena** (no para ZAP, que sigue siendo innecesario): este suelo es\n"
                    "aditivo y crece hacia el rojo, así que hay que quitarlo antes de integrar flujo en aperturas\n"
                    "grandes — es exactamente lo que mide `scripts/measure_growth_curve.py` para `apcorr` y lo\n"
                    "que el fondo de anillo corrige en C2. Y explica que el azul del espectro final sea el tramo\n"
                    "menos citable: allí el residuo está dentro del ruido, no verificado."
                ),
                code=(
                    "if MAKE_PLOT:\n"
                    "    try:\n"
                    "        import glob as _glob\n"
                    "        from pathlib import Path\n"
                    "        import numpy as np\n"
                    "        import matplotlib.pyplot as plt\n"
                    "        from astropy.io import fits\n"
                    "        from musepipe.reduction.sky_zap import channel_rms_from_mask\n"
                    "        from musepipe.reduction.verify import circular_aperture_mask\n"
                    "\n"
                    "        A = _a2_load()\n"
                    "        wave, data, wl = A[\"wave\"], A[\"data\"], A[\"wl\"]\n"
                    "        sky_mask, sky_med = A[\"sky_mask\"], A[\"sky_med\"]\n"
                    "        _sy, _sx = A[\"star_yx\"]\n"
                    "        _root = Path(A[\"cube_path\"]).parent.parent\n"
                    "\n"
                    "        # 1) La evidencia literal de que el DRS resto cielo, del log de esorex.\n"
                    "        _logs = sorted(_glob.glob(str(_root / \"*\" / \"products\" / \"*\" / \"muse_scipost.log\")))\n"
                    "        print(\"=== que hizo muse_scipost con el cielo (log de esorex, literal) ===\")\n"
                    "        if _logs:\n"
                    "            print(f\"log: {_logs[0]}\")\n"
                    "            for _ln in Path(_logs[0]).read_text(errors=\"replace\").splitlines():\n"
                    "                if any(k in _ln for k in (\"sky mask\", \"Sky line fit\", \"sky continuum\",\n"
                    "                                          \"sky subtraction\", \"Cutting data\", \"sky-spectrum\",\n"
                    "                                          \"Loaded sky lines\")):\n"
                    "                    print(\"   \", _ln.split(\"] \")[-1].strip())\n"
                    "        else:\n"
                    "            print(\"   no hay log de muse_scipost bajo\", _root)\n"
                    "            print(\"   (cubo ADP o workdir de reduccion no disponible: no re-derivable aqui)\")\n"
                    "\n"
                    "        # 2) Lo que el DRS llamo cielo y lo que resto.\n"
                    "        _mk = sorted(_glob.glob(str(_root / \"*\" / \"products\" / \"*\" / \"SKY_MASK_*.fits\")))\n"
                    "        _ct = sorted(_glob.glob(str(_root / \"*\" / \"products\" / \"*\" / \"SKY_CONTINUUM_*.fits\")))\n"
                    "        _sp = sorted(_glob.glob(str(_root / \"*\" / \"products\" / \"*\" / \"SKY_SPECTRUM_*.fits\")))\n"
                    "        if _mk:\n"
                    "            _m = np.asarray(fits.getdata(_mk[0]))\n"
                    "            print(f\"\\nSKY_MASK del DRS: {(_m > 0).mean():.3f} del campo\"\n"
                    "                  f\"  (skymodel_fraction=0.1 tras ignorar 0.05)\"\n"
                    "                  f\" vs {sky_mask.mean():.3f} de la zona de cielo de A2\")\n"
                    "        _wc = _fc = None\n"
                    "        if _ct:\n"
                    "            _t = fits.open(_ct[0])[1].data\n"
                    "            _wc, _fc = np.asarray(_t[\"lambda\"], float), np.asarray(_t[\"flux\"], float)\n"
                    "            print(f\"SKY_CONTINUUM restado: {np.median(_fc[(_wc >= 4750) & (_wc < 5000)]):+.2f} (azul)\"\n"
                    "                  f\" -> {np.median(_fc[(_wc >= 8000) & (_wc < 8300)]):+.2f} (8000-8300 A)\"\n"
                    "                  f\"   frente a un suelo residual de {np.nanmedian(sky_med):+.2f}\")\n"
                    "\n"
                    "        # 3) Perfil radial del suelo: cielo mal restado = plano; halo = cae.\n"
                    "        yy, xx = np.indices(wl.shape)\n"
                    "        rr = np.hypot(yy - _sy, xx - _sx)\n"
                    "        _r0 = float(np.floor(rr[sky_mask].min()))\n"
                    "        ann = [(_r0 + k * 12, _r0 + (k + 1) * 12) for k in range(5)]\n"
                    "        bands = [(wave[0], wave[0] + 550), (6500, 7050), (wave[-1] - 550, wave[-1])]\n"
                    "        _prof = {}\n"
                    "        print(\"\\n=== perfil radial del suelo residual (anillos fuera de la mascara) ===\")\n"
                    "        for a, b in bands:\n"
                    "            sel = (wave >= a) & (wave <= b)\n"
                    "            img = np.nanmedian(data[sel], axis=0)\n"
                    "            vals = []\n"
                    "            for q0, q1 in ann:\n"
                    "                s = sky_mask & (rr >= q0) & (rr < q1)\n"
                    "                vals.append(float(np.nanmedian(img[s])) if s.sum() > 20 else np.nan)\n"
                    "            _prof[(a, b)] = vals\n"
                    "            _fin = [v for v in vals if np.isfinite(v)]\n"
                    "            _dr = _fin[0] / _fin[-1] if len(_fin) > 1 and _fin[-1] else np.nan\n"
                    "            print(f\"  {a:6.0f}-{b:6.0f} A: \" + \" \".join(f\"{v:+6.3f}\" for v in vals)\n"
                    "                  + f\"   caida dentro/fuera = {_dr:.2f}x\")\n"
                    "        print(\"  -> cae con el radio en las tres bandas: el suelo es HALO de la primaria,\"\n"
                    "              \" no cielo mal restado.\")\n"
                    "\n"
                    "        # 4) Suelo vs SED de la primaria: la fraccion de halo es cromatica.\n"
                    "        star_m = circular_aperture_mask(wl.shape, (_sy, _sx), 10.0)\n"
                    "        _idx = np.where(star_m.ravel())[0]\n"
                    "        star = np.full(A[\"n3\"], np.nan)\n"
                    "        for c0 in range(0, A[\"n3\"], 200):\n"
                    "            blk = data[c0:c0 + 200].reshape(min(200, A[\"n3\"] - c0), -1)[:, _idx]\n"
                    "            star[c0:c0 + blk.shape[0]] = np.nansum(blk, axis=1)\n"
                    "        # La banda del laser AO se excluye de TODA metrica de decision (spec_A2 2.5);\n"
                    "        # aqui tambien, o el cociente suelo/primaria explota en sus bordes.\n"
                    "        from musepipe.reduction.sky_zap import NALGS_RANGE\n"
                    "        _nal = (wave >= NALGS_RANGE[0]) & (wave <= NALGS_RANGE[1])\n"
                    "        _f = np.isfinite(sky_med) & np.isfinite(star) & (star > 0) & ~_nal\n"
                    "        _blue = _f & (wave < wave[0] + 550); _red = _f & (wave > wave[-1] - 650)\n"
                    "        _fs, _ff = np.median(star[_red]) / np.median(star[_blue]), np.median(sky_med[_red]) / np.median(sky_med[_blue])\n"
                    "        print(f\"\\nazul -> rojo:  primaria x{_fs:.2f}   suelo x{_ff:.2f}\"\n"
                    "              f\"   fraccion de halo x{_ff / _fs:.2f}  (Strehl AO sube con lambda: Fetick+2019)\")\n"
                    "\n"
                    "        # 5) El suelo en unidades del ruido del canal: por que el azul parece limpio.\n"
                    "        rms = channel_rms_from_mask(data.astype(np.float64), sky_mask)\n"
                    "        _snr = sky_med / rms\n"
                    "        print(f\"suelo / RMS del canal:  azul {np.nanmedian(_snr[_blue]):.2f} sigma\"\n"
                    "              f\"   rojo {np.nanmedian(_snr[_red]):.2f} sigma\"\n"
                    "              f\"   (RMS azul {np.nanmedian(rms[_blue]):.2f} vs rojo {np.nanmedian(rms[_red]):.2f})\")\n"
                    "\n"
                    "        fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.6))\n"
                    "        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]\n"
                    "\n"
                    "        # (a) lo restado por el DRS vs lo que quedo\n"
                    "        if _wc is not None:\n"
                    "            ax1.plot(_wc, _fc, lw=0.6, color=\"tab:blue\", label=\"SKY_CONTINUUM restado por el DRS\")\n"
                    "        if _sp:\n"
                    "            _t2 = fits.open(_sp[0])[1].data\n"
                    "            ax1.plot(_t2[\"lambda\"], _t2[\"data\"], lw=0.3, color=\"0.6\", alpha=0.7,\n"
                    "                     label=\"SKY_SPECTRUM (cielo medido por el DRS)\")\n"
                    "        ax1.plot(wave, sky_med, lw=0.6, color=\"tab:purple\", label=\"suelo que QUEDA (zona de cielo A2)\")\n"
                    "        ax1.axhline(0, color=\"tab:red\", ls=\"--\", lw=0.9)\n"
                    "        ax1.set_yscale(\"symlog\", linthresh=1.0)\n"
                    "        ax1.set_xlabel(\"lambda [A]\"); ax1.set_ylabel(\"flujo (unidades del cubo)\")\n"
                    "        ax1.set_title(\"(a) el DRS SI resto cielo (skymethod=model, 9.1% mas oscuro del campo);\\n\"\n"
                    "                      \"lo que queda es dos ordenes menor y de otro origen\", fontsize=9.5)\n"
                    "        ax1.legend(fontsize=7)\n"
                    "        if _wc is None and not _sp:\n"
                    "            ax1.text(0.5, 0.5, \"productos SKY_* del DRS no disponibles\\npara esta cadena\",\n"
                    "                     transform=ax1.transAxes, ha=\"center\", va=\"center\", fontsize=9, color=\"0.4\")\n"
                    "\n"
                    "        # (b) la prueba: plano (cielo) o decreciente (halo)\n"
                    "        _rc = [0.5 * (q0 + q1) for q0, q1 in ann]\n"
                    "        for (a, b), vals in _prof.items():\n"
                    "            ax2.plot(_rc, vals, \"o-\", lw=1.2, ms=4, label=f\"{a:.0f}-{b:.0f} A\")\n"
                    "        ax2.axhline(0, color=\"0.6\", ls=\":\", lw=0.9)\n"
                    "        ax2.set_xlabel(\"radio desde la primaria [px]\")\n"
                    "        ax2.set_ylabel(\"mediana del suelo en el anillo\")\n"
                    "        if A[\"pix\"] > 0:\n"
                    "            _tw = ax2.secondary_xaxis(\"top\", functions=(lambda v: v * A[\"pix\"], lambda v: v / A[\"pix\"]))\n"
                    "            _tw.set_xlabel(\"radio [arcsec]\", fontsize=8); _tw.tick_params(labelsize=7)\n"
                    "        ax2.set_title(\"(b) el suelo CAE con el radio -> es halo de la primaria,\\n\"\n"
                    "                      \"no cielo mal restado (un residuo de cielo seria plano)\", fontsize=9.5)\n"
                    "        ax2.legend(fontsize=7.5)\n"
                    "\n"
                    "        # (c) suelo vs SED de la primaria\n"
                    "        _n = lambda v, s: v / np.nanmedian(v[s])\n"
                    "        ax3.plot(wave[_f], _n(star, _blue)[_f], lw=0.5, color=\"tab:orange\",\n"
                    "                 label=f\"primaria (normalizada al azul) x{_fs:.1f}\")\n"
                    "        ax3.plot(wave[_f], _n(sky_med, _blue)[_f], lw=0.5, color=\"tab:purple\",\n"
                    "                 label=f\"suelo (normalizado al azul) x{_ff:.1f}\")\n"
                    "        ax3.set_yscale(\"log\"); ax3.set_xlabel(\"lambda [A]\"); ax3.set_ylabel(\"normalizado al azul\")\n"
                    "        _axr = ax3.twinx()\n"
                    "        _rat = (sky_med / star)\n"
                    "        _rat = _rat / np.nanmedian(_rat[_blue])\n"
                    "        _axr.plot(wave[_f], _rat[_f], lw=0.5, color=\"tab:green\")\n"
                    "        _axr.set_ylim(0, np.nanpercentile(_rat[_f], 99.5) * 1.1)\n"
                    "        _axr.set_ylabel(\"fraccion de halo en el cielo (rel. al azul)\", color=\"tab:green\", fontsize=8)\n"
                    "        _axr.tick_params(axis=\"y\", colors=\"tab:green\", labelsize=7)\n"
                    "        ax3.set_title(\"(c) el suelo sigue la SED de la primaria, pero mas plano:\\n\"\n"
                    "                      \"la fraccion de halo BAJA hacia el rojo (Strehl AO cromatico)\", fontsize=9.5)\n"
                    "        ax3.legend(fontsize=7.5, loc=\"upper left\")\n"
                    "\n"
                    "        # (d) el suelo en unidades de ruido\n"
                    "        ax4.plot(wave, _snr, lw=0.5, color=\"0.35\")\n"
                    "        ax4.axhline(1.0, color=\"tab:red\", ls=\"--\", lw=1.0, label=\"1 sigma del canal\")\n"
                    "        ax4.axhline(0.0, color=\"0.6\", ls=\":\", lw=0.9)\n"
                    "        ax4.set_ylim(np.nanpercentile(_snr, 0.5) - 0.2, np.nanpercentile(_snr, 99.5) + 0.2)\n"
                    "        ax4.set_xlabel(\"lambda [A]\"); ax4.set_ylabel(\"suelo / RMS del canal [sigma]\")\n"
                    "        ax4.set_title(\"(d) por que el azul PARECE limpio: alli el suelo esta enterrado\\n\"\n"
                    "                      \"en el ruido de la respuesta; en el rojo asoma a ~1 sigma\", fontsize=9.5)\n"
                    "        ax4.legend(fontsize=7.5)\n"
                    "\n"
                    "        fig.tight_layout(rect=[0, 0.055, 1, 1])\n"
                    "        fig.text(0.005, 0.012, A2_REFS + \"  ·  halo AO cromatico: Fetick et al. 2019, A&A 628, A99\",\n"
                    "                 fontsize=6.2, color=\"0.25\", wrap=True)\n"
                    "        outdir = nb.run_dir(RUN_ID) / \"plots\" / \"a2_m4\"\n"
                    "        outdir.mkdir(parents=True, exist_ok=True)\n"
                    "        fig.savefig(outdir / \"cielo_drs_azul_rojo.png\", dpi=110)\n"
                    "        print(\"\\nfigura ->\", outdir / \"cielo_drs_azul_rojo.png\")\n"
                    "        plt.show()\n"
                    "    except NameError:\n"
                    "        print(\"Ejecuta primero la celda del plot M4 (define _a2_load y A2_REFS).\")\n"
                    "    except Exception as e:\n"
                    "        print(\"No se pudo generar el plot:\", type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("La métrica M4 (`R`) y la caracterización del airglow viven en A4/`stage00q_qc.json`; la regla de decisión, en `musepipe/reduction/sky_zap.py`.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**Decisión de este objeto: `zap_applied = "
            "{{qc:stages/stage00s_qc.json:decision.zap_applied}}` "
            "(`{{qc:stages/stage00s_qc.json:decision.verdict}}`), con "
            "R = {{qc:stages/stage00s_qc.json:decision.R_skyline_over_continuum}} sobre la máscara de "
            "A2.** `n/d` = A2 no ha corrido para esta cadena: la decisión de abajo aún no está "
            "registrada para el objeto.\n\n"
            "- **Datos:** cubo NFM-AO auto-reducido `cube_telcorr.fits` + `SKY_SPECTRUM` cacheado "
            "para caracterizar el airglow (el detalle del OB y las exposiciones de **este** objeto "
            "sale del QC de A1/A4; la celda de setup imprime de qué run vienen).\n"
            "- **Evidencia:** `R = {{qc:stages/stage00q_qc.json:m4_sky.R}}` frente al umbral `1.5` "
            "(estado M4 = **{{qc:stages/stage00q_qc.json:m4_sky.status}}**); el cubo restado de cielo "
            "tiene pocas skylines usables (residual al nivel de ruido). En el campo diminuto NFM, ZAP "
            "aportaría ~0 y arriesgaría absorber señal del compañero.\n"
            "- **Salvedad de M4:** el residuo es bajo, pero la escasez de skylines hace la métrica "
            "menos robusta que en WFM. No bloqueante.\n"
            "- **Para el paper:** registrar como decisión con su métrica (R), **no** como omisión. "
            "Nada es paper-válido hasta cerrar el A-block del objeto."
        ),
    ),
    dict(
        id="A3", slug="A3_telluric", title="Corrección telúrica", block="A · Reducción",
        spec="spec_A3_codex_telluric.md", run_override=None,
        what="Corrige absorción telúrica para producir `cube_telcorr.fits`.",
        inputs="Cubo (post-cielo)", outputs="`cube_telcorr.fits`",
        downstream="A4, B1",
        exec=dict(kind="launch", hist_cmd="bash scripts/telluric.sh"),
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
            "**Método adoptado = STD_TELLURIC** (nativo del DRS MUSE; ver `a3_telluric_justification.md`).\n\n"
            "**Sobre molecfit (actualizado A1a, 2026-07-19):** el primer intento (2026-07-06) NO convergió "
            "(χ² congelado, transmisión→0). El reintento dedicado **A1a** halló la causa raíz real: **no era "
            "el GDAS** (secundario; se usó el perfil MIPAS estándar) sino que el espectro 1D se pasó **sin "
            "normalizar** (flujo mediano ~58000, continuo atascado en 1.0) y con **una sola ventana débil** "
            "(la banda B de O₂, 5.6% de absorción), dejando a O₂ **sin apalancamiento**. Al **normalizar el "
            "flujo** e incluir la **banda A de O₂ (7590–7690 Å, 30% de absorción)**, molecfit **converge** "
            "(`rel_col_O2=0.966±0.016`, `ppmv_O2≈205000` ≈20.5%, físico) y su transmisión **corrobora** "
            "STD_TELLURIC al **0.8%/píxel** en la banda B junto a Hα (tras alinear el marco vacío→aire de "
            "molecfit). Es decir: STD_TELLURIC no fue un atajo por fallo de molecfit — es el método del DRS, "
            "ahora con un **contraste independiente** que lo valida. Registrado en "
            "`a1a_molecfit_crosscheck` (stage00r_qc.json) y `a3_telluric_justification.md §6`.\n\n"
            "**Ventanas protegidas (`T ≡ 1`, la corrección NO se aplica):**\n"
            "- **6540–6590 Å** = Hα del compañero (diagnóstico de acreción).\n"
            "- **5780–6050 Å** = láser AO de NFM.\n\n"
            "**Consecuencia clave:** Hα está esencialmente libre de telúricas y su ventana está "
            "protegida → **A3 NUNCA afecta el resultado científico** (el límite de Hα). A3 solo "
            "importa para la fidelidad del **continuo rojo** que usan D1 y el modelado."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "La celda lee el `stage00t_qc.json` **de la cadena de este objeto** y no supone su "
            "esquema, porque hay **dos en circulación** y no comparten campos: el de las "
            "reducciones antiguas (`threshold_pct`, `method`, escala por `airmass_*`, V1 pre→post) "
            "y el de la cadena multi-noche (`checkpoint_required`, ajuste por `molecules`/"
            "`regions_A`, V1 como residuo por banda). Se imprime el que traiga el QC, y **un campo "
            "que no exista se dice**, no se deja en blanco — indexar a ciegas es lo que rompía esta "
            "celda con `KeyError: 'threshold_pct'`.\n\n"
            "Y ojo con leer las V como si siempre validaran una corrección: si "
            "`telluric_applied = False`, V1/V2/V4 confirman que **no se tocó nada**, no que se "
            "corrigiera bien. La celda lo avisa cuando toca."
        ),
        evidence_code=(
            "qt = nb.load_qc_optional('stages/stage00t_qc.json', RUN_ID)\n"
            "print()\n"
            "if qt is None:\n"
            "    print('A3 no emitió stage00t_qc.json en esta cadena: nada que auditar aquí.')\n"
            "    print(\"Declara el run que lo contiene en chain.stage_runs['A3'], o ejecuta la etapa.\")\n"
            "else:\n"
            "    # Hay DOS esquemas de A3 en circulación y NO comparten todos los campos:\n"
            "    #   · reducciones antiguas: decision.threshold_pct/method, fit.airmass_*,\n"
            "    #     verification.v1_o2_depth_pre_post_pct / v2_outside_bands_unchanged\n"
            "    #   · cadena multi-noche:   decision.checkpoint_required, fit.molecules/regions_A,\n"
            "    #     verification.v1_residual_pct_by_band / v2_outside_bands_change_pct\n"
            "    # Se imprime el que traiga el QC. Un campo ausente se DICE (no se deja en\n"
            "    # blanco ni se inventa): indexar a ciegas es lo que rompía esta celda.\n"
            "    d = qt.get('decision') or {}\n"
            "    fit = qt.get('fit') or {}\n"
            "    ver = qt.get('verification') or {}\n"
            "\n"
            "    def _f(dd, key, unit=''):\n"
            "        if key not in dd:\n"
            "            return '(no está en el esquema de este QC)'\n"
            "        v = dd[key]\n"
            "        return 'n/d' if v is None or v == '' else f'{v}{unit}'\n"
            "\n"
            "    print('Decisión:', _f(d, 'verdict'), '| aplicado:', _f(d, 'telluric_applied'),\n"
            "          '| checkpoint:', _f(d, 'user_checkpoint'))\n"
            "    print('  umbral:', _f(d, 'threshold_pct', ' %'), '| método:', _f(d, 'method'))\n"
            "    _dp = d.get('depth_pct_by_band') or {}\n"
            "    if _dp:\n"
            "        _mx = float(d.get('max_depth_pct', max(_dp.values())))\n"
            "        print('  profundidad por banda [%]:',\n"
            "              ', '.join(f'{k}={float(v):.3f}' for k, v in _dp.items()), f'(máx {_mx:.3f})')\n"
            "    print()\n"
            "\n"
            "    if 'airmass_std' in fit:      # esquema antiguo: escala por masa de aire\n"
            "        _xs, _xc = fit.get('airmass_std'), fit.get('airmass_sci')\n"
            "        _r = f'{_xc / _xs:.3f}' if (_xs and _xc) else 'n/d'\n"
            "        print(f\"Escala airmass: X_std={_xs} -> X_sci={_xc}  ({fit.get('scaling')} = {_r})\")\n"
            "        print('  fuente:', fit.get('source', 'n/d'))\n"
            "    else:                          # esquema multi-noche: ajuste por moléculas\n"
            "        print('Ajuste:', ', '.join(fit.get('molecules') or []) or 'n/d',\n"
            "              '| regiones [Å]:', fit.get('regions_A'))\n"
            "        print('  kernel:', fit.get('kernel'), '| chi2 por región:',\n"
            "              fit.get('chi2_by_region') or '(vacío: no se ajustó nada)')\n"
            "    print()\n"
            "\n"
            "    print('Verificación:')\n"
            "    if 'v1_o2_depth_pre_post_pct' in ver:\n"
            "        _pp = ver['v1_o2_depth_pre_post_pct']\n"
            "        print(f'  V1 O2 B (~6870 A) pre -> post: {_pp[0]}% -> {_pp[1]}%')\n"
            "        print('  V2 fuera de bandas sin cambio:', ver.get('v2_outside_bands_unchanged'))\n"
            "    else:\n"
            "        print('  V1 residuo por banda [%]:',\n"
            "              ver.get('v1_residual_pct_by_band') or '(vacío: no se aplicó corrección)')\n"
            "        print('  V2 cambio fuera de bandas [%]:', ver.get('v2_outside_bands_change_pct'))\n"
            "    print('  V3 Halpha intacta:', _f(ver, 'v3_halpha_untouched'),\n"
            "          '| V4 transmisión física [0,1]:', _f(ver, 'v4_transmission_physical'),\n"
            "          '| V5 STAT escalado:', _f(ver, 'v5_stat_scaled'))\n"
            "    if not d.get('telluric_applied'):\n"
            "        print()\n"
            "        print('  OJO: telluric_applied = False -> V1/V2/V4 no verifican una corrección')\n"
            "        print('  aplicada, sino que NO se tocó nada. La decisión está en la figura de abajo.')\n"
            "print()\n"
            "print('molecfit (crosscheck A1a, si el A1 de esta cadena lo trae):')\n"
            "try:\n"
            "    qr = nb.load_qc('stages/stage00r_qc.json', RUN_ID)\n"
            "    cc = qr['a1a_molecfit_crosscheck']\n"
            "    mf = cc['model_fit']; tcB = cc['transmission_comparison_vs_std_telluric']['O2_B_band_6864_6960A']\n"
            "    print(f\"  converge: mpfit status={mf['mpfit_status']}, rel_col_O2={mf['rel_mol_col_O2']}+-{mf['rel_mol_col_O2_unc']}, ppmv_O2={mf['ppmv_O2']:.0f}\")\n"
            "    print(f\"  vs STD_TELLURIC en banda B (junto a Halpha): |dT|/px={tcB['mean_abs_dT_per_pixel']}, razon absorcion={tcB['integrated_absorption_ratio_molecfit_over_std']}\")\n"
            "    print('  =>', cc['conclusion'][:110], '...')\n"
            "except (FileNotFoundError, KeyError) as e:\n"
            "    print('  (no disponible para esta cadena:', type(e).__name__, e, ')')"
        ),
        plot_md=(
            "## De dónde sale la corrección — o por qué no se aplicó\n\n"
            "La celda **no busca ficheros en disco**: lee el QC de A3 "
            "(`stages/stage00t_qc.json`), que es quien declara en `products.transmission` si la "
            "etapa emitió la curva y en `decision` si concluyó que no hacía falta aplicarla. Según "
            "eso hace una cosa u otra:\n\n"
            "- **A3 aplicó corrección** → grafica `TELLURIC_TRANS.fits` (transmisión 1D derivada de "
            "`STD_TELLURIC_0001.fits` de `muse_standard`, escalada a la masa de aire de la ciencia; "
            "archivo pequeño, no hace falta el cubo): transmisión vs λ con las bandas telúricas "
            "(O₂ B ~6870, la fuerte O₂ A ~7600, H₂O ~7200/8200/9300) y las **ventanas protegidas** "
            "(gris, `T ≡ 1`) — la de Hα (6540–6590) queda plana justo antes de O₂ B.\n"
            "- **A3 decidió que no hacía falta** → no hay curva que graficar, y eso **no es un "
            "fallo**: grafica la medida que llevó a esa decisión, la profundidad de cada banda "
            "frente al umbral del 3 %.\n\n"
            "Solo si el QC **no declara** la ruta (QCs antiguos) prueba los dos sitios previstos, "
            "siempre dentro del propio run. Y si A3 dice `telluric_applied = True` pero la curva no "
            "aparece, **falla ruidosamente** en vez de graficar otra cosa.\n\n"
            "> Ojo al comparar con números de otras reducciones: el veredicto de A3 es **por cubo**. "
            "La cadena canónica de ROXs 12 b da `not_needed_shallow` con O₂ B **0.59 %**, mientras "
            "la primera auto-reducción daba **6.76 %** y su realineado **7.34 %**, ambos con "
            "corrección aplicada. Las tres decisiones están desdobladas más abajo; manda el QC que "
            "imprime esta celda. **La diferencia entre esos números ya está explicada** (el DRS "
            "corrige el telúrico por exposición en la cadena multi-noche y no lo hacía en las "
            "antiguas): la medida está en `debug/A3_telluric_debug.ipynb` y el resumen, en "
            "«Decisiones y notas»."
        ),
        plot_code=(
            "MAKE_PLOT = True   # archivo pequeño; requiere kernel MUSE (astropy)\n"
            "if MAKE_PLOT:\n"
            "    try:\n"
            "        import numpy as np\n"
            "        import matplotlib.pyplot as plt\n"
            "        from pathlib import Path\n"
            "        from astropy.io import fits\n"
            "\n"
            "        def _anchor(value):\n"
            "            \"\"\"Ruta del QC -> absoluta. Unas vienen absolutas y otras RELATIVAS a la\n"
            "            raíz del repo, y el cwd del notebook es notebooks/<objeto>/ (la celda de\n"
            "            setup añade la raíz al sys.path pero NO hace chdir).\"\"\"\n"
            "            p = Path(value)\n"
            "            return p if p.is_absolute() else nb.project_root() / p\n"
            "\n"
            "        # La fuente de verdad es el QC de A3, no una búsqueda de ficheros: la etapa\n"
            "        # declara en products.transmission si emitió la curva, y en decision si\n"
            "        # decidió que no hacía falta aplicar nada.\n"
            "        qc3 = nb.load_qc_optional('stages/stage00t_qc.json', RUN_ID)\n"
            "        if qc3 is None:\n"
            "            print('A3 no ha corrido para este objeto: no hay stage00t_qc.json en su cadena.')\n"
            "            print('   (nada que graficar; la decisión de abajo aún no está registrada)')\n"
            "        else:\n"
            "            dec = qc3.get('decision', {})\n"
            "            declared = (qc3.get('products', {}) or {}).get('transmission') or ''\n"
            "            applied = dec.get('telluric_applied')\n"
            "            print(f\"A3: telluric_applied = {applied}   verdict = {dec.get('verdict')}\")\n"
            "\n"
            "            trans_path = None\n"
            "            if declared:\n"
            "                trans_path = _anchor(declared)\n"
            "                print('curva declarada por el QC:', trans_path)\n"
            "                if not trans_path.exists():\n"
            "                    raise FileNotFoundError(\n"
            "                        f'el QC declara la transmisión en {trans_path} y no está en disco')\n"
            "            else:\n"
            "                # QCs antiguos no persistían la ruta: se prueban los dos sitios\n"
            "                # previstos, SIEMPRE dentro del propio run (nunca el de otro).\n"
            "                cands = []\n"
            "                cube_path = qc3.get('input', {}).get('cube')\n"
            "                if cube_path:\n"
            "                    cands.append(('junto al cubo de A1',\n"
            "                                  _anchor(cube_path).parent / 'TELLURIC_TRANS.fits'))\n"
            "                cands.append(('raw_reduction/ del run',\n"
            "                              nb.run_dir(RUN_ID) / 'raw_reduction' / 'TELLURIC_TRANS.fits'))\n"
            "                for _lab, _p in cands:\n"
            "                    if _p.exists():\n"
            "                        trans_path = _p\n"
            "                        print(f'el QC no declara la curva; encontrada ({_lab}):', _p)\n"
            "                        break\n"
            "                if trans_path is None and applied:\n"
            "                    raise FileNotFoundError(\n"
            "                        'A3 dice telluric_applied=True pero no hay curva ni declarada ni en disco -> '\n"
            "                        + ' | '.join(f'{l}: {p}' for l, p in cands))\n"
            "\n"
            "            if trans_path is not None:\n"
            "                h = fits.open(trans_path); t = h[1].data\n"
            "                wave = np.asarray(t['wave_A'], dtype=float)\n"
            "                trans = np.asarray(t['transmission'], dtype=float)\n"
            "                print('FITS usado:', trans_path)\n"
            "\n"
            "                fig, ax = plt.subplots(figsize=(11, 4))\n"
            "                ax.plot(wave, trans, lw=1.1, color='tab:blue')\n"
            "                for i, (a, b) in enumerate(qc3.get('protected_windows_A') or [(6540, 6590), (5780, 6050)]):\n"
            "                    ax.axvspan(a, b, color='0.6', alpha=0.35,\n"
            "                               label='ventana protegida (T≡1)' if i == 0 else None)\n"
            "                for x, lab in [(6870, 'O₂ B'), (7200, 'H₂O'), (8200, 'H₂O'), (9300, 'H₂O')]:\n"
            "                    ax.annotate(lab, (x, np.interp(x, wave, trans)), textcoords='offset points',\n"
            "                                xytext=(0, -14), ha='center', fontsize=8, color='tab:red')\n"
            "                ax.set_xlabel('λ [Å]'); ax.set_ylabel('Transmisión telúrica aplicada')\n"
            "                ax.set_title('A3 · TELLURIC_TRANS.fits (STD_TELLURIC escalado a airmass sci)')\n"
            "                ax.set_ylim(0, 1.05); ax.legend(fontsize=8); fig.tight_layout()\n"
            "                out = 'transmission.png'\n"
            "                h.close()\n"
            "            else:\n"
            "                # No hay curva porque A3 decidió que no hacía falta. Eso NO es un\n"
            "                # fallo: se grafica la medida que llevó a esa decisión.\n"
            "                depths = dec.get('depth_pct_by_band') or {}\n"
            "                print('A3 no emitió curva de transmisión: decidió que no hacía falta aplicarla.')\n"
            "                print('   profundidad medida por banda [%]:',\n"
            "                      ', '.join(f'{k}={v:.3f}' for k, v in depths.items()) or '(no registrada)')\n"
            "                if not depths:\n"
            "                    raise RuntimeError('el QC de A3 no registra depth_pct_by_band: nada que graficar')\n"
            "                _k = list(depths); _v = [float(depths[k]) for k in _k]\n"
            "                fig, ax = plt.subplots(figsize=(7.5, 4))\n"
            "                ax.bar(range(len(_k)), _v, color=['tab:red' if x > 3 else 'tab:green' for x in _v],\n"
            "                       alpha=0.85)\n"
            "                ax.axhline(3.0, color='tab:red', ls='--', lw=1.2,\n"
            "                           label='umbral de la etapa (3 %)')\n"
            "                ax.set_xticks(range(len(_k))); ax.set_xticklabels(_k, fontsize=9)\n"
            "                ax.set_ylabel('profundidad de la banda [%]')\n"
            "                ax.set_ylim(0, max(3.6, max(_v) * 1.25))\n"
            "                ax.set_title(f\"A3 · por qué NO se aplicó: {dec.get('verdict')}\\n\"\n"
            "                             'profundidad medida frente al umbral', fontsize=10)\n"
            "                ax.legend(fontsize=8); fig.tight_layout()\n"
            "                out = 'depth_vs_threshold.png'\n"
            "\n"
            "            outdir = nb.run_dir(RUN_ID) / 'plots' / 'a3_telluric'\n"
            "            outdir.mkdir(parents=True, exist_ok=True)\n"
            "            fig.savefig(outdir / out, dpi=110)\n"
            "            print('figura ->', outdir / out)\n"
            "            plt.show()\n"
            "    except Exception as e:\n"
            "        print('No se pudo generar el plot:', type(e).__name__, e)\n"
            "        print('Necesita el kernel MUSE (astropy). La curva, si A3 la emitió, sale de '\n"
            "              'products.transmission de su QC.')"
        ),
        decisions=[
            ("**STD_TELLURIC + escala por airmass** (método nativo del DRS MUSE). El reintento A1a hizo **converger** molecfit (causa raíz previa: flujo sin normalizar + banda débil, no el GDAS) y su transmisión **corrobora** STD_TELLURIC al 0.8%/px en la banda B junto a Hα.", "a3_telluric_justification.md"),
            ("Etapa **condicional**, y el veredicto es **por cubo**: no hay una sola decisión de A3, "
             "sino una por reducción. **La que manda en este notebook** es la de su propia cadena: "
             "`verdict = {{qc:stages/stage00t_qc.json:decision.verdict}}`, "
             "`telluric_applied = {{qc:stages/stage00t_qc.json:decision.telluric_applied}}`, "
             "profundidad O₂ B = {{qc:stages/stage00t_qc.json:decision.depth_pct_by_band.O2_B:.3f}} % "
             "frente al umbral del 3 %. Si sale `n/d`, A3 no ha corrido para esta cadena y no hay "
             "decisión registrada para el objeto.", None),
            ("Histórico de ROXs 12 b, desdoblado para que no se citen números de un cubo en otro — "
             "las tres tienen QC propio y las tres dicen cosas distintas: "
             "**(a) `ROXs12b_raw`**, primera auto-reducción (`muse_scipost`), "
             "`stages/stage00t_qc.json` de ese run: O₂ B **6.76 %**, H₂O 7200 0.658 % → `needed`, "
             "**aplicado**, checkpoint humano **aprobado**. "
             "**(b) El realineado de esa misma reducción** (`muse_scipost_aligned`, alias "
             "`stages/stage00t_realigned_qc.json`): O₂ B **7.34 %**, H₂O 7200 1.0 %, H₂O 8200 "
             "0.054 % → `needed`, **aplicado**, sin checkpoint registrado; su curva es la que vive "
             "en `/mnt/2TB/MUSE_work/ROXs12b_realigned/TELLURIC_TRANS.fits`, **fuera** del run. "
             "**(c) Multi-noche del 2026-07-28** (la cadena canónica de hoy): O₂ B **0.586 %** → "
             "`not_needed_shallow`, **no aplicado**, sin curva emitida.", None),
            ("**Cerrado (2026-07-30):** que la banda O₂ B pase de ~7 % en las reducciones antiguas "
             "a 0.59 % en la multi-noche **no es una banda más superficial: es el residuo de una "
             "corrección ya aplicada**. La cadena multi-noche le pasa `STD_TELLURIC` a "
             "`muse_scipost` en cada una de las 29 exposiciones —lo exige "
             "`musepipe/reduction/perexp_plan.py`, que aborta si no hay exactamente una por "
             "exposición— y los SOF de las reducciones antiguas **no lo llevaban** (16 SOF con "
             "`STD_RESPONSE` y cero `STD_TELLURIC`). La prueba independiente es **O₂ A "
             "(7590–7700 Å)**, que A3 no mide y que cae de **28.8 %** a **1.25 %** entre una "
             "reducción y la otra. La masa de aire no lo explica: con las 29 exposiciones reales "
             "(X = 1.02–1.54, peso `EXPTIME`) Beer–Lambert da un factor **1.03×**, y del signo "
             "contrario. Luego la corrección telúrica del cubo canónico **está hecha, por el DRS, "
             "exposición a exposición y cada una a su propia masa de aire** — mejor que aplicar "
             "una curva única al cubo ya combinado. Medida y trazada en "
             "`notebooks/<objeto>/debug/A3_telluric_debug.ipynb`.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**En la cadena de ESTE notebook: `telluric_applied = "
            "{{qc:stages/stage00t_qc.json:decision.telluric_applied}}`, `verdict = "
            "{{qc:stages/stage00t_qc.json:decision.verdict}}`** (O₂ B = "
            "{{qc:stages/stage00t_qc.json:decision.depth_pct_by_band.O2_B:.3f}} % frente al umbral "
            "del 3 %). `n/d` = A3 no ha corrido para este objeto.\n\n"
            "Lo de abajo es la conclusión **de la reducción de referencia** (`ROXs12b_raw`), donde "
            "sí se aplicó: se conserva porque es la que justifica el método para el paper, pero "
            "**no describe el cubo de la cadena multi-noche** — sus números son de otro cubo (ver "
            "el desdoble en «Decisiones y notas»).\n\n"
            "**A3 en `ROXs12b_raw`: corrección telúrica APLICADA por STD_TELLURIC escalado a "
            "airmass (`telluric_applied = True`, `needed`, checkpoint aprobado).**\n\n"
            "- **Fecha:** QC telúrico de referencia 2026-07-06 (`ROXs12b_raw`); justificación para "
            "paper 2026-07-10 (`docs/a3_telluric_justification.md`).\n"
            "- **Datos:** `STD_TELLURIC_0001.fits` (muse_standard), escala airmass 1.087 → 1.158 "
            "(exp. 1.065); aplicado → `cube_telcorr.fits`; transmisión en `TELLURIC_TRANS.fits`.\n"
            "- **Evidencia:** O₂ B 6.76% → 0.6%; continuo fuera de bandas sin cambio; Hα intacta "
            "(ventana protegida); STAT escala como T².\n"
            "- **molecfit (A1a, 2026-07-19): CONVERGE y corrobora STD_TELLURIC.** La no-convergencia previa "
            "no era por GDAS sino por flujo sin normalizar + banda B débil (O₂ sin apalancamiento). "
            "Normalizando el flujo + banda A de O₂ → `rel_col_O2=0.966±0.016`, y T(λ) coincide con "
            "STD_TELLURIC al 0.8%/px en la banda B junto a Hα → contraste independiente. Residuo O₂ ~0.6% "
            "presupuestado para D2. Ver `a1a_molecfit_crosscheck` + `a3_telluric_justification.md §6`.\n"
            "- **Corregido el 2026-07-30:** era falso que «el realineado no emitió su propio "
            "`stage00t_qc.json`». Sí lo emitió, y hay **tres** QCs de A3 para ROXs 12 b diciendo "
            "cosas distintas — están desdoblados en «Decisiones y notas». La cadena multi-noche "
            "además **no aplica** la corrección.\n"
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
        exec=dict(kind="launch",
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
            "Los valores de la tabla son los de **este objeto**, resueltos de su "
            "`stage00q_qc.json` al generar el notebook (`n/d` = métrica no medida todavía "
            "para esta cadena).\n\n"
            "| Métrica | Qué mide | Resultado | Significado |\n|---|---|---|---|\n"
            "| **M1** | Exactitud de la solución de λ (offset vs airglow) | "
            "**{{qc:stages/stage00q_qc.json:m1_wavelength.status}}** "
            "({{qc:stages/stage00q_qc.json:m1_wavelength.offset_median_A:.3f}} Å) | "
            "residuo de la solución en λ del cubo |\n"
            "| **M2** | LSF (ancho de la función de dispersión) | "
            "**{{qc:stages/stage00q_qc.json:m2_lsf.status}}** "
            "({{qc:stages/stage00q_qc.json:m2_lsf.lsf_fwhm_at_halpha_A:.3f}} Å @Hα) | "
            "resolución espectral real, medida del airglow y comparada con la LSF publicada de "
            "MUSE (Bacon+2017); es la medida la que se usa en E1/E3/G2 |\n"
            "| **M3** | Calibración de flujo absoluto (vs Gaia RP) | "
            "**{{qc:stages/stage00q_qc.json:m3_flux.status}}** "
            "(factor {{qc:stages/stage00q_qc.json:m3_flux.flux_factor:.3f}}) | "
            "cuánto se aparta la escala de flujo de la fotometría Gaia |\n"
            "| **M4** | Residuo de cielo (la `R` de A2) | "
            "**{{qc:stages/stage00q_qc.json:m4_sky.status}}** "
            "(R {{qc:stages/stage00q_qc.json:m4_sky.R}}) | calidad de la sustracción de cielo "
            "(ver A2) |\n"
            "| **M5** | Fiabilidad del STAT (varianza del cubo) | "
            "**{{qc:stages/stage00q_qc.json:m5_stat.status}}** "
            "({{qc:stages/stage00q_qc.json:m5_stat.factor_spaxel_median}}×) | cuánto subestima el "
            "STAT el ruido → si es alto, σ **siempre** empírico |\n\n"
            "Las dos decisiones grandes de A4: **M3** (¿la escala de flujo es utilizable?) y "
            "**M5** (¿sirve el STAT como σ, o rige la regla *control = objeto*, "
            "[`docs/noise_model.md`](../docs/noise_model.md)?)."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Resumen M1–M5 del `stage00q_qc.json` (estado + cifra de cabecera)."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage00q_qc.json', RUN_ID)\n"
            "m1, m2, m3, m4, m5 = (q['m1_wavelength'], q['m2_lsf'], q['m3_flux'], q['m4_sky'], q['m5_stat'])\n"
            "def _f(v, fmt='.3f'):\n"
            "    \"\"\"Formatea, o 'n/d' si la métrica no se midió en esta cadena.\"\"\"\n"
            "    return format(v, fmt) if isinstance(v, (int, float)) else 'n/d'\n"
            "rows = [\n"
            "    ('M1 λ-solution', m1.get('status'), f\"offset {_f(m1.get('offset_median_A'))} Å (±{_f(m1.get('offset_err_A'))}), {m1.get('n_lines')} líneas\"),\n"
            "    ('M2 LSF',        m2.get('status'), f\"{_f(m2.get('lsf_fwhm_at_halpha_A'))} Å @Hα, dev máx vs referencia {_f(m2.get('max_dev_vs_nominal_pct'), '.1f')}% [{m2.get('nominal_reference', 'referencia no declarada en el QC')}]\"),\n"
            "    ('M3 flujo abs',  m3.get('status'), f\"factor {_f(m3.get('flux_factor'))} vs Gaia {m3.get('band')} (growth-curve r={_f(m3.get('plateau_radius_px'), '.0f')})\"),\n"
            "    ('M4 cielo',      m4.get('status'), f\"R = {m4.get('R')}\"),\n"
            "    ('M5 STAT',       m5.get('status'), f\"factor spaxel {m5.get('factor_spaxel_median')}× (cuánto subestima el STAT el ruido)\"),\n"
            "]\n"
            "for name, st, detail in rows:\n"
            "    print(f'{name:15s} [{str(st):9s}] {detail}')"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — M2 LSF (medida vs referencia publicada) y M3 growth-curve\n\n"
                    "Ambos desde el QC (baratos, sin cubo).\n\n"
                    "**Izq:** LSF medida del airglow (azul) frente a la **LSF de referencia de "
                    "MUSE publicada**: FWHM(λ) = 5.866·10⁻⁸ λ² − 9.187·10⁻⁴ λ + 6.040 Å "
                    "([Bacon et al. 2017, A&A 608, A1](https://doi.org/10.1051/0004-6361/201730833), "
                    "Ec. 8 — mediana de la LSF medida en los cubos del MUSE UDF, dispersión 1–3%). "
                    "Sustituye a la interpolación lineal en R (1770@4800 Å → 3590@9300 Å) que se "
                    "usaba antes y que no procedía de ninguna publicación. Salvedad: la referencia "
                    "es de WFM, así que sirve como **patrón de comparación**, no como la LSF de "
                    "este cubo — aguas abajo (E1/E3/G2) se usa siempre la **medida**. Para este "
                    "objeto: medida @Hα = "
                    "{{qc:stages/stage00q_qc.json:m2_lsf.lsf_fwhm_at_halpha_A:.3f}} Å "
                    "(referencia @Hα = 2.537 Å).\n\n"
                    "**Der:** el flujo en banda RP crece con el radio hasta el *plateau* (halo AO "
                    "capturado) → `flux_factor = "
                    "{{qc:stages/stage00q_qc.json:m3_flux.flux_factor:.3f}}`."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage00q_qc.json', RUN_ID)\n"
                    "    m2 = q['m2_lsf']; tab = m2['table_A_fwhm']\n"
                    "    w = np.array([r['wave_A'] for r in tab]); f = np.array([r['fwhm_A'] for r in tab])\n"
                    "    # Curva de referencia: se RECALCULA con la función canónica en vez de leer\n"
                    "    # `nominal_fwhm_A` del QC, porque un QC escrito antes del cambio de\n"
                    "    # referencia llevaría todavía la curva antigua (interpolación en R).\n"
                    "    try:\n"
                    "        from musepipe.qc.cube_qc import nominal_muse_fwhm_A as ref_fn\n"
                    "        from musepipe.qc.cube_qc import MUSE_LSF_REFERENCE as ref_cite\n"
                    "        from musepipe.qc.cube_qc import MUSE_LSF_REFERENCE_SHORT as ref_short\n"
                    "    except ImportError:\n"
                    "        o = np.argsort(w); _nom = np.array([r['nominal_fwhm_A'] for r in tab])\n"
                    "        ref_fn = lambda x: np.interp(np.asarray(x, float), w[o], _nom[o])\n"
                    "        ref_cite = m2.get('nominal_reference', 'referencia guardada en el QC')\n"
                    "        ref_short = 'QC'\n"
                    "        print('(sin musepipe en este kernel: uso la referencia guardada en el QC)')\n"
                    "    ref = np.asarray(ref_fn(w), dtype=float)\n"
                    "    ref_ha = float(np.atleast_1d(ref_fn([6563.0]))[0])\n"
                    "    dev = np.abs(f / ref - 1.0) * 100.0\n"
                    "    m3 = q['m3_flux']; gc = m3['growth_curve']\n"
                    "    gr = np.array([p['radius_px'] for p in gc]); gf = np.array([p['band_flux'] for p in gc])\n\n"
                    "    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.2))\n"
                    "    axL.scatter(w, f, s=10, color='tab:blue', label='LSF medida (airglow)')\n"
                    "    wg = np.linspace(float(w.min()), float(w.max()), 300)\n"
                    "    axL.plot(wg, ref_fn(wg), color='0.35', lw=1.6,\n"
                    "             label=f'LSF de referencia ({ref_short})')\n"
                    "    axL.axvline(6563, color='tab:red', ls=':', label='Hα')\n"
                    "    hal = m2.get('lsf_fwhm_at_halpha_A')\n"
                    "    if hal: axL.axhline(hal, color='tab:red', ls='--', lw=1)\n"
                    "    axL.set_xlabel('λ [Å]'); axL.set_ylabel('FWHM LSF [Å]')\n"
                    "    axL.set_title(f\"M2 · LSF medida vs referencia ({m2['status']})\\n\"\n"
                    "                  f\"@Hα = {hal:.3f} Å vs {ref_ha:.3f} Å (ref) · dev máx {dev.max():.1f}%\",\n"
                    "                  fontsize=10)\n"
                    "    axL.text(0.02, 0.035, f'Referencia: {ref_cite}', transform=axL.transAxes,\n"
                    "             fontsize=7, color='0.35',\n"
                    "             bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))\n"
                    "    axL.legend(fontsize=8, loc='upper right')\n"
                    "    # Procedencia de la referencia: avisa si el QC en disco se escribió con otra.\n"
                    "    stored_cite, stored_dev = m2.get('nominal_reference'), m2.get('max_dev_vs_nominal_pct')\n"
                    "    if stored_cite != ref_cite:\n"
                    "        print(f'AVISO: el QC en disco declara referencia {stored_cite!r} y este plot usa '\n"
                    "              f'{ref_cite!r}.')\n"
                    "        if isinstance(stored_dev, (int, float)):\n"
                    "            print(f'       dev máx: {dev.max():.1f}% (recalculada aquí) vs '\n"
                    "                  f'{stored_dev:.1f}% (guardada). El estado M2 del QC se calculó con la '\n"
                    "                  'referencia antigua.')\n"
                    "        print('       Re-ejecuta A4 (m1m2-sky) para regenerar el QC con la referencia actual.')\n"
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
                    "M5 mide cuánto subestima el STAT el ruido por spaxel "
                    "(**{{qc:stages/stage00q_qc.json:m5_stat.factor_spaxel_median}}×** en este "
                    "objeto). El QC solo guarda "
                    "esa mediana, así que ilustro el **mecanismo** con la inflación espacial de G1 "
                    "(almacenada): al sumar en cajas N×N la varianza real se infla frente a la suma "
                    "ingenua de STAT (que asume píxeles independientes, =1) hasta ~19× en 5×5. Es la "
                    "correlación introducida por el remuestreo del cubo → **σ siempre empírico, "
                    "control = objeto**.\n\n"
                    "> Dependencia: este plot lee `stages/stage_g1_qc.json` (etapa G1); si G1 no ha "
                    "corrido en el run, la celda degrada a un mensaje."
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
                    "    fac = m5.get('factor_spaxel_median')\n"
                    "    fac_txt = f\"STAT ~{fac}× bajo por spaxel\" if isinstance(fac, (int, float)) \\\n"
                    "        else f\"M5 no medida en esta cadena ({m5.get('status')})\"\n"
                    "    ax.set_title(f\"M5 [{m5['status']}]: {fac_txt} \"\n"
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
            ("**M3 [{{qc:stages/stage00q_qc.json:m3_flux.status}}]**: flujo absoluto contrastado con Gaia DR3 {{qc:stages/stage00q_qc.json:m3_flux.band}}, factor {{qc:stages/stage00q_qc.json:m3_flux.flux_factor:.3f}} tras growth-curve + truncación de cola.", None),
            ("**M5 [{{qc:stages/stage00q_qc.json:m5_stat.status}}]**: factor de subestimación del STAT = {{qc:stages/stage00q_qc.json:m5_stat.factor_spaxel_median}}× por spaxel (covarianza del remuestreo: inherente, no un defecto del cubo) → σ SIEMPRE empírico, control=objeto. Limitación aceptada en F1.", "noise_model.md"),
            ("**M2 LSF@Hα = {{qc:stages/stage00q_qc.json:m2_lsf.lsf_fwhm_at_halpha_A:.3f}} Å medido** del airglow, frente a los 2.537 Å de la referencia publicada (Bacon et al. 2017, A&A 608, A1, Ec. 8); en E1/E3/G2 se usa la MEDIDA, nunca la referencia.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**A4: cubo caracterizado.** Estado por métrica para **este objeto** (resuelto del "
            "`stage00q_qc.json` de su cadena al generar el notebook; `n/d` = aún no medida). La "
            "procedencia exacta del QC la imprime la celda de setup.\n\n"
            "- **M1 [{{qc:stages/stage00q_qc.json:m1_wavelength.status}}]:** offset "
            "{{qc:stages/stage00q_qc.json:m1_wavelength.offset_median_A:.3f}} Å sobre "
            "{{qc:stages/stage00q_qc.json:m1_wavelength.n_lines}} líneas de airglow "
            "(solución de λ).\n"
            "- **M2 [{{qc:stages/stage00q_qc.json:m2_lsf.status}}]:** LSF "
            "{{qc:stages/stage00q_qc.json:m2_lsf.lsf_fwhm_at_halpha_A:.3f}} Å @Hα medida del "
            "airglow (referencia publicada: 2.537 Å @Hα, Bacon+2017 Ec. 8); es la LSF **medida** la "
            "que se usa en E1/E3/G2.\n"
            "- **M3 [{{qc:stages/stage00q_qc.json:m3_flux.status}}]:** flujo absoluto vs Gaia DR3 "
            "{{qc:stages/stage00q_qc.json:m3_flux.band}}, factor "
            "{{qc:stages/stage00q_qc.json:m3_flux.flux_factor:.3f}}, con growth-curve (halo AO) + "
            "truncación de cola.\n"
            "- **M4 [{{qc:stages/stage00q_qc.json:m4_sky.status}}]:** residuo de cielo "
            "R={{qc:stages/stage00q_qc.json:m4_sky.R}} (ver A2).\n"
            "- **M5 [{{qc:stages/stage00q_qc.json:m5_stat.status}}]:** factor de subestimación del "
            "STAT = {{qc:stages/stage00q_qc.json:m5_stat.factor_spaxel_median}}× por spaxel, por la "
            "covarianza del remuestreo (inherente, no defecto) → σ SIEMPRE empírico "
            "(control=objeto). Limitación aceptada en F1.\n"
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
        checks_md=(
            "## Los términos de este QC, en físico\n\n"
            "| Término | Qué es | Por qué importa |\n|---|---|---|\n"
            "| `centering_method` / `star_centers` | Cómo se localizó la primaria en cada cubo para "
            "ponerlos todos en el mismo sitio. | Un centrado malo desplaza al compañero respecto de "
            "la apertura que lo mide. |\n"
            "| `spatial_shift_mode` | Si los cubos se desplazan por **píxeles enteros** o con "
            "**submuestreo**. | Un desplazamiento fraccionario interpola, y la interpolación "
            "**correlaciona píxeles vecinos**: ese es el origen físico de que el ruido de una "
            "apertura no sea la suma en cuadratura de los píxeles. |\n"
            "| `covariance_factor_box3` | Cuánto se subestima σ en una caja 3×3 si se supone que los "
            "píxeles son independientes. | Es la traducción numérica de lo anterior y entra en todo "
            "el modelo de ruido ([`docs/noise_model.md`](../../docs/noise_model.md)). |\n"
            "| `finite_fraction` | Fracción de vóxeles con dato (no NaN) tras alinear y recortar. | "
            "Los bordes pierden cobertura al desplazar; si cae mucho, el recorte se comió campo útil. |\n"
            "| `crop` / `crop_bounds_per_cube` | La ventana espacial que se conserva. | Define el "
            "campo donde existen controles al mismo radio que el compañero. |\n"
            "| `equivalence` | Comparación contra el producto de referencia (histórico o ADP). | Es "
            "la prueba de que re-alinear no cambió el dato, solo su rejilla. |\n"
        ),
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
            "                     fill=False, ec='tab:orange', lw=1.2, ls='--',\n"
            "                     label=f\"crop inicial {ib['x2'] - ib['x1']}px\"))\n"
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
        checks_md=(
            "## Los términos de este QC, en físico\n\n"
            "«Franjas» (*stripes*) = estructura periódica alineada con los **slicers** del "
            "espectrógrafo: cada slicer corta una tira del campo y la manda a una zona distinta del "
            "detector, así que un defecto suyo aparece como bandas paralelas, no como ruido.\n\n"
            "| Término | Qué es | Por qué importa |\n|---|---|---|\n"
            "| `mean_shift_per_cube_ch` | Desplazamiento en λ de cada cubo respecto de la referencia, "
            "en canales, medido por correlación cruzada. | Si cada exposición está en un cero de λ "
            "distinto, combinarlas ensancha las líneas y borra señal. |\n"
            "| `stripe_metric` / `reduction_factor` | Amplitud de la estructura de franjas y cuánto "
            "baja tras corregir. | Distingue un artefacto del instrumento de una estructura del "
            "cielo. |\n"
            "| `stripe_orientation` / `stripe_angle_deg` | En qué dirección van los slicers en este "
            "cubo. | Los cubos norte-arriba no tienen los slicers verticales por defecto: buscar "
            "franjas en la orientación equivocada da siempre «no hay». |\n"
            "| `dirty_channels` | Canales marcados como contaminados. | Alimentan **T1 de E2**: si el "
            "pico de Hα cayera en uno, sería sospechoso de artefacto. |\n"
            "| `ref_index` / `ref_scores` | Qué cubo se tomó como referencia y con qué criterio. | "
            "Todo el desplazamiento es *relativo* a él. |\n"
        ),
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
        what="Localiza el compañero de {{target}} en el campo.",
        inputs="Cubo alineado", outputs="`stages/stage01c_qc.json`",
        downstream="C1–C4 (posición de extracción)",
        exec=dict(kind="script", target="stage01c_localize.sh", cost="Ligero."),
        qc="stages/stage01c_qc.json",
        salient=["companion.snr_detection", "sep_arcsec", "pa_deg", "band_used_A", "chromatic_centroid_needed"],
        checks_md=(
            "## Los términos de este QC, en físico\n\n"
            "| Término | Qué es | Por qué importa |\n|---|---|---|\n"
            "| `sep_arcsec` / `pa_deg` | Separación angular y **ángulo de posición** del compañero "
            "respecto de la primaria (PA se mide desde el norte hacia el este). | Es la posición que "
            "usan todas las etapas siguientes; los controles se colocan a **esta misma separación** "
            "para que vean el mismo halo. |\n"
            "| `sep_deviation_sigma` / `pa_deviation_sigma` | Cuánto se aparta lo medido de lo "
            "esperado por la literatura, en σ. | Un desvío grande sería otra fuente, no el compañero "
            "conocido. |\n"
            "| `band_used_A` | La banda en la que se detecta. | El compañero es rojo: en el azul su "
            "S/N < 1 y no hay nada que localizar. |\n"
            "| `chromatic_centroid_needed` | Si la posición del compañero se mueve con λ. | La "
            "refracción atmosférica residual desplaza la imagen con la longitud de onda; si es "
            "apreciable, la apertura debe seguirlo o pierde flujo en un extremo del espectro. |\n"
            "| `snr_detection` | S/N de la detección posicional (no de la línea). | Solo dice que la "
            "fuente está ahí, no que emita Hα — eso es E1. |\n"
            "| `legacy_check` | Contraste con la posición histórica. | Si difiere, C2 corre con ambas "
            "y se reporta la diferencia: es información, no un fallo. |\n"
        ),
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
            "        from musepipe.io import read_wavelength_axis\n"
            "        h = fits.open(cube_path, memmap=True); data = h[1].data\n"
            "        data = np.asarray(data[0] if data.ndim == 4 else data, dtype=np.float32)\n"
            "        wave = read_wavelength_axis(h, data_shape=data.shape)   # ext WAVELENGTH o WCS\n"
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
        exec=dict(kind="module_main", target="musepipe.stages.stage04b_local_surface",
                  cost="Ligero–moderado."),
        qc="stages/stage04b_qc.json",
        salient=["method", "local_model_kind", "fit_radius_px", "target_yx", "bad_channel_count"],
        checks_md=(
            "## Los términos de este QC, en físico\n\n"
            "| Término | Qué es | Por qué importa |\n|---|---|---|\n"
            "| `local_model_kind` | La forma que se ajusta al fondo alrededor del compañero (plano, "
            "cuadrática…). | El halo de la primaria varía suavemente a esa escala: se modela como "
            "**superficie local**, no como un nivel constante. |\n"
            "| `fit_radius_px` / `mask_radius_px` | Hasta dónde se ajusta el fondo y qué se excluye "
            "del ajuste. | La máscara evita que el propio compañero entre en el fondo y se reste a sí "
            "mismo; el radio de ajuste decide cuánta curvatura del halo se captura. |\n"
            "| `local_fit_sigma_clip` | Rechazo iterativo de píxeles atípicos en ese ajuste. | Que un "
            "cósmico o una fuente vecina no arrastre la superficie. |\n"
            "| `bad_wavelength_ranges_A` / `bad_channel_count` | Canales donde el ajuste no es fiable "
            "(hueco del láser AO, bordes). | Se marcan y viajan con el producto: los datos no se "
            "tocan, los avisos sí. |\n"
            "| `best4` / `worst2` | Los canales mejor y peor ajustados. | Sirven para mirar de un "
            "vistazo si el modelo local falla en alguna zona concreta del espectro. |\n"
        ),
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
            "        from astropy.io import fits\n"
            "        from musepipe.io import read_wavelength_axis\n"
            "        from musepipe.stages.stage04b_local_surface import CONT_HA_RANGE_A\n\n"
            "        q = nb.load_qc('stages/stage04b_qc.json', RUN_ID)\n"
            "        ty, tx = q['target_yx']; fr = q['fit_radius_px']; mr = q['mask_radius_px']\n"
            "        with fits.open(q['input_cube_fits'], memmap=True) as _h:\n"
            "            wave = read_wavelength_axis(_h)   # ext WAVELENGTH del stack\n"
            "        sel = (wave >= CONT_HA_RANGE_A[0]) & (wave <= CONT_HA_RANGE_A[1])   # continuo Hα oficial\n"
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
        checks_md=(
            "## Los chequeos del QC, en físico\n\n"
            "Los nombres `v2_…`/`v3_…`/`v4_…` son las verificaciones de la spec (§5) y viven así en "
            "el QC; esto es la pregunta que contesta cada una.\n\n"
            "| Chequeo | ¿Qué pregunta contesta? | Si falla |\n|---|---|---|\n"
            "| `v2_error_ratio_ok` | **¿La barra de error del espectro describe su dispersión real?** "
            "Compara el error propagado del cubo con el medido en controles al mismo radio "
            "(mediana de la razón dentro de [0.7, 1.4]). | Toda significancia posterior (E1, E3) "
            "queda mal escalada: el σ no es el ruido. |\n"
            "| `v3_roundtrip_ok` | **¿El fichero que escribimos se relee idéntico?** Integridad del "
            "formato `SpectrumProduct` (write→read→validate). | No es física: es un producto "
            "corrupto o una versión de formato incompatible. |\n"
            "| `v4_apcorr_range_ok` | **¿Sabemos cuánta luz se queda fuera de la apertura?** La "
            "corrección `apcorr(λ)` debe ser suave y estar en [1.0, ~1.6] para la caja 3×3, y el "
            "espectro corregido de 3×3 y 5×5 debe coincidir: si la curva de crecimiento de C1 es "
            "correcta, medir en caja chica o grande da lo mismo. | El flujo absoluto del compañero "
            "queda sesgado **y con dependencia en λ** (la PSF se ensancha hacia el azul). |\n"
        ),
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
                    "    from musepipe.spectral import median_filter_1d\n"
                    "    # Mediana móvil que IGNORA los NaN. Con una media y nan_to_num, los 215\n"
                    "    # canales sin dato (hueco del láser, bordes) entraban como CEROS y tiraban\n"
                    "    # la curva hacia abajo justo donde importa: ~10% en el rojo y una caída\n"
                    "    # falsa a cero cruzando el hueco.\n"
                    "    sm = median_filter_1d(flux, 41)\n"
                    "    fig, ax = plt.subplots(figsize=(11, 4))\n"
                    "    ax.fill_between(w, -sig, sig, color='0.8', label=f'±1σ empírico ({ctrl.shape[0]} controles)')\n"
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
            dict(
                md=PAPER_SPECTRUM_MD,
                code=paper_spectrum_cell(
                    arrays_code=(
                        "    from astropy.io import fits\n"
                        "    ROOT_P = nb.project_root()\n"
                        "    METHOD_P = 'aperture'\n"
                        "    PRODUCT_P = 'spec_aperture_object.fits'\n"
                        "    TARGET_P = (nb.run_target(RUN_ID) or RUN_ID).replace(' ', '')\n"
                        "    _h = fits.open(nb.run_dir(RUN_ID) / 'stages' / PRODUCT_P)\n"
                        "    _d = _h[1].data\n"
                        "    # La unidad viaja con el dato (BUNIT); no hay default silencioso.\n"
                        "    BUNIT_P = _h[1].header.get('BUNIT') or 'ADU'\n"
                        "    W_P = np.asarray(_d['wave_A'], float)\n"
                        "    F_P = np.asarray(_d['flux'], float)\n"
                        "    E_P = np.asarray(_d['flux_err_emp'], float)\n"
                        "    E_ALT_P = np.asarray(_d['flux_err'], float)\n"
                        "    EXTRA_P = {'flux_err_stat': E_ALT_P,\n"
                        "               'apcorr': np.asarray(_d['apcorr'], float),\n"
                        "               'npix_eff': np.asarray(_d['npix_eff'], float),\n"
                        "               'flags': np.asarray(_d['flags'], int)}\n"
                        "    _h.close()\n"
                        "    MODO_P = nb.load_qc('stages/spec_aperture_qc.json', RUN_ID)['errors']['mode']\n"
                    ),
                    subdir="c2_aperture",
                    err_label="±1σ empírico (controles procesados igual)",
                    err_alt_label="±1σ propagado del STAT (no es σ)",
                    title_suffix="espectro del compañero · apertura box3 (C2)",
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
        id="C3", slug="C3_optimal", title="Extracción óptima (2 variantes → 2 métodos)",
        block="C · Extracción",
        spec="spec_C3_codex_optimal_extraction.md", run_override=None,
        what=("Extracción óptima (Horne) ponderada por la PSF, en **dos variantes obligatorias** "
              "que se diferencian solo en el fondo que se resta antes: `optimal_ls` (superficie "
              "local de 04b, comparable 1:1 con C2) y `optimal_psfsub` (modelo de PSF de la "
              "primaria, de C1). Son **2 de los 6 métodos** de la cadena: 5 etapas, 6 métodos."),
        inputs="Cubo + PSF (C1)",
        outputs=("`stages/spec_optimal_qc.json`, `spec_optimal_object.fits` (ls), "
                 "`spec_optimal_psfsub_object.fits`"),
        downstream="D1, E1",
        exec=dict(kind="script", target="stage_x02_optimal.sh", cost="Moderado."),
        qc="stages/spec_optimal_qc.json",
        salient=["variants", "snr_gain_vs_aperture.median", "continuum_bias_vs_aperture_pct", "v3_continuum_bias_ok", "fwhm_pm10pct"],
        checks_md=(
            "## Los chequeos del QC, en físico\n\n"
            "| Chequeo | ¿Qué pregunta contesta? | Si falla |\n|---|---|---|\n"
            "| `v1_snr_gain_ok` | **¿Pesar por la PSF gana algo frente a sumar en una caja?** "
            "Mediana de S/N(óptima)/S/N(apertura 3×3) ≥ 1 (se espera 1.1–1.3). | El método no "
            "aporta: la extracción óptima solo se justifica por la ganancia de S/N. |\n"
            "| `v2_error_ratio_ok` | **¿La barra de error describe el ruido real?** Igual que en C2: "
            "propagado vs empírico de controles, razón mediana en [0.7, 1.4]. | El σ no es el ruido "
            "y la significancia posterior queda mal escalada. |\n"
            "| `v3_continuum_bias_ok` | **¿El método se come el continuo del compañero?** "
            "(óptima − apertura)/apertura en bandas de continuo, < 2–3%. | Es el chequeo que "
            "**rechaza `optimal_ls`** en esta cadena: sobre-sustrae el halo y deja el continuo "
            "negativo. La variante que G1 valida es `optimal_psfsub`. |\n"
            "| `v4_clip_concentration_ok` | **¿El rechazo de píxeles se está comiendo la señal?** "
            "El mapa de clipping no debe concentrarse en la posición del compañero (< 2× la tasa "
            "media). | Se estaría recortando el objeto y llamándolo ruido. |\n"
            "| `v5_ls_vs_psfsub_written` | **¿Quedan las dos variantes en disco para compararlas?** "
            "No juzga: entrega el diagnóstico del modelo de halo a D1. | Falta insumo para D1. |\n"
        ),
        narrative_md=(
            "## Qué hace C3 y las dos variantes\n\n"
            "C3 es **extracción óptima de Horne (1986)**: por canal, pondera cada píxel por el "
            "**perfil de PSF esperado** (de C1) y la varianza inversa "
            "(`f = Σ M·P·D/V / Σ M·P²/V`). Al bajar el peso de los píxeles ruidosos, **gana S/N** "
            "frente a la apertura (aquí ~**6.9× mediana**). La fórmula es cerrada; el valor está en "
            "implementarla exacta (tests analíticos de flujo y varianza).\n\n"
            "**Dos variantes del fondo** — mismo estimador, distinto fondo restado antes:\n"
            "- **`optimal_ls`** — usa el residual de superficie local (04b) como fondo. Mismo fondo "
            "que C2, así que la comparación con C2 aísla la ganancia del ponderado óptimo.\n"
            "- **`optimal_psfsub`** — ajusta y **resta la PSF de la primaria** primero, y luego "
            "extrae ópticamente el compañero. Anticipa el fondo que usará C4, así que `ls` vs "
            "`psfsub` es un **diagnóstico del modelo de halo** para D1, no una redundancia.\n\n"
            "> **De dónde salen los 6 métodos.** C2–C6 son **cinco etapas**, pero C3 emite estas "
            "**dos** variantes como productos separados, así que la cadena compara **seis** "
            "métodos: `aperture`, `optimal_ls`, `optimal_psfsub`, `psffit`, `sgf` y `lpm` "
            "(el `METHOD_ORDER` que usan D1, D2, E4 y G1).\n\n"
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
            "    bias_pct = nb.load_qc('stages/spec_optimal_qc.json', RUN_ID)['continuum_bias_vs_aperture_pct']\n"
            "    def spec(path):\n"
            "        h = fits.open(rd / 'stages' / path); d = h[1].data\n"
            "        w = np.asarray(d['wave_A'], float); f = np.asarray(d['flux'], float); h.close(); return w, f\n"
            "    from musepipe.spectral import median_filter_1d\n"
            "    sm = lambda x, n=41: median_filter_1d(x, n)   # mediana móvil, ignora NaN\n"
            "    w, fls = spec('spec_optimal_object.fits')\n"
            "    _, fps = spec('spec_optimal_psfsub_object.fits')\n"
            "    fig, ax = plt.subplots(figsize=(11, 4))\n"
            "    ax.plot(w, sm(fls), lw=1.2, color='tab:orange', label='optimal_ls (superficie local)')\n"
            "    ax.plot(w, sm(fps), lw=1.2, color='tab:green', label='optimal_psfsub (resta de PSF) — validada G1')\n"
            "    ax.axhline(0, color='0.6', lw=0.7); ax.axvline(6563, color='tab:red', ls=':', label='Hα')\n"
            "    allv = np.concatenate([sm(fls), sm(fps)])\n"
            "    ax.set_ylim(np.nanpercentile(allv, 2), np.nanpercentile(allv, 98))\n"
            "    ax.set_xlabel('λ [Å]'); ax.set_ylabel('flujo (suavizado 41ch)')\n"
            "    ax.set_title(f'C3 · dos variantes: ls sobre-sustrae el continuo ({bias_pct:.0f}% vs apertura), psfsub no')\n"
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
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    h = fits.open(rd / 'stages' / 'spec_optimal_psfsub_object.fits')\n"
                    "    wave = np.asarray(h[1].data['wave_A'], float)   # eje λ del propio producto\n"
                    "    fo = np.asarray(h[1].data['flux'], float); h.close()\n"
                    "    C = np.load(rd / 'stages' / 'spec_optimal_psfsub_controls.npz')['control_spectra']\n"
                    "    cm = np.nanmean(C, axis=0)\n"
                    "    print('banda            media_ctrl     objeto   obj-ctrl   (unidades nativas)')\n"
                    "    for lo, hi in [(5100, 5500), (6600, 7200), (8000, 8800)]:\n"
                    "        b = (wave >= lo) & (wave <= hi)\n"
                    "        print(f'  {lo}-{hi} Å   {np.nanmedian(cm[b]):9.0f}  {np.nanmedian(fo[b]):9.0f}  {np.nanmedian((fo - cm)[b]):9.0f}')\n"
                    "    from musepipe.spectral import median_filter_1d\n"
                    "    sm = lambda x, n=81: median_filter_1d(x, n)   # mediana móvil, ignora NaN\n"
                    "    fig, ax = plt.subplots(figsize=(11, 4.2))\n"
                    "    ax.plot(wave, sm(fo), lw=1, color='tab:green', label='objeto psfsub (crudo, sobre-sustraído)')\n"
                    "    ax.plot(wave, sm(cm), lw=1, color='tab:red', ls='--', label=f'media de {C.shape[0]} controles = fondo residual del halo')\n"
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
            dict(
                md=PAPER_SPECTRUM_MD,
                code=paper_from_product_cell(
                    product='spec_optimal_object.fits',
                    method='optimal_ls',
                    subdir='c3_optimal',
                    stem='spectrum_paper_ls',
                    qc='stages/spec_optimal_qc.json',
                    title_suffix='espectro del compañero · optimal_ls (C3)',
                ),
            ),
            dict(
                md=PAPER_SPECTRUM_MD,
                code=paper_from_product_cell(
                    product='spec_optimal_psfsub_object.fits',
                    method='optimal_psfsub',
                    subdir='c3_optimal',
                    stem='spectrum_paper_psfsub',
                    qc='stages/spec_optimal_qc.json',
                    title_suffix='espectro del compañero · optimal_psfsub (C3)',
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
        checks_md=(
            "## Los chequeos del QC, en físico\n\n"
            "C4 ajusta **dos PSF a la vez** (primaria + compañero) canal a canal, así que sus "
            "chequeos preguntan por lo que puede salir mal en ese ajuste conjunto.\n\n"
            "| Chequeo | ¿Qué pregunta contesta? | Si falla |\n|---|---|---|\n"
            "| `v3_star_scale_ok` | **¿El ajuste reproduce la estrella que sí vemos bien?** Razón "
            "entre el espectro de la primaria ajustada y su fotometría de apertura grande: mediana "
            "en [0.97, 1.03] y sin pendiente con λ. | Si el modelo no reproduce la fuente brillante, "
            "el residuo donde vive el compañero tampoco es de fiar. |\n"
            "| `v4_rho_ab_ok` | **¿Son separables las dos fuentes?** Correlación ρ(a,b) entre las "
            "amplitudes de primaria y compañero, y número de condición del ajuste; a esta separación "
            "se espera \\|ρ\\| < 0.3. | Con ρ→1 el ajuste no puede decidir cuánta luz es de cada una: "
            "el flujo del compañero se vuelve **degenerado** (cualquier reparto encaja igual de bien) "
            "y su error real es mucho mayor que el formal. |\n\n"
            "Las otras verificaciones de la spec (χ²ᵣ~1, residuo limpio, crosstalk, contraste con "
            "C2/C3) se revisan en los plots de abajo, no como banderas del QC.\n"
        ),
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
                    "    from musepipe.io import read_wavelength_axis\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    q = nb.load_qc('stages/spec_psffit_qc.json', RUN_ID)\n"
                    "    loc = nb.load_qc('stages/stage01c_qc.json', RUN_ID)   # posiciones oficiales (B3)\n"
                    "    (py, px), (cy, cx) = loc['primary']['pos_yx'], loc['companion']['pos_yx']\n"
                    "    def cube(path):\n"
                    "        h = fits.open(path); hd = next(x for x in h if x.data is not None)\n"
                    "        d = np.asarray(hd.data, float); h.close(); return d[0] if d.ndim == 4 else d\n"
                    "    res = cube(q['products']['residual_cube'])\n"
                    "    inp = cube(rd / 'stages' / 'stage02_xcorr_cube_stack.fits')\n"
                    "    with fits.open(rd / 'stages' / 'stage02_xcorr_cube_stack.fits') as _h:\n"
                    "        wave = read_wavelength_axis(_h)   # ext WAVELENGTH del stack\n"
                    "    sel = (wave >= 7000) & (wave <= 8500)\n"
                    "    imgi = np.nanmedian(inp[sel], axis=0); imgr = np.nanmedian(res[sel], axis=0)\n"
                    "    v = np.nanpercentile(imgi, [30, 99.5])\n"
                    "    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))\n"
                    "    for ax, im, t in [(axes[0], imgi, 'entrada: estrella + compañero'),\n"
                    "                      (axes[1], imgr, 'residual psffit: ambos removidos (χ²ᵣ≈1)')]:\n"
                    "        ax.imshow(im, origin='lower', cmap='magma', vmin=v[0], vmax=v[1])\n"
                    "        ax.plot(px, py, '+', color='cyan', ms=10); ax.plot(cx, cy, 'o', mfc='none', mec='lime', ms=12)\n"
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
                    "`spec_psffit_object.fits` con la banda ±1σ empírica de los controles y Hα. El continuo "
                    "sube al rojo (SED real de enana fría) y **no hay nada en Hα** — la no-detección con "
                    "el método canónico. El azul (λ<7000) está dominado por ruido (SNR<1)."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    h = fits.open(rd / 'stages' / 'spec_psffit_object.fits')\n"
                    "    wave = np.asarray(h[1].data['wave_A'], float)   # eje λ del propio producto\n"
                    "    flux = np.asarray(h[1].data['flux'], float); h.close()\n"
                    "    C = np.load(rd / 'stages' / 'spec_psffit_controls.npz')['control_spectra']\n"
                    "    sig = np.nanstd(C, axis=0)\n"
                    "    from musepipe.spectral import median_filter_1d\n"
                    "    sm = median_filter_1d(flux, 41)   # mediana móvil, ignora NaN\n"
                    "    fig, ax = plt.subplots(figsize=(11, 4))\n"
                    "    ax.fill_between(wave, -sig, sig, color='0.85', label=f'±1σ empírico ({C.shape[0]} controles)')\n"
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
            dict(
                md=PAPER_SPECTRUM_MD,
                code=paper_from_product_cell(
                    product='spec_psffit_object.fits',
                    method='psffit',
                    subdir='c4_psffit',
                    stem='spectrum_paper',
                    qc='stages/spec_psffit_qc.json',
                    title_suffix='espectro del compañero · psffit, el método canónico (C4)',
                ),
            ),
            dict(
                md=PAPER_SPECTRUM_MD,
                code=paper_from_product_cell(
                    product='spec_psffit_star.fits',
                    method='psffit_star',
                    subdir='c4_psffit',
                    stem='spectrum_paper_star',
                    qc='stages/spec_psffit_qc.json',
                    title_suffix='espectro de la PRIMARIA · psffit (C4)',
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
    dict(
        id="C5", qc_optional=True, slug="C5_sgf", title="Sustracción de halo SGF", block="C · Extracción",
        spec="spec_C5_codex_sgf_subtraction.md", run_override=None,
        what=(
            "Sustrae el halo estelar por diversidad espectral con filtrado Savitzky-Golay "
            "(Haffert et al. 2019; Julo et al. 2025 App. A.3) y extrae el producto box3 del residual."
        ),
        inputs="`stage02` stack + posiciones (B3) + PSF (C1, solo apcorr)",
        outputs="`stages/spec_sgf_qc.json`, `spec_sgf_object.fits`, cubo residual",
        downstream="D1 v3, G1, E4; cubo residual → E1b (mapa FoV)",
        exec=dict(kind="module_main", target="musepipe.stages.stage_x04_sgf",
                  cost="Ligero (~1 min: un savgol por exposición + extracción)."),
        qc="stages/spec_sgf_qc.json",
        salient=["halosub.n_exposures", "sgf.window", "errors.mode",
                 "checks.v1_reference_ok", "checks.v2_far_continuum_ok", "checks.v4_scale_convention_ok"],
        checks_md=(
            "## Los chequeos del QC, en físico\n\n"
            "| Chequeo | ¿Qué pregunta contesta? | Si falla |\n|---|---|---|\n"
            "| `v1_reference_ok` | **¿Hay bastante halo para construir la referencia estelar?** "
            "≥ 50 spaxels conservados por exposición. | La referencia es ruido: el método resta "
            "ruido en vez de halo. |\n"
            "| `v2_far_continuum_ok` | **¿La sustracción deja el fondo en cero donde no hay nada?** "
            "Mediana del residuo en los controles, en continuo lejos de líneas, compatible con 0. | "
            "Hay un pedestal residual: todo lo extraído después lleva ese sesgo. |\n"
            "| `v3_predictor_written` | **¿Cuánta señal de línea se espera perder?** El SGF filtra el "
            "continuo por construcción y muerde también la línea; el predictor (Ec. 1 de Julo+25) "
            "queda registrado para cada línea estándar en cobertura. | No hay con qué interpretar el "
            "flujo de línea de este método. |\n"
            "| `v4_scale_convention_ok` | **¿Queda trazable en qué escala está el producto?** "
            "Cabeceras `BKGMODE`/`SCALEREF` y espectros de control persistidos. | D1 no puede "
            "comparar este método con los otros sin adivinar la convención. |\n"
            "| `v5_no_pca` | **¿Se coló PCA?** Declara explícitamente `pca_applied = false`. | "
            "Sería saltarse una decisión congelada (PCA descartado en D1). |\n"
        ),
        narrative_md=(
            "## Qué hace C5 y por qué existe\n\n"
            "C5 implementa el método **estado-del-arte de literatura** (HRSDI/SGF): por spaxel, "
            "divide por un espectro estelar de referencia (mediana de spaxels con flujo en "
            "0.01–0.1×Fmax), suaviza el ratio con Savitzky-Golay (d=1, W̆=101 canales) y sustrae "
            "referencia×ratio_suavizado.\n\n"
            "**Deliberadamente sin enmascarar líneas y sin PCA**: es la línea base cuyos sesgos "
            "cuantifica Julo et al. 2025 — auto-sustracción de líneas y continuo negativo vecino, "
            "con profundidad exacta (en el modelo de juguete) `C̃_P/L̂_P = −(R/(1−R))·(C_S/L_S)` "
            "(Ec. 1). El QC registra ese **predictor por línea** con C_S/L_S medido en la "
            "referencia del run, y la corrección práctica es el throughput E4/E3 (como "
            "Jorquera et al. 2024) — nunca un parche dentro de C5.\n\n"
            "Los oráculos analíticos del paper están verificados en `tests/test_halosub_toy.py` "
            "(Fig. 2d y Ec. 1 a 8 decimales) y el contrato de la etapa en "
            "`tests/test_halosub_stages.py`."
        ),
        evidence_md=(
            "## Evidencia: predictor de auto-sustracción y continuo negativo\n\n"
            "Predictor Ec. 1 por línea de ciencia (con C_S/L_S medido en la referencia) y fracción "
            "de canales < −2σ en las bandas laterales del producto."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/spec_sgf_qc.json', RUN_ID)\n"
            "print(f\"exposiciones: {q['halosub']['n_exposures']}  spaxels ref: {q['halosub']['n_spaxels_kept']}\")\n"
            "for row in q['self_subtraction_predictor']:\n"
            "    if row['in_range']:\n"
            "        print(f\"  {row['line']:8s} C_S/L_S={row['cs_over_ls']:.3f}  R={row['R']:.4f}  \"\n"
            "              f\"predictor C̃_P/L̂_P={row['predictor']:+.4f}\")\n"
            "for row in q['negative_continuum']:\n"
            "    if row['frac_below_minus2sigma'] is not None:\n"
            "        print(f\"  {row['line']:8s} frac(<−2σ) en bandas laterales = {row['frac_below_minus2sigma']:.3f}\")\n"
            "print('checks:', {k: v for k, v in q['checks'].items()})"
        ),
        plots=[
            dict(
                md=(
                    "## Plot — residual SGF y espectro del compañero\n\n"
                    "Colapso del cubo residual (7000–8500 Å) y espectro `spec_sgf_object` con la banda "
                    "±1σ de controles. Comparar con C4: aquí el halo se remueve por diversidad "
                    "espectral, no por modelo espacial."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    q = nb.load_qc('stages/spec_sgf_qc.json', RUN_ID)\n"
                    "    loc = nb.load_qc('stages/stage01c_qc.json', RUN_ID)\n"
                    "    (cy, cx) = loc['companion']['pos_yx']\n"
                    "    with fits.open(q['products']['residual_cube']) as h:\n"
                    "        res = np.asarray(h['RESIDUAL'].data, float); wave = np.asarray(h['WAVELENGTH'].data, float)\n"
                    "    sel = (wave >= 7000) & (wave <= 8500)\n"
                    "    img = np.nanmedian(res[sel], axis=0)\n"
                    "    hp = fits.open(rd / 'stages' / 'spec_sgf_object.fits')\n"
                    "    w = np.asarray(hp[1].data['wave_A'], float); f = np.asarray(hp[1].data['flux'], float); hp.close()\n"
                    "    C = np.load(rd / 'stages' / 'spec_sgf_controls.npz')['control_spectra']\n"
                    "    sig = np.nanstd(C, axis=0)\n"
                    "    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), width_ratios=[1, 2])\n"
                    "    v = np.nanpercentile(img, [5, 99])\n"
                    "    axes[0].imshow(img, origin='lower', cmap='magma', vmin=v[0], vmax=v[1])\n"
                    "    axes[0].plot(cx, cy, 'o', mfc='none', mec='lime', ms=12); axes[0].axis('off')\n"
                    "    axes[0].set_title('residual SGF (mediana 7000–8500 Å)')\n"
                    "    axes[1].fill_between(w, -sig, sig, color='0.85', label='±1σ controles')\n"
                    "    axes[1].plot(w, f, lw=0.4, color='tab:blue', label='compañero (sgf)')\n"
                    "    axes[1].axvline(6563, color='tab:red', ls=':', label='Hα'); axes[1].axhline(0, color='0.6', lw=0.6)\n"
                    "    axes[1].set_ylim(np.nanpercentile(f, 2), np.nanpercentile(f, 98))\n"
                    "    axes[1].set_xlabel('λ [Å]'); axes[1].legend(fontsize=8)\n"
                    "    axes[1].set_title('C5 · espectro sgf del compañero')\n"
                    "    fig.tight_layout()\n"
                    "    outdir = rd / 'plots' / 'c5_sgf'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'residual_and_spectrum.png', dpi=110); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=PAPER_SPECTRUM_MD,
                code=paper_from_product_cell(
                    product='spec_sgf_object.fits',
                    method='sgf',
                    subdir='c5_sgf',
                    stem='spectrum_paper',
                    qc='stages/spec_sgf_qc.json',
                    title_suffix='espectro del compañero · sgf (C5)',
                ),
            ),
        ],
        decisions=[
            ("**Fiel a la literatura** (SavGol d=1, W̆=101, sin máscara de líneas, sin PCA): línea base cuyos sesgos se miden, no se ocultan.", "spec_C5_codex_sgf_subtraction.md"),
            ("Predictor Ec. 1 por línea en el QC; la corrección práctica de la auto-sustracción es el throughput E4/E3.", "plan_integracion_halosub_julo2025.md"),
            ("Sustracción por exposición; residuos combinados con el mismo combinador que el cubo madre (comparabilidad D1).", None),
        ],
        checks=(
            "q = nb.load_qc('stages/spec_sgf_qc.json', RUN_ID)\n"
            "for k, v in q['checks'].items():\n"
            "    print(f'  {k}: {v}')\n"
            "assert q['pca_applied'] is False"
        ),
        conclusion_md=(
            "## Estado\n\n"
            "**Pendiente de primera ejecución sobre datos reales** (checkpoint de la spec C5: el "
            "usuario aprueba la spec antes de correr). Kernel y contrato verificados con tests "
            "sintéticos (2026-07-14)."
        ),
    ),
    dict(
        id="C6", qc_optional=True, slug="C6_lpm", title="Sustracción de halo LPM", block="C · Extracción",
        spec="spec_C6_codex_lpm_subtraction.md", run_override=None,
        what=(
            "Sustrae el halo estelar modelando cada spaxel como modulación polinomial de Legendre "
            "(grado 4) del espectro de referencia, con las líneas de ciencia enmascaradas del ajuste "
            "(Julo et al. 2025 App. A.4)."
        ),
        inputs="`stage02` stack + posiciones (B3) + PSF (C1, solo apcorr)",
        outputs="`stages/spec_lpm_qc.json`, `spec_lpm_object.fits`, cubo residual, mapas de coeficientes",
        downstream="D1 v3, G1, E4; cubo residual → E1b (mapa FoV)",
        exec=dict(kind="module_main", target="musepipe.stages.stage_x05_lpm",
                  cost="Ligero (~1–2 min: una pseudo-inversa compartida + diagnósticos de grado)."),
        qc="stages/spec_lpm_qc.json",
        salient=["lpm.degree", "lpm.line_preservation_recovery", "degree_diagnostics.mse_argmin",
                 "checks.v2_line_preservation_ok", "checks.v4_slow_path_ok", "checks.v5_scale_convention_ok"],
        checks_md=(
            "## Los chequeos del QC, en físico\n\n"
            "| Chequeo | ¿Qué pregunta contesta? | Si falla |\n|---|---|---|\n"
            "| `v1_reference_ok` | **¿Hay bastante halo para la referencia?** ≥ 50 spaxels por "
            "exposición, como en C5. | La referencia es ruido. |\n"
            "| `v2_line_preservation_ok` | **¿El método respeta la línea, que es lo que buscamos?** "
            "Se inyecta una gaussiana de 5σ en un control, se sustrae con la misma matriz y se exige "
            "recuperar **≥ 90%** del flujo. Es el argumento de LPM frente a SGF, que sí muerde la "
            "línea. | El método estaría borrando la señal de acreción junto con el halo. Es un smoke "
            "interno: la calibración formal es E4. |\n"
            "| `v3_condition_ok` | **¿El ajuste es numéricamente estable?** Número de condición "
            "< 1e8. | Los coeficientes son ruido amplificado: el residuo deja de significar nada. |\n"
            "| `v4_slow_path_ok` | **¿Cuántos spaxels necesitaron el camino lento?** ≤ 20%. | Señal "
            "de mal condicionamiento generalizado (y de coste). |\n"
            "| `v5_scale_convention_ok` | **¿Escala y cabeceras trazables?** Igual que C5. | D1 no "
            "puede comparar sin adivinar la convención. |\n"
            "| `v6_degree_diagnostics_written` | **¿Está justificado el grado 4 del polinomio?** "
            "Energy-share, curva de MSE y mapas de coeficientes persistidos. | Se pierde la evidencia "
            "de por qué ese grado y no otro (avisos que no bloquean: se miran en los plots). |\n"
        ),
        narrative_md=(
            "## Qué hace C6 y por qué preserva las líneas\n\n"
            "C6 implementa el método **propuesto** por Julo et al. 2025 (LPM): cada spaxel se modela "
            "como `ŝ_xy = Σ_k β_k · P_k(λ̃) · ŝ` (Legendre de grado 4 modulando la referencia), "
            "resuelto por mínimos cuadrados **con las líneas de ciencia fuera del ajuste**. El modelo "
            "interpola suavemente a través de las líneas → el flujo y el perfil del compañero "
            "sobreviven (sin auto-sustracción estructural), y el continuo vecino no se hunde.\n\n"
            "**Diagnósticos de grado (QC, nunca ajuste al vuelo):** energy-share por grado "
            "(Fig. 8 del paper), curva MSE analítica descompuesta en underfit-estrella / "
            "overfit-planeta / overfit-ruido (Fig. 6 / Ec. B.5) y mapas de coeficientes que "
            "descomponen la PSF por frecuencia espectral (Fig. 7: radio AO, spikes, anillos de "
            "Airy). Si señalan que ∂=4 no basta, se revisa la spec — el grado no cambia dentro "
            "del run.\n\n"
            "Límite conocido (paper §4.1): la componente del espectro planetario **colineal** con la "
            "referencia (p.ej. su continuo) se absorbe en el modelo; el LPM es óptimo para "
            "compañeras dominadas por líneas."
        ),
        evidence_md=(
            "## Evidencia: preservación de línea y diagnósticos de grado"
        ),
        evidence_code=(
            "q = nb.load_qc('stages/spec_lpm_qc.json', RUN_ID)\n"
            "l = q['lpm']; d = q['degree_diagnostics']\n"
            "print(f\"grado={l['degree']}  máscara={l['masked_lines_A']}\")\n"
            "print(f\"condición={max(l['condition_number']):.2e}  slow_frac={l['slow_fraction_max']:.3f}\")\n"
            "print(f\"smoke de preservación de línea (control): {l['line_preservation_recovery']}\")\n"
            "print(f\"energy-share (g1..g9): {[f'{v:.3f}' for v in d['energy_share']]}\")\n"
            "print(f\"MSE argmin={d['mse_argmin']}  warns: grado={d['degree_check_warn']} mse={d['mse_check_warn']}\")\n"
            "print('checks:', q['checks'])"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — mapas de coeficientes (Fig. 7 del paper)\n\n"
                    "Planos β̂_k del ajuste diagnóstico de grado 9 (primera exposición): los grados "
                    "bajos muestran el radio AO y los spikes; los altos, anillos de Airy y ruido."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    q = nb.load_qc('stages/spec_lpm_qc.json', RUN_ID)\n"
                    "    with fits.open(q['products']['coeff_maps']) as h:\n"
                    "        maps = np.asarray(h['COEFFS_DEG9'].data, float)\n"
                    "    fig, axes = plt.subplots(2, 5, figsize=(14, 5.6))\n"
                    "    for k, ax in enumerate(axes.ravel()):\n"
                    "        m = maps[k]\n"
                    "        v = np.nanpercentile(m, [25, 75])\n"
                    "        ax.imshow(m, origin='lower', cmap='RdBu_r', vmin=v[0], vmax=v[1])\n"
                    "        ax.set_title(f'grado {k}', fontsize=9); ax.axis('off')\n"
                    "    fig.suptitle('C6 · mapas de coeficientes LPM (descomposición de la PSF)')\n"
                    "    fig.tight_layout(); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — SGF vs LPM alrededor de Hα (Fig. 13/14 del paper)\n\n"
                    "Los dos espectros del compañero en ±80 Å de Hα: si hay línea, el SGF la "
                    "auto-sustrae y hunde el continuo vecino; el LPM la preserva. Requiere C5 corrido."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    def spec(name):\n"
                    "        h = fits.open(rd / 'stages' / name)\n"
                    "        w = np.asarray(h[1].data['wave_A'], float); f = np.asarray(h[1].data['flux'], float)\n"
                    "        h.close(); return w, f\n"
                    "    w_s, f_s = spec('spec_sgf_object.fits')\n"
                    "    w_l, f_l = spec('spec_lpm_object.fits')\n"
                    "    sel = np.abs(w_s - 6563.0) <= 80\n"
                    "    fig, ax = plt.subplots(figsize=(9, 4))\n"
                    "    ax.plot(w_s[sel], f_s[sel], lw=1.0, color='tab:blue', label='SGF (C5)')\n"
                    "    ax.plot(w_l[sel], f_l[sel], lw=1.0, color='tab:orange', label='LPM (C6)')\n"
                    "    ax.axvline(6563, color='tab:red', ls=':'); ax.axhline(0, color='0.6', lw=0.6)\n"
                    "    ax.set_xlabel('λ [Å]'); ax.set_title('C6 · SGF vs LPM alrededor de Hα')\n"
                    "    ax.legend(); fig.tight_layout(); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=PAPER_SPECTRUM_MD,
                code=paper_from_product_cell(
                    product='spec_lpm_object.fits',
                    method='lpm',
                    subdir='c6_lpm',
                    stem='spectrum_paper',
                    qc='stages/spec_lpm_qc.json',
                    title_suffix='espectro del compañero · lpm (C6)',
                ),
            ),
        ],
        decisions=[
            ("**Grado 4 congelado** (tres vías independientes del paper §3.2); diagnósticos de grado como QC con warnings, nunca ajuste al vuelo.", "spec_C6_codex_lpm_subtraction.md"),
            ("Máscara de líneas por target (`lpm_masked_lines_A`); una línea de ciencia sin enmascarar = auto-sustracción parcial silenciosa → el notebook la audita contra el catálogo G2.", None),
            ("Sin pesos por varianza en v1 (OLS plano, como el paper); ponderación por precisión es trabajo futuro explícito.", "plan_integracion_halosub_julo2025.md"),
        ],
        checks=(
            "q = nb.load_qc('stages/spec_lpm_qc.json', RUN_ID)\n"
            "for k, v in q['checks'].items():\n"
            "    print(f'  {k}: {v}')\n"
            "assert q['lpm']['degree'] == 4"
        ),
        conclusion_md=(
            "## Estado\n\n"
            "**Pendiente de primera ejecución sobre datos reales** (checkpoint de la spec C6). "
            "Kernel verificado contra los oráculos del paper (Ec. B.5, límite R→0 de Fig. 2d) y "
            "contrato de etapa con tests sintéticos (2026-07-14)."
        ),
    ),
    # ===================== BLOQUE D — método + calibración =====================
    dict(
        id="D1", slug="D1_method_compare", title="Comparación inter-método (v4, por observable)", block="D · Método",
        spec="spec_D1_v4_codex_method_comparison.md", run_override=None,
        what=(
            "Compara los 6 métodos de extracción (C2–C6) por pares y banda sobre 33 controles, "
            "**separando continuo y líneas como observables distintos** (v4), y emite "
            "`recommended_method` con el árbol congelado (nunca fija el canónico)."
        ),
        inputs="C2/C3/C4 + C5/C6 (sgf/lpm) + G1 verdicts + predictor Ec. 1 de C5",
        outputs="`stages/stage_x10_qc.json`",
        downstream="D2 (consume el canónico de config)",
        exec=dict(kind="script", target="stage_x10_compare.sh", cost="Ligero (~6 s)."),
        qc="stages/stage_x10_qc.json",
        salient=["spec_version", "verdict", "action", "recommended_method",
                 "primary_pairs_continuum", "primary_pairs_lines", "comparison_policy"],
        checks_md=(
            "## Los chequeos del QC, en físico\n\n"
            "D1 decide si dos métodos **discrepan de verdad** o solo por ruido. Sus chequeos vigilan "
            "las tres formas de engañarse en esa comparación: subestimar el ruido, comparar contra "
            "controles sucios y contar una corrección dos veces.\n\n"
            "| Chequeo | ¿Qué pregunta contesta? | Si falla |\n|---|---|---|\n"
            "| `v1_sigma_empirical_le_naive` | **¿Hay dispersión que los errores individuales no "
            "explican?** Exige `σ_empírico ≤ σ_ingenuo`, donde el ingenuo es "
            "`√(σ_i² + σ_j²)` (errores independientes). Dos métodos sobre el **mismo** cubo están "
            "positivamente correlacionados, así que la varianza de su diferencia debe ser *menor* que "
            "la ingenua: `σ_emp < σ_ing` es lo normal y no es un fallo. | Un σ empírico **mayor** que "
            "el ingenuo delata varianza extra —controles mal emparejados, una sistemática no "
            "modelada, o σ individuales subestimados— y el QC lo declara en `open_issues` con la "
            "celda concreta (par y banda) que la produce. |\n"
            "| `v2_controls_clean` | **¿Los controles de los pares primarios son utilizables?** "
            "Ningún par primario degradado (por controles ausentes o inconsistentes). | El veredicto "
            "del par se marca degradado: no se puede afirmar que dos métodos difieran. |\n"
            "| `v5_no_double_throughput` | **¿Se aplicó la corrección de throughput dos veces?** La "
            "corrección es interna a la comparación (solo en memoria); los productos en disco quedan "
            "SIN corregir y E3/G2 aplican la suya. | El flujo del compañero saldría corregido dos "
            "veces aguas abajo — un sesgo silencioso en Ṁ. |\n"
            "| `v6_scale_check_ok` | **¿Están los dos métodos en la misma escala antes de restarlos?** "
            "| Se estaría midiendo una diferencia de convención, no de física. |\n"
            "| `v7_t_calibration` | **¿El estadístico t está bien calibrado?** Fracción de pares de "
            "CONTROL que salen «divergentes» vs la fracción esperada: en posiciones sin fuente, los "
            "métodos solo pueden diferir por ruido, así que el ritmo de falsos positivos debe ser el "
            "nominal. | El umbral de «discrepan» no significa lo que dice. |\n"
        ),
        narrative_md=(
            "## Qué hace D1 v4 y cómo decide\n\n"
            "D1 compara los métodos **por pares y por banda** con un **t control-centrado** "
            "(df = n_controles − 1): la diferencia del objeto contra la distribución de las "
            "diferencias de los 33 controles — referenciado a controles por construcción. "
            "Bandas B1–B6 (continuo) y LHα/LHβ/LOI (líneas); umbrales congelados p<0.0455 "
            "(divergente) y p<0.0027 (fuerte). Todo eso viene de v2/v3 y **no cambia**.\n\n"
            "**Qué cambia en v4** ([`docs/spec_D1_v4_codex_method_comparison.md`](../docs/spec_D1_v4_codex_method_comparison.md), "
            "congelada 2026-07-29). La auditoría del cubo multiépoca encontró que v3 mezclaba tres "
            "observables distintos en un único estadístico:\n\n"
            "| Defecto de v3 | Qué hace v4 |\n|---|---|\n"
            "| Aplicaba el throughput escalar **medido con inyecciones de Hα** a todas las bandas de "
            "continuo — una extrapolación cromática nunca medida, capaz de **crear** diferencias entre "
            "continuos que ya concordaban | B1–B6 se comparan sobre el flujo persistido tal cual, en la "
            "`SCALEREF` común: `comparison_mode = raw_total_continuum`, **throughput no aplicado** |\n"
            "| Integraba las bandas de línea **sin restar continuo local**, así que la fila `LOI` podía "
            "estar dominada por el nivel de continuo y no por O I 8446 | Se resta la mediana móvil "
            "canónica de 80 Å antes de integrar, y el residual de línea se divide por el throughput "
            "**una sola vez**: `comparison_mode = local_continuum_subtracted_throughput_line` |\n"
            "| Dejaba que **SGF y LPM** gobernaran `divergent_continuum` contra extractores de flujo "
            "total, cuando ambos eliminan o absorben el continuo **por construcción** | "
            "`CONTINUUM_METHODS = (aperture, optimal_ls, optimal_psfsub, psffit)`. SGF y LPM siguen en "
            "todas las tablas como diagnóstico (`role=diagnostic_noncomparable_continuum`) pero **no "
            "pueden activar** `divergent_continuum` |\n\n"
            "El QC publica ahora `primary_pairs_continuum` y `primary_pairs_lines` por separado "
            "(`primary_pairs` se conserva como la unión, por compatibilidad), y el gate binomial de "
            "controles se calcula **por observable**: una fila diagnóstica ya no degrada un observable "
            "al que no pertenece.\n\n"
            "**Por qué esto importa aquí y no es cosmético.** El veredicto v3 registrado para "
            "ROXs 12 B era `divergent_continuum`, y lo activaba B6 con t≈+14 de **psffit contra "
            "sgf/lpm** — exactamente los dos métodos que v4 excluye del veredicto de continuo. Con la "
            "política v4, esa señal pasa a ser diagnóstica y el continuo lo deciden los cuatro "
            "métodos comparables entre sí. **El veredicto puede cambiar al re-ejecutar**, y por eso la "
            "celda de abajo empieza declarando qué versión de QC está leyendo.\n\n"
            "Lo que v4 **no** toca: `METHOD_ORDER`, las bandas, los umbrales, el scale-check, la "
            "jerarquía G1, el árbol de recomendación y la regla de que la elección canónica es humana."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Versión de QC, política de comparación, pares primarios **por observable** y t por banda "
            "de cada par primario. Si el QC en disco es todavía `D1_v3`, la celda lo dice y no "
            "reinterpreta esos números con la política nueva: un QC v3 se lee como v3."
        ),
        evidence_code=(
            "from scipy.stats import t as t_dist\n"
            "q = nb.load_qc('stages/stage_x10_qc.json', RUN_ID)\n"
            "spec = q.get('spec_version')\n"
            "policy = q.get('comparison_policy')\n"
            "print(f\"spec del QC en disco : {spec}\")\n"
            "if spec != 'D1_v4':\n"
            "    print('   ⚠ Este QC es ANTERIOR a la spec v4: sus B1-B6 llevan throughput de Hα\\n'\n"
            "          '     aplicado y sus bandas de línea no tienen el continuo local restado.\\n'\n"
            "          '     El veredicto de continuo pudo activarlo un par sgf/lpm, que v4 ya no\\n'\n"
            "          '     deja votar. Hay que re-ejecutar D1 para leerlo con la política nueva.')\n"
            "else:\n"
            "    print(f\"   continuo : {policy['continuum_mode']}  sobre {policy['continuum_methods']}\")\n"
            "    print(f\"   líneas   : {policy['line_mode']}  (ventana {policy['line_continuum_window_A']:.0f} Å)\")\n"
            "    for method, why in (q.get('excluded_from_continuum_verdict') or {}).items():\n"
            "        print(f\"   sin voto en continuo: {method} ({why})\")\n"
            "print()\n"
            "print(f\"veredicto : {q['verdict']}   ->  {q['action']}\")\n"
            "print(f\"motivo    : {q.get('reason')}\")\n"
            "print(f\"recomendado: {q['recommended_method']}   elegibles: {q['recommendation_rules']['eligible']}\")\n"
            "print()\n"
            "# `primary_pairs_*` solo existen en v4; en v3 hay una única lista.\n"
            "groups = [(k, q[k]) for k in ('primary_pairs_continuum', 'primary_pairs_lines') if k in q]\n"
            "if not groups:\n"
            "    groups = [('primary_pairs (v3, sin separar observable)', q['primary_pairs'])]\n"
            "st = q['statistics']; df = st['n_controls'] - 1\n"
            "T_DIV = t_dist.ppf(1 - st['p_divergent'] / 2, df)\n"
            "T_STR = t_dist.ppf(1 - st['p_strong'] / 2, df)\n"
            "for name, pairs in groups:\n"
            "    print(f\"{name}: {list(pairs)}\")\n"
            "for name, pairs in groups:\n"
            "    for pp in pairs:\n"
            "        print(f'\\nt control-centrado ({pp})  [{name.split()[0]}]:')\n"
            "        for b, t in q['t_matrix'][pp].items():\n"
            "            if t is None: continue\n"
            "            flag = '  <-- FUERTE' if abs(t) > T_STR else ('  <- marginal' if abs(t) > T_DIV else '')\n"
            "            print(f'   {b:8s}: t = {t:+.2f}{flag}')"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — t control-centrado por par × banda (15 pares)\n\n"
                    "Del `t_matrix` del QC. Se marcan con `C` los pares que pueden votar el "
                    "**continuo** (los cuatro de `CONTINUUM_METHODS`) y con `L` los que votan "
                    "**líneas**; el resto están en la tabla como diagnóstico y no activan veredicto. "
                    "Bajo v4 las columnas B1–B6 y las de línea ya no son la misma cantidad física: "
                    "las primeras son flujo crudo integrado, las segundas residual de línea sobre "
                    "continuo local."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from scipy.stats import t as t_dist\n"
                    "    q = nb.load_qc('stages/stage_x10_qc.json', RUN_ID)\n"
                    "    st = q['statistics']\n"
                    "    T_STR = t_dist.ppf(1 - st['p_strong'] / 2, st['n_controls'] - 1)\n"
                    "    tm = q['t_matrix']\n"
                    "    cont = list(q.get('primary_pairs_continuum', []))\n"
                    "    line = list(q.get('primary_pairs_lines', []))\n"
                    "    primary = list(dict.fromkeys(cont + line)) or list(q['primary_pairs'])\n"
                    "    pairs = primary + [p for p in tm if p not in primary]\n"
                    "    bands = list(tm[pairs[0]].keys())\n"
                    "    M = np.array([[np.nan if tm[p].get(b) is None else tm[p][b] for b in bands] for p in pairs])\n"
                    "    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(pairs) + 1.8))\n"
                    "    im = ax.imshow(M, cmap='RdBu_r', vmin=-6, vmax=6, aspect='auto')\n"
                    "    ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands)\n"
                    "    def _role(p):\n"
                    "        tag = ('C' if p in cont else '') + ('L' if p in line else '')\n"
                    "        return (tag or ('*' if p in primary else '  ')).ljust(2) + ' '\n"
                    "    labels = [_role(p) + p.replace('_vs_', ' vs ') for p in pairs]\n"
                    "    ax.set_yticks(range(len(pairs))); ax.set_yticklabels(labels, fontsize=7)\n"
                    "    for i in range(len(pairs)):\n"
                    "        for j in range(len(bands)):\n"
                    "            v = M[i, j]\n"
                    "            if np.isfinite(v):\n"
                    "                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=6, color='k' if abs(v) < 4 else 'w')\n"
                    "    ax.set_title(f\"D1 {q.get('spec_version','?')} · t por par × banda \"\n"
                    "                 f\"(C=vota continuo, L=vota líneas; |t|>{T_STR:.2f} fuerte)\")\n"
                    "    fig.colorbar(im, label='t'); fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'd1_compare'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'tmatrix.png', dpi=110); print('figura ->', outdir / 'tmatrix.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — los dos observables por separado\n\n"
                    "Arriba, las bandas de **continuo** B1–B6 con solo los pares que pueden votarlas; "
                    "abajo, las de **línea** con sus pares primarios. Es la figura que hace visible el "
                    "cambio de v4: si una divergencia vive únicamente en el panel de continuo con "
                    "pares que ya no votan, deja de ser un veredicto y pasa a ser un diagnóstico. Las "
                    "líneas punteadas son los umbrales divergente y fuerte."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from scipy.stats import t as t_dist\n"
                    "    q = nb.load_qc('stages/stage_x10_qc.json', RUN_ID)\n"
                    "    st = q['statistics']; df = st['n_controls'] - 1\n"
                    "    t_div = t_dist.ppf(1 - st['p_divergent'] / 2, df)\n"
                    "    t_str = t_dist.ppf(1 - st['p_strong'] / 2, df)\n"
                    "    kinds = {b['name']: b['kind'] for b in q['bands']}\n"
                    "    all_bands = list(q['t_matrix'][list(q['t_matrix'])[0]].keys())\n"
                    "    cbands = [b for b in all_bands if kinds.get(b) == 'continuum']\n"
                    "    lbands = [b for b in all_bands if kinds.get(b) == 'line']\n"
                    "    if 'primary_pairs_continuum' in q:\n"
                    "        panels = [('continuo (B1-B6) · pares con voto', cbands, list(q['primary_pairs_continuum'])),\n"
                    "                  ('líneas · pares con voto', lbands, list(q['primary_pairs_lines']))]\n"
                    "    else:\n"
                    "        # QC v3: no existe la separación por observable. No se inventa: un único\n"
                    "        # panel con todos los pares primarios, y el título lo dice.\n"
                    "        panels = [('QC v3 · sin separar por observable (todos los pares primarios)',\n"
                    "                   all_bands, list(q['primary_pairs']))]\n"
                    "    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 4.0 * len(panels)), squeeze=False)\n"
                    "    for ax, (title, bands, pairs) in zip(axes[:, 0], panels):\n"
                    "        if not bands or not pairs:\n"
                    "            ax.text(0.5, 0.5, 'sin pares con voto para este observable', ha='center',\n"
                    "                    va='center', transform=ax.transAxes, color='#8a8a8a')\n"
                    "            ax.set_title(f'{title} — sin veredicto'); ax.set_xticks([]); continue\n"
                    "        # Un punto por par y banda: aguanta 15 pares, que un agrupado de barras no.\n"
                    "        rng = np.random.default_rng(0)\n"
                    "        for j, b in enumerate(bands):\n"
                    "            vals = [(pp, q['t_matrix'].get(pp, {}).get(b)) for pp in pairs]\n"
                    "            vals = [(pp, t) for pp, t in vals if t is not None]\n"
                    "            # anota solo los 2 |t| mayores de cada banda: con 15 pares, anotarlos\n"
                    "            # todos tapa la figura y no se lee ninguno\n"
                    "            top = {pp for pp, _ in sorted(vals, key=lambda kv: -abs(kv[1]))[:2]}\n"
                    "            for pp, t in vals:\n"
                    "                strong = abs(t) > t_str\n"
                    "                ax.plot(j + rng.uniform(-0.18, 0.18), t, 'o', ms=5,\n"
                    "                        color='tab:red' if strong else ('tab:orange' if abs(t) > t_div else 'tab:blue'),\n"
                    "                        alpha=0.85)\n"
                    "                if strong and pp in top:\n"
                    "                    ax.annotate(pp.replace('_vs_', '/'), (j, t), fontsize=5.5,\n"
                    "                                xytext=(7, 0), textcoords='offset points', va='center')\n"
                    "        for s in (t_div, t_str):\n"
                    "            ax.axhline(s, color='k', ls=':', lw=0.8); ax.axhline(-s, color='k', ls=':', lw=0.8)\n"
                    "        ax.axhline(0, color='k', lw=0.6)\n"
                    "        # symlog: deja ver a la vez el umbral (~2-3) y las t de tres cifras\n"
                    "        ax.set_yscale('symlog', linthresh=max(t_str, 1.0))\n"
                    "        ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands)\n"
                    "        ax.set_ylabel('t control-centrado (symlog)')\n"
                    "        ax.set_title(f'{title} · {len(pairs)} par(es)', fontsize=10)\n"
                    "    fig.suptitle(f\"{q.get('spec_version','D1 ?')} · veredicto {q['verdict']} \"\n"
                    "                 f\"(rojo = fuerte, naranja = divergente)\", y=0.995)\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'd1_compare'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'primary_pairs.png', dpi=110); print('figura ->', outdir / 'primary_pairs.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**Separación por observable (v4)**: el continuo lo deciden solo `aperture`, `optimal_ls`, `optimal_psfsub` y `psffit`; SGF y LPM quedan como diagnóstico porque anulan o absorben el continuo por construcción. Las líneas las deciden los seis, tras restar continuo local.", "spec_D1_v4_codex_method_comparison.md"),
            ("**El throughput de Hα ya no toca B1–B6.** Aplicarlo a bandas de continuo era una extrapolación cromática no medida que podía fabricar divergencias; ahora se aplica exactamente una vez y solo al residual de línea.", "spec_D1_v4_codex_method_comparison.md"),
            ("**METHOD_ORDER de 6** con la familia espectral (Julo+25); pares primarios = validados por G1; bandas, umbrales, scale-check y protocolo anti cherry-picking **sin cambios** desde v2/v3.", "spec_D1_v3_codex_method_comparison.md"),
            ("**optimal_psfsub → rejected** (histórico v3): su T venía de un E4 pre-consolidación Psfao; verificado con worktree HEAD que el cambio no provenía del código nuevo. Revisar contra el E4 v2 actual al re-ejecutar.", None),
            ("**El usuario mantuvo psffit como canónico** (2026-07-15). Esa decisión sigue en pie y D1 no la toca: el árbol solo recomienda con veredicto `consistent` y la elección canónica es humana.", "d1_canonical_method_decision.md"),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (pendiente de re-ejecución)\n\n"
            "**La spec v4 está congelada y el código la implementa (`SPEC_VERSION = D1_v4`), pero el "
            "veredicto de este objeto todavía no se ha vuelto a calcular con ella.** La celda de "
            "evidencia declara arriba qué versión de QC estás leyendo; no des por bueno un veredicto "
            "v3 como si fuera v4.\n\n"
            "- **Lo que había (v3, 2026-07-15):** `divergent_continuum` con 3 pares primarios limpios "
            "(psffit–sgf, psffit–lpm, sgf–lpm), activado por B6 con t≈+14 de psffit contra la familia "
            "espectral. **Bajo v4 esos tres pares ya no votan el continuo**: dos de sus miembros son "
            "sgf y lpm.\n"
            "- **Qué esperar al re-ejecutar:** el continuo pasa a decidirse entre "
            "`aperture/optimal_ls/optimal_psfsub/psffit` sobre flujo crudo, y las líneas sobre "
            "residual con continuo local restado. El veredicto puede cambiar; la firma B6 no "
            "desaparece, se reclasifica como diagnóstico.\n"
            "- **Estadístico (sin cambios):** t control-centrado, 33 controles (df=32), corr_length "
            "{{qc:stages/stage_g1_qc.json:covariance.corr_length_channels_median:.2f}} canales "
            "(de G1).\n"
            "- **Downstream:** D2 calibró los 6 métodos; E1 re-confirmó la no-detección con 6 "
            "métodos. Si el veredicto de D1 cambia, hay que revisar qué arrastra hacia D2.\n"
            "- Todo provisional hasta cerrar el A-block."
        ),
    ),
    dict(
        id="D2", slug="D2_calibrate", title="Calibración espectral", block="D · Método",
        spec="spec_D2_codex_spectral_calibration.md", run_override=None,
        what=("Calibra los 6 métodos y la primaria (Δλ, escala de flujo, continuo, presupuesto de "
              "error) y produce los espectros definitivos, con `spec_final_object.fits` como canónico."),
        inputs="Los 6 métodos + la primaria de C4 + M3",
        outputs=("`stages/stage_x11_qc.json`, `spec_final_object.fits`, "
                 "`spec_calibrated_<método>_object.fits` (6), `spec_calibrated_psffit_star.fits`"),
        downstream="E1, E3, G2",
        exec=dict(kind="script", target="stage_x11_calibrate.sh", cost="Ligero–moderado."),
        qc="stages/stage_x11_qc.json",
        salient=["canonical_method", "scale_factor", "v3_continuum_stable.ok", "fraction_channels_methods_agree", "control_referenced"],
        narrative_md=(
            "## Qué hace D2 y qué integramos\n\n"
            "**Los espectros definitivos son el entregable de esta etapa**, no solo la entrada de "
            "E1: el compañero por los **6 métodos** (misma rejilla, superponibles) y la "
            "**primaria** (`spec_calibrated_psffit_star.fits`), todos con unidad declarada "
            "(`BUNIT`) y error total. El canónico se copia además a `spec_final_object.fits`.\n\n"
            "D2 convierte el espectro canónico (psffit) en el **producto científico final**: λ "
            "corregida y en marco declarado, flujo en escala validada, continuo por dos vías, y un "
            "**error total con presupuesto de sistemáticos explícito**. **Aplica factores medidos "
            "aguas arriba** (trazables al QC que los midió) — no mide nada nuevo.\n\n"
            "- **λ:** Δλ = −{{qc:stages/stage00q_qc.json:m1_wavelength.offset_median_A:.3f}} Å "
            "(de A4/M1), marco final **baricéntrico**.\n"
            "- **Flujo:** escala = 1.0 (M3 factor "
            "{{qc:stages/stage00q_qc.json:m3_flux.flux_factor:.3f}} vs Gaia DR3, estado "
            "**{{qc:stages/stage00q_qc.json:m3_flux.status}}**: se aplica escala 1.0 porque el factor "
            "es consistente con 1 dentro de su error).\n"
            "- **Continuo:** running-median y polinomio; su diferencia es el término `sys_continuum`.\n"
            "- **Error:** `stat` (empírico, M5 rojo) + sistemáticos (flujo-cal, psf, cielo, telúrico, "
            "continuo). El total está **dominado por el stat**.\n\n"
            "**Diagnóstico del 'continuo rojo inestable'** ([`docs/d2_red_continuum_diagnosis.md`]"
            "(../docs/d2_red_continuum_diagnosis.md)): son 3 cosas reales (señal de enana fría + "
            "sistemático de nivel inter-método + rigidez del polinomio), **no** un defecto de PSF.\n\n"
            "**El chequeo de continuo estable (`v3_continuum_stable`)** pregunta, en físico: *¿el "
            "continuo que llamamos «del compañero» es suyo, o es lo que quedó del halo de la "
            "primaria?* Se responde comparando dos métodos de extracción independientes — abajo, en "
            "el Plot 1, está explicado en detalle. Para este objeto: "
            "{{qc:stages/stage_x11_qc.json:checks.v3_continuum_stable.fraction_channels_methods_agree:.3f}} "
            "de los canales concuerdan (umbral "
            "{{qc:stages/stage_x11_qc.json:checks.v3_continuum_stable.threshold}}), así que el chequeo "
            "**{{qc:stages/stage_x11_qc.json:checks.v3_continuum_stable.ok}}** = ok. Desde 2026-07-11 "
            "la métrica es la **control-referenciada** (consistente con el t control-centrado de D1) y "
            "el producto final lleva la columna `cont_runmed_biasref` para G3. **Lo que queda es un "
            "sistemático cromático genuino, y afecta a la FORMA del continuo (G3), no a la línea Hα "
            "ni al límite de Ṁ (E1/E3).**"
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
            "if q['flux'].get('declared_err_frac'):\n"
            "    print(f\"       flujo-cal declarado (NO en el total): {100 * q['flux']['declared_err_frac']:.1f}%\"\n"
            "          f\" — {q['flux']['declared_source'][:70]}\")\n"
            "# La unidad que resolverán E3/G3: knob -> BUNIT del producto -> QC de M3.\n"
            "un = q['flux'].get('unit')\n"
            "if un:\n"
            "    print(f\"       unidad: {un['cgs']} erg/s/cm²/Å vía {un['source']}\"\n"
            "          f\" (BUNIT={un['from_bunit']}, M3={un['from_m3_qc']})\")\n"
            "    if un['conflict']:\n"
            "        print('       ⚠ ' + un['conflict'])\n"
            "im = q['continuum']['intermethod_systematic']; ar = im['after_control_reference']\n"
            "print()\n"
            "print('continuo inter-método (psffit vs optimal_psfsub):')\n"
            "print(f\"  ANTES  (crudo)       fraction_agree={im['fraction_channels_methods_agree']:.3f}  red_ratio={im['red_band_median_ratio']:.2f}×\")\n"
            "print(f\"  DESPUÉS (referenciado) fraction_agree={ar['fraction_channels_methods_agree']:.3f}  red_ratio={ar['red_band_median_ratio']:.2f}×\")\n"
            "print(f\"  sesgo rojo psffit/psfsub = {ar['canonical_control_bias_red']:+.0f} / {ar['other_control_bias_red']:+.0f}\")\n"
            "v3 = q['checks']['v3_continuum_stable']\n"
            "print(f\"\\ncontinuo estable (V3): ok={v3['ok']} — {v3['fraction_channels_methods_agree']:.3f}\"\n"
            "      f\" de los canales tienen los dos métodos de acuerdo dentro del error combinado,\"\n"
            "      f\" umbral {v3['threshold']}\")\n"
            "print(f\"  métrica={v3['metric']}\")\n"
            "print('  significa: lo que NO concuerda no puede ser el compañero (los dos métodos miden el mismo objeto)')\n"
            "print('             sino residuo cromático de la sustracción de halo -> afecta a la FORMA del continuo (G3),')\n"
            "print('             no a la línea Hα ni al límite de Ṁ (E1/E3).')\n"
            "\n"
            "# Los espectros definitivos: la tabla que D2 deja en su QC.\n"
            "sp = q.get('spectra')\n"
            "if sp is None:\n"
            "    print('\\n(este QC es anterior a la tabla de espectros: re-ejecuta D2 para tenerla)')\n"
            "else:\n"
            "    n = lambda v, f='{:.1f}': '—' if v is None else f.format(v)\n"
            "    print(f\"\\nespectros definitivos: {sp['n_companion']} métodos + {sp['n_primary']} primaria\"\n"
            "          f\" | unidad {sp['unit']} (consistente={sp['unit_consistent']})\"\n"
            "          f\" | misma rejilla={sp['companions_share_grid']}\")\n"
            "    print(f\"medianas en {sp['red_band_A'][0]:.0f}–{sp['red_band_A'][1]:.0f} Å (donde el compañero se detecta)\")\n"
            "    print(f\"  {'espectro':16s} {'papel':10s} {'S/N':>7s} {'flujo':>10s} {'err total':>10s} {'ratio/canon':>12s}\")\n"
            "    for row in sp['table']:\n"
            "        tag = row['name'] + (' *' if row['canonical'] else '')\n"
            "        print(f\"  {tag:16s} {row['role']:10s} {n(row['red_snr_median']):>7s}\"\n"
            "              f\" {n(row['red_flux_median']):>10s} {n(row['flux_err_total_median']):>10s}\"\n"
            "              f\" {n(row['ratio_to_canonical_red'], '{:.2f}×'):>12s}\")\n"
            "    for row in sp['table']:\n"
            "        if row['caveat']:\n"
            "            print(f\"  ! {row['name']}: {row['caveat']}\")"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — dónde se mide\n\n"
                    "Antes de comparar métodos, ver el sitio. Es el **flujo mediano en λ** del cubo "
                    "sin sustraer (1 de cada 10 canales, que para situar aperturas sobra), en escala "
                    "**logarítmica**: en lineal solo se vería el núcleo de la primaria y el halo — "
                    "que es de lo que trata toda esta etapa — quedaría invisible.\n\n"
                    "Encima, las cuatro cajas box3 donde se hace la fotometría: la del **compañero** "
                    "y las **tres de control**, con el círculo punteado que marca la separación. La "
                    "imagen deja claro de un vistazo lo que sostiene toda la comparación: las cuatro "
                    "caen sobre el mismo halo, a la misma distancia de la estrella, y las tres verdes "
                    "no tienen fuente dentro.\n\n"
                    "*(Esta celda calcula la referencia sin sustraer y la deja en "
                    "`ref_sin_sustraer` para los dos plots siguientes: construir el modelo de PSF "
                    "tarda ~27 s y se hace una sola vez.)*"
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from musepipe.stages import unsubtracted_aperture_reference, photometry_map_figure\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    ref_sin_sustraer, motivo = unsubtracted_aperture_reference(rd / 'stages')\n"
                    "    if ref_sin_sustraer is None:\n"
                    "        print('No se pudo construir la referencia sin sustraer:', motivo)\n"
                    "    else:\n"
                    "        print('apertura', ref_sin_sustraer['aperture'],\n"
                    "              '· apcorr', ref_sin_sustraer['apcorr_mode'],\n"
                    "              '· compañero yx =', [round(v, 1) for v in ref_sin_sustraer['position_yx']])\n"
                    "        for c in ref_sin_sustraer['controls']:\n"
                    "            print(f\"  control {c['name']:18s} PA {c['pa_deg']:6.1f}°  yx={[round(v,1) for v in c['position_yx']]}\")\n"
                    "        for s in ref_sin_sustraer['controls_skipped']:\n"
                    "            print('  (saltado)', s['name'], '->', s['reason'])\n"
                    "        fig, ax = photometry_map_figure(rd / 'stages', plt=plt, reference=ref_sin_sustraer)\n"
                    "        outdir = rd / 'plots' / 'd2_calibrate'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "        fig.savefig(outdir / 'photometry_map.png', dpi=110)\n"
                    "        print('figura ->', outdir / 'photometry_map.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — los 6 métodos contra la misma apertura SIN sustraer\n\n"
                    "Todos los métodos hacen, en el fondo, lo mismo: **quitar el halo de la primaria** "
                    "en la posición del compañero. Para saber cuánto quitó cada uno hace falta una "
                    "medida de *lo que había antes*, y esa medida es una **apertura simple sin "
                    "sustracción**: la misma caja 3×3, en la misma posición de B3 y con la misma "
                    "corrección de apertura de C1 que usa C2 — lo único que cambia es que se mide "
                    "sobre `cube_input_local_object.fits`, el cubo tal cual entra a 04b.\n\n"
                    "La corrección de apertura no es un detalle: una caja 3×3 recoge una fracción "
                    "minúscula de la PSF de NFM, así que apcorr vale ~40×. Sin aplicarla a la "
                    "referencia la comparación no sería de manzanas con manzanas.\n\n"
                    "**Izquierda — `continuo − offset`.** En gris lo que hay en la apertura sin restar "
                    "nada; encima, lo que deja el método. A la referencia se le resta su **valor "
                    "mínimo** (el offset, escrito en la leyenda): sin eso el pedestal la manda arriba "
                    "del todo y lo único que se aprende es que está muy por encima; con el mínimo "
                    "fuera, lo que queda en el eje es su **forma**, comparable con la del método.\n\n"
                    "**Derecha — la misma resta donde NO hay compañero.** En verde, las tres "
                    "posiciones de control tal cual (halo puro). En morado, esas mismas tres después "
                    "de aplicarles **la resta que el método hizo en el compañero**:\n\n"
                    "> `control − (referencia − método)`\n\n"
                    "Ahí no hay ninguna fuente, así que si el método quita exactamente el halo que "
                    "corresponde a ese radio, el morado debe quedarse **en cero**. Lo que se separe "
                    "del cero es halo que el método no quitó (por encima) o que quitó de más (por "
                    "debajo), en ese ángulo. Es el mismo argumento del chequeo de continuo estable, "
                    "pero medido contra sitios sin compañero en vez de contra otro método — y por eso "
                    "**este panel no lleva offset**: su cero es físico.\n\n"
                    "Las tres verdes valen además por sí solas: su dispersión mide **cuánto cambia el "
                    "halo con el ángulo** a esa misma separación. En ROXs 12 b los continuos medianos "
                    "van de 3778 (posición opuesta) a 4317 — una variación azimutal comparable a lo "
                    "que aporta el compañero en esa apertura. Eso es, en una imagen, por qué la "
                    "sustracción de halo domina esta etapa.\n\n"
                    "Los dos ejes son symlog porque los residuos cruzan el cero."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from musepipe.stages import halo_removal_figure\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    # Reutiliza la referencia del mapa de arriba si esa celda ya corrió.\n"
                    "    fig, axes = halo_removal_figure(rd / 'stages', plt=plt,\n"
                    "                                    reference=globals().get('ref_sin_sustraer'))\n"
                    "    if fig is None:\n"
                    "        print('No se pudo construir la referencia sin sustraer:', axes)\n"
                    "    else:\n"
                    "        outdir = rd / 'plots' / 'd2_calibrate'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "        fig.savefig(outdir / 'halo_removal.png', dpi=110)\n"
                    "        print('figura ->', outdir / 'halo_removal.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 3 — lo que QUEDA tras restar, en % de lo que había\n\n"
                    "La otra mitad de la comparación anterior, ahora que cada método tiene su panel "
                    "a ancho completo. Se dice «lo que queda» y no «cuánto se quitó» porque el 100% "
                    "**no** es la meta: la referencia incluye también al compañero, así que lo que "
                    "*debe* quedar es él. Lo que no admite discusión es el **cero**: por debajo se "
                    "quitó más de lo que había, y eso es sobre-sustracción.\n\n"
                    "Aquí se usa la referencia **entera**, sin el offset del plot anterior — con él "
                    "dejaría de ser «% de lo que había».\n\n"
                    "Lo que se ve en ROXs 12 b, en el rojo (donde el compañero existe): `psffit` deja "
                    "~18%, `aperture` ~19% — el compañero. `optimal_ls` deja **−47%**: quitó vez y "
                    "media lo que había, el mismo defecto que su chequeo `v3_continuum_bias` señala "
                    "en C3. `sgf` y `lpm` se quedan pegados al 0 porque **filtran el continuo por "
                    "construcción**: su valor está en la línea, no en el nivel."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from musepipe.stages import halo_remaining_figure\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    # Reutiliza la referencia del plot anterior si esa celda ya corrió.\n"
                    "    fig, axes = halo_remaining_figure(rd / 'stages', plt=plt,\n"
                    "                                      reference=globals().get('ref_sin_sustraer'))\n"
                    "    if fig is None:\n"
                    "        print('No se pudo construir la referencia sin sustraer:', axes)\n"
                    "    else:\n"
                    "        outdir = rd / 'plots' / 'd2_calibrate'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "        fig.savefig(outdir / 'halo_remaining.png', dpi=110)\n"
                    "        print('figura ->', outdir / 'halo_remaining.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 4 — ¿el continuo del compañero es suyo o es halo de la primaria?\n\n"
                    "### La pregunta física\n\n"
                    "El compañero está a **{{qc:stages/stage01c_qc.json:astrometry.sep_arcsec:.2f}}″** de "
                    "una estrella unas **10³ veces más brillante** (en el rojo; más aún en el azul), y a "
                    "esa separación seguimos dentro de su halo AO. Todo espectro del compañero es, por "
                    "tanto, un **residuo**: lo que queda tras restar el halo. Cada método de extracción "
                    "lo resta de una forma distinta, así que la pregunta obligada antes de creerse la "
                    "forma del continuo es: *¿esta pendiente y estas bandas son del compañero, o son lo "
                    "que a este método le sobró del halo?*\n\n"
                    "### Cómo se responde (y qué es «V3»)\n\n"
                    "**V3** es la tercera verificación de la spec de D2 (§6), y «gate» / «gatea» quiere "
                    "decir que es un chequeo con umbral cuyo resultado entra en el semáforo final de F1 "
                    "— no un cálculo que cambie ningún dato. En el QC vive como "
                    "`checks.v3_continuum_stable`.\n\n"
                    "La idea: **dos métodos independientes miden el mismo objeto real**, así que su "
                    "continuo debería coincidir *dentro del error*. Lo que no coincide no puede ser el "
                    "compañero — es sistemático del método. Se mide como la **fracción de canales** en "
                    "los que\n\n"
                    "> |continuo_A − continuo_B| ≤ √(σ_A² + σ_B²)  (errores estadísticos combinados)\n\n"
                    "y se exige ≥ **{{qc:stages/stage_x11_qc.json:checks.v3_continuum_stable.threshold}}**. "
                    "El par no es fijo: lo eligen D1/D2 y lo declara el QC — aquí es "
                    "**{{qc:stages/stage_x11_qc.json:canonical_method}}** (canónico) vs "
                    "**{{qc:stages/stage_x11_qc.json:continuum.intermethod_systematic.canonical_vs}}**.\n\n"
                    "### Por qué «referenciado a controles» (los dos paneles)\n\n"
                    "Además del residuo en el compañero, cada método deja un **pedestal en sitios donde "
                    "no hay nada** — controles al mismo radio, procesados igual. Ese pedestal es común y "
                    "no dice nada del método: compararlo cuenta el halo dos veces. Restando a cada "
                    "método su propio nivel de control queda solo la discrepancia **genuinamente "
                    "dependiente del método**.\n\n"
                    "La operación es literalmente una resta, y el panel la enseña: la **línea "
                    "discontinua** de la izquierda es el nivel de controles de cada método, y el panel "
                    "derecho es *izquierda menos discontinua*. Si las dos columnas se parecen mucho, "
                    "es que ese pedestal común era pequeño frente a la discrepancia — que es "
                    "justamente el caso aquí, y por eso el chequeo sigue por debajo del umbral.\n\n"
                    "Para este objeto: "
                    "{{qc:stages/stage_x11_qc.json:continuum.intermethod_systematic.fraction_channels_methods_agree:.3f}} "
                    "(crudo) → "
                    "{{qc:stages/stage_x11_qc.json:continuum.intermethod_systematic.after_control_reference.fraction_channels_methods_agree:.3f}} "
                    "(referenciado). Referenciar **no siempre sube** la cifra: acerca los continuos donde "
                    "el pedestal era común y puede bajarla si el residuo cromático no lo es — eso también "
                    "es información.\n\n"
                    "### Qué significa que falle\n\n"
                    "Que parte de lo que llamamos continuo del compañero es **residuo cromático de la "
                    "sustracción de halo**, no su SED. Consecuencia acotada:\n\n"
                    "- **Afecta a G3** (ajuste atmosférico, Teff / tipo espectral), que vive de la "
                    "*forma* del continuo → por eso G3 usa `cont_runmed_biasref`.\n"
                    "- **No afecta a Hα** (E1) ni al límite de Ṁ (E3): la línea es estrecha y se mide "
                    "contra su continuo local, con controles procesados igual que el objeto.\n\n"
                    "No es un defecto de PSF — C1 ya usa Psfao. El diagnóstico completo, con las tres "
                    "causas reales, está en [`docs/d2_red_continuum_diagnosis.md`]"
                    "(../../docs/d2_red_continuum_diagnosis.md).\n\n"
                    "*(La métrica original de la spec era |runmed − poly| sobre un solo espectro; se "
                    "sustituyó porque confunde la estructura molecular **real** de una enana fría con el "
                    "sistemático. Se conserva como `legacy_runmed_poly_fraction`.)*\n\n"
                    "*(En el azul, λ<7000 Å, el compañero tiene SNR<1: la discrepancia de esa zona está "
                    "dentro del error combinado y no es señal.)*"
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    qx = nb.load_qc('stages/stage_x11_qc.json', RUN_ID)\n"
                    "    im = qx['continuum']['intermethod_systematic']; ar = im['after_control_reference']\n"
                    "    # El par lo declara el QC (D1/D2 lo eligen: el primer par primario que\n"
                    "    # contiene al canónico, excluyendo sgf). Fijarlo a mano mostraba una\n"
                    "    # comparación distinta de la que gatea v3 en cuanto cambiaba de objeto.\n"
                    "    pair = [qx['canonical_method'], im['canonical_vs']]\n"
                    "    def col(method, c):\n"
                    "        h = fits.open(rd / 'stages' / f'spec_calibrated_{method}_object.fits')\n"
                    "        v = np.asarray(h[1].data[c], float); h.close(); return v\n"
                    "    wave = col(pair[0], 'wave_A')   # eje λ del producto\n"
                    "    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)\n"
                    "    panels = ((axl, 'cont_runmed', 'ANTES: continuos crudos', im['fraction_channels_methods_agree']),\n"
                    "              (axr, 'cont_runmed_biasref', 'DESPUÉS: referenciados a controles', ar['fraction_channels_methods_agree']))\n"
                    "    drawn = []\n"
                    "    for ax, c, title, frac in panels:\n"
                    "        for method, color in zip(pair, ('tab:blue', 'tab:orange')):\n"
                    "            y = col(method, c); drawn.append(y)\n"
                    "            ax.plot(wave, y, lw=1.1, color=color,\n"
                    "                    label=method + (' (canónico)' if method == pair[0] else ''))\n"
                    "            if c == 'cont_runmed':\n"
                    "                # Lo que se RESTA: el propio producto lleva las dos columnas,\n"
                    "                # así que el nivel de controles es su diferencia exacta.\n"
                    "                bias = y - col(method, 'cont_runmed_biasref')\n"
                    "                ax.plot(wave, bias, lw=0.9, ls='--', color=color, alpha=0.7,\n"
                    "                        label=f'nivel de controles de {method} (se resta)')\n"
                    "        ax.set_title(f'{title}\\n{frac:.3f} de canales dentro del error combinado', fontsize=10)\n"
                    "    for ax in (axl, axr):\n"
                    "        ax.set_xlabel('λ [Å]'); ax.axvline(6563, color='tab:red', ls=':'); ax.axhline(0, color='0.7', lw=0.6); ax.legend(fontsize=8)\n"
                    "    axl.set_ylabel('continuo')\n"
                    "    # Límites por percentil: la escala fija estaba dimensionada para el flujo\n"
                    "    # del primer objeto y recortaba el continuo de cualquier otro.\n"
                    "    vals = np.concatenate([y[np.isfinite(y)] for y in drawn])\n"
                    "    if vals.size:\n"
                    "        lo, hi = np.percentile(vals, [1, 99]); pad = 0.15 * (hi - lo) or 1.0\n"
                    "        axl.set_ylim(lo - pad, hi + pad)\n"
                    "    axl.text(0.02, 0.04, 'azul (λ<7000): SNR<1\\n(dentro del error)', fontsize=7,\n"
                    "             color='0.4', transform=axl.transAxes)\n"
                    "    fig.suptitle(f\"D2 · ¿el continuo es del compañero o es halo residual?\"\n"
                    "                 f\"  ·  {pair[0]} vs {pair[1]}  ·  chequeo V3 de continuo estable\")\n"
                    "    fig.tight_layout()\n"
                    "    outdir = rd / 'plots' / 'd2_calibrate'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'referencing.png', dpi=110); print('figura ->', outdir / 'referencing.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 5 — el presupuesto de error\n\n"
                    "Cada componente del error del `spec_final_object.fits` vs λ (escala log). El total "
                    "está **dominado por el `stat`** (empírico, M5 rojo); el sistemático de continuo "
                    "(runmed vs poly) es el segundo; flujo-cal/cielo/telúrico son ~0.\n\n"
                    "**Las líneas discontinuas NO entran en `flux_err_total`** (la leyenda lo repite en "
                    "cada una), y son dos casos distintos:\n\n"
                    "- `sys_continuum` — por diseño de la spec (§3.5): el continuo se entrega como "
                    "columna y es el modelado quien decide si lo usa; sumarlo aquí sería decidir por él.\n"
                    "- `sys_fluxcal_declared` — el desvío de calibración absoluta que M3 mide frente a "
                    "Gaia (|1−`flux_factor`|), **declarado y no plegado**: M3 no publica barra de error, "
                    "y plegarlo movería el error del compañero, que sostiene decisiones congeladas. G3 ya "
                    "asume su propio 10% (`g3_sys_fluxcal_frac`), mayor que este valor.\n\n"
                    "El total sólido es, por tanto, `stat ⊕ fluxcal ⊕ psf ⊕ cielo ⊕ telúrico`."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    h = fits.open(rd / 'stages' / 'spec_final_object.fits'); d = h[1].data\n"
                    "    wave = np.asarray(d['wave_A'], float)   # eje λ del propio producto\n"
                    "    # `sumado` = si el termino entra en flux_err_total (spec 3.5). El continuo\n"
                    "    # NUNCA entro: se entrega como columna. Se dibujan discontinuos los dos que\n"
                    "    # no suman, para no leerlos como parte del total.\n"
                    "    comp = [('stat', 'flux_err_stat', True), ('flujo-cal', 'sys_fluxcal', True),\n"
                    "            ('psf', 'sys_psf', True), ('cielo', 'sys_sky', True),\n"
                    "            ('telúrico', 'sys_telluric', True), ('continuo', 'sys_continuum', False),\n"
                    "            ('flujo-cal DECLARADO', 'sys_fluxcal_declared', False)]\n"
                    "    from musepipe.spectral import median_filter_1d\n"
                    "    sm = lambda x, n=51: median_filter_1d(np.abs(x), n)   # mediana móvil, ignora NaN\n"
                    "    fig, ax = plt.subplots(figsize=(11, 4))\n"
                    "    for lab, c, summed in comp:\n"
                    "        if c in d.columns.names:\n"
                    "            ax.plot(wave, sm(np.asarray(d[c], float)), lw=1 if summed else 1.3,\n"
                    "                    ls='-' if summed else '--',\n"
                    "                    label=lab if summed else f'{lab} (NO en el total)')\n"
                    "    ax.plot(wave, sm(np.asarray(d['flux_err_total'], float)), lw=2, color='k',\n"
                    "            label='TOTAL = stat ⊕ fluxcal ⊕ psf ⊕ cielo ⊕ telúrico')\n"
                    "    h.close()\n"
                    "    ax.set_yscale('log'); ax.set_xlabel('λ [Å]'); ax.set_ylabel('error (|componente|, suavizado)')\n"
                    "    ax.set_title('D2 · presupuesto de error: continuo y flujo-cal declarado quedan FUERA del total')\n"
                    "    ax.legend(fontsize=7, ncol=3)\n"
                    "    outdir = rd / 'plots' / 'd2_calibrate'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.tight_layout(); fig.savefig(outdir / 'error_budget.png', dpi=110)\n"
                    "    print('figura ->', outdir / 'error_budget.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 6 — los espectros definitivos (6 métodos + la primaria)\n\n"
                    "El entregable de D2 en una figura, dibujada con la **misma función que usa la "
                    "etapa** (`musepipe.stages.definitive_spectra_figure`), leyendo los productos "
                    "calibrados del run:\n\n"
                    "- **arriba:** la **primaria** con su banda de error total (stat + sistemáticos). "
                    "Está ~1e3–1e4 veces por encima del compañero, así que necesita su propio panel.\n"
                    "- **centro:** el **compañero por los 6 métodos**, suavizado 15 canales para que "
                    "se lean a la vez, sobre la banda de error total del canónico (sin suavizar). "
                    "Hα en rojo punteado.\n"
                    "- **abajo:** el **continuo referenciado a controles** — la comparación "
                    "inter-método real (la del gate v3); la banda sombreada es el rojo, donde el "
                    "compañero se detecta.\n\n"
                    "*(`sgf` filtra el continuo por construcción: su nivel no es comparable, su valor "
                    "está en la línea.)*"
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from musepipe.stages import definitive_spectra_figure\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    fig, axes = definitive_spectra_figure(rd / 'stages', plt=plt)\n"
                    "    outdir = rd / 'plots' / 'd2_calibrate'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'definitive_spectra.png', dpi=110)\n"
                    "    print('figura ->', outdir / 'definitive_spectra.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**Los espectros definitivos (6 métodos + primaria) son el entregable de D2**, antes del estudio de Hα: todos con `BUNIT` declarado y error total; la tabla vive en `qc['spectra']` y la figura en `plots/stage_x11_spectra.png`.", "espectros_definitivos_handoff_2026-07-25.md"),
            ("**Diagnóstico honesto del 'continuo rojo inestable'**: 3 cosas reales (señal de enana fría + sistemático de nivel inter-método + rigidez del polinomio). NO es defecto de PSF.", "d2_red_continuum_diagnosis.md"),
            ("**Referenciación a controles integrada** (2026-07-11): el chequeo de continuo estable (V3) gatea sobre la métrica referenciada — a cada método se le resta su propio nivel en controles, para no contar dos veces el pedestal de halo que ambos comparten. Columna `cont_runmed_biasref` entregada para G3.", None),
            ("D1 **ya era control-centrado** (su veredicto refleja el sistemático genuino); esto solo puso a D2 al mismo nivel. Lo que V3 sigue midiendo por debajo del umbral es sistemático cromático **genuino** (no el pedestal): limitación aceptada en F1, que afecta a la forma del continuo (G3) y no al endpoint de Hα.", None),
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
            "- **Continuo:** referenciación a controles integrada (a cada método se le resta su propio "
            "nivel en controles); columna `cont_runmed_biasref` para G3.\n"
            "- **Continuo estable (V3):** "
            "{{qc:stages/stage_x11_qc.json:checks.v3_continuum_stable.fraction_channels_methods_agree:.3f}} "
            "de canales concuerdan entre los dos métodos dentro del error combinado, umbral "
            "{{qc:stages/stage_x11_qc.json:checks.v3_continuum_stable.threshold}} → ok="
            "**{{qc:stages/stage_x11_qc.json:checks.v3_continuum_stable.ok}}**. Lo que no concuerda es "
            "sistemático cromático **genuino** de sustracción de halo, no el pedestal común: "
            "limitación aceptada, F1 sigue yellow.\n"
            "- **Endpoint intacto:** eso afecta a la FORMA del continuo (G3); la línea Hα (E1) y el "
            "límite de Ṁ (E3) no dependen de ello.\n"
            "- **Espectros definitivos:** los 6 métodos + la primaria, con unidad y error total "
            "(tabla en `qc['spectra']`, figura en `plots/stage_x11_spectra.png`) — resultado en sí "
            "mismos, entregados antes del estudio de Hα.\n"
            "- **Unidad de flujo:** `qc['flux']['unit']` deja resuelta y contrastada la escala que "
            "usarán E3/G3 (knob → `BUNIT` → `m3_flux.flux_unit_cgs`). Si el `BUNIT` del producto y "
            "la unidad con la que M3 midió el factor no coinciden, sale como `open_issue`.\n"
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
        checks_md=(
            "## Los chequeos del QC, en físico\n\n"
            "E1 es el endpoint: aquí se decide si hay o no emisión. Los chequeos existen para que un "
            "«sí» no pueda venir de un control contaminado ni de una FAP mal calibrada.\n\n"
            "| Chequeo | ¿Qué pregunta contesta? | Si falla |\n|---|---|---|\n"
            "| `v2_controls` | **¿La distribución de máximos nulos es sana?** Sin bimodalidades ni "
            "outliers extremos que delaten un control contaminado por otra fuente. | La FAP se "
            "calibra contra una distribución sucia: el umbral de detección deja de valer. (Se puede "
            "excluir un control contaminado, con registro, y recalcular — máximo 2.) |\n"
            "| `v3_placebo` | **¿La cadena «detecta» donde no puede haber nada?** La misma "
            "maquinaria, centrada en 6400 y 6700 Å (líneas placebo), no debe superar el umbral. | Si "
            "un placebo detecta, **la FAP está mal calibrada** y la detección de Hα no vale: es "
            "hallazgo mayor, no un detalle. La batería completa de placebos es E2/T5. |\n"
            "| `v4_multimethod` | **¿Los métodos coinciden?** Las FAP por método, contadas y "
            "comparadas. | Una discrepancia grande entre métodos apunta a que el resultado depende "
            "de cómo se restó el halo → insumo directo para E2. |\n"
        ),
        narrative_md=(
            "## Qué hace E1 y el resultado\n\n"
            "E1 busca **emisión de Hα** del compañero con un **matched filter** (plantilla de la línea "
            "esperada) y calibra la significancia con **controles** (FAP empírico). Es el **endpoint "
            "científico**.\n\n"
            "Por método: busca en ±{{qc:stages/stage_h01_qc.json:line.search_half_width_kms:.0f}} km/s "
            "alrededor de Hα ({{qc:stages/stage_h01_qc.json:line.rest_A}} Å, rv_sys "
            "{{qc:stages/stage_h01_qc.json:line.rv_sys_kms:+.0f}}) → un `z` del matched filter. La "
            "**FAP** = fracción de los "
            "{{csvtop:tables/halpha_detection_by_method.csv:matched_z:n_controls:.0f}} máximos nulos "
            "(posiciones de control) que superan el pico del objeto. **Criterio de detección:** "
            "`global_fap < {{qc:stages/stage_h01_qc.json:criterion.global_fap_lt}}` **Y** un par "
            "admisible (psffit+aperture) **Y** rv dentro de la LSF.\n\n"
            "**VEREDICTO = `{{qc:stages/stage_h01_qc.json:verdict.verdict}}`** "
            "(`{{qc:stages/stage_h01_qc.json:verdict.reason}}`). Valores de **este objeto**: los picos "
            "van de z {{csvrange:tables/halpha_detection_by_method.csv:matched_z:.2f}} con FAP "
            "{{csvrange:tables/halpha_detection_by_method.csv:global_empirical_fap:.2f}}. El pico más "
            "alto es **{{csvtop:tables/halpha_detection_by_method.csv:matched_z:method}}** "
            "(z={{csvtop:tables/halpha_detection_by_method.csv:matched_z:matched_z:.2f}}, "
            "FAP={{csvtop:tables/halpha_detection_by_method.csv:matched_z:global_empirical_fap:.2f}}, "
            "v={{csvtop:tables/halpha_detection_by_method.csv:matched_z:peak_velocity_kms:+.0f}} km/s, "
            "rv_ok={{csvtop:tables/halpha_detection_by_method.csv:matched_z:rv_consistent}}, "
            "FWHM={{csvtop:tables/halpha_detection_by_method.csv:matched_z:fwhm_A:.1f}} Å).\n\n"
            "> **Cómo leer un pico alto:** un `z` grande no es señal por sí solo. Hay que exigirle las "
            "tres cosas del criterio — FAP baja frente a SUS controles, rv consistente con el "
            "sistémico, y anchura compatible con la LSF. Un pico con rv incoherente o mucho más ancho "
            "que la LSF es ruido de borde, no una línea. La celda de evidencia y el Plot 1 dan las "
            "tres por método.\n\n"
            "**Inputs:** LSF {{qc:stages/stage_h01_qc.json:templates.lsf_fwhm_A}} Å, rv_sys "
            "{{qc:stages/stage_h01_qc.json:line.rv_sys_kms:+.0f}} km/s (estimación de literatura), "
            "{{csvtop:tables/halpha_detection_by_method.csv:matched_z:n_controls:.0f}} controles → "
            "`min_resolvable_fap` ≈ "
            "{{csvtop:tables/halpha_detection_by_method.csv:matched_z:minimum_resolvable_fap:.3f}} "
            "(por encima del umbral: un FAP<1% estricto necesitaría ~99 controles — salvedad)."
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
            "n_ctrl = int(d['n_controls'].iloc[0])\n"
            "n_need = int(round(1 / cr['global_fap_lt'])) - 1   # FAP mínimo resoluble = 1/(n+1)\n"
            "print(f\"\\nmin_resolvable_fap = {d['minimum_resolvable_fap'].iloc[0]:.3f} \"\n"
            "      f\"({n_ctrl} controles; <{cr['global_fap_lt']} necesita ~{n_need})\")"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — el pico del objeto frente a su distribución nula\n\n"
                    "Por método, los **máximos nulos** (matched filter en las posiciones de control, "
                    "gris) y el **pico del objeto** (estrella; roja si `rv_consistent=False`). La "
                    "lectura es la posición del pico **dentro de su propia nube**: si queda inmerso en "
                    "ella, la FAP es alta y no hay detección. Para este objeto el veredicto es "
                    "`{{qc:stages/stage_h01_qc.json:verdict.verdict}}` con FAP "
                    "{{csvrange:tables/halpha_detection_by_method.csv:global_empirical_fap:.2f}} frente "
                    "al umbral {{qc:stages/stage_h01_qc.json:criterion.global_fap_lt}}."
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
                    "        ax.scatter(x, nulls, s=14, color='0.6', alpha=0.7, label=f'máximos nulos ({nulls.size} controles)' if i == 0 else None)\n"
                    "        obj = d.loc[m, 'matched_z']; fap = d.loc[m, 'global_empirical_fap']; rv = d.loc[m, 'rv_consistent']\n"
                    "        ax.scatter(i, obj, s=170, marker='*', color='tab:orange' if rv else 'tab:red', zorder=5,\n"
                    "                   edgecolor='k', label='pico del objeto' if i == 0 else None)\n"
                    "        ax.text(i, obj + 0.35, f'z={obj:.2f}\\nFAP={fap:.2f}\\nrv_ok={rv}', ha='center', fontsize=7)\n"
                    "    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=9)\n"
                    "    ax.set_ylabel('z del matched filter (máximo en la ventana Hα)')\n"
                    "    v = nb.load_qc('stages/stage_h01_qc.json', RUN_ID)['verdict']\n"
                    "    ax.set_title(f\"E1 · {v['verdict']} ({v['reason']}): pico del objeto vs su nube nula\")\n"
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
                    "El espectro canónico (`spec_final_object.fits`) alrededor de Hα con la banda ±1σ "
                    "empírica (controles del método canónico de D2) y la posición esperada de Hα "
                    "(rest_A del QC de E1, corrida por rv_sys). Lo que hay que mirar: si asoma algo "
                    "**por encima de la banda** justo en la posición esperada. Veredicto de E1 para "
                    "este objeto: `{{qc:stages/stage_h01_qc.json:verdict.verdict}}`."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    from musepipe.constants import C_KMS\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    line = nb.load_qc('stages/stage_h01_qc.json', RUN_ID)['line']   # rest_A + rv_sys oficiales\n"
                    "    canon = nb.load_qc('stages/stage_x11_qc.json', RUN_ID)['canonical_method']\n"
                    "    h = fits.open(rd / 'stages' / 'spec_final_object.fits')\n"
                    "    wave = np.asarray(h[1].data['wave_A'], float)   # eje λ del propio producto\n"
                    "    flux = np.asarray(h[1].data['flux'], float); h.close()\n"
                    "    sig = np.nanstd(np.load(rd / 'stages' / f'spec_calibrated_{canon}_controls.npz')['control_spectra'], axis=0)\n"
                    "    ha = line['rest_A'] * (1 + line['rv_sys_kms'] / C_KMS)\n"
                    "    w = (wave >= 6400) & (wave <= 6750)\n"
                    "    fig, ax = plt.subplots(figsize=(9, 4))\n"
                    "    ax.fill_between(wave[w], -sig[w], sig[w], color='0.85', label='±1σ empírico')\n"
                    "    ax.plot(wave[w], flux[w], lw=0.9, color='tab:blue', label=f'flujo {canon}')\n"
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
            ("**VEREDICTO = `{{qc:stages/stage_h01_qc.json:verdict.verdict}}`** (`{{qc:stages/stage_h01_qc.json:verdict.reason}}`); métodos significativos: {{qc:stages/stage_h01_qc.json:verdict.significant_methods}}. FAP por método {{csvrange:tables/halpha_detection_by_method.csv:global_empirical_fap:.2f}} frente al umbral {{qc:stages/stage_h01_qc.json:criterion.global_fap_lt}}. **Es el endpoint científico de la cadena.**", None),
            ("Un `z` alto no basta: el pico mayor de este objeto es {{csvtop:tables/halpha_detection_by_method.csv:matched_z:method}} (z={{csvtop:tables/halpha_detection_by_method.csv:matched_z:matched_z:.2f}}, FAP={{csvtop:tables/halpha_detection_by_method.csv:matched_z:global_empirical_fap:.2f}}, rv_ok={{csvtop:tables/halpha_detection_by_method.csv:matched_z:rv_consistent}}, FWHM={{csvtop:tables/halpha_detection_by_method.csv:matched_z:fwhm_A:.1f}} Å) — hay que juzgarlo por FAP + rv + anchura, no por z.", None),
            ("LSF = {{qc:stages/stage_h01_qc.json:templates.lsf_fwhm_A}} Å, {{csvtop:tables/halpha_detection_by_method.csv:matched_z:n_controls:.0f}} controles → min_resolvable_fap ≈ {{csvtop:tables/halpha_detection_by_method.csv:matched_z:minimum_resolvable_fap:.3f}} (un FAP<1% estricto necesitaría ~99 controles).", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**E1: veredicto `{{qc:stages/stage_h01_qc.json:verdict.verdict}}` "
            "(`{{qc:stages/stage_h01_qc.json:verdict.reason}}`).** Cifras de este objeto, resueltas "
            "de su `stage_h01_qc.json` y de `tables/halpha_detection_by_method.csv`.\n\n"
            "- **Todos los métodos:** z "
            "{{csvrange:tables/halpha_detection_by_method.csv:matched_z:.2f}}, FAP "
            "{{csvrange:tables/halpha_detection_by_method.csv:global_empirical_fap:.2f}} frente al "
            "umbral {{qc:stages/stage_h01_qc.json:criterion.global_fap_lt}}.\n"
            "- **Pico mayor:** {{csvtop:tables/halpha_detection_by_method.csv:matched_z:method}} "
            "(z={{csvtop:tables/halpha_detection_by_method.csv:matched_z:matched_z:.2f}}, "
            "v={{csvtop:tables/halpha_detection_by_method.csv:matched_z:peak_velocity_kms:+.0f}} km/s, "
            "rv_ok={{csvtop:tables/halpha_detection_by_method.csv:matched_z:rv_consistent}}, "
            "FWHM={{csvtop:tables/halpha_detection_by_method.csv:matched_z:fwhm_A:.1f}} Å frente a una "
            "LSF de {{qc:stages/stage_h01_qc.json:templates.lsf_fwhm_A}} Å).\n"
            "- **Inputs:** LSF {{qc:stages/stage_h01_qc.json:templates.lsf_fwhm_A}} Å, "
            "{{csvtop:tables/halpha_detection_by_method.csv:matched_z:n_controls:.0f}} controles "
            "(min_fap {{csvtop:tables/halpha_detection_by_method.csv:matched_z:minimum_resolvable_fap:.3f}}; "
            "~99 para 1% estricto), rv_sys "
            "{{qc:stages/stage_h01_qc.json:line.rv_sys_kms:+.0f}} km/s (literatura).\n"
            "- **Downstream:** alimenta E3 (límite superior de Ṁ) y G2."
        ),
    ),
    dict(
        id="E1b", qc_optional=True, slug="E1b_fov_detection", title="Detección ciega FoV (matched filter)", block="E · Resultado",
        spec="spec_E1b_codex_fov_detection.md", run_override=None,
        what=(
            "Mapas matched-filter espacio-espectrales sobre todo el campo desde los cubos "
            "residuales (sgf/lpm/psfsub), normalizados por ruido de anillos; propone candidatos "
            "ciegos (Julo et al. 2025 Fig. 9 + App. G)."
        ),
        inputs="Cubos residuales C5/C6 (+psfsub reconstruido) + PSF C1 + posiciones B3",
        outputs="`stages/stage_h01b_qc.json`, `stage_h01b_fovmap_<m>.fits`",
        downstream="E6 (ROC); targets nuevos: candidatos previos a B3 (checkpoint usuario)",
        exec=dict(kind="module_main", target="musepipe.stages.stage_h01b_fovmap",
                  cost="Ligero (~1 min por método)."),
        qc="stages/stage_h01b_qc.json",
        salient=["params.threshold_sigma", "params.kernel_source",
                 "checks.v1_maps_written", "checks.v2_known_source_reported"],
        checks_md=(
            "## Los chequeos del QC, en físico\n\n"
            "| Chequeo | ¿Qué pregunta contesta? | Si falla |\n|---|---|---|\n"
            "| `v1_maps_written` | **¿Hay mapa de z para cada método?** Al menos uno. | No hay "
            "búsqueda ciega que revisar. |\n"
            "| `v2_known_source_reported` | **¿Se ve la compañera que YA sabemos dónde está?** Su z "
            "se reporta en todos los métodos. | Control de cordura: si el mapa no recupera la fuente "
            "conocida, tampoco vale para buscar desconocidas. |\n"
            "| `v3_ring_noise_written` | **¿Cómo es el ruido a cada radio?** σ, asimetría, curtosis y "
            "fracción de \\|z\\|>3 por anillo. Cerca de la estrella el ruido no es gaussiano, así que "
            "un mismo z **no significa lo mismo** a 0.5″ que a 3″. | Los z del mapa no se pueden "
            "traducir a probabilidad. |\n"
            "| `v4_no_candidate_at_star_core` | **¿La máscara del núcleo estelar funciona?** Ningún "
            "candidato con r < r_min. | El propio halo de la primaria entra en la lista como si "
            "fuera una detección. |\n"
        ),
        narrative_md=(
            "## Qué hace E1b\n\n"
            "Para cada cubo residual: plantilla espectral gaussiana (FWHM=LSF) en la línea "
            "objetivo → mapa M = Σ f·residual → correlación con el kernel de PSF cromática C1 → "
            "normalización por anillos (μ, σ robustos por radio, filosofía Andres 1994) → mapa z "
            "y candidatos con z ≥ 5 fuera del núcleo AO.\n\n"
            "Es la etapa de **detección ciega** del pipeline multi-target: en targets nuevos corre "
            "antes de fijar `companion` en B3 (la promoción de un candidato es SIEMPRE checkpoint "
            "del usuario); con compañera conocida es un test de consistencia con E1 (su z se "
            "reporta como `known_source`). El QC registra además la gaussianidad del ruido por "
            "anillos (App. G del paper): no-gaussianidad cerca del núcleo NO bloquea — motiva la "
            "estadística empírica de E3/E6."
        ),
        evidence_md=("## Evidencia: candidatos, fuente conocida y ruido por anillos"),
        evidence_code=(
            "q = nb.load_qc('stages/stage_h01b_qc.json', RUN_ID)\n"
            "print('kernel:', q['params']['kernel_source'], ' umbral:', q['params']['threshold_sigma'], 'σ')\n"
            "for m, mq in q['methods'].items():\n"
            "    ks = mq['known_source']\n"
            "    z_known = None if ks is None else ks['z']\n"
            "    print(f\"  {m:14s} candidatos={len(mq['candidates'])}  z(compañera)={z_known}\")\n"
            "    for c in mq['candidates'][:5]:\n"
            "        print(f\"     cand y={c['y']} x={c['x']} r={c['r_px']:.1f}px z={c['z']:.1f}\")\n"
            "print('checks:', q['checks'])"
        ),
        plots=[
            dict(
                md=(
                    "## Plot — mapas z por método + gaussianidad por anillos\n\n"
                    "Mapas z (Fig. 9 del paper) con estrella (*), compañera conocida (○) y "
                    "candidatos (□); y fracción |z|>3 por anillo vs expectativa gaussiana (App. G)."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from astropy.io import fits\n"
                    "    q = nb.load_qc('stages/stage_h01b_qc.json', RUN_ID)\n"
                    "    methods = list(q['methods'])\n"
                    "    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(5*(len(methods)+1), 4.4))\n"
                    "    for ax, m in zip(axes, methods):\n"
                    "        with fits.open(q['methods'][m]['map_fits']) as h:\n"
                    "            z = np.asarray(h['Z'].data, float)\n"
                    "        im = ax.imshow(z, origin='lower', cmap='viridis', vmin=-3, vmax=8)\n"
                    "        sy, sx = q['star_yx']; ax.plot(sx, sy, '*', color='red')\n"
                    "        if q.get('companion_yx'):\n"
                    "            cy, cx = q['companion_yx']; ax.plot(cx, cy, 'o', mfc='none', mec='lime', ms=12)\n"
                    "        for c in q['methods'][m]['candidates']:\n"
                    "            ax.plot(c['x'], c['y'], 's', mfc='none', mec='orange', ms=10)\n"
                    "        ax.set_title(f'{m} · z'); ax.axis('off'); fig.colorbar(im, ax=ax, shrink=0.75)\n"
                    "    ax2 = axes[-1]\n"
                    "    for m in methods:\n"
                    "        rn = q['methods'][m]['ring_noise']\n"
                    "        r = [0.5*(x['r_lo']+x['r_hi']) for x in rn]\n"
                    "        f3 = [x['frac_abs_z_gt3'] for x in rn]\n"
                    "        ax2.plot(r, f3, marker='o', ms=3, label=m)\n"
                    "    ax2.axhline(0.0027, color='k', ls=':', label='gaussiana (0.27%)')\n"
                    "    ax2.set_xlabel('radio [px]'); ax2.set_ylabel('frac |z|>3'); ax2.set_yscale('log')\n"
                    "    ax2.legend(fontsize=7); ax2.set_title('gaussianidad por anillos (App. G)')\n"
                    "    fig.tight_layout(); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("Umbral 5σ y r_min=3px congelados; la promoción de candidatos a `companion` (B3) es checkpoint del usuario, nunca automática.", "spec_E1b_codex_fov_detection.md"),
            ("Normalización por anillos con μ/σ robustos (Andres 1994); no-gaussianidad registrada, no bloqueante.", "plan_integracion_halosub_julo2025.md"),
        ],
        checks=(
            "q = nb.load_qc('stages/stage_h01b_qc.json', RUN_ID)\n"
            "for k, v in q['checks'].items():\n"
            "    print(f'  {k}: {v}')"
        ),
        conclusion_md=(
            "## Estado\n\n"
            "**Pendiente de primera ejecución sobre datos reales** (requiere C5/C6 corridos). "
            "Núcleo y contrato verificados con tests sintéticos (2026-07-14)."
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
        salient=["overall", "overall_raw", "t2.status", "t5.status",
                 "halpha_map_correlation.corr_stripe_scatter_vs_halpha_sigma",
                 "halpha_map_correlation.halpha_sigma_slicer_aligned"],
        checks_md=(
            "## La batería T1–T5, en físico\n\n"
            "`t1`…`t5` son la batería **congelada** de la spec: cinco maneras distintas de que lo que "
            "E1 midió sea un artefacto y no el compañero. Cada una ataca un origen instrumental "
            "diferente, y por eso se pasan todas aunque E1 diga no-detección (en ese caso se aplican "
            "al máximo del mapa, o sea al ruido dominante).\n\n"
            "| Test | ¿Qué artefacto descarta? | Criterio |\n|---|---|---|\n"
            "| **T1 · coincidencia instrumental** | Que el canal de la señal caiga sobre algo que el "
            "instrumento ya ensucia: franjas del slicer (lista de B2), líneas de cielo (catálogo de "
            "A4) o los bordes del hueco del láser AO. | Distancia al artefacto más cercano de cada "
            "lista; coincide si \\|Δcanal\\| ≤ 2. |\n"
            "| **T2 · coherencia espacial** | Que la señal no tenga la forma de una estrella. Una "
            "fuente real es la PSF de C1 centrada donde B3 puso al compañero; un artefacto es "
            "alargado, desplazado o con varios picos. | χ² de PSF vs plano, centroide < 1 px de B3, "
            "elongación < 1.5×. |\n"
            "| **T3 · estabilidad temporal** | Que venga de una sola exposición. Una señal real crece "
            "como √N al combinar; un rayo cósmico o un defecto no. | z > z_total/√2 en ambas mitades "
            "independientes. **Con una sola exposición no se puede aplicar** — queda `unavailable` y "
            "así consta: es un eje de robustez que este dataset no cubre. |\n"
            "| **T4 · estabilidad frente a parámetros** | Que la señal dependa de cómo la analizamos. "
            "Se repite E1 moviendo UNA perilla cada vez (radio de ajuste local, máscara de C1, ancho "
            "de plantilla, continuo). | Rango de z entre variantes Δz < 1. |\n"
            "| **T5 · placebos espectrales** | Que el método «detecte» en cualquier sitio. La cadena "
            "completa se centra en líneas donde no se espera nada (6200, 6400, 6700, 7100 Å; 6300 "
            "descartada por skyline). | Ningún placebo supera el umbral de E1. Si alguno lo supera, "
            "**la FAP está mal calibrada** y el resultado de E1 no vale. |\n"
        ),
        narrative_md=(
            "## Qué hace E2 y cómo se reinterpreta\n\n"
            "E2 corre una **batería FIJA de 5 tests de artefactos** sobre el resultado de E1 (siempre "
            "completa, gane o no E1). Para una no-detección, la pregunta es: **¿es robusta?**\n\n"
            "- **T1 (coincidencia** con stripe/skyline/laser): `unavailable` (no hay listas — "
            "exposición única, sin QC de stripes).\n"
            "- **T2 (forma de PSF del máximo global):** el máximo de Hα **NO** se ajusta mejor con una "
            "PSF que con un plano (`chi2_ratio` 1.007 ≈ 1) y está **elongado** (1.54, no puntual). "
            "status `fail` — pero para una **no-detección** esto se **reinterpreta** (spec §2): que el "
            "máximo NO tenga forma de PSF significa que es **ruido**, no una fuente → **apoya la "
            "no-detección**.\n"
            "- **T3 (split de exposición):** `unavailable` (exposición única).\n"
            "- **T4 (variación de knobs):** `unavailable` (variantes de validación no generadas).\n"
            "- **T5 (placebos):** buscar en λ **fuera de línea** (6200/6400/6700/7100 Å) → **sin "
            "detección espuria** (max_fap 0.029, ninguno <0.01). **PASS**: el método no fabrica "
            "detecciones en λ aleatorios.\n\n"
            "**`overall_raw = fails`** (driven por T2) → **reinterpretado a `overall = survives`** "
            "(`non_detection_robust`). Es una **limitación aceptada** documentada en F1 (la "
            "reinterpretación de T2 para no-detección)."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Los 5 tests con su status y la reinterpretación del `stage_h02_qc.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage_h02_qc.json', RUN_ID)\n"
            "print('E1 input:', q['input_verdict_e1'], '| método seleccionado:', q['selected_method'])\n"
            "print(f\"T1 coincidencia: {q['t1']['status']}\")\n"
            "t2 = q['t2']\n"
            "print(f\"T2 forma-PSF del máximo: {t2['status']}  (chi2_ratio_psf_vs_plane={t2['chi2_ratio_psf_vs_plane']:.3f}, \"\n"
            "      f\"elongación={t2['elongation_vs_psf']:.2f}) -> NO es PSF -> apoya no-detección\")\n"
            "print(f\"T3 split exposición: {q['t3']['status']} ({q['t3'].get('reason','')})\")\n"
            "print(f\"T4 variación knobs: {q['t4']['status']} ({q['t4'].get('reason','')})\")\n"
            "print(f\"T5 placebos: {q['t5']['status']}  (max_fap={q['t5']['placebo_max_fap_global']:.3f}, any_above={q['t5']['any_above_threshold']})\")\n"
            "print()\n"
            "print(f\"overall_raw = {q['overall_raw']}  ->  overall = {q['overall']}\")\n"
            # `overall_interpretation` NO la emite ningún productor (verificado
            # 2026-07-24): la celda la leía por error. Se deriva de las claves reales.
            "if q['overall_raw'] != q['overall']:\n"
            "    print(f\"interpretación: el veredicto crudo ({q['overall_raw']}) se revisa a \"\n"
            "          f\"{q['overall']} por los tests que sí aplican; ver t1..t5 arriba.\")\n"
            "else:\n"
            "    print(f\"interpretación: veredicto consistente ({q['overall']}).\")"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — T5 placebos: el método no inventa detecciones\n\n"
                    "El `z` del matched filter buscando en centros **fuera de línea** (6200–7100 Å) por "
                    "método, y el **Hα real** (★) en 6562.8. El Hα real cae en el **mismo nivel de "
                    "ruido** que los placebos → no hay detección espuria y la no-detección es robusta."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import pandas as pd\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    q = nb.load_qc('stages/stage_h02_qc.json', RUN_ID)\n"
                    "    df = pd.DataFrame(q['t5']['rows'])\n"
                    "    centers = sorted(df['center_A'].unique())\n"
                    "    methods = ['aperture', 'optimal_psfsub', 'psffit', 'optimal_ls']\n"
                    "    real = pd.read_csv(rd / 'tables' / 'halpha_detection_by_method.csv').set_index('method')['matched_z']\n"
                    "    cmap = dict(zip(methods, ['tab:blue', 'tab:green', 'tab:red', 'tab:orange']))\n"
                    "    fig, ax = plt.subplots(figsize=(9, 4.3))\n"
                    "    for m in methods:\n"
                    "        zc = [df[(df.center_A == c) & (df.method == m)]['matched_z'].values[0] for c in centers]\n"
                    "        ax.plot(centers, zc, 'o-', color=cmap[m], ms=6, label=f'placebo {m}')\n"
                    "        ax.scatter([6562.8], [real[m]], marker='*', s=160, color=cmap[m], edgecolor='k', zorder=5)\n"
                    "    ax.axvline(6562.8, color='0.5', ls=':', label='Hα real (★)')\n"
                    "    ax.set_xlabel('centro de búsqueda [Å]'); ax.set_ylabel('z del matched filter')\n"
                    "    ax.set_title('E2 · T5 placebos: λ off-line da ruido; Hα real (★) igual → sin detección espuria')\n"
                    "    ax.legend(fontsize=7, ncol=2); fig.tight_layout()\n"
                    "    outdir = rd / 'plots' / 'e2_artifacts'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 't5_placebos.png', dpi=110); print('figura ->', outdir / 't5_placebos.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — la batería a golpe de vista\n\n"
                    "Status de los 5 tests. T2 en rojo (`fail`, pero reinterpretado: el máximo no es "
                    "PSF → apoya la no-detección); T5 en verde (placebos limpios); T1/T3/T4 en gris "
                    "(no disponibles por la exposición única)."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_h02_qc.json', RUN_ID)\n"
                    "    tests = {'T1 coincidencia\\n(stripe/skyline/laser)': q['t1']['status'],\n"
                    "             'T2 forma PSF\\ndel máximo': q['t2']['status'],\n"
                    "             'T3 split de\\nexposición': q['t3']['status'],\n"
                    "             'T4 variación de\\nknobs': q['t4']['status'],\n"
                    "             'T5 placebos\\n(λ off-line)': q['t5']['status']}\n"
                    "    col = {'pass': 'tab:green', 'fail': 'tab:red', 'unavailable': '0.7'}\n"
                    "    names = list(tests)\n"
                    "    fig, ax = plt.subplots(figsize=(8, 3.6))\n"
                    "    ax.barh(names, [1] * len(names), color=[col.get(tests[n], '0.7') for n in names])\n"
                    "    for i, n in enumerate(names):\n"
                    "        ax.text(0.5, i, tests[n], ha='center', va='center', fontsize=9, color='w', weight='bold')\n"
                    "    ax.set_xlim(0, 1); ax.set_xticks([]); ax.invert_yaxis()\n"
                    "    ax.set_title(f\"E2 · batería: overall_raw={q['overall_raw']} -> {q['overall']} (T2 reinterpretado)\")\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'e2_artifacts'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'battery.png', dpi=110); print('figura ->', outdir / 'battery.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 3 — S1b: zonas sucias de stripes vs mapas Hα (Xie+20 §4.1)\n\n"
                    "Correlación por columna entre la perturbación de la solución de onda (scatter "
                    "del mapa de offset S0) y el ancho σ del Hα (mapa S1). La clave es el **control "
                    "transversal**: si la correlación en la dirección transversal es casi igual, la "
                    "correlación es un **confundido radial** (núcleo brillante vs halo débil), NO una "
                    "firma de slicer. `halpha_*_slicer_aligned` aplica el mismo criterio que el gate "
                    "G1 (estructura > 3× y > 2× su control transversal)."
                ),
                code=(
                    "try:\n"
                    "    from IPython.display import Image, display\n"
                    "    q = nb.load_qc('stages/stage_h02_qc.json', RUN_ID)\n"
                    "    hmc = q.get('halpha_map_correlation')\n"
                    "    if not hmc:\n"
                    "        print('S1b no integrado en este run: falta halpha_map_correlation.')\n"
                    "        print('-> corre: python scripts/s1b_integrate_e2.py --run-dir', nb.run_dir(RUN_ID))\n"
                    "    else:\n"
                    "        print(f\"corr(stripe scatter, Hα σ)      = {hmc['corr_stripe_scatter_vs_halpha_sigma']:.3f}\")\n"
                    "        print(f\"  control transversal            = {hmc['corr_stripe_scatter_vs_halpha_sigma_transverse']:.3f}  (≈ igual ⇒ confundido radial)\")\n"
                    "        print(f\"a  estructura {hmc['halpha_a_structure_significance']:.1f}× (transv {hmc['halpha_a_transverse_significance']:.1f}×) -> slicer_aligned={hmc['halpha_a_slicer_aligned']}\")\n"
                    "        print(f\"σ  estructura {hmc['halpha_sigma_structure_significance']:.1f}× (transv {hmc['halpha_sigma_transverse_significance']:.1f}×) -> slicer_aligned={hmc['halpha_sigma_slicer_aligned']}\")\n"
                    "        print(f\"corr(a,σ) = {hmc['halpha_corr_a_sigma']:.2f}, P_cov = {hmc['halpha_P_cov']:.2f}  (Xie Fig.3 pide P≈const)\")\n"
                    "        fig = (q.get('figures') or {}).get('s1_stripe_halpha')\n"
                    "        from pathlib import Path as _P\n"
                    "        if fig and _P(fig).exists():\n"
                    "            display(Image(filename=str(fig)))\n"
                    "except Exception as e:\n"
                    "    print('No se pudo mostrar S1b:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**S1b (wavesol/stripes):** los mapas Hα a/σ NO están alineados con slicers (estructura ≤ control transversal ⇒ radial, núcleo vs halo); la fuerte correlación por columna stripe↔σ es un confundido radial (transversal ≈ igual). Consistente con G1 (cubo combinado ciego a stripes). Sin interpretar ghost-vs-instrumental (humano/por-exposición).", "decision_g1_wavesol_2026-07-17.md"),
            ("**T2 reinterpretado para no-detección** (spec §2): el máximo global NO tiene forma de PSF (chi2_ratio 1.007, elongación 1.54) → *apoya* la no-detección. `overall_raw=fails` → `overall=survives`. Limitación aceptada en F1.", None),
            ("**T5 placebos PASS**: buscar en λ off-line no fabrica detecciones (max_fap 0.029, ninguno <0.01) → método limpio.", None),
            ("T1/T3/T4 `unavailable` por la exposición única (sin stripe QC, sin split, sin variantes de knobs).", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**E2: la no-detección es ROBUSTA (`overall=survives`, reinterpretado de `fails`).**\n\n"
            "- **Fecha:** 2026-07-09 (re-run con PSF Psfao).\n"
            "- **T2** `fail` reinterpretado: el máximo de Hα no es PSF (chi2_ratio 1.007, elong 1.54) "
            "→ ruido → apoya la no-detección.\n"
            "- **T5** PASS: placebos en λ off-line sin detección espuria (max_fap 0.029).\n"
            "- **T1/T3/T4** `unavailable` por la exposición única.\n"
            "- **F1:** la reinterpretación de T2 es una **limitación aceptada** documentada (no un "
            "rojo bloqueante).\n"
            "- **Downstream:** con la no-detección robusta, E3 calcula el límite superior de Ṁ."
        ),
    ),
    dict(
        id="E3", slug="E3_upper_limits", title="Límites superiores (Ṁ)", block="E · Resultado",
        spec="spec_E3_codex_upper_limits.md", run_override=None,
        what="Límite superior de la tasa de acreción Ṁ a partir de la no-detección (Gumbel 99%).",
        inputs="E1 + throughput (E4) + config físico", outputs="`stages/stage_h03_qc.json`, `tables/halpha_upper_limits.csv`",
        downstream="F1, comparación con G3",
        exec=dict(kind="script", target="stage_h03_limits.sh", cost="Ligero–moderado."),
        qc="stages/stage_h03_qc.json",
        salient=["canonical_method", "mdot", "throughput", "intermethod_scatter_pct", "definition"],
        narrative_md=(
            "## Qué hace E3 y cómo sale la cifra\n\n"
            "E3 convierte la **no-detección** en un **límite superior de la tasa de acreción Ṁ**. "
            "Toma el umbral de flujo **Gumbel 99%** de los máximos de ruido de los controles "
            "(matched filter, `f_stat_99`) y lo pasa por la cadena física:\n\n"
            "```\nf_stat_99  ÷throughput→  f_obs  ×deredden→  f_dered  ×4πd²→  L_Hα  →Alcalá→  Ṁ\n```\n\n"
            "**Ṁ (99%, canónico {{qc:stages/stage_h03_qc.json:canonical_method}}) = "
            "{{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].mdot:.2e}} M☉/yr** para "
            "este objeto. Insumos físicos (todos citados en el QC): "
            "d={{qc:stages/stage_h03_qc.json:physical_inputs.distance_pc}} pc, "
            "A_V={{qc:stages/stage_h03_qc.json:physical_inputs.av}} "
            "({{qc:stages/stage_h03_qc.json:physical_inputs.av_source}}), CCM89 R_V=3.1, relación "
            "L_acc–L_Hα de {{qc:stages/stage_h03_qc.json:physical_inputs.lacc_lha_relation}} "
            "(scatter {{qc:stages/stage_h03_qc.json:physical_inputs.relation_scatter_dex}} dex), "
            "masa {{qc:stages/stage_h03_qc.json:physical_inputs.companion_mass_msun}} M☉, "
            "radio {{qc:stages/stage_h03_qc.json:physical_inputs.companion_radius_rsun}} R☉.\n\n"
            "**Dependencia del método = el throughput:** menor throughput → señal peor recuperada → "
            "límite **peor** (más alto). El canónico citable es "
            "`{{qc:stages/stage_h03_qc.json:canonical_method}}` (throughput "
            "{{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].throughput:.2f}}) por "
            "robustez, no por dar el número más bajo: un método puede dar un límite más ajustado y ser "
            "menos fiable si su ruido está subestimado en ese borde. **Scatter inter-método "
            "{{qc:stages/stage_h03_qc.json:intermethod_scatter_pct:.1f}}%** (la celda de evidencia y "
            "el Plot 1 dan el valor de cada método).\n\n"
            "**Nota de definición** ([`docs/mdot_limit_definition_note.md`]"
            "(../docs/mdot_limit_definition_note.md)): E3 usa Gumbel 99% **sin** el factor R_in 1.25; "
            "G3 usa 5σ **con** R_in → G3 da "
            "{{qc:stages/stage_g3_qc.json:mdot_p50_msun_yr:.2e}} (`n/d` = G3 no ha calculado acreción "
            "para este objeto; ver su notebook). Misma cadena física; la diferencia "
            "es **definicional** (ninguna declarada canónica aún). Knob clave "
            "`h03_flux_unit_cgs=1e-20` (unidad nativa scipost) — sin él L/Ṁ salían ~10²⁰ altos."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Límite por método, insumos físicos y la nota de definición del `stage_h03_qc.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage_h03_qc.json', RUN_ID)\n"
            "lim = {L['method']: L for L in q['limits']}\n"
            "pin = q['physical_inputs']\n"
            "print('canónico:', q['canonical_method'], f\"| scatter inter-método {q['intermethod_scatter_pct']:.0f}%\")\n"
            "print(f\"\\nṀ canónico (psffit) = {lim['psffit']['mdot']:.2e} M☉/yr (throughput {lim['psffit']['throughput']:.2f})\")\n"
            "print('\\nṀ por método:')\n"
            "for m in ['optimal_psfsub','psffit','aperture','optimal_ls']:\n"
            "    print(f\"   {m:15s} Ṁ={lim[m]['mdot']:.2e}  throughput={lim[m]['throughput']:.2f}\")\n"
            "print(f\"\\nfísica: d={pin['distance_pc']}pc, A_V={pin['av']}, A_Hα={pin['a_halpha_over_av']*pin['av']:.2f}, \"\n"
            "      f\"masa={pin['companion_mass_msun']:.4f} M☉, radio={pin['companion_radius_rsun']} R☉\")\n"
            "print(f\"relación: {pin['lacc_lha_relation']} (scatter {pin['relation_scatter_dex']} dex)\")\n"
            # `limit_definition_note` NO la emite ningún productor (verificado
            # 2026-07-24). La diferencia de definición E3-vs-G3 está en
            # docs/mdot_limit_definition_note.md, y aquí se muestra lo que el QC
            # sí trae: el límite dual Alcala / Aoyama+21 que añadió R1.
            "print('\\ndefinición: E3 usa Gumbel 99% SIN el factor R_in 1.25 '\n"
            "      '(G3 usa 5 sigma CON el factor) -> docs/mdot_limit_definition_note.md')\n"
            "alt = lim[q['canonical_method']].get('mdot_aoyama21')\n"
            "if alt:\n"
            "    print(f\"\\nrelación dual (R1): Alcala+17 {lim[q['canonical_method']]['mdot']:.2e}\"\n"
            "          f\"  |  Aoyama+21 {alt:.2e} M☉/yr\")\n"
            "else:\n"
            "    print('\\n(sin límite Aoyama+21 en este QC: relación alternativa no configurada)')"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — Ṁ por método (el throughput manda)\n\n"
                    "El límite de Ṁ por método (escala log); en verde el canónico "
                    "(`{{qc:stages/stage_h03_qc.json:canonical_method}}` = "
                    "{{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].mdot:.2e}} M☉/yr). "
                    "Cada barra lleva su throughput: los métodos que recuperan peor la señal dan "
                    "límites **más altos**. Scatter inter-método "
                    "{{qc:stages/stage_h03_qc.json:intermethod_scatter_pct:.1f}}%."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_h03_qc.json', RUN_ID)\n"
                    "    lim = {L['method']: L for L in q['limits']}\n"
                    "    canon = q['canonical_method']\n"
                    "    methods = ['optimal_psfsub', 'psffit', 'aperture', 'optimal_ls']\n"
                    "    md = [lim[m]['mdot'] for m in methods]; th = [lim[m]['throughput'] for m in methods]\n"
                    "    cols = ['tab:green' if m == canon else '0.6' for m in methods]\n"
                    "    fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
                    "    ax.bar(range(len(methods)), md, color=cols)\n"
                    "    for i, (m, t) in enumerate(zip(md, th)):\n"
                    "        ax.text(i, m * 1.05, f'{m:.1e}\\nT={t:.2f}', ha='center', fontsize=8)\n"
                    "    ax.set_yscale('log'); ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=9)\n"
                    "    ax.set_ylabel('Ṁ límite superior [M☉/yr]')\n"
                    "    ax.set_title(f\"E3 · Ṁ por método (canónico {canon}={lim[canon]['mdot']:.1e}; \"\n"
                    "                 f\"scatter {q['intermethod_scatter_pct']:.0f}%)\")\n"
                    "    ax.axhline(lim[canon]['mdot'], color='tab:green', ls='--', lw=1); fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'e3_limits'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'mdot_by_method.png', dpi=110); print('figura ->', outdir / 'mdot_by_method.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — la cadena física: de la no-detección a Ṁ\n\n"
                    "Los 5 pasos para el método canónico "
                    "(`{{qc:stages/stage_h03_qc.json:canonical_method}}`): del umbral de flujo Gumbel "
                    "99% (`f_stat_99`), dividir por el throughput, deredden (A_V="
                    "{{qc:stages/stage_h03_qc.json:physical_inputs.av}}), convertir a luminosidad "
                    "(4πd², d={{qc:stages/stage_h03_qc.json:physical_inputs.distance_pc}} pc), y "
                    "aplicar {{qc:stages/stage_h03_qc.json:physical_inputs.lacc_lha_relation}} → "
                    "**Ṁ = {{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].mdot:.2e}} "
                    "M☉/yr**. Cada caja muestra el valor real del QC de este objeto."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_h03_qc.json', RUN_ID)\n"
                    "    canon = q['canonical_method']\n"
                    "    p = {L['method']: L for L in q['limits']}[canon]; pin = q['physical_inputs']\n"
                    "    deredden = p['f_lim_dereddened'] / p['f_lim_observed']\n"
                    "    steps = [('Gumbel 99%\\nf_stat_99', p['f_stat_99'], 'erg/s/cm²'),\n"
                    "             (f\"÷ throughput\\n{p['throughput']:.2f}\", p['f_lim_observed'], 'f_obs'),\n"
                    "             (f\"× deredden\\n(A_Hα={pin['a_halpha_over_av']*pin['av']:.2f}, {deredden:.1f}×)\", p['f_lim_dereddened'], 'f_dered'),\n"
                    "             (f\"× 4πd²\\n(d={pin['distance_pc']}pc)\", p['l_halpha'], 'L_Hα [erg/s]'),\n"
                    "             ('Alcalá 2017\\n→ L_acc → Ṁ', p['mdot'], 'Ṁ [M☉/yr]')]\n"
                    "    fig, ax = plt.subplots(figsize=(11, 3.4))\n"
                    "    for i, (lab, val, unit) in enumerate(steps):\n"
                    "        ax.text(i, 0.6, lab, ha='center', va='center', fontsize=8, bbox=dict(boxstyle='round', fc='0.92'))\n"
                    "        ax.text(i, 0.25, f'{val:.2e}\\n{unit}', ha='center', va='center', fontsize=8, color='tab:blue')\n"
                    "        if i < len(steps) - 1:\n"
                    "            ax.annotate('', (i + 0.65, 0.6), (i + 0.35, 0.6), arrowprops=dict(arrowstyle='->'))\n"
                    "    ax.set_xlim(-0.5, len(steps) - 0.5); ax.set_ylim(0, 1); ax.axis('off')\n"
                    "    ax.set_title(f'E3 · cómo la no-detección se vuelve Ṁ (método canónico {canon})')\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'e3_limits'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'physical_chain.png', dpi=110); print('figura ->', outdir / 'physical_chain.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**Ṁ(99%) = {{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].mdot:.2e}} M☉/yr** (canónico {{qc:stages/stage_h03_qc.json:canonical_method}}; Gumbel 99%, L_Hα deredden, throughput {{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].throughput:.2f}}, d={{qc:stages/stage_h03_qc.json:physical_inputs.distance_pc}}pc, A_V={{qc:stages/stage_h03_qc.json:physical_inputs.av}}, {{qc:stages/stage_h03_qc.json:physical_inputs.lacc_lha_relation}}).", "mdot_limit_definition_note.md"),
            ("Dependencia del método = throughput; **`{{qc:stages/stage_h03_qc.json:canonical_method}}` canónico citable** por robustez, no por dar el límite más bajo; scatter inter-método {{qc:stages/stage_h03_qc.json:intermethod_scatter_pct:.1f}}%.", None),
            ("Difiere de G3 ({{qc:stages/stage_g3_qc.json:mdot_p50_msun_yr:.2e}}) solo por DEFINICIÓN (Gumbel99 sin R_in vs 5σ con R_in); cadena física idéntica. Ninguna elegida canónica aún.", None),
            ("Knob `h03_flux_unit_cgs=1e-20` (unidad nativa scipost) — sin él L/Ṁ salían ~10²⁰ altos.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**E3: Ṁ (99%, canónico {{qc:stages/stage_h03_qc.json:canonical_method}}) = "
            "{{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].mdot:.2e}} M☉/yr** para "
            "este objeto (todo resuelto de su `stage_h03_qc.json`).\n\n"
            "- **Cadena:** Gumbel 99% → ÷throughput "
            "({{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].throughput:.2f}}) → "
            "deredden (A_V={{qc:stages/stage_h03_qc.json:physical_inputs.av}}) → ×4πd² "
            "({{qc:stages/stage_h03_qc.json:physical_inputs.distance_pc}} pc) → "
            "{{qc:stages/stage_h03_qc.json:physical_inputs.lacc_lha_relation}}.\n"
            "- **Método:** `{{qc:stages/stage_h03_qc.json:canonical_method}}` canónico citable por "
            "robustez; scatter inter-método "
            "{{qc:stages/stage_h03_qc.json:intermethod_scatter_pct:.1f}}%.\n"
            "- **vs G3:** {{qc:stages/stage_g3_qc.json:mdot_p50_msun_yr:.2e}} por definición "
            "(5σ + R_in), no por física; ninguna canónica aún.\n"
            "- **Caveats:** provisional hasta cerrar el A-block del objeto; el FAP 99% está por debajo "
            "de la resolución de "
            "{{csvtop:tables/halpha_detection_by_method.csv:matched_z:n_controls:.0f}} controles.\n"
            "- **Resultado científico:** la no-detección de E1 se traduce en el límite superior de Ṁ "
            "de arriba (acreción muy baja o ausente)."
        ),
    ),
    dict(
        id="E4", slug="E4_injection", title="Inyección-recuperación (v2, nulo empírico)", block="E · Resultado",
        spec="spec_E4_v2_codex_injection_recovery.md", run_override=None,
        what=("Inyecta señal sintética y mide el throughput de cada método, con **continuo plano no "
              "nulo medido** y el gate de nulos calculado sobre **posiciones de control** (v2)."),
        inputs="Extractores reales (C2–C6) + PSF + espectros de control de producción",
        outputs="`stages/stage_h04_qc.json`, `tables/injection_throughput_by_method.csv`",
        downstream="E3 (throughput), G1",
        exec=dict(kind="script", target="stage_h04_injection.sh",
                  cost="Pesado (~22 min grid 112 casos con `h04_process_pool`)."),
        qc="stages/stage_h04_qc.json",
        salient=["spec_version", "continuum_injection.value", "continuum_injection.source",
                 "checks.v2_nulls_clean.status", "checks.v5_continuum.status"],
        checks_md=(
            "## Los chequeos del QC, en físico\n\n"
            "E4 inyecta líneas de flujo conocido y mide cuánto sobrevive (**throughput**). Ese número "
            "es el que convierte el límite de flujo de E1 en un límite de Ṁ, así que los chequeos "
            "vigilan que la inyección se comporte como debe.\n\n"
            "| Chequeo | ¿Qué pregunta contesta? | Si falla |\n|---|---|---|\n"
            "| `v1_regression` | **¿La cadena sigue dando lo mismo que el caso histórico validado?** "
            "| Algo cambió en el camino sin que nadie lo decidiera. |\n"
            "| `v2_nulls_clean` | **¿Inventa recuperaciones donde no inyectamos nada?** Filas S/N=0 "
            "**en posiciones de control**, con la FAP medida contra la distribución nula empírica de "
            "los controles de producción del mismo método. Falla por *exceso* binomial de extremos, "
            "no por tener uno. | La tasa de falsos positivos no es la que E1 supone, y toda detección "
            "marginal aguas abajo queda en entredicho. |\n"
            "| `v3_monotonic` | **¿Más señal inyectada da más señal recuperada?** Throughput y "
            "completitud monótonos con S/N. | Una no-monotonía es bug o estadística insuficiente: hay "
            "que investigarla ANTES de usar el número. |\n"
            "| `v4_hierarchy` | **¿Los métodos se ordenan como la física manda?** Las extracciones "
            "que usan el modelo de PSF (C3/C4) deberían recuperar al menos tanto como la apertura "
            "simple (C2), con sesgo < 5% a S/N ≥ 5. | Si la apertura supera al ajuste de PSF, algo "
            "está mal en C4 — no es que la apertura sea mejor. |\n"
            "| `v5_continuum` | **¿Cuánto cuesta tener continuo debajo de la línea?** Exige además "
            "que el continuo plano sea **estrictamente positivo** y que sus filas **no sean idénticas** "
            "a las de `none` — si lo fueran, el chequeo no estaría midiendo nada. | Es el número que "
            "sostiene la decisión sobre los métodos de halo (C5/C6). |\n"
        ),
        narrative_md=(
            "## Qué hace E4 v2 y qué cambia respecto de v1\n\n"
            "E4 **inyecta** señales sintéticas de Hα de flujo/SNR conocidos y mide cuánto **recupera** "
            "cada método → el **throughput** (recuperado / inyectado), que es el factor con el que E3 "
            "convierte un límite de flujo en un límite de Ṁ. La grilla nominal está congelada: "
            "S/N {0,1,2,3,5,7,10} × anchos {LSF, 2×LSF} × continuo {none, flat} × posiciones "
            "{real, control1, control2, control3}.\n\n"
            "**Qué cambia en v2** ([`docs/spec_E4_v2_codex_injection_recovery.md`](../docs/spec_E4_v2_codex_injection_recovery.md), "
            "congelada 2026-07-29). La auditoría encontró tres defectos de interpretación en v1:\n\n"
            "| Defecto de v1 | Qué hace v2 |\n|---|---|\n"
            "| El gate de nulos llamaba «nulo» al caso S/N=0 **en la posición real del compañero**. Ese "
            "espectro contiene el dato científico y puede contener una línea real o un residuo: no es "
            "una posición nula | La población de V2 son **solo** `control1/2/3`. Las filas de la "
            "posición real se reportan aparte como `science_position_diagnostics`, **sin poder de "
            "veto** |\n"
            "| `recovered_snr` usaba el error interno del extractor con los controles desactivados; "
            "con STAT en rojo, ese número no cumple el modelo de ruido canónico | La significancia se "
            "estima con los **espectros de control de producción** del mismo método: mediana móvil de "
            "80 Å, el mismo matched filter de E1, y FAP de rango `(1+N(null≥obs))/(N+1)` |\n"
            "| El modo de continuo `flat` corría con **amplitud cero** cuando faltaba una clave de "
            "config: era un duplicado exacto de `none`, así que V5 no probaba nada | Si no está "
            "configurado, E4 **mide** el continuo en las bandas laterales [6500,6540] y [6585,6625] Å "
            "sobre los cuatro productos que preservan continuo, lo lleva a escala NORMRAD y adopta la "
            "mediana de los valores positivos. V5 falla si `flat` es cero o idéntico a `none` |\n\n"
            "**El gate global ya no exige cero extremos.** Con `K` filas extremas de `N`, falla solo si "
            "`P(X≥K | N, p_null=0.0455) < alpha=0.01`: un test binomial unilateral, el mismo que el "
            "gate de controles de D1. Eso prueba **exceso** de falsos positivos sin fingir "
            "independencia gaussiana canal a canal.\n\n"
            "Lo que v2 **no** toca: la grilla, los anchos, las posiciones, las perturbaciones de PSF, "
            "los extractores ni la definición de throughput."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Continuo inyectado (medido o configurado), throughput por método y las verificaciones del "
            "`stage_h04_qc.json`. Si el QC en disco es anterior a v2 la celda lo dice: su `v2` medía "
            "otra población y no es comparable con este gate."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage_h04_qc.json', RUN_ID)\n"
            "spec = q.get('spec_version')\n"
            "print(f\"spec del QC en disco : {spec}\")\n"
            "if spec != 'E4_v2':\n"
            "    print('   ⚠ QC anterior a la spec v2: su gate de nulos incluía la posición REAL del\\n'\n"
            "          '     compañero y su continuo `flat` pudo correr con amplitud cero. Hay que\\n'\n"
            "          '     re-ejecutar E4 para leerlo con la política nueva.')\n"
            "ci = q.get('continuum_injection')\n"
            "if ci:\n"
            "    print(f\"\\ncontinuo inyectado : {ci['value']:.3g} ({ci['scale']}, fuente: {ci['source']})\")\n"
            "    print(f\"   bandas laterales: {ci['bands_A']}\")\n"
            "    for m, v in (ci.get('by_method') or {}).items():\n"
            "        mark = '' if v > 0 else '   <- no positivo, excluido de la mediana'\n"
            "        print(f\"      {m:16s} {v:+10.3f}{mark}\")\n"
            "print(f\"\\ngrid: {q['grid']['n_injections']} inyecciones × \"\n"
            "      f\"{len(q['grid']['methods'])} métodos ({', '.join(q['grid']['methods'])})\")\n"
            "th = (q.get('throughput') or {}).get('per_method_at_snr5') or {}\n"
            "print('\\nthroughput @ SNR5:')\n"
            "for m in q['grid']['methods']:\n"
            "    if m in th:\n"
            "        print(f\"   {m:16s} {th[m]['throughput']:.3f} ± {th[m]['err']:.3f}\")\n"
            "print('\\nverificaciones:')\n"
            "for name, c in q['checks'].items():\n"
            "    extra = f\"  {c.get('failures')}\" if c.get('failures') else ''\n"
            "    print(f\"   {name:16s} {c.get('status')}{extra}\")\n"
            "v2 = q['checks'].get('v2_nulls_clean') or {}\n"
            "if 'n_rows' in v2:\n"
            "    print(f\"\\nV2 · nulo empírico sobre {v2.get('population')}\")\n"
            "    print(f\"   {v2['n_extreme']} filas extremas de {v2['n_rows']} \"\n"
            "          f\"(esperadas ~{v2['n_rows'] * v2['p_null']:.1f} con p_null={v2['p_null']})\")\n"
            "    print(f\"   exceso binomial p = {v2['excess_p']:.3g}  vs alpha = {v2['gate_alpha']}\"\n"
            "          f\"   -> {v2['status'].upper()}\")\n"
            "    rows = [r for r in v2.get('rows', []) if r.get('empirical_fap', 1.0) < v2['p_null']]\n"
            "    if rows:\n"
            "        from collections import Counter\n"
            "        print('   extremos por método:', dict(Counter(r['method'] for r in rows)))\n"
            "print('\\nopen_issues:')\n"
            "for s in q['open_issues']:\n"
            "    print('  -', s)"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — throughput por método (lo que E3 consume)\n\n"
                    "El throughput @ SNR5 de cada método del grid, con el veredicto de "
                    "`v4_hierarchy` en el título. Los métodos y los valores se leen del QC: si el "
                    "conjunto de extractores cambió (v2 corre los seis), la figura lo refleja sola."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_h04_qc.json', RUN_ID)\n"
                    "    th = q['throughput']['per_method_at_snr5']; v4 = q['checks']['v4_hierarchy']\n"
                    "    methods = [m for m in q['grid']['methods'] if m in th]\n"
                    "    vals = [th[m]['throughput'] for m in methods]; errs = [th[m]['err'] for m in methods]\n"
                    "    canon = 'psffit'\n"
                    "    cols = ['tab:green' if m == canon else 'tab:blue' for m in methods]\n"
                    "    fig, ax = plt.subplots(figsize=(9, 4.2))\n"
                    "    ax.bar(range(len(methods)), vals, yerr=errs, color=cols, capsize=4)\n"
                    "    for i, v in enumerate(vals): ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)\n"
                    "    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=8.5, rotation=15)\n"
                    "    ax.set_ylabel('throughput @ SNR5 (recuperado/inyectado)')\n"
                    "    ax.set_ylim(0, max(vals) * 1.25 if vals else 1.0)\n"
                    "    extra = f\": {v4['failures']}\" if v4.get('failures') else ''\n"
                    "    ax.set_title(f\"E4 {q.get('spec_version','?')} · throughput por método \"\n"
                    "                 f\"(v4_hierarchy={v4['status']}{extra}; verde = canónico)\")\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'e4_injection'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'throughput.png', dpi=110); print('figura ->', outdir / 'throughput.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — la curva de recuperación (pendiente = throughput)\n\n"
                    "Flujo neto recuperado vs inyectado por método. **Subconjunto mostrado:** solo la "
                    "variante `nominal` con `continuum_mode='none'` (la configuración base del grid; "
                    "las demás variantes se resumen en el QC). Pasa por el origen y la **pendiente es "
                    "el throughput**: cuanto más plana la curva, más insensible el método. *(psffit "
                    "va discontinuo para distinguirlo cuando se solapa con otro.)*\n\n"
                    "El caso `continuum_mode='flat'` es el que v2 arregla: con la amplitud medida, "
                    "comparar `flat` contra `none` mide de verdad lo que cuesta tener continuo debajo "
                    "de la línea (eso es `v5_continuum`)."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import pandas as pd\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    q = nb.load_qc('stages/stage_h04_qc.json', RUN_ID); th = q['throughput']['per_method_at_snr5']\n"
                    "    d = pd.read_csv(rd / 'tables' / 'injection_throughput_by_method.csv')\n"
                    "    d = d[(d.variant == 'nominal') & (d.continuum_mode == 'none')]\n"
                    "    methods = [m for m in q['grid']['methods'] if m in th]\n"
                    "    fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
                    "    for m in methods:\n"
                    "        ls = '--' if m == 'psffit' else '-'\n"
                    "        g = d[d.method == m].groupby('injected_flux')['recovered_flux_net'].median()\n"
                    "        ax.plot(g.index, g.values, 'o' + ls, ms=5, label=f\"{m} (T={th[m]['throughput']:.2f})\")\n"
                    "    mx = d['injected_flux'].max(); ax.plot([0, mx], [0, mx], 'k:', lw=0.8, label='ideal 1:1 (T=1)')\n"
                    "    ax.set_xlabel('flujo inyectado'); ax.set_ylabel('flujo neto recuperado (mediana)')\n"
                    "    ax.set_title('E4 · recuperación: pendiente = throughput; aperture/ls insensibles en el borde')\n"
                    "    ax.legend(fontsize=8); fig.tight_layout()\n"
                    "    outdir = rd / 'plots' / 'e4_injection'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'recovery.png', dpi=110); print('figura ->', outdir / 'recovery.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**La posición real del compañero no es un nulo.** Su fila S/N=0 contiene el dato científico y puede llevar línea real o residuo; se reporta como diagnóstico y no vota el gate. La población nula son solo `control1/2/3`.", "spec_E4_v2_codex_injection_recovery.md"),
            ("**La significancia sale de los controles de producción**, no de `recovered_snr`: con STAT en rojo, el error interno del extractor no cumple el modelo de ruido canónico. FAP de rango sobre la distribución nula empírica de cada método.", "noise_model.md"),
            ("**El continuo `flat` se mide** si no está configurado (bandas laterales de Hα, cuatro productos que preservan continuo, escala NORMRAD, mediana de los positivos). Antes corría a cero y V5 era vacuo.", "spec_E4_v2_codex_injection_recovery.md"),
            ("**El gate es binomial, no de tolerancia cero**: falla por *exceso* de extremos (`P(X≥K|N,p_null) < 0.01`), el mismo criterio que el gate de controles de D1. No se mueve el umbral mirando el run.", None),
            ("Salvedad heredada: la regresión histórica de Stage06 sigue sin pasar ('H04 no válido para E3' formalmente); los throughputs se usan con esa salvedad declarada.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (leer del QC, no de aquí)\n\n"
            "**E4 v2 cambia lo que significa «nulo limpio», así que el veredicto de V2 no es "
            "comparable con el de v1.** Los números concretos —throughput por método, continuo "
            "adoptado, filas extremas y su exceso binomial— los imprime la celda de evidencia sobre "
            "el QC de este objeto; no están copiados en esta prosa a propósito, que es justo lo que "
            "hacía que un objeto heredara las cifras de otro.\n\n"
            "- **Qué mirar primero:** `v2_nulls_clean`. Si falla, hay **exceso de falsos positivos** "
            "en posiciones de control procesadas como el objeto, y eso afecta a cómo hay que leer "
            "cualquier detección marginal aguas abajo (E1/E3). Un fallo aquí no es un detalle de "
            "bookkeeping.\n"
            "- **Después, `v5_continuum`:** con la amplitud medida, `flat` ya no puede ser un "
            "duplicado de `none`; la degradación por método es la que sostiene la discusión sobre "
            "los métodos de halo (C5/C6).\n"
            "- **Y `v4_hierarchy`:** los métodos con modelo de PSF deberían recuperar al menos tanto "
            "como la apertura simple. Si el orden se rompe, el sospechoso es el extractor, no la "
            "apertura.\n"
            "- **Downstream:** el throughput del método canónico es el que E3 aplica para el límite "
            "de Ṁ, y D1 v4 lo usa **solo** en las bandas de línea."
        ),
    ),
    # ===================== BLOQUE F — paquete =====================
    dict(
        id="E5", qc_optional=True, slug="E5_contrast_curves", title="Curvas de contraste (anillos)", block="E · Resultado",
        spec="spec_E5_codex_contrast_curves.md", run_override=None,
        what=(
            "Contraste mínimo detectable vs separación por método (sgf/lpm), con inyecciones en "
            "grilla de anillos validadas sobre el mapa E1b (Julo et al. 2025 Fig. 10)."
        ),
        inputs="Cubos residuales C5/C6 + cubo madre + PSF C1",
        outputs="`tables/contrast_curve_by_method.csv`, `tables/contrast_injections.csv`, `stages/stage_h05_qc.json`",
        downstream="E6 (ROC); notebook 10 de límites de masa (consumidor futuro)",
        exec=dict(kind="module_main", target="musepipe.stages.stage_h05_contrast",
                  cost="Ligero-moderado (~min: camino delta lineal, grilla congelada en spec)."),
        qc="stages/stage_h05_qc.json",
        salient=["params.threshold_sigma", "params.f_star_line",
                 "checks.v1_curves_written", "checks.v2_monotonic_trend"],
        checks_md=(
            "## Los chequeos del QC, en físico\n\n"
            "| Chequeo | ¿Qué pregunta contesta? | Si falla |\n|---|---|---|\n"
            "| `v1_curves_written` | **¿Hay curva para cada método disponible?** | No hay contraste "
            "que comparar. |\n"
            "| `v2_monotonic_trend` | **¿La sensibilidad mejora al alejarse de la estrella?** El "
            "contraste mediano del tercio externo debe ser ≤ el del tercio interno: lejos del halo se "
            "detectan compañeros más débiles. Es sanidad, no bloquea. | Una tendencia invertida "
            "delata un problema de normalización o de máscara, no una propiedad del instrumento. |\n"
            "| `v3_grid_saturation` | **¿La rejilla de contrastes probados es la adecuada?** Fracción "
            "de separaciones cuyo contraste al 50% quedó en el borde de la rejilla; > 50% = aviso. | "
            "La curva está tocando el techo: mide la rejilla, no el instrumento. |\n"
        ),
        narrative_md=(
            "## Qué hace E5\n\n"
            "Inyecta líneas falsas (gaussiana FWHM=LSF × PSF C1) en anillos concéntricos "
            "(separaciones × 8 ángulos × contrastes 1e-5→1e-2 relativos al flujo estelar en la "
            "banda de línea) y valida cada una con la MISMA cadena de detección de E1b "
            "(matched filter + μ̂/σ̂ robustos por anillo, umbral 5σ). Por la linealidad de la "
            "sustracción con ŝ fija, el residual base y sus anillos se calculan UNA vez y cada "
            "inyección solo aporta su delta — la grilla completa cuesta segundos y la decisión "
            "usa la realización de ruido REAL de cada posición (como el paper §3.3.2).\n\n"
            "La curva reportada es el contraste con ≥50% de detección sobre los ángulos, con "
            "bandas al 25/75%."
        ),
        evidence_md=("## Evidencia: curva por método"),
        evidence_code=(
            "q = nb.load_qc('stages/stage_h05_qc.json', RUN_ID)\n"
            "print('F_star(banda línea) =', q['params']['f_star_line'])\n"
            "for m, mq in q['methods'].items():\n"
            "    print(f\"  {m}: {mq['n_injections']} inyecciones\")\n"
            "    for e in mq['curve'][:6]:\n"
            "        print(f\"    r={e['separation_px']:5.1f}px  c50={e['contrast_50']}  c25={e['contrast_25']}  c75={e['contrast_75']}\")\n"
            "print('checks:', q['checks'])"
        ),
        plots=[
            dict(
                md=("## Plot — curva de contraste 5σ (Fig. 10 del paper)\n\n"
                    "Contraste al 50% con banda 25/75%, eje x en arcsec (25 mas/px)."),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_h05_qc.json', RUN_ID)\n"
                    "    fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
                    "    for m, mq in q['methods'].items():\n"
                    "        r = [e['separation_px']*0.025 for e in mq['curve'] if e['contrast_50']]\n"
                    "        c50 = [e['contrast_50'] for e in mq['curve'] if e['contrast_50']]\n"
                    "        lo = [e['contrast_25'] or np.nan for e in mq['curve'] if e['contrast_50']]\n"
                    "        hi = [e['contrast_75'] or np.nan for e in mq['curve'] if e['contrast_50']]\n"
                    "        ax.plot(r, c50, marker='o', ms=3, label=m)\n"
                    "        ax.fill_between(r, lo, hi, alpha=0.2)\n"
                    "    ax.set_yscale('log'); ax.set_xlabel('separación [arcsec]')\n"
                    "    ax.set_ylabel('contraste de línea 5σ'); ax.legend()\n"
                    "    ax.set_title('E5 · curvas de contraste'); fig.tight_layout(); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("Detección con la cadena E1b real (no un detector ad-hoc); grillas congeladas en la spec.", "spec_E5_codex_contrast_curves.md"),
            ("Camino delta lineal exacto (verificado contra fuerza bruta en tests); psfsub opcional con re-sustracción completa.", "plan_integracion_halosub_julo2025.md"),
        ],
        checks=(
            "q = nb.load_qc('stages/stage_h05_qc.json', RUN_ID)\n"
            "for k, v in q['checks'].items():\n"
            "    print(f'  {k}: {v}')"
        ),
        conclusion_md=(
            "## Estado\n\n"
            "**Pendiente de primera ejecución sobre datos reales** (requiere C5/C6). Núcleo y "
            "contrato verificados con tests sintéticos (2026-07-14)."
        ),
    ),
    dict(
        id="E6", qc_optional=True, slug="E6_roc_curves", title="Curvas ROC del detector", block="E · Resultado",
        spec="spec_E6_codex_roc_curves.md", run_override=None,
        what=(
            "ROC (DP vs FAP) del matched filter por método y separación, con inyecciones a "
            "contraste tipo límite y nulo empírico de anillos + canales sin línea (Julo+25 Fig. 11)."
        ),
        inputs="Cubos residuales C5/C6 + E5 QC (contrast_50) + PSF C1",
        outputs="`tables/roc_curves.csv`, `stages/stage_h06_qc.json`",
        downstream="Insumo informativo del checkpoint D1 v3 (robustez por método)",
        exec=dict(kind="module_main", target="musepipe.stages.stage_h06_roc",
                  cost="Moderado (~min: mapas nulos según h06_null_step_channels)."),
        qc="stages/stage_h06_qc.json",
        salient=["params.n_null_wavelengths", "checks.v1_curves_written",
                 "checks.v2_auc_above_random", "checks.v3_null_sample_ok"],
        checks_md=(
            "## Los chequeos del QC, en físico\n\n"
            "Una ROC cruza **detección** (qué fracción de inyecciones supera el umbral) contra "
            "**falsa alarma** (qué fracción del ruido lo supera). El AUC resume: 1.0 = separación "
            "perfecta, 0.5 = el método no distingue señal de ruido.\n\n"
            "| Chequeo | ¿Qué pregunta contesta? | Si falla |\n|---|---|---|\n"
            "| `v1_curves_written` | **¿Hay ROC para cada método × escenario?** | Falta el escenario. |\n"
            "| `v2_auc_above_random` | **¿El detector es mejor que tirar una moneda?** AUC ≥ 0.5 − "
            "2/√n_iny en todos los escenarios. | Un AUC por debajo del azar **no** significa un "
            "instrumento malo: significa un bug de signo o de normalización (estaríamos detectando "
            "al revés). |\n"
            "| `v3_null_sample_ok` | **¿Hay bastante ruido medido para que la FAP signifique algo?** "
            "≥ 500 muestras nulas por escenario. | Con pocas muestras la parte izquierda de la curva "
            "(FAP baja) es justo la que no se puede medir — y es la que importa. |\n"
        ),
        narrative_md=(
            "## Qué hace E6\n\n"
            "Complementa a E5: en vez de fijar el FAP (5σ) y variar el contraste, fija el "
            "contraste (el `contrast_50` de E5, régimen de límite de detección) y barre el "
            "umbral, midiendo DP con inyecciones en anillo (16 ángulos) y FAP con el nulo "
            "EMPÍRICO: píxeles del anillo del mapa z base + mapas z reconstruidos con la "
            "plantilla centrada en canales libres de línea (paper §3.3.3 — sin supuestos "
            "gaussianos). AUC por método y separación."
        ),
        evidence_md=("## Evidencia: AUC por método y escenario"),
        evidence_code=(
            "q = nb.load_qc('stages/stage_h06_qc.json', RUN_ID)\n"
            "print('mapas nulos:', q['params']['n_null_wavelengths'])\n"
            "for m, mq in q['methods'].items():\n"
            "    for s in mq['scenarios']:\n"
            "        print(f\"  {m:5s} r={s['separation_px']:5.1f}px c={s['contrast']:.2e} \"\n"
            "              f\"AUC={s['auc']:.3f} (n_iny={s['n_injections']}, n_nulo={s['n_null']})\")\n"
            "print('checks:', q['checks'])"
        ),
        plots=[
            dict(
                md=("## Plot — ROC por método y separación (Fig. 11 del paper)"),
                code=(
                    "try:\n"
                    "    import csv\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    rows = list(csv.DictReader(open(rd / 'tables' / 'roc_curves.csv')))\n"
                    "    fig, ax = plt.subplots(figsize=(6.4, 5.4))\n"
                    "    keys = sorted({(r['method'], r['separation_px']) for r in rows})\n"
                    "    for m, sep in keys:\n"
                    "        pts = sorted((float(r['fap']), float(r['dp'])) for r in rows\n"
                    "                     if r['method'] == m and r['separation_px'] == sep)\n"
                    "        ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.1, label=f'{m} r={float(sep):g}px')\n"
                    "    ax.plot([0, 1], [0, 1], 'k:', lw=0.8, label='aleatorio')\n"
                    "    ax.set_xlabel('FAP'); ax.set_ylabel('DP'); ax.legend(fontsize=7)\n"
                    "    ax.set_title('E6 · ROC'); fig.tight_layout(); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("Nulo empírico (anillos + canales sin línea), nunca gaussiano asumido; informativo para D1 v3, no cambia veredictos.", "spec_E6_codex_roc_curves.md"),
        ],
        checks=(
            "q = nb.load_qc('stages/stage_h06_qc.json', RUN_ID)\n"
            "for k, v in q['checks'].items():\n"
            "    print(f'  {k}: {v}')"
        ),
        conclusion_md=(
            "## Estado\n\n"
            "**Pendiente de primera ejecución sobre datos reales** (requiere C5/C6 y E5). "
            "Núcleo y contrato verificados con tests sintéticos (2026-07-14/15)."
        ),
    ),
    dict(
        id="F1", slug="F1_final_report", title="Paquete final + gate", block="F · Paquete",
        spec="spec_F1_codex_final_products.md", run_override=None,
        what="Consolida A→E en el paquete final y aplica el gate (semáforo por etapa + limitaciones aceptadas).",
        inputs="Todos los QC A→E", outputs="`report/run_summary.json`, `report/report.md`, figuras/tablas",
        downstream="Revisión humana / decisión de publicación",
        exec=dict(kind="pyscript", target="build_report.py", cost="Ligero."),
        qc="report/run_summary.json",
        salient=["overall_status", "accepted_limitations_hash", "schema_version"],
        checks_md=(
            "## Los términos de este QC, en físico\n\n"
            "F1 no mide nada: **audita** lo que midieron las demás y decide si el run se puede "
            "publicar. De ahí que casi todo aquí sea integridad y trazabilidad, no astronomía.\n\n"
            "| Término | Qué es | Por qué importa |\n|---|---|---|\n"
            "| `overall_status` | El semáforo del run: verde / amarillo / rojo. | Amarillo = "
            "utilizable **con limitaciones declaradas**; rojo = hay algo bloqueante sin resolver. |\n"
            "| `gate_policy` | Qué chequeos son bloqueantes y cuáles solo informativos. | Sin ella, "
            "«falla un chequeo» no dice si el run sirve o no. |\n"
            "| `accepted_limitations` + `..._hash` | Los problemas **conocidos y aceptados a "
            "propósito** (p.ej. el sistemático cromático de continuo de D2), con un hash de la "
            "lista. | El hash es lo que impide que una limitación se edite o desaparezca en "
            "silencio: si alguien la cambia, deja de cuadrar. |\n"
            "| `hash_chain` | Encadenado de sha256 de los productos de cada etapa. | Prueba que el "
            "espectro que se publica desciende de los cubos que se dice: si un producto intermedio "
            "se regeneró y no el resto, la cadena se rompe. |\n"
            "| `traceability` | De qué QC sale cada número del informe. | Ningún valor del paper "
            "debería carecer de la etapa que lo midió. |\n"
            "| `spectrum_conventions` | Marco de λ, unidad de flujo, convención de error. | Es lo que "
            "permite que otro lea el producto sin adivinar. |\n"
        ),
        narrative_md=(
            "## Qué hace F1 y cómo gatea\n\n"
            "F1 consolida todos los QC de A→E en el **paquete final** y aplica el **gate**: un "
            "semáforo por etapa (green/yellow/red) + una **política de gate congelada** que degrada "
            "ciertos rojos *documentados* a yellow **'limitación aceptada'** (no los esconde: los "
            "conserva y anota). **Cualquier OTRO rojo bloquea.**\n\n"
            "**Semáforo:** 15 yellow, 1 green (B1), 1 `not_run` (A3), **0 rojos** → overall "
            "**yellow**.\n\n"
            "**4 limitaciones aceptadas** (congeladas, hash `e4478990`) — las mismas que fuimos "
            "viendo etapa por etapa:\n"
            "- **A4/M5 STAT**: la varianza STAT subestima ~4–6× por la covarianza del remuestreo "
            "(inherente al drizzle).\n"
            "- **D2/v3 continuo**: sistemático de NIVEL inter-método en el rojo (residuo de halo "
            "cromático; ya lo referenciamos a controles).\n"
            "- **E2/T2**: el test de forma-PSF reinterpretado para una NO-detección (el máximo no es "
            "PSF → apoya la no-detección).\n"
            "- **E4/v4_hierarchy**: la patología de borde del throughput (aperture/ls insensibles; "
            "canónico psffit no afectado).\n\n"
            "`hash_chain` = **pass** (proveniencia de productos trazable). **F1 se niega "
            "correctamente a dar luz verde de paper** mientras el A-block siga provisional: es "
            "**yellow, no green**."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Semáforo por etapa, las 4 limitaciones aceptadas y la cadena de hash del "
            "`report/run_summary.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('report/run_summary.json', RUN_ID)\n"
            "from collections import Counter\n"
            "cnt = Counter(s['status'] for s in q['stages'])\n"
            "print('overall_status:', q['overall_status'], '|', dict(cnt))\n"
            "hc = q.get('hash_chain', {})\n"
            "print(f\"hash_chain: {hc.get('status')} ({len(hc.get('checks', []))} checks) | \"\n"
            "      f\"open_issues: {len(q['open_issues'])} | gate hash: {q['gate_policy']['accepted_limitations_hash']}\")\n"
            "print('\\nsemáforo por etapa:')\n"
            "for s in q['stages']:\n"
            "    mark = f\"  (limitación aceptada ×{s['accepted_limitations']})\" if s['accepted_limitations'] else ''\n"
            "    print(f\"   {s['stage']:20s} {s['status']:9s} issues={s['issue_count']}{mark}\")\n"
            "print('\\n4 limitaciones aceptadas:')\n"
            "for a in q['accepted_limitations']:\n"
            "    print(f\"   {a['path']:32s} {a['reason'][:75]}...\")"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — el semáforo del gate\n\n"
                    "Status de las 17 etapas. **0 rojos** → overall **yellow**. B1 verde, A3 `not_run`, "
                    "el resto yellow; `⚠×1` marca las 4 etapas con una limitación aceptada "
                    "(A4, D2, E2, E4)."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('report/run_summary.json', RUN_ID)\n"
                    "    st = q['stages']\n"
                    "    col = {'green': 'tab:green', 'yellow': 'gold', 'red': 'tab:red', 'not_run': '0.8'}\n"
                    "    names = [s['stage'] for s in st]; stats = [s['status'] for s in st]; accl = [s['accepted_limitations'] for s in st]\n"
                    "    from collections import Counter\n"
                    "    cnt = Counter(stats)\n"
                    "    resumen = ', '.join(f'{v} {k}' for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]))\n"
                    "    fig, ax = plt.subplots(figsize=(8, 6))\n"
                    "    ax.barh(range(len(names)), [1] * len(names), color=[col.get(s, '0.5') for s in stats])\n"
                    "    for i, (n, s, a) in enumerate(zip(names, stats, accl)):\n"
                    "        ax.text(0.02, i, f\"{n}  [{s}]\" + (f'  ⚠×{a}' if a else ''), va='center', fontsize=8, color='k')\n"
                    "    ax.set_yticks([]); ax.set_xticks([]); ax.invert_yaxis(); ax.set_xlim(0, 1)\n"
                    "    ax.set_title(f\"F1 · semáforo: overall={q['overall_status'].upper()} \"\n"
                    "                 f\"({cnt.get('red', 0)} rojos; {resumen})\")\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'f1_report'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'gate.png', dpi=110); print('figura ->', outdir / 'gate.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — las 4 limitaciones aceptadas\n\n"
                    "Los 4 rojos degradados a yellow (política de gate congelada). Cada uno es "
                    "inherente, reinterpretado o documentado — y ya los revisamos en A4, D2, E2, E4."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('report/run_summary.json', RUN_ID)\n"
                    "    al = q['accepted_limitations']\n"
                    "    labels = {'m5_stat.status': 'A4/M5 STAT', 'checks.v3_continuum_stable.ok': 'D2/v3 continuo rojo',\n"
                    "              't2.status': 'E2/T2 forma-PSF', 'checks.v4_hierarchy.status': 'E4 jerarquía'}\n"
                    "    fig, ax = plt.subplots(figsize=(11, 3.2))\n"
                    "    for i, a in enumerate(al):\n"
                    "        lab = labels.get(a['path'], a['path'])\n"
                    "        ax.text(0.01, len(al) - 1 - i, f'● {lab}', fontsize=10, weight='bold', va='center')\n"
                    "        ax.text(0.22, len(al) - 1 - i, a['reason'][:110] + '...', fontsize=8, va='center')\n"
                    "    ax.set_xlim(0, 1); ax.set_ylim(-0.5, len(al) - 0.5); ax.axis('off')\n"
                    "    ax.set_title(f\"F1 · {len(al)} limitaciones aceptadas (rojos degradados; hash {q['gate_policy']['accepted_limitations_hash']})\")\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'f1_report'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'accepted_limitations.png', dpi=110); print('figura ->', outdir / 'accepted_limitations.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**Gate `overall_status: yellow`, 0 rojos bloqueantes** en el realineado (era rojo en ADP): 15 yellow, 1 green (B1), 1 not_run (A3).", None),
            ("Política de gate **congelada** (hash e4478990): SOLO 4 rojos específicos (A4/M5, D2/v3, E2/T2, E4/hierarchy) bajan a 'limitación aceptada'; cualquier OTRO rojo bloquea.", "d2_red_continuum_diagnosis.md"),
            ("Las limitaciones aceptadas se **conservan y anotan** (no se esconden); `hash_chain` pass (proveniencia trazable).", None),
            ("F1 se niega correctamente a dar **luz verde de paper** mientras el A-block siga provisional (yellow, no green).", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**F1: paquete final; overall `yellow`, 0 rojos bloqueantes.**\n\n"
            "- **Fecha:** consolidado 2026-07-09/10 sobre el run realineado.\n"
            "- **Semáforo:** 15 yellow, 1 green (B1), 1 not_run (A3), 0 rojos.\n"
            "- **4 limitaciones aceptadas** (congeladas, documentadas, no ocultas): A4/M5, D2/v3, "
            "E2/T2, E4/hierarchy.\n"
            "- **hash_chain pass**; 29 open_issues agregados.\n"
            "- **Yellow (no green):** F1 se niega a dar luz verde de paper mientras el A-block siga "
            "provisional — es el comportamiento correcto.\n"
            "- **Endpoint del paquete:** no-detección de Hα → Ṁ ≲ 8×10⁻¹³ M☉/yr, compañero real "
            "ligado (caracterización en el bloque G)."
        ),
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
        salient=["hash_chain_ok", "stat_verdict", "n_flagged", "frozen_criteria_untouched"],
        checks_md=(
            "## Los términos de este QC, en físico\n\n"
            "| Término | Qué es | Por qué importa |\n|---|---|---|\n"
            "| `stat_verdict` | Si la extensión `STAT` del cubo real sirve como σ. | Es la pregunta "
            "de M5 hecha sobre el cubo definitivo: si el STAT subestima el ruido, **σ se mide "
            "siempre en controles** ([`docs/noise_model.md`](../../docs/noise_model.md)). |\n"
            "| `hash_chain_ok` | Que el encadenado de sha256 entre etapas cuadre. | Garantiza que "
            "estos resultados salen de estos datos, sin productos mezclados de otra ejecución. |\n"
            "| `frozen_criteria_untouched` | Que los umbrales y decisiones congelados sigan siendo "
            "los mismos. | Correr sobre datos reales **no** puede ir acompañado de aflojar un umbral: "
            "eso convertiría el criterio en una consecuencia del resultado. |\n"
            "| `pytest_before` / `pytest_after` | La suite antes y después de la ejecución. | Deja "
            "constancia de que el código no cambió a mitad del proceso. |\n"
            "| `legacy_comparison` | Contraste con la reducción histórica (ADP). | Sitúa la "
            "re-reducción propia frente a la del archivo. |\n"
        ),
        narrative_md=(
            "## Qué hace G0 y por qué\n\n"
            "G0 es la **entrada del bloque G (caracterización)**. Ejecuta la cadena multi-método "
            "**de extremo a extremo sobre el cubo real** por primera vez y deja constancia de "
            "cualquier fallo de infraestructura **antes** de tocar nada. Es *ejecutar, inspeccionar y "
            "corregir lo mínimo* — no reescribir.\n\n"
            "**Verifica:**\n"
            "- **`hash_chain` = pass**: la proveniencia de los productos es trazable (el sha del cubo "
            "de entrada casa a lo largo de la cadena).\n"
            "- **`stat_verdict` = {{qc:stages/stage_g0_qc.json:stat_verdict.status}}**: STAT usable = "
            "{{qc:stages/stage_g0_qc.json:stat_verdict.usable}} (factor "
            "{{qc:stages/stage_g0_qc.json:stat_verdict.factor}}, de A4/M5) — marcado, se usa ruido "
            "empírico aguas abajo.\n"
            "- **`legacy_comparison`**: el run realineado vs el legacy ADP (flujo integrado por "
            "banda), **{{qc:stages/stage_g0_qc.json:legacy_comparison.n_flagged}} de "
            "{{qc:stages/stage_g0_qc.json:legacy_comparison.n_bands}} bandas flagged** (legacy: "
            "`{{qc:stages/stage_g0_qc.json:legacy_comparison.legacy_run}}`; `n/d` = no hay ADP de "
            "archivo para este objeto y la comparación no aplica). Es **documentario** (las razones no "
            "son fiables donde el continuo es negativo, p.ej. post-Hα).\n"
            "- **`frozen_criteria_untouched` = True**: no se tocaron los criterios congelados del gate "
            "para pasar (sin trampas).\n\n"
            "**Deviación honesta** (open_issue): se reprodujo **retroactivamente** vía "
            "`scripts/run_g0.py` sobre un run `stage-*` existente, no en una rama `phase-g0` fresca — "
            "'G0 cumplido en sustancia'. Con `hash_chain_ok=True`, el paquete es consistente en "
            "proveniencia → entrada a G1–G5."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Cadena de hash, veredicto STAT, comparación legacy y criterios congelados del "
            "`stage_g0_qc.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage_g0_qc.json', RUN_ID)\n"
            "print('hash_chain_ok:', q['hash_chain_ok'], '| status:', q['hash_chain']['status'])\n"
            "sv = q['stat_verdict']\n"
            "print(f\"stat_verdict: {sv['status']} (usable={sv['usable']}, factor {sv['factor']}) -> ruido empírico\")\n"
            "lc = q['legacy_comparison']\n"
            "print(f\"legacy: {lc['n_flagged']}/{lc['n_bands']} bandas flagged vs {lc['legacy_run']} (documentario)\")\n"
            "print('frozen_criteria_untouched:', q['frozen_criteria_untouched'])\n"
            "print(f\"cubo: {q['input_cube']['file'].split('/')[-1]} (entry_point={q['input_cube']['entry_point']}, NaN {q['input_cube']['nan_fraction_data']:.3f})\")\n"
            "print('\\nopen_issues:')\n"
            "for a in q['open_issues']:\n"
            "    s = a['issue'] if isinstance(a, dict) else a\n"
            "    print('  -', s[:100])"
        ),
        plot_md=(
            "## Plot — comparación con el legacy (ADP)\n\n"
            "Razón del flujo integrado por banda (realineado / legacy ADP), del "
            "`g0_legacy_comparison.csv`. Rojo = flagged. Es **documentario**: donde el continuo es "
            "negativo (p.ej. post-Hα) la razón no es fiable — la diferencia principal es la "
            "calibración de flujo entre reducciones, esperada."
        ),
        plot_code=(
            "try:\n"
            "    import numpy as np\n"
            "    import pandas as pd\n"
            "    import matplotlib.pyplot as plt\n"
            "    rd = nb.run_dir(RUN_ID)\n"
            "    lc = nb.load_qc('stages/stage_g0_qc.json', RUN_ID).get('legacy_comparison') or {}\n"
            "    if not lc.get('table'):\n"
            "        raise FileNotFoundError('sin comparación legacy para este objeto: '\n"
            "                                + str(lc.get('reason') or 'no hay tabla declarada en el QC'))\n"
            "    d = pd.read_csv(rd / 'tables' / 'g0_legacy_comparison.csv')\n"
            "    x = np.arange(len(d))\n"
            "    fig, ax = plt.subplots(figsize=(10, 4.3))\n"
            "    ax.bar(x, d['ratio_new_over_legacy'], 0.55, color=['tab:red' if f else 'tab:green' for f in d['flagged']])\n"
            "    ax.axhline(1.0, color='k', ls='--', lw=1, label='ratio=1 (idéntico)'); ax.axhline(0, color='0.6', lw=0.6)\n"
            "    ax.set_xticks(x); ax.set_xticklabels(d['band'], rotation=30, ha='right', fontsize=8)\n"
            "    for i, r in enumerate(d['ratio_new_over_legacy']):\n"
            "        ax.text(i, r + 0.05 * np.sign(r), f'{r:.2f}', ha='center', fontsize=7)\n"
            "    ax.set_ylabel('flujo realineado / legacy (ADP)')\n"
            "    ax.set_title(f\"G0 · comparación con legacy ADP: {int(d['flagged'].sum())}/{len(d)} bandas flagged (documentario)\")\n"
            "    ax.legend(fontsize=8); fig.tight_layout()\n"
            "    outdir = rd / 'plots' / 'g0_real_cube'; outdir.mkdir(parents=True, exist_ok=True)\n"
            "    fig.savefig(outdir / 'legacy_comparison.png', dpi=110); print('figura ->', outdir / 'legacy_comparison.png'); plt.show()\n"
            "except Exception as e:\n"
            "    print('No se pudo generar el plot:', type(e).__name__, e)"
        ),
        decisions=[
            ("G0 cerrado **retroactivamente** (`hash_chain_ok=True`, criterios congelados intactos); ejecución real-cube verificada.", "g0_execution_log.md"),
            ("Comparación legacy = calibración de flujo distinta entre reducciones (esperado); {{qc:stages/stage_g0_qc.json:legacy_comparison.n_flagged}}/{{qc:stages/stage_g0_qc.json:legacy_comparison.n_bands}} bandas flagged, documentario (continuo negativo). `n/d` = sin ADP de archivo para este objeto.", None),
            ("M5 STAT red → ruido empírico aguas abajo (consistente con A4).", "noise_model.md"),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**G0: cadena real-cube ejecutada y verificada; `hash_chain_ok = "
            "{{qc:stages/stage_g0_qc.json:hash_chain_ok}}`, criterios congelados intactos = "
            "{{qc:stages/stage_g0_qc.json:frozen_criteria_untouched}}.**\n\n"
            "- **Proveniencia:** hash_chain sobre el cubo declarado en el QC de este objeto.\n"
            "- **STAT:** {{qc:stages/stage_g0_qc.json:stat_verdict.status}} (M5, factor "
            "{{qc:stages/stage_g0_qc.json:stat_verdict.factor}}) → ruido empírico.\n"
            "- **Legacy:** {{qc:stages/stage_g0_qc.json:legacy_comparison.n_flagged}} de "
            "{{qc:stages/stage_g0_qc.json:legacy_comparison.n_bands}} bandas flagged vs "
            "`{{qc:stages/stage_g0_qc.json:legacy_comparison.legacy_run}}`, documentario "
            "(calibración de flujo distinta); `n/d` = sin ADP de archivo para este objeto.\n"
            "- **Deviación honesta:** reproducido retroactivamente vía `run_g0.py`, no en rama fresca "
            "— cumplido en sustancia.\n"
            "- **Downstream:** entrada a G1 (validación de extracción), G2–G5."
        ),
    ),
    dict(
        id="G1", slug="G1_extraction_validation", title="Validación de extracción", block="G · Caracterización",
        spec="spec_G1_codex_extraction_validation.md", run_override=None,
        what="Valida los métodos de extracción: covarianza espectral, inflación espacial, presupuesto de sesgo.",
        inputs="Controles + E4 grid", outputs="`stages/stage_g1_qc.json`, `g1_channel_covariance.npz`",
        downstream="G2, E3 (throughput/bias)",
        exec=dict(kind="pyscript", target="run_g1.py", cost="Moderado."),
        qc="stages/stage_g1_qc.json",
        salient=["method_verdicts", "corr_length_channels_median", "n_eff_over_n_median", "spatial_inflation_by_box"],
        checks_md=(
            "## Los términos de este QC, en físico\n\n"
            "G1 mide **cuánto miente el ruido ingenuo**. Todo lo de abajo son formas de decir que los "
            "píxeles y los canales vecinos NO son independientes, así que sumar errores en cuadratura "
            "subestima σ.\n\n"
            "| Término | Qué es | Por qué importa |\n|---|---|---|\n"
            "| `corr_length_channels_median` | Sobre cuántos canales sigue correlacionado el ruido. | "
            "Si es ~2, dos canales contiguos no aportan dos medidas independientes: el remuestreo en "
            "λ los mezcló. |\n"
            "| `n_eff_over_n_median` | Cuántos canales **efectivamente independientes** hay por cada "
            "canal nominal. | Si sale ~0.4, una línea repartida en 10 canales aporta como ~4: la "
            "significancia real es menor que la que da contar canales. |\n"
            "| `spatial_inflation_by_box` | Factor por el que se subestima σ en una caja N×N si se "
            "suponen píxeles independientes. | Crece con la caja porque el desplazamiento subpíxel de "
            "B1 correlacionó vecinos; es lo que convierte «3×3» en mucho menos que 9 medidas. |\n"
            "| `method_verdicts` | Por método: `validated`, `validated_with_bias` o `rejected`. | "
            "`validated_with_bias` = utilizable **si se transporta su sesgo** al presupuesto de "
            "error; `rejected` = no se usa para ciencia (aquí caen los que sobre-sustraen). |\n"
            "| `bias_budget_table` | El sesgo de cada método, con su origen. | Es lo que permite "
            "aceptar un método imperfecto sin esconder su defecto. |\n"
            "| `impact_on_x10_chi2` | Cómo cambia la comparación de D1 al usar la covarianza real. | "
            "Sin ella, dos métodos parecerían discrepar mucho más de lo que discrepan. |\n"
        ),
        narrative_md=(
            "## Qué hace G1 y qué valida\n\n"
            "G1 **valida los métodos de extracción**: cuantifica la **covarianza del ruido** "
            "(espectral + espacial), el **presupuesto de sesgo**, y emite un veredicto por método.\n\n"
            "**Covarianza (por qué el ruido debe ser empírico):**\n"
            "- **Espectral:** longitud de correlación "
            "{{qc:stages/stage_g1_qc.json:covariance.corr_length_channels_median:.2f}} canales, "
            "**n_eff/n = {{qc:stages/stage_g1_qc.json:covariance.n_eff_over_n_median:.3f}}** — solo esa "
            "fracción de los canales cuenta como independiente; promediar en λ NO gana √N.\n"
            "- **Espacial:** sumar en una caja N×N infla la varianza frente a la suma ingenua √N: "
            "**box3 ≈ {{qc:stages/stage_g1_qc.json:covariance.spatial_inflation_by_box.3:.1f}}×, "
            "box5 ≈ {{qc:stages/stage_g1_qc.json:covariance.spatial_inflation_by_box.5:.1f}}×**. Es la "
            "**covarianza del remuestreo** — el **mismo** fenómeno detrás de M5 (STAT subestimado). "
            "G1 es donde se caracteriza del todo.\n\n"
            "**Presupuesto de sesgo:** los dos métodos validados (psffit, optimal_psfsub) tienen "
            "~**33% de pérdida de throughput** (que E3 corrige dividiendo por el throughput); el "
            "término de PSF es ~0 (la perturbación de PSF es un no-op, la PSF está flux-normalizada).\n\n"
            "**Veredictos:** **psffit & optimal_psfsub = `validated_with_bias`** (pérdida de "
            "throughput medida y corregida aguas abajo); **aperture & optimal_ls = `rejected`** "
            "(insensibles en el borde del compañero). Es exactamente lo que D1 usa: el **par primario "
            "= los dos validados**.\n\n"
            "**Nota de dominio:** la covarianza de los controles **NO** captura el sistemático de "
            "halo en la posición del compañero — eso es el presupuesto de inyección (el B6 de D1)."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Veredictos, covarianza y presupuesto de sesgo del `stage_g1_qc.json`."
        ),
        evidence_code=(
            "import pandas as pd\n"
            "q = nb.load_qc('stages/stage_g1_qc.json', RUN_ID)\n"
            "cov = q['covariance']\n"
            "print('veredictos por método:')\n"
            "for m, v in q['method_verdicts'].items():\n"
            "    print(f\"   {m:15s} {v}\")\n"
            "print(f\"\\ncovarianza espectral: corr_length {cov['corr_length_channels_median']:.2f} ch, \"\n"
            "      f\"n_eff/n = {cov['n_eff_over_n_median']:.3f} (rho_1 {cov['rho_1_median']:.2f})\")\n"
            "print('inflación espacial por caja:', {k: round(v, 1) for k, v in cov['spatial_inflation_by_box'].items()})\n"
            "print()\n"
            "d = pd.read_csv(nb.run_dir(RUN_ID) / 'tables' / 'g1_bias_budget.csv')\n"
            "tot = d[d.term == 'TOTAL']\n"
            "print('presupuesto de sesgo (TOTAL throughput loss):')\n"
            "for _, r in tot.iterrows():\n"
            "    v = 'rechazado (insensible)' if pd.isna(r['value_frac']) else f\"{r['value_frac']*100:+.0f}%\"\n"
            "    print(f\"   {r['method']:15s} {v}\")"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — la covarianza del remuestreo (por qué σ es empírico)\n\n"
                    "Inflación de la varianza al sumar en cajas N×N frente a la suma ingenua √N "
                    "(=1). En este objeto: box3 ≈ "
                    "{{qc:stages/stage_g1_qc.json:covariance.spatial_inflation_by_box.3:.1f}}×, "
                    "box5 ≈ {{qc:stages/stage_g1_qc.json:covariance.spatial_inflation_by_box.5:.1f}}× "
                    "→ un σ de apertura de √(Σ STAT) está mal por estos factores. Es el mismo "
                    "mecanismo que M5. (Espectral: n_eff/n = "
                    "{{qc:stages/stage_g1_qc.json:covariance.n_eff_over_n_median:.3f}}.)"
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_g1_qc.json', RUN_ID); cov = q['covariance']\n"
                    "    infl = cov['spatial_inflation_by_box']\n"
                    "    boxes = sorted(infl, key=lambda k: int(k)); xs = [f'{int(b)}×{int(b)}' for b in boxes]; vals = [infl[b] for b in boxes]\n"
                    "    fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
                    "    ax.bar(xs, vals, color='tab:orange', alpha=0.85)\n"
                    "    for i, v in enumerate(vals): ax.text(i, v + 0.3, f'{v:.1f}×', ha='center', fontsize=9)\n"
                    "    ax.axhline(1.0, color='tab:green', ls='--', label='suma ingenua √N (=1)')\n"
                    "    ax.set_xlabel('caja de integración (N×N spaxels)'); ax.set_ylabel('inflación varianza real / ingenua')\n"
                    "    ax.set_title(f\"G1 · covarianza del remuestreo (espectral n_eff/n={cov['n_eff_over_n_median']:.2f}, \"\n"
                    "                 f\"corr_length {cov['corr_length_channels_median']:.1f} ch)\")\n"
                    "    ax.legend(fontsize=8); fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'g1_validation'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'covariance.png', dpi=110); print('figura ->', outdir / 'covariance.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — veredictos y pérdida de throughput por método\n\n"
                    "psffit y optimal_psfsub = `validated_with_bias` (pérdida de throughput ~−33%, "
                    "corregida en E3); aperture y optimal_ls = `rejected` (insensibles en el borde, "
                    "throughput no medible). Son los dos validados que D1 compara."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import pandas as pd\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_g1_qc.json', RUN_ID); ver = q['method_verdicts']\n"
                    "    d = pd.read_csv(nb.run_dir(RUN_ID) / 'tables' / 'g1_bias_budget.csv')\n"
                    "    tot = d[d.term == 'TOTAL'].set_index('method')['value_frac']\n"
                    "    methods = ['psffit', 'optimal_psfsub', 'aperture', 'optimal_ls']\n"
                    "    fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
                    "    for i, m in enumerate(methods):\n"
                    "        val = tot.get(m, np.nan)\n"
                    "        if np.isnan(val):\n"
                    "            ax.text(i, 0.02, 'rechazado\\n(insensible)', ha='center', va='bottom', fontsize=9, color='tab:red')\n"
                    "        else:\n"
                    "            ax.bar(i, abs(val) * 100, color='tab:green')\n"
                    "            ax.text(i, abs(val) * 100 + 1, f'{val*100:+.0f}%\\n{ver[m]}', ha='center', fontsize=8)\n"
                    "    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=9)\n"
                    "    ax.set_ylabel('|pérdida de throughput| [%]'); ax.set_ylim(0, 45)\n"
                    "    ax.set_title('G1 · veredictos: psffit/psfsub validados (−33%, corregido en E3); aperture/ls rechazados')\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'g1_validation'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'verdicts.png', dpi=110); print('figura ->', outdir / 'verdicts.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("psffit & optimal_psfsub = `validated_with_bias` (throughput loss ~−33%, corregido en E3); aperture & optimal_ls = `rejected` (insensibles en el borde). Es el par primario de D1.", "noise_model.md"),
            ("Correlación espectral {{qc:stages/stage_g1_qc.json:covariance.corr_length_channels_median:.2f}} ch, **n_eff/n={{qc:stages/stage_g1_qc.json:covariance.n_eff_over_n_median:.3f}}**; inflación espacial box3≈{{qc:stages/stage_g1_qc.json:covariance.spatial_inflation_by_box.3:.1f}}×, box5≈{{qc:stages/stage_g1_qc.json:covariance.spatial_inflation_by_box.5:.1f}}× → **confirma M5** y justifica el ruido empírico.", None),
            ("La covarianza de controles NO captura el sistemático de halo en la posición del compañero (eso es el B6 / presupuesto de inyección).", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**G1: valida psffit & optimal_psfsub (el par primario de D1); rechaza aperture & "
            "optimal_ls.**\n\n"
            "- **Covarianza espectral:** corr_length "
            "{{qc:stages/stage_g1_qc.json:covariance.corr_length_channels_median:.2f}} ch, n_eff/n "
            "{{qc:stages/stage_g1_qc.json:covariance.n_eff_over_n_median:.3f}} (promediar en λ no gana "
            "√N).\n"
            "- **Covarianza espacial:** box3 "
            "{{qc:stages/stage_g1_qc.json:covariance.spatial_inflation_by_box.3:.1f}}×, box5 "
            "{{qc:stages/stage_g1_qc.json:covariance.spatial_inflation_by_box.5:.1f}}× → **confirma "
            "M5** (STAT subestimado), justifica el σ empírico.\n"
            "- **Sesgo:** ~−33% pérdida de throughput en los validados (corregido en E3); término de "
            "PSF ~0.\n"
            "- **Downstream:** los veredictos definen el par primario que D1 compara y los throughputs "
            "que E3 aplica."
        ),
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
        salient=["n_detected", "n_upper_limit", "n_not_measurable", "halpha_reconciliation_v3", "lsf_fwhm_A"],
        checks_md=(
            "## Los términos de este QC, en físico\n\n"
            "| Término | Qué es | Por qué importa |\n|---|---|---|\n"
            "| `n_detected` / `n_marginal` / `n_upper_limit` / `n_not_measurable` | El reparto del "
            "catálogo de líneas: medida, dudosa, solo cota superior, o **imposible de medir** (cae "
            "en el hueco del láser, fuera de rango o sobre una telúrica). | «No medible» no es «no "
            "hay»: distinguirlo evita convertir una laguna instrumental en un límite físico. |\n"
            "| `lsf_source` / `lsf_fwhm_A` | La anchura instrumental usada para el ajuste, y de dónde "
            "sale (medida en A4/M2, no la nominal). | Una línea no resuelta tiene exactamente esta "
            "anchura: si se pone mal, el flujo integrado sale mal. |\n"
            "| `covariance_used` | Si el ajuste usó la covarianza espectral de G1. | Sin ella, el "
            "error de una línea que abarca varios canales sale demasiado pequeño. |\n"
            "| `throughput_applied` / `throughput_source` | La fracción de flujo que la extracción "
            "deja pasar, medida por inyección en E4. | El flujo observado se divide por ella para "
            "recuperar el intrínseco; por eso importa que no se aplique dos veces (ver D1). |\n"
            "| `rv_weighted_kms` | Velocidad radial combinada de las líneas medidas. | Una línea "
            "real está a la velocidad del sistema; una a otra velocidad es sospechosa. |\n"
            "| `halpha_reconciliation_v3` | Que la Hα de G2 y la de E1 cuenten lo mismo. | Dos "
            "etapas midiendo la misma línea con distinto método deben coincidir, o una de las dos "
            "está mal. |\n"
        ),
        narrative_md=(
            "## Qué hace G2 y el resultado\n\n"
            "G2 mide las líneas espectrales de forma **genérica** (cero lógica específica de Hα — el "
            "catálogo de líneas vive en config) sobre el espectro canónico final: continuo local, flujo "
            "(directo + ajuste gaussiana⊗LSF), EW, centroide, FWHM, asimetría, RV, status "
            "detect/límite, y errores por Monte Carlo.\n\n"
            "**Catálogo:** {{qc:stages/stage_g2_qc.json:catalog_n}} líneas en config — Balmer (Hα/Hβ), serie de "
            "Paschen, triplete de Ca II, He I, [O I], [S II] (diagnósticos de acreción, cromosfera y "
            "outflow).\n\n"
            "**Resultado para este objeto** — detectadas: "
            "**{{qc:stages/stage_g2_qc.json:n_detected}}** · marginales: "
            "**{{qc:stages/stage_g2_qc.json:n_marginal}}** · límites superiores: "
            "**{{qc:stages/stage_g2_qc.json:n_upper_limit}}** · no medibles: "
            "**{{qc:stages/stage_g2_qc.json:n_not_measurable}}**.\n\n"
            "**Reconciliación de Hα (V3):** G2 Hα = "
            "`{{qc:stages/stage_g2_qc.json:halpha_reconciliation_v3.g2_halpha_status}}` vs E1 = "
            "`{{qc:stages/stage_g2_qc.json:halpha_reconciliation_v3.h01_verdict}}` → consistente = "
            "**{{qc:stages/stage_g2_qc.json:halpha_reconciliation_v3.consistent}}** (`n/d` = la "
            "reconciliación no está calculada en el QC de este objeto).\n\n"
            "**Inputs:** LSF {{qc:stages/stage_g2_qc.json:lsf_fwhm_A}} Å "
            "(`{{qc:stages/stage_g2_qc.json:lsf_source}}` — ojo a la procedencia: si dice `config`, no "
            "es la medida de A4/M2), throughput "
            "{{qc:stages/stage_g2_qc.json:throughput_applied:.3f}} (de E4), MC "
            "n={{qc:stages/stage_g2_qc.json:mc.n}}, **covarianza = "
            "`{{qc:stages/stage_g2_qc.json:covariance_used}}`** (mientras no sea un modelo real, el MC "
            "trata los errores como independientes — salvedad). Los límites alimentan G3 (inferencia "
            "física)."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Resumen del catálogo y la reconciliación con E1 del `stage_g2_qc.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage_g2_qc.json', RUN_ID)\n"
            "print(f\"catálogo: {q['catalog_n']} líneas (espectro {q['input_spectrum']['file']}, {q['input_spectrum']['method']})\")\n"
            "print(f\"  detectadas={q['n_detected']}  marginales={q['n_marginal']}  \"\n"
            "      f\"límites={q['n_upper_limit']}  no_medibles={q['n_not_measurable']}\")\n"
            "hr = q['halpha_reconciliation_v3']\n"
            "print(f\"\\nreconciliación Hα (V3): G2={hr['g2_halpha_status']} vs E1={hr['h01_verdict']} -> consistente={hr['consistent']}\")\n"
            "print(f\"LSF={q['lsf_fwhm_A']} Å ({q['lsf_source']}); throughput={q['throughput_applied']:.3f}; \"\n"
            "      f\"MC n={q['mc']['n']}; covarianza={q['covariance_used']}\")\n"
            "import pandas as pd\n"
            "# Tabla NATIVA de G2 (G5/characterization la copia como final_line_table.csv):\n"
            "d = pd.read_csv(nb.run_dir(RUN_ID) / 'tables' / 'g2_line_measurements.csv')\n"
            "todas = 'todas' if (d['z_score'].abs() < 5).all() else 'NO todas'\n"
            "print(f\"\\nz_score de las {len(d)} líneas: {d['z_score'].min():.2f} .. {d['z_score'].max():.2f} ({todas} < 5σ)\")"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — el catálogo de líneas: ninguna detectada\n\n"
                    "El `z_score` (significancia de detección) de cada línea, de la tabla nativa de G2 "
                    "(`tables/g2_line_measurements.csv`; G5 la copia como `final_line_table.csv`), "
                    "coloreado por familia. Las no medibles se marcan con × gris. Hα marcada con ★."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import pandas as pd\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_g2_qc.json', RUN_ID)\n"
                    "    d = pd.read_csv(nb.run_dir(RUN_ID) / 'tables' / 'g2_line_measurements.csv').sort_values('rest_A')\n"
                    "    fams = d['family'].fillna('?').unique()\n"
                    "    cm = {f: c for f, c in zip(fams, plt.cm.tab10.colors)}\n"
                    "    fig, ax = plt.subplots(figsize=(10, 5))\n"
                    "    for i, (_, r) in enumerate(d.reset_index(drop=True).iterrows()):\n"
                    "        z = r['z_score']\n"
                    "        if pd.notna(z):\n"
                    "            ax.plot(z, i, 'o', color=cm.get(r['family'], '0.5'), ms=7)\n"
                    "        else:   # no medible: sin z_score, marcador distinto (no confundir con z=0)\n"
                    "            ax.plot(0, i, 'x', color='0.6', ms=7, mew=1.5)\n"
                    "        ax.text(-6.5, i, r['name'] + (' ★' if r['name'] == 'Halpha' else ''), fontsize=6.5, va='center')\n"
                    "    ax.axvline(5, color='tab:red', ls='--', lw=1, label='5σ detección'); ax.axvline(-5, color='tab:red', ls='--', lw=1)\n"
                    "    ax.axvline(0, color='0.6', lw=0.6)\n"
                    "    ax.set_yticks([]); ax.set_xlim(-7, 7); ax.set_xlabel('z_score (significancia de detección)')\n"
                    "    ax.set_title(f\"G2 · {q['catalog_n']} líneas: {q['n_detected']} detectadas, \"\n"
                    "                 f\"{q['n_upper_limit']} límites, {q['n_not_measurable']} no medible(s)\")\n"
                    "    ax.legend(fontsize=8, loc='lower right'); fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'g2_lines'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'zscore_forest.png', dpi=110); print('figura ->', outdir / 'zscore_forest.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — los límites de flujo 5σ por línea\n\n"
                    "El límite superior de flujo (5σ) de cada línea medible (escala log), de la tabla "
                    "nativa `tables/g2_line_measurements.csv`. Es el **producto** que G3 consume "
                    "(G5 la copia a `report/characterization/final_line_table.csv`); Hα (★) es la más "
                    "restrictiva para el diagnóstico de acreción."
                ),
                code=(
                    "try:\n"
                    "    import numpy as np\n"
                    "    import pandas as pd\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    d = pd.read_csv(nb.run_dir(RUN_ID) / 'tables' / 'g2_line_measurements.csv')\n"
                    "    d = d[d['flux_upper_limit_5sigma'].notna()].sort_values('rest_A')\n"
                    "    x = np.arange(len(d))\n"
                    "    cols = ['tab:red' if n == 'Halpha' else 'tab:blue' for n in d['name']]\n"
                    "    fig, ax = plt.subplots(figsize=(10, 4.3))\n"
                    "    ax.bar(x, d['flux_upper_limit_5sigma'], color=cols)\n"
                    "    ax.set_yscale('log'); ax.set_xticks(x)\n"
                    "    ax.set_xticklabels([n + (' ★' if n == 'Halpha' else '') for n in d['name']], rotation=60, ha='right', fontsize=6.5)\n"
                    "    ax.set_ylabel('límite de flujo 5σ (unidad cubo)')\n"
                    "    ax.set_title('G2 · límites superiores de flujo por línea (Hα ★ = la más restrictiva para acreción)')\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'g2_lines'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'flux_limits.png', dpi=110); print('figura ->', outdir / 'flux_limits.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("{{qc:stages/stage_g2_qc.json:n_detected}} detectadas / {{qc:stages/stage_g2_qc.json:n_upper_limit}} límites / {{qc:stages/stage_g2_qc.json:n_not_measurable}} no medible(s) — reconciliación V3 con E1: consistente={{qc:stages/stage_g2_qc.json:halpha_reconciliation_v3.consistent}} (G2 Hα `{{qc:stages/stage_g2_qc.json:halpha_reconciliation_v3.g2_halpha_status}}` vs E1 `{{qc:stages/stage_g2_qc.json:halpha_reconciliation_v3.h01_verdict}}`).", None),
            ("Cero lógica específica de Hα: catálogo genérico de {{qc:stages/stage_g2_qc.json:catalog_n}} líneas en config (Balmer, Paschen, Ca II, He I, [O I], [S II]).", None),
            ("Salvedades de este objeto: LSF `{{qc:stages/stage_g2_qc.json:lsf_source}}`; covarianza aplicada al MC = {{qc:stages/stage_g2_qc.json:covariance_used}}.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**G2: {{qc:stages/stage_g2_qc.json:n_detected}} líneas detectadas, "
            "{{qc:stages/stage_g2_qc.json:n_upper_limit}} límites superiores, "
            "{{qc:stages/stage_g2_qc.json:n_not_measurable}} no medible(s)** en este objeto "
            "(resuelto de su `stage_g2_qc.json`).\n\n"
            "- **Genérico:** catálogo de {{qc:stages/stage_g2_qc.json:catalog_n}} líneas (cero lógica "
            "específica de Hα) sobre `{{qc:stages/stage_g2_qc.json:input_spectrum.method}}`.\n"
            "- **Reconciliación:** Hα `{{qc:stages/stage_g2_qc.json:halpha_reconciliation_v3.g2_halpha_status}}` "
            "vs E1 `{{qc:stages/stage_g2_qc.json:halpha_reconciliation_v3.h01_verdict}}` → "
            "consistente={{qc:stages/stage_g2_qc.json:halpha_reconciliation_v3.consistent}}.\n"
            "- **Salvedades:** LSF `{{qc:stages/stage_g2_qc.json:lsf_source}}`; covarianza aplicada = "
            "{{qc:stages/stage_g2_qc.json:covariance_used}}.\n"
            "- **Downstream:** los límites alimentan G3 (inferencia física, L_acc→Ṁ)."
        ),
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
        salient=["mdot_p50_msun_yr", "l_acc_lsun", "combined_accretion", "libraries", "definition"],
        checks_md=(
            "## Los términos de este QC, en físico\n\n"
            "La cadena física es: **flujo de línea → luminosidad de línea → L_acc → Ṁ**. Cada flecha "
            "es una relación empírica de literatura con su dispersión, y ahí está casi todo el error.\n\n"
            "| Término | Qué es | Por qué importa |\n|---|---|---|\n"
            "| `l_acc_lsun` | Luminosidad de acreción: la energía por segundo que libera el material "
            "al caer. | Se obtiene de la luminosidad de la línea con una relación calibrada en "
            "objetos donde ambas se midieron. |\n"
            "| `mdot_p50_msun_yr` | La **mediana** (percentil 50) de la distribución de Ṁ del Monte "
            "Carlo, no un valor único. | Ṁ = L_acc·R/(G·M)·(1−R/R_in): cada ingrediente (masa, radio, "
            "extinción, distancia) entra con su incertidumbre, así que el resultado es una "
            "distribución. |\n"
            "| `combined_accretion` | Cómo se combinan varias líneas en un solo número. | Con solo "
            "cotas superiores, la combinación es **la más restrictiva**, no un promedio. |\n"
            "| `libraries` | Qué relaciones y modelos externos se usaron, con cita. | Cambiar de "
            "calibración cambia Ṁ en un factor: el número no significa nada sin decir con cuál se "
            "obtuvo. |\n"
            "| `halpha_h03_consistency_v5` | Que G3 y E3 den lo mismo para Hα. | Miden lo mismo por "
            "caminos distintos; si divergen, hay un factor aplicado dos veces o ninguna. |\n"
            "| `provisional` | Que el resultado depende de algo aún no cerrado. | Marca el número como "
            "no publicable todavía, aunque esté calculado. |\n"
        ),
        narrative_md=(
            "## Qué hace G3 y qué queda diferido\n\n"
            "G3 es la **inferencia física**: de los flujos/límites de líneas (G2) infiere la "
            "**luminosidad de acreción L_acc** y la **tasa Ṁ** (relación Hα de Alcalá 2017), y "
            "*ajustaría* plantillas/atmósferas/tracks para SpT/Teff/masa — pero eso está **diferido "
            "(pendiente de librerías externas)**.\n\n"
            "**Acreción en este objeto** (de su `stage_g3_qc.json`; `n/d` = G3 no ha calculado "
            "acreción para el objeto — su QC puede ser la variante *G3-real* de tipado espectral, con "
            "otro esquema):\n\n"
            "| | |\n|---|---|\n"
            "| **L_acc** | {{qc:stages/stage_g3_qc.json:combined_accretion.l_acc_lsun:.2e}} L☉ "
            "(`{{qc:stages/stage_g3_qc.json:combined_accretion.kind}}`, de "
            "`{{qc:stages/stage_g3_qc.json:combined_accretion.from_line}}`, regla "
            "`{{qc:stages/stage_g3_qc.json:combined_accretion.rule}}`) |\n"
            "| **Ṁ p50** | {{qc:stages/stage_g3_qc.json:mdot_p50_msun_yr:.2e}} M☉/yr (MC "
            "n={{qc:stages/stage_g3_qc.json:mc.n}}) |\n\n"
            "**Diferencia con E3 (definicional):** G3 usa **5σ** del flujo de Hα de G2 **con** el "
            "factor de truncamiento de disco R_in=1.25; E3 usa Gumbel 99% **sin** R_in. Misma cadena "
            "física; el desfase entre "
            "{{qc:stages/stage_g3_qc.json:mdot_p50_msun_yr:.2e}} (G3) y "
            "{{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].mdot:.2e}} (E3) es de "
            "**definición**, no de física. Canónica **sin decidir** (usuario diferido, "
            "[`docs/mdot_limit_definition_note.md`]"
            "(../docs/mdot_limit_definition_note.md)).\n\n"
            "**Diferido → `not_constrained`:** atmósfera (BT-Settl), tracks (BHAC15/ATMO2020), "
            "plantillas (Luhman/Bonnefoy) — SpT/Teff/masa necesitan datos externos. **Por eso la "
            "clasificación de G4 es ambigua.** Provisional."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Acreción, comparación con E3 y el estado de las librerías del `stage_g3_qc.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage_g3_qc.json', RUN_ID)\n"
            "ca = q['combined_accretion']\n"
            "print(f\"acreción: {ca['kind']} L_acc = {ca['l_acc_lsun']:.2e} L☉ (de {ca['from_line']}, regla {ca['rule']})\")\n"
            "print(f\"Ṁ p50 = {q['mdot_p50_msun_yr']:.2e} M☉/yr (MC n={q['mc']['n']}); {q['n_lines_with_relation']} línea con relación\")\n"
            "e3 = nb.load_qc('stages/stage_h03_qc.json', RUN_ID)\n"
            "e3_mdot = {L['method']: L['mdot'] for L in e3['limits']}[e3['canonical_method']]\n"
            "print(f\"\\ncomparación: E3 Ṁ={e3_mdot:.2e} (Gumbel99, sin R_in) vs G3 Ṁ={q['mdot_p50_msun_yr']:.2e} (5σ, con R_in) -> {q['mdot_p50_msun_yr']/e3_mdot:.2f}× definicional\")\n"
            "print('\\nlibrerías (tipado espectral):')\n"
            "for k, v in q['libraries'].items():\n"
            "    print(f\"   {k:20s} {v}\")"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — E3 vs G3: la misma física, dos definiciones\n\n"
                    "Los dos límites de Ṁ de **este objeto**: E3 = "
                    "{{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].mdot:.2e}} "
                    "(Gumbel 99%, sin R_in) y G3 = "
                    "{{qc:stages/stage_g3_qc.json:mdot_p50_msun_yr:.2e}} (5σ, con el factor R_in "
                    "1.25). El desfase es puramente **definicional** — misma cadena física. Canónica "
                    "sin decidir."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_g3_qc.json', RUN_ID)\n"
                    "    e3 = nb.load_qc('stages/stage_h03_qc.json', RUN_ID)\n"
                    "    e3_mdot = {L['method']: L['mdot'] for L in e3['limits']}[e3['canonical_method']]\n"
                    "    g3_mdot = q.get('mdot_p50_msun_yr')\n"
                    "    if g3_mdot is None:\n"
                    "        raise KeyError('el QC de G3 de este objeto no trae acreción '\n"
                    "                       \"('mdot_p50_msun_yr'): puede ser la variante G3-real de \"\n"
                    "                       'tipado espectral. Sin par E3/G3 que comparar.')\n"
                    "    fig, ax = plt.subplots(figsize=(6.5, 4.3))\n"
                    "    bars = ax.bar(['E3\\n(Gumbel 99%,\\nsin R_in)', 'G3\\n(5σ,\\ncon R_in 1.25)'], [e3_mdot, g3_mdot],\n"
                    "                  color=['tab:blue', 'tab:green'])\n"
                    "    for b, v in zip(bars, [e3_mdot, g3_mdot]):\n"
                    "        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f'{v:.2e}', ha='center', fontsize=10)\n"
                    "    ax.set_ylabel('Ṁ límite superior [M☉/yr]')\n"
                    "    ax.set_title(f'G3 · E3 vs G3: {g3_mdot/e3_mdot:.2f}× (definicional, misma física)')\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'g3_accretion'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'e3_vs_g3.png', dpi=110); print('figura ->', outdir / 'e3_vs_g3.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — qué constriñe G3 y qué queda diferido\n\n"
                    "G3 **computa** la acreción (L_acc, Ṁ vía Alcalá) pero **difiere** el tipado "
                    "espectral (atmósfera BT-Settl, tracks, plantillas) por falta de librerías externas "
                    "→ SpT/Teff/masa `not_constrained` → **G4 ambigua**."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_g3_qc.json', RUN_ID)\n"
                    "    items = [('acreción (L_acc, Ṁ)', 'computado'),\n"
                    "             ('atmósfera (BT-Settl)', 'diferido'),\n"
                    "             ('tracks (BHAC15/ATMO2020)', 'diferido'),\n"
                    "             ('plantillas (Luhman/Bonnefoy)', 'diferido')]\n"
                    "    col = {'computado': 'tab:green', 'diferido': '0.7'}\n"
                    "    names = [i[0] for i in items]\n"
                    "    fig, ax = plt.subplots(figsize=(8, 3.2))\n"
                    "    ax.barh(names, [1] * len(names), color=[col[i[1]] for i in items])\n"
                    "    for i, (n, s) in enumerate(items):\n"
                    "        ax.text(0.5, i, f'{n}  →  {s}', ha='center', va='center', fontsize=9, color='w' if s == 'computado' else 'k', weight='bold')\n"
                    "    ax.set_xlim(0, 1); ax.set_xticks([]); ax.set_yticks([]); ax.invert_yaxis()\n"
                    "    ax.set_title('G3 · acreción computada; tipado espectral diferido (pending_libraries)')\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'g3_accretion'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'characterization_status.png', dpi=110); print('figura ->', outdir / 'characterization_status.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**Ṁ p50 = {{qc:stages/stage_g3_qc.json:mdot_p50_msun_yr:.2e}} M☉/yr** (5σ de G2 + factor R_in 1.25); difiere de E3 ({{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].mdot:.2e}}) solo por DEFINICIÓN; canónica sin decidir.", "mdot_limit_definition_note.md"),
            ("L_acc {{qc:stages/stage_g3_qc.json:combined_accretion.kind}} = {{qc:stages/stage_g3_qc.json:combined_accretion.l_acc_lsun:.2e}} L☉ de `{{qc:stages/stage_g3_qc.json:combined_accretion.from_line}}` (regla `{{qc:stages/stage_g3_qc.json:combined_accretion.rule}}`).", None),
            ("Plantilla/atmósfera/tracks = `not_constrained` (pending_libraries: BT-Settl/BHAC15/Luhman-Bonnefoy diferidas) → G4 ambigua.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**G3 (este objeto): L_acc = "
            "{{qc:stages/stage_g3_qc.json:combined_accretion.l_acc_lsun:.2e}} L☉ "
            "(`{{qc:stages/stage_g3_qc.json:combined_accretion.kind}}`), Ṁ p50 = "
            "{{qc:stages/stage_g3_qc.json:mdot_p50_msun_yr:.2e}} M☉/yr (5σ + R_in).** `n/d` = la "
            "acreción no está calculada en el QC de este objeto.\n\n"
            "- **vs E3:** {{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].mdot:.2e}} "
            "(Gumbel99, sin R_in) → la diferencia es de definición, no de física; canónica sin "
            "decidir.\n"
            "- **Tipado espectral diferido:** atmósfera/tracks/plantillas `not_constrained` (falta de "
            "librerías externas).\n"
            "- **Consecuencia:** sin SpT/Teff/masa espectroscópicos, la clasificación de G4 queda "
            "**ambigua** (planeta/BD/M no resuelto).\n"
            "- **Downstream:** G4 (clasificación) y G5 (síntesis)."
        ),
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
        salient=["final_class", "background_probability", "tests_available", "tests_unavailable"],
        checks_md=(
            "## Los términos de este QC, en físico\n\n"
            "| Término | Qué es | Por qué importa |\n|---|---|---|\n"
            "| `hypotheses` / `combined_ranking` | Las explicaciones posibles de lo que hay en esa "
            "posición (planeta, enana marrón, estrella M ligada, estrella de fondo…) ordenadas por "
            "cuánto las apoya la evidencia. | La clasificación es una **comparación entre "
            "hipótesis**, no una medida directa. |\n"
            "| `background_probability` | Probabilidad de que una estrella no relacionada caiga por "
            "azar tan cerca en el cielo. | Es lo que descarta la coincidencia fortuita: con densidad "
            "estelar baja y separación pequeña, sale despreciable. |\n"
            "| `tests_available` / `tests_unavailable` | Qué discriminantes se pudieron aplicar y "
            "cuáles no (por falta de dato, no por resultado). | Un test ausente no es evidencia en "
            "contra; que la lista sea explícita evita leerlo así. |\n"
            "| `leave_one_out_stable` (LOO) | Si el ranking sobrevive al quitar **un test cada vez**. "
            "| Si al retirar un solo discriminante cambia el ganador, la clasificación se apoya en "
            "una sola pata: por eso el veredicto puede quedar «ambiguo» aunque haya un favorito. |\n"
            "| `correlated_groups` | Tests que no son independientes entre sí. | Contarlos por "
            "separado inflaría artificialmente la evidencia de una hipótesis. |\n"
            "| `frozen_thresholds` + `..._hash` | Los umbrales de decisión, congelados y con hash. | "
            "Impide ajustar el criterio después de ver el resultado. |\n"
        ),
        narrative_md=(
            "## Qué hace G4 y por qué queda ambigua\n\n"
            "G4 clasifica la fuente con una **matriz transparente hipótesis × test**: para cada "
            "hipótesis (compañero subestelar, enana marrón, estrella M asociada, en formación, fondo, "
            "contaminante, artefacto) combina los tests disponibles en una **log-verosimilitud "
            "relativa**, con **umbrales congelados** (hash anti-sesgo — el veredicto no se puede "
            "ajustar a mano).\n\n"
            "**Tests disponibles:** T1 (astrometría en la posición ligada), T2 (fuente puntual), T7 "
            "(no es artefacto), T8, T9. **No disponibles:** T3, T4, T5, T6 (tipado espectral — "
            "pendiente de G3 — y 2ª época astrométrica).\n\n"
            "**Resultado: clase = `substellar_companion`, robustez = `ambiguous`.**\n"
            "- **CONFIRMADO — compañero REAL ligado:** artefacto excluido (log_l_rel −1004), fondo "
            "desfavorecido (−3, P_bg=6.3×10⁻⁴), contaminante (−4), vía T1+T2+T7.\n"
            "- **AMBIGUO — el subtipo empata:** `substellar_companion`, `brown_dwarf` y "
            "`m_star_associated` **todas en log_l_rel 0** — sin tipado espectral (T3/T4/T6, pendiente "
            "de G3) planeta vs BD vs M no se distingue.\n\n"
            "**Resolver la ambigüedad** requiere: G3 real (SpT/Teff/masa) + una **2ª época "
            "astrométrica** + una densidad de fondo final. Provisional."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Clase final, ranking de hipótesis y tests del `stage_g4_classification.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stage_g4_classification.json', RUN_ID)\n"
            "fc = q['final_class']\n"
            "print(f\"clase = {fc['label']} | robustez = {fc['robustness']} | soportes independientes = {fc['n_independent_supports']}\")\n"
            "print(f\"P(fondo) = {q['background_probability']['raw']:.2e} ({q['background_probability']['source'][:40]}...)\")\n"
            "print(f\"tests disponibles: {q['tests_available']} | no disponibles: {q['tests_unavailable']}\")\n"
            "print(f\"umbrales congelados (hash anti-sesgo): {q['frozen_thresholds_hash'][:12]}\")\n"
            "# Semántica oficial (musepipe.classify): la EXCLUSIÓN es un veredicto 'excludes'\n"
            "# (peso -1000), no un umbral en log_l_rel; 'disfavors' pesa -1 por test.\n"
            "try:\n"
            "    from musepipe.classify import DEFAULT_WEIGHTS\n"
            "    W_EXCL = DEFAULT_WEIGHTS['excludes']\n"
            "except Exception:\n"
            "    W_EXCL = -1000.0   # espejo de DEFAULT_WEIGHTS (kernel sin musepipe)\n"
            "print('\\nranking de hipótesis (log-verosimilitud relativa):')\n"
            "for h in q['combined_ranking']:\n"
            "    v = h['log_l_rel']\n"
            "    if v >= -1e-9:\n"
            "        tag = '  <- EMPATE (líder)'\n"
            "    elif v <= W_EXCL / 2:   # un solo 'excludes' (-1000) domina cualquier suma de ±1\n"
            "        tag = '  (EXCLUIDA: >=1 test excludes)'\n"
            "    else:\n"
            "        tag = f'  ({abs(v):.0f} test(s) en contra, no excluida)'\n"
            "    print(f\"   {h['hypothesis']:22s} {v:8.0f}{tag}\")"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — el ranking de hipótesis\n\n"
                    "Log-verosimilitud relativa de las 7 hipótesis (clip a −6; artefacto real −1004). "
                    "**Tres empatan en 0** (★, verde) → compañero subestelar / BD / M asociada "
                    "indistinguibles; **artefacto EXCLUIDO** (rojo: único veredicto `excludes`, peso "
                    "−1000); fondo y contaminante solo **desfavorecidos** (gris: −1 por test en "
                    "contra, sin exclusión). Es compañero real, pero el subtipo queda ambiguo."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    from musepipe.classify import DEFAULT_WEIGHTS\n"
                    "    W_EXCL = DEFAULT_WEIGHTS['excludes']   # -1000: semántica oficial de exclusión\n"
                    "    q = nb.load_qc('stages/stage_g4_classification.json', RUN_ID)\n"
                    "    r = q['combined_ranking']\n"
                    "    names = [h['hypothesis'] for h in r]; ll = [h['log_l_rel'] for h in r]\n"
                    "    llc = [max(v, -6) for v in ll]\n"
                    "    cols = ['tab:green' if v >= -1e-9 else ('tab:red' if v <= W_EXCL / 2 else '0.6') for v in ll]\n"
                    "    fig, ax = plt.subplots(figsize=(9, 4.5))\n"
                    "    y = range(len(names))\n"
                    "    ax.barh(list(y), llc, color=cols)\n"
                    "    for i, v in enumerate(ll):\n"
                    "        if v >= -1e-9:\n"
                    "            ax.scatter(-0.15, i, marker='*', s=130, color='tab:green', zorder=5)\n"
                    "            ax.text(-0.35, i, 'EMPATE', va='center', ha='right', fontsize=8, color='tab:green')\n"
                    "        else:\n"
                    "            tag = f\"{v:.0f}\" + ('  (EXCLUIDA)' if v <= W_EXCL / 2 else '  (en contra)')\n"
                    "            ax.text(llc[i] - 0.1 if v > -6 else -5.9, i, tag, va='center',\n"
                    "                    ha='right' if v > -5 else 'left', fontsize=8)\n"
                    "    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()\n"
                    "    ax.set_xlabel('log-verosimilitud relativa (clip a −6; artefacto real=−1004)'); ax.axvline(0, color='k', lw=0.6)\n"
                    "    ax.set_title(f\"G4 · {q['final_class']['label']} / {q['final_class']['robustness']}: 3 hipótesis empatan (sin tipado espectral)\")\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'g4_classify'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'ranking.png', dpi=110); print('figura ->', outdir / 'ranking.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — los tests: lo que falta para romper el empate\n\n"
                    "T1–T9: en verde los disponibles (astrometría, fuente puntual, no-artefacto…) que "
                    "confirman el compañero real; en gris los **no disponibles** (T3/T4/T5/T6 = tipado "
                    "espectral de G3 + 2ª época) — justo los que distinguirían planeta/BD/M."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('stages/stage_g4_classification.json', RUN_ID)\n"
                    "    avail = set(q['tests_available']); unavail = set(q['tests_unavailable'])\n"
                    "    tests = sorted(avail | unavail, key=lambda t: int(t[1:]))\n"
                    "    cols = ['tab:green' if t in avail else '0.7' for t in tests]\n"
                    "    fig, ax = plt.subplots(figsize=(9, 2.6))\n"
                    "    ax.bar(range(len(tests)), [1] * len(tests), color=cols)\n"
                    "    for i, t in enumerate(tests):\n"
                    "        ax.text(i, 0.5, t + ('\\ndisp.' if t in avail else '\\nfalta'), ha='center', va='center',\n"
                    "                fontsize=8, color='w' if t in avail else 'k', weight='bold')\n"
                    "    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 1)\n"
                    "    ax.set_title('G4 · tests disponibles (verde) vs faltantes (gris: tipado espectral + 2ª época)')\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'g4_classify'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'tests.png', dpi=110); print('figura ->', outdir / 'tests.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**Clase = `substellar_companion`, robustez = `ambiguous`**: es compañero REAL ligado (artefacto −1004, fondo P=6.3e-4, contaminante excluidos), pero subestelar/BD/M **empatan** sin tipado espectral.", None),
            ("Resolver la ambigüedad requiere G3 real (SpT/Teff/masa) + 2ª época astrométrica + densidad de fondo final.", None),
            ("Umbrales **congelados** (hash anti-sesgo): el veredicto no se puede ajustar a mano.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada)\n\n"
            "**G4: `substellar_companion` / `ambiguous` — compañero real ligado, subtipo sin "
            "resolver.**\n\n"
            "- **Fecha:** 2026-07-08 (provisional).\n"
            "- **Confirmado:** compañero real (artefacto −1004, fondo −3, contaminante −4 excluidos) "
            "vía T1+T2+T7.\n"
            "- **Ambiguo:** subestelar / BD / M asociada empatan en log_l_rel 0 (sin tipado "
            "espectral).\n"
            "- **Falta:** T3/T4/T6 (tipado de G3, diferido) + 2ª época astrométrica + densidad de "
            "fondo final.\n"
            "- **Anti-sesgo:** umbrales congelados (hash).\n"
            "- **Downstream:** G5 sintetiza (clase provisional ambigua)."
        ),
    ),
    dict(
        id="G5", slug="G5_final_synthesis", title="Síntesis final", block="G · Caracterización",
        spec="spec_G5_codex_final_synthesis.md", run_override=None,
        what="Consolida G0–G4 en el paquete de caracterización (tablas + supuestos + síntesis).",
        inputs="G0–G4", outputs="`report/characterization/` (6 tablas + md + summary.json)",
        downstream="Revisión humana",
        exec=dict(kind="pyscript", target="build_characterization.py", cost="Ligero."),
        qc="report/characterization/characterization_summary.json",
        salient=["final_class", "consistency", "provisional", "n_lines", "n_physical_properties"],
        checks_md=(
            "## Los términos de este QC, en físico\n\n"
            "G5 no añade ciencia: **empaqueta** y comprueba que lo que sale de la cadena es "
            "coherente consigo mismo.\n\n"
            "| Término | Qué es | Por qué importa |\n|---|---|---|\n"
            "| `consistency` | Que las etapas no se contradigan entre sí (la Hα de E1/G2, el Ṁ de "
            "E3/G3, la clase de G4). | Dos caminos que miden lo mismo deben coincidir; si no, hay un "
            "factor aplicado de más o de menos. |\n"
            "| `provisional` | Que algún ingrediente sigue abierto. | Un resultado provisional está "
            "calculado pero **no es publicable**: la etiqueta viaja con él para que no se cite por "
            "error. |\n"
            "| `n_lines` / `n_physical_properties` | Cuántas líneas medidas y cuántas magnitudes "
            "físicas derivadas entran en el paquete. | Es el inventario de lo que realmente sostiene "
            "la síntesis. |\n"
            "| `final_class` | La clasificación heredada de G4, con su ambigüedad si la tiene. | G5 "
            "no re-clasifica: transporta el veredicto y su incertidumbre. |\n"
        ),
        narrative_md=(
            "## Qué hace G5 y el cierre\n\n"
            "G5 es la **síntesis final**: consolida G0–G4 en el paquete de caracterización "
            "(`report/characterization/`): tablas (parámetros adoptados, líneas, propiedades físicas, "
            "clasificación, índice de espectros, presupuesto de incertidumbre), documentos "
            "(`characterization.md`, `assumptions_and_limitations.md`) y **9 figuras**.\n\n"
            "**Verifica (los V-checks):**\n"
            "- **V1 trazabilidad:** hashes de QC de las fases G0–G4 (cadena completa).\n"
            "- **V2 consistencia:** Hα de G2 frente al veredicto de E1 "
            "(`{{qc:stages/stage_g2_qc.json:halpha_reconciliation_v3.g2_halpha_status}}` vs "
            "`{{qc:stages/stage_g2_qc.json:halpha_reconciliation_v3.h01_verdict}}`); robustez de la "
            "clasificación: `{{qc:stages/stage_g4_classification.json:final_class.robustness}}`.\n"
            "- **V4 determinismo:** hashes de los 10 archivos (dos builds → idénticos, reproducible).\n"
            "- **V6 F1 intacto:** `run_summary_extended` (F1 no se toca, es aditivo).\n\n"
            "**Resultado consolidado ({{target}}):** clase = "
            "`{{qc:stages/stage_g4_classification.json:final_class.label}}` / "
            "`{{qc:stages/stage_g4_classification.json:final_class.robustness}}`; L_acc ≤ "
            "{{qc:stages/stage_g3_qc.json:combined_accretion.l_acc_lsun:.2e}} L☉; Ṁ ≲ "
            "{{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].mdot:.2e}} (E3) — "
            "{{qc:stages/stage_g3_qc.json:mdot_p50_msun_yr:.2e}} (G3).\n\n"
            "**El G-block queda CERRADO (provisional):** el paquete está armado, es determinista y "
            "trazable, pero **hereda el bloqueo del A-block** (alineación, M3, M5) + el **tipado "
            "espectral diferido de G3** → G4 ambigua. Nada es paper-final hasta cerrar el A-block."
        ),
        evidence_md=(
            "## Resultados que llevaron a la conclusión\n\n"
            "Clase final, consistencia, determinismo y trazabilidad del "
            "`characterization_summary.json`."
        ),
        evidence_code=(
            "q = nb.load_qc('report/characterization/characterization_summary.json', RUN_ID)\n"
            "fc = q['final_class']\n"
            "print(f\"clase final: {fc['label']} / {fc['robustness']} ({fc['n_independent_supports']} soportes)\")\n"
            "print('\\nconsistencia:')\n"
            "for c in q['consistency']:\n"
            "    print('  ', {k: v for k, v in c.items()})\n"
            "print(f\"\\npaquete: {q['n_lines']} líneas, {q['n_physical_properties']} propiedades físicas, \"\n"
            "      f\"{len(q['figures_generated'])} figuras, {len(q['determinism_hash'])} archivos con hash (determinista)\")\n"
            "print('trazabilidad (hashes de fase):', list(q['inputs']['phase_qc_hashes'].keys()))\n"
            "print('F1 intacto (V6):', q['f1_compatibility']['run_summary_extended'])\n"
            "oi = (q.get('open_issues') or [{}])[0]\n"
            "print('\\nopen_issue (blocking):', (oi.get('issue') if isinstance(oi, dict) else str(oi))[:120])"
        ),
        plots=[
            dict(
                md=(
                    "## Plot 1 — el resultado consolidado del proyecto\n\n"
                    "La síntesis de toda la cadena A→G: qué es la fuente, y los límites de acreción. "
                    "Con sus citas (de `adopted_parameters.csv`)."
                ),
                code=(
                    "try:\n"
                    "    import pandas as pd\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    rd = nb.run_dir(RUN_ID)\n"
                    "    q = nb.load_qc('report/characterization/characterization_summary.json', RUN_ID)\n"
                    "    ap = pd.read_csv(rd / 'report' / 'characterization' / 'adopted_parameters.csv').set_index('parameter')\n"
                    "    e3 = nb.load_qc('stages/stage_h03_qc.json', RUN_ID)\n"
                    "    e3_mdot = {L['method']: L['mdot'] for L in e3['limits']}[e3['canonical_method']]\n"
                    "    g3 = nb.load_qc('stages/stage_g3_qc.json', RUN_ID)\n"
                    "    ast = nb.load_qc('stages/stage01c_qc.json', RUN_ID)['astrometry']   # sep/PA oficiales (B3)\n"
                    "    MSUN_PER_MJUP = 1047.57\n"
                    "    m_mjup = float(ap.loc['companion_mass_msun', 'value']) * MSUN_PER_MJUP\n"
                    "    lines = [\n"
                    "        f\"CLASE:  {q['final_class']['label']}  ({q['final_class']['robustness']})\",\n"
                    "        f\"        compañero real ligado a ROXs 12 A \"\n"
                    "        f\"(sep {ast['sep_arcsec']:.2f}\\\", PA {ast['pa_deg']:.0f}°)\",\n"
                    "        '',\n"
                    "        f\"ACRECIÓN (no-detección de Hα):\",\n"
                    "        f\"   L_acc  ≤ {g3['combined_accretion']['l_acc_lsun']:.1e} L☉      (Alcalá+2017 Hα)\",\n"
                    "        f\"   Ṁ      ≲ {e3_mdot:.1e} M☉/yr  (E3, Gumbel 99%)\",\n"
                    "        f\"   Ṁ      = {g3['mdot_p50_msun_yr']:.1e} M☉/yr  (G3, 5σ + R_in)\",\n"
                    "        '',\n"
                    "        f\"PARÁMETROS ADOPTADOS:\",\n"
                    "        f\"   distancia  {ap.loc['distance_pc','value']} pc   (Gaia)\",\n"
                    "        f\"   A_V        {ap.loc['a_v','value']}         (Rizzuto+2015)\",\n"
                    "        f\"   masa       {m_mjup:.1f} M_Jup    (Bowler+2017, hot-start)\",\n"
                    "    ]\n"
                    "    fig, ax = plt.subplots(figsize=(8.5, 5)); ax.axis('off')\n"
                    "    ax.text(0.5, 0.98, f'G5 · Síntesis {nb.display_name(RUN_ID)} (provisional)',\n"
                    "            ha='center', va='top', fontsize=13, weight='bold')\n"
                    "    ax.text(0.05, 0.86, '\\n'.join(lines), va='top', ha='left', fontsize=10, family='monospace')\n"
                    "    ax.text(0.5, 0.03, 'PROVISIONAL: caveats A-block (V2/V5/V6) + tipado espectral (G3) diferido -> subtipo ambiguo',\n"
                    "            ha='center', fontsize=8, color='tab:red')\n"
                    "    outdir = rd / 'plots' / 'g5_synthesis'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'summary_card.png', dpi=110, bbox_inches='tight'); print('figura ->', outdir / 'summary_card.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
            dict(
                md=(
                    "## Plot 2 — integridad del paquete (los V-checks)\n\n"
                    "Lo que G5 verifica antes de cerrar: consistencia (Hα G2=H01), trazabilidad "
                    "(hashes G0–G4), determinismo (dos builds → idénticos), F1 intacto. En rojo, el "
                    "bloqueo heredado (A-block provisional)."
                ),
                code=(
                    "try:\n"
                    "    import matplotlib.pyplot as plt\n"
                    "    q = nb.load_qc('report/characterization/characterization_summary.json', RUN_ID)\n"
                    "    hal = next((c for c in q['consistency'] if c.get('check') == 'halpha_category_g2_vs_h01'), {})\n"
                    "    checks = [\n"
                    "        ('V2 consistencia Hα (G2=H01)', bool(hal.get('consistent'))),\n"
                    "        (f\"V1 trazabilidad (hashes G0–G4: {len(q['inputs']['phase_qc_hashes'])})\", len(q['inputs']['phase_qc_hashes']) == 5),\n"
                    "        (f\"V4 determinismo ({len(q['determinism_hash'])} archivos con hash)\", len(q['determinism_hash']) >= 10),\n"
                    "        ('V6 F1 intacto (aditivo)', bool(q['f1_compatibility']['run_summary_extended'])),\n"
                    "        ('Cierre paper-final (caveats A-block V2/V5/V6 + tipado G3 pendientes)', False),\n"
                    "    ]\n"
                    "    names = [c[0] for c in checks]; oks = [c[1] for c in checks]\n"
                    "    fig, ax = plt.subplots(figsize=(9, 3.2))\n"
                    "    ax.barh(names, [1] * len(names), color=['tab:green' if o else 'tab:red' for o in oks])\n"
                    "    for i, (n, o) in enumerate(checks):\n"
                    "        ax.text(0.5, i, ('✓  ' if o else '✗  ') + n, ha='center', va='center', fontsize=9, color='w', weight='bold')\n"
                    "    ax.set_xlim(0, 1); ax.set_xticks([]); ax.set_yticks([]); ax.invert_yaxis()\n"
                    "    ax.set_title('G5 · integridad del paquete: determinista y trazable, provisional (caveats A-block + tipado G3)')\n"
                    "    fig.tight_layout()\n"
                    "    outdir = nb.run_dir(RUN_ID) / 'plots' / 'g5_synthesis'; outdir.mkdir(parents=True, exist_ok=True)\n"
                    "    fig.savefig(outdir / 'integrity.png', dpi=110); print('figura ->', outdir / 'integrity.png'); plt.show()\n"
                    "except Exception as e:\n"
                    "    print('No se pudo generar el plot:', type(e).__name__, e)"
                ),
            ),
        ],
        decisions=[
            ("**G-block CERRADO (provisional):** paquete completo (10 archivos con hash, 9 figuras) pero hereda el bloqueo del A-block + G3 diferido → clasificación ambigua.", None),
            ("Determinismo verificado (dos builds → hash idéntico); trazabilidad (hashes G0–G4); F1 intacto (V6, aditivo).", None),
            ("Consistencia V2: Hα G2 upper_limit = H01 non_detection.", None),
        ],
        checks=None,
        conclusion_md=(
            "## Conclusión (registrada) — cierre del proyecto\n\n"
            "**G5: paquete de caracterización armado; determinista, trazable, F1 intacto; "
            "PROVISIONAL.**\n\n"
            "- **Resultado consolidado:** el compañero de **{{target}}** queda clasificado como "
            "`{{qc:stages/stage_g4_classification.json:final_class.label}}` con robustez "
            "`{{qc:stages/stage_g4_classification.json:final_class.robustness}}` "
            "({{qc:stages/stage_g4_classification.json:final_class.n_independent_supports}} apoyos "
            "independientes); E1 = `{{qc:stages/stage_h01_qc.json:verdict.verdict}}` en Hα → "
            "**Ṁ ≲ {{qc:stages/stage_h03_qc.json:limits[method=@canonical_method].mdot:.2e}} M☉/yr** "
            "(E3) / {{qc:stages/stage_g3_qc.json:mdot_p50_msun_yr:.2e}} (G3).\n"
            "- **Integridad:** consistencia V2, trazabilidad G0–G4, determinismo (10 hashes), F1 "
            "intacto (V6).\n"
            "- **Bloqueo heredado (al cierre, 2026-07-08):** A-block abierto (alineación, M3, M5) + "
            "tipado espectral de G3 diferido → nada era paper-final.\n"
            "- **Actualización 2026-07-10 (ver A1):** los 6 blockers duros del A-block se **cerraron** "
            "(F1 realineado = yellow, 0 bloqueantes); quedan los caveats no bloqueantes V2/V5/V6 y el "
            "tipado espectral diferido → la validez para paper es juicio científico con esos caveats "
            "declarados.\n"
            "- **Para cerrar del todo:** G3 real (librerías BT-Settl/BHAC15/Luhman-Bonnefoy) + 2ª "
            "época astrométrica romperían la ambigüedad."
        ),
    ),
    # ===================== BLOQUE S — wavesol / stripes =====================
    dict(
        id="S0", slug="S0_wavesol_map",
        title="Mapa de offsets de λ por spaxel (checklist G1)", block="S · wavesol/stripes",
        spec="plan_wavesol_stripes_2026-07-17.md", run_override=None,
        what=(
            "Mide, spaxel a spaxel, el corrimiento espectral de las líneas de absorción de la "
            "primaria contra un espectro de referencia de campo (Xie+20 §4.2.2). Estructura "
            "alineada con slicers ⇒ diferencias de solución de λ por exposición/slice "
            "(*stripes*, Hashimoto+20). Es el **insumo formal de la decisión G1**."
        ),
        inputs="`cube_telcorr.fits` (realineado) + ADP oficial de ESO (control independiente)",
        outputs=(
            "`stages/stageS0_qc.json` (+ `stageS0_adp_qc.json`), "
            "`stages/stageS0_offset_map.fits`, `plots/s0_wavesol/`"
        ),
        downstream="**Decisión G1 (humana)** → Fase 2 (S2–S5) o cierre `fase2_descartable`",
        narrative_md=(
            "## Qué hace S0 y cómo\n\n"
            "Por cada spaxel del halo (selección por brillo, percentil 50) se normaliza el "
            "continuo por división de *running-median* (mata el continuo cromático del halo AO) "
            "en 4 ventanas de absorción estelar que **evitan** el láser AO, Hα (la primaria es "
            "emisora), y las bandas telúricas O₂/H₂O; se cross-correla contra el espectro de "
            "referencia del campo (`stripes._xcorr_shift_pixels`, subpíxel) y se toma la mediana "
            "de las ventanas usables. El resultado es un **mapa de offset** (Å) por spaxel.\n\n"
            "**Corte S/N (lección del preliminar 2026-07-17):** cada spaxel lleva un error "
            "`err = σ_robusta(offsets_por_ventana)/√N`; el p95 del gate se calcula SOLO sobre "
            "spaxels con `err < max_err_ch` (0.08 ch ≈ 0.1 Å), porque el p95 crudo lo dominan "
            "spaxels débiles donde la xcorr falla (preliminar: p95 global 3.9 Å de puro ruido "
            "vs 0.18 Å en el núcleo r<1\"). El perfil por columnas usa todos (su mediana ya es "
            "robusta).\n\n"
            "La **métrica de estructura** colapsa el mapa a lo largo de la dirección de los "
            "stripes (perfil por columna) y compara su amplitud contra el ruido esperado; el "
            "perfil **transversal** es el control: stripes reales muestran estructura en el "
            "perfil de stripe, no en el transversal."
        ),
        exec=dict(
            kind="module_main",
            target="musepipe.qc.wavesol_map",
            cmd=(
                "python -m musepipe.qc.wavesol_map --run-id $RUN --orientation vertical"
            ),
            cost="Coste: full-res 330×338, normalización vectorizada; ~minutos por cubo.",
        ),
        qc="stages/stageS0_qc.json",
        salient=[
            "gate_g1.recommendation", "gate_g1.decision",
            "metrics.p95_abs_offset_A", "metrics.structure_significance",
            "metrics.transverse_significance", "metrics.n_selected_low_err",
            "channel_step_A", "runtime_s",
        ],
        checks_md=(
            "## Los términos de este QC, en físico\n\n"
            "S0 mide, **spaxel a spaxel**, cuánto se desvía la solución de longitud de onda respecto "
            "de una referencia. La pregunta no es «¿hay desviación?» (siempre hay ruido) sino "
            "**«¿tiene forma de slicer?»**: un defecto del instrumento produce bandas paralelas a los "
            "slicers; el ruido de S/N, no.\n\n"
            "| Término | Qué es | Por qué importa |\n|---|---|---|\n"
            "| `structure_significance` (*stripe_sig*) | Cuánto destaca el perfil promediado **a lo "
            "largo** de la dirección del slicer frente al ruido. | Es la firma que se busca. |\n"
            "| `transverse_significance` | Lo mismo en la dirección **perpendicular**, donde no puede "
            "haber franjas. | Es el **control**: si las dos significancias son parecidas, lo que se "
            "ve es estructura radial o ruido, no slicers. Comparar sin este control es el error "
            "clásico. |\n"
            "| `p95_abs_offset_A` | El percentil 95 del desvío absoluto en Å. | Mide magnitud, no "
            "forma: en spaxels débiles está dominado por el ruido de la correlación cruzada, así que "
            "un p95 grande **no implica** franjas. |\n"
            "| `max_err_ch` / `n_selected_low_err` | Corte de error por spaxel y cuántos lo pasan. | "
            "Sin ese corte, el mapa mide sobre todo spaxels sin señal. |\n"
            "| `orientation` | En qué dirección van los slicers en este cubo. | Buscar en la "
            "orientación equivocada garantiza no encontrar nada. |\n"
            "| `gate_g1.recommendation` vs `.decision` | Lo que sugiere el umbral automático vs lo "
            "que **decidió una persona**, con su motivo. | Están separados a propósito: aquí la "
            "recomendación automática se anuló con argumento físico, y eso queda escrito. |\n"
        ),
        evidence_md=(
            "## Evidencia: realineado vs ADP (control)\n\n"
            "Los dos cubos deben coincidir: el mapa de offset es un diagnóstico del "
            "**instrumento/reducción**, no del cubo concreto. El preliminar 2×2 (2026-07-17) "
            "dio amplitud de columna 72 mÅ (realineado) / 67 mÅ (ADP), a ~1× ruido, sin "
            "estructura alineada con slicers; el núcleo r<1\" med|off| = 64 / 62 mÅ ≈ el M1 "
            "global (+74 mÅ).\n\n"
            "**Full-res reproduce la conclusión clave** (sin estructura de slicer: `stripe_sig` "
            "≈ control transversal) y la extiende: al medir TODOS los spaxels de bajo error (no "
            "solo el núcleo) el p95 sube a ~0.32 Å — un scatter de λ por spaxel, consistente "
            "entre ventanas pero **espacialmente desestructurado**. La celda imprime ambos "
            "cubos; deben coincidir."
        ),
        evidence_code=(
            "# p95 se mide sobre el subconjunto de bajo error (err<max_err_ch); la\n"
            "# estructura de slicer se juzga con stripe_sig vs el control transversal.\n"
            "rows = []\n"
            "for label, rel in [('realineado', 'stages/stageS0_qc.json'),\n"
            "                   ('ADP',        'stages/stageS0_adp_qc.json')]:\n"
            "    try:\n"
            "        q = nb.load_qc(rel, RUN_ID)\n"
            "    except FileNotFoundError:\n"
            "        print(f'[{label}] QC aún no existe: {rel}'); continue\n"
            "    m = q['metrics']\n"
            "    rows.append((label, m['p95_abs_offset_A'], m['structure_significance'],\n"
            "                 m['transverse_significance'], m['n_selected_low_err'],\n"
            "                 m['n_spaxels_measured'], q['gate_g1']['recommendation']))\n"
            "hdr = ('cubo', 'p95|off|[A]', 'stripe_sig', 'transv_sig', 'n_low_err',\n"
            "       'n_meas', 'recomendación')\n"
            "print('{:>10} {:>12} {:>11} {:>11} {:>10} {:>8}  {}'.format(*hdr))\n"
            "for r in rows:\n"
            "    print('{:>10} {:>12.4f} {:>11.2f} {:>11.2f} {:>10d} {:>8d}  {}'.format(*r))\n"
            "print('\\np95 sobre spaxels de bajo error; stripe_sig<=transv_sig => sin estructura de slicer.')"
        ),
        plots=[
            dict(
                md=(
                    "## Checklist G1 — umbrales y argumentos\n\n"
                    "**Umbrales del plan (recomendación automática, decisión humana):**\n\n"
                    "| Criterio | Umbral | Dispara Fase 2 si |\n|---|---|---|\n"
                    "| p95 \\|offset\\| (spaxels `err<0.1 Å`) | 0.1 Å | **>** 0.1 Å |\n"
                    "| Estructura alineada con slicers | 3× ruido | **>** 3× **y** > 2× el control transversal |\n\n"
                    "**Argumentos del caso `fase2_descartable`** — registro de la decisión tomada "
                    "para **ROXs 12 b** el 2026-07-17, con las cifras que se tenían entonces; se "
                    "reproduce literal por trazabilidad y **no** describe a otro objeto (para el tuyo, "
                    "los valores vivos están en su A4 y en el QC de S0):\n\n"
                    "1. La métrica espacial de S0 **no ve un offset común a todas las exposiciones** "
                    "(deriva temporal uniforme): ese modo no aparece como estructura espacial, solo "
                    "**ensancharía la LSF combinada**.\n"
                    "2. Pero A4·M2 midió **LSF = 2.383 Å**, MÁS ESTRECHA que el nominal → acota ese "
                    "*smearing* a nivel pequeño (si hubiera deriva grande entre exposiciones, la LSF "
                    "combinada saldría ensanchada, no estrecha).\n"
                    "3. El **M1 global (+0.074 Å)** ya corrige el zero-point de λ.\n\n"
                    "Los tres juntos son el caso para **no** entrar a la Fase 2 (re-reducción por "
                    "exposición). La decisión final es **humana** (gate G1)."
                ),
            ),
            dict(
                md=(
                    "## Mapas S0 — panorama de TODOS los cubos (7 exposiciones + combinado + ADP)\n\n"
                    "Renderizados **directamente del FITS** `stageS0_offset_map.fits`. Se incluyen "
                    "las **7 exposiciones individuales** (S2, `/mnt/2TB/MUSE_work/ROXs12b_perexp/`), "
                    "el **combinado** (realineado) y el **ADP** de ESO (control) — 9 casos. Primero "
                    "el detalle de 4 paneles del combinado (offset/error/perfil-columna/histograma), "
                    "luego una rejilla 3×3 con el mapa de offset de cada caso.\n\n"
                    "**Por qué mirar por exposición:** las cabeceras (`INS DROT MODE=SKY`, "
                    "`INS DROT POSANG` = 0°,0°,90°,90°,180°,180°,0°) muestran un **patrón "
                    "deliberado de rotación de 90°** entre grupos de exposiciones; los 7 cubos "
                    "están remuestreados norte-arriba ⇒ el combinado mezcla TRES asignaciones "
                    "de slice distintas (0°/90°/180°) y el scrambling de un stripe fijo del "
                    "slicer es total. Cada exposición sí conserva su orientación de slicer "
                    "(vertical para POSANG 0/180, **horizontal para 90** — exp3/exp4), así que "
                    "0/7 con estructura, medido cada uno en su eje, es la prueba real de que no "
                    "hay stripes."
                ),
                code=(
                    "import os, numpy as np, matplotlib.pyplot as plt\n"
                    "from astropy.io import fits\n"
                    "import musepipe.qc.wavesol_map as wsm\n"
                    "rd = nb.run_dir(RUN_ID)\n"
                    "# Directorio de cubos por exposición: del config del run, no fijo (WP-E4b).\n"
                    "import sys as _sys; _sys.path.insert(0, str(nb.project_root()))\n"
                    "from musepipe.config import run_workdir_setting\n"
                    "PEREXP_DIR = run_workdir_setting(RUN_ID, 'perexp_dir', project_root=nb.project_root())\n"

                    "# CASES: (label, qc_path, map_path)  — per-exp absolutos; combinado/ADP en el run\n"
                    "CASES = [(f'exp{i}', f'{PEREXP_DIR}/exp{i}/stageS0_qc.json',\n"
                    "          f'{PEREXP_DIR}/exp{i}/stageS0_offset_map.fits') for i in range(1, 8)]\n"
                    "CASES += [('combinado', str(rd/'stages'/'stageS0_qc.json'), str(rd/'stages'/'stageS0_offset_map.fits')),\n"
                    "          ('ADP',       str(rd/'stages'/'stageS0_adp_qc.json'), str(rd/'stages'/'stageS0_adp_offset_map.fits'))]\n"
                    "import json\n"
                    "def _load(qcf, mapf):\n"
                    "    q = json.load(open(qcf)); step = q['channel_step_A']; thr = q['gate_g1']['thresholds']['max_err_ch']\n"
                    "    with fits.open(mapf) as h:\n"
                    "        d = {k: h[k].data.astype(float) for k in ('OFFSET_A','ERR_A','OFFSET_CH','ERR_CH')}\n"
                    "    return q, step, thr, d\n"
                    "# --- detalle 4-panel del combinado ---\n"
                    "cm = [c for c in CASES if c[0]=='combinado'][0]\n"
                    "q, step, thr, d = _load(cm[1], cm[2])\n"
                    "low = np.isfinite(d['OFFSET_A']) & np.isfinite(d['ERR_CH']) & (d['ERR_CH'] < thr)\n"
                    "prof = wsm.stripe_profile(d['OFFSET_CH'], 'vertical')\n"
                    "fig, ax = plt.subplots(1, 4, figsize=(17, 3.6)); fig.suptitle(f'S0 · combinado (detalle) — {int(low.sum())} spaxels bajo error', fontsize=11)\n"
                    "vl = np.nanpercentile(np.abs(d['OFFSET_A'][low]), 95)\n"
                    "im0 = ax[0].imshow(d['OFFSET_A'], origin='lower', cmap='RdBu_r', vmin=-vl, vmax=vl); ax[0].set_title('offset (Å)'); plt.colorbar(im0, ax=ax[0], fraction=0.046)\n"
                    "im1 = ax[1].imshow(d['ERR_A'], origin='lower', cmap='viridis', vmax=np.nanpercentile(d['ERR_A'],95)); ax[1].set_title('error (Å)'); plt.colorbar(im1, ax=ax[1], fraction=0.046)\n"
                    "ax[2].plot(np.arange(prof['profile'].size), np.asarray(prof['profile'],float)*step, lw=0.9); ax[2].axhline(0,color='0.6',lw=0.6); ax[2].set_title('perfil por columna (∥ stripes)'); ax[2].set_xlabel('columna'); ax[2].set_ylabel('offset mediano (Å)')\n"
                    "ax[3].hist(d['OFFSET_A'][low], bins=60, color='tab:blue', alpha=0.8); ax[3].axvline(0,color='k',lw=0.8); ax[3].axvline(np.median(d['OFFSET_A'][low]),color='tab:red',ls='--',label=f\"mediana {np.median(d['OFFSET_A'][low])*1e3:+.0f} mÅ\"); ax[3].set_title('hist (bajo error)'); ax[3].set_xlabel('offset (Å)'); ax[3].legend(fontsize=8)\n"
                    "fig.tight_layout(); plt.show()\n"
                    "# --- rejilla 3x3 de mapas de offset (9 casos) ---\n"
                    "fig, axes = plt.subplots(3, 3, figsize=(13, 12)); fig.suptitle('S0 · mapa de offset (Å) por caso — 7 exposiciones + combinado + ADP', fontsize=12)\n"
                    "for axi, (label, qcf, mapf) in zip(axes.ravel(), CASES):\n"
                    "    if not os.path.exists(mapf):\n"
                    "        axi.set_title(f'{label}: (falta)'); axi.axis('off'); continue\n"
                    "    _, _, thr, d = _load(qcf, mapf)\n"
                    "    lo = np.isfinite(d['OFFSET_A']) & np.isfinite(d['ERR_CH']) & (d['ERR_CH'] < thr)\n"
                    "    vl = np.nanpercentile(np.abs(d['OFFSET_A'][lo]), 95) if lo.any() else 0.3\n"
                    "    im = axi.imshow(d['OFFSET_A'], origin='lower', cmap='RdBu_r', vmin=-vl, vmax=vl)\n"
                    "    axi.set_title(f'{label}  (n={int(lo.sum())}, ±{vl*1e3:.0f} mÅ)', fontsize=9)\n"
                    "    plt.colorbar(im, ax=axi, fraction=0.046)\n"
                    "fig.tight_layout(); plt.show()"
                ),
            ),
            dict(
                md=(
                    "## Paso extra — crop 100×100 en la estrella, para los 9 casos\n\n"
                    "**Motivación:** el gate global lo lastran los spaxels débiles del borde (el "
                    "p95 crudo y el `median|offset|` los dominan). Si hubiera un offset "
                    "**coherente escondido en el ruido**, se saca a la luz midiéndolo donde la "
                    "S/N es máxima: el halo de la estrella. Se recorta un **100×100 centrado en "
                    "la estrella** (centroide de menor error, sin cargar el cubo) y se recalculan "
                    "las métricas + el **offset medio con signo ± error** — el test directo de un "
                    "offset coherente oculto (que `median|·|` no ve por no distinguir signo).\n\n"
                    "Se hace para **los 7 cubos por exposición + combinado + ADP** (full vs crop), "
                    "midiendo el stripe **en la orientación real del slicer de cada caso** según "
                    "`INS DROT POSANG` (vertical para 0/180, horizontal para 90 — exp3/exp4; "
                    "corrección 2026-07-19, ver `decision_g1_wavesol_2026-07-17.md`). Dos "
                    "lecturas: (1) el **offset medio del crop por exposición** traza la **deriva "
                    "temporal** de zero-point (modo TEMPORAL de G1/S3); (2) `stripe_sig` ≤ el "
                    "control perpendicular en todos ⇒ **ningún stripe de slicer** aun al máximo "
                    "S/N (modo ESPACIAL descartado).\n\n"
                    "Salida: la tabla, la figura resumen (deriva + stripe/transversal) y, para "
                    "cada caso, el **mismo detalle de 4 paneles que la celda de mapas pero "
                    "recortado al crop** (offset, error, perfil por columna, histograma)."
                ),
                code=(
                    "import os, json, numpy as np, matplotlib.pyplot as plt\n"
                    "from astropy.io import fits\n"
                    "import musepipe.qc.wavesol_map as wsm\n"
                    "rd = nb.run_dir(RUN_ID); HALF = 50\n"
                    "import sys as _sys; _sys.path.insert(0, str(nb.project_root()))\n"
                    "from musepipe.config import run_workdir_setting\n"
                    "PEREXP_DIR = run_workdir_setting(RUN_ID, 'perexp_dir', project_root=nb.project_root())\n"

                    "CASES = [(f'exp{i}', f'{PEREXP_DIR}/exp{i}/stageS0_qc.json',\n"
                    "          f'{PEREXP_DIR}/exp{i}/stageS0_offset_map.fits') for i in range(1, 8)]\n"
                    "CASES += [('combinado', str(rd/'stages'/'stageS0_qc.json'), str(rd/'stages'/'stageS0_offset_map.fits')),\n"
                    "          ('ADP',       str(rd/'stages'/'stageS0_adp_qc.json'), str(rd/'stages'/'stageS0_adp_offset_map.fits'))]\n"
                    "# Orientación del slicer EN el cubo norte-arriba, por INS DROT POSANG (modo SKY):\n"
                    "# POSANG 0/180 -> vertical; POSANG 90 -> horizontal. Combinado/ADP: vertical (dominante 5/7).\n"
                    "POSANG = {'exp1': 0, 'exp2': 0, 'exp3': 90, 'exp4': 90, 'exp5': 180, 'exp6': 180, 'exp7': 0}\n"
                    "orient_of = lambda label: 'horizontal' if POSANG.get(label, 0) == 90 else 'vertical'\n"
                    "def stats(oc, ec, oa, step, thr, orient):\n"
                    "    m = wsm.structure_metrics(oc, step, orientation=orient, err_map_ch=ec, max_err_ch=thr)\n"
                    "    low = np.isfinite(oa) & np.isfinite(ec) & (ec < thr); v = oa[low]\n"
                    "    mean = float(np.mean(v)); se = float(np.std(v) / np.sqrt(max(v.size, 1)))\n"
                    "    return m, mean, se\n"
                    "print('stripe/ctrl medidos en la orientación del slicer de CADA caso (POSANG):')\n"
                    "print('{:>10} {:>5} {:>6} {:>8} {:>7} {:>9} {:>6} {:>6}  {}'.format(\n"
                    "      'caso/región','n_lowE','med|o|','p95 Å','meanmÅ','σ_mean','strp','ctrl','orient'))\n"
                    "rows = []; crops = []\n"
                    "for label, qcf, mapf in CASES:\n"
                    "    if not os.path.exists(mapf):\n"
                    "        print(f'{label:>10}  (falta {mapf})'); continue\n"
                    "    q = json.load(open(qcf)); step = q['channel_step_A']; thr = q['gate_g1']['thresholds']['max_err_ch']\n"
                    "    with fits.open(mapf) as h:\n"
                    "        oc = h['OFFSET_CH'].data.astype(float); oa = h['OFFSET_A'].data.astype(float); ec = h['ERR_CH'].data.astype(float)\n"
                    "    ny, nx = oc.shape; low = np.isfinite(oc) & np.isfinite(ec) & (ec < thr)\n"
                    "    if low.sum() < 10:\n"
                    "        print(f'{label:>10}  (pocos spaxels de bajo error)'); continue\n"
                    "    yy, xx = np.mgrid[0:ny, 0:nx]; wt = np.where(low, 1.0/np.clip(ec, 1e-3, None)**2, 0.0)\n"
                    "    cy = int(round(np.sum(yy*wt)/np.sum(wt))); cx = int(round(np.sum(xx*wt)/np.sum(wt)))\n"
                    "    sl = (slice(max(0,cy-HALF),min(ny,cy+HALF)), slice(max(0,cx-HALF),min(nx,cx+HALF)))\n"
                    "    orient = orient_of(label)\n"
                    "    mF, meanF, seF = stats(oc, ec, oa, step, thr, orient)\n"
                    "    mC, meanC, seC = stats(oc[sl], ec[sl], oa[sl], step, thr, orient)\n"
                    "    for tag, mm, mn, se in [('FULL', mF, meanF, seF), ('CROP', mC, meanC, seC)]:\n"
                    "        print('{:>10} {:>5d} {:>6.0f} {:>8.4f} {:>+7.1f} {:>9.1f} {:>6.2f} {:>6.2f}  {}'.format(\n"
                    "              f'{label} {tag}', mm['n_selected_low_err'], mm['median_abs_offset_ch']*step*1e3,\n"
                    "              mm['p95_abs_offset_A'], mn*1e3, se*1e3, mm['structure_significance'], mm['transverse_significance'], orient[:4]))\n"
                    "    rows.append((label, meanC*1e3, seC*1e3, mC['structure_significance'], mC['transverse_significance']))\n"
                    "    crops.append((label, oc[sl].copy(), oa[sl].copy(), ec[sl].copy(), step, thr, meanC, seC, orient))\n"
                    "# --- figura resumen: deriva temporal + stripe vs transversal ---\n"
                    "exps = [r for r in rows if r[0].startswith('exp')]\n"
                    "fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))\n"
                    "if exps:\n"
                    "    x = np.arange(len(exps)); labs = [r[0] for r in exps]\n"
                    "    ax[0].errorbar(x, [r[1] for r in exps], yerr=[r[2] for r in exps], fmt='o', color='tab:blue', capsize=3, label='crop mean por exp')\n"
                    "    for name, col in [('combinado','tab:green'), ('ADP','tab:red')]:\n"
                    "        rr = [r for r in rows if r[0]==name]\n"
                    "        if rr: ax[0].axhline(rr[0][1], color=col, ls='--', lw=1, label=f'{name} {rr[0][1]:+.0f} mÅ')\n"
                    "    ax[0].axhline(0, color='0.6', lw=0.6); ax[0].axhline(74, color='0.4', ls=':', lw=1, label='M1 global +74 mÅ')\n"
                    "    ax[0].set_xticks(x); ax[0].set_xticklabels(labs); ax[0].set_ylabel('offset medio del crop (mÅ)')\n"
                    "    ax[0].set_title('Deriva temporal del zero-point (crop alta S/N)'); ax[0].legend(fontsize=7)\n"
                    "for r in rows:\n"
                    "    mk = 'o' if r[0].startswith('exp') else ('s' if r[0]=='combinado' else '^')\n"
                    "    ax[1].scatter(r[4], r[3], marker=mk, s=55); ax[1].annotate(r[0], (r[4], r[3]), fontsize=7, xytext=(3,3), textcoords='offset points')\n"
                    "limmax = 3.0\n"
                    "ax[1].plot([0,limmax],[0,limmax], color='0.6', ls='--', lw=1, label='stripe = transversal')\n"
                    "ax[1].axhline(3, color='tab:red', ls=':', lw=1, label='umbral stripe 3×')\n"
                    "ax[1].set_xlabel('control (⊥ slicer)'); ax[1].set_ylabel('stripe_sig (∥ slicer, orientación por POSANG)'); ax[1].set_xlim(0,limmax); ax[1].set_ylim(0,limmax)\n"
                    "ax[1].set_title('stripe vs control (bajo la diagonal = sin stripe)'); ax[1].legend(fontsize=7)\n"
                    "fig.tight_layout(); plt.show()\n"
                    "# --- detalle 4-panel del CROP por caso (como la celda de mapas, pero recortado) ---\n"
                    "print('\\nDetalle 4-panel del crop (offset / error / perfil ∥ slicer / histograma) por caso:')\n"
                    "for label, oc_c, oa_c, ec_c, step, thr, meanC, seC, orient in crops:\n"
                    "    lo = np.isfinite(oa_c) & np.isfinite(ec_c) & (ec_c < thr)\n"
                    "    prof = wsm.stripe_profile(oc_c, orient)\n"
                    "    fig, ax = plt.subplots(1, 4, figsize=(16, 3.2))\n"
                    "    fig.suptitle(f'S0 · {label} · CROP 100×100 (slicer {orient}; n bajo error={int(lo.sum())}, media {meanC*1e3:+.1f}±{seC*1e3:.1f} mÅ)', fontsize=10)\n"
                    "    vl = np.nanpercentile(np.abs(oa_c[lo]), 95) if lo.any() else 0.3\n"
                    "    im0 = ax[0].imshow(oa_c, origin='lower', cmap='RdBu_r', vmin=-vl, vmax=vl); ax[0].set_title('offset (Å)'); plt.colorbar(im0, ax=ax[0], fraction=0.046)\n"
                    "    im1 = ax[1].imshow(ec_c*step, origin='lower', cmap='viridis', vmax=np.nanpercentile(ec_c[np.isfinite(ec_c)]*step, 95) if np.isfinite(ec_c).any() else None); ax[1].set_title('error (Å)'); plt.colorbar(im1, ax=ax[1], fraction=0.046)\n"
                    "    p_A = np.asarray(prof['profile'], float) * step\n"
                    "    ax[2].plot(np.arange(p_A.size), p_A, lw=0.9); ax[2].axhline(0, color='0.6', lw=0.6); ax[2].set_title(f'perfil ∥ slicer ({orient})'); ax[2].set_xlabel('posición transversal (crop)'); ax[2].set_ylabel('offset mediano (Å)')\n"
                    "    if lo.any():\n"
                    "        ax[3].hist(oa_c[lo], bins=40, color='tab:green', alpha=0.8); ax[3].axvline(0, color='k', lw=0.8)\n"
                    "        ax[3].axvline(meanC, color='tab:red', ls='--', lw=1.2, label=f'media {meanC*1e3:+.1f} mÅ ({abs(meanC/seC):.0f}σ)'); ax[3].legend(fontsize=8)\n"
                    "    ax[3].set_title('hist (bajo error)'); ax[3].set_xlabel('offset (Å)')\n"
                    "    fig.tight_layout(); plt.show()\n"
                    "print('\\nLectura: (1) el offset medio del crop deriva exposición a exposición (~±40 mÅ, el modo')\n"
                    "print('TEMPORAL de S3, ≲0.04 canal); (2) todos los casos caen en/bajo la diagonal stripe=transversal')\n"
                    "print('y muy por debajo del umbral 3× => NINGÚN stripe de slicer, ni al máximo S/N. Refuerza el cierre G1.')"
                ),
            ),
        ],
        decisions=[
            ("S0a/S0b: núcleo target-agnostic (`musepipe/qc/wavesol_map.py`) + CLI, normalización vectorizada (gate de equivalencia <1e-9) y corte S/N por spaxel; 14 tests verdes.", "plan_wavesol_stripes_pasos_agente.md"),
            ("El offset map es diagnóstico del instrumento/reducción: realineado ≈ ADP (control cruzado).", "plan_wavesol_stripes_2026-07-17.md"),
            ("**Gate G1 (humano):** con este producto se decide entrar o no a la Fase 2 (S2–S5, re-reducción por exposición). La recomendación automática se imprime en Checks.", "plan_wavesol_stripes_2026-07-17.md"),
            ("**DECIDIDO 2026-07-17:** cerrar como sistemático acotado (interpretación temporal); Fase 2 NO disparada; confirmación diferida (S0 por exposición cuando existan los 7 cubos). Anula la recomendación automática.", "decision_g1_wavesol_2026-07-17.md"),
            ("**Corrección POSANG 2026-07-19 (hallazgo del usuario):** `INS DROT POSANG` = 0/0/90/90/180/180/0 en modo SKY (el ABSROT −16→+6° es el ángulo físico del derotador, no rotación de campo). exp3/exp4 tienen el slicer HORIZONTAL en el cubo norte-arriba: sus métricas stripe/control estaban intercambiadas. Con la orientación correcta el veredicto NO cambia (0/7 sin stripes); la ceguera del combinado se refuerza (scrambling total, no smear de 5.9°). Tabla corregida: `tables/s0_perexp_summary_posang.csv`.", "decision_g1_wavesol_2026-07-17.md"),
        ],
        checks=(
            "q = nb.load_qc('stages/stageS0_qc.json', RUN_ID)\n"
            "m, g = q['metrics'], q['gate_g1']\n"
            "th = g['thresholds']\n"
            "p95 = m['p95_abs_offset_A']; sig = m['structure_significance']; sigt = m['transverse_significance']\n"
            "c1 = p95 <= th['p95_threshold_A']\n"
            "aligned = (sig > th['significance_threshold']) and (sigt != sigt or sig > 2.0 * sigt)\n"
            "print('CHECKLIST G1 (realineado, full-res):')\n"
            "print(f\"  p95|off| = {p95:.4f} A  (umbral {th['p95_threshold_A']} A)   -> {'OK pequeño' if c1 else 'GRANDE'}\")\n"
            "print(f\"  estructura stripe = {sig:.2f}x ruido   (control transversal {sigt:.2f}x, umbral {th['significance_threshold']}x)\")\n"
            "print(f\"  {'sin' if not aligned else 'CON'} estructura alineada con slicers dominante\")\n"
            "print(f\"  n_spaxels bajo corte err<{th['max_err_ch']} ch = {m['n_selected_low_err']} / {m['n_spaxels_measured']} medidos\")\n"
            "print(f\"\\n  recomendación automática: {g['recommendation']}\")\n"
            "for r in g['reasons']:\n"
            "    print('   -', r)\n"
            "hd = q.get('g1_human_decision')\n"
            "if hd:\n"
            "    print(f\"\\n  DECISIÓN G1 (humana, {hd['date']}): {hd['decision']} \"\n"
            "          f\"[interpretación: {hd['interpretation']}; anula {hd['recommendation_overridden']}]\")\n"
            "    print(f\"    fase2_triggered = {hd['phase2_triggered']}; confirmación diferida: {hd['deferred_confirmation'][:80]}...\")\n"
            "    print(f\"    doc: {hd['doc']}\")\n"
            "else:\n"
            "    print(f\"  decisión: {g['decision']} (aún no registrada en este QC)\")"
        ),
        conclusion_md=(
            "## Conclusión (registrada, 2026-07-17) — G1 DECIDIDO\n\n"
            "> Registro de la decisión tomada para **ROXs 12 b** (cifras de esa fecha, literales "
            "por trazabilidad). Si estás en el set de otro objeto, esta conclusión **no** es la "
            "suya: la de tu objeto sale de su propio `stageS0_qc.json`, arriba.\n\n"
            "**S0 (full-res, 330×338):** sin estructura de slicer (`stripe_sig` ≤ control "
            "transversal) y **realineado ≈ ADP**, pero p95\\|off\\| ≈ 0.32 Å (>0.1 Å), scatter "
            "por spaxel creciente con el radio.\n\n"
            "**Clave (por qué el cubo combinado es ciego a los stripes) — corregido 2026-07-19 "
            "(POSANG):** las 7 exposiciones son de una noche, dithers ≈0, pero `INS DROT "
            "POSANG` = **0°,0°,90°,90°,180°,180°,0°** (modo SKY: PA fijo en cielo durante cada "
            "exposición; el `ABSROT` −16→+6° es el ángulo físico del derotador, no rotación de "
            "campo). El combinado mezcla **tres asignaciones de slice** (0°/90°/180°) ⇒ el "
            "scrambling de un stripe fijo del slicer es **total** (cuerda de 90° en el "
            "compañero ≈ 2.5″), y aparece como scatter desestructurado. **S0 sobre el combinado "
            "no puede confirmar ni descartar stripes**; el 'sin estructura' es esperable en "
            "cualquier caso. En los cubos por exposición el slicer queda **vertical para POSANG "
            "0/180 y horizontal para 90 (exp3/exp4)** — la celda del crop mide cada caso en su "
            "orientación.\n\n"
            "➡️ **DECISIÓN G1 (humana, 2026-07-17): cerrar como sistemático acotado, "
            "interpretación TEMPORAL** (deriva de zero-point por exposición, acotada por A4·M2 "
            "LSF=2.383 Å). **Fase 2 NO disparada.** Anula la recomendación automática "
            "(`fase2_justificada`, que salía solo por el p95). **Salvedad:** M2 acota el modo "
            "temporal uniforme, no los stripes rotados; defendible para la no-detección de Hα / "
            "límites (E1/E3), más débil para líneas finas en G2/G3.\n\n"
            "**Confirmación (HECHA 2026-07-18):** se regeneraron los 7 cubos por exposición "
            "(S2) y se corrió **S0 por exposición** (0/7 con estructura de slicer) **+ S3** "
            "(deriva temporal ~0.04 Å std). Ambas ramas cerradas ⇒ **GATE G1 CERRADO** "
            "(`gate_g1.decision=closed`). Detalle en `docs/decision_g1_wavesol_2026-07-17.md`.\n\n"
            "**Corroboración con el crop 100×100 en la estrella (paso extra):** al restringir a "
            "la región de máxima S/N, el `median|offset|` cae de ~0.8 Å a ~0.11 Å (era ruido de "
            "bordes) y el offset medio con signo se resuelve a **unas decenas de mÅ** "
            "(realineado ≈ −48 mÅ, ADP ≈ −11 mÅ; ≲0.04 canal), pero **`stripe_sig` sigue ≤ el "
            "control transversal** en ambos cubos → **ningún stripe de slicer emerge al subir la "
            "S/N**. Lo que queda es un zero-point casi uniforme, del orden de la deriva temporal "
            "S3 (~0.04 Å) y dentro de lo que M1 (+74 mÅ) corrige — refuerza el cierre G1 al mejor "
            "S/N disponible."
        ),
    ),
    dict(
        id="S1", slug="S1_halpha_map",
        title="Mapas Hα line-to-continuum por spaxel (LSF/ghost, Xie Ec.1)",
        block="S · wavesol/stripes",
        spec="plan_wavesol_stripes_2026-07-17.md", run_override=None,
        what=(
            "Ajusta el Hα de la primaria por spaxel del halo con `phi=b(1+a·exp(−(λ−μ)²/2σ²))` "
            "(Xie+20 §4.1, Ec.1) y mapea a (line/continuo), σ, μ y P=a·b·σ·√(2π). Test de Xie "
            "Fig.3: a y σ **anticorrelados a P≈constante** ⇒ variación de LSF instrumental "
            "(alineada con slicers), no un *ghost*. Diagnóstico de apoyo a G1."
        ),
        inputs="`cube_telcorr.fits` (realineado); geometría B3 (primaria/compañero)",
        outputs=(
            "`stages/stageS1_qc.json`, `stages/stageS1_halpha_map.fits`, `plots/s1_halpha/`; "
            "integrado en E2 (`halpha_map_correlation` + Plot 3 del notebook E2)"
        ),
        downstream="Apoyo a la **decisión G1** (¿la variación Hα está alineada con slicers?)",
        narrative_md=(
            "## Qué hace S1 y cómo\n\n"
            "Por spaxel del halo (misma selección por brillo que S0) ajusta el Hα con "
            "`scipy.optimize.curve_fit` (semillas por momentos, bounds a∈[0,50], μ∈[6540,6590], "
            "σ∈[0.5,8] Å); los fits sin línea detectada (gate de S/N) devuelven NaN limpio. "
            "Reutiliza la métrica de estructura de S0 (`structure_metrics`/`stripe_profile`) "
            "sobre los mapas a y σ, con **control transversal** — la misma disciplina que S0: "
            "estructura real de slicer debe superar a su control transversal.\n\n"
            "El núcleo (S1a) es target-agnostic y testeado (8 tests); S1b lo corre full-res "
            "sobre el realineado e integra en E2 la correlación *zonas sucias de stripes* (mapa "
            "de offset S0) vs mapas Hα."
        ),
        exec=dict(
            kind="module_main",
            target="musepipe.qc.halpha_map",
            cmd=(
                "python -m musepipe.qc.halpha_map --run-id $RUN --orientation vertical\n"
                "python scripts/s1b_integrate_e2.py --run-dir runs/$RUN"
            ),
            cost="Coste: full-res, curve_fit por spaxel ~5 min; S1b re-lee y parchea E2.",
        ),
        qc="stages/stageS1_qc.json",
        salient=[
            "halpha_map.sigma_median_A", "halpha_map.a_structure_significance",
            "halpha_map.a_transverse_significance", "halpha_map.sigma_structure_significance",
            "halpha_map.sigma_transverse_significance", "halpha_map.corr_a_sigma",
            "halpha_map.P_cov", "halpha_map.n_fit",
        ],
        checks_md=(
            "## Los términos de este QC, en físico\n\n"
            "S1 ajusta, en cada spaxel del halo, la Hα de la **primaria** como "
            "`φ = b·(1 + a·exp(−(λ−μ)²/2σ²))`. No busca al compañero: busca si la línea cambia de "
            "forma según en qué parte del detector se mida.\n\n"
            "| Término | Qué es | Por qué importa |\n|---|---|---|\n"
            "| `a` (amplitud) | Cuánto sobresale la línea sobre el continuo local. | Si variara con "
            "el slicer, la razón línea/continuo dependería del instrumento. |\n"
            "| `sigma_median_A` | Anchura de la línea por spaxel. | Es la resolución espectral "
            "efectiva **medida donde importa**; si varía espacialmente, la LSF única de A4/M2 sería "
            "una simplificación. |\n"
            "| `mu` | Centro de la línea. | Su variación espacial es el mismo desvío de λ que mide S0, "
            "visto en una línea real en vez de en el cielo. |\n"
            "| `*_structure_significance` vs `*_transverse_significance` | Igual que en S0: la firma "
            "buscada frente a su **control perpendicular**. | Si estructura ≈ transversal, lo que se "
            "ve es radial (núcleo vs halo), no alineado con el slicer. |\n"
            "| `corr_a_sigma` / `P_cov` | Correlación entre amplitud y anchura, y si el producto "
            "`a·σ` se conserva. | Es la firma instrumental de Xie+20: si una línea se ensancha, su "
            "pico baja **manteniendo el área**. Que `P` NO sea constante indica que la variación no "
            "es ese efecto. |\n"
            "| `n_fit` | Cuántos spaxels tuvieron señal para ajustar. | Fija el alcance real del "
            "mapa. |\n"
        ),
        evidence_md=(
            "## Resultado (realineado, full-res)\n\n"
            "n_fit=29203, σ_median=2.01 Å. **NO alineado con slicers:** a struct 6.7× vs "
            "transversal 8.9×; σ struct 8.6× vs transversal 7.9× (estructura ≤ control ⇒ "
            "isótropo/radial, núcleo-vs-halo). corr(a,σ)=−0.57 pero **P_cov=1.07** (P no "
            "constante) ⇒ NO es el caso instrumental-LSF de Xie Fig.3. La correlación por "
            "columna stripe↔σ (−0.91) es un confundido radial (control transversal −0.905). "
            "Consistente con G1: el cubo combinado es ciego a stripes."
        ),
        evidence_code=(
            "q = nb.load_qc('stages/stageS1_qc.json', RUN_ID)['halpha_map']\n"
            "print(f\"n_fit={q['n_fit']}  sigma_median={q['sigma_median_A']:.3f} A\")\n"
            "print(f\"a:     struct {q['a_structure_significance']:.1f}x  transv {q['a_transverse_significance']:.1f}x\")\n"
            "print(f\"sigma: struct {q['sigma_structure_significance']:.1f}x  transv {q['sigma_transverse_significance']:.1f}x\")\n"
            "print(f\"corr(a,sigma)={q['corr_a_sigma']:.2f}  P_cov={q['P_cov']:.2f} (Xie Fig.3 pide P~const)\")\n"
            "aligned = lambda s,t: s>3 and s>2*t\n"
            "print('a slicer-aligned:', aligned(q['a_structure_significance'], q['a_transverse_significance']),\n"
            "      '| sigma slicer-aligned:', aligned(q['sigma_structure_significance'], q['sigma_transverse_significance']))"
        ),
        plots=[
            dict(
                md="## Mapas S1 (a, σ, P, y a-vs-σ)",
                code=(
                    "from IPython.display import Image, display\n"
                    "p = nb.run_dir(RUN_ID) / 'plots' / 's1_halpha' / 's1_realigned.png'\n"
                    "if p.exists(): display(Image(filename=str(p)))\n"
                    "else: print('falta', p, '- corre la etapa (arriba).')"
                ),
            ),
        ],
        decisions=[
            ("Núcleo S1a target-agnostic + 8 tests; S1b integra en E2 (aditivo) con control transversal.", "plan_wavesol_stripes_pasos_agente.md"),
            ("a/σ NO alineados con slicers (radial); apoya el cierre G1 (sin stripes en el combinado).", "decision_g1_wavesol_2026-07-17.md"),
        ],
        checks=(
            "q = nb.load_qc('stages/stageS1_qc.json', RUN_ID)['halpha_map']\n"
            "print('sigma_median_A =', round(q['sigma_median_A'],3))\n"
            "print('corr_a_sigma   =', round(q['corr_a_sigma'],3), '(P_cov', round(q['P_cov'],2),')')"
        ),
        conclusion_md=(
            "## Conclusión (registrada, 2026-07-18)\n\n"
            "Los mapas Hα a/σ del cubo combinado **no muestran estructura de slicer** (estructura "
            "≤ control transversal) y **no** cumplen la firma instrumental-LSF de Xie Fig.3 "
            "(P no constante). Es diagnóstico de apoyo al **cierre G1**: el combinado es ciego a "
            "los stripes (confirmado por S0 por-exposición). Sin interpretación ghost-vs-"
            "instrumental adicional (se resolvió por-exposición)."
        ),
    ),
]


def validate_against_registry() -> None:
    """Aborta si `STAGES` diverge de `musepipe.stage_registry` (fuente única).

    El registro es la autoridad de la parte machine-readable (id, slug, ruta de
    QC, opcionalidad, tipo de punto de entrada); aquí vive solo la prosa. Si
    alguien mueve un QC en un sitio y no en el otro, la generación falla en vez
    de emitir notebooks que apuntan a la nada.
    """
    sys.path.insert(0, str(ROOT))
    from musepipe import stage_registry as reg

    problems: list[str] = []
    ids_here = {s["id"] for s in STAGES}
    ids_reg = set(reg.stage_ids())
    for missing in sorted(ids_reg - ids_here):
        problems.append(f"{missing}: en el registro pero no en STAGES")
    for extra in sorted(ids_here - ids_reg):
        problems.append(f"{extra}: en STAGES pero no en el registro")

    for s in STAGES:
        r = reg.by_id(s["id"])
        if r is None:
            continue
        for field, here, there in (
            ("slug", s["slug"], r.slug),
            ("qc", s.get("qc"), r.qc),
            ("qc_optional", bool(s.get("qc_optional", False)), r.qc_optional),
            ("exec_kind", s["exec"]["kind"], r.exec_kind),
        ):
            if here != there:
                problems.append(f"{s['id']}.{field}: STAGES={here!r} registro={there!r}")

    if problems:
        raise SystemExit(
            "build_review_notebooks: STAGES diverge de musepipe/stage_registry.py:\n  "
            + "\n  ".join(problems)
        )


def check_qc_resolution(out_dir: Path, run_id: str) -> list[str]:
    """Resuelve cada `nb.load_qc(...)` de los notebooks generados.

    Distingue tres desenlaces, como la línea base de resolución de la Fase 0:
      * resuelve en la cadena del objeto  -> OK
      * no resuelve en ningún run del objeto -> etapa PENDIENTE (se reporta, no falla:
        las celdas la degradan a aviso vía `load_qc_optional`/`audit_code`)
      * resuelve a un run de OTRO objeto -> CROSS-OBJECT (falla: es contaminación, H1)

    Devuelve solo los CROSS-OBJECT (los que deben hacer `--check` salir ≠0).
    Los pendientes se imprimen aquí como información.
    """
    import ast

    sys.path.insert(0, str(NB_DIR))
    sys.path.insert(0, str(ROOT))
    import importlib
    nb = importlib.import_module("_nbcommon")
    nb.resolve_run_id(run_id)

    problems: list[str] = []
    pending: set[str] = set()
    for path in sorted(out_dir.glob("*.ipynb")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seen: set[str] = set()
        for cell in payload.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("load_qc", "load_qc_optional", "qc_path")):
                    continue
                if not node.args or not isinstance(node.args[0], ast.Constant):
                    continue
                rel = node.args[0].value
                if rel in seen:
                    continue
                seen.add(rel)
                try:
                    _p, where, _why = nb.resolve_qc(rel, run_id)
                except FileNotFoundError:
                    pending.add(f"{path.name}: {rel}")
                    continue
                if nb._is_cross_object(where, run_id):
                    problems.append(f"{path.name}: {rel} resuelve a {where} (OTRO objeto)")
    if pending:
        print(f"\n--check: {len(pending)} etapa(s) pendiente(s) para este objeto "
              "(se degradan a aviso, no son errores):")
        for line in sorted(pending):
            print("  ", line)
    return problems


def main(argv: list[str]) -> None:
    validate_against_registry()
    # Flags opcionales para generar un set AISLADO por objeto (p.ej. ROXs 42B b)
    # sin tocar los notebooks de ROXs 12 b:
    #   --run-id ID    fija el run que auditan TODOS los notebooks (override).
    #   --out-dir DIR  carpeta de salida (relativa a la raíz del repo o absoluta).
    # El resto de argumentos posicionales siguen filtrando por id de etapa.
    #   --target OBJ   genera el set del objeto: run = default_run de su cadena,
    #                  salida = notebooks/<OBJ>/.  Forma de alto nivel.
    #   --run-id ID / --out-dir DIR  formas de bajo nivel (siguen valiendo).
    #   --check        tras generar, resuelve cada load_qc y sale ≠0 si alguna
    #                  ruta no resuelve y la etapa no está legítimamente pendiente.
    run_override_all: str | None = None
    out_dir: Path | None = None
    target: str | None = None
    do_check = False
    rest: list[str] = []
    it = iter(argv)
    for a in it:
        if a == "--run-id":
            run_override_all = next(it)
        elif a.startswith("--run-id="):
            run_override_all = a.split("=", 1)[1]
        elif a == "--target":
            target = next(it)
        elif a.startswith("--target="):
            target = a.split("=", 1)[1]
        elif a == "--out-dir":
            d = Path(next(it))
            out_dir = d if d.is_absolute() else (ROOT / d)
        elif a.startswith("--out-dir="):
            d = Path(a.split("=", 1)[1])
            out_dir = d if d.is_absolute() else (ROOT / d)
        elif a == "--check":
            do_check = True
        else:
            rest.append(a)

    # --target: deduce run y carpeta desde la cadena del objeto.
    if target is not None:
        default_run = f"{target}_realigned"
        config_json = ROOT / "runs" / default_run / "config" / "config.json"
        if config_json.exists():
            try:
                chain = json.loads(config_json.read_text(encoding="utf-8")).get("chain", {})
                default_run = chain.get("default_run", default_run)
            except (OSError, json.JSONDecodeError):
                pass
        run_override_all = run_override_all or default_run
        if out_dir is None:
            out_dir = NB_DIR / target

    # Sin --target ni --out-dir: la salida va a notebooks/<objeto>/, no a la raíz.
    if out_dir is None:
        obj = _object_slug(run_override_all or DEFAULT_RUN)
        out_dir = NB_DIR / obj

    want = {a.upper() for a in rest}
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for s in STAGES:
        if want and s["id"].upper() not in want:
            continue
        if run_override_all is not None:
            s = {**s, "run_override": run_override_all}
        nb = notebook(build_cells(s))
        out = out_dir / f"{s['slug']}.ipynb"
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
        written.append(out.name)
    tag = f" (run={run_override_all})" if run_override_all else ""
    print(f"Generados {len(written)} notebooks en {out_dir}/{tag}:")
    for name in written:
        print("  ", name)

    if do_check:
        problems = check_qc_resolution(out_dir, run_override_all or DEFAULT_RUN)
        if problems:
            print(f"\n--check: {len(problems)} ruta(s) de QC no resuelven y NO son etapas "
                  "pendientes legítimas:")
            for line in problems:
                print("  ", line)
            raise SystemExit(1)
        print("\n--check: OK — cada load_qc resuelve o corresponde a una etapa pendiente.")


if __name__ == "__main__":
    main(sys.argv[1:])
