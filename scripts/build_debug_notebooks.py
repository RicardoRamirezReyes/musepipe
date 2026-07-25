#!/usr/bin/env python3
"""Constructor OPCIONAL de notebooks de análisis («debug») por método.

Los notebooks de `notebooks/<Objeto>/` auditan la cadena: llaman a `musepipe` y
enseñan lo que la etapa decidió. Estos son lo contrario — hacen **el proceso
dentro del notebook**, con el código a la vista y editable, para poder probar,
cambiar y ajustar **sin tocar la cadena general**.

De ahí la decisión incómoda de este generador: **copia el fuente de las
funciones numéricas** dentro del notebook en vez de importarlas. Duplicar código
es normalmente una mala idea, así que la copia se defiende con dos cosas que van
en cada notebook generado:

1. una **celda de deriva** que compara el fuente copiado con el que hoy tiene
   `musepipe` y avisa, nombrando la función, si la cadena cambió y este notebook
   se quedó atrás;
2. una **celda de comparación** contra el producto real de la etapa: con las
   perillas por defecto debe salir idéntico, y en cuanto se toca algo dice qué
   se movió y dónde.

El corte de qué se copia: **solo lo numérico de la etapa**. El modelo de PSF
(`evaluate_psf_model`) se importa de `musepipe`, porque es de C1 y no es lo que
se ajusta aquí; el ensamblado del `SpectrumProduct` (cabeceras, provenance) va
como código plano del notebook, más legible que las 164 líneas del original.

Salida: `notebooks/<Objeto>/debug/`, una subcarpeta a propósito — `--check` del
constructor principal y `tests/test_notebook_qc_resolution.py` recorren
`notebooks/<obj>/*.ipynb` sin recursión, así que estos no interfieren con la
validación de la cadena.

Uso:

    python scripts/build_debug_notebooks.py --target ROXs12b
    python scripts/build_debug_notebooks.py --target ROXs42Bb C2
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


def _load_main_builder():
    """El constructor principal, para reutilizar sus celdas y su metadata."""
    spec = importlib.util.spec_from_file_location(
        "build_review_notebooks", ROOT / "scripts" / "build_review_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("build_review_notebooks", module)
    spec.loader.exec_module(module)
    return module


#: Funciones que se copian al notebook, por etapa, en orden de dependencia.
#: Solo lo NUMÉRICO de la etapa: lo que un usuario querría tocar para probar.
INLINE_SOURCES = {
    "C2": [
        ("musepipe/stats.py", ["finite_values", "robust_sigma", "robust_sigma_axis0"]),
        ("musepipe/apertures.py", [
            "angular_separation_deg", "aperture_weights", "same_radius_control_positions",
        ]),
        ("musepipe/extraction/aperture.py", [
            "_as_cube", "_npix_eff", "aperture_spectrum", "annulus_background_spectrum",
            "aperture_stat_error", "control_aperture_spectra", "_flag_window", "channel_flags",
            "aperture_correction_from_psf",
        ]),
    ],
}


def extract_sources(stage_id):
    """`[(modulo, nombre, fuente, sha12)]` de las funciones a copiar.

    Se leen con `ast` sobre el fichero, no con `inspect`: así este generador
    sigue siendo stdlib pura y se puede correr sin el entorno científico.
    """
    out = []
    for rel, names in INLINE_SOURCES[stage_id]:
        text = (ROOT / rel).read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        tree = ast.parse(text)
        found = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in names:
            node = found.get(name)
            if node is None:
                raise KeyError(f"{rel}: no se encuentra la función {name!r}")
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            # El sha se calcula sobre el MISMO texto que se copia (ya recortado):
            # si se calculara sobre el texto con su salto final, el chequeo de
            # deriva del notebook daría falsa alarma nada más generarlo.
            src = "".join(lines[start:node.end_lineno]).rstrip("\n")
            sha = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
            out.append((rel, name, src, sha))
    return out


def needed_imports(sources):
    """Los `import` de módulo que la copia necesita, deducidos del propio código.

    Se detectan en vez de escribirse a mano: cuando este generador cubra C3–C6,
    una función copiada que use `warnings` o `math` traerá su import sola en vez
    de fallar al ejecutar el notebook.
    """
    used = set()
    for rel, _name, src, _sha in sources:
        tree = ast.parse(src)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.value.id for n in ast.walk(tree)
                  if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
        module = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in module.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    if local in names:
                        used.add(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module != "__future__":
                for alias in node.names:
                    local = alias.asname or alias.name
                    if local in names:
                        used.add(f"from {node.module} import {alias.name}")
    return sorted(used)


def needed_constants(sources):
    """Constantes de módulo que usa la copia (`FLAG_BAD_WINDOW`, umbrales…).

    Misma idea que `needed_imports`: se deducen del código copiado. Son parte
    del contrato de la etapa (los bits de `flags`, por ejemplo), así que viajan
    con la copia en vez de importarse a escondidas.
    """
    out, seen = [], set()
    for rel, _name, src, _sha in sources:
        names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
        text = (ROOT / rel).read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        for node in ast.parse(text).body:
            targets = []
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            if not targets or not (set(targets) & names):
                continue
            key = (rel, tuple(targets))
            if key in seen:
                continue
            seen.add(key)
            out.append("".join(lines[node.lineno - 1:node.end_lineno]).rstrip("\n"))
    return out


def _knobs_from_config(run_id):
    """Los valores que la cadena usa en ESTE run, para escribirlos como literales."""
    path = ROOT / "runs" / run_id / "config" / "config.json"
    cfg = {}
    if path.exists():
        try:
            cfg = json.loads(path.read_text(encoding="utf-8")).get("config", {})
        except (OSError, ValueError):
            cfg = {}
    return {
        "x01_aperture_correction": cfg.get("x01_aperture_correction", "auto"),
        "x01_annulus_bkg_px": cfg.get("x01_annulus_bkg_px", [8.0, 14.0, 30.0]),
        "x01_control_apertures": int(cfg.get("x01_control_apertures", 8)),
        "x01_control_exclude_angle_deg": float(cfg.get("x01_control_exclude_angle_deg", 25.0)),
        "x01_error_mode": cfg.get("x01_error_mode", "auto"),
        "x01_wings_intact_apcorr": bool(cfg.get("x01_wings_intact_apcorr", True)),
        "x01_bad_windows_A": cfg.get("x01_bad_windows_A", []),
        "x01_skyline_windows_A": cfg.get("x01_skyline_windows_A", []),
        "x01_interpolated_windows_A": cfg.get("x01_interpolated_windows_A", []),
    }


def build_c2_cells(mb, target, run_id):
    """Las celdas de `C2_aperture_debug` para un objeto."""
    md, code = mb.md, mb.code
    sources = extract_sources("C2")
    knobs = _knobs_from_config(run_id)
    inline_src = "\n\n\n".join(src for _rel, _name, src, _sha in sources)
    shas = {f"{rel}:{name}": sha for rel, name, _src, sha in sources}

    cells = [
        md(
            f"# C2 · apertura — notebook de análisis (`debug`)\n\n"
            f"**Objeto:** {target}  |  **Run:** `{run_id}`  |  "
            f"**Spec:** [`docs/spec_C2_codex_aperture_extraction.md`]"
            f"(../../../docs/spec_C2_codex_aperture_extraction.md)\n\n"
            "Este notebook **no llama a la cadena**: rehace la extracción por apertura aquí "
            "dentro, con el código a la vista, para que puedas **probar, cambiar y ajustar sin "
            "tocar `musepipe`**. El notebook de auditoría equivalente es "
            f"[`../C2_aperture.ipynb`](../C2_aperture.ipynb), que sí llama a la etapa.\n\n"
            "Cómo está montado, y por qué:\n\n"
            "1. **Perillas** arriba del todo, con el valor que usa la cadena para este run.\n"
            "2. **Las funciones numéricas, copiadas literalmente** de `musepipe`. Se copian "
            "(en vez de importarse) para que puedas editarlas: todo lo que viene después usa "
            "estos nombres locales.\n"
            "3. **Chequeo de deriva** — avisa si `musepipe` cambió y esta copia se quedó atrás.\n"
            "4. El proceso **paso a paso**, cada uno con su diagnóstico.\n"
            "5. **Comparación con el producto de la cadena**: con las perillas por defecto debe "
            "salir *idéntico*; en cuanto cambias algo, te dice qué se movió y dónde.\n\n"
            "> Lo que NO se copia: `evaluate_psf_model` (es de C1, no es lo que se ajusta aquí) "
            "y el ensamblado del `SpectrumProduct`, que va como código plano más abajo."
        ),
        code(
            "import json, sys\n"
            "from pathlib import Path\n\n"
            "import numpy as np\n"
            "from astropy.io import fits\n"
            "import matplotlib.pyplot as plt\n\n"
            "_here = Path.cwd()\n"
            "ROOT = next(p for p in (_here, *_here.parents) if (p / 'musepipe').is_dir())\n"
            "sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'notebooks'))\n"
            "import _nbcommon as nb\n\n"
            f"RUN_ID = nb.resolve_run_id({run_id!r})\n"
            "RD = nb.run_dir(RUN_ID); SD = RD / 'stages'\n"
            "CFG = json.loads((RD / 'config' / 'config.json').read_text(encoding='utf-8'))['config']\n"
            "# El objeto se DERIVA del run (cadena declarada en su config), no se\n"
            "# escribe: un literal aquí haría que un objeto nuevo heredase el nombre\n"
            "# del primero, que es lo que vigila tests/test_no_hardcoded_target.py.\n"
            "TARGET = nb.run_target(RUN_ID) or nb.display_name(RUN_ID)\n"
            "print('objeto :', TARGET, '·', nb.display_name(RUN_ID))\n"
            "print('run    :', RUN_ID)\n"
            "print('stages :', SD)"
        ),
        md(
            "## 1 · Perillas\n\n"
            "Los valores son los que **la cadena usa en este run** (leídos de su `config.json` al "
            "generar el notebook). Cambia cualquiera y vuelve a ejecutar desde aquí: la celda de "
            "comparación del final te dirá exactamente qué efecto tuvo."
        ),
        code(
            "APERTURE            = {'kind': 'box', 'size': 3}   # la caja que se compara con la cadena\n"
            f"APCORR_MODE         = {knobs['x01_aperture_correction']!r}   # cfg x01_aperture_correction\n"
            f"WINGS_INTACT        = {knobs['x01_wings_intact_apcorr']!r}   # cfg x01_wings_intact_apcorr\n"
            f"ANNULUS_BKG_PX      = {knobs['x01_annulus_bkg_px']!r}   # cfg x01_annulus_bkg_px\n"
            f"N_CONTROLS          = {knobs['x01_control_apertures']!r}   # cfg x01_control_apertures\n"
            f"EXCLUDE_ANGLE_DEG   = {knobs['x01_control_exclude_angle_deg']!r}   # cfg x01_control_exclude_angle_deg\n"
            f"ERROR_MODE          = {knobs['x01_error_mode']!r}   # cfg x01_error_mode\n"
            f"BAD_WINDOWS_A       = {knobs['x01_bad_windows_A']!r}\n"
            f"SKYLINE_WINDOWS_A   = {knobs['x01_skyline_windows_A']!r}\n"
            f"INTERPOLATED_WIN_A  = {knobs['x01_interpolated_windows_A']!r}\n"
            "print('perillas listas; APERTURE =', APERTURE)"
        ),
        md(
            "## 2 · Entradas\n\n"
            "Las mismas que toma C2, y **de dónde sale cada una**. Ojo al cubo: cuando la "
            "corrección de apertura está activa, C2 **no** extrae del residual de 04b sino del "
            "cubo crudo de B2 con un fondo de anillo (*wings-intact*), porque el residual de 04b "
            "se come las alas del compañero y rompería la consistencia box3/box5 de la curva de "
            "crecimiento. Esa decisión se replica aquí."
        ),
        code(
            "qc_b3 = json.loads((SD / 'stage01c_qc.json').read_text(encoding='utf-8'))\n"
            "OBJECT_YX = tuple(float(v) for v in qc_b3['companion']['pos_yx'])\n"
            "STAR_YX   = tuple(float(v) for v in qc_b3['primary']['pos_yx'])\n\n"
            "psf_path = SD / 'psf_model.json'\n"
            "PSF_MODEL = json.loads(psf_path.read_text(encoding='utf-8')) if psf_path.exists() else None\n\n"
            "# ¿wings-intact? Misma condición que la etapa.\n"
            "use_raw = (PSF_MODEL is not None and str(APCORR_MODE).lower() in ('auto', 'psf_growth_curve')\n"
            "           and WINGS_INTACT and (SD / 'stage02_xcorr_cube_stack.fits').exists())\n"
            "CUBE_PATH = SD / ('stage02_xcorr_cube_stack.fits' if use_raw else 'cube_residual_local_object.fits')\n"
            "ANNULUS = list(ANNULUS_BKG_PX) if use_raw else None\n\n"
            "with fits.open(CUBE_PATH) as h:\n"
            "    if 'CUBES' in h:\n"
            "        CUBE = np.asarray(h['CUBES'].data, dtype=float)\n"
            "        WAVE = np.asarray(h['WAVELENGTH'].data, dtype=float)\n"
            "    else:\n"
            "        CUBE = np.asarray(h[0].data, dtype=float)\n"
            "        WAVE = np.asarray(fits.getdata(SD / 'stage02_xcorr_cube_stack.fits', 'WAVELENGTH'), dtype=float)\n"
            "if CUBE.ndim == 4:\n"
            "    CUBE = CUBE[0]\n\n"
            "# STAT y sus factores (A4/M5 + B1): el STAT crudo subestima el ruido de apertura.\n"
            "qc00 = json.loads((SD / 'stage00q_qc.json').read_text(encoding='utf-8'))\n"
            "qc01 = json.loads((SD / 'stage01_qc.json').read_text(encoding='utf-8'))\n"
            "m5 = qc00.get('m5_stat', {})\n"
            "STAT_FACTOR = float(CFG.get('x01_stat_factor_box3', m5.get('factor_box3_median', 1.0)) or 1.0)\n"
            "COV_FACTOR  = float(CFG.get('x01_covariance_factor_box3',\n"
            "                            qc01.get('stat', {}).get('covariance_factor_box3', 1.0)) or 1.0)\n"
            "STAT_STATUS = str(CFG.get('x01_stat_status', m5.get('status', 'unknown')))\n"
            "stat_src = Path(CFG.get('x01_stat_cube_fits') or (SD / 'stage02_xcorr_cube_stack.fits'))\n"
            "STAT_CUBE = None\n"
            "if stat_src.exists():\n"
            "    with fits.open(stat_src, memmap=True) as h:\n"
            "        if 'STAT' in h:\n"
            "            s = np.asarray(h['STAT'].data, dtype=float)\n"
            "            STAT_CUBE = s[0] if s.ndim == 4 else s\n"
            "    if STAT_CUBE is not None and STAT_CUBE.shape != CUBE.shape:\n"
            "        print('STAT descartado por forma:', STAT_CUBE.shape, '!=', CUBE.shape); STAT_CUBE = None\n\n"
            "print('cubo      :', CUBE_PATH.name, CUBE.shape, '| wings-intact:', use_raw)\n"
            "print('fondo     :', 'anillo ' + str(ANNULUS) if ANNULUS else 'ninguno (residual 04b)')\n"
            "print('compañero :', [round(v, 2) for v in OBJECT_YX], ' primaria:', [round(v, 2) for v in STAR_YX])\n"
            "print('PSF       :', (PSF_MODEL or {}).get('form', 'sin modelo'))\n"
            "print(f'STAT      : factor={STAT_FACTOR:.3f} covarianza={COV_FACTOR:.3f} estado={STAT_STATUS}')"
        ),
        md(
            "## 3 · Las funciones numéricas, copiadas de `musepipe`\n\n"
            "Copia **literal** del fuente, para que puedas editarla. Todo lo que viene después "
            "usa estos nombres locales, así que un cambio aquí se propaga al resultado — y la "
            "comparación del final lo cuantifica.\n\n"
            + "\n".join(f"- `{name}` — de `{rel}`" for rel, name, _s, _h in sources)
        ),
        code(
            "# ------------------------------------------------------------------\n"
            "# COPIA EDITABLE. Fuente: musepipe (ver el chequeo de deriva abajo).\n"
            "# ------------------------------------------------------------------\n"
            + "\n".join(needed_imports(sources)) + "\n"
            "from musepipe.psf import evaluate_psf_model   # de C1: no es lo que se ajusta aquí\n\n"
            + "\n".join(needed_constants(sources)) + "\n\n\n"
            + inline_src
        ),
        md(
            "## 4 · Chequeo de deriva\n\n"
            "Compara el fuente copiado arriba con el que **hoy** tiene `musepipe`. Si alguien "
            "cambió la cadena, esta celda lo dice nombrando la función: es lo que evita que este "
            "notebook siga dando resultados «de la cadena» cuando ya no lo son."
        ),
        code(
            "import ast as _ast, hashlib as _hashlib\n\n"
            f"_SHAS = {json.dumps(shas, indent=4)}\n\n"
            "def chequeo_de_deriva(shas=_SHAS, root=ROOT):\n"
            "    problemas = []\n"
            "    for key, sha in shas.items():\n"
            "        rel, name = key.rsplit(':', 1)\n"
            "        text = (root / rel).read_text(encoding='utf-8')\n"
            "        lines = text.splitlines(keepends=True)\n"
            "        node = next((n for n in _ast.parse(text).body\n"
            "                     if isinstance(n, _ast.FunctionDef) and n.name == name), None)\n"
            "        if node is None:\n"
            "            problemas.append(f'{key}: ya no existe en musepipe'); continue\n"
            "        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1\n"
            "        src = ''.join(lines[start:node.end_lineno]).rstrip('\\n')\n"
            "        actual = _hashlib.sha256(src.encode('utf-8')).hexdigest()[:12]\n"
            "        if actual != sha:\n"
            "            problemas.append(f'{key}: la copia es {sha}, musepipe tiene {actual}')\n"
            "    return problemas\n\n"
            "_deriva = chequeo_de_deriva()\n"
            "if _deriva:\n"
            "    print('DERIVA — la cadena cambió y esta copia se quedó atrás:')\n"
            "    for p in _deriva:\n"
            "        print('  ·', p)\n"
            "    print(f'\\nRegenera el notebook: python scripts/build_debug_notebooks.py'\n"
            "          f' --target {TARGET} C2')\n"
            "else:\n"
            "    print(f'sin deriva: las {len(_SHAS)} funciones copiadas son las de musepipe')"
        ),
        md(
            "## 5 · Paso 1 — la apertura\n\n"
            "Los pesos de la caja y el número **efectivo** de píxeles por canal: `npix_eff` no es "
            "9 fijo, baja donde hay NaN, y es lo que después convierte el fondo por píxel del "
            "anillo en fondo de la apertura."
        ),
        code(
            "weights = aperture_weights(CUBE.shape[1], CUBE.shape[2], OBJECT_YX, APERTURE)\n"
            "raw_flux, npix_eff = aperture_spectrum(CUBE, OBJECT_YX, APERTURE)\n"
            "print('píxeles con peso:', int((weights > 0).sum()), '| npix_eff mediano:', float(np.nanmedian(npix_eff)))\n\n"
            "yy, xx = np.nonzero(weights)\n"
            "y0, y1, x0, x1 = yy.min() - 4, yy.max() + 5, xx.min() - 4, xx.max() + 5\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))\n"
            "a1.imshow(np.nanmedian(CUBE[::20, y0:y1, x0:x1], axis=0), origin='lower', cmap='magma')\n"
            "a1.imshow(np.where(weights[y0:y1, x0:x1] > 0, 1.0, np.nan), origin='lower', cmap='cool', alpha=0.45)\n"
            "a1.set_title(f'la apertura sobre el dato ({APERTURE[\"kind\"]}{APERTURE[\"size\"]})', fontsize=9)\n"
            "a2.plot(WAVE, npix_eff, lw=0.8); a2.set_xlabel('λ [Å]'); a2.set_ylabel('npix_eff')\n"
            "a2.set_title('píxeles efectivos por canal', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 6 · Paso 2 — fondo de anillo\n\n"
            "Solo cuando la extracción es *wings-intact* (cubo crudo). El anillo se mide "
            "**excluyendo la primaria**, y se resta multiplicado por `npix_eff` para pasar de "
            "fondo por píxel a fondo de la apertura."
        ),
        code(
            "if ANNULUS is not None:\n"
            "    bkg = annulus_background_spectrum(CUBE, OBJECT_YX, ANNULUS[0], ANNULUS[1],\n"
            "                                      exclude_yx=STAR_YX,\n"
            "                                      exclude_radius=(ANNULUS[2] if len(ANNULUS) > 2 else 30.0))\n"
            "    raw_flux_bkgsub = raw_flux - bkg * npix_eff\n"
            "    print('fondo mediano por píxel:', round(float(np.nanmedian(bkg)), 3))\n"
            "    print('resta mediana a la apertura:', round(float(np.nanmedian(bkg * npix_eff)), 2))\n"
            "else:\n"
            "    bkg = None\n"
            "    raw_flux_bkgsub = raw_flux\n"
            "    print('sin fondo de anillo (se extrae del residual de 04b)')\n"
            "raw_flux = raw_flux_bkgsub"
        ),
        md(
            "## 7 · Paso 3 — controles y error empírico\n\n"
            "**La regla que gobierna todo el modelo de ruido**: los controles se procesan "
            "*exactamente igual* que el objeto — misma caja, mismo radio a la primaria, mismo "
            "fondo de anillo. σ es su dispersión, no el STAT. Ver "
            "[`docs/noise_model.md`](../../../docs/noise_model.md)."
        ),
        code(
            "controls_yx, control_spectra, control_npix = control_aperture_spectra(\n"
            "    CUBE, OBJECT_YX, STAR_YX, APERTURE,\n"
            "    n_controls=N_CONTROLS, exclude_angle_deg=EXCLUDE_ANGLE_DEG)\n"
            "control_bkgsub = control_spectra\n"
            "if ANNULUS is not None and control_spectra.shape[0]:\n"
            "    control_bkgsub = control_spectra.copy()\n"
            "    for k, yx in enumerate(controls_yx):\n"
            "        cb = annulus_background_spectrum(CUBE, yx, ANNULUS[0], ANNULUS[1],\n"
            "                                        exclude_yx=STAR_YX,\n"
            "                                        exclude_radius=(ANNULUS[2] if len(ANNULUS) > 2 else 30.0))\n"
            "        control_bkgsub[k] = control_spectra[k] - cb * control_npix[k]\n"
            "if control_bkgsub.shape[0] >= 2:\n"
            "    raw_err_emp = robust_sigma_axis0(control_bkgsub)\n"
            "else:\n"
            "    raw_err_emp = np.full(WAVE.size, robust_sigma(raw_flux), dtype=float)\n"
            "print(f'{len(controls_yx)} controles | σ empírico mediano = {float(np.nanmedian(raw_err_emp)):.2f}')\n\n"
            "fig, ax = plt.subplots(figsize=(11, 3.4))\n"
            "for c in control_bkgsub:\n"
            "    ax.plot(WAVE, c, lw=0.4, alpha=0.35, color='0.6')\n"
            "ax.plot(WAVE, raw_flux, lw=0.6, color='tab:blue', label='objeto')\n"
            "ax.plot(WAVE, raw_err_emp, lw=1.0, color='tab:red', label='σ empírico (dispersión de controles)')\n"
            "ax.set_xlabel('λ [Å]'); ax.legend(fontsize=8)\n"
            "ax.set_title('objeto vs controles procesados igual', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 8 · Paso 4 — error por STAT, corrección de apertura y flags\n\n"
            "El STAT del cubo **nunca** se usa crudo: lleva el factor de M5 y el de covarianza de "
            "B1 (el desplazamiento subpíxel correlacionó píxeles vecinos). Y `apcorr(λ)` sale de "
            "la curva de crecimiento de C1 — en NFM vale decenas, porque una caja 3×3 recoge una "
            "fracción minúscula de la PSF."
        ),
        code(
            "stat_usable = (STAT_CUBE is not None and str(ERROR_MODE).lower() != 'empirical'\n"
            "               and STAT_STATUS.lower() != 'red')\n"
            "if stat_usable:\n"
            "    raw_err = aperture_stat_error(STAT_CUBE, OBJECT_YX, APERTURE,\n"
            "                                  stat_factor=STAT_FACTOR, covariance_factor=COV_FACTOR)\n"
            "    error_mode = 'stat'\n"
            "else:\n"
            "    raw_err = np.asarray(raw_err_emp, dtype=float)\n"
            "    error_mode = 'empirical'\n\n"
            "apcorr, apcorr_mode, norm_radius = aperture_correction_from_psf(\n"
            "    WAVE, APERTURE, PSF_MODEL, center_yx=OBJECT_YX, correction_mode=APCORR_MODE)\n"
            "flags = channel_flags(WAVE, bad_windows_A=BAD_WINDOWS_A, skyline_windows_A=SKYLINE_WINDOWS_A,\n"
            "                      interpolated_windows_A=INTERPOLATED_WIN_A)\n"
            "print(f'error: modo={error_mode} | apcorr: modo={apcorr_mode} mediana={float(np.nanmedian(apcorr)):.1f}'\n"
            "      f' (norm_radius={norm_radius:g} px) | canales marcados: {int((flags != 0).sum())}')\n\n"
            "fig, ax = plt.subplots(figsize=(11, 3.2))\n"
            "ax.plot(WAVE, apcorr, lw=1.0)\n"
            "ax.set_xlabel('λ [Å]'); ax.set_ylabel('apcorr'); ax.set_title('corrección de apertura vs λ', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 9 · Paso 5 — el espectro\n\n"
            "El ensamblado va aquí como código plano (no copiado): flujo y errores en escala "
            "física es multiplicar por `apcorr`."
        ),
        code(
            "flux         = raw_flux * apcorr\n"
            "flux_err     = raw_err * apcorr\n"
            "flux_err_emp = raw_err_emp * apcorr\n"
            "print('flujo mediano:', round(float(np.nanmedian(flux)), 2),\n"
            "      '| error mediano:', round(float(np.nanmedian(flux_err)), 2))\n\n"
            "from musepipe.spectral import median_filter_1d\n"
            "fig, ax = plt.subplots(figsize=(11, 3.6))\n"
            "ax.fill_between(WAVE, -flux_err_emp, flux_err_emp, color='0.85', label='±σ empírico')\n"
            "ax.plot(WAVE, flux, lw=0.3, color='0.5', alpha=0.7)\n"
            "ax.plot(WAVE, median_filter_1d(flux, 41), lw=1.2, color='tab:blue', label='flujo (mediana 41 canales)')\n"
            "ax.axvline(6563, color='tab:red', ls=':', label='Hα')\n"
            "ax.set_xlabel('λ [Å]'); ax.legend(fontsize=8)\n"
            "ax.set_title('C2 rehecho en el notebook', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 10 · Comparación con la cadena\n\n"
            "Contra `spec_aperture_object.fits`, el producto que escribió la etapa. **Con las "
            "perillas por defecto debe salir idéntico** (a precisión de coma flotante): si no lo "
            "es, o la copia se desvió o alguna entrada no es la que usó la cadena. En cuanto "
            "cambias una perilla, esta celda mide exactamente qué se movió."
        ),
        code(
            "from musepipe.extraction.product import SpectrumProduct\n\n"
            "cadena = SpectrumProduct.read(SD / 'spec_aperture_object.fits')\n"
            "ref_flux = np.asarray(cadena.flux, dtype=float)\n"
            "ref_err  = np.asarray(cadena.flux_err, dtype=float)\n\n"
            "def _compara(nombre, mio, suyo, rtol=1e-9):\n"
            "    finito = np.isfinite(mio) & np.isfinite(suyo)\n"
            "    dif = np.abs(mio - suyo)[finito]\n"
            "    escala = np.maximum(np.abs(suyo)[finito], 1e-30)\n"
            "    iguales = np.isclose(mio[finito], suyo[finito], rtol=rtol, atol=0.0)\n"
            "    print(f'  {nombre:14s} idénticos {100 * iguales.mean():6.2f}% de {finito.sum()} canales'\n"
            "          f' | máx |Δ| = {dif.max():.3e} ({100 * (dif / escala).max():.2e}%)')\n"
            "    return bool(iguales.all())\n\n"
            "print('mi resultado vs la cadena:')\n"
            "ok = _compara('flujo', flux, ref_flux)\n"
            "ok &= _compara('error', flux_err, ref_err)\n"
            "ok &= _compara('apcorr', apcorr, np.asarray(cadena.apcorr, dtype=float))\n"
            "print()\n"
            "print('IDÉNTICO: la copia reproduce la cadena.' if ok else\n"
            "      'DIFIERE — si has tocado una perilla, es lo esperado; si no, revisa el chequeo de deriva.')\n\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.plot(WAVE, median_filter_1d(ref_flux, 41), lw=1.6, color='0.6', label='cadena')\n"
            "a1.plot(WAVE, median_filter_1d(flux, 41), lw=1.0, color='tab:blue', ls='--', label='este notebook')\n"
            "a1.legend(fontsize=8); a1.set_ylabel('flujo (mediana 41 ch)')\n"
            "a2.plot(WAVE, flux - ref_flux, lw=0.7, color='tab:purple')\n"
            "a2.axhline(0, color='0.7', lw=0.6)\n"
            "a2.set_ylabel('este − cadena'); a2.set_xlabel('λ [Å]')\n"
            "a1.set_title('comparación con el producto de la cadena', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 11 · Y contra el QC\n\n"
            "Los números que la etapa publicó en `spec_aperture_qc.json`, al lado de los de "
            "aquí. Sirve para ver si un cambio de perilla mueve algo que después mira D1 o E1."
        ),
        code(
            "qc = json.loads((SD / 'spec_aperture_qc.json').read_text(encoding='utf-8'))\n"
            "# El QC no publica el número de controles: viene del .npz que escribe la etapa.\n"
            "ctrl_npz = SD / 'spec_aperture_controls.npz'\n"
            "n_ctrl_cadena = int(np.load(ctrl_npz)['control_spectra'].shape[0]) if ctrl_npz.exists() else None\n"
            "err_qc = qc.get('errors') or {}\n"
            "mios = {\n"
            "    'aperture_correction.median': float(np.nanmedian(apcorr)),\n"
            "    'errors.mode': error_mode,\n"
            "    'errors.stat_factor_box3': STAT_FACTOR,\n"
            "    'errors.covariance_factor_box3': COV_FACTOR,\n"
            "    'n_controles': len(controls_yx),\n"
            "    'flujo mediano': float(np.nanmedian(flux)),\n"
            "    'canales marcados': int((flags != 0).sum()),\n"
            "}\n"
            "suyos = {\n"
            "    'aperture_correction.median': (qc.get('aperture_correction') or {}).get('median'),\n"
            "    'errors.mode': err_qc.get('mode'),\n"
            "    'errors.stat_factor_box3': err_qc.get('stat_factor_box3'),\n"
            "    'errors.covariance_factor_box3': err_qc.get('covariance_factor_box3'),\n"
            "    'n_controles': n_ctrl_cadena,\n"
            "    'flujo mediano': float(np.nanmedian(ref_flux)),\n"
            "    'canales marcados': (qc.get('flags') or {}).get('n_flagged'),\n"
            "}\n"
            "def _fmt(x):\n"
            "    return f'{x:20.4f}' if isinstance(x, float) else f'{str(x):>20s}'\n"
            "print(f\"{'clave':32s} {'este notebook':>20s} {'cadena':>20s}\")\n"
            "for k, v in mios.items():\n"
            "    w = suyos.get(k)\n"
            "    marca = '' if (w is None or (isinstance(v, float) and isinstance(w, (int, float))\n"
            "                                 and np.isclose(v, w, rtol=1e-9))\n"
            "                   or v == w) else '   <-- difiere'\n"
            "    print(f'{k:32s} {_fmt(v)} {_fmt(w)}{marca}')"
        ),
    ]
    return cells


BUILDERS = {"C2": ("C2_aperture_debug", build_c2_cells)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True, help="slug del objeto (targets/<slug>.json)")
    parser.add_argument("--run-id", default=None, help="run a auditar (por defecto, el de la cadena)")
    parser.add_argument("stages", nargs="*", help="etapas a generar (por defecto, todas)")
    args = parser.parse_args(argv)

    mb = _load_main_builder()
    run_id = args.run_id
    if run_id is None:
        default_run = f"{args.target}_realigned"
        config_json = ROOT / "runs" / default_run / "config" / "config.json"
        if config_json.exists():
            try:
                chain = json.loads(config_json.read_text(encoding="utf-8")).get("chain", {})
                default_run = chain.get("default_run", default_run)
            except (OSError, ValueError):
                pass
        run_id = default_run

    wanted = {s.upper() for s in args.stages} or set(BUILDERS)
    out_dir = NB_DIR / args.target / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stage_id, (slug, builder) in BUILDERS.items():
        if stage_id not in wanted:
            continue
        cells = builder(mb, args.target, run_id)
        path = out_dir / f"{slug}.ipynb"
        path.write_text(json.dumps(mb.notebook(cells), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        written.append(path.name)
    print(f"Generados {len(written)} notebooks de análisis en {out_dir} (run={run_id}):")
    for name in written:
        print("  ", name)


if __name__ == "__main__":
    main(sys.argv[1:])
