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
    # C3 comparte con C2 el fondo de anillo, los flags y la apcorr — es el mismo
    # contrato de la etapa — y añade lo suyo: el estimador optimo de Horne y el
    # ajuste de la PSF de la primaria, que es lo que separa las dos variantes.
    "C3": [
        ("musepipe/stats.py", ["finite_values", "robust_sigma", "robust_sigma_axis0"]),
        ("musepipe/apertures.py", [
            "angular_separation_deg", "aperture_weights", "same_radius_control_positions",
        ]),
        ("musepipe/extraction/aperture.py", [
            "_as_cube", "annulus_background_spectrum", "_flag_window", "channel_flags",
            "aperture_correction_from_psf",
        ]),
        ("musepipe/extraction/optimal.py", [
            "circular_window_indices", "normalized_psf_window", "covariance_factor_for_npix",
            "_channel_estimate", "estimate_variance_cube", "optimal_raw_spectrum",
            "control_optimal_spectra", "psf_image", "fit_primary_psf_model_cube",
        ]),
    ],
    # C4 ajusta DOS PSF a la vez por canal. Lo copiado es el ajuste entero: la
    # region, la matriz de diseño, el estimador por canal y los controles.
    "C4": [
        ("musepipe/stats.py", ["finite_values", "robust_sigma", "robust_sigma_axis0"]),
        ("musepipe/apertures.py", ["angular_separation_deg", "same_radius_control_positions"]),
        ("musepipe/extraction/aperture.py", ["_flag_window", "channel_flags"]),
        ("musepipe/extraction/optimal.py", ["covariance_factor_for_npix", "estimate_variance_cube"]),
        ("musepipe/extraction/psffit.py", [
            "PsfFitCubeResult", "fit_region_mask", "psf_pair_design", "_correlation",
            "_fit_one_channel", "fit_psffit_cube", "control_psffit_spectra",
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
        # También clases: C4 copia la dataclass que sus funciones construyen.
        found = {n.name: n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in names:
            node = found.get(name)
            if node is None:
                raise KeyError(f"{rel}: no se encuentra {name!r}")
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




def drift_cell(code, shas, stage_id):
    """La celda que compara la copia con el fuente actual de `musepipe`.

    Es la que sostiene toda la idea de copiar codigo, asi que vive en un solo
    sitio y no una vez por etapa.
    """
    return code(
        "import ast as _ast, hashlib as _hashlib\n\n"
        f"_SHAS = {json.dumps(shas, indent=4)}\n\n"
        "def chequeo_de_deriva(shas=_SHAS, root=ROOT):\n"
        "    problemas = []\n"
        "    for key, sha in shas.items():\n"
        "        rel, name = key.rsplit(':', 1)\n"
        "        text = (root / rel).read_text(encoding='utf-8')\n"
        "        lines = text.splitlines(keepends=True)\n"
        "        node = next((n for n in _ast.parse(text).body\n"
        "                     if isinstance(n, (_ast.FunctionDef, _ast.ClassDef)) and n.name == name),\n"
        "                    None)\n"
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
        "    print(f'\\nRegenera: python scripts/build_debug_notebooks.py"
        f" --target {{TARGET}} {stage_id}')\n"
        "else:\n"
        "    print(f'sin deriva: las {len(_SHAS)} piezas copiadas son las de musepipe')"
    )


def build_c2_cells(mb, target, run_id):
    """Las celdas de `C2_aperture_debug` para un objeto."""
    md, code = mb.md, mb.code
    sources = extract_sources("C2")
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
            "Salen del **config resuelto de la etapa**, no del `config.json` crudo: C2 rellena "
            "defaults que no están escritos en el run, y copiarlos a mano es exactamente cómo se "
            "consigue un notebook que no reproduce la cadena. Cambia lo que quieras **debajo** de "
            "la lectura y vuelve a ejecutar: la comparación del final dirá qué efecto tuvo."
        ),
        code(
            "from musepipe.stages.stage_x01_aperture import stage_x01_config_from_run\n\n"
            "# `project_root=ROOT` no es opcional: musepipe resuelve rutas contra el cwd, y\n"
            "# el cwd de un notebook es su propia carpeta, no la raíz del repo.\n"
            "X01 = stage_x01_config_from_run(RUN_ID, project_root=ROOT)   # run + defaults de la etapa\n"
            "APERTURE            = {'kind': 'box', 'size': 3}   # la caja que se compara con la cadena\n"
            "APCORR_MODE         = X01.get('x01_aperture_correction', 'auto')\n"
            "WINGS_INTACT        = bool(X01.get('x01_wings_intact_apcorr', True))\n"
            "ANNULUS_BKG_PX      = X01.get('x01_annulus_bkg_px', [8.0, 14.0, 30.0])\n"
            "N_CONTROLS          = int(X01.get('x01_control_apertures', 8))\n"
            "EXCLUDE_ANGLE_DEG   = float(X01.get('x01_control_exclude_angle_deg', 25.0))\n"
            "ERROR_MODE          = X01.get('x01_error_mode', 'auto')\n"
            "BAD_WINDOWS_A       = X01.get('x01_bad_windows_A', [])\n"
            "SKYLINE_WINDOWS_A   = X01.get('x01_skyline_windows_A', [])\n"
            "INTERPOLATED_WIN_A  = X01.get('x01_interpolated_windows_A', [])\n\n"
            "# ---- a partir de aquí, cambia lo que quieras probar ----\n\n"
            "for _k, _v in {'apertura': APERTURE, 'apcorr': APCORR_MODE, 'wings-intact': WINGS_INTACT,\n"
            "               'anillo fondo': ANNULUS_BKG_PX, 'controles': N_CONTROLS,\n"
            "               'modo error': ERROR_MODE}.items():\n"
            "    print(f'  {_k:14s} {_v}')"
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
        drift_cell(code, shas, "C2"),
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
            "## 10 · Validación externa contra `photutils` (opcional)\n\n"
            "La fotometría de la cadena **no usa `photutils` ni nada estilo DAOPHOT**: es una "
            "máscara de pesos propia (`musepipe/apertures.py`), binaria, y la caja se centra en el "
            "**píxel entero más cercano** al centroide de B3. Dos preguntas legítimas salen de ahí, "
            "y esta celda las contesta con una implementación independiente:\n\n"
            "1. **¿La máscara propia es correcta?** Misma caja, mismo centro entero, "
            "`method='center'`: tiene que dar **exactamente** lo mismo. Si no, hay un bug de "
            "índices o de convención de píxel.\n"
            "2. **¿Cuánto cuesta redondear el centro al píxel?** La misma caja centrada en la "
            "posición **fraccionaria** con `method='subpixel'`. Esa diferencia es una aproximación "
            "real de la cadena, no un error: mide cuánto flujo entra o sale por el desplazamiento.\n"
            "3. **¿Y si la apertura fuera circular con cobertura exacta?** `CircularAperture` con "
            "`method='exact'`, cada una con **su propia** `apcorr`. Si el modelo de PSF de C1 es "
            "bueno, el flujo total debe coincidir aunque las aperturas sean distintas — es la misma "
            "lógica de la verificación box3 vs box5 de la spec.\n\n"
            "> `photutils` **no está en `environment.yml`**: es una dependencia solo de análisis y "
            "la cadena no la necesita. Si no está, la celda lo dice y sigue.\n"
            "> Para activarla: `conda install -c conda-forge photutils` en el entorno `MUSE`."
        ),
        code(
            "try:\n"
            "    from photutils.aperture import (CircularAperture, RectangularAperture,\n"
            "                                    aperture_photometry)\n"
            "except ImportError:\n"
            "    print('photutils no está instalado: validación externa omitida.')\n"
            "    print('  conda install -c conda-forge photutils   (entorno MUSE)')\n"
            "else:\n"
            "    paso = 20   # 1 de cada N canales: la comparación no necesita los 3681\n"
            "    canales = np.arange(0, WAVE.size, paso)\n"
            "    yc, xc = OBJECT_YX\n"
            "    yi, xi = int(round(yc)), int(round(xc))\n"
            "    size = float(APERTURE['size'])\n"
            "    # photutils toma (x, y) y sitúa el centro del píxel en coordenada entera,\n"
            "    # igual que los índices de numpy: las posiciones son directamente comparables.\n"
            "    caja_entera = RectangularAperture([(xi, yi)], w=size, h=size, theta=0.0)\n"
            "    caja_frac   = RectangularAperture([(xc, yc)], w=size, h=size, theta=0.0)\n"
            "    circulo     = CircularAperture([(xc, yc)], r=size / 2.0)\n"
            "    def _fot(ap, metodo, **kw):\n"
            "        return np.array([float(aperture_photometry(CUBE[i], ap, method=metodo, **kw)['aperture_sum'][0])\n"
            "                         for i in canales])\n"
            "    pu_entera = _fot(caja_entera, 'center')\n"
            "    pu_frac   = _fot(caja_frac, 'subpixel', subpixels=32)\n"
            "    pu_circ   = _fot(circulo, 'exact')\n"
            "    # OJO: `raw_flux` ya lleva restado el fondo de anillo; para comparar con\n"
            "    # photutils, que solo suma, hay que devolvérselo.\n"
            "    mio_sin_fondo = raw_flux[canales] + (bkg[canales] * npix_eff[canales]\n"
            "                                        if bkg is not None else 0.0)\n"
            "    # Solo canales CON dato: el hueco del láser y los bordes son NaN en el\n"
            "    # cubo, y un NaN suelto convertiría el veredicto en «difieren».\n"
            "    fin = np.isfinite(mio_sin_fondo) & np.isfinite(pu_entera)\n"
            "    d = np.abs(mio_sin_fondo - pu_entera)[fin]\n"
            "    rel = d / np.maximum(np.abs(pu_entera[fin]), 1e-30)\n"
            "    igual = bool(np.allclose(mio_sin_fondo[fin], pu_entera[fin], rtol=1e-9, atol=0.0))\n"
            "    print('1) máscara propia vs photutils (misma caja, centro entero, method=center)')\n"
            "    print(f'   {fin.sum()} canales con dato | máx |Δ| = {d.max():.3e}'\n"
            "          f'  ({100 * rel.max():.2e}%)  ->  ' + ('IDÉNTICAS' if igual else 'DIFIEREN'))\n"
            "    # El efecto del centrado se mide DONDE HAY SEÑAL: en el azul el compañero\n"
            "    # tiene S/N<1 y un cociente por canal solo mediría ruido.\n"
            "    rojo = fin & (WAVE[canales] >= 7500) & (WAVE[canales] <= 9000)\n"
            "    med_ent, med_frac = np.nanmedian(pu_entera[rojo]), np.nanmedian(pu_frac[rojo])\n"
            "    print(f'2) centrar en el píxel vs en la posición real ({yc - yi:+.2f}, {xc - xi:+.2f} px),'\n"
            "          f' medido en 7500–9000 Å:')\n"
            "    print(f'   caja en el píxel {med_ent:10.2f} | caja en la posición real {med_frac:10.2f}'\n"
            "          f' -> {100 * (med_frac - med_ent) / abs(med_ent):+.2f}%')\n"
            "    dif_centro = 100 * (pu_frac - pu_entera) / np.maximum(np.abs(pu_entera), 1e-30)\n"
            "    print(f'   (por canal en esa banda: mediana {np.nanmedian(dif_centro[rojo]):+.2f}%,'\n"
            "          f' p5..p95 {np.nanpercentile(dif_centro[rojo], 5):+.2f}..'\n"
            "          f'{np.nanpercentile(dif_centro[rojo], 95):+.2f}%)')\n"
            "    # 3) flujo TOTAL: cada apertura con su propia corrección\n"
            "    apc_circ, _m, _r = aperture_correction_from_psf(\n"
            "        WAVE, {'kind': 'circle', 'radius_px': size / 2.0}, PSF_MODEL,\n"
            "        center_yx=OBJECT_YX, correction_mode=APCORR_MODE)\n"
            "    total_caja = pu_entera * apcorr[canales]\n"
            "    total_circ = pu_circ * apc_circ[canales]\n"
            "    razon = np.nanmedian(total_circ[rojo]) / np.nanmedian(total_caja[rojo])\n"
            "    print(f'3) flujo TOTAL en 7500–9000 Å: círculo r={size / 2:.1f} px (exact) / '\n"
            "          f'caja {size:.0f}×{size:.0f} = {razon:.4f}')\n"
            "    print('   (1.0 = la curva de crecimiento de C1 es consistente entre aperturas;'\n"
            "          ' es la misma prueba que box3 vs box5 de la spec)')\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)\n"
            "    ax1.plot(WAVE[canales], dif_centro, lw=0.8, color='tab:purple')\n"
            "    ax1.axhline(0, color='0.7', lw=0.6)\n"
            "    ax1.axvspan(7500, 9000, color='tab:red', alpha=0.07)\n"
            "    ax1.set_ylim(*np.nanpercentile(dif_centro[fin], [2, 98]))\n"
            "    ax1.set_ylabel('centrado [%]')\n"
            "    ax1.set_title('coste de redondear el centro al píxel (banda sombreada = donde el '\n"
            "                  'compañero se detecta)', fontsize=9)\n"
            "    ax2.plot(WAVE[canales], total_caja, lw=0.9, label=f'caja {size:.0f}×{size:.0f} × apcorr')\n"
            "    ax2.plot(WAVE[canales], total_circ, lw=0.9,\n"
            "             label=f'círculo r={size / 2:.1f} px (exact) × su apcorr')\n"
            "    ax2.axvspan(7500, 9000, color='tab:red', alpha=0.07)\n"
            "    ax2.set_xlabel('λ [Å]'); ax2.set_ylabel('flujo total'); ax2.legend(fontsize=8)\n"
            "    fig.tight_layout(); plt.show()"
        ),
        md(
            "## 11 · Comparación con la cadena\n\n"
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
            "## 12 · Y contra el QC\n\n"
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




def build_c3_cells(mb, target, run_id):
    """Las celdas de `C3_optimal_debug`: el estimador óptimo y sus DOS variantes."""
    md, code = mb.md, mb.code
    sources = extract_sources("C3")
    inline_src = "\n\n\n".join(src for _rel, _name, src, _sha in sources)
    shas = {f"{rel}:{name}": sha for rel, name, _src, sha in sources}

    return [
        md(
            f"# C3 · extracción óptima — notebook de análisis (`debug`)\n\n"
            f"**Objeto:** {target}  |  **Run:** `{run_id}`  |  "
            f"**Spec:** [`docs/spec_C3_codex_optimal_extraction.md`]"
            f"(../../../docs/spec_C3_codex_optimal_extraction.md)\n\n"
            "Rehace C3 **dentro del notebook**, con el código a la vista y editable, para probar "
            "y ajustar sin tocar `musepipe`. El notebook de auditoría es "
            "[`../C3_optimal.ipynb`](../C3_optimal.ipynb).\n\n"
            "**C3 produce dos métodos, no uno.** El estimador es el mismo — Horne (1986): por "
            "canal, cada píxel pesa por el perfil de PSF esperado y por la inversa de su varianza, "
            "`f = Σ M·P·D/V ÷ Σ M·P²/V` — y lo que cambia es **el cubo del que se extrae**:\n\n"
            "| variante | cubo | por qué existe |\n|---|---|---|\n"
            "| `optimal_ls` | residual de superficie local (04b) | **mismo fondo que C2**, así que "
            "compararlos aísla la ganancia del ponderado óptimo |\n"
            "| `optimal_psfsub` | cubo de B2 menos el **modelo de PSF de la primaria**, ajustado "
            "aquí canal a canal | anticipa el fondo de C4; `ls` vs `psfsub` es el diagnóstico del "
            "modelo de halo que consume D1 |\n\n"
            "Aquí se hacen **las dos**, en paralelo, y la comparación final las contrasta por "
            "separado contra sus productos de la cadena."
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
            "TARGET = nb.run_target(RUN_ID) or nb.display_name(RUN_ID)\n"
            "print('objeto :', TARGET, '·', nb.display_name(RUN_ID))\n"
            "print('run    :', RUN_ID)"
        ),
        md(
            "## 1 · Perillas\n\n"
            "Salen del **config resuelto de la etapa**, no del `config.json` crudo: C3 rellena "
            "defaults que no están escritos en el run (`x02_local_bkg_annulus_px` hereda de "
            "`x01_annulus_bkg_px`, el radio de ajuste de la primaria de `psf_norm_radius_px`…), y "
            "copiarlos a mano es exactamente cómo se consigue un notebook que no reproduce la "
            "cadena. Cambia lo que quieras **debajo** de la lectura y vuelve a ejecutar.\n\n"
            "`WINDOW_RADIUS_PX` es la que más mueve el resultado: define hasta dónde llega el "
            "ponderado, y la fracción de PSF que queda fuera la recupera después `apcorr`."
        ),
        code(
            "from musepipe.stages.stage_x02_optimal import stage_x02_config_from_run\n\n"
            "# `project_root=ROOT` no es opcional: musepipe resuelve rutas contra el cwd, y\n"
            "# el cwd de un notebook es su propia carpeta, no la raíz del repo.\n"
            "X02 = stage_x02_config_from_run(RUN_ID, project_root=ROOT)   # run + defaults de la etapa\n"
            "WINDOW_RADIUS_PX     = float(X02.get('x02_window_radius_px', 8.0))\n"
            "CLIP_SIGMA           = float(X02.get('x02_clip_sigma', 4.0))\n"
            "CLIP_MAX_ITER        = int(X02.get('x02_clip_max_iter', 2))\n"
            "APCORR_MODE          = X02.get('x02_aperture_correction', 'auto')\n"
            "ERROR_MODE           = X02.get('x02_error_mode', 'auto')\n"
            "N_CONTROLS           = int(X02.get('x02_control_apertures', 8))\n"
            "EXCLUDE_ANGLE_DEG    = float(X02.get('x02_control_exclude_angle_deg', 25.0))\n"
            "LOCAL_BKG_ANNULUS_PX = X02.get('x02_local_bkg_annulus_px')\n"
            "PRIMARY_FIT_RADIUS   = float(X02.get('x02_primary_fit_radius_px', 25.0))\n"
            "PRIMARY_EXCL_RADIUS  = float(X02.get('x02_primary_exclude_radius_px', WINDOW_RADIUS_PX))\n"
            "BAD_WINDOWS_A        = X02.get('x02_bad_windows_A', [])\n"
            "SKYLINE_WINDOWS_A    = X02.get('x02_skyline_windows_A', [])\n"
            "INTERPOLATED_WIN_A   = X02.get('x02_interpolated_windows_A', [])\n\n"
            "# ---- a partir de aquí, cambia lo que quieras probar ----\n\n"
            "for _k, _v in sorted({'ventana (px)': WINDOW_RADIUS_PX, 'clip σ': CLIP_SIGMA,\n"
            "                      'clip iter': CLIP_MAX_ITER, 'apcorr': APCORR_MODE,\n"
            "                      'modo error': ERROR_MODE, 'controles': N_CONTROLS,\n"
            "                      'anillo fondo': LOCAL_BKG_ANNULUS_PX,\n"
            "                      'radio ajuste primaria': PRIMARY_FIT_RADIUS,\n"
            "                      'radio exclusión compañero': PRIMARY_EXCL_RADIUS}.items()):\n"
            "    print(f'  {_k:26s} {_v}')"
        ),
        md(
            "## 2 · Entradas — **los dos cubos**\n\n"
            "`ls` sale del residual de 04b; `psfsub` del cubo de B2. Los dos deben tener la misma "
            "forma: la cadena lo exige y para aquí si no (serían dos rejillas distintas)."
        ),
        code(
            "qc_b3 = json.loads((SD / 'stage01c_qc.json').read_text(encoding='utf-8'))\n"
            "OBJECT_YX = tuple(float(v) for v in qc_b3['companion']['pos_yx'])\n"
            "STAR_YX   = tuple(float(v) for v in qc_b3['primary']['pos_yx'])\n"
            "PSF_MODEL = json.loads((SD / 'psf_model.json').read_text(encoding='utf-8'))\n\n"
            "with fits.open(SD / 'stage02_xcorr_cube_stack.fits') as h:\n"
            "    STAGE02 = np.asarray(h['CUBES'].data, dtype=float)\n"
            "    WAVE = np.asarray(h['WAVELENGTH'].data, dtype=float)\n"
            "    STAT_CUBE = np.asarray(h['STAT'].data, dtype=float) if 'STAT' in h else None\n"
            "if STAGE02.ndim == 4:\n"
            "    STAGE02 = STAGE02[0]\n"
            "if STAT_CUBE is not None and STAT_CUBE.ndim == 4:\n"
            "    STAT_CUBE = STAT_CUBE[0]\n"
            "LS_CUBE = np.asarray(fits.getdata(SD / 'cube_residual_local_object.fits'), dtype=float)\n"
            "assert LS_CUBE.shape == STAGE02.shape, (LS_CUBE.shape, STAGE02.shape)\n\n"
            "qc00 = json.loads((SD / 'stage00q_qc.json').read_text(encoding='utf-8'))\n"
            "qc01 = json.loads((SD / 'stage01_qc.json').read_text(encoding='utf-8'))\n"
            "m5 = qc00.get('m5_stat', {})\n"
            "# El STAT crudo se multiplica por el factor POR SPAXEL de M5 (aquí no es 1:\n"
            "# el DRS subestima la varianza) antes de entrar como peso del estimador.\n"
            "STAT_FACTOR = float(X02.get('x02_stat_factor_spaxel',\n"
            "                            m5.get('factor_spaxel_median', 1.0)) or 1.0)\n"
            "COV_FACTOR  = float(X02.get('x02_covariance_factor_box3',\n"
            "                            qc01.get('stat', {}).get('covariance_factor_box3', 1.0)) or 1.0)\n"
            "STAT_STATUS = str(X02.get('x02_stat_status', m5.get('status', 'unknown')))\n"
            "print('cubos    :', STAGE02.shape, '| STAT:', 'sí' if STAT_CUBE is not None else 'no')\n"
            "print('compañero:', [round(v, 2) for v in OBJECT_YX], ' primaria:', [round(v, 2) for v in STAR_YX])\n"
            f"print(f'STAT     : factor={{STAT_FACTOR:.3f}} covarianza={{COV_FACTOR:.3f}} estado={{STAT_STATUS}}')"
        ),
        md(
            "## 3 · Las funciones numéricas, copiadas de `musepipe`\n\n"
            "Copia **literal**; edítalas y el resultado cambia. Se importan solo "
            "`evaluate_psf_model` (es de C1) y `run_channel_chunks` (paralelismo, no física).\n\n"
            + "\n".join(f"- `{name}` — de `{rel}`" for rel, name, _s, _h in sources)
        ),
        code(
            "# ------------------------------------------------------------------\n"
            "# COPIA EDITABLE. Fuente: musepipe (ver el chequeo de deriva abajo).\n"
            "# ------------------------------------------------------------------\n"
            + "\n".join(needed_imports(sources)) + "\n"
            "from musepipe.psf import evaluate_psf_model      # de C1\n"
            "from musepipe.parallel import run_channel_chunks  # paralelismo, no física\n\n"
            + "\n".join(needed_constants(sources)) + "\n\n\n"
            + inline_src
        ),
        md("## 4 · Chequeo de deriva"),
        drift_cell(code, shas, "C3"),
        md(
            "## 5 · El modelo de la primaria (lo que separa las dos variantes)\n\n"
            "`psfsub` necesita restar la primaria antes de extraer. El ajuste es canal a canal, "
            "con la PSF de C1, **excluyendo un disco alrededor del compañero** para no absorberlo "
            "en el modelo de la estrella — si ese radio se queda corto, el modelo se come parte "
            "del compañero y `psfsub` sale bajo. Es una de las perillas interesantes de tocar.\n\n"
            "*(Es la celda cara: ajusta un modelo por canal. Un par de minutos.)*"
        ),
        code(
            "primary_model, psfsub_meta = fit_primary_psf_model_cube(\n"
            "    STAGE02, WAVE, STAR_YX, PSF_MODEL,\n"
            "    variance_zyx=STAT_CUBE,\n"
            "    fit_radius_px=PRIMARY_FIT_RADIUS,\n"
            "    exclude_centers_yx=[OBJECT_YX],\n"
            "    exclude_radius_px=PRIMARY_EXCL_RADIUS)\n"
            "PSFSUB_CUBE = STAGE02 - primary_model\n"
            "print('ajuste de la primaria:', {kk: psfsub_meta[kk] for kk in list(psfsub_meta)[:4]})\n\n"
            "iz = int(np.argmin(np.abs(WAVE - 7500)))\n"
            "y, x = int(round(STAR_YX[0])), int(round(STAR_YX[1]))\n"
            "sl = (slice(y - 40, y + 41), slice(x - 40, x + 41))\n"
            "fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))\n"
            "for ax, img, t in ((axes[0], STAGE02[iz][sl], 'B2 (con primaria)'),\n"
            "                   (axes[1], primary_model[iz][sl], 'modelo de la primaria'),\n"
            "                   (axes[2], PSFSUB_CUBE[iz][sl], 'residual = psfsub')):\n"
            "    v = np.nanpercentile(np.abs(img), 99)\n"
            "    ax.imshow(img, origin='lower', cmap='magma', vmin=-0.1 * v, vmax=v)\n"
            "    ax.set_title(f'{t}  (λ={WAVE[iz]:.0f} Å)', fontsize=8)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 6 · Las dos extracciones\n\n"
            "El mismo estimador sobre los dos cubos. `optimal_raw_spectrum` devuelve además la "
            "varianza propagada, `npix_eff` y la fracción de píxeles rechazados por canal — el "
            "clipping es el que hay que vigilar: si se concentra en el compañero, se está "
            "recortando la señal (es el chequeo `v4_clip_concentration` de la spec)."
        ),
        code(
            "def extrae(cube, variance, etiqueta):\n"
            "    # El peso del estimador es 1/varianza, y la varianza lleva el factor de M5.\n"
            "    if variance is not None:\n"
            "        variance = np.asarray(variance, dtype=float) * STAT_FACTOR\n"
            "    else:\n"
            "        variance = estimate_variance_cube(cube)\n"
            "    bkg = None\n"
            "    if LOCAL_BKG_ANNULUS_PX is not None:\n"
            "        bkg = annulus_background_spectrum(\n"
            "            cube, OBJECT_YX, LOCAL_BKG_ANNULUS_PX[0], LOCAL_BKG_ANNULUS_PX[1],\n"
            "            exclude_yx=STAR_YX,\n"
            "            exclude_radius=(LOCAL_BKG_ANNULUS_PX[2] if len(LOCAL_BKG_ANNULUS_PX) > 2 else 30.0))\n"
            "    raw = optimal_raw_spectrum(cube, variance, WAVE, OBJECT_YX, PSF_MODEL,\n"
            "                               window_radius_px=WINDOW_RADIUS_PX,\n"
            "                               clip_sigma=CLIP_SIGMA, clip_max_iter=CLIP_MAX_ITER,\n"
            "                               n_jobs=1, bkg_spectrum=bkg)\n"
            "    cov = covariance_factor_for_npix(raw['npix_eff'], COV_FACTOR)\n"
            "    raw_var = raw['variance'] * cov\n"
            "    ctrl_yx, ctrl = control_optimal_spectra(\n"
            "        cube, variance, WAVE, OBJECT_YX, STAR_YX, PSF_MODEL,\n"
            "        window_radius_px=WINDOW_RADIUS_PX, clip_sigma=CLIP_SIGMA,\n"
            "        clip_max_iter=CLIP_MAX_ITER, n_controls=N_CONTROLS,\n"
            "        exclude_angle_deg=EXCLUDE_ANGLE_DEG, n_jobs=1,\n"
            "        local_bkg_annulus_px=LOCAL_BKG_ANNULUS_PX)\n"
            "    err_emp = (robust_sigma_axis0(ctrl) if ctrl.shape[0] >= 2\n"
            "               else np.full(WAVE.size, robust_sigma(raw['flux'])))\n"
            "    usable = (variance is not None and str(ERROR_MODE).lower() != 'empirical'\n"
            "              and STAT_STATUS.lower() != 'red')\n"
            "    err = np.sqrt(np.clip(raw_var, 0.0, np.inf)) if usable else np.asarray(err_emp, float)\n"
            "    modo = 'stat' if usable else 'empirical'\n"
            "    apert = {'kind': 'circle', 'radius_px': float(WINDOW_RADIUS_PX),\n"
            "             'name': f'optimal_r{float(WINDOW_RADIUS_PX):g}'}\n"
            "    apcorr, apcorr_mode, _nr = aperture_correction_from_psf(\n"
            "        WAVE, apert, PSF_MODEL, center_yx=OBJECT_YX, correction_mode=APCORR_MODE)\n"
            "    print(f'{etiqueta:8s} modo={modo:9s} apcorr={float(np.nanmedian(apcorr)):6.2f} '\n"
            "          f'npix_eff={float(np.nanmedian(raw[\"npix_eff\"])):6.1f} '\n"
            "          f'clip_medio={100 * float(np.nanmedian(raw[\"clip_fraction\"])):.2f}%')\n"
            "    return {'raw': raw, 'flux': raw['flux'] * apcorr, 'err': err * apcorr,\n"
            "            'err_emp': np.asarray(err_emp, float) * apcorr, 'apcorr': apcorr,\n"
            "            'modo': modo, 'controles': ctrl, 'n_ctrl': len(ctrl_yx)}\n\n"
            "LS     = extrae(LS_CUBE, STAT_CUBE, 'ls')\n"
            "PSFSUB = extrae(PSFSUB_CUBE, STAT_CUBE, 'psfsub')"
        ),
        md(
            "## 7 · Las dos variantes, una al lado de la otra\n\n"
            "Es la comparación que D1 consume. Una diferencia **estructurada** entre ellas no es "
            "ruido: es el modelo de halo, porque el objeto y el estimador son los mismos y lo "
            "único que cambia es qué se restó antes."
        ),
        code(
            "from musepipe.spectral import median_filter_1d\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.plot(WAVE, median_filter_1d(LS['flux'], 41), lw=1.1, label='optimal_ls')\n"
            "a1.plot(WAVE, median_filter_1d(PSFSUB['flux'], 41), lw=1.1, label='optimal_psfsub')\n"
            "a1.axhline(0, color='0.7', lw=0.6); a1.axvline(6563, color='tab:red', ls=':')\n"
            "a1.legend(fontsize=8); a1.set_ylabel('flujo (mediana 41 ch)')\n"
            "a2.plot(WAVE, median_filter_1d(LS['flux'] - PSFSUB['flux'], 41), lw=1.0, color='tab:purple')\n"
            "a2.axhline(0, color='0.7', lw=0.6)\n"
            "a2.set_ylabel('ls − psfsub'); a2.set_xlabel('λ [Å]')\n"
            "a1.set_title('las dos variantes: mismo estimador, distinto fondo', fontsize=9)\n"
            "fig.tight_layout(); plt.show()\n"
            "for nombre, v in (('ls', LS), ('psfsub', PSFSUB)):\n"
            "    print(f\"{nombre:8s} flujo mediano = {float(np.nanmedian(v['flux'])):9.2f}\"\n"
            "          f\"  error mediano = {float(np.nanmedian(v['err'])):8.2f}  controles = {v['n_ctrl']}\")"
        ),
        md(
            "## 8 · Comparación con la cadena\n\n"
            "Cada variante contra **su** producto. Con las perillas por defecto deben salir "
            "idénticas; si tocas `WINDOW_RADIUS_PX` o el radio de exclusión de la primaria, aquí "
            "se ve exactamente cuánto se movió cada una."
        ),
        code(
            "from musepipe.extraction.product import SpectrumProduct\n\n"
            "def compara(nombre, mio, fichero, rtol=1e-9):\n"
            "    ref = SpectrumProduct.read(SD / fichero)\n"
            "    ok = True\n"
            "    print(f'{nombre} vs {fichero}:')\n"
            "    for clave, a, b in (('flujo', mio['flux'], np.asarray(ref.flux, float)),\n"
            "                        ('error', mio['err'], np.asarray(ref.flux_err, float)),\n"
            "                        ('apcorr', mio['apcorr'], np.asarray(ref.apcorr, float))):\n"
            "        fin = np.isfinite(a) & np.isfinite(b)\n"
            "        d = np.abs(a - b)[fin]\n"
            "        ig = np.isclose(a[fin], b[fin], rtol=rtol, atol=0.0)\n"
            "        print(f'   {clave:7s} idénticos {100 * ig.mean():6.2f}% de {fin.sum()} canales'\n"
            "              f' | máx |Δ| = {d.max():.3e}')\n"
            "        ok &= bool(ig.all())\n"
            "    return ok, ref\n\n"
            "ok_ls, ref_ls = compara('optimal_ls    ', LS, 'spec_optimal_object.fits')\n"
            "ok_ps, ref_ps = compara('optimal_psfsub', PSFSUB, 'spec_optimal_psfsub_object.fits')\n"
            "print()\n"
            "print('IDÉNTICO: la copia reproduce la cadena.' if (ok_ls and ok_ps) else\n"
            "      'DIFIERE — si has tocado una perilla, es lo esperado; si no, revisa el chequeo de deriva.')\n\n"
            "fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)\n"
            "for ax, (nombre, mio, ref) in zip(axes, (('optimal_ls', LS, ref_ls),\n"
            "                                         ('optimal_psfsub', PSFSUB, ref_ps))):\n"
            "    ax.plot(WAVE, median_filter_1d(np.asarray(ref.flux, float), 41), lw=1.6,\n"
            "            color='0.6', label='cadena')\n"
            "    ax.plot(WAVE, median_filter_1d(mio['flux'], 41), lw=1.0, ls='--',\n"
            "            color='tab:blue', label='este notebook')\n"
            "    ax.set_ylabel(nombre, fontsize=9); ax.legend(fontsize=8)\n"
            "axes[-1].set_xlabel('λ [Å]')\n"
            "fig.tight_layout(); plt.show()"
        ),
    ]


def build_c4_cells(mb, target, run_id):
    """Las celdas de `C4_psffit_debug`: el ajuste simultáneo de dos PSF."""
    md, code = mb.md, mb.code
    sources = extract_sources("C4")
    inline_src = "\n\n\n".join(src for _rel, _name, src, _sha in sources)
    shas = {f"{rel}:{name}": sha for rel, name, _src, sha in sources}

    return [
        md(
            f"# C4 · psffit — notebook de análisis (`debug`)\n\n"
            f"**Objeto:** {target}  |  **Run:** `{run_id}`  |  "
            f"**Spec:** [`docs/spec_C4_codex_psf_fitting.md`]"
            f"(../../../docs/spec_C4_codex_psf_fitting.md)\n\n"
            "Rehace C4 **dentro del notebook**. Es el **método canónico** de la cadena, y el "
            "único que no mide un residuo: en cada canal ajusta **dos PSF a la vez** —la primaria "
            "y el compañero— resolviendo un sistema lineal de dos amplitudes. El halo no se resta "
            "antes, se ajusta *junto con* la fuente.\n\n"
            "Eso trae su propio modo de fallo, y es el que hay que vigilar aquí: si las dos PSF se "
            "parecen demasiado en la región de ajuste, el sistema no puede repartir la luz entre "
            "ellas. La correlación **ρ(a,b)** mide justo eso, y es el chequeo `v4_rho_ab_ok` del "
            "QC.\n\n"
            "> **El ajuste es por canal y cuesta ~11 min para los 3681.** Por eso el notebook trae "
            "una perilla de submuestreo: cada canal se ajusta de forma independiente, así que "
            "quedarse con 1 de cada N no cambia el resultado de esos canales — verificado "
            "comparando dos submuestreos distintos, que salen **bit a bit iguales** en los canales "
            "comunes. Ponla a 1 para recorrer los 3681.\n\n"
            "> Frente al producto **guardado** por la cadena el acuerdo es de redondeo (~1e-12 en "
            "relativo), no bit a bit como en C2 y C3: aquí hay un sistema lineal por canal, no una "
            "suma. Por eso la comparación usa `rtol=1e-9`."
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
            "TARGET = nb.run_target(RUN_ID) or nb.display_name(RUN_ID)\n"
            "print('objeto :', TARGET, '·', nb.display_name(RUN_ID))\n"
            "print('run    :', RUN_ID)"
        ),
        md(
            "## 1 · Perillas\n\n"
            "Los dos radios son **la** decisión de esta etapa: definen la región donde se ajustan "
            "las dos PSF. Agrandar el del compañero mete más halo en el ajuste; encogerlo deja "
            "menos píxeles para separar las dos fuentes. Las dos cosas se ven en ρ(a,b)."
        ),
        code(
            "from musepipe.stages.stage_x03_psffit import stage_x03_config_from_run\n\n"
            "# `project_root=ROOT`: musepipe resuelve rutas contra el cwd, que en un\n"
            "# notebook es su propia carpeta, no la raíz del repo.\n"
            "X03 = stage_x03_config_from_run(RUN_ID, project_root=ROOT)\n"
            "STAR_RADIUS_PX    = float(X03.get('x03_star_radius_px', 20.0))\n"
            "COMP_RADIUS_PX    = float(X03.get('x03_comp_radius_px', 12.0))\n"
            "ERROR_MODE        = X03.get('x03_error_mode', 'auto')\n"
            "N_CONTROLS        = int(X03.get('x03_control_apertures', 8))\n"
            "EXCLUDE_ANGLE_DEG = float(X03.get('x03_control_exclude_angle_deg', 25.0))\n"
            "BAD_WINDOWS_A     = X03.get('x03_bad_windows_A', [])\n"
            "SKYLINE_WINDOWS_A = X03.get('x03_skyline_windows_A', [])\n"
            "INTERPOLATED_WIN_A = X03.get('x03_interpolated_windows_A', [])\n\n"
            "# Submuestreo: 1 de cada N canales. El ajuste es independiente por canal,\n"
            "# así que estos salen idénticos a los de la cadena completa. Pon 1 (y ~11\n"
            "# min de paciencia) para comparar los 3681.\n"
            "PASO_CANALES = 20\n\n"
            "# ---- a partir de aquí, cambia lo que quieras probar ----\n\n"
            "print(f'radios: primaria {STAR_RADIUS_PX:.0f} px · compañero {COMP_RADIUS_PX:.0f} px'\n"
            "      f' | controles {N_CONTROLS} | 1 de cada {PASO_CANALES} canales')"
        ),
        md(
            "## 2 · Entradas\n\n"
            "C4 extrae del **cubo de B2**, sin sustracción previa: la primaria entra en el ajuste "
            "como una de las dos componentes."
        ),
        code(
            "qc_b3 = json.loads((SD / 'stage01c_qc.json').read_text(encoding='utf-8'))\n"
            "COMP_YX = tuple(float(v) for v in qc_b3['companion']['pos_yx'])\n"
            "STAR_YX = tuple(float(v) for v in qc_b3['primary']['pos_yx'])\n"
            "PSF_MODEL = json.loads((SD / 'psf_model.json').read_text(encoding='utf-8'))\n\n"
            "with fits.open(SD / 'stage02_xcorr_cube_stack.fits') as h:\n"
            "    CUBE_FULL = np.asarray(h['CUBES'].data, dtype=float)\n"
            "    WAVE_FULL = np.asarray(h['WAVELENGTH'].data, dtype=float)\n"
            "    STAT_FULL = np.asarray(h['STAT'].data, dtype=float) if 'STAT' in h else None\n"
            "if CUBE_FULL.ndim == 4:\n"
            "    CUBE_FULL = CUBE_FULL[0]\n"
            "if STAT_FULL is not None and STAT_FULL.ndim == 4:\n"
            "    STAT_FULL = STAT_FULL[0]\n\n"
            "qc00 = json.loads((SD / 'stage00q_qc.json').read_text(encoding='utf-8'))\n"
            "qc01 = json.loads((SD / 'stage01_qc.json').read_text(encoding='utf-8'))\n"
            "m5 = qc00.get('m5_stat', {})\n"
            "STAT_FACTOR = float(X03.get('x03_stat_factor_spaxel',\n"
            "                            m5.get('factor_spaxel_median', 1.0)) or 1.0)\n"
            "COV_FACTOR  = float(X03.get('x03_covariance_factor_box3',\n"
            "                            qc01.get('stat', {}).get('covariance_factor_box3', 1.0)) or 1.0)\n"
            "STAT_STATUS = str(X03.get('x03_stat_status', m5.get('status', 'unknown')))\n\n"
            "CANALES = np.arange(0, WAVE_FULL.size, PASO_CANALES)\n"
            "CUBE = CUBE_FULL[CANALES]\n"
            "WAVE = WAVE_FULL[CANALES]\n"
            "STAT = None if STAT_FULL is None else STAT_FULL[CANALES]\n"
            "print('cubo     :', CUBE_FULL.shape, '-> se ajustan', WAVE.size, 'canales')\n"
            "print('primaria :', [round(v, 2) for v in STAR_YX],\n"
            "      ' compañero:', [round(v, 2) for v in COMP_YX])\n"
            "sep = float(np.hypot(COMP_YX[0] - STAR_YX[0], COMP_YX[1] - STAR_YX[1]))\n"
            "print(f'separación: {sep:.1f} px | radios de ajuste {STAR_RADIUS_PX:.0f}/{COMP_RADIUS_PX:.0f} px'\n"
            "      f\" -> las regiones {'SE SOLAPAN' if sep < STAR_RADIUS_PX + COMP_RADIUS_PX else 'no se solapan'}\")\n"
            f"print(f'STAT     : factor={{STAT_FACTOR:.3f}} covarianza={{COV_FACTOR:.3f}} estado={{STAT_STATUS}}')"
        ),
        md(
            "## 3 · Las funciones copiadas de `musepipe`\n\n"
            "Incluye la dataclass del resultado, porque el ajuste la construye.\n\n"
            + "\n".join(f"- `{name}` — de `{rel}`" for rel, name, _s, _h in sources)
        ),
        code(
            "# ------------------------------------------------------------------\n"
            "# COPIA EDITABLE. Fuente: musepipe (ver el chequeo de deriva abajo).\n"
            "# ------------------------------------------------------------------\n"
            + "\n".join(needed_imports(sources)) + "\n"
            "from dataclasses import dataclass\n"
            "from musepipe.psf import evaluate_psf_model      # de C1\n"
            "from musepipe.parallel import run_channel_chunks  # paralelismo, no física\n\n"
            + "\n".join(needed_constants(sources)) + "\n\n\n"
            + inline_src
        ),
        md("## 4 · Chequeo de deriva"),
        drift_cell(code, shas, "C4"),
        md(
            "## 5 · La región de ajuste y las dos PSF\n\n"
            "La máscara es la unión de dos discos. Y las dos columnas de la matriz de diseño son "
            "las dos PSF normalizadas: **si se parecen dentro de la máscara, el ajuste no puede "
            "separarlas**, y eso es exactamente lo que mide ρ(a,b) más abajo."
        ),
        code(
            "mask = fit_region_mask(CUBE.shape[1:], STAR_YX, COMP_YX,\n"
            "                       star_radius_px=STAR_RADIUS_PX, comp_radius_px=COMP_RADIUS_PX)\n"
            "iz = int(np.argmin(np.abs(WAVE - 7500)))\n"
            "design = psf_pair_design(CUBE.shape[1:], WAVE[iz], STAR_YX, COMP_YX, PSF_MODEL)\n"
            "print('píxeles en la región de ajuste:', int(mask.sum()))\n\n"
            "ys, xs = np.nonzero(mask)\n"
            "sl = (slice(ys.min() - 3, ys.max() + 4), slice(xs.min() - 3, xs.max() + 4))\n"
            "fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))\n"
            "axes[0].imshow(mask[sl], origin='lower', cmap='gray')\n"
            "axes[0].set_title('región de ajuste (unión de dos discos)', fontsize=8)\n"
            "for ax, i, t in ((axes[1], 0, 'PSF de la primaria'), (axes[2], 1, 'PSF del compañero')):\n"
            "    img = design[i][sl]\n"
            "    ax.imshow(img, origin='lower', cmap='magma',\n"
            "              vmax=np.nanpercentile(img, 99.5))\n"
            "    ax.set_title(f'{t}  (λ={WAVE[iz]:.0f} Å)', fontsize=8)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 6 · El ajuste, canal a canal\n\n"
            "Dos amplitudes por canal, con sus covarianzas. Los tres diagnósticos que importan:\n\n"
            "- **χ²ᵣ ~ 1** dice que el modelo describe el dato con el error que declara el STAT "
            "(verificación V1 de la spec).\n"
            "- **ρ(a,b)** es la degeneración: con \\|ρ\\|→1 el ajuste no puede decidir cuánta luz es "
            "de cada fuente, y el error real del compañero es mucho mayor que el formal. A esta "
            "separación se espera \\|ρ\\| < 0.3 (`v4_rho_ab_ok`).\n"
            "- El **número de condición** avisa de lo mismo por la vía numérica."
        ),
        code(
            "variance = (estimate_variance_cube(CUBE) if STAT is None\n"
            "            else np.asarray(STAT, dtype=float) * STAT_FACTOR)\n"
            "res = fit_psffit_cube(CUBE, variance, WAVE, STAR_YX, COMP_YX, PSF_MODEL,\n"
            "                      star_radius_px=STAR_RADIUS_PX, comp_radius_px=COMP_RADIUS_PX,\n"
            "                      n_jobs=1)\n"
            "print(f'χ²ᵣ mediano   = {float(np.nanmedian(res.chi2r)):.3f}')\n"
            "print(f'|ρ(a,b)| mediano = {float(np.nanmedian(np.abs(res.rho_ab))):.3f}'\n"
            "      f'  (p95 {float(np.nanpercentile(np.abs(res.rho_ab), 95)):.3f})')\n"
            "print(f'condición mediana = {float(np.nanmedian(res.condition_number)):.1f}')\n\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)\n"
            "a1.plot(WAVE, res.chi2r, lw=0.7); a1.axhline(1.0, color='tab:red', ls='--', lw=0.8)\n"
            "a1.set_ylabel('χ²ᵣ'); a1.set_ylim(0, np.nanpercentile(res.chi2r, 99))\n"
            "a2.plot(WAVE, res.rho_ab, lw=0.7, color='tab:purple')\n"
            "for lim in (-0.3, 0.3):\n"
            "    a2.axhline(lim, color='tab:red', ls='--', lw=0.8)\n"
            "a2.set_ylabel('ρ(a,b)'); a2.set_xlabel('λ [Å]')\n"
            "a1.set_title('¿describe el modelo al dato? ¿y puede separar las dos fuentes?', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 7 · Errores y controles\n\n"
            "El error formal sale de la covarianza del ajuste, inflado por el factor de covarianza "
            "espacial. El empírico se mide re-ajustando **el mismo par de PSF** en posiciones de "
            "control al mismo radio: mide la estabilidad del ajuste, no el ruido de fotones — y por "
            "eso en D2 viaja en columna aparte."
        ),
        code(
            "cov = covariance_factor_for_npix(res.npix_eff_comp, COV_FACTOR)\n"
            "comp_var = res.covariance[:, 1, 1] * cov\n"
            "star_var = res.covariance[:, 0, 0] * cov\n"
            "controls_yx, star_ctrl, comp_ctrl = control_psffit_spectra(\n"
            "    CUBE, variance, WAVE, STAR_YX, COMP_YX, PSF_MODEL,\n"
            "    star_radius_px=STAR_RADIUS_PX, comp_radius_px=COMP_RADIUS_PX,\n"
            "    n_controls=N_CONTROLS, exclude_angle_deg=EXCLUDE_ANGLE_DEG, n_jobs=1)\n"
            "comp_err_emp = (robust_sigma_axis0(comp_ctrl) if comp_ctrl.shape[0] >= 2\n"
            "                else np.full(WAVE.size, robust_sigma(res.coeffs[:, 1])))\n"
            "star_err_emp = (robust_sigma_axis0(star_ctrl) if star_ctrl.shape[0] >= 2\n"
            "                else np.full(WAVE.size, robust_sigma(res.coeffs[:, 0])))\n"
            "usable = (STAT is not None and str(ERROR_MODE).lower() != 'empirical'\n"
            "          and STAT_STATUS.lower() != 'red')\n"
            "comp_err = np.sqrt(np.clip(comp_var, 0.0, np.inf)) if usable else comp_err_emp\n"
            "star_err = np.sqrt(np.clip(star_var, 0.0, np.inf)) if usable else star_err_emp\n"
            "modo = 'stat' if usable else 'empirical'\n"
            "print(f'{len(controls_yx)} controles | modo de error: {modo}')\n"
            "print(f'  compañero: formal {float(np.nanmedian(np.sqrt(comp_var))):8.2f}'\n"
            "      f'  empírico {float(np.nanmedian(comp_err_emp)):8.2f}')\n"
            "print(f'  primaria : formal {float(np.nanmedian(np.sqrt(star_var))):8.2f}'\n"
            "      f'  empírico {float(np.nanmedian(star_err_emp)):8.2f}')"
        ),
        md(
            "## 8 · Los dos espectros\n\n"
            "C4 entrega **dos** productos: el compañero (`spec_psffit_object.fits`, el canónico de "
            "toda la cadena) y la primaria (`spec_psffit_star.fits`, que D2 calibra desde 2026-07-25). "
            "Aquí `apcorr` es 1: el ajuste devuelve directamente el flujo total de cada fuente, no "
            "el de una apertura."
        ),
        code(
            "comp_flux = res.coeffs[:, 1]\n"
            "star_flux = res.coeffs[:, 0]\n"
            "from musepipe.spectral import median_filter_1d\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)\n"
            "a1.plot(WAVE, median_filter_1d(star_flux, 11), lw=1.0, color='k')\n"
            "a1.set_ylabel('primaria'); a1.set_title('las dos componentes del ajuste', fontsize=9)\n"
            "a2.fill_between(WAVE, -comp_err_emp, comp_err_emp, color='0.85', label='±σ empírico')\n"
            "a2.plot(WAVE, median_filter_1d(comp_flux, 11), lw=1.0, color='tab:blue')\n"
            "a2.axvline(6563, color='tab:red', ls=':', label='Hα')\n"
            "a2.set_ylabel('compañero'); a2.set_xlabel('λ [Å]'); a2.legend(fontsize=8)\n"
            "fig.tight_layout(); plt.show()\n"
            "print(f'razón primaria/compañero (mediana): '\n"
            "      f'{float(np.nanmedian(star_flux) / np.nanmedian(comp_flux)):.0f}×')"
        ),
        md(
            "## 9 · Comparación con la cadena\n\n"
            "Los dos productos, **solo en los canales ajustados** (el submuestreo no cambia el "
            "resultado de un canal: el ajuste es independiente canal a canal). Con las perillas por "
            "defecto debe salir idéntico."
        ),
        code(
            "from musepipe.extraction.product import SpectrumProduct\n\n"
            "def compara(nombre, mio_flux, mio_err, fichero, rtol=1e-9):\n"
            "    ref = SpectrumProduct.read(SD / fichero)\n"
            "    ok = True\n"
            "    print(f'{nombre} vs {fichero}:')\n"
            "    for clave, a, b in (('flujo', mio_flux, np.asarray(ref.flux, float)[CANALES]),\n"
            "                        ('error', mio_err, np.asarray(ref.flux_err, float)[CANALES])):\n"
            "        fin = np.isfinite(a) & np.isfinite(b)\n"
            "        d = np.abs(a - b)[fin]\n"
            "        ig = np.isclose(a[fin], b[fin], rtol=rtol, atol=0.0)\n"
            "        print(f'   {clave:6s} idénticos {100 * ig.mean():6.2f}% de {fin.sum()} canales'\n"
            "              f' | máx |Δ| = {d.max():.3e}')\n"
            "        ok &= bool(ig.all())\n"
            "    return ok\n\n"
            "ok = compara('compañero', comp_flux, comp_err, 'spec_psffit_object.fits')\n"
            "ok &= compara('primaria ', star_flux, star_err, 'spec_psffit_star.fits')\n"
            "print()\n"
            "print('IDÉNTICO: la copia reproduce la cadena.' if ok else\n"
            "      'DIFIERE — si has tocado una perilla, es lo esperado; si no, revisa el chequeo de deriva.')"
        ),
    ]


BUILDERS = {
    "C2": ("C2_aperture_debug", build_c2_cells),
    "C3": ("C3_optimal_debug", build_c3_cells),
    "C4": ("C4_psffit_debug", build_c4_cells),
}


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
