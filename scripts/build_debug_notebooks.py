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
        # La apertura de C2 tambien viaja: la seccion 8 repite la extraccion con
        # ella para ver cuanto del resultado depende de la ventana y del peso.
        ("musepipe/extraction/aperture.py", [
            "_as_cube", "_npix_eff", "aperture_spectrum", "annulus_background_spectrum",
            "azimuthal_background_spectrum", "local_plane_background_spectrum",
            "aperture_stat_error",
            "control_aperture_spectra", "_flag_window", "channel_flags",
            "aperture_correction_from_psf",
        ]),
        ("musepipe/extraction/optimal.py", [
            "circular_window_indices", "normalized_psf_window", "covariance_factor_for_npix",
            "_channel_estimate", "estimate_variance_cube", "optimal_raw_spectrum",
            "local_background_spectrum", "control_optimal_spectra", "psf_image",
            "fit_primary_psf_model_cube",
        ]),
    ],
    # C4 ajusta DOS PSF a la vez por canal. Lo copiado es el ajuste entero: la
    # region, la matriz de diseño, el estimador por canal y los controles.
    "C4": [
        ("musepipe/stats.py", ["finite_values", "robust_sigma", "robust_sigma_axis0"]),
        ("musepipe/apertures.py", [
            "angular_separation_deg", "aperture_weights", "same_radius_control_positions",
        ]),
        # La apertura no la ajusta C4, pero la seccion 8 mide la primaria con
        # ella para contrastar el psffit por un camino independiente: es codigo
        # numerico del notebook, asi que viaja copiado como todo lo demas.
        ("musepipe/extraction/aperture.py", [
            "_as_cube", "_npix_eff", "aperture_spectrum", "_flag_window", "channel_flags",
            "aperture_correction_from_psf",
        ]),
        ("musepipe/extraction/optimal.py", ["covariance_factor_for_npix", "estimate_variance_cube"]),
        ("musepipe/extraction/psffit.py", [
            "PsfFitCubeResult", "fit_region_mask", "psf_pair_design", "_correlation",
            "_fit_one_channel", "fit_psffit_cube", "control_psffit_spectra",
        ]),
    ],
}

#: Lo que C5 y C6 tienen en comun: la referencia estelar y la apertura de C2.
_HALOSUB_COMUN = [
    ("musepipe/stats.py", ["finite_values", "robust_sigma", "robust_sigma_axis0"]),
    ("musepipe/apertures.py", [
        "angular_separation_deg", "aperture_weights", "same_radius_control_positions",
    ]),
    ("musepipe/extraction/aperture.py", [
        "_as_cube", "_npix_eff", "aperture_spectrum", "annulus_background_spectrum",
        "aperture_stat_error", "control_aperture_spectra", "_flag_window", "channel_flags",
        "aperture_correction_from_psf",
    ]),
    ("musepipe/halosub.py", [
        "select_reference_spaxels", "reference_spectrum", "safe_reference", "fill_nan_along_axis0",
    ]),
]

INLINE_SOURCES["C5"] = _HALOSUB_COMUN + [
    ("musepipe/halosub.py", ["SgfResult", "sgf_subtract"]),
]
INLINE_SOURCES["C6"] = _HALOSUB_COMUN + [
    ("musepipe/halosub.py", [
        "LpmResult", "lpm_design_matrix", "lpm_fit_mask", "lpm_subtract",
        "lpm_coefficient_energy_share",
    ]),
]


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
    # Lo que la copia define no se importa: el `def` local lo pisaria igual, pero
    # importar algo que acto seguido se redefine confunde a quien lee.
    copiados = {name for _rel, name, _src, _sha in sources}
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
                    if local in names and local not in copiados:
                        used.add(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                # Los imports RELATIVOS se resuelven a absolutos: en el notebook
                # no hay paquete que los ancle. Sin esto, `from .spectral import
                # STANDARD_LINE_WINDOWS_A` se perdía y la copia reventaba al
                # ejecutarse (le pasó a C6).
                modulo = node.module or ""
                if node.level:
                    partes = Path(rel).with_suffix("").parts
                    paquete = list(partes[:-1])
                    if node.level > 1:
                        paquete = paquete[: -(node.level - 1)]
                    modulo = ".".join([*paquete, modulo]) if modulo else ".".join(paquete)
                for alias in node.names:
                    local = alias.asname or alias.name
                    if local in names and local not in copiados:
                        # El alias importa: halosub usa `legendre as npleg`.
                        sufijo = f" as {alias.asname}" if alias.asname else ""
                        used.add(f"from {modulo} import {alias.name}{sufijo}")
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
            "# Resolución de las figuras EN PANTALLA. `savefig` guarda a 300 dpi, pero\n"
            "# lo que se ve dentro del notebook lo fija el backend inline, que va a 100\n"
            "# dpi por defecto y sale borroso. `retina` dobla los píxeles sin cambiar el\n"
            "# tamaño aparente; fuera de IPython no hace nada y queda el rcParam.\n"
            "import matplotlib as mpl\n"
            "mpl.rcParams['figure.dpi'] = 120\n"
            "mpl.rcParams['savefig.dpi'] = 200\n"
            "try:\n"
            "    from matplotlib_inline.backend_inline import set_matplotlib_formats\n"
            "    set_matplotlib_formats('retina')\n"
            "except Exception:\n"
            "    pass\n"
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
            "# La unidad, con la MISMA regla que usa la cadena: el stack de B2 no\n"
            "# lleva BUNIT (se escribió antes de que B1/B2 lo propagaran), así que\n"
            "# `resolve_bunit` cae al cubo de entrada del run. Leer solo la\n"
            "# cabecera del stack dejaba las colorbars sin unidad.\n"
            "from musepipe.io import resolve_bunit\n"
            "_stack_bunit = str(fits.getheader(CUBE_PATH, 0).get('BUNIT', '') or\n"
            "                   fits.getheader(CUBE_PATH, 1).get('BUNIT', '')) or None\n"
            "BUNIT = resolve_bunit(X01, stack_bunit=_stack_bunit)\n"
            "UNIDAD = BUNIT or 'sin unidad declarada'   # etiqueta de las colorbars\n"
            "print('cubo      :', CUBE_PATH.name, CUBE.shape, '| wings-intact:', use_raw,\n"
            "      '| BUNIT:', BUNIT or 'sin declarar')\n"
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
            "# `evaluate_psf_model` viene de C1 y `run_channel_chunks` es paralelismo:\n"
            "# no son lo que se ajusta aquí, por eso se importan en vez de copiarse.\n\n"
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
            "## 5 · Paso 1 — dónde se mide\n\n"
            "Antes de sumar nada, ver el sitio. Dos vistas del **mismo** dato, con escalas "
            "distintas a propósito:\n\n"
            "- **izquierda**, el campo entero en escala **logarítmica**: así se ve el halo de la "
            "primaria, que es lo que domina y lo que hay que quitar. El compañero es invisible "
            "aquí, y esa es la lección.\n"
            "- **derecha**, un zoom sobre el compañero **con el halo ya restado** (el residual de "
            "04b) y escala por percentiles: ahora sí se ve la fuente que estamos midiendo.\n\n"
            "En las dos, la caja de extracción; en la izquierda, además, el anillo de fondo y la "
            "primaria. La imagen es la **mediana en λ** (1 de cada 20 canales), y el zoom la toma "
            "solo en el rojo (7500–9000 Å), que es donde el compañero se detecta."
        ),
        code(
            "from matplotlib.colors import LogNorm\n"
            "from matplotlib.patches import Circle, Rectangle\n\n"
            "weights = aperture_weights(CUBE.shape[1], CUBE.shape[2], OBJECT_YX, APERTURE)\n"
            "raw_flux, npix_eff = aperture_spectrum(CUBE, OBJECT_YX, APERTURE)\n"
            "print('píxeles con peso:', int((weights > 0).sum()),\n"
            "      '| npix_eff mediano:', float(np.nanmedian(npix_eff)))\n\n"
            "campo = np.nanmedian(CUBE[::20], axis=0)\n"
            "rojo_ch = (WAVE >= 7500) & (WAVE <= 9000)\n"
            "resid_path = SD / 'cube_residual_local_object.fits'\n"
            "zoom_src = (np.nanmedian(np.asarray(fits.getdata(resid_path), float)[rojo_ch][::10], axis=0)\n"
            "            if resid_path.exists() else campo)\n"
            "titulo_zoom = 'halo restado (residual 04b)' if resid_path.exists() else 'sin residual 04b en disco'\n\n"
            "yc, xc = OBJECT_YX\n"
            "size = float(APERTURE['size'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))\n"
            "pos = campo[np.isfinite(campo) & (campo > 0)]\n"
            "a1.imshow(campo, origin='lower', cmap='magma',\n"
            "          norm=LogNorm(vmin=np.percentile(pos, 60), vmax=np.percentile(pos, 99.9)))\n"
            "a1.add_patch(Rectangle((xc - size/2 - 0.5, yc - size/2 - 0.5), size, size,\n"
            "                       fill=False, edgecolor='tab:cyan', lw=1.6))\n"
            "if ANNULUS is not None:\n"
            "    for r, ls_ in ((ANNULUS[0], '-'), (ANNULUS[1], '-')):\n"
            "        a1.add_patch(Circle((xc, yc), r, fill=False, edgecolor='tab:cyan', lw=0.9, ls=ls_, alpha=0.8))\n"
            "a1.plot(STAR_YX[1], STAR_YX[0], marker='+', ms=11, mew=1.6, color='w')\n"
            "a1.annotate('primaria', (STAR_YX[1], STAR_YX[0]), textcoords='offset points',\n"
            "            xytext=(0, 9), ha='center', fontsize=7, color='w')\n"
            "a1.annotate('compañero', (xc, yc), textcoords='offset points', xytext=(0, 10),\n"
            "            ha='center', fontsize=7, color='tab:cyan')\n"
            "im1 = a1.get_images()[0]\n"
            "cb1 = fig.colorbar(im1, ax=a1, shrink=0.82)\n"
            "cb1.set_label(f'flujo mediano [{UNIDAD}] · escala LOG', fontsize=7)\n"
            "a1.set_title(f'campo, escala log ({np.percentile(pos, 60):.3g}–{np.percentile(pos, 99.9):.3g}):'\n"
            "             ' manda el halo (el compañero no se ve)', fontsize=9)\n"
            "a1.set_xlabel('x [px]'); a1.set_ylabel('y [px]')\n\n"
            "h = 12\n"
            "y0, x0 = int(round(yc)) - h, int(round(xc)) - h\n"
            "recorte = zoom_src[y0:y0 + 2*h + 1, x0:x0 + 2*h + 1]\n"
            "fin_r = recorte[np.isfinite(recorte)]\n"
            "a2.imshow(recorte, origin='lower', cmap='viridis',\n"
            "          vmin=np.percentile(fin_r, 5), vmax=np.percentile(fin_r, 99.5),\n"
            "          extent=[x0 - 0.5, x0 + 2*h + 0.5, y0 - 0.5, y0 + 2*h + 0.5])\n"
            "a2.add_patch(Rectangle((xc - size/2 - 0.5, yc - size/2 - 0.5), size, size,\n"
            "                       fill=False, edgecolor='tab:red', lw=1.8))\n"
            "im2 = a2.get_images()[0]\n"
            "cb2 = fig.colorbar(im2, ax=a2, shrink=0.82)\n"
            "cb2.set_label(f'flujo mediano [{UNIDAD}] · escala LINEAL', fontsize=7)\n"
            "a2.set_title(f'zoom al compañero · {titulo_zoom}\\n"
            "escala lineal, percentiles 5–99.5 "
            "({np.percentile(fin_r, 5):.3g}–{np.percentile(fin_r, 99.5):.3g}) · mediana en 7500–9000 Å',\n"
            "             fontsize=9)\n"
            "a2.set_xlabel('x [px]')\n"
            "fig.tight_layout(); plt.show()\n\n"
            "fig, ax = plt.subplots(figsize=(11, 2.6))\n"
            "ax.plot(WAVE, npix_eff, lw=0.8)\n"
            "ax.set_xlabel('λ [Å]'); ax.set_ylabel('npix_eff')\n"
            "ax.set_title('píxeles efectivos por canal: no son 9 fijos, bajan donde hay NaN', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 6 · Paso 2 — el fondo de anillo, y dónde se mide\n\n"
            "Solo cuando la extracción es *wings-intact* (cubo crudo). Se toma la **mediana por "
            "canal** de los píxeles del anillo entre `ANNULUS[0]` y `ANNULUS[1]` alrededor del "
            "compañero, **excluyendo** un disco de `ANNULUS[2]` px alrededor de la primaria — sin "
            "esa exclusión el anillo mediría el halo de la estrella, no el fondo local.\n\n"
            "Se resta multiplicado por `npix_eff`, para pasar de fondo **por píxel** a fondo **de "
            "la apertura**. La figura enseña exactamente qué píxeles entran."
        ),
        code(
            "if ANNULUS is not None:\n"
            "    bkg = annulus_background_spectrum(CUBE, OBJECT_YX, ANNULUS[0], ANNULUS[1],\n"
            "                                      exclude_yx=STAR_YX,\n"
            "                                      exclude_radius=(ANNULUS[2] if len(ANNULUS) > 2 else 30.0))\n"
            "    raw_flux_bkgsub = raw_flux - bkg * npix_eff\n"
            "    print(f'fondo mediano por píxel      : {float(np.nanmedian(bkg)):8.3f}')\n"
            "    print(f'resta mediana a la apertura  : {float(np.nanmedian(bkg * npix_eff)):8.2f}'\n"
            "          f'  (= fondo × npix_eff)')\n"
            "    # Los mismos pixeles que usa `annulus_background_spectrum`, dibujados.\n"
            "    yy, xx = np.indices(CUBE.shape[1:], dtype=float)\n"
            "    rr = np.hypot(yy - OBJECT_YX[0], xx - OBJECT_YX[1])\n"
            "    rr_star = np.hypot(yy - STAR_YX[0], xx - STAR_YX[1])\n"
            "    anillo = (rr >= ANNULUS[0]) & (rr <= ANNULUS[1])\n"
            "    excl = rr_star <= (ANNULUS[2] if len(ANNULUS) > 2 else 30.0)\n"
            "    usados = anillo & ~excl\n"
            "    print(f'píxeles del anillo           : {int(anillo.sum())}'\n"
            "          f' | usados tras excluir la primaria: {int(usados.sum())}')\n"
            "    h2 = int(ANNULUS[1]) + 4\n"
            "    y0, x0 = int(round(OBJECT_YX[0])) - h2, int(round(OBJECT_YX[1])) - h2\n"
            "    sl = (slice(max(y0, 0), y0 + 2*h2 + 1), slice(max(x0, 0), x0 + 2*h2 + 1))\n"
            "    fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "    img = np.nanmedian(CUBE[::20], axis=0)[sl]\n"
            "    fin_i = img[np.isfinite(img)]\n"
            "    b1.imshow(img, origin='lower', cmap='magma',\n"
            "              vmin=np.percentile(fin_i, 5), vmax=np.percentile(fin_i, 99))\n"
            "    cbb = fig.colorbar(b1.get_images()[0], ax=b1, shrink=0.8)\n"
            "    cbb.set_label(f'flujo mediano [{UNIDAD}] · lineal', fontsize=7)\n"
            "    b1.set_title(f'dato (mediana en λ) · escala lineal, percentiles 5–99'\n"
            "                 f' ({np.percentile(fin_i, 5):.3g}–{np.percentile(fin_i, 99):.3g})', fontsize=9)\n"
            "    mascara = np.where(usados[sl], 1.0, np.where(anillo[sl], 0.4, np.nan))\n"
            "    b2.imshow(img, origin='lower', cmap='gray',\n"
            "              vmin=np.percentile(fin_i, 5), vmax=np.percentile(fin_i, 99))\n"
            "    b2.imshow(mascara, origin='lower', cmap='cool', alpha=0.55, vmin=0, vmax=1)\n"
            "    b2.set_title('anillo: en claro lo usado, en oscuro lo excluido\\n(disco de la primaria)',\n"
            "                 fontsize=9)\n"
            "    fig.tight_layout(); plt.show()\n"
            "else:\n"
            "    bkg = None\n"
            "    raw_flux_bkgsub = raw_flux\n"
            "    print('sin fondo de anillo (se extrae del residual de 04b)')\n"
            "raw_flux = raw_flux_bkgsub"
        ),
        md(
            "## 7 · Paso 3 — controles y error empírico\n\n"
            "**La regla que gobierna todo el modelo de ruido**: σ no sale del STAT del cubo, sale "
            "de medir **lo mismo donde no hay nada**. Los controles son aperturas idénticas "
            "(misma caja, mismo radio a la primaria, mismo fondo de anillo) repartidas en ángulo, "
            "excluyendo un cono alrededor del compañero.\n\n"
            "**Cómo se calcula σ empírico**, canal a canal:\n\n"
            "1. se extrae el espectro de cada uno de los N controles, con el mismo procedimiento;\n"
            "2. para **cada canal**, se mira la dispersión de esos N valores;\n"
            "3. esa dispersión se mide con `robust_sigma_axis0`, que usa la **MAD** (mediana de "
            "las desviaciones absolutas × 1.4826) en vez de la desviación típica, para que un "
            "control contaminado no infle σ.\n\n"
            "Es decir: σ(λ) **no** es la dispersión del objeto a lo largo de λ, sino la dispersión "
            "**entre posiciones** en ese canal. Ver [`docs/noise_model.md`]"
            "(../../../docs/noise_model.md).\n\n"
            "En la figura: **cada línea gris es un control** (su espectro completo); la azul es el "
            "**objeto**, que es la **suma de la apertura box3** —no el píxel más brillante—, ya con "
            "el fondo restado; la roja es el σ que sale del paso 3."
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
            "print(f'{len(controls_yx)} controles a {np.hypot(*(np.array(OBJECT_YX) - np.array(STAR_YX))):.1f} px'\n"
            "      f' de la primaria | σ empírico mediano = {float(np.nanmedian(raw_err_emp)):.2f}')\n"
            "print('σ(λ) = dispersión ENTRE los controles en ese canal (MAD robusta), no a lo largo de λ')\n\n"
            "# Donde estan los N controles: mismo radio a la primaria, repartidos en\n"
            "# angulo, con un cono excluido alrededor del compañero.\n"
            "campo_c = np.nanmedian(CUBE[::20], axis=0)\n"
            "pos_c = campo_c[np.isfinite(campo_c) & (campo_c > 0)]\n"
            "sep_px = float(np.hypot(*(np.array(OBJECT_YX) - np.array(STAR_YX))))\n"
            "figc, axc = plt.subplots(figsize=(6.6, 6.0))\n"
            "imc = axc.imshow(campo_c, origin='lower', cmap='magma',\n"
            "                 norm=LogNorm(vmin=np.percentile(pos_c, 60), vmax=np.percentile(pos_c, 99.9)))\n"
            "cbc = figc.colorbar(imc, ax=axc, shrink=0.82)\n"
            "cbc.set_label('flujo mediano · escala LOG', fontsize=7)\n"
            "axc.add_patch(Circle((STAR_YX[1], STAR_YX[0]), sep_px, fill=False,\n"
            "                     edgecolor='w', lw=0.8, ls=':', alpha=0.6))\n"
            "axc.plot(STAR_YX[1], STAR_YX[0], marker='+', ms=11, mew=1.6, color='w')\n"
            "axc.add_patch(Rectangle((OBJECT_YX[1] - size/2 - 0.5, OBJECT_YX[0] - size/2 - 0.5),\n"
            "                        size, size, fill=False, edgecolor='tab:cyan', lw=1.8))\n"
            "axc.annotate('compañero', (OBJECT_YX[1], OBJECT_YX[0]), textcoords='offset points',\n"
            "             xytext=(0, 10), ha='center', fontsize=7, color='tab:cyan')\n"
            "for k, (cy, cx) in enumerate(controls_yx):\n"
            "    axc.add_patch(Rectangle((cx - size/2 - 0.5, cy - size/2 - 0.5), size, size,\n"
            "                            fill=False, edgecolor='tab:green', lw=1.2))\n"
            "    axc.annotate(str(k), (cx, cy), textcoords='offset points', xytext=(0, 7),\n"
            "                 ha='center', fontsize=6, color='tab:green')\n"
            "axc.set_title(f'los {len(controls_yx)} controles: mismo radio ({sep_px:.0f} px) que el compañero,'\n"
            "              f'\\ncono de ±{EXCLUDE_ANGLE_DEG:.0f}° excluido a su alrededor', fontsize=9)\n"
            "axc.set_xlabel('x [px]'); axc.set_ylabel('y [px]')\n"
            "figc.tight_layout(); plt.show()\n\n"
            "from musepipe.reduction.telluric import TELLURIC_BANDS\n"
            "fig, ax = plt.subplots(figsize=(11, 3.6))\n"
            "for j, (lo, hi) in enumerate(BAD_WINDOWS_A):\n"
            "    ax.axvspan(lo, hi, color='0.85', zorder=0,\n"
            "               label='ventana ignorada (flags)' if j == 0 else None)\n"
            "for j, (nombre, (lo, hi)) in enumerate(TELLURIC_BANDS.items()):\n"
            "    ax.axvspan(lo, hi, color='tab:orange', alpha=0.13, zorder=0,\n"
            "               label='bandas telúricas (A3)' if j == 0 else None)\n"
            "    ax.annotate(nombre, ((lo + hi) / 2, 0.97), xycoords=('data', 'axes fraction'),\n"
            "                ha='center', va='top', fontsize=6, color='tab:orange')\n"
            "for i, c in enumerate(control_bkgsub):\n"
            "    ax.plot(WAVE, c, lw=0.4, alpha=0.35, color='0.6',\n"
            "            label=f'{len(control_bkgsub)} controles (uno por línea)' if i == 0 else None)\n"
            "ax.plot(WAVE, raw_flux, lw=0.6, color='tab:blue', label='objeto = suma de la caja box3')\n"
            "ax.plot(WAVE, raw_err_emp, lw=1.0, color='tab:red', label='σ empírico = dispersión entre controles')\n"
            "ax.set_xlabel('λ [Å]'); ax.legend(fontsize=8)\n"
            "ax.set_title('el objeto y los controles, procesados igual', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 8 · Paso 4 — error por STAT y corrección de apertura\n\n"
            "El STAT del cubo **nunca** se usa crudo: lleva el factor de M5 (el DRS subestima la "
            "varianza) y el de covarianza de B1 (el desplazamiento subpíxel correlacionó píxeles "
            "vecinos, así que sumar N píxeles no da √N).\n\n"
            "### Qué es `apcorr` (y qué NO es)\n\n"
            "**No es un porcentaje: es un factor multiplicativo adimensional.** Si `f` es la "
            "fracción de la PSF que cae dentro de la apertura, entonces\n\n"
            "> `apcorr = 1 / f`,  y por tanto  `f = 1 / apcorr`\n\n"
            "Una caja 3×3 en NFM recoge `f ≈ 0.022`, o sea el **2.2%** de la luz de la fuente, así "
            "que `apcorr ≈ 45`. Leído al revés: **`apcorr = 100` significaría que la caja recoge "
            "el 1%**, no el 100%. Cuanto mayor el número, *menos* luz entra en la apertura.\n\n"
            "Y no, no es «el flujo que llega de la fuente a ese lugar»: es el factor con el que "
            "hay que multiplicar lo medido para recuperar el flujo **total** de la fuente.\n\n"
            "- **De dónde sale**: del modelo de PSF de **C1** (`psf_model.json`). Para cada λ se "
            "evalúa la PSF, se suma dentro de la **misma máscara de apertura** y se toma "
            "`apcorr = 1 / fracción`. La PSF está normalizada dentro de `norm_radius_px` (25 px), "
            "así que «flujo total» significa *el que hay dentro de ese radio*.\n"
            "- **Depende de λ** porque la PSF se ensancha hacia el azul: más luz fuera de la caja, "
            "corrección mayor.\n"
            "- **Cómo se aplica**: multiplicando, a **flujo y errores por igual** "
            "(`flux = raw × apcorr`), así que no cambia la relación señal-ruido; solo la escala.\n"
            "- Si no hay modelo de PSF, `apcorr = 1` y el producto lo declara (`APCMODE=none`) en "
            "vez de fingir una corrección."
        ),
        code(
            "# El STAT se propaga SIEMPRE que exista, aunque la etapa no lo use como\n"
            "# sigma: es la segunda estimación con la que compararlo. Si solo se\n"
            "# calculara cuando manda, en los objetos con M5 rojo (que son los que\n"
            "# más interesa vigilar) no habría con qué comparar.\n"
            "raw_err_stat = (aperture_stat_error(STAT_CUBE, OBJECT_YX, APERTURE,\n"
            "                                    stat_factor=STAT_FACTOR,\n"
            "                                    covariance_factor=COV_FACTOR)\n"
            "                if STAT_CUBE is not None else None)\n"
            "stat_usable = (raw_err_stat is not None and str(ERROR_MODE).lower() != 'empirical'\n"
            "               and STAT_STATUS.lower() != 'red')\n"
            "if stat_usable:\n"
            "    raw_err = raw_err_stat\n"
            "    error_mode = 'stat'\n"
            "else:\n"
            "    raw_err = np.asarray(raw_err_emp, dtype=float)\n"
            "    error_mode = 'empirical'\n"
            "if raw_err_stat is not None and not stat_usable:\n"
            "    razon = float(np.nanmedian(raw_err_stat) / np.nanmedian(raw_err_emp))\n"
            "    print(f'el STAT NO se usa como σ (estado {STAT_STATUS}), pero se propaga'\n"
            "          f' igual para poder compararlo: STAT/empírico = {razon:.2f}×')\n\n"
            "apcorr, apcorr_mode, norm_radius = aperture_correction_from_psf(\n"
            "    WAVE, APERTURE, PSF_MODEL, center_yx=OBJECT_YX, correction_mode=APCORR_MODE)\n"
            "flags = channel_flags(WAVE, bad_windows_A=BAD_WINDOWS_A, skyline_windows_A=SKYLINE_WINDOWS_A,\n"
            "                      interpolated_windows_A=INTERPOLATED_WIN_A)\n"
            "print(f'error: modo={error_mode} (STAT {STAT_STATUS})')\n"
            "print(f'apcorr: modo={apcorr_mode} | mediana={float(np.nanmedian(apcorr)):.1f}'\n"
            "      f' -> la caja recoge el {100 / float(np.nanmedian(apcorr)):.2f}% de la PSF'\n"
            "      f' | norm_radius={norm_radius:g} px')\n"
            "print(f'canales marcados por flags: {int((flags != 0).sum())}')\n\n"
            "fig, ax = plt.subplots(figsize=(11, 3.4))\n"
            "ax.plot(WAVE, apcorr, lw=1.0, color='tab:blue')\n"
            "ax.set_xlabel('λ [Å]'); ax.set_ylabel('apcorr  [adimensional]', color='tab:blue')\n"
            "ax.tick_params(axis='y', labelcolor='tab:blue')\n"
            "# El mismo dato leído como fracción de PSF capturada, que es lo intuitivo.\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(WAVE, 100.0 / apcorr, lw=0.0)\n"
            "ax2.set_ylim(100.0 / np.array(ax.get_ylim())[::-1])\n"
            "ax2.set_ylabel('luz capturada por la caja  [%]', color='tab:grey')\n"
            "ax2.tick_params(axis='y', labelcolor='tab:grey')\n"
            "ax.set_title('corrección de apertura vs λ: sube al azul porque la PSF se ensancha'\n"
            "             ' (más luz fuera de la caja)', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 9 · Paso 5 — el espectro, binado con su error\n\n"
            "El flujo por canal es demasiado ruidoso para leerse, y una mediana móvil suaviza pero "
            "**no dice cuánto vale lo que se ve**. Aquí se **bina**: se agrupan `BIN_CANALES` "
            "canales, el flujo es la media y el error se propaga como gaussiano independiente,\n\n"
            "> σ_bin = √(Σ σᵢ²) / n\n\n"
            "que para σ constante es el conocido σ/√n. Así cada punto lleva su barra y se puede "
            "juzgar si el espectro está por encima de cero.\n\n"
            "> **Aviso que la propia cadena mide**: esa fórmula supone canales **independientes**, y "
            "no lo son. G1 midió `n_eff/n ≈ 0.43` (el remuestreo en λ correlacionó canales "
            "vecinos), así que el error binado gaussiano está **subestimado en ~√(1/0.43) ≈ 1.5×**. "
            "La celda dibuja las dos barras: la gaussiana y la corregida por `n_eff`."
        ),
        code(
            "BIN_CANALES = 25        # cámbialo y vuelve a ejecutar\n"
            "N_EFF_OVER_N = 0.43     # medido por G1 (docs/noise_model.md)\n\n"
            "flux          = raw_flux * apcorr\n"
            "flux_err      = raw_err * apcorr\n"
            "flux_err_emp  = raw_err_emp * apcorr\n"
            "flux_err_stat = None if raw_err_stat is None else raw_err_stat * apcorr\n\n"
            "def binea(wave, flux, err, n):\n"
            "    \"\"\"Media por bloques de n canales, con error gaussiano independiente.\"\"\"\n"
            "    n = int(n)\n"
            "    corte = (wave.size // n) * n\n"
            "    w = wave[:corte].reshape(-1, n)\n"
            "    f = flux[:corte].reshape(-1, n)\n"
            "    e = err[:corte].reshape(-1, n)\n"
            "    bueno = np.isfinite(f) & np.isfinite(e)\n"
            "    cuenta = bueno.sum(axis=1)\n"
            "    with np.errstate(invalid='ignore', divide='ignore'):\n"
            "        wb = np.nanmean(np.where(bueno, w, np.nan), axis=1)\n"
            "        fb = np.nansum(np.where(bueno, f, 0.0), axis=1) / np.maximum(cuenta, 1)\n"
            "        eb = np.sqrt(np.nansum(np.where(bueno, e, 0.0) ** 2, axis=1)) / np.maximum(cuenta, 1)\n"
            "    vacio = cuenta == 0\n"
            "    fb[vacio] = np.nan; eb[vacio] = np.nan\n"
            "    return wb, fb, eb, cuenta\n\n"
            "wb, fb, eb, cuenta = binea(WAVE, flux, flux_err_emp, BIN_CANALES)\n"
            "eb_corr = eb / np.sqrt(N_EFF_OVER_N)   # canales correlacionados (G1)\n"
            "rojo_b = (wb >= 7500) & (wb <= 9000)\n"
            "snr = np.abs(fb) / eb_corr\n"
            "print(f'binado de {BIN_CANALES} canales -> {np.isfinite(fb).sum()} puntos')\n"
            "print(f'  flujo mediano en 7500–9000 Å : {np.nanmedian(fb[rojo_b]):9.2f}')\n"
            "print(f'  error binado gaussiano       : {np.nanmedian(eb[rojo_b]):9.2f}')\n"
            "print(f'  corregido por n_eff/n={N_EFF_OVER_N:.2f}   : {np.nanmedian(eb_corr[rojo_b]):9.2f}'\n"
            "      f'  (×{1/np.sqrt(N_EFF_OVER_N):.2f})')\n"
            "print(f'  S/N mediano en esa banda     : {np.nanmedian(snr[rojo_b]):9.2f}')\n\n"
            "fig, ax = plt.subplots(figsize=(11, 4))\n"
            "ax.plot(WAVE, flux, lw=0.3, color='0.75', label='flujo por canal (crudo)')\n"
            "ax.errorbar(wb, fb, yerr=eb_corr, fmt='o', ms=3, lw=0.9, color='tab:blue',\n"
            "            ecolor='tab:blue', alpha=0.9,\n"
            "            label=f'binado {BIN_CANALES} ch, ±σ corregido por n_eff')\n"
            "ax.errorbar(wb, fb, yerr=eb, fmt='none', lw=1.8, ecolor='tab:orange', alpha=0.8,\n"
            "            label='±σ gaussiano (subestima: canales correlacionados)')\n"
            "ax.axhline(0, color='0.5', lw=0.7)\n"
            "ax.axvline(6563, color='tab:red', ls=':', label='Hα')\n"
            "ax.set_ylim(*np.nanpercentile(fb[np.isfinite(fb)], [1, 99]) * np.array([2.5, 2.5]))\n"
            "ax.set_xlabel('λ [Å]'); ax.set_ylabel('flujo (apcorr aplicada)'); ax.legend(fontsize=8)\n"
            "ax.set_title('C2 rehecho en el notebook, binado con su error', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 10 · Tres controles con geometría fija\n\n"
            "Los N controles del paso 3 los coloca el pipeline repartidos en ángulo. Aquí se miran "
            "**tres sitios elegidos a mano**, a la misma distancia de la primaria que el compañero: "
            "el **opuesto** (PA + 180°) y los dos **perpendiculares** (PA ± 90°). Se colocan con "
            "los helpers de B3, así que heredan su convención de PA y el `north_angle_deg` del run.\n\n"
            "Cada uno pasa por **exactamente el mismo proceso** que el compañero: misma caja, mismo "
            "fondo de anillo, misma `apcorr`. Ahí no hay ninguna fuente, así que lo que se vea es "
            "halo residual y ruido — y su dispersión dice cuánto cambia el fondo con el ángulo a "
            "esa misma separación."
        ),
        code(
            "from musepipe.stages.stage01c_localize import position_from_sep_pa\n\n"
            "astro = qc_b3.get('astrometry') or {}\n"
            "escala = qc_b3.get('pixel_scale_arcsec')\n"
            "norte = (qc_b3.get('wcs_orientation') or {}).get('north_angle_deg', 0.0)\n"
            "TRES = [('opuesto', 180.0), ('perpendicular +90', 90.0), ('perpendicular -90', -90.0)]\n"
            "fig, axes = plt.subplots(len(TRES), 1, figsize=(11, 2.5 * len(TRES)), sharex=True)\n"
            "for ax_i, (nombre, delta) in zip(np.atleast_1d(axes), TRES):\n"
            "    pos = position_from_sep_pa((float(STAR_YX[0]), float(STAR_YX[1])),\n"
            "                               float(astro['sep_arcsec']),\n"
            "                               float(astro['pa_deg']) + delta,\n"
            "                               float(escala), north_angle_deg=float(norte or 0.0))\n"
            "    f_c, npix_c = aperture_spectrum(CUBE, pos, APERTURE)\n"
            "    if ANNULUS is not None:\n"
            "        b_c = annulus_background_spectrum(CUBE, pos, ANNULUS[0], ANNULUS[1],\n"
            "                                         exclude_yx=STAR_YX,\n"
            "                                         exclude_radius=(ANNULUS[2] if len(ANNULUS) > 2 else 30.0))\n"
            "        f_c = f_c - b_c * npix_c\n"
            "    f_c = f_c * apcorr\n"
            "    wbc, fbc, ebc, _n = binea(WAVE, f_c, flux_err_emp, BIN_CANALES)\n"
            "    rb = (wbc >= 7500) & (wbc <= 9000)\n"
            "    for lo, hi in BAD_WINDOWS_A:\n"
            "        ax_i.axvspan(lo, hi, color='0.85', zorder=0)\n"
            "    ax_i.errorbar(wbc, fbc, yerr=ebc / np.sqrt(N_EFF_OVER_N), fmt='o', ms=2.5, lw=0.8,\n"
            "                  color='tab:green')\n"
            "    ax_i.axhline(0, color='0.5', lw=0.7)\n"
            "    pa = (float(astro['pa_deg']) + delta) % 360.0\n"
            "    ax_i.set_ylabel(f'{nombre}\\nPA {pa:.0f}°', fontsize=8)\n"
            "    print(f'{nombre:18s} PA {pa:6.1f}°  yx={[round(v,1) for v in pos]}'\n"
            "          f'  mediana rojo = {np.nanmedian(fbc[rb]):9.2f}')\n"
            "np.atleast_1d(axes)[-1].set_xlabel('λ [Å]')\n"
            "np.atleast_1d(axes)[0].set_title('el mismo proceso donde NO hay compañero'\n"
            "                                 ' (mismo radio, tres ángulos)', fontsize=9)\n"
            "fig.tight_layout(); plt.show()\n"
            "print()\n"
            "print('compañero, para comparar   mediana rojo =',\n"
            "      f'{np.nanmedian(fb[(wb >= 7500) & (wb <= 9000)]):9.2f}')"
        ),
        md(
            "## 11 · Validación externa contra `photutils` (opcional)\n\n"
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
            "    # La curva por canal es RUIDO: se bina igual que el espectro, con su\n"
            "    # error, para que se vea que la mediana de banda es lo que manda.\n"
            "    paso_bin = max(1, len(canales) // 40)\n"
            "    wb_c, fb_c, eb_c, _n = binea(WAVE[canales], dif_centro,\n"
            "                                 np.abs(dif_centro) * 0 + np.nanstd(dif_centro), paso_bin)\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)\n"
            "    ax1.plot(WAVE[canales], dif_centro, lw=0.5, color='0.75', label='por canal (crudo)')\n"
            "    ax1.errorbar(wb_c, fb_c, yerr=eb_c, fmt='o', ms=3, lw=0.9, color='tab:purple',\n"
            "                 label=f'binado {paso_bin} puntos')\n"
            "    ax1.legend(fontsize=7)\n"
            "    ax1.axhline(0, color='0.7', lw=0.6)\n"
            "    # Umbral de interpretacion: por encima del 10% la diferencia ya no es\n"
            "    # un detalle de centrado, es una discrepancia que hay que explicar.\n"
            "    for lim in (-10, 10):\n"
            "        ax1.axhline(lim, color='tab:red', ls=':', lw=1.1,\n"
            "                    label='±10% (demasiado alto)' if lim > 0 else None)\n"
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
            "## 12 · Comparación con la cadena\n\n"
            "Contra `spec_aperture_object.fits`, el producto que escribió la etapa. **Con las "
            "perillas por defecto debe salir idéntico** (a precisión de coma flotante): si no lo "
            "es, o la copia se desvió o alguna entrada no es la que usó la cadena. En cuanto "
            "cambias una perilla, esta celda mide exactamente qué se movió."
        ),
        code(
            "from musepipe.extraction.product import SpectrumProduct\n"
            "from musepipe.spectral import median_filter_1d\n\n"
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
            "## 13 · Y contra el QC\n\n"
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
        md(
            "## 14 · Por qué el continuo se va a negativo en el azul\n\n"
            "En algunos objetos el continuo del extremo azul queda **por debajo de cero** en la "
            "figura anterior. La primera sospecha razonable es un error de formulación: que en vez "
            "de multiplicar por `apcorr` se estuviera restando algo. **No es eso**, y esta celda "
            "lo comprueba elemento a elemento:\n\n"
            "> `producto = (crudo − fondo × npix_eff) × apcorr`\n\n"
            "Cuando pasa, lo que se lee en la tabla de abajo son dos cosas encadenadas:\n\n"
            "1. **En el azul el anillo mide más que la caja.** El fondo por píxel del anillo es "
            "*mayor* que el flujo por píxel dentro de la caja, así que `crudo − fondo × npix_eff` "
            "sale **negativo** antes de tocar la corrección de apertura. El halo AO de la "
            "primaria es **cromático** (mucho peor en el azul) y **no es plano**: a la distancia "
            "del compañero cae con el radio, así que la mediana del anillo —que abarca radios "
            "mayores y menores— no representa el fondo justo debajo de la caja.\n"
            "2. **`apcorr` amplifica ese negativo.** En el azul la PSF es más ancha, la caja "
            "recoge menos luz y la corrección es varias veces la del rojo: multiplica el signo "
            "negativo por un factor grande y lo convierte en un continuo negativo llamativo.\n\n"
            "La celda mide los tres fondos por píxel —caja, anillo y halo de la primaria a la "
            "misma distancia— así que el diagnóstico sale **para este objeto**, no heredado de "
            "otro: en unos el anillo queda por encima de la caja y el continuo azul se va a "
            "negativo, en otros no.\n\n"
            "O sea: el signo viene del **modelo de fondo**, no de la aritmética. Es de la misma "
            "familia que el residuo de halo AO documentado para C3/psfsub, y es el motivo por el "
            "que el nivel absoluto en el azul no se cita sin la calibración de D2."
        ),
        code(
            "BANDA_AZUL = (4800.0, 5600.0)   # cámbiala y vuelve a ejecutar\n\n"
            "sel = (WAVE >= BANDA_AZUL[0]) & (WAVE <= BANDA_AZUL[1]) & (flags == 0)\n"
            "crudo = raw_flux + (bkg * npix_eff if ANNULUS is not None else 0.0)\n"
            "resta = crudo - raw_flux\n"
            "med = lambda v: float(np.nanmedian(np.asarray(v, dtype=float)[sel]))\n"
            "print(f'banda {BANDA_AZUL[0]:.0f}-{BANDA_AZUL[1]:.0f} Å, medianas por canal:')\n"
            "print(f'  crudo (suma de la caja)      : {med(crudo):10.2f}')\n"
            "print(f'  fondo × npix_eff             : {med(resta):10.2f}')\n"
            "marca = '   <-- ya negativo ANTES de apcorr' if med(raw_flux) < 0 else ''\n"
            "print(f'  crudo − fondo × npix_eff     : {med(raw_flux):10.2f}{marca}')\n"
            "print(f'  apcorr                       : {med(apcorr):10.2f}')\n"
            "print(f'  producto                     : {med(flux):10.2f}')\n"
            "# La comprobación literal: no hay ninguna resta escondida en el último paso.\n"
            "esperado = raw_flux * apcorr\n"
            "print(f'\\n¿producto == (crudo − fondo·npix) × apcorr, bit a bit?  '\n"
            "      f'{np.array_equal(flux, esperado, equal_nan=True)}')\n\n"
            "# Y de dónde sale el signo: el fondo POR PÍXEL, en tres sitios.\n"
            "img = np.nanmedian(CUBE[sel], axis=0)\n"
            "yy, xx = np.indices(img.shape, dtype=float)\n"
            "rr_obj = np.hypot(yy - OBJECT_YX[0], xx - OBJECT_YX[1])\n"
            "rr_star = np.hypot(yy - STAR_YX[0], xx - STAR_YX[1])\n"
            "r_comp = float(np.hypot(OBJECT_YX[0] - STAR_YX[0], OBJECT_YX[1] - STAR_YX[1]))\n"
            "en_caja = rr_obj <= 1.5\n"
            "if ANNULUS is not None:\n"
            "    en_anillo = ((rr_obj >= ANNULUS[0]) & (rr_obj <= ANNULUS[1])\n"
            "                 & (rr_star > (ANNULUS[2] if len(ANNULUS) > 2 else 30.0)))\n"
            "else:\n"
            "    en_anillo = np.zeros_like(en_caja)\n"
            "# El halo de la primaria a la MISMA distancia que el compañero, mirando\n"
            "# alrededor: es la referencia justa, y la que el anillo no reproduce.\n"
            "en_halo = (np.abs(rr_star - r_comp) <= 1.5) & (rr_obj > 6.0)\n"
            "print('\\nfondo por píxel en esa banda:')\n"
            "for nombre, mascara in (('dentro de la caja', en_caja),\n"
            "                        ('anillo del fondo', en_anillo),\n"
            "                        ('halo a la misma distancia de la primaria', en_halo)):\n"
            "    if mascara.any():\n"
            "        print(f'  {nombre:42s}: {float(np.nanmedian(img[mascara])):7.3f}')\n"
            "if med(raw_flux) < 0:\n"
            "    print('  -> el anillo mide MÁS que la caja: la resta deja el continuo negativo,')\n"
            "    print('     y apcorr (grande en el azul) lo amplifica. No hay error de fórmula.')\n"
            "else:\n"
            "    print('  -> aquí la caja queda por encima del anillo: en este objeto el continuo')\n"
            "    print('     azul NO se va a negativo. El mecanismo es el mismo, el signo no.')"
        ),
        md(
            mb.PAPER_SPECTRUM_MD
            + "\n\n> **Aquí sale de TUS números**, los recalculados arriba, no del producto de "
            "la cadena: si has tocado una perilla, la figura y la tabla llevan ese cambio. "
            "Por eso se escriben con el sufijo `_debug`, en `plots/c2_aperture_debug/` y "
            "`tables/…_debug.ecsv`, y no pisan lo que exporta el notebook de auditoría."
        ),
        code(
            mb.paper_spectrum_cell(
                arrays_code=(
                    "    ROOT_P = ROOT\n"
                    "    METHOD_P = 'aperture'\n"
                    "    PRODUCT_P = 'recalculado en C2_aperture_debug (no leído de disco)'\n"
                    "    TARGET_P = str(TARGET).replace(' ', '') + '_debug'\n"
                    "    BUNIT_P = BUNIT or 'ADU'\n"
                    "    W_P, F_P = WAVE, flux\n"
                    "    # La banda es la empírica; la línea, el STAT propagado —\n"
                    "    # calculado exista o no el modo `stat`, para que las dos\n"
                    "    # curvas salgan en todos los objetos.\n"
                    "    E_P, E_ALT_P = flux_err_emp, flux_err_stat\n"
                    "    if not np.isfinite(E_P).any():\n"
                    "        E_P, E_ALT_P = flux_err, None\n"
                    "    EXTRA_P = {'flux_err_stat': flux_err_stat, 'apcorr': apcorr,\n"
                    "               'npix_eff': npix_eff, 'flags': flags}\n"
                    "    MODO_P = error_mode\n"
                ),
                subdir="c2_aperture_debug",
                err_label="±1σ empírico (controles procesados igual)",
                err_alt_label="±1σ propagado del STAT (no es σ)",
                title_suffix="apertura box3, rehecha en el notebook (C2 debug)",
            )
        ),
        md(
            "## 16 · Prueba — ¿y si `apcorr` fuera constante?\n\n"
            "Una prueba, **no un cambio de la cadena**: se rehace el último paso con una "
            "corrección de apertura **plana**, `apcorr = 40` en todo el rango, en vez de la "
            "curva cromática que sale de la PSF de C1. La celda de comparación de arriba sigue "
            "midiendo contra la cadena con las perillas por defecto; esto vive aparte y escribe "
            "sus propios ficheros con sufijo `_apcorr40`.\n\n"
            "**Qué se está suponiendo.** `apcorr = 1/f` con `f` la fracción de la PSF que cae "
            "dentro de la caja; ponerla constante equivale a decir que **esa fracción no depende "
            "de λ**. Es falso —la PSF AO se ensancha hacia el azul, así que la caja captura menos "
            "y la corrección real sube— pero es exactamente la prueba que hace falta para separar "
            "dos cosas que se confunden en la figura anterior:\n\n"
            "- lo que en el espectro es **del compañero**, y\n"
            "- lo que es **de la corrección**: una curva que va de ~118× a ~21× multiplica el azul "
            "por casi seis veces más que el rojo, y cualquier error en el modelo de PSF entra "
            "amplificado y **con pendiente**.\n\n"
            "**Cómo leerlo.** Con `apcorr` plana el espectro es el crudo reescalado: conserva "
            "la forma medida, y la pendiente azul→rojo que se ve **es la del dato**. La diferencia "
            "entre las dos curvas es, punto por punto, lo que la corrección cromática está "
            "añadiendo a la SED. Si el continuo azul negativo se atenúa mucho al aplanar `apcorr`, "
            "confirma lo de la celda 14: el signo lo pone el fondo, pero **el tamaño lo pone la "
            "corrección**.\n\n"
            "> Esto **no** valida `apcorr = 40` como alternativa: la corrección cromática es la "
            "física correcta y es la que usa la cadena. Es un banco de pruebas para ver cuánto "
            "del resultado depende de ella."
        ),
        code(
            "from musepipe.paper_spectrum import (paper_spectrum_figure, pretty_flux_unit,\n"
            "                                     spectrum_table_meta, write_spectrum_table)\n"
            "from musepipe.telluric_lines import measured_transmission\n"
            "from musepipe.spectral import median_filter_1d\n\n"
            "APCORR_CONST = 40.0                 # la perilla de esta prueba\n"
            "AZUL, ROJO = (4800.0, 5600.0), (7500.0, 9000.0)\n\n"
            "flux_const = raw_flux * APCORR_CONST\n"
            "err_const  = raw_err_emp * APCORR_CONST    # el empírico, que es el que manda\n\n"
            "def _med(v, banda):\n"
            "    sel = (WAVE >= banda[0]) & (WAVE <= banda[1]) & (flags == 0)\n"
            "    return float(np.nanmedian(np.asarray(v, dtype=float)[sel]))\n\n"
            "print(f'apcorr de la cadena : mediana {float(np.nanmedian(apcorr)):6.1f}'\n"
            "      f' | azul {_med(apcorr, AZUL):6.1f} -> rojo {_med(apcorr, ROJO):6.1f}'\n"
            "      f'  (razón azul/rojo = {_med(apcorr, AZUL) / _med(apcorr, ROJO):.1f}×)')\n"
            "print(f'apcorr de la prueba : {APCORR_CONST:6.1f} en todo el rango (razón 1.0×)')\n"
            "print()\n"
            "cab = f\"{'banda':22s} {'cromática':>12s} {'constante':>12s} {'const/crom':>11s}\"\n"
            "print(cab); print('-' * len(cab))\n"
            "for nombre, banda in (('azul %.0f-%.0f Å' % AZUL, AZUL),\n"
            "                      ('rojo %.0f-%.0f Å' % ROJO, ROJO)):\n"
            "    a, b = _med(flux, banda), _med(flux_const, banda)\n"
            "    razon = (b / a) if a not in (0.0,) and np.isfinite(a) else float('nan')\n"
            "    print(f'{nombre:22s} {a:12.1f} {b:12.1f} {razon:11.2f}')\n"
            "# La pendiente de la SED: cuánto sube el continuo del azul al rojo en cada caso.\n"
            "for etiqueta, v in (('cromática', flux), ('constante', flux_const)):\n"
            "    azul_v, rojo_v = _med(v, AZUL), _med(v, ROJO)\n"
            "    if azul_v > 0:\n"
            "        print(f'  pendiente rojo/azul ({etiqueta:9s}): {rojo_v / azul_v:7.2f}×')\n"
            "    else:\n"
            "        # Con el continuo azul negativo el cociente no es una pendiente:\n"
            "        # cambia de signo y crece sin límite cerca del cero.\n"
            "        print(f'  pendiente rojo/azul ({etiqueta:9s}): no se puede leer,'\n"
            "              f' el continuo azul es negativo ({azul_v:.1f})')\n\n"
            "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,\n"
            "                               gridspec_kw={'height_ratios': [1, 2]})\n"
            "ax1.plot(WAVE, apcorr, lw=0.9, color='tab:purple', label='apcorr de la cadena (PSF de C1)')\n"
            "ax1.axhline(APCORR_CONST, color='tab:brown', ls='--', lw=1.2,\n"
            "            label=f'apcorr = {APCORR_CONST:g} (prueba)')\n"
            "ax1.set_ylabel('apcorr [adim.]'); ax1.legend(fontsize=8)\n"
            "ax1.set_title('la corrección de apertura, y la misma puesta plana', fontsize=9)\n"
            "ax2.axhline(0, color='0.6', lw=0.7)\n"
            "ax2.plot(WAVE, median_filter_1d(flux, 41), lw=1.2, color='tab:blue',\n"
            "         label='espectro con apcorr cromática (= la cadena)')\n"
            "ax2.plot(WAVE, median_filter_1d(flux_const, 41), lw=1.2, color='tab:brown',\n"
            "         label=f'espectro con apcorr = {APCORR_CONST:g}')\n"
            "ax2.axvline(6562.8, color='tab:red', ls=':', lw=0.8, label='Hα')\n"
            "amb = np.concatenate([median_filter_1d(flux, 41), median_filter_1d(flux_const, 41)])\n"
            "ax2.set_ylim(*np.nanpercentile(amb[np.isfinite(amb)], [1, 99]))\n"
            "ax2.set_xlabel('λ [Å]'); ax2.set_ylabel('flujo [' + pretty_flux_unit(BUNIT) + ']')\n"
            "ax2.legend(fontsize=8)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "### La misma figura de paper, con la corrección plana\n\n"
            "Se dibuja y **se exporta** igual que la del apartado anterior, para poder ponerlas "
            "una al lado de otra. Ficheros con sufijo `_apcorr40`: no pisan nada."
        ),
        code(
            "MALOS_P = (np.asarray(flags, dtype=int) & FLAG_BAD_WINDOW) != 0\n"
            "trans = measured_transmission(RUN_ID, project_root=ROOT)\n"
            "fig, _ejes = paper_spectrum_figure(\n"
            "    WAVE, flux_const, err_const, bad_channels=MALOS_P, transmission=trans,\n"
            "    err_label='±1σ empírico × apcorr constante',\n"
            "    title=nb.display_name(RUN_ID) + ' · apertura box3 con apcorr = '\n"
            "          + format(APCORR_CONST, 'g') + ' (prueba, C2 debug)',\n"
            "    flux_label='flujo [' + pretty_flux_unit(BUNIT) + ']')\n"
            "outdir = nb.run_dir(RUN_ID) / 'plots' / 'c2_aperture_debug'\n"
            "outdir.mkdir(parents=True, exist_ok=True)\n"
            "DPI_P = 300\n"
            "for ext in ('png', 'pdf'):\n"
            "    fig.savefig(outdir / ('spectrum_paper_apcorr40.' + ext), dpi=DPI_P)\n"
            "objetivo = (nb.run_dir(RUN_ID) / 'tables'\n"
            "            / ('spec_aperture_' + str(TARGET).replace(' ', '') + '_apcorr40.ecsv'))\n"
            "tabla = write_spectrum_table(\n"
            "    objetivo, WAVE, flux_const, err_const,\n"
            "    extra_columns={'apcorr': np.full_like(WAVE, APCORR_CONST),\n"
            "                   'npix_eff': npix_eff, 'flags': flags},\n"
            "    units={'flux': BUNIT, 'flux_err': BUNIT},\n"
            "    meta=spectrum_table_meta(\n"
            "        run_id=RUN_ID, target=str(TARGET), method='aperture',\n"
            "        product='prueba en C2_aperture_debug: apcorr constante',\n"
            "        flux_unit=BUNIT, error_mode='empirical',\n"
            "        extra={'apcorr_mode': 'constante (prueba, NO es la cadena)',\n"
            "               'apcorr_value': float(APCORR_CONST),\n"
            "               'figure': str(outdir / 'spectrum_paper_apcorr40.pdf')}))\n"
            "print('figura ->', outdir / 'spectrum_paper_apcorr40.pdf')\n"
            "print('tabla  ->', tabla)\n"
            "print('   la meta dice que apcorr es constante: quien reciba el fichero suelto')\n"
            "print('   no puede confundirlo con el producto de la cadena.')\n"
            "plt.show()"
        ),
        md(mb.STAR_REFERENCE_MD),
        code(
            mb.paper_from_product_cell(
                product="spec_psffit_star.fits",
                method="psffit_star",
                subdir="c2_aperture_debug",
                stem="spectrum_paper_star_ref",
                qc="stages/spec_psffit_qc.json",
                title_suffix="espectro de la PRIMARIA (producto de C4, referencia)",
            )
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
            "# Resolución de las figuras EN PANTALLA. `savefig` guarda a 300 dpi, pero\n"
            "# lo que se ve dentro del notebook lo fija el backend inline, que va a 100\n"
            "# dpi por defecto y sale borroso. `retina` dobla los píxeles sin cambiar el\n"
            "# tamaño aparente; fuera de IPython no hace nada y queda el rcParam.\n"
            "import matplotlib as mpl\n"
            "mpl.rcParams['figure.dpi'] = 120\n"
            "mpl.rcParams['savefig.dpi'] = 200\n"
            "try:\n"
            "    from matplotlib_inline.backend_inline import set_matplotlib_formats\n"
            "    set_matplotlib_formats('retina')\n"
            "except Exception:\n"
            "    pass\n"
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
            "CUBE_PATH = SD / 'stage02_xcorr_cube_stack.fits'\n"
            "with fits.open(CUBE_PATH) as h:\n"
            "    STAGE02 = np.asarray(h['CUBES'].data, dtype=float)\n"
            "    WAVE = np.asarray(h['WAVELENGTH'].data, dtype=float)\n"
            "    STAT_CUBE = np.asarray(h['STAT'].data, dtype=float) if 'STAT' in h else None\n"
            "    _stack_bunit = str(h[0].header.get('BUNIT', '')\n"
            "                       or h['CUBES'].header.get('BUNIT', '')) or None\n"
            "# La unidad, con la MISMA regla que la cadena: este stack no lleva BUNIT\n"
            "# (se escribió antes de que B1/B2 lo propagaran), así que `resolve_bunit`\n"
            "# cae al cubo de entrada del run. Sin esto las colorbars no tienen unidad.\n"
            "from musepipe.io import resolve_bunit\n"
            "BUNIT = resolve_bunit(X02, stack_bunit=_stack_bunit)\n"
            "UNIDAD = BUNIT or 'sin unidad declarada'\n"
            "if STAGE02.ndim == 4:\n"
            "    STAGE02 = STAGE02[0]\n"
            "if STAT_CUBE is not None and STAT_CUBE.ndim == 4:\n"
            "    STAT_CUBE = STAT_CUBE[0]\n"
            "LS_04B = np.asarray(fits.getdata(SD / 'cube_residual_local_object.fits'), dtype=float)\n"
            "assert LS_04B.shape == STAGE02.shape, (LS_04B.shape, STAGE02.shape)\n"
            "# Wings-intact (2026-07-26): con anillo configurado, LS extrae del cubo CRUDO,\n"
            "# igual que C2. Antes salia del residual de 04b y encima se le restaba el\n"
            "# anillo: dos fondos sobre un cubo que no es homogeneo (04b solo resta\n"
            "# alrededor del objeto). Ver reports/20260726/auditoria_c3_doble_sustraccion.\n"
            "WINGS_INTACT_LS = bool(X02.get('x02_wings_intact_ls', True))\n"
            "LS_CUBE = (STAGE02 if (WINGS_INTACT_LS and LOCAL_BKG_ANNULUS_PX is not None)\n"
            "           else LS_04B)\n"
            "print('cubo de ls:', 'crudo de B2 (wings-intact)' if LS_CUBE is STAGE02\n"
            "      else 'residual de 04b (historico)')\n\n"
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
            f"print(f'STAT     : factor={{STAT_FACTOR:.3f}} covarianza={{COV_FACTOR:.3f}} estado={{STAT_STATUS}}')\n"
            "print('BUNIT    :', UNIDAD)"
        ),
        md(
            "## 3 · De dónde sale el espectro\n\n"
            "Antes de nada, **dónde** se mide. A diferencia de C2, que suma una caja 3×3, aquí la "
            "extracción usa una **ventana circular** de `WINDOW_RADIUS_PX` px alrededor del "
            "compañero: no es una apertura que se sume: es el conjunto de píxeles que entran en "
            "el estimador, cada uno con su peso.\n\n"
            "Las dos imágenes son el **mismo** cubo (la mediana en λ) con **dos escalas**, porque "
            "una sola no puede enseñar las dos cosas:\n\n"
            "- **logarítmica**: enseña el halo AO de la primaria, que es el fondo contra el que "
            "hay que medir y lo que las dos variantes tratan de forma distinta;\n"
            "- **lineal recortada al entorno del compañero**: enseña la fuente, que en log queda "
            "aplastada contra el halo.\n\n"
            "Marcado encima: la **ventana de extracción** (círculo continuo), el **anillo de "
            "fondo** cuando está configurado (dos círculos punteados) con el **disco excluido** "
            "alrededor de la primaria, y las dos posiciones que vienen de B3."
        ),
        code(
            "from matplotlib.colors import LogNorm\n"
            "from matplotlib.patches import Circle\n\n"
            "# La mediana se toma en la BANDA ROJA, no en todo el rango: es donde el\n"
            "# compañero se detecta. Promediando los 3681 canales manda el azul, que\n"
            "# es casi todo halo y ruido, y la fuente no asoma ni con el halo quitado.\n"
            "BANDA_IMAGEN = (7500.0, 9000.0)     # cámbiala y vuelve a ejecutar\n"
            "_ch = (WAVE >= BANDA_IMAGEN[0]) & (WAVE <= BANDA_IMAGEN[1])\n"
            "img_med = np.nanmedian(LS_CUBE[_ch], axis=0)   # el cubo del que extrae `ls`\n"
            "# El recorte enmarca a las DOS fuentes: el disco excluido alrededor de\n"
            "# la primaria es parte de lo que se está explicando, y con un recorte\n"
            "# centrado en el compañero se dibujaba fuera de la imagen.\n"
            "MARGEN_PX = 16.0\n"
            "_ny, _nx = img_med.shape\n"
            "sl = (slice(int(max(0, min(STAR_YX[0], OBJECT_YX[0]) - MARGEN_PX)),\n"
            "            int(min(_ny, max(STAR_YX[0], OBJECT_YX[0]) + MARGEN_PX + 1))),\n"
            "      slice(int(max(0, min(STAR_YX[1], OBJECT_YX[1]) - MARGEN_PX)),\n"
            "            int(min(_nx, max(STAR_YX[1], OBJECT_YX[1]) + MARGEN_PX + 1))))\n"
            "y0, x0 = sl[0].start, sl[1].start\n"
            "rec = img_med[sl]\n\n"
            "def _marcas(ax):\n"
            "    ax.add_patch(Circle((OBJECT_YX[1] - x0, OBJECT_YX[0] - y0), WINDOW_RADIUS_PX,\n"
            "                        fill=False, color='tab:cyan', lw=1.4))\n"
            "    if LOCAL_BKG_ANNULUS_PX is not None:\n"
            "        for r in LOCAL_BKG_ANNULUS_PX[:2]:\n"
            "            ax.add_patch(Circle((OBJECT_YX[1] - x0, OBJECT_YX[0] - y0), float(r),\n"
            "                                fill=False, color='tab:orange', lw=0.9, ls=':'))\n"
            "        r_ex = float(LOCAL_BKG_ANNULUS_PX[2]) if len(LOCAL_BKG_ANNULUS_PX) > 2 else 30.0\n"
            "        ax.add_patch(Circle((STAR_YX[1] - x0, STAR_YX[0] - y0), r_ex,\n"
            "                            fill=False, color='tab:red', lw=0.9, ls='--'))\n"
            "    ax.plot(STAR_YX[1] - x0, STAR_YX[0] - y0, '*', color='w', ms=11, mec='k')\n"
            "    ax.plot(OBJECT_YX[1] - x0, OBJECT_YX[0] - y0, '+', color='tab:cyan', ms=9)\n\n"
            "# Tercera vista, SOLO PARA VER: se le quita al campo el perfil radial\n"
            "# mediano alrededor de la primaria. El halo AO es casi simétrico en\n"
            "# azimut, así que al restarlo lo que sobresale es el compañero. No\n"
            "# entra en ninguna cuenta — es lo que hacen C5/C6, no C3.\n"
            "yy_f, xx_f = np.indices(img_med.shape, dtype=float)\n"
            "r_bin = np.hypot(yy_f - STAR_YX[0], xx_f - STAR_YX[1]).astype(int)\n"
            "perfil = np.full(int(r_bin.max()) + 1, np.nan)\n"
            "for b in range(perfil.size):\n"
            "    m_ = r_bin == b\n"
            "    if m_.any():\n"
            "        perfil[b] = np.nanmedian(img_med[m_])\n"
            "sin_halo = (img_med - perfil[r_bin])[sl]\n\n"
            "fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.0))\n"
            "# Escala LOG sobre los positivos: el halo abarca varios órdenes de magnitud.\n"
            "pos = rec[np.isfinite(rec) & (rec > 0)]\n"
            "vmin = float(np.nanpercentile(pos, 30)) if pos.size else 1e-3\n"
            "vmax = float(np.nanpercentile(pos, 99.9)) if pos.size else 1.0\n"
            "im1 = a1.imshow(rec, origin='lower', cmap='magma',\n"
            "                norm=LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, vmin * 10)))\n"
            "a1.set_title(f'mediana {BANDA_IMAGEN[0]:.0f}-{BANDA_IMAGEN[1]:.0f} Å · LOG (el halo de la primaria)', fontsize=9)\n"
            "cb1 = fig.colorbar(im1, ax=a1, fraction=0.046)\n"
            "cb1.set_label(f'flujo mediano [{UNIDAD}]', fontsize=7)\n"
            "# Escala LINEAL acotada por lo que hay CERCA del compañero, no por la\n"
            "# primaria: si no, el compañero es un píxel indistinguible del fondo.\n"
            "yy_, xx_ = np.indices(rec.shape, dtype=float)\n"
            "cerca = np.hypot(yy_ - (OBJECT_YX[0] - y0), xx_ - (OBJECT_YX[1] - x0)) <= 2 * WINDOW_RADIUS_PX\n"
            "lo, hi = np.nanpercentile(rec[cerca & np.isfinite(rec)], [5, 99.5])\n"
            "im2 = a2.imshow(rec, origin='lower', cmap='viridis', vmin=lo, vmax=hi)\n"
            "a2.set_title('zoom al compañero · escala LINEAL a su entorno', fontsize=9)\n"
            "cb2 = fig.colorbar(im2, ax=a2, fraction=0.046)\n"
            "cb2.set_label(f'flujo mediano [{UNIDAD}]', fontsize=7)\n"
            "v3 = float(np.nanpercentile(np.abs(sin_halo[cerca]), 98))\n"
            "im3 = a3.imshow(sin_halo, origin='lower', cmap='RdBu_r', vmin=-v3, vmax=v3)\n"
            "a3.set_title('zoom · perfil radial del halo restado (solo para ver)', fontsize=9)\n"
            "cb3 = fig.colorbar(im3, ax=a3, fraction=0.046, extend='both')\n"
            "cb3.set_label(f'flujo − halo azimutal [{UNIDAD}]', fontsize=7)\n"
            "for ax in (a1, a2, a3):\n"
            "    _marcas(ax)\n"
            "    # Que un círculo grande no estire los ejes y encoja la imagen.\n"
            "    ax.set_xlim(-0.5, rec.shape[1] - 0.5); ax.set_ylim(-0.5, rec.shape[0] - 0.5)\n"
            "    ax.set_xlabel('x [px]'); ax.set_ylabel('y [px]')\n"
            "# El panel derecho hace zoom: a esta separación el compañero ocupa unos\n"
            "# pocos píxeles y en el campo entero no se distingue del fondo.\n"
            "_zoom = 2.6 * float(WINDOW_RADIUS_PX)\n"
            "for ax in (a2, a3):\n"
            "    ax.set_xlim(OBJECT_YX[1] - x0 - _zoom, OBJECT_YX[1] - x0 + _zoom)\n"
            "    ax.set_ylim(OBJECT_YX[0] - y0 - _zoom, OBJECT_YX[0] - y0 + _zoom)\n"
            "# Cuánto sobresale el compañero una vez quitado el halo azimutal.\n"
            "_en_win = (np.hypot(yy_ - (OBJECT_YX[0] - y0), xx_ - (OBJECT_YX[1] - x0))\n"
            "           <= float(WINDOW_RADIUS_PX))\n"
            "print(f'con el halo azimutal quitado, en la ventana: mediana'\n"
            "      f' {float(np.nanmedian(sin_halo[_en_win])):7.2f}, pico'\n"
            "      f' {float(np.nanmax(sin_halo[_en_win])):7.2f} {UNIDAD}')\n"
            "fig.suptitle('dónde se extrae: ventana (cian), anillo de fondo (naranja punteado),'\n"
            "             ' disco excluido de la primaria (rojo)', fontsize=9)\n"
            "fig.tight_layout(); plt.show()\n\n"
            "# El mismo criterio que `circular_window_indices` (que se copia más\n"
            "# abajo): distancia al centro <= radio. Aquí a mano para no depender\n"
            "# de una celda posterior.\n"
            "yy_g, xx_g = np.indices(LS_CUBE.shape[1:], dtype=float)\n"
            "n_win = int((np.hypot(yy_g - OBJECT_YX[0], xx_g - OBJECT_YX[1])\n"
            "             <= float(WINDOW_RADIUS_PX)).sum())\n"
            "print(f'ventana de extracción: círculo de r={float(WINDOW_RADIUS_PX):g} px'\n"
            "      f' -> {n_win} píxeles (una caja 3×3 tendría 9)')\n"
            "sep = float(np.hypot(OBJECT_YX[0] - STAR_YX[0], OBJECT_YX[1] - STAR_YX[1]))\n"
            "print(f'separación compañero-primaria: {sep:.1f} px')\n"
            "if LOCAL_BKG_ANNULUS_PX is not None:\n"
            "    print(f'anillo de fondo: {LOCAL_BKG_ANNULUS_PX[0]:g}-{LOCAL_BKG_ANNULUS_PX[1]:g} px'\n"
            "          f' | excluye r<{(LOCAL_BKG_ANNULUS_PX[2] if len(LOCAL_BKG_ANNULUS_PX) > 2 else 30.0):g} px'\n"
            "          ' de la primaria')\n"
            "else:\n"
            "    print('sin anillo de fondo configurado: `ls` extrae del residual de 04b')"
        ),
        md(
            '### El mismo residuo, resuelto en 8 bandas\n\n'
            'La vista de arriba es una **mediana de la banda** `BANDA_IMAGEN` (7500–9000 Å por defecto), no luz blanca ni un canal suelto: colapsa ~1200 canales en una imagen. Eso hace visible al compañero, pero **promedia toda la estructura cromática** del halo, que es justo lo que se quiere mirar cuando la sospecha está en la resta.\n\n'
            'Aquí se repite la operación —quitar el perfil radial mediano alrededor de la primaria— en **8 bandas** que cubren el rango completo. Cada panel sigue siendo una integración de ~460 canales, así que el ruido no domina, pero ahora se ve **cómo cambia con λ** lo que queda tras quitar el halo simétrico.\n\n'
            'Qué buscar, y qué significaría:\n\n'
            '- **Estructura fija que no cambia con λ** (una mancha, un borde): algo del detector o del combinado; no es halo.\n\n'
            '- **Estructura que se mueve o se ensancha hacia el azul**: speckles de la AO. El halo no es liso ni simétrico, y ahí es donde una mediana de anillo deja de representar el fondo bajo el compañero.\n\n'
            '- **Un gradiente que crece hacia el azul en la posición del compañero**: la firma de la sobre-sustracción que se ve en el espectro.\n\n'
            '> Sigue siendo **solo para ver**: no entra en ninguna cuenta de la extracción.\n\n'
            '**Constancia de por qué el fondo se pasa.** El anillo (8–14 px alrededor del compañero) barre radios *estelares* de ~57 a ~85 px y el halo cae un factor ~1.6 cada 10 px ahí. La intuición dice que el lado interior, más brillante, tira la mediana hacia arriba — **y es falsa**: la curvatura del arco mete más área por fuera, así que la mediana del radio estelar dentro del anillo cae 0.7 px MÁS LEJOS de la estrella y el anillo **sub**-estima el halo. Lo que de verdad pasa es que el compañero **está en un mínimo local**: medido en la banda azul, por píxel, caja 4.126, anillo 3–6 px 4.600, anillo 8–14 px 4.280, azimutal al mismo radio estelar 5.111. Cualquier promedio de su entorno le quita de más, y esa diferencia se resta en **cada uno** de los ~200 píxeles de la ventana antes de multiplicar por `apcorr`. Registrado, con la errata, en [`reports/20260727/sesgo_anillo_y_ventana_2026-07-27.md`](../../../reports/20260727/sesgo_anillo_y_ventana_2026-07-27.md).\n\n'
        ),
        code(
            'N_BANDAS = 8\n'
            'bordes = np.linspace(float(np.nanmin(WAVE)), float(np.nanmax(WAVE)), N_BANDAS + 1)\n'
            '\n'
            'def quita_halo(img):\n'
            '    """Le resta a la imagen el perfil radial mediano alrededor de la primaria."""\n'
            '    perfil_r = np.full(int(r_bin.max()) + 1, np.nan)\n'
            '    for b in range(perfil_r.size):\n'
            '        m_ = r_bin == b\n'
            '        if m_.any():\n'
            '            perfil_r[b] = np.nanmedian(img[m_])\n'
            '    return img - perfil_r[r_bin]\n'
            '\n'
            'fig, ejes = plt.subplots(4, 2, figsize=(9.5, 15))\n'
            'for k, ax in enumerate(ejes.ravel()):\n'
            '    lo_k, hi_k = bordes[k], bordes[k + 1]\n'
            '    sel_k = (WAVE >= lo_k) & (WAVE < hi_k)\n'
            '    n_ch = int(sel_k.sum())\n'
            '    if n_ch == 0:\n'
            '        ax.set_axis_off()\n'
            '        continue\n'
            '    resid_k = quita_halo(np.nanmedian(LS_CUBE[sel_k], axis=0))[sl]\n'
            '    v_k = float(np.nanpercentile(np.abs(resid_k[cerca]), 98))\n'
            "    im_k = ax.imshow(resid_k, origin='lower', cmap='RdBu_r', vmin=-v_k, vmax=v_k)\n"
            "    ax.set_title(f'{lo_k:.0f}-{hi_k:.0f} Å  ({n_ch} canales)', fontsize=8)\n"
            '    cb_k = fig.colorbar(im_k, ax=ax, fraction=0.046)\n'
            '    cb_k.ax.tick_params(labelsize=6)\n'
            '    ax.add_patch(Circle((OBJECT_YX[1] - x0, OBJECT_YX[0] - y0), WINDOW_RADIUS_PX,\n'
            "                        fill=False, color='k', lw=0.8))\n"
            "    ax.plot(OBJECT_YX[1] - x0, OBJECT_YX[0] - y0, '+', color='k', ms=7)\n"
            '    ax.set_xlim(OBJECT_YX[1] - x0 - _zoom, OBJECT_YX[1] - x0 + _zoom)\n'
            '    ax.set_ylim(OBJECT_YX[0] - y0 - _zoom, OBJECT_YX[0] - y0 + _zoom)\n'
            '    ax.set_xticks([]); ax.set_yticks([])\n'
            '    # Lo que queda DENTRO de la ventana, que es lo que acaba en el espectro.\n'
            '    yy_k, xx_k = np.indices(resid_k.shape, dtype=float)\n'
            '    dentro_k = (np.hypot(yy_k - (OBJECT_YX[0] - y0), xx_k - (OBJECT_YX[1] - x0))\n'
            '                <= float(WINDOW_RADIUS_PX))\n'
            "    print(f'  {lo_k:6.0f}-{hi_k:6.0f} Å  ({n_ch:4d} ch):'\n"
            "          f' residuo en la ventana, mediana {float(np.nanmedian(resid_k[dentro_k])):8.3f}'\n"
            "          f' | pico {float(np.nanmax(resid_k[dentro_k])):8.3f}')\n"
            "fig.suptitle('halo azimutal restado, banda a banda (solo para ver)'\n"
            "             '  ·  círculo negro = ventana de extracción', fontsize=9)\n"
            'fig.tight_layout(); plt.show()\n'
            '\n'
        ),
        md(
            "## 4 · Las funciones numéricas, copiadas de `musepipe`\n\n"
            "Copia **literal**; edítalas y el resultado cambia. Se importan solo "
            "`evaluate_psf_model` (es de C1) y `run_channel_chunks` (paralelismo, no física).\n\n"
            + "\n".join(f"- `{name}` — de `{rel}`" for rel, name, _s, _h in sources)
        ),
        code(
            "# ------------------------------------------------------------------\n"
            "# COPIA EDITABLE. Fuente: musepipe (ver el chequeo de deriva abajo).\n"
            "# ------------------------------------------------------------------\n"
            + "\n".join(needed_imports(sources)) + "\n"
            "# `evaluate_psf_model` (C1) y `run_channel_chunks` (paralelismo) se importan\n"
            "# arriba: no son lo que se ajusta aquí.\n\n"
            + "\n".join(needed_constants(sources)) + "\n\n\n"
            + inline_src
        ),
        md("## 5 · Chequeo de deriva"),
        drift_cell(code, shas, "C3"),
        md(
            "## 6 · El modelo de la primaria (lo que separa las dos variantes)\n\n"
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
            "LAMBDA_VISTA = 7500.0    # canal que se dibuja; cámbialo y vuelve a ejecutar\n"
            "iz = int(np.argmin(np.abs(WAVE - LAMBDA_VISTA)))\n"
            "# El recorte tiene que contener a las DOS fuentes: centrado en la\n"
            "# primaria con ±40 px dejaba al compañero (a ~70 px) fuera de cuadro,\n"
            "# así que sus marcas no se veían y sus medianas salían NaN.\n"
            "MARGEN_PX = 16.0\n"
            "_ny, _nx = STAGE02.shape[1:]\n"
            "sl = (slice(int(max(0, min(STAR_YX[0], OBJECT_YX[0]) - MARGEN_PX)),\n"
            "            int(min(_ny, max(STAR_YX[0], OBJECT_YX[0]) + MARGEN_PX + 1))),\n"
            "      slice(int(max(0, min(STAR_YX[1], OBJECT_YX[1]) - MARGEN_PX)),\n"
            "            int(min(_nx, max(STAR_YX[1], OBJECT_YX[1]) + MARGEN_PX + 1))))\n"
            "sy0, sx0 = sl[0].start, sl[1].start\n"
            "dato, modelo_i = STAGE02[iz][sl], primary_model[iz][sl]\n"
            "resid = PSFSUB_CUBE[iz][sl]\n\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))\n"
            "# Dato y modelo comparten escala LOG y los MISMOS límites: si cada uno\n"
            "# llevara la suya, dos imágenes distintas se verían iguales.\n"
            "pos = np.concatenate([dato[np.isfinite(dato) & (dato > 0)].ravel(),\n"
            "                      modelo_i[np.isfinite(modelo_i) & (modelo_i > 0)].ravel()])\n"
            "v_lo = float(np.nanpercentile(pos, 40)) if pos.size else 1e-3\n"
            "v_hi = float(np.nanpercentile(pos, 99.99)) if pos.size else 1.0\n"
            "norma = LogNorm(vmin=max(v_lo, 1e-6), vmax=max(v_hi, v_lo * 10))\n"
            "for ax, img, titulo in ((axes[0], dato, 'B2 (con primaria)'),\n"
            "                        (axes[1], modelo_i, 'modelo de la primaria')):\n"
            "    im = ax.imshow(img, origin='lower', cmap='magma', norm=norma)\n"
            "    ax.set_title(f'{titulo}  ·  LOG', fontsize=9)\n"
            "    cb = fig.colorbar(im, ax=ax, fraction=0.046)\n"
            "    cb.set_label(f'flujo [{UNIDAD}]', fontsize=7)\n"
            "# El residuo es la resta de dos números grandes: se va a los dos signos\n"
            "# y su interés está ALREDEDOR DE CERO. Con una escala secuencial\n"
            "# anclada al máximo (el core de la estrella) todo lo demás sale negro,\n"
            "# que es justo lo que no se quiere mirar. Va en divergente y simétrica,\n"
            "# con los límites tomados FUERA del core: allí es donde el residuo\n"
            "# importa, porque es el fondo sobre el que se mide el compañero.\n"
            "yy_r, xx_r = np.indices(resid.shape, dtype=float)\n"
            "r_core = np.hypot(yy_r - (STAR_YX[0] - sy0), xx_r - (STAR_YX[1] - sx0))\n"
            "fuera = np.isfinite(resid) & (r_core > 6.0)\n"
            "v = float(np.nanpercentile(np.abs(resid[fuera]), 98)) if fuera.any() else 1.0\n"
            "im = axes[2].imshow(resid, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v)\n"
            "axes[2].set_title('residual = psfsub  ·  LINEAL simétrica', fontsize=9)\n"
            "cb = fig.colorbar(im, ax=axes[2], fraction=0.046, extend='both')\n"
            "cb.set_label(f'residuo [{UNIDAD}]', fontsize=7)\n"
            "for ax in axes:\n"
            "    ax.plot(STAR_YX[1] - sx0, STAR_YX[0] - sy0, '*', color='w', ms=11, mec='k')\n"
            "    ax.plot(OBJECT_YX[1] - sx0, OBJECT_YX[0] - sy0, '+', color='tab:cyan', ms=9)\n"
            "    ax.add_patch(Circle((OBJECT_YX[1] - sx0, OBJECT_YX[0] - sy0),\n"
            "                        float(PRIMARY_EXCL_RADIUS), fill=False,\n"
            "                        color='tab:cyan', lw=1.0, ls='--'))\n"
            "    ax.add_patch(Circle((STAR_YX[1] - sx0, STAR_YX[0] - sy0),\n"
            "                        float(PRIMARY_FIT_RADIUS), fill=False,\n"
            "                        color='w', lw=0.9, ls=':'))\n"
            "    # Los círculos se salen del recorte; sin fijar los límites estiran\n"
            "    # los ejes y la imagen queda flotando en un marco blanco.\n"
            "    ax.set_xlim(-0.5, resid.shape[1] - 0.5)\n"
            "    ax.set_ylim(-0.5, resid.shape[0] - 0.5)\n"
            "    ax.set_xlabel('x [px]')\n"
            "axes[0].set_ylabel('y [px]')\n"
            "fig.suptitle(f'λ = {WAVE[iz]:.0f} Å  ·  punteado blanco: radio de ajuste;'\n"
            "             ' discontinuo cian: disco excluido del compañero', fontsize=9)\n"
            "fig.tight_layout(); plt.show()\n\n"
            "# Cuánto queda tras restar, donde se mide: si el residuo en la ventana\n"
            "# no está centrado en cero, `psfsub` arrastra un pedestal.\n"
            "en_ventana = (np.hypot(yy_r - (OBJECT_YX[0] - sy0), xx_r - (OBJECT_YX[1] - sx0))\n"
            "              <= float(WINDOW_RADIUS_PX))\n"
            "print(f'en la ventana del compañero (λ={WAVE[iz]:.0f} Å):')\n"
            "print(f'  B2       mediana {float(np.nanmedian(dato[en_ventana])):9.3f} {UNIDAD}')\n"
            "print(f'  modelo   mediana {float(np.nanmedian(modelo_i[en_ventana])):9.3f}')\n"
            "print(f'  residual mediana {float(np.nanmedian(resid[en_ventana])):9.3f}'\n"
            "      '   <- lo que psfsub deja bajo el compañero')\n"
            "print(f'  residual mediana fuera del core (referencia): '\n"
            "      f'{float(np.nanmedian(resid[fuera])):9.3f}')"
        ),
        md(
            "## 7 · El peso óptimo, píxel a píxel\n\n"
            "Aquí es donde C3 se diferencia de C2, y conviene verlo en un canal antes de "
            "lanzarlo sobre los 3681. La extracción óptima de Horne no suma: **promedia con "
            "pesos**,\n\n"
            "> `f = Σ (P·D/V) / Σ (P²/V)`\n\n"
            "con `D` el dato, `V` la varianza y `P` el **perfil de PSF normalizado** en la "
            "ventana (de C1, evaluado a esa λ). Cada píxel pesa `P/V`: mucho donde se espera "
            "señal y poco ruido, casi nada en el borde de la ventana. Por eso gana S/N frente a "
            "sumar una caja, donde todos los píxeles pesan igual y los del borde solo meten "
            "ruido.\n\n"
            "Todo lo de esta sección es **del compañero**, no de la primaria: la ventana está "
            "centrada en `OBJECT_YX` y el cubo es el de la variante que elijas en `CUBO_VISTO` "
            "(`PSFSUB_CUBE`, ya con la primaria restada, o `LS_CUBE`). El título de la figura lo "
            "dice en cada ejecución.\n\n"
            "Los cuatro mapas de abajo son, en la ventana y a esa λ: el **dato**, el **perfil "
            "`P`** que hace de peso, la **varianza** que entra como `1/V`, y la **contribución "
            "de cada píxel al flujo final**. Ese último es el que contesta «¿de dónde sale este "
            "número?».\n\n"
            '### ¿Y el peso no debería calcularse sobre la estrella central?\n\n'
            'Hay dos cosas distintas metidas en esa pregunta, y la respuesta es **sí a una y no a la otra**.\n\n'
            '**La FORMA de la PSF sí sale de la estrella.** El compañero es demasiado débil para medir su propio perfil. El modelo `psf_model.json` que usa esta celda lo midió **C1 sobre la primaria**, que es la única fuente con señal de sobra. Así que la PSF *es* la de la estrella central.\n\n'
            '**Pero se evalúa CENTRADA EN EL COMPAÑERO.** Eso es lo que hace `normalized_psf_window(PSF_MODEL, λ, OBJECT_YX, ...)`: coge la forma medida en la estrella y la coloca en la posición del compañero. Y tiene que ser así porque el estimador de Horne **mide el flujo de la fuente cuyo perfil usas como peso**: `f = Σ(P·D/V) / Σ(P²/V)` es la amplitud que mejor ajusta `D ≈ f·P`. Si `P` estuviera centrado en la estrella, el número que saldría sería la amplitud del **halo de la estrella** en esa ventana, no el flujo del compañero.\n\n'
            'Y hay una segunda razón, más práctica: a esta separación el perfil de la estrella ya no tiene pico sobre la ventana del compañero, es una **rampa suave**. Un peso casi plano y creciente hacia la estrella pesaría más justo donde el fondo es más brillante y más ruidoso — exactamente lo contrario de «óptimo». La celda mide las dos cosas (el contraste de cada perfil y el número que saldría con cada uno) para que no haya que creérselo.\n\n'
            '> Entonces, ¿dónde entra la estrella? **Como fondo, no como peso.** Es lo que separa a las dos variantes: `psfsub` ajusta y resta el modelo de PSF de la primaria antes de extraer, y C4 va más lejos y ajusta las dos PSF a la vez. La PSF de la estrella se usa, pero para *quitarla*, no para pesar.\n\n'
            '> **El supuesto que queda**: que la PSF no cambie de forma entre la estrella y el compañero. En AO no es exacto (anisoplanatismo), y el QC lo acota moviendo la FWHM ±10 % (`psf_sensitivity`).\n\n'
            "### Por qué la contribución tiene valores positivos y negativos\n\n"
            "Porque el **dato** los tiene. La contribución de un píxel es\n\n"
            "> `cᵢ = (Pᵢ · Dᵢ / Vᵢ) / Σ(P²/V)`,  y  `Σᵢ cᵢ = f`\n\n"
            "y como `Pᵢ > 0` y `Vᵢ > 0`, **el signo de `cᵢ` es el signo de `Dᵢ`**. En un canal "
            "suelto el compañero está por debajo del ruido, así que la mitad de los píxeles "
            "tienen el dato por debajo de cero (fluctuación, o resta de fondo/PSF que se pasó) y "
            "restan al total. Eso es lo normal y lo correcto: el estimador **no** es una suma de "
            "cosas positivas, es un promedio ponderado, y el flujo sale de la **cancelación** de "
            "ruido alrededor de un valor pequeño.\n\n"
            "Lo que sí sería sospechoso es un patrón: si los negativos se agruparan en un lado "
            "(gradiente de fondo mal restado) o justo en el centro (sobre-sustracción de la "
            "primaria). Repartidos como sal y pimienta, es ruido.\n\n"
            "Y en rojo, los píxeles que el **σ-clipping** descartó: el estimador itera "
            "`CLIP_SIGMA` veces quitando los que se desvían del modelo `f·P`. Es una protección "
            "contra cósmicos y píxeles malos, pero si el rechazo se concentra **sobre el "
            "compañero** se estaría recortando la señal y llamándolo ruido — es exactamente lo "
            "que vigila el chequeo `v4_clip_concentration` de la spec.\n\n"
            "### Qué es el perfil radial de la última figura\n\n"
            "El eje x es la **distancia de cada píxel al centro del compañero** (`OBJECT_YX`, el "
            "que fijó B3), en píxeles; el eje y es su flujo, con la barra `√V`. Los puntos son "
            "los píxeles de la ventana, sin ordenar ni promediar: cada uno es un píxel.\n\n"
            "Lo que caracteriza es **si el dato tiene la forma que el estimador supone**. La "
            "línea azul es `f·P`: el modelo que resulta del ajuste, o sea *el mismo perfil de "
            "PSF, escalado al flujo que salió*. Si el dato siguiera una campana más ancha o más "
            "estrecha que esa línea, el peso estaría mal puesto y el estimador sería solo "
            "distinto, no óptimo (es lo que mide `psf_sensitivity` en el QC, moviendo la FWHM "
            "±10 %). Y si hubiera una **pendiente** en los puntos que la línea no sigue, sería "
            "fondo sin restar dentro de la ventana.\n\n"
            "> **Lo que se ve en el perfil radial no es una detección, y está bien.** En *un* "
            "canal el compañero queda por debajo de la barra de error: el modelo `f·P` sale "
            "casi plano. La señal aparece al juntar los 3681 canales, que es lo que hace la "
            "sección siguiente. Si en un solo canal se viera un pico limpio, habría que "
            "desconfiar."
        ),
        code(
            "LAMBDA_PESO = LAMBDA_VISTA      # el mismo canal de la sección anterior\n"
            "CUBO_VISTO = PSFSUB_CUBE        # prueba con LS_CUBE para ver la otra variante\n\n"
            "ETIQUETA_CUBO = ('psfsub (primaria restada)' if CUBO_VISTO is PSFSUB_CUBE\n"
            "                 else 'ls (cubo crudo + anillo)')\n"
            "iz2 = int(np.argmin(np.abs(WAVE - LAMBDA_PESO)))\n"
            "wy, wx = circular_window_indices(CUBO_VISTO.shape[1:], OBJECT_YX, WINDOW_RADIUS_PX)\n"
            "P = normalized_psf_window(PSF_MODEL, WAVE[iz2], OBJECT_YX, wy, wx)\n"
            "D = CUBO_VISTO[iz2][wy, wx]\n"
            "if STAT_CUBE is not None:\n"
            "    V = STAT_CUBE[iz2][wy, wx] * STAT_FACTOR\n"
            "else:\n"
            "    V = np.full(P.shape, float(robust_sigma(D)) ** 2)\n"
            "valido = np.isfinite(D) & np.isfinite(V) & (V > 0)\n"
            "f_ch, var_ch, neff, frac_clip, recortados = _channel_estimate(\n"
            "    D, V, P, valido, clip_sigma=CLIP_SIGMA, clip_max_iter=CLIP_MAX_ITER)\n\n"
            "usados = valido & ~recortados\n"
            "r_px = np.hypot(wy - OBJECT_YX[0], wx - OBJECT_YX[1])\n"
            "contrib = np.zeros_like(D)\n"
            "denom = float(np.sum(P[usados] ** 2 / V[usados]))\n"
            "contrib[usados] = P[usados] * D[usados] / V[usados] / denom   # suman f_ch\n"
            "orden = np.argsort(P)[::-1]\n"
            "top = orden[:max(1, P.size // 10)]\n"
            "print(f'λ = {WAVE[iz2]:.0f} Å · ventana de {P.size} px'\n"
            "      f' ({int(usados.sum())} usados, {int(recortados.sum())} recortados'\n"
            "      f' = {100 * frac_clip:.1f}%)')\n"
            "print(f'  f = {f_ch:.3f} ± {np.sqrt(var_ch):.3f} {UNIDAD}   (npix_eff = {neff:.1f})')\n"
            "print(f'  píxeles que restan al total (dato < 0): '\n"
            "      f'{int((contrib[usados] < 0).sum())} de {int(usados.sum())}'\n"
            "      '  <- normal: en un canal el compañero está bajo el ruido')\n"
            "print(f'  el 10% de píxeles con más peso aporta el'\n"
            "      f' {100 * float(np.sum(contrib[top])) / f_ch:.0f}% del flujo'\n"
            "      f'  <- esto es lo que una caja reparte por igual')\n"
            "print(f'  suma simple de la ventana (sin pesos): {float(np.nansum(D[usados])):.3f}'\n"
            "      '  (no es comparable: no lleva apcorr ni normalización)')\n\n"
            '# La pregunta de arriba, medida: el MISMO modelo de PSF centrado en la\n'
            '# primaria, sobre los mismos pixeles de la ventana del compañero.\n'
            'P_star = normalized_psf_window(PSF_MODEL, WAVE[iz2], STAR_YX, wy, wx)\n'
            'contraste = lambda q: float(np.nanmax(q) / np.nanmedian(q))\n'
            'f_star_w = float(np.sum(P_star[usados] * D[usados] / V[usados])\n'
            '                 / np.sum(P_star[usados] ** 2 / V[usados]))\n'
            'print()\n'
            "print('el peso, centrado donde toca y centrado en la primaria:')\n"
            "print(f'  P centrado en el COMPAÑERO : contraste max/mediana = {contraste(P):7.1f}'\n"
            "      f'  ->  f = {f_ch:9.3f}')\n"
            "print(f'  P centrado en la PRIMARIA  : contraste max/mediana = {contraste(P_star):7.1f}'\n"
            "      f'  ->  f = {f_star_w:9.3f}')\n"
            "print('  el segundo NO es el flujo del compañero: es la amplitud del halo de la')\n"
            "print('  estrella en esta ventana. Un peso casi plano pesa mas donde el fondo es')\n"
            "print('  mas brillante y mas ruidoso, que es lo contrario de optimo.')\n"
            'print()\n'
            '\n'
            "# El chequeo `v4_clip_concentration` de la spec, en este canal: si el\n"
            "# rechazo se ceba en el núcleo del compañero, se está recortando la\n"
            "# señal y llamándola ruido.\n"
            "RADIO_NUCLEO_PX = 2.0\n"
            "nucleo = valido & (r_px <= RADIO_NUCLEO_PX)\n"
            "if nucleo.any() and valido.any():\n"
            "    tasa_nucleo = float(recortados[nucleo].mean())\n"
            "    tasa_total = float(recortados[valido].mean())\n"
            "    razon = tasa_nucleo / tasa_total if tasa_total > 0 else 0.0\n"
            "    print(f'  clipping: {100 * tasa_total:5.1f}% en la ventana,'\n"
            "          f' {100 * tasa_nucleo:5.1f}% en el núcleo (r<={RADIO_NUCLEO_PX:g} px)'\n"
            "          f' -> {razon:.1f}×')\n"
            "    if razon > 2.0:\n"
            "        print('    AVISO: el rechazo se concentra sobre el compañero.'\n"
            "              ' Es lo que vigila v4_clip_concentration: el modelo f·P se'\n"
            "              ' queda corto en el núcleo y sus píxeles salen como outliers.')\n\n"
            "def _mapa(valores):\n"
            "    \"\"\"Los píxeles dispersos de la ventana, de vuelta a una imagen.\"\"\"\n"
            "    m = np.full((wy.max() - wy.min() + 1, wx.max() - wx.min() + 1), np.nan)\n"
            "    m[wy - wy.min(), wx - wx.min()] = valores\n"
            "    return m\n\n"
            "paneles = ((_mapa(D), f'dato D [{UNIDAD}]', 'viridis', None),\n"
            "           (_mapa(P), 'perfil P (normalizado, Σ=1)', 'magma', None),\n"
            "           (_mapa(V), f'varianza V [{UNIDAD}²]', 'cividis', None),\n"
            "           (_mapa(contrib), f'contribución al flujo [{UNIDAD}]', 'RdBu_r', 'sim'))\n"
            "fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))\n"
            "for ax, (img, titulo, cmap, modo) in zip(axes, paneles):\n"
            "    if modo == 'sim':\n"
            "        v = float(np.nanpercentile(np.abs(img), 99.5))\n"
            "        im = ax.imshow(img, origin='lower', cmap=cmap, vmin=-v, vmax=v)\n"
            "    else:\n"
            "        lo, hi = np.nanpercentile(img, [1, 99.5])\n"
            "        im = ax.imshow(img, origin='lower', cmap=cmap, vmin=lo, vmax=hi)\n"
            "    ax.set_title(titulo, fontsize=8)\n"
            "    fig.colorbar(im, ax=ax, fraction=0.046)\n"
            "    if recortados.any():\n"
            "        ax.plot(wx[recortados] - wx.min(), wy[recortados] - wy.min(), 'x',\n"
            "                color='tab:red', ms=5, mew=1.2)\n"
            "    ax.plot(OBJECT_YX[1] - wx.min(), OBJECT_YX[0] - wy.min(), '+',\n"
            "            color='w', ms=9)\n"
            "    ax.set_xticks([]); ax.set_yticks([])\n"
            "fig.suptitle(f'ventana del COMPAÑERO sobre el cubo {ETIQUETA_CUBO}'\n"
            "             f'  ·  λ = {WAVE[iz2]:.0f} Å  ·  × roja = recortado por el σ-clipping',\n"
            "             fontsize=9)\n"
            "fig.tight_layout(); plt.show()\n\n"
            "# El perfil radial: si el modelo `f·P` no sigue al dato, el peso está\n"
            "# mal puesto y el estimador no es óptimo, solo distinto.\n"
            "fig, ax = plt.subplots(figsize=(7.5, 3.6))\n"
            "ax.errorbar(r_px[usados], D[usados], yerr=np.sqrt(V[usados]), fmt='o', ms=3,\n"
            "            lw=0.6, color='0.4', alpha=0.8, label='dato ± √V')\n"
            "if recortados.any():\n"
            "    ax.plot(r_px[recortados], D[recortados], 'x', color='tab:red', ms=6,\n"
            "            label='recortado por clipping')\n"
            "ord_r = np.argsort(r_px)\n"
            "ax.plot(r_px[ord_r], (f_ch * P)[ord_r], lw=1.6, color='tab:blue',\n"
            "        label='modelo ajustado  f · P')\n"
            "ax.axhline(0, color='0.7', lw=0.6)\n"
            "ax.set_xlabel('distancia al centro del compañero [px]')\n"
            "ax.set_ylabel(f'flujo [{UNIDAD}]')\n"
            "ax.set_title(f'perfil radial del COMPAÑERO en su ventana · cubo {ETIQUETA_CUBO}'\n"
            "             f' · λ = {WAVE[iz2]:.0f} Å', fontsize=9)\n"
            "ax.legend(fontsize=8); fig.tight_layout(); plt.show()"
        ),
        md(
            "## 8 · Las dos extracciones\n\n"
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
            "    # El STAT propagado se guarda SIEMPRE (exista o no el modo `stat`):\n"
            "    # es la segunda estimación contra la que se contrasta el empírico.\n"
            "    err_stat = np.sqrt(np.clip(raw_var, 0.0, np.inf)) if variance is not None else None\n"
            "    err = err_stat if usable else np.asarray(err_emp, float)\n"
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
            "            'err_stat': None if err_stat is None else err_stat * apcorr,\n"
            "            'modo': modo, 'controles': ctrl, 'n_ctrl': len(ctrl_yx)}\n\n"
            "LS     = extrae(LS_CUBE, STAT_CUBE, 'ls')\n"
            "PSFSUB = extrae(PSFSUB_CUBE, STAT_CUBE, 'psfsub')"
        ),
        md(
            '### Doble resultado: las mismas cuentas con una apertura tipo C2\n\n'
            'Hasta aquí todo sale del estimador óptimo sobre una **ventana circular de `WINDOW_RADIUS_PX`** px. Pero si la sospecha está en la resta del fondo, conviene saber **cuánto del resultado depende de la geometría y del peso**, y no solo del modelo de halo.\n\n'
            'Así que se repite la extracción con la **receta de C2**: suma en una **caja 3×3**, mismo fondo de anillo, misma `apcorr` — sin pesos y sobre 9 píxeles en vez de ~200. Se hace sobre **los dos cubos**, así que a partir de aquí hay cuatro espectros:\n\n'
            '| | ventana óptima (r = `WINDOW_RADIUS_PX`) | apertura box3 (C2) |\n\n'
            '|---|---|---|\n\n'
            '| cubo de `ls` | `LS` | `LS_AP` |\n\n'
            '| cubo de `psfsub` | `PSFSUB` | `PSFSUB_AP` |\n\n'
            'Cómo leerlo: la apertura y la ventana óptima **ven fondos distintos**. La caja 3×3 mide 9 píxeles pegados al compañero; la ventana circular abarca hasta 8 px, donde el gradiente del halo ya es apreciable. Si el negativo del azul apareciera solo en la ventana grande, sería un problema de **extensión**; si aparece en las dos por igual, el fondo que se resta está mal **en el propio píxel del compañero**, y la geometría no tiene la culpa.\n\n'
            '> Ojo con comparar los niveles en crudo: la caja suma 9 píxeles y la ventana ~200, así que sus cuentas no son comparables. Lo que sí lo es, y es lo que hay que mirar, es **el espectro ya corregido por `apcorr`** — que es precisamente lo que hace `apcorr`: llevar los dos al flujo total de la fuente.\n\n'
        ),
        code(
            "from musepipe.spectral import median_filter_1d\n"
            '\n'
            "APERTURA_C2 = {'kind': 'box', 'size': 3}    # la misma que usa C2\n"
            '\n'
            'def _fondo_anillo(cube, pos):\n'
            '    if LOCAL_BKG_ANNULUS_PX is None:\n'
            '        return None\n'
            '    r_ex = LOCAL_BKG_ANNULUS_PX[2] if len(LOCAL_BKG_ANNULUS_PX) > 2 else 30.0\n'
            '    return annulus_background_spectrum(cube, pos, LOCAL_BKG_ANNULUS_PX[0],\n'
            '                                       LOCAL_BKG_ANNULUS_PX[1],\n'
            '                                       exclude_yx=STAR_YX, exclude_radius=r_ex)\n'
            '\n'
            'def extrae_apertura(cube, etiqueta):\n'
            '    """La receta de C2 sobre el mismo cubo: caja, anillo, apcorr. Sin pesos."""\n'
            '    raw_f, npix = aperture_spectrum(cube, OBJECT_YX, APERTURA_C2)\n'
            '    bkg = _fondo_anillo(cube, OBJECT_YX)\n'
            '    if bkg is not None:\n'
            '        raw_f = raw_f - bkg * npix\n'
            '    ctrl_yx, ctrl, ctrl_npix = control_aperture_spectra(\n'
            '        cube, OBJECT_YX, STAR_YX, APERTURA_C2,\n'
            '        n_controls=N_CONTROLS, exclude_angle_deg=EXCLUDE_ANGLE_DEG)\n'
            '    # Control = objeto: el mismo fondo, restado igual, o el sigma no vale.\n'
            '    if bkg is not None and ctrl.shape[0]:\n'
            '        ctrl = np.asarray(ctrl, dtype=float).copy()\n'
            '        for k, pos in enumerate(ctrl_yx):\n'
            '            cb = _fondo_anillo(cube, pos)\n'
            '            if cb is not None:\n'
            '                ctrl[k] = ctrl[k] - cb * ctrl_npix[k]\n'
            '    err_emp = (robust_sigma_axis0(ctrl) if ctrl.shape[0] >= 2\n'
            '               else np.full(WAVE.size, robust_sigma(raw_f)))\n'
            '    err_stat = None\n'
            '    if STAT_CUBE is not None:\n'
            '        err_stat = aperture_stat_error(STAT_CUBE, OBJECT_YX, APERTURA_C2,\n'
            '                                       stat_factor=STAT_FACTOR,\n'
            '                                       covariance_factor=COV_FACTOR)\n'
            '    apc, _m, _r = aperture_correction_from_psf(\n'
            '        WAVE, APERTURA_C2, PSF_MODEL, center_yx=OBJECT_YX, correction_mode=APCORR_MODE)\n'
            "    print(f'{etiqueta:11s} apcorr={float(np.nanmedian(apc)):7.2f}'\n"
            "          f' npix={float(np.nanmedian(npix)):5.1f}'\n"
            "          f' flujo mediano={float(np.nanmedian(raw_f * apc)):10.2f}')\n"
            "    return {'raw': {'flux': raw_f}, 'flux': raw_f * apc,\n"
            "            'err_emp': np.asarray(err_emp, float) * apc,\n"
            "            'err': np.asarray(err_emp, float) * apc,\n"
            "            'err_stat': None if err_stat is None else err_stat * apc,\n"
            "            'apcorr': apc, 'modo': 'empirical', 'controles': ctrl,\n"
            "            'n_ctrl': len(ctrl_yx)}\n"
            '\n'
            "LS_AP     = extrae_apertura(LS_CUBE, 'ls · box3')\n"
            "PSFSUB_AP = extrae_apertura(PSFSUB_CUBE, 'psfsub · box3')\n"
            '\n'
            'ROJO_CMP = (7500.0, 9000.0)\n'
            'sel_cmp = (WAVE >= ROJO_CMP[0]) & (WAVE <= ROJO_CMP[1])\n'
            'print()\n'
            "print(f'flujo mediano en {ROJO_CMP[0]:.0f}-{ROJO_CMP[1]:.0f} Å, ya con apcorr:')\n"
            "for nombre, opt, ap in (('ls', LS, LS_AP), ('psfsub', PSFSUB, PSFSUB_AP)):\n"
            "    f_o = float(np.nanmedian(opt['flux'][sel_cmp]))\n"
            "    f_a = float(np.nanmedian(ap['flux'][sel_cmp]))\n"
            "    print(f'  {nombre:8s} óptima {f_o:9.1f} | apertura box3 {f_a:9.1f}'\n"
            '          f\' | apertura/óptima {f_a / f_o if f_o else float("nan"):6.2f}\')\n'
            '\n'
            'fig, ejes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)\n'
            "for ax, (nombre, opt, ap) in zip(ejes, (('optimal_ls', LS, LS_AP),\n"
            "                                        ('optimal_psfsub', PSFSUB, PSFSUB_AP))):\n"
            "    ax.plot(WAVE, median_filter_1d(opt['flux'], 41), lw=1.3, color='tab:blue',\n"
            "            label=f'ventana óptima r={float(WINDOW_RADIUS_PX):g} px')\n"
            "    ax.plot(WAVE, median_filter_1d(ap['flux'], 41), lw=1.3, color='tab:orange',\n"
            "            label='apertura box3 (receta de C2)')\n"
            "    ax.axhline(0, color='0.5', lw=0.7)\n"
            "    ax.axvline(6562.8, color='tab:red', ls=':', lw=0.8)\n"
            "    ax.set_ylabel(f'{nombre}  [{UNIDAD}]', fontsize=8)\n"
            '    ax.legend(fontsize=7)\n'
            "ejes[-1].set_xlabel('λ [Å]')\n"
            "ejes[0].set_title('el mismo fondo, dos geometrías: si el negativo del azul está en las'\n"
            "                  ' dos, no es de la ventana', fontsize=9)\n"
            'fig.tight_layout(); plt.show()\n'
            '\n'
        ),
        md(
            "## 9 · Las dos variantes, una al lado de la otra\n\n"
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
            "## 10 · El espectro binado con su error (los dos métodos)\n\n"
            "El flujo por canal es demasiado ruidoso para leerse, y una mediana móvil suaviza "
            "pero **no dice cuánto vale lo que se ve**. Aquí se **bina**: se agrupan "
            "`BIN_CANALES` canales, el flujo es la media y el error se propaga como gaussiano "
            "independiente,\n\n"
            "> σ_bin = √(Σ σᵢ²) / n\n\n"
            "que para σ constante es el conocido σ/√n. Así cada punto lleva su barra y se puede "
            "juzgar si el espectro está por encima de cero — y, sobre todo, **comparar las dos "
            "variantes con una barra de error delante**, que es lo que decide si su diferencia "
            "es real o es ruido.\n\n"
            "> **Aviso que la propia cadena mide**: esa fórmula supone canales **independientes**, "
            "y no lo son. G1 midió `n_eff/n ≈ 0.43` (el remuestreo en λ correlacionó canales "
            "vecinos), así que el error binado gaussiano está **subestimado en ~√(1/0.43) ≈ "
            "1.5×**. Se dibujan las dos barras."
        ),
        code(
            "BIN_CANALES = 25        # cámbialo y vuelve a ejecutar\n"
            "N_EFF_OVER_N = 0.43     # medido por G1 (docs/noise_model.md)\n"
            "ROJO_A = (7500.0, 9000.0)\n\n"
            "def binea(wave, flujo, err, n):\n"
            "    \"\"\"Media por bloques de n canales, con error gaussiano independiente.\"\"\"\n"
            "    n = int(n)\n"
            "    corte = (wave.size // n) * n\n"
            "    w = wave[:corte].reshape(-1, n)\n"
            "    f = np.asarray(flujo, dtype=float)[:corte].reshape(-1, n)\n"
            "    e = np.asarray(err, dtype=float)[:corte].reshape(-1, n)\n"
            "    bueno = np.isfinite(f) & np.isfinite(e)\n"
            "    cuenta = bueno.sum(axis=1)\n"
            "    with np.errstate(invalid='ignore', divide='ignore'):\n"
            "        wb = np.nanmean(np.where(bueno, w, np.nan), axis=1)\n"
            "        fb = np.nansum(np.where(bueno, f, 0.0), axis=1) / np.maximum(cuenta, 1)\n"
            "        eb = np.sqrt(np.nansum(np.where(bueno, e, 0.0) ** 2, axis=1)) / np.maximum(cuenta, 1)\n"
            "    vacio = cuenta == 0\n"
            "    fb[vacio] = np.nan; eb[vacio] = np.nan\n"
            "    return wb, fb, eb, cuenta\n\n"
            "fig, ejes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)\n"
            "for ax, (nombre, v) in zip(ejes, (('optimal_ls', LS), ('optimal_psfsub', PSFSUB))):\n"
            "    wb, fb, eb, _ = binea(WAVE, v['flux'], v['err_emp'], BIN_CANALES)\n"
            "    eb_corr = eb / np.sqrt(N_EFF_OVER_N)   # canales correlacionados (G1)\n"
            "    rojo_b = (wb >= ROJO_A[0]) & (wb <= ROJO_A[1])\n"
            "    n_pts = int(np.isfinite(fb).sum())\n"
            "    f_med = float(np.nanmedian(fb[rojo_b]))\n"
            "    e_med = float(np.nanmedian(eb_corr[rojo_b]))\n"
            "    snr_med = float(np.nanmedian(np.abs(fb[rojo_b]) / eb_corr[rojo_b]))\n"
            "    print(f'{nombre:15s} {n_pts:3d} puntos de {BIN_CANALES} ch |'\n"
            "          f' en {ROJO_A[0]:.0f}-{ROJO_A[1]:.0f} Å: flujo {f_med:9.1f}'\n"
            "          f' ± {e_med:6.1f} -> S/N {snr_med:5.2f}')\n"
            "    ax.plot(WAVE, v['flux'], lw=0.3, color='0.78', label='flujo por canal')\n"
            "    ax.errorbar(wb, fb, yerr=eb_corr, fmt='o', ms=3, lw=0.9, color='tab:blue',\n"
            "                ecolor='tab:blue', alpha=0.9,\n"
            "                label=f'binado {BIN_CANALES} ch, ±σ corregido por n_eff')\n"
            "    ax.errorbar(wb, fb, yerr=eb, fmt='none', lw=1.8, ecolor='tab:orange', alpha=0.8,\n"
            "                label='±σ gaussiano (subestima: canales correlacionados)')\n"
            "    ax.axhline(0, color='0.5', lw=0.7)\n"
            "    ax.axvline(6562.8, color='tab:red', ls=':', label='Hα')\n"
            "    fin_b = np.isfinite(fb)\n"
            "    if fin_b.any():\n"
            "        ax.set_ylim(*np.nanpercentile(fb[fin_b], [1, 99]) * np.array([2.5, 2.5]))\n"
            "    ax.set_ylabel(f'{nombre}  [{UNIDAD}]', fontsize=8)\n"
            "ejes[0].legend(fontsize=7, ncol=2)\n"
            "ejes[-1].set_xlabel('λ [Å]')\n"
            "ejes[0].set_title('las dos variantes del COMPAÑERO, binadas y con su error',\n"
            "                  fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 11 · Tres controles con geometría fija (los dos métodos)\n\n"
            "Los `N_CONTROLS` controles de la extracción los reparte el pipeline en ángulo. Aquí "
            "se miran **tres sitios elegidos a mano**, a la misma distancia de la primaria que el "
            "compañero: el **opuesto** (PA + 180°) y los dos **perpendiculares** (PA ± 90°). Se "
            "colocan con los helpers de B3, así que heredan su convención de PA y el "
            "`north_angle_deg` del run.\n\n"
            "Cada uno pasa por **exactamente el mismo proceso** que el compañero —misma ventana, "
            "mismo fondo, mismo estimador óptimo, misma `apcorr`— y para **las dos variantes**. "
            "Ahí no hay ninguna fuente, así que lo que se vea es halo residual y ruido.\n\n"
            "Es la prueba directa de qué separa a `ls` de `psfsub`: **no es lo que hacen sobre el "
            "compañero, es lo que dejan donde no hay nada**. Si una deja los controles centrados "
            "en cero y la otra no, esa otra arrastra un pedestal que en el compañero no se "
            "distingue de flujo."
        ),
        code(
            "from musepipe.stages.stage01c_localize import position_from_sep_pa\n"
            "from musepipe.spectral import median_filter_1d\n\n"
            "astro = qc_b3.get('astrometry') or {}\n"
            "escala = qc_b3.get('pixel_scale_arcsec')\n"
            "norte = (qc_b3.get('wcs_orientation') or {}).get('north_angle_deg', 0.0)\n"
            "TRES = [('opuesto', 180.0), ('perpendicular +90', 90.0), ('perpendicular -90', -90.0)]\n\n"
            "def extrae_en(cube, pos):\n"
            "    \"\"\"El mismo estimador de la sección 8, en otra posición.\"\"\"\n"
            "    var = (np.asarray(STAT_CUBE, dtype=float) * STAT_FACTOR\n"
            "           if STAT_CUBE is not None else estimate_variance_cube(cube))\n"
            "    bkg = None\n"
            "    if LOCAL_BKG_ANNULUS_PX is not None:\n"
            "        r_ex = LOCAL_BKG_ANNULUS_PX[2] if len(LOCAL_BKG_ANNULUS_PX) > 2 else 30.0\n"
            "        bkg = annulus_background_spectrum(\n"
            "            cube, pos, LOCAL_BKG_ANNULUS_PX[0], LOCAL_BKG_ANNULUS_PX[1],\n"
            "            exclude_yx=STAR_YX, exclude_radius=r_ex)\n"
            "    crudo = optimal_raw_spectrum(cube, var, WAVE, pos, PSF_MODEL,\n"
            "                                 window_radius_px=WINDOW_RADIUS_PX,\n"
            "                                 clip_sigma=CLIP_SIGMA, clip_max_iter=CLIP_MAX_ITER,\n"
            "                                 n_jobs=1, bkg_spectrum=bkg)\n"
            "    return crudo['flux']\n\n"
            "sel_r = (WAVE >= 7500) & (WAVE <= 9000)\n"
            "fig, ejes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)\n"
            "for ax, (nombre, v, cubo) in zip(ejes, (('optimal_ls', LS, LS_CUBE),\n"
            "                                        ('optimal_psfsub', PSFSUB, PSFSUB_CUBE))):\n"
            "    print(f'{nombre}:  (mediana en 7500-9000 Å)')\n"
            "    med_obj = float(np.nanmedian(v['flux'][sel_r]))\n"
            "    print(f'  {\"compañero\":22s} {med_obj:10.1f}')\n"
            "    ax.plot(WAVE, median_filter_1d(v['flux'], 41), lw=1.4, color='tab:blue',\n"
            "            label='compañero')\n"
            "    for etiqueta, delta in TRES:\n"
            "        pos = position_from_sep_pa((float(STAR_YX[0]), float(STAR_YX[1])),\n"
            "                                   float(astro['sep_arcsec']),\n"
            "                                   float(astro['pa_deg']) + delta,\n"
            "                                   float(escala), north_angle_deg=float(norte or 0.0))\n"
            "        f_c = extrae_en(cubo, pos) * v['apcorr']\n"
            "        ax.plot(WAVE, median_filter_1d(f_c, 41), lw=0.9, alpha=0.8, label=etiqueta)\n"
            "        print(f'  {etiqueta:22s} {float(np.nanmedian(f_c[sel_r])):10.1f}'\n"
            "              f'    en y={pos[0]:.1f} x={pos[1]:.1f}')\n"
            "    ax.axhline(0, color='0.5', lw=0.7)\n"
            "    ax.set_ylabel(f'{nombre}  [{UNIDAD}]', fontsize=8)\n"
            "    ax.legend(fontsize=7, ncol=4)\n"
            "ejes[-1].set_xlabel('λ [Å]')\n"
            "ejes[0].set_title('el compañero y tres posiciones sin fuente, procesadas igual'\n"
            "                  '  (mediana móvil 41 ch)', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            '## 12 · ¿El negativo del azul es de la RESTA o de la corrección de flujo?\n\n'
            'En el extremo azul el continuo se va por debajo de cero, **y también en los controles** — o sea que no es del compañero, es del procedimiento. Esta celda separa los dos únicos sitios donde puede haberse metido:\n\n'
            '1. **La resta** (fondo de anillo en `ls`, modelo de PSF de la primaria en `psfsub`): si se pasa, el espectro sale negativo **ya en cuentas**, antes de tocar nada más.\n\n'
            '2. **La corrección de flujo** (`apcorr`, la curva de crecimiento de C1): multiplica, y en el azul multiplica mucho más que en el rojo porque la PSF se ensancha.\n\n'
            'La distinción es fácil de hacer y no admite discusión: **`apcorr` es siempre positiva, así que no puede cambiar el signo**. Si el crudo ya es negativo, el error está en la resta y `apcorr` solo lo amplifica. Si el crudo está en cero y el corregido no, sería al revés — pero eso es aritméticamente imposible, y por eso el diagnóstico es limpio.\n\n'
            'Lo que la figura sí añade sobre esa lógica es **cuánto** aporta cada uno, y con qué significancia: el panel de arriba es el espectro **crudo** (antes de `apcorr`, en las cuentas del cubo) para el objeto y los controles; el del medio, lo mismo ya corregido; el de abajo, la `apcorr` que los separa. Si objeto y controles caen juntos en el azul, la resta se está pasando **por igual en todas partes**, que es la firma de un modelo de fondo mal escalado y no de algo local al compañero.\n\n'
            '> El eje dice «cuentas del cubo» y no ADU porque el cubo ya viene escalado por el DRS: la unidad es la de `BUNIT`. Lo que importa aquí no es la unidad, es que **todavía no se ha aplicado `apcorr`**.\n\n'
        ),
        code(
            'BANDA_AZUL = (4800.0, 5600.0)\n'
            'BANDA_ROJA = (7500.0, 9000.0)\n'
            '\n'
            'def _med_en(v, banda):\n'
            '    sel = (WAVE >= banda[0]) & (WAVE <= banda[1])\n'
            '    return float(np.nanmedian(np.asarray(v, dtype=float)[sel]))\n'
            '\n'
            'fig, ejes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)\n'
            "for nombre, v, color in (('optimal_ls', LS, 'tab:blue'),\n"
            "                         ('optimal_psfsub', PSFSUB, 'tab:green')):\n"
            "    crudo = np.asarray(v['raw']['flux'], dtype=float)      # ANTES de apcorr\n"
            "    ctrl = np.asarray(v['controles'], dtype=float)          # controles, tambien crudos\n"
            '    ctrl_med = np.nanmedian(ctrl, axis=0) if ctrl.ndim == 2 and ctrl.shape[0] else None\n'
            '    ejes[0].plot(WAVE, median_filter_1d(crudo, 41), lw=1.2, color=color, label=nombre)\n'
            '    if ctrl_med is not None:\n'
            "        ejes[0].plot(WAVE, median_filter_1d(ctrl_med, 41), lw=1.0, ls='--', color=color,\n"
            "                     alpha=0.7, label=nombre + ' · mediana de los controles')\n"
            "    ejes[1].plot(WAVE, median_filter_1d(v['flux'], 41), lw=1.2, color=color, label=nombre)\n"
            '    if ctrl_med is not None:\n'
            "        ejes[1].plot(WAVE, median_filter_1d(ctrl_med * v['apcorr'], 41), lw=1.0, ls='--',\n"
            '                     color=color, alpha=0.7)\n'
            '    # La tabla: donde esta el signo, y cuanto lo amplifica la correccion.\n'
            "    print(nombre + ':')\n"
            "    for etiqueta, banda in (('azul', BANDA_AZUL), ('rojo', BANDA_ROJA)):\n"
            '        c_obj = _med_en(crudo, banda)\n'
            "        c_ctl = _med_en(ctrl_med, banda) if ctrl_med is not None else float('nan')\n"
            "        ap = _med_en(v['apcorr'], banda)\n"
            "        print(f'  {etiqueta} {banda[0]:.0f}-{banda[1]:.0f} Å  crudo objeto {c_obj:9.3f}'\n"
            "              f' | crudo controles {c_ctl:9.3f} | apcorr {ap:6.2f}'\n"
            "              f' | corregido {c_obj * ap:10.1f}')\n"
            "    signo = 'YA es negativo en crudo -> el problema esta en la RESTA'\n"
            '    if _med_en(crudo, BANDA_AZUL) >= 0:\n'
            "        signo = 'el crudo NO es negativo en el azul'\n"
            "    print('  ' + signo)\n"
            '\n'
            "ejes[2].plot(WAVE, LS['apcorr'], lw=1.0, color='tab:purple')\n"
            "ejes[2].set_ylabel('apcorr [adim.]')\n"
            "ejes[2].set_xlabel('λ [Å]')\n"
            "for ax, titulo, unidad in ((ejes[0], 'crudo: ANTES de la corrección de apertura', UNIDAD),\n"
            "                           (ejes[1], 'corregido: × apcorr', UNIDAD)):\n"
            "    ax.axhline(0, color='0.5', lw=0.7)\n"
            "    ax.set_ylabel('flujo [' + unidad + ']', fontsize=8)\n"
            '    ax.set_title(titulo, fontsize=9)\n'
            '    ax.legend(fontsize=7, ncol=2)\n'
            'for ax in ejes:\n'
            "    ax.axvspan(BANDA_AZUL[0], BANDA_AZUL[1], color='tab:blue', alpha=0.06)\n"
            "    ax.axvspan(BANDA_ROJA[0], BANDA_ROJA[1], color='tab:red', alpha=0.06)\n"
            "fig.suptitle('el mismo espectro antes y después de apcorr: apcorr multiplica,'\n"
            "             ' no puede cambiar el signo', fontsize=9)\n"
            'fig.tight_layout(); plt.show()\n'
            '\n'
            "# Queda una tercera posibilidad que conviene descartar aquí mismo: que el\n"
            "# negativo venga de MÁS ARRIBA (sustracción de cielo en A2/ZAP) y estas\n"
            "# restas solo lo hereden. Se mide el nivel del propio cubo lejos de las\n"
            "# dos fuentes: si el cubo ya llegara negativo, el problema no sería de C3.\n"
            "R_LEJOS_ESTRELLA_PX, R_LEJOS_COMPANERO_PX = 60.0, 20.0\n"
            "yy_f, xx_f = np.indices(LS_CUBE.shape[1:], dtype=float)\n"
            "lejos = ((np.hypot(yy_f - STAR_YX[0], xx_f - STAR_YX[1]) > R_LEJOS_ESTRELLA_PX)\n"
            "         & (np.hypot(yy_f - OBJECT_YX[0], xx_f - OBJECT_YX[1]) > R_LEJOS_COMPANERO_PX))\n"
            "print()\n"
            "print(f'nivel del CUBO en los MISMOS {int(lejos.sum())} píxeles'\n"
            "      f' (r>{R_LEJOS_ESTRELLA_PX:.0f} px de la primaria y'\n"
            "      f' r>{R_LEJOS_COMPANERO_PX:.0f} px del compañero),'\n"
            "      ' medido en dos bandas:')\n"
            "for etiqueta, banda in (('azul', BANDA_AZUL), ('rojo', BANDA_ROJA)):\n"
            "    sel_b = (WAVE >= banda[0]) & (WAVE <= banda[1])\n"
            "    img_b = np.nanmedian(LS_CUBE[sel_b], axis=0)\n"
            "    print(f'  {etiqueta} {banda[0]:.0f}-{banda[1]:.0f} Å:'\n"
            "          f' {float(np.nanmedian(img_b[lejos])):8.3f} por píxel')\n"
            "print('  (misma región en las dos filas: lo que cambia es la banda, no la')\n"
            "print('  distancia. Y NO es un nivel de cielo: ahí fuera todavía hay halo,')\n"
            "print('  por eso el rojo sale por encima del azul.)')\n"
            "print('  si sale POSITIVO, el cubo no llega negativo a esta etapa: el signo lo')\n"
            "print('  mete la resta LOCAL de aquí (anillo en ls, modelo de PSF en psfsub),')\n"
            "print('  no la sustracción de cielo de A2/A3.')\n"
        ),
        md(
            "## 13 · Comparación con la cadena\n\n"
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
        md(
            "## 14 · Figura de paper y tabla — las dos variantes\n\n"
            + mb.PAPER_SPECTRUM_MD.split("\n\n", 1)[1]
            + "\n\n> Aquí sale **dos veces**, una por variante, de los números recalculados en "
            "este notebook (no del producto de la cadena): si has tocado una perilla, la figura "
            "y la tabla la llevan. Ficheros con sufijo `_debug`, que no pisan nada."
        ),
        code(
            mb.paper_spectrum_cell(
                arrays_code=(
                    "    ROOT_P = ROOT\n"
                    "    METHOD_P = 'optimal_ls'\n"
                    "    PRODUCT_P = 'recalculado en C3_optimal_debug (variante ls)'\n"
                    "    TARGET_P = str(TARGET).replace(' ', '') + '_debug'\n"
                    "    BUNIT_P = BUNIT or 'ADU'\n"
                    "    W_P, F_P = WAVE, LS['flux']\n"
                    "    E_P, E_ALT_P = LS['err_emp'], LS['err_stat']\n"
                    "    EXTRA_P = {'flux_err_stat': LS['err_stat'], 'apcorr': LS['apcorr'],\n"
                    "               'flags': channel_flags(WAVE, bad_windows_A=BAD_WINDOWS_A,\n"
                    "                                      skyline_windows_A=SKYLINE_WINDOWS_A,\n"
                    "                                      interpolated_windows_A=INTERPOLATED_WIN_A)}\n"
                    "    MODO_P = LS['modo']\n"
                ),
                subdir="c3_optimal_debug",
                stem="spectrum_paper_ls",
                err_label="±1σ empírico (controles procesados igual)",
                err_alt_label="±1σ propagado del STAT (no es σ)",
                title_suffix="optimal_ls, rehecha en el notebook (C3 debug)",
            )
        ),
        code(
            mb.paper_spectrum_cell(
                arrays_code=(
                    "    ROOT_P = ROOT\n"
                    "    METHOD_P = 'optimal_psfsub'\n"
                    "    PRODUCT_P = 'recalculado en C3_optimal_debug (variante psfsub)'\n"
                    "    TARGET_P = str(TARGET).replace(' ', '') + '_debug'\n"
                    "    BUNIT_P = BUNIT or 'ADU'\n"
                    "    W_P, F_P = WAVE, PSFSUB['flux']\n"
                    "    E_P, E_ALT_P = PSFSUB['err_emp'], PSFSUB['err_stat']\n"
                    "    EXTRA_P = {'flux_err_stat': PSFSUB['err_stat'], 'apcorr': PSFSUB['apcorr'],\n"
                    "               'flags': channel_flags(WAVE, bad_windows_A=BAD_WINDOWS_A,\n"
                    "                                      skyline_windows_A=SKYLINE_WINDOWS_A,\n"
                    "                                      interpolated_windows_A=INTERPOLATED_WIN_A)}\n"
                    "    MODO_P = PSFSUB['modo']\n"
                ),
                subdir="c3_optimal_debug",
                stem="spectrum_paper_psfsub",
                err_label="±1σ empírico (controles procesados igual)",
                err_alt_label="±1σ propagado del STAT (no es σ)",
                title_suffix="optimal_psfsub, rehecha en el notebook (C3 debug)",
            )
        ),
        md(mb.STAR_REFERENCE_MD),
        code(
            mb.paper_from_product_cell(
                product="spec_psffit_star.fits",
                method="psffit_star",
                subdir="c3_optimal_debug",
                stem="spectrum_paper_star_ref",
                qc="stages/spec_psffit_qc.json",
                title_suffix="espectro de la PRIMARIA (producto de C4, referencia)",
            )
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
            "# Resolución de las figuras EN PANTALLA. `savefig` guarda a 300 dpi, pero\n"
            "# lo que se ve dentro del notebook lo fija el backend inline, que va a 100\n"
            "# dpi por defecto y sale borroso. `retina` dobla los píxeles sin cambiar el\n"
            "# tamaño aparente; fuera de IPython no hace nada y queda el rcParam.\n"
            "import matplotlib as mpl\n"
            "mpl.rcParams['figure.dpi'] = 120\n"
            "mpl.rcParams['savefig.dpi'] = 200\n"
            "try:\n"
            "    from matplotlib_inline.backend_inline import set_matplotlib_formats\n"
            "    set_matplotlib_formats('retina')\n"
            "except Exception:\n"
            "    pass\n"
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
            "INTERPOLATED_WIN_A = X03.get('x03_interpolated_windows_A', [])\n"
            "# Solo para el contraste de la seccion 8: psffit no usa apcorr (da el\n"
            "# flujo total directamente), pero la apertura con la que se compara si.\n"
            "APCORR_MODE       = X03.get('x03_aperture_correction', 'auto')\n\n"
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
            "CUBE_PATH = SD / 'stage02_xcorr_cube_stack.fits'\n"
            "with fits.open(CUBE_PATH) as h:\n"
            "    CUBE_FULL = np.asarray(h['CUBES'].data, dtype=float)\n"
            "    WAVE_FULL = np.asarray(h['WAVELENGTH'].data, dtype=float)\n"
            "    STAT_FULL = np.asarray(h['STAT'].data, dtype=float) if 'STAT' in h else None\n"
            "    _stack_bunit = str(h[0].header.get('BUNIT', '')\n"
            "                       or h['CUBES'].header.get('BUNIT', '')) or None\n"
            "# La unidad, con la regla de la cadena: el stack de B2 no la declara y\n"
            "# `resolve_bunit` cae al cubo de entrada del run.\n"
            "from musepipe.io import resolve_bunit\n"
            "BUNIT = resolve_bunit(X03, stack_bunit=_stack_bunit)\n"
            "UNIDAD = BUNIT or 'sin unidad declarada'\n"
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
            f"print(f'STAT     : factor={{STAT_FACTOR:.3f}} covarianza={{COV_FACTOR:.3f}} estado={{STAT_STATUS}}')\n"
            "print('BUNIT    :', UNIDAD)"
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
            "# `evaluate_psf_model` (C1) y `run_channel_chunks` (paralelismo) se importan\n"
            "# arriba: no son lo que se ajusta aquí.\n\n"
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
            "from matplotlib.colors import LogNorm\n\n"
            "mask = fit_region_mask(CUBE.shape[1:], STAR_YX, COMP_YX,\n"
            "                       star_radius_px=STAR_RADIUS_PX, comp_radius_px=COMP_RADIUS_PX)\n"
            "iz = int(np.argmin(np.abs(WAVE - 7500)))\n"
            "design = psf_pair_design(CUBE.shape[1:], WAVE[iz], STAR_YX, COMP_YX, PSF_MODEL)\n"
            "print('píxeles en la región de ajuste:', int(mask.sum()))\n\n"
            "ys, xs = np.nonzero(mask)\n"
            "sl = (slice(max(ys.min() - 3, 0), ys.max() + 4),\n"
            "      slice(max(xs.min() - 3, 0), xs.max() + 4))\n"
            "sy0, sx0 = sl[0].start, sl[1].start\n"
            "# `psf_pair_design` apila las columnas en el ÚLTIMO eje: la forma es\n"
            "# (ny, nx, 5), no (5, ny, nx). Con `design[i]` se cogía la fila i de la\n"
            "# imagen, no la componente i, y los paneles salían en blanco.\n"
            "print('forma de la matriz de diseño:', design.shape,\n"
            "      '-> columnas: PSF primaria, PSF compañero, constante, y, x')\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))\n"
            "axes[0].imshow(mask[sl], origin='lower', cmap='gray')\n"
            "axes[0].set_title('región de ajuste (unión de dos discos)', fontsize=9)\n"
            "# Las dos PSF comparten escala LOG y los MISMOS límites: la pregunta es\n"
            "# si se parecen dentro de la máscara, y con escalas distintas dos\n"
            "# perfiles diferentes se ven idénticos.\n"
            "compos = [design[..., 0][sl], design[..., 1][sl]]\n"
            "juntos = np.concatenate([c[np.isfinite(c) & (c > 0)].ravel() for c in compos])\n"
            "norma = LogNorm(vmin=max(float(np.nanpercentile(juntos, 55)), 1e-12),\n"
            "                vmax=float(np.nanmax(juntos)))\n"
            "for ax, img, t in ((axes[1], compos[0], 'PSF de la primaria'),\n"
            "                   (axes[2], compos[1], 'PSF del compañero')):\n"
            "    im = ax.imshow(img, origin='lower', cmap='magma', norm=norma)\n"
            "    ax.contour(mask[sl], levels=[0.5], colors='tab:cyan', linewidths=0.8)\n"
            "    ax.set_title(f'{t}  ·  LOG', fontsize=9)\n"
            "    cb = fig.colorbar(im, ax=ax, fraction=0.046)\n"
            "    cb.set_label('PSF normalizada [adim.]', fontsize=7)\n"
            "for ax in axes:\n"
            "    ax.plot(STAR_YX[1] - sx0, STAR_YX[0] - sy0, '*', color='w', ms=11, mec='k')\n"
            "    ax.plot(COMP_YX[1] - sx0, COMP_YX[0] - sy0, '+', color='tab:cyan', ms=9)\n"
            "    ax.set_xlabel('x [px]')\n"
            "axes[0].set_ylabel('y [px]')\n"
            "fig.suptitle(f'λ = {WAVE[iz]:.0f} Å · contorno cian = borde de la región de ajuste',\n"
            "             fontsize=9)\n"
            "fig.tight_layout(); plt.show()\n\n"
            "# El número que resume el panel: cuánto se parecen DENTRO de la máscara.\n"
            "a = design[..., 0][mask]; b = design[..., 1][mask]\n"
            "fin = np.isfinite(a) & np.isfinite(b)\n"
            "coseno = float(np.dot(a[fin], b[fin])\n"
            "               / np.sqrt(np.dot(a[fin], a[fin]) * np.dot(b[fin], b[fin])))\n"
            "print(f'solape de las dos columnas en la máscara: {coseno:.4f}'\n"
            "      '   (1 = indistinguibles, 0 = ortogonales)')"
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
            "el de una apertura.\n\n"
            "### La primaria, contrastada con fotometría de apertura\n\n"
            "El flujo de la primaria es la única de las dos componentes que se puede **verificar "
            "por otro camino**: es brillante y está aislada, así que una **apertura circular "
            "simple** —la misma `aperture_spectrum` que usa C2, con la corrección de apertura de "
            "la PSF de C1— debe dar lo mismo. Si las dos curvas se separan, el problema no está "
            "en el compañero: está en el modelo de PSF o en la curva de crecimiento, y entonces "
            "el flujo del compañero (que **no** se puede contrastar así) hereda ese error.\n\n"
            "Por eso van juntas, con la diferencia debajo. Es el mismo espíritu que el chequeo "
            "`v3_star_scale` de la spec, pero con la cuenta a la vista y con el radio de la "
            "apertura como perilla: si la corrección de apertura fuera correcta, **el resultado "
            "no debería depender del radio**.\n\n"
            "### Sobre el suavizado\n\n"
            "Sí, las curvas llevan **mediana móvil** (`SUAVIZADO_CH` canales) — antes iba fija a "
            "11 y no se decía, que es justo lo que hace que un espectro parezca mejor de lo que "
            "es. Ahora el dato **por canal** va detrás en gris y el suavizado es una perilla: "
            "ponla a 1 y se ve el espectro crudo.\\n\\n"
            "> Y hay un **segundo** motivo por el que esto se ve más liso que el mismo espectro "
            "en C2: el notebook ajusta **1 de cada `PASO_CANALES` canales** (20 por defecto). "
            "La línea «por canal» son ~175 puntos, no 3681. El submuestreo no cambia el valor de "
            "ningún canal —el ajuste es independiente canal a canal, y por eso la comparación "
            "con la cadena sale idéntica— pero sí cambia el aspecto. Pon `PASO_CANALES = 1` y "
            "unos 11 minutos si quieres verlo entero."
        ),
        code(
            "from musepipe.spectral import median_filter_1d\n\n"
            "SUAVIZADO_CH = 11        # 1 = sin suavizar, y se ve el dato crudo\n"
            "RADIO_APERTURA_PX = 10.0 # apertura circular sobre la primaria\n\n"
            "comp_flux = res.coeffs[:, 1]\n"
            "star_flux = res.coeffs[:, 0]\n"
            "suave = lambda v: (median_filter_1d(v, SUAVIZADO_CH) if SUAVIZADO_CH > 1\n"
            "                   else np.asarray(v, dtype=float))\n\n"
            "# --- el otro camino: fotometría de apertura sobre la primaria ---\n"
            "# Misma función que usa C2 para el compañero, y la misma corrección de\n"
            "# apertura de la PSF de C1, para que las dos curvas signifiquen lo mismo\n"
            "# (flujo TOTAL de la fuente) y sean comparables sin más.\n"
            "APERTURA_STAR = {'kind': 'circle', 'radius_px': float(RADIO_APERTURA_PX),\n"
            "                 'name': f'star_r{RADIO_APERTURA_PX:g}'}\n"
            "star_ap_raw, star_npix = aperture_spectrum(CUBE, STAR_YX, APERTURA_STAR)\n"
            "star_apcorr, _modo_ap, _nr = aperture_correction_from_psf(\n"
            "    WAVE, APERTURA_STAR, PSF_MODEL, center_yx=STAR_YX, correction_mode=APCORR_MODE)\n"
            "star_ap = star_ap_raw * star_apcorr\n"
            "print(f'apertura sobre la primaria: círculo r={RADIO_APERTURA_PX:g} px'\n"
            "      f' ({float(np.nanmedian(star_npix)):.0f} px), apcorr mediano'\n"
            "      f' {float(np.nanmedian(star_apcorr)):.2f}× -> recoge el'\n"
            "      f' {100 / float(np.nanmedian(star_apcorr)):.0f}% de la PSF')\n\n"
            "# Solo donde el ajuste corrió (el submuestreo deja canales sin ajustar).\n"
            "ajustados = np.isfinite(star_flux)\n"
            "dif = np.where(ajustados, star_ap - star_flux, np.nan)\n"
            "rel = 100.0 * dif / np.where(np.abs(star_flux) > 0, star_flux, np.nan)\n"
            "print(f'primaria: psffit vs apertura -> diferencia relativa mediana'\n"
            "      f' {float(np.nanmedian(rel)):+.1f}%'\n"
            "      f' (p16..p84: {float(np.nanpercentile(rel[np.isfinite(rel)], 16)):+.1f}'\n"
            "      f' .. {float(np.nanpercentile(rel[np.isfinite(rel)], 84)):+.1f}%)')\n\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.0), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.plot(WAVE, star_flux, lw=0.3, color='0.8')\n"
            "a1.plot(WAVE, star_ap, lw=0.3, color='0.8')\n"
            "a1.plot(WAVE, suave(star_flux), lw=1.2, color='k', label='psffit (ajuste de dos PSF)')\n"
            "a1.plot(WAVE, suave(star_ap), lw=1.2, color='tab:orange',\n"
            "        label=f'apertura r={RADIO_APERTURA_PX:g} px × apcorr')\n"
            "a1.set_ylabel(f'primaria [{UNIDAD}]')\n"
            "a1.set_title(f'la primaria por dos caminos independientes'\n"
            "             f'  (líneas: mediana móvil de {SUAVIZADO_CH} ch; gris: por canal)',\n"
            "             fontsize=9)\n"
            "a1.legend(fontsize=8)\n"
            "a2.axhline(0, color='0.6', lw=0.7)\n"
            "a2.plot(WAVE, dif, lw=0.3, color='0.8')\n"
            "a2.plot(WAVE, suave(dif), lw=1.1, color='tab:purple')\n"
            "fin_d = np.isfinite(dif)\n"
            "if fin_d.any():\n"
            "    a2.set_ylim(*np.nanpercentile(dif[fin_d], [1, 99]))\n"
            "a2.set_ylabel(f'apertura − psffit\\n[{UNIDAD}]', fontsize=8)\n"
            "a2.set_xlabel('λ [Å]')\n"
            "fig.tight_layout(); plt.show()\n\n"
            "fig, ax = plt.subplots(figsize=(11, 3.6))\n"
            "ax.fill_between(WAVE, -comp_err_emp, comp_err_emp, color='0.85',\n"
            "                label='±σ empírico (controles)')\n"
            "ax.plot(WAVE, comp_flux, lw=0.3, color='0.6', label='por canal (sin suavizar)')\n"
            "if SUAVIZADO_CH > 1:\n"
            "    ax.plot(WAVE, suave(comp_flux), lw=1.2, color='tab:blue',\n"
            "            label=f'mediana móvil {SUAVIZADO_CH} ch')\n"
            "ax.axvline(6562.8, color='tab:red', ls=':', label='Hα')\n"
            "fin_c = np.isfinite(comp_flux)\n"
            "if fin_c.any():\n"
            "    ax.set_ylim(*np.nanpercentile(comp_flux[fin_c], [1, 99]))\n"
            "ax.set_ylabel(f'compañero [{UNIDAD}]'); ax.set_xlabel('λ [Å]')\n"
            "ax.set_title('el compañero: el dato por canal, y su mediana móvil encima', fontsize=9)\n"
            "ax.legend(fontsize=8); fig.tight_layout(); plt.show()\n"
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
        md(
            "## 10 · Figura de paper y tabla — el compañero y la primaria\n\n"
            + mb.PAPER_SPECTRUM_MD.split("\n\n", 1)[1]
            + "\n\n> Dos veces: el compañero (el espectro canónico de la cadena) y la "
            "primaria. Salen de los números recalculados aquí, con sufijo `_debug`.\n\n"
            "> **Ojo con la rejilla**: el notebook ajusta 1 de cada `PASO_CANALES` canales, "
            "así que la figura y la tabla llevan esos ~175 puntos, no los 3681. Pon "
            "`PASO_CANALES = 1` para exportar el espectro completo.\n\n"
            "> Y la línea fina no es el STAT propagado sino el **error formal del ajuste** "
            "(la diagonal de la matriz de covarianza): en psffit es esa la segunda "
            "estimación con la que se contrasta el empírico."
        ),
        code(
            mb.paper_spectrum_cell(
                arrays_code=(
                    "    ROOT_P = ROOT\n"
                    "    METHOD_P = 'psffit'\n"
                    "    PRODUCT_P = 'recalculado en C4_psffit_debug (compañero)'\n"
                    "    TARGET_P = str(TARGET).replace(' ', '') + '_debug'\n"
                    "    BUNIT_P = BUNIT or 'ADU'\n"
                    "    W_P, F_P = WAVE, comp_flux\n"
                    "    E_P, E_ALT_P = comp_err_emp, np.sqrt(np.clip(comp_var, 0.0, np.inf))\n"
                    "    EXTRA_P = {'flux_err_stat': np.sqrt(np.clip(comp_var, 0.0, np.inf))}\n"
                    "    MODO_P = modo\n"
                ),
                subdir="c4_psffit_debug",
                stem="spectrum_paper",
                err_label="±1σ empírico (controles procesados igual)",
                err_alt_label="±1σ formal del ajuste (matriz de covarianza)",
                title_suffix="psffit del compañero, rehecho en el notebook (C4 debug)",
            )
        ),
        code(
            mb.paper_spectrum_cell(
                arrays_code=(
                    "    ROOT_P = ROOT\n"
                    "    METHOD_P = 'psffit_star'\n"
                    "    PRODUCT_P = 'recalculado en C4_psffit_debug (primaria)'\n"
                    "    TARGET_P = str(TARGET).replace(' ', '') + '_debug'\n"
                    "    BUNIT_P = BUNIT or 'ADU'\n"
                    "    W_P, F_P = WAVE, star_flux\n"
                    "    E_P, E_ALT_P = star_err_emp, np.sqrt(np.clip(star_var, 0.0, np.inf))\n"
                    "    EXTRA_P = {'flux_err_stat': np.sqrt(np.clip(star_var, 0.0, np.inf))}\n"
                    "    MODO_P = modo\n"
                ),
                subdir="c4_psffit_debug",
                stem="spectrum_paper_star",
                err_label="±1σ empírico (controles procesados igual)",
                err_alt_label="±1σ formal del ajuste (matriz de covarianza)",
                title_suffix="psffit de la PRIMARIA, rehecho en el notebook (C4 debug)",
            )
        ),
        md(mb.STAR_REFERENCE_MD),
        code(
            mb.paper_from_product_cell(
                product="spec_psffit_star.fits",
                method="psffit_star",
                subdir="c4_psffit_debug",
                stem="spectrum_paper_star_ref",
                qc="stages/spec_psffit_qc.json",
                title_suffix="espectro de la PRIMARIA (producto de C4, referencia)",
            )
        ),
    ]


#: Lo propio de cada metodo de halo, para no duplicar el constructor.
HALOSUB_META = {
    "C5": {
        "slug": "C5_sgf_debug", "metodo": "sgf", "prefijo": "x04",
        "titulo": "sustracción de halo SGF",
        "producto": "spec_sgf_object.fits", "qc": "spec_sgf_qc.json",
        "spec": "spec_C5_codex_sgf_subtraction.md",
        "config": "from musepipe.stages.stage_x04_sgf import stage_x04_config_from_run as _cfg_from_run",
        "que_hace": (
            "**SGF** (Haffert+19; Julo+25 App. A.3) explota la **diversidad espectral**: el halo de "
            "la estrella tiene la misma forma espectral en todos los spaxels, y el compañero no. "
            "Cada spaxel se divide por el espectro estelar de referencia, se **suaviza con un filtro "
            "Savitzky-Golay** —que sigue lo lento y se come lo estrecho— y lo suavizado se toma como "
            "el halo y se resta.\n\n"
            "> Por construcción, **el SGF se lleva también el continuo del compañero**: lo que "
            "sobrevive es la línea, no el nivel. Eso no es un defecto, es su definición — y es la "
            "razón de que su continuo no sea comparable con el de los demás métodos (aviso "
            "automático en la tabla de D2)."
        ),
        "perillas": (
            "# OJO: las perillas del método NO llevan prefijo de etapa (`sgf_*`, no `x04_*`).\n"
            "SGF_WINDOW = int(X0.get('sgf_window', 101))   # ancho del filtro, en canales\n"
            "SGF_DEGREE = int(X0.get('sgf_degree', 1))     # grado del polinomio local\n"
        ),
        "sustraccion_md": (
            "## 7 · La sustracción SGF\n\n"
            "Por spaxel: `cociente = espectro / referencia`, se filtra con Savitzky-Golay de "
            "`SGF_WINDOW` canales y grado `SGF_DEGREE`, y el halo estimado es "
            "`suavizado × referencia`. El residual es lo que queda.\n\n"
            "La ventana es **la** perilla: cuanto más ancha, menos se come de la línea, pero peor "
            "sigue las variaciones lentas del halo."
        ),
        "sustraccion_code": (
            "res = sgf_subtract(CUBE, s_hat, window=SGF_WINDOW, degree=SGF_DEGREE)\n"
            "residual = res.residual_cube\n"
            "print('ventana', SGF_WINDOW, 'canales · grado', SGF_DEGREE)\n"
            "yc, xc = int(round(OBJECT_YX[0])), int(round(OBJECT_YX[1]))\n"
            "fig, ax = plt.subplots(figsize=(11, 3.6))\n"
            "ax.plot(WAVE, CUBE[:, yc, xc], lw=0.5, color='0.6', label='spaxel del compañero (crudo)')\n"
            "ax.plot(WAVE, CUBE[:, yc, xc] - residual[:, yc, xc], lw=1.0, color='tab:orange',\n"
            "        label='halo estimado por el SGF')\n"
            "ax.plot(WAVE, residual[:, yc, xc], lw=0.8, color='tab:blue', label='residual')\n"
            "ax.axvline(6563, color='tab:red', ls=':', label='Hα')\n"
            "ax.set_xlabel('λ [Å]'); ax.legend(fontsize=8)\n"
            "ax.set_title('SGF en el spaxel central del compañero', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
    },
    "C6": {
        "slug": "C6_lpm_debug", "metodo": "lpm", "prefijo": "x05",
        "titulo": "sustracción de halo LPM",
        "producto": "spec_lpm_object.fits", "qc": "spec_lpm_qc.json",
        "spec": "spec_C6_codex_lpm_subtraction.md",
        "config": "from musepipe.stages.stage_x05_lpm import stage_x05_config_from_run as _cfg_from_run",
        "que_hace": (
            "**LPM** (Julo+25 App. A.4) parte de la misma idea que el SGF pero **no filtra**: modela "
            "cada spaxel como el espectro de referencia **modulado por un polinomio de Legendre** de "
            "grado bajo,\n\n"
            "> `ŝ_xy(λ) = Σ_k β_k · P_k(λ̃) · ŝ(λ)`\n\n"
            "y ajusta los `β_k` por mínimos cuadrados **con las líneas de ciencia enmascaradas**. "
            "Esa máscara es la diferencia clave con el SGF: como la línea no entra en el ajuste, el "
            "modelo no puede aprenderla, y por eso **el LPM preserva la línea** (su chequeo "
            "`v2_line_preservation_ok` exige recuperar ≥90% de una línea inyectada)."
        ),
        "perillas": (
            "# OJO: la perilla del método NO lleva prefijo de etapa (`lpm_degree`, no `x05_*`).\n"
            "LPM_DEGREE = int(X0.get('lpm_degree', 4))   # grado del polinomio de Legendre\n"
        ),
        "sustraccion_md": (
            "## 7 · La sustracción LPM\n\n"
            "La matriz de diseño tiene una columna por grado: `P_k(λ̃) · ŝ(λ)`. El ajuste es por "
            "mínimos cuadrados sobre los canales **no enmascarados** (fuera de Hα, Hβ, O I…), y el "
            "modelo se evalúa después en **todos** los canales — incluidos los de la línea, que es "
            "donde se quiere que no haya aprendido nada.\n\n"
            "Subir el grado sigue mejor el halo pero se acerca a poder absorber la línea; es la "
            "perilla que el diagnóstico de energía por grado vigila."
        ),
        "sustraccion_code": (
            "mask_fit = lpm_fit_mask(WAVE, s_hat)\n"
            "res = lpm_subtract(CUBE, WAVE, s_hat, degree=LPM_DEGREE)\n"
            "residual = res.residual_cube\n"
            "print('grado', LPM_DEGREE, '| canales usados en el ajuste:', int(mask_fit.sum()),\n"
            "      f'de {WAVE.size} ({100 * mask_fit.mean():.1f}%)')\n"
            "print('energía por grado:', np.round(lpm_coefficient_energy_share(res.coeffs), 3))\n"
            "yc, xc = int(round(OBJECT_YX[0])), int(round(OBJECT_YX[1]))\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)\n"
            "design = lpm_design_matrix(WAVE, s_hat, degree=LPM_DEGREE)\n"
            "for k in range(design.shape[1]):\n"
            "    a1.plot(WAVE, design[:, k], lw=0.8, label=f'P{k}(λ̃)·ŝ')\n"
            "a1.set_ylabel('columnas del diseño'); a1.legend(fontsize=7, ncol=5)\n"
            "a1.set_title('la base: la referencia modulada por Legendre', fontsize=9)\n"
            "a2.plot(WAVE, CUBE[:, yc, xc], lw=0.5, color='0.6', label='spaxel del compañero (crudo)')\n"
            "a2.plot(WAVE, CUBE[:, yc, xc] - residual[:, yc, xc], lw=1.0, color='tab:orange',\n"
            "        label='halo modelado')\n"
            "a2.plot(WAVE, residual[:, yc, xc], lw=0.8, color='tab:blue', label='residual')\n"
            "a2.fill_between(WAVE, *a2.get_ylim(), where=~mask_fit, color='tab:red', alpha=0.10,\n"
            "                label='canales EXCLUIDOS del ajuste (líneas)')\n"
            "a2.axvline(6563, color='tab:red', ls=':')\n"
            "a2.set_xlabel('λ [Å]'); a2.legend(fontsize=8)\n"
            "fig.tight_layout(); plt.show()"
        ),
    },
}


def build_halosub_cells(mb, target, run_id, stage_id):
    """Celdas de C5/C6: mismo esqueleto, distinta sustracción."""
    md, code = mb.md, mb.code
    meta = HALOSUB_META[stage_id]
    sources = extract_sources(stage_id)
    inline_src = "\n\n\n".join(src for _rel, _name, src, _sha in sources)
    shas = {f"{rel}:{name}": sha for rel, name, _src, sha in sources}
    pref = meta["prefijo"]

    return [
        md(
            f"# {stage_id} · {meta['titulo']} — notebook de análisis (`debug`)\n\n"
            f"**Objeto:** {target}  |  **Run:** `{run_id}`  |  "
            f"**Spec:** [`docs/{meta['spec']}`](../../../docs/{meta['spec']})\n\n"
            + meta["que_hace"] + "\n\n"
            "El esqueleto de la etapa son cuatro pasos: **elegir los spaxels de referencia**, "
            "**construir el espectro estelar de referencia**, **restar el halo** (lo propio de cada "
            "método) y **extraer una apertura box3** del cubo residual — esta última con la misma "
            "maquinaria que C2, controles y `apcorr` incluidos."
        ),
        code(
            "import json, sys\n"
            "from pathlib import Path\n\n"
            "import numpy as np\n"
            "from astropy.io import fits\n"
            "import matplotlib.pyplot as plt\n"
            "# Resolución de las figuras EN PANTALLA. `savefig` guarda a 300 dpi, pero\n"
            "# lo que se ve dentro del notebook lo fija el backend inline, que va a 100\n"
            "# dpi por defecto y sale borroso. `retina` dobla los píxeles sin cambiar el\n"
            "# tamaño aparente; fuera de IPython no hace nada y queda el rcParam.\n"
            "import matplotlib as mpl\n"
            "mpl.rcParams['figure.dpi'] = 120\n"
            "mpl.rcParams['savefig.dpi'] = 200\n"
            "try:\n"
            "    from matplotlib_inline.backend_inline import set_matplotlib_formats\n"
            "    set_matplotlib_formats('retina')\n"
            "except Exception:\n"
            "    pass\n"
            "from matplotlib.colors import LogNorm\n\n"
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
            "Del **config resuelto de la etapa**: rellena defaults que el run no escribe. Cambia lo "
            "que quieras debajo de la lectura."
        ),
        code(
            meta["config"] + "\n\n"
            "# `project_root=ROOT`: musepipe resuelve rutas contra el cwd, que en un\n"
            "# notebook es su propia carpeta, no la raíz del repo.\n"
            "X0 = _cfg_from_run(RUN_ID, project_root=ROOT)\n"
            + meta["perillas"] +
            "FLUX_LO = float(X0.get('halosub_flux_mask_lo', 0.10))   # percentil bajo de la máscara\n"
            "FLUX_HI = float(X0.get('halosub_flux_mask_hi', 0.90))   # percentil alto\n"
            "EXCLUDE_RADIUS_PX = float(X0.get('halosub_exclude_radius_px', 3.0))\n"
            f"N_CONTROLS = int(X0.get('{pref}_control_apertures', 8))\n"
            f"EXCLUDE_ANGLE_DEG = float(X0.get('{pref}_control_exclude_angle_deg', 25.0))\n"
            f"APCORR_MODE = X0.get('{pref}_aperture_correction', 'auto')\n"
            f"ERROR_MODE = X0.get('{pref}_error_mode', 'auto')\n"
            f"ANNULUS = X0.get('{pref}_annulus_bkg_px')\n"
            f"BAD_WINDOWS_A = X0.get('{pref}_bad_windows_A', [])\n\n"
            "# ---- a partir de aquí, cambia lo que quieras probar ----\n\n"
            "print('máscara de referencia: percentiles', FLUX_LO, '-', FLUX_HI,\n"
            "      '| excluye', EXCLUDE_RADIUS_PX, 'px alrededor del compañero')\n"
            "print('extracción: controles', N_CONTROLS, '| apcorr', APCORR_MODE,\n"
            "      '| anillo', ANNULUS)"
        ),
        md(
            "## 2 · Entradas\n\n"
            "El cubo de B2 (la etapa trabaja **por exposición** si el run las tiene; aquí se usa el "
            "stack, que es lo que hay en estos runs) y las posiciones de B3."
        ),
        code(
            "qc_b3 = json.loads((SD / 'stage01c_qc.json').read_text(encoding='utf-8'))\n"
            "OBJECT_YX = tuple(float(v) for v in qc_b3['companion']['pos_yx'])\n"
            "STAR_YX   = tuple(float(v) for v in qc_b3['primary']['pos_yx'])\n"
            "psf_path = SD / 'psf_model.json'\n"
            "PSF_MODEL = json.loads(psf_path.read_text(encoding='utf-8')) if psf_path.exists() else None\n\n"
            "CUBE_PATH = SD / 'stage02_xcorr_cube_stack.fits'\n"
            "with fits.open(CUBE_PATH) as h:\n"
            "    CUBE = np.asarray(h['CUBES'].data, dtype=float)\n"
            "    WAVE = np.asarray(h['WAVELENGTH'].data, dtype=float)\n"
            "    STAT_CUBE = np.asarray(h['STAT'].data, dtype=float) if 'STAT' in h else None\n"
            "    _stack_bunit = str(h[0].header.get('BUNIT', '')\n"
            "                       or h['CUBES'].header.get('BUNIT', '')) or None\n"
            "if CUBE.ndim == 4:\n"
            "    CUBE = CUBE[0]\n"
            "if STAT_CUBE is not None and STAT_CUBE.ndim == 4:\n"
            "    STAT_CUBE = STAT_CUBE[0]\n"
            "# La unidad, con la regla de la cadena: el stack de B2 no la declara y\n"
            "# `resolve_bunit` cae al cubo de entrada del run.\n"
            "from musepipe.io import resolve_bunit\n"
            "BUNIT = resolve_bunit(X0, stack_bunit=_stack_bunit)\n"
            "UNIDAD = BUNIT or 'sin unidad declarada'\n\n"
            "qc00 = json.loads((SD / 'stage00q_qc.json').read_text(encoding='utf-8'))\n"
            "qc01 = json.loads((SD / 'stage01_qc.json').read_text(encoding='utf-8'))\n"
            "m5 = qc00.get('m5_stat', {})\n"
            f"STAT_FACTOR = float(X0.get('{pref}_stat_factor_box3', m5.get('factor_box3_median', 1.0)) or 1.0)\n"
            f"COV_FACTOR  = float(X0.get('{pref}_covariance_factor_box3',\n"
            "                            qc01.get('stat', {}).get('covariance_factor_box3', 1.0)) or 1.0)\n"
            f"STAT_STATUS = str(X0.get('{pref}_stat_status', m5.get('status', 'unknown')))\n"
            "print('cubo     :', CUBE.shape, '| compañero', [round(v, 1) for v in OBJECT_YX],\n"
            "      '| primaria', [round(v, 1) for v in STAR_YX])\n"
            "print(f'STAT     : factor={STAT_FACTOR:.3f} covarianza={COV_FACTOR:.3f} estado={STAT_STATUS}')"
        ),
        md(
            "## 3 · Las funciones copiadas de `musepipe`\n\n"
            + "\n".join(f"- `{name}` — de `{rel}`" for rel, name, _s, _h in sources)
        ),
        code(
            "# ------------------------------------------------------------------\n"
            "# COPIA EDITABLE. Fuente: musepipe (ver el chequeo de deriva abajo).\n"
            "# ------------------------------------------------------------------\n"
            + "\n".join(needed_imports(sources)) + "\n"
            "# `evaluate_psf_model` (C1) y el catálogo de líneas se importan arriba:\n"
            "# no son lo que se ajusta aquí.\n\n"
            + "\n".join(needed_constants(sources)) + "\n\n\n"
            + inline_src
        ),
        md("## 4 · Chequeo de deriva"),
        drift_cell(code, shas, stage_id),
        md(
            "## 5 · Paso 1 — los spaxels de referencia\n\n"
            "El halo se estima **con el propio campo**: se eligen los spaxels cuyo flujo está entre "
            "dos percentiles —ni saturados por el núcleo de la estrella ni dominados por el ruido "
            "del borde— **excluyendo un disco alrededor del compañero**, para no meterlo en su "
            "propia referencia. La spec exige ≥50 spaxels: por debajo de eso no hay diversidad "
            "espectral que explotar."
        ),
        code(
            "keep, keep_qc = select_reference_spaxels(\n"
            "    CUBE, flux_lo_frac=FLUX_LO, flux_hi_frac=FLUX_HI,\n"
            "    exclude_yx=[OBJECT_YX], exclude_radius_px=EXCLUDE_RADIUS_PX)\n"
            "print('spaxels conservados:', keep_qc['n_spaxels_kept'],\n"
            "      '| mínimo que exige la spec: 50')\n\n"
            "campo = np.nanmedian(CUBE[::20], axis=0)\n"
            "pos = campo[np.isfinite(campo) & (campo > 0)]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))\n"
            "im1 = a1.imshow(campo, origin='lower', cmap='magma',\n"
            "                norm=LogNorm(vmin=np.percentile(pos, 60), vmax=np.percentile(pos, 99.9)))\n"
            "fig.colorbar(im1, ax=a1, shrink=0.8).set_label('flujo mediano · escala LOG', fontsize=7)\n"
            "a1.set_title('el campo', fontsize=9)\n"
            "a2.imshow(campo, origin='lower', cmap='gray',\n"
            "          norm=LogNorm(vmin=np.percentile(pos, 60), vmax=np.percentile(pos, 99.9)))\n"
            "a2.imshow(np.where(keep, 1.0, np.nan), origin='lower', cmap='cool', alpha=0.55,\n"
            "          vmin=0, vmax=1)\n"
            "a2.plot(OBJECT_YX[1], OBJECT_YX[0], marker='o', ms=9, mfc='none', mec='tab:red', mew=1.5)\n"
            "a2.annotate('compañero (excluido)', (OBJECT_YX[1], OBJECT_YX[0]),\n"
            "            textcoords='offset points', xytext=(0, 11), ha='center',\n"
            "            fontsize=7, color='tab:red')\n"
            "n_keep = int(keep_qc['n_spaxels_kept'])\n"
            "a2.set_title(f'spaxels de referencia ({n_keep}) en color', fontsize=9)\n"
            "for ax in (a1, a2):\n"
            "    ax.set_xlabel('x [px]')\n"
            "a1.set_ylabel('y [px]')\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 6 · Paso 2 — el espectro estelar de referencia\n\n"
            "La mediana de los spaxels elegidos, normalizada. Es **el espectro del halo**: lo que "
            "los dos métodos van a escalar y restar en cada spaxel."
        ),
        code(
            "s_hat = reference_spectrum(CUBE, keep)\n"
            "print('referencia: mediana', round(float(np.nanmedian(s_hat)), 4),\n"
            "      '| canales no finitos:', int((~np.isfinite(s_hat)).sum()))\n"
            "fig, ax = plt.subplots(figsize=(11, 3.2))\n"
            "ax.plot(WAVE, s_hat, lw=0.7)\n"
            "ax.axvline(6563, color='tab:red', ls=':', label='Hα')\n"
            "ax.set_xlabel('λ [Å]'); ax.set_ylabel('referencia ŝ'); ax.legend(fontsize=8)\n"
            "ax.set_title('espectro estelar de referencia (mediana de los spaxels elegidos)', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(meta["sustraccion_md"]),
        code(meta["sustraccion_code"]),
        md(
            "## 8 · Paso 4 — la apertura sobre el residual\n\n"
            "A partir de aquí es **exactamente C2**: caja box3 en la posición de B3 sobre el cubo "
            "residual, controles al mismo radio procesados igual, σ empírico de su dispersión y "
            "`apcorr` de la curva de crecimiento de C1. Por eso estos métodos comparten convención "
            "de flujo con el resto."
        ),
        code(
            "APERTURE = {'kind': 'box', 'size': 3}\n"
            "raw_flux, npix_eff = aperture_spectrum(residual, OBJECT_YX, APERTURE)\n"
            "if ANNULUS is not None:\n"
            "    bkg = annulus_background_spectrum(residual, OBJECT_YX, ANNULUS[0], ANNULUS[1],\n"
            "                                     exclude_yx=STAR_YX,\n"
            "                                     exclude_radius=(ANNULUS[2] if len(ANNULUS) > 2 else 30.0))\n"
            "    raw_flux = raw_flux - bkg * npix_eff\n"
            "controls_yx, control_spectra, control_npix = control_aperture_spectra(\n"
            "    residual, OBJECT_YX, STAR_YX, APERTURE,\n"
            "    n_controls=N_CONTROLS, exclude_angle_deg=EXCLUDE_ANGLE_DEG)\n"
            "raw_err_emp = (robust_sigma_axis0(control_spectra) if control_spectra.shape[0] >= 2\n"
            "               else np.full(WAVE.size, robust_sigma(raw_flux)))\n"
            "usable = (STAT_CUBE is not None and str(ERROR_MODE).lower() != 'empirical'\n"
            "          and STAT_STATUS.lower() != 'red')\n"
            "# El STAT se propaga SIEMPRE que exista, aunque la etapa no lo use como\n"
            "# sigma: es la segunda estimación con la que contrastar el empírico.\n"
            "raw_err_stat = (aperture_stat_error(STAT_CUBE, OBJECT_YX, APERTURE,\n"
            "                                    stat_factor=STAT_FACTOR,\n"
            "                                    covariance_factor=COV_FACTOR)\n"
            "                if STAT_CUBE is not None else None)\n"
            "raw_err = raw_err_stat if usable else np.asarray(raw_err_emp, float)\n"
            "apcorr, apcorr_mode, _nr = aperture_correction_from_psf(\n"
            "    WAVE, APERTURE, PSF_MODEL, center_yx=OBJECT_YX, correction_mode=APCORR_MODE)\n"
            "flux = raw_flux * apcorr\n"
            "flux_err_emp = np.asarray(raw_err_emp, float) * apcorr\n"
            "flux_err_stat = None if raw_err_stat is None else raw_err_stat * apcorr\n"
            "modo_err = 'stat' if usable else 'empirical'\n"
            "print(f'{len(controls_yx)} controles | modo error: {modo_err}'\n"
            "      f' | apcorr mediana {float(np.nanmedian(apcorr)):.1f}')\n\n"
            "def binea(wave, flux, err, n):\n"
            "    n = int(n); corte = (wave.size // n) * n\n"
            "    w = wave[:corte].reshape(-1, n); f = flux[:corte].reshape(-1, n)\n"
            "    e = err[:corte].reshape(-1, n)\n"
            "    bueno = np.isfinite(f) & np.isfinite(e); cuenta = bueno.sum(axis=1)\n"
            "    with np.errstate(invalid='ignore', divide='ignore'):\n"
            "        wb = np.nanmean(np.where(bueno, w, np.nan), axis=1)\n"
            "        fb = np.nansum(np.where(bueno, f, 0.0), axis=1) / np.maximum(cuenta, 1)\n"
            "        eb = np.sqrt(np.nansum(np.where(bueno, e, 0.0) ** 2, axis=1)) / np.maximum(cuenta, 1)\n"
            "    fb[cuenta == 0] = np.nan; eb[cuenta == 0] = np.nan\n"
            "    return wb, fb, eb\n\n"
            "BIN_CANALES = 25\n"
            "N_EFF_OVER_N = 0.43   # medido por G1 (docs/noise_model.md)\n"
            "wb, fb, eb = binea(WAVE, flux, flux_err_emp, BIN_CANALES)\n"
            "rojo = (wb >= 7500) & (wb <= 9000)\n"
            "print(f'flujo mediano en 7500–9000 Å: {np.nanmedian(fb[rojo]):9.2f}'\n"
            "      f' ± {np.nanmedian(eb[rojo] / np.sqrt(N_EFF_OVER_N)):.2f}')\n"
            "fig, ax = plt.subplots(figsize=(11, 3.8))\n"
            "for lo, hi in BAD_WINDOWS_A:\n"
            "    ax.axvspan(lo, hi, color='0.85', zorder=0)\n"
            "ax.plot(WAVE, flux, lw=0.3, color='0.75', label='por canal')\n"
            "ax.errorbar(wb, fb, yerr=eb / np.sqrt(N_EFF_OVER_N), fmt='o', ms=3, lw=0.9,\n"
            "            color='tab:blue', label=f'binado {BIN_CANALES} ch, ±σ corregido por n_eff')\n"
            "ax.axhline(0, color='0.5', lw=0.7); ax.axvline(6563, color='tab:red', ls=':', label='Hα')\n"
            "ax.set_ylim(*np.nanpercentile(fb[np.isfinite(fb)], [1, 99]) * np.array([2.5, 2.5]))\n"
            "ax.set_xlabel('λ [Å]'); ax.legend(fontsize=8)\n"
            f"ax.set_title('{stage_id} rehecho en el notebook', fontsize=9)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 9 · Comparación con la cadena\n\n"
            f"Contra `{meta['producto']}`. Con las perillas por defecto debe salir idéntico; si "
            "cambias la ventana del filtro, el grado o la máscara de referencia, aquí se ve cuánto "
            "se movió."
        ),
        code(
            "from musepipe.extraction.product import SpectrumProduct\n"
            "from musepipe.spectral import median_filter_1d\n\n"
            f"cadena = SpectrumProduct.read(SD / '{meta['producto']}')\n"
            "ref_flux = np.asarray(cadena.flux, float)\n"
            "ok = True\n"
            "for clave, a, b in (('flujo', flux, ref_flux),\n"
            "                    ('apcorr', apcorr, np.asarray(cadena.apcorr, float))):\n"
            "    fin = np.isfinite(a) & np.isfinite(b)\n"
            "    ig = np.isclose(a[fin], b[fin], rtol=1e-9, atol=0.0)\n"
            "    print(f'  {clave:7s} idénticos {100 * ig.mean():6.2f}% de {fin.sum()} canales'\n"
            "          f' | máx |Δ| = {np.abs(a - b)[fin].max():.3e}')\n"
            "    ok &= bool(ig.all())\n"
            "print()\n"
            "print('IDÉNTICO: la copia reproduce la cadena.' if ok else\n"
            "      'DIFIERE — si has tocado una perilla, es lo esperado; si no, revisa el chequeo de deriva.')\n\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.plot(WAVE, median_filter_1d(ref_flux, 41), lw=1.6, color='0.6', label='cadena')\n"
            "a1.plot(WAVE, median_filter_1d(flux, 41), lw=1.0, ls='--', color='tab:blue',\n"
            "        label='este notebook')\n"
            "a1.legend(fontsize=8); a1.set_ylabel('flujo (mediana 41 ch)')\n"
            "a2.plot(WAVE, flux - ref_flux, lw=0.7, color='tab:purple')\n"
            "a2.axhline(0, color='0.7', lw=0.6)\n"
            "a2.set_ylabel('este − cadena'); a2.set_xlabel('λ [Å]')\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 10 · Figura de paper y tabla\n\n"
            + mb.PAPER_SPECTRUM_MD.split("\n\n", 1)[1]
            + "\n\n> Sale de los números recalculados aquí, no del producto de la cadena: "
            "si has tocado una perilla, la figura y la tabla la llevan. Ficheros con sufijo "
            "`_debug`, que no pisan lo que exporta el notebook de auditoría."
        ),
        code(
            mb.paper_spectrum_cell(
                arrays_code=(
                    "    ROOT_P = ROOT\n"
                    f"    METHOD_P = {meta['metodo']!r}\n"
                    f"    PRODUCT_P = 'recalculado en {meta['slug']}'\n"
                    "    TARGET_P = str(TARGET).replace(' ', '') + '_debug'\n"
                    "    BUNIT_P = BUNIT or 'ADU'\n"
                    "    W_P, F_P = WAVE, flux\n"
                    "    E_P, E_ALT_P = flux_err_emp, flux_err_stat\n"
                    "    EXTRA_P = {'flux_err_stat': flux_err_stat, 'apcorr': apcorr,\n"
                    "               'npix_eff': npix_eff,\n"
                    "               'flags': channel_flags(WAVE, bad_windows_A=BAD_WINDOWS_A)}\n"
                    "    MODO_P = modo_err\n"
                ),
                subdir=meta["slug"].lower(),
                err_label="±1σ empírico (controles procesados igual)",
                err_alt_label="±1σ propagado del STAT (no es σ)",
                title_suffix=f"{meta['metodo']}, rehecho en el notebook",
            )
        ),
        md(mb.STAR_REFERENCE_MD),
        code(
            mb.paper_from_product_cell(
                product="spec_psffit_star.fits",
                method="psffit_star",
                subdir=meta["slug"].lower(),
                stem="spectrum_paper_star_ref",
                qc="stages/spec_psffit_qc.json",
                title_suffix="espectro de la PRIMARIA (producto de C4, referencia)",
            )
        ),
    ]


BUILDERS = {
    "C2": ("C2_aperture_debug", build_c2_cells),
    "C3": ("C3_optimal_debug", build_c3_cells),
    "C4": ("C4_psffit_debug", build_c4_cells),
    "C5": (HALOSUB_META["C5"]["slug"], lambda mb, tg, run: build_halosub_cells(mb, tg, run, "C5")),
    "C6": (HALOSUB_META["C6"]["slug"], lambda mb, tg, run: build_halosub_cells(mb, tg, run, "C6")),
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
