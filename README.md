# MUSE accretion pipeline

Pipeline por etapas para reduccion, diagnostico y extraccion espectral de
cubos MUSE. Los productos de cada ejecucion viven en
`runs/<RUN_ID>/{config,stages,tables,plots,logs}` y no se versionan en Git.

## Estado del refactor

La logica reutilizable se esta extrayendo desde notebooks hacia `musepipe/`.
Las etapas para objetos lejanos ya migradas son:

| Etapa | Implementacion canonica | Interfaz historica |
|---|---|---|
| 04b, sustraccion local | `musepipe/stages/stage04b_local_surface.py` | `legacy/Far_04b_local_surface_subtraction.ipynb` |
| 06, inyeccion local | `musepipe/stages/stage06_local_surface_injection.py` | `legacy/Far_06_inject_halpha_signal.ipynb` |
| 06 C2, resumen PCA | `musepipe/stages/stage06_pca_c2_summary.py` | `legacy/06c_pca_c2_multiline_40.ipynb` |
| 07, lineas de acrecion | `musepipe/stages/stage07_accretion_lines.py` | `legacy/Far_07_accretion_line_spectra.ipynb` |
| 07b, robustez de Halpha | `musepipe/stages/stage07b_halpha_robustness.py` | `legacy/Far_07b_halpha_robustness_checks.ipynb` |
| 08, espectro completo | `stage08_full_spectrum_for_modeling.py` | `legacy/Far_08_full_spectrum_for_modeling.ipynb` |
| 08c, falsos positivos | `musepipe/stages/stage08c_look_elsewhere.py` | API `run_stage08c()` |

`Far_` se conserva por compatibilidad y contexto historico. No identifica una
segunda implementacion: los notebooks migrados son interfaces sobre el codigo
compartido.

### `legacy/` — notebooks de la version antigua

Los 31 notebooks que vivian en la raiz (`00_*`–`11_*`, `01b_*`, `02b/02c_*`,
`03b_*`, `04c/04d_*`, `06c_*`, `A1_F1_run_inspection`, `Far_*`) y su arnes de
validacion `validate_o2b.sh` se movieron a `legacy/`, que **no se versiona**
(`.gitignore`). Son instantaneas historicas: la interfaz actual es
`notebooks/<objeto>/` sobre `musepipe/` + `scripts/`.

Si `legacy/` no esta en tu copia, recuperalo del historial:

```bash
git checkout 225a8fc -- '*.ipynb' validate_o2b.sh   # ultimo commit con ellos en la raiz
```

Los planes y reports fechados de `docs/` los siguen citando por su ruta
antigua en la raiz: son registros de su momento y no se reescriben.

`ROXs12b_short` y `ROXs12b` fueron validados sobre copias aisladas. Para
`ROXs12b`, Stage 04b coincide exactamente con el baseline en todos los datos
FITS y 06 local, 07, 07b y 08 pasan sus contratos y revision visual. El run
original no fue modificado.

## Mapa de etapas (spec ↔ código)

Cadena ROXs 12 (A→G). Los QC viven en `runs/<RUN>/stages/`.

| Spec | Qué hace | Módulo canónico | Script | QC |
|---|---|---|---|---|
| A1 | Reducción raw (esorex) | `musepipe/reduction/esorex_driver.py` | `scripts/reduce_raw.sh` | `stage00r_qc.json` |
| A2 | Decisión ZAP (cielo) | `musepipe/reduction/sky_zap.py` | `scripts/sky_zap.sh` | `stage00s_qc.json` |
| A3 | Corrección telúrica | `musepipe/reduction/` (telluric) | `scripts/telluric.sh` | `stage00t_qc.json` |
| A4 | QC del cubo (M1–M5) | `musepipe/qc/cube_qc.py` | `scripts/cube_qc.sh` | `stage00q_qc.json` |
| B1 | Carga/alineación/crop | `musepipe/stages/stage01_align.py` | — | `stage01_qc.json` |
| B2 | Xcorr / franjas | `musepipe/stages/stage02_xcorr.py` | `scripts/stage02_xcorr.sh` | `stage02_xcorr_qc.json` |
| B3 | Localización del compañero | `musepipe/stages/stage01c_localize.py` | `scripts/stage01c_localize.sh` | `stage01c_qc.json` |
| C1 | PSF cromática (Moffat/Psfao) | `musepipe/stages/stage_e01_psf.py` (+`stage_e01_psfao.py`) | `scripts/stage_e01_psf.sh` | `stage_e01_qc.json`, `psf_model.json` |
| 04b | Fondo local (superficie) | `musepipe/stages/stage04b_local_surface.py` | — | `stage04b_qc.json` |
| C2 | Extracción por apertura | `musepipe/stages/stage_x01_aperture.py` | `scripts/stage_x01_aperture.sh` | `spec_aperture_qc.json` |
| C3 | Extracción óptima — **2 variantes**: `optimal_ls` y `optimal_psfsub` | `musepipe/stages/stage_x02_optimal.py` | `scripts/stage_x02_optimal.sh` | `spec_optimal_qc.json` |
| C4 | Ajuste de PSF (psffit) | `musepipe/stages/stage_x03_psffit.py` | `scripts/stage_x03_psffit.sh` | `spec_psffit_qc.json` |
| C5 | Sustracción de halo SGF | `musepipe/stages/stage_x04_sgf.py` (+`halosub_stage.py`) | — | `spec_sgf_qc.json` |
| C6 | Sustracción de halo LPM | `musepipe/stages/stage_x05_lpm.py` (+`halosub_stage.py`) | — | `spec_lpm_qc.json` |
| D1 | Comparación inter-método | `musepipe/stages/stage_x10_compare.py` | `scripts/stage_x10_compare.sh` | `stage_x10_qc.json` |
| D2 | Calibración espectral | `musepipe/stages/stage_x11_calibrate.py` | `scripts/stage_x11_calibrate.sh` | `stage_x11_qc.json` |
| E1 | Detección Hα | `musepipe/stages/stage_h01_detect.py` | `scripts/stage_h01_detect.sh` | `stage_h01_qc.json` |
| E1b | Detección ciega en el FoV | `musepipe/stages/stage_h01b_fovmap.py` | — | `stage_h01b_qc.json` |
| E2 | Batería de artefactos | `musepipe/stages/stage_h02_artifacts.py` | `scripts/stage_h02_artifacts.sh` | `stage_h02_qc.json` |
| E3 | Límites superiores (Ṁ) | `musepipe/stages/stage_h03_limits.py` | `scripts/stage_h03_limits.sh` | `stage_h03_qc.json` |
| E4 | Inyección-recuperación | `musepipe/stages/stage_h04_injection.py` | `scripts/stage_h04_injection.sh` | `stage_h04_qc.json` |
| E5 | Curvas de contraste | `musepipe/stages/stage_h05_contrast.py` | — | `stage_h05_qc.json` |
| E6 | Curvas ROC | `musepipe/stages/stage_h06_roc.py` | — | `stage_h06_qc.json` |
| F1 | Paquete final + gate | `musepipe/report.py` | `scripts/build_report.py` | `report/run_summary.json` |
| G0 | Ejecución cubo real | `musepipe/g0.py` | — | `stage_g0_qc.json` |
| G1 | Validación de extracción | `musepipe/covariance.py` | — | `stage_g1_qc.json` |
| G2 | Medición de líneas | `musepipe/lines.py` + `stages/stage_g2_measure_lines.py` | — | `stage_g2_qc.json` |
| G3 | Inferencia física | `musepipe/models/` + `stages/stage_g3_accretion.py` | — | `stage_g3_qc.json` |
| G4 | Clasificación de fuente | `musepipe/classify.py` + `stages/stage_g4_classify.py` | — | `stage_g4_classification.json` |
| G5 | Síntesis final | `musepipe/characterization.py` | `scripts/build_characterization.py` | `report/characterization/` |
| S0 | Mapa de solución de onda | `musepipe/qc/wavesol_map.py` | — | `stageS0_qc.json` |
| S1 | Mapas de Hα (LSF, línea/continuo) | `musepipe/qc/halpha_map.py` | — | `stageS1_qc.json` |

`musepipe/stage_registry.py` es la fuente de verdad legible por máquina de esta
tabla (QC de cada etapa, `qc_aliases` por perfil de reducción, `exec_kind` y
comando de lanzamiento); los notebooks y `scripts/build_review_notebooks.py`
validan contra él. B2 escribe `stage02_xcorr_qc.json` (el `stage02_qc.json` de
los runs antiguos es el de la etapa previa).

**Cinco etapas de extracción, seis métodos.** C2–C6 son cinco etapas, pero C3
emite **dos** variantes como productos separados —`optimal_ls` (fondo = la
superficie local de 04b, comparable 1:1 con C2) y `optimal_psfsub` (fondo = el
modelo de PSF de la primaria, de C1)— así que la cadena compara seis métodos:
`aperture`, `optimal_ls`, `optimal_psfsub`, `psffit`, `sgf` y `lpm`. Ese es el
`METHOD_ORDER` que usan D1, D2, E4 y G1, y la comparación `ls` vs `psfsub` es un
diagnóstico del modelo de halo, no una redundancia.

Además del canónico `spec_final_object.fits`, **D2 entrega los espectros
definitivos**: los seis métodos calibrados (`spec_calibrated_<método>_object.fits`)
y la primaria (`spec_calibrated_psffit_star.fits`), todos con `BUNIT` y
presupuesto de error, con su tabla en `stage_x11_qc.json` → `spectra` y la figura
`plots/stage_x11_spectra.png`.

La cadena histórica de objetos lejanos (04b→06→07→07b→08→08c) y la cadena cercana/PCA
(notebooks 00–06, `LkCa_15`) se documentan más arriba en este README.

## Notebooks de revisión (A1 → G5)

`notebooks/` contiene **un set por objeto** (`notebooks/ROXs12b/`,
`notebooks/ROXs42Bb/`), con un notebook por spec de la cadena canónica
(`A1_raw_reduction.ipynb` … `G5_final_synthesis.ipynb`), interfaces delgadas
sobre `musepipe/` + `scripts/`. Cada set audita la cadena de su objeto (declarada
en la clave `chain` de `runs/<run>/config/config.json`); las etapas de reducción
se pueden lanzar desde el propio notebook. Índice y uso en
[`notebooks/README.md`](notebooks/README.md). Se regeneran con
`python scripts/build_review_notebooks.py --target <objeto>` (`--check` valida que
cada QC resuelve). Añadir un objeto = crear su run+config y su
`targets/<slug>.json`, sin tocar código. Ver
[`docs/plan_multiobjeto_notebooks_2026-07-24.md`](docs/plan_multiobjeto_notebooks_2026-07-24.md).

### Notebooks de análisis (`debug/`)

`scripts/build_debug_notebooks.py` es un constructor **opcional y aparte** que
escribe en `notebooks/<Objeto>/debug/`. Los de revisión auditan la cadena
(llaman a `musepipe`); estos hacen **el proceso dentro del notebook**, con las
funciones numéricas **copiadas literalmente** del código, para poder probar,
cambiar y ajustar **sin tocar la cadena general**.

```bash
python scripts/build_debug_notebooks.py --target ROXs12b        # todas las etapas cubiertas
python scripts/build_debug_notebooks.py --target ROXs42Bb C2    # solo una
```

Copiar código es normalmente mala idea, así que cada notebook lleva dos
defensas: una **celda de deriva**, que compara el fuente copiado con el que hoy
tiene `musepipe` y avisa nombrando la función, y una **celda de comparación**
contra el producto real de la etapa — con las perillas por defecto debe salir
idéntico (lo verifica `tests/test_debug_notebooks.py` ejecutando el notebook
entero; son los tests marcados `slow`), y en cuanto se cambia una perilla dice
qué se movió y cuánto.

Las perillas se leen del **config resuelto de la etapa**, no del `config.json`
crudo: la etapa rellena defaults que el run no escribe (C3 hereda el anillo de
fondo de C2), y copiarlos a mano fue justo lo que hizo que el primer C3 no
reprodujera la cadena.

| notebook | qué rehace |
|---|---|
| `A3_telluric_debug` | la decisión telúrica entera —continuo local, profundidad por banda, umbral— sobre las tres reducciones del objeto, con el espectro **antes y después** de corregir y las tres hipótesis del salto O₂ B ~7 % → 0.59 % |
| `C2_aperture_debug` | apertura box3, fondo de anillo, controles, error empírico y por STAT, apcorr. Además, dos estudios: el **anillo** (8 aperturas repartidas sobre él, con el mismo proceso que el compañero, y el sesgo de su mediana medido contra los sectores a ±90°) y el **tamaño óptimo** por S/N empírica |
| `C3_optimal_debug` | el estimador de Horne y **las dos variantes**: `optimal_ls` y `optimal_psfsub`, incluyendo el ajuste de la PSF de la primaria que las separa. Y el estudio del anillo **sobre los datos con el halo restado** (modelo de PSF frente a perfil radial medido): si restarlo aplana el gradiente azimutal y su sesgo |
| `C4_psffit_debug` | el ajuste simultáneo de **dos PSF** por canal (el método canónico): región de ajuste, matriz de diseño, χ²ᵣ y ρ(a,b), controles y los dos espectros. Trae perilla de submuestreo de canales |
| `C5_sgf_debug` | selección de spaxels de referencia, espectro estelar de referencia, filtrado Savitzky-Golay y apertura sobre el residual |
| `C6_lpm_debug` | lo mismo, pero modelando con Legendre y **con las líneas enmascaradas del ajuste**: la base del diseño y la energía por grado |

## Seleccionar un run

Si se pasa `run_id` a la API, ese valor es explicito. En caso contrario, el run
se resuelve en este orden:

1. Variable de entorno `MUSE_RUN_ID`.
2. Primera linea no comentada de `active_run.txt`.

`active_run.txt` es local y esta ignorado por Git. Para una ejecucion puntual,
la variable de entorno evita cambiar el archivo:

```bash
MUSE_RUN_ID=ROXs12b python -c "from musepipe.config import load_run_config; print(load_run_config().run_id)"
```

Inspeccionar configuracion no escribe productos:

```bash
MUSE_RUN_ID=ROXs12b_short python -c "from musepipe.stages import stage04b_config_from_run; print(stage04b_config_from_run())"
```

Por defecto, una discrepancia entre la carpeta seleccionada y
`config/config.json` produce un error. No se recomienda usar
`allow_run_id_mismatch=True` salvo para diagnosticar un run historico conocido.

## Flujo para objetos lejanos

Con `stage04b_input_mode = native_stage02`, la ruta principal es:

```text
01_load_align_crop
  -> 02_xcorr_stripes
  -> 04b local surface
  -> 06 local-surface injection (validacion opcional)
  -> 07 accretion-line spectra (inspeccion opcional)
  -> 07b Halpha robustness
  -> 08 full spectrum
  -> 08c look-elsewhere / false-alarm calibration
```

Stage 03 es opcional si 04b usa `fakecont_stage03`. Stage 04/PCA no es parte de
la sustraccion local, aunque un `stage04_qc.json` existente puede aportar las
coordenadas historicas de los objetos.

Las llamadas siguientes escriben o reemplazan productos dentro del run activo:

```bash
MUSE_RUN_ID=ROXs12b python -c "from musepipe.stages import run_stage04b; run_stage04b()"
MUSE_RUN_ID=ROXs12b python -c "from musepipe.stages import run_stage06_local; run_stage06_local()"
MUSE_RUN_ID=ROXs12b python -c "from musepipe.stages import run_stage07; run_stage07()"
MUSE_RUN_ID=ROXs12b python -c "from musepipe.stages import run_stage07b; run_stage07b()"
MUSE_RUN_ID=ROXs12b python stage08_full_spectrum_for_modeling.py
MUSE_RUN_ID=ROXs12b python -c "from musepipe.stages import run_stage08c; run_stage08c()"
```

Antes de ejecutarlas sobre un run cientifico, confirma el run y revisa los
productos existentes. Los notebooks `legacy/Far_04b`, `legacy/Far_07`,
`legacy/Far_07b` y `legacy/Far_08` ofrecen la misma ruta con figuras de
inspeccion.

### PCA C2 de LkCa 15: producto historico, no un flujo vigente

El barrido PCA C2 de nueve casos (`stage06_pca_c2_40`, LkCa 15, 2026-06-21) es
un **resultado ya calculado**, no un paso que se re-ejecute hoy:

- Sus nueve `*_recovery.csv` de entrada los escribia **solo**
  `legacy/06c_pca_c2_multiline_40.ipynb`; ningun modulo ni script versionado los
  produce, y `legacy/` no se versiona. Desde un clon limpio ese paso no es
  reproducible.
- La etapa no esta en la cadena canonica: no aparece en
  `musepipe/stage_registry.py`, y `LkCa_15` no tiene ficha en `targets/` (no
  tiene set en `notebooks/<objeto>/`).

Lo que si sigue vigente es la **consolidacion** de esos productos, que no repite
PCA ni abre el cubo grande:

```bash
MPLBACKEND=Agg python -c "from musepipe.stages import run_stage06_pca_c2_summary; run_stage06_pca_c2_summary('LkCa_15')"
```

**Para inyeccion-recuperacion nueva, el camino es la cadena canonica**: E4
(`stage_h04_injection`, throughput y curva de recuperacion) y, si se quieren
curvas, E5 (contraste) y E6 (ROC). LkCa 15 ya se corrio por la cadena moderna el
2026-07-15 (B1→B2→B3→C1→C5→C6→E1b, ver
[`docs/smoke_lkca15_2026-07-15.md`](docs/smoke_lkca15_2026-07-15.md)).

## Estructura

```text
musepipe/
  config.py       seleccion y validacion de runs
  paths.py        rutas estandar por run
  io.py           FITS/CSV/JSON + resolucion de la unidad de flujo (BUNIT -> cgs)
  stage_registry.py  fuente de verdad de la cadena (QC, exec_kind, lanzamiento)
  stats.py        estadistica robusta
  spectral.py     mascaras y operaciones espectrales
  apertures.py    aperturas y controles al mismo radio
  localfit.py     ajuste y sustraccion de superficies locales
  psf.py, stripes.py, covariance.py, injection.py, halosub.py, parallel.py
  stages/         implementaciones de etapas migradas
  extraction/     SpectrumProduct (contenedor canonico) y los extractores
  reduction/      driver de esorex, cielo/ZAP, telurico, combinacion
  qc/             QC de cubo y frame, censo de ghosts, STAT empirica, mapas S0/S1
  models/         BT-Settl, extincion, relaciones de acrecion, tracks, plantillas
  report.py (F1), characterization.py (G5), classify.py (G4), lines.py (G2), g0.py
  telluric_lines.py  bandas teluricas + curva de transmision medida por A3
  paper_spectrum.py  figura de publicacion (sin binar, con lineas) + export ECSV
notebooks/        un set de revision por objeto (generado por scripts/)
scripts/          lanzadores por etapa y trabajos largos
targets/          ficha por objeto (alias, referencias)
tests/            pruebas unitarias y sinteticas
docs/             guias cientificas, specs congeladas y planes de trabajo
runs/             datos y productos locales, ignorados por Git
legacy/           notebooks + arnes de la version antigua, locales e ignorados por Git
```

## Entorno

El entorno reproducible está declarado en `environment.yml`. Para crearlo:

```bash
conda env create --file environment.yml
conda activate MUSE
```

Para actualizar un entorno `MUSE` existente con la misma especificación:

```bash
conda env update --name MUSE --file environment.yml --prune
```

Las versiones están fijadas al entorno usado para validar la suite.
`maoppy` se instala desde PyPI y es necesario para el perfil
`stage01_profile=maoppy_refined`; sin él, Stage 01 usa su fallback de centrado
por pico.

## Verificacion

Desde la raiz del proyecto:

```bash
python -m pytest tests/ -q                         # suite completa (699 pruebas)
python -m pytest tests/test_h03_chain.py -q        # un solo archivo
python -m pytest tests/ -q -m "not external_data"  # sin las que piden bibliotecas externas
python -m compileall musepipe tests stage08_full_spectrum_for_modeling.py
```

El runner es **pytest** (`pytest.ini` fija `testpaths` y el marcador
`external_data`), aunque casi todas las pruebas son clases `unittest.TestCase`.
No hay linter ni formateador configurado. El entorno reproducible esta fijado en
`environment.yml`.

## Documentacion

- `docs/roxs12b_clean_spectrum_pipeline.md`: uso operativo y productos de la
  cadena local para ROXs12b.
- `docs/refactor_plan_far_objects.md`: fases, decisiones y criterios del
  refactor.
- `docs/plan_mejora_y_sugerencias.md`: diagnostico cientifico y prioridades
  posteriores.
- `docs/00_config_parameters.md`: referencia de configuracion existente.
- `docs/noise_model.md`: modelo de ruido canonico (STAT subestima ~4x, inflacion
  espacial en apertura, correlacion espectral, origen en el shift subpixel de stage01;
  regla control=objeto). Toda etapa debe citarlo en vez de re-derivar sigma.
- `docs/a3_telluric_justification.md`: justificacion de la correccion telurica por
  STD_TELLURIC (molecfit no convergio) para el paper.
