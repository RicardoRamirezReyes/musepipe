# MUSE accretion pipeline

Pipeline por etapas para reduccion, diagnostico y extraccion espectral de
cubos MUSE. Los productos de cada ejecucion viven en
`runs/<RUN_ID>/{config,stages,tables,plots,logs}` y no se versionan en Git.

## Estado del refactor

La logica reutilizable se esta extrayendo desde notebooks hacia `musepipe/`.
Las etapas para objetos lejanos ya migradas son:

| Etapa | Implementacion canonica | Interfaz historica |
|---|---|---|
| 04b, sustraccion local | `musepipe/stages/stage04b_local_surface.py` | `Far_04b_local_surface_subtraction.ipynb` |
| 06, inyeccion local | `musepipe/stages/stage06_local_surface_injection.py` | `Far_06_inject_halpha_signal.ipynb` |
| 06 C2, resumen PCA | `musepipe/stages/stage06_pca_c2_summary.py` | `06c_pca_c2_multiline_40.ipynb` |
| 07, lineas de acrecion | `musepipe/stages/stage07_accretion_lines.py` | `Far_07_accretion_line_spectra.ipynb` |
| 07b, robustez de Halpha | `musepipe/stages/stage07b_halpha_robustness.py` | `Far_07b_halpha_robustness_checks.ipynb` |
| 08, espectro completo | `stage08_full_spectrum_for_modeling.py` | `Far_08_full_spectrum_for_modeling.ipynb` |
| 08c, falsos positivos | `musepipe/stages/stage08c_look_elsewhere.py` | API `run_stage08c()` |

`Far_` se conserva por compatibilidad y contexto historico. No identifica una
segunda implementacion: los notebooks migrados son interfaces sobre el codigo
compartido.

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
| B2 | Xcorr / franjas | `musepipe/stages/stage02_xcorr.py` | `scripts/stage02_xcorr.sh` | `stage02_qc.json` |
| B3 | Localización del compañero | `musepipe/stages/stage01c_localize.py` | `scripts/stage01c_localize.sh` | `stage01c_qc.json` |
| C1 | PSF cromática (Moffat/Psfao) | `musepipe/stages/stage_e01_psf.py` (+`stage_e01_psfao.py`) | `scripts/stage_e01_psf.sh` | `stage_e01_qc.json`, `psf_model.json` |
| 04b | Fondo local (superficie) | `musepipe/stages/stage04b_local_surface.py` | — | `stage04b_qc.json` |
| C2 | Extracción por apertura | `musepipe/stages/stage_x01_aperture.py` | `scripts/stage_x01_aperture.sh` | `spec_aperture_qc.json` |
| C3 | Extracción óptima | `musepipe/stages/stage_x02_optimal.py` | `scripts/stage_x02_optimal.sh` | `spec_optimal_qc.json` |
| C4 | Ajuste de PSF (psffit) | `musepipe/stages/stage_x03_psffit.py` | `scripts/stage_x03_psffit.sh` | `spec_psffit_qc.json` |
| D1 | Comparación inter-método | `musepipe/stages/stage_x10_compare.py` | `scripts/stage_x10_compare.sh` | `stage_x10_qc.json` |
| D2 | Calibración espectral | `musepipe/stages/stage_x11_calibrate.py` | `scripts/stage_x11_calibrate.sh` | `stage_x11_qc.json` |
| E1 | Detección Hα | `musepipe/stages/stage_h01_detect.py` | `scripts/stage_h01_detect.sh` | `stage_h01_qc.json` |
| E2 | Batería de artefactos | `musepipe/stages/stage_h02_artifacts.py` | `scripts/stage_h02_artifacts.sh` | `stage_h02_qc.json` |
| E3 | Límites superiores (Ṁ) | `musepipe/stages/stage_h03_limits.py` | `scripts/stage_h03_limits.sh` | `stage_h03_qc.json` |
| E4 | Inyección-recuperación | `musepipe/stages/stage_h04_injection.py` | `scripts/stage_h04_injection.sh` | `stage_h04_qc.json` |
| F1 | Paquete final + gate | `musepipe/report.py` | `scripts/build_report.py` | `report/run_summary.json` |
| G0 | Ejecución cubo real | `musepipe/g0.py` | — | `stage_g0_qc.json` |
| G1 | Validación de extracción | `musepipe/covariance.py` | — | `stage_g1_qc.json` |
| G2 | Medición de líneas | `musepipe/lines.py` + `stages/stage_g2_measure_lines.py` | — | `stage_g2_qc.json` |
| G3 | Inferencia física | `musepipe/models/` + `stages/stage_g3_accretion.py` | — | `stage_g3_qc.json` |
| G4 | Clasificación de fuente | `musepipe/classify.py` + `stages/stage_g4_classify.py` | — | `stage_g4_classification.json` |
| G5 | Síntesis final | `musepipe/characterization.py` | `scripts/build_characterization.py` | `report/characterization/` |

La cadena histórica de objetos lejanos (04b→06→07→07b→08→08c) y la cadena cercana/PCA
(notebooks 00–06, `LkCa_15`) se documentan más arriba en este README.

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

Los nueve casos PCA de C2 se ejecutan desde
`06c_pca_c2_multiline_40.ipynb`. Una vez completos, su consolidación no repite
PCA ni abre el cubo grande:

```bash
MPLBACKEND=Agg python -c "from musepipe.stages import run_stage06_pca_c2_summary; run_stage06_pca_c2_summary('LkCa_15')"
```

Antes de ejecutarlas sobre un run cientifico, confirma el run y revisa los
productos existentes. Los notebooks `Far_04b`, `Far_07`, `Far_07b` y `Far_08` ofrecen la
misma ruta con figuras de inspeccion.

## Estructura

```text
musepipe/
  config.py       seleccion y validacion de runs
  paths.py        rutas estandar por run
  io.py           lectura y escritura de FITS, CSV y JSON
  stats.py        estadistica robusta
  spectral.py     mascaras y operaciones espectrales
  apertures.py    aperturas y controles al mismo radio
  localfit.py     ajuste y sustraccion de superficies locales
  stages/         implementaciones de etapas migradas
tests/            pruebas unitarias y sinteticas
docs/             guias cientificas y planes de trabajo
runs/             datos y productos locales, ignorados por Git
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

Las versiones están fijadas al entorno usado para la validación de 52 pruebas.
`maoppy` se instala desde PyPI y es necesario para el perfil
`stage01_profile=maoppy_refined`; sin él, Stage 01 usa su fallback de centrado
por pico.

## Verificacion

Desde la raiz del proyecto:

```bash
python -m unittest discover -s tests
python -m compileall musepipe tests stage08_full_spectrum_for_modeling.py
```

La suite actual contiene 52 pruebas y el entorno reproducible esta fijado en
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
