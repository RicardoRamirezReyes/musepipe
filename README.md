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
| 07, lineas de acrecion | `musepipe/stages/stage07_accretion_lines.py` | `Far_07_accretion_line_spectra.ipynb` |
| 07b, robustez de Halpha | `musepipe/stages/stage07b_halpha_robustness.py` | `Far_07b_halpha_robustness_checks.ipynb` |
| 08, espectro completo | `stage08_full_spectrum_for_modeling.py` | `Far_08_full_spectrum_for_modeling.ipynb` |

`Far_` se conserva por compatibilidad y contexto historico. No identifica una
segunda implementacion: los notebooks migrados son interfaces sobre el codigo
compartido.

`ROXs12b_short` y `ROXs12b` fueron validados sobre copias aisladas. Para
`ROXs12b`, Stage 04b coincide exactamente con el baseline en todos los datos
FITS y 06 local, 07, 07b y 08 pasan sus contratos y revision visual. El run
original no fue modificado.

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

Las versiones están fijadas al entorno usado para la validación de 45 pruebas.
`maoppy` se instala desde PyPI y es necesario para el perfil
`stage01_profile=maoppy_refined`; sin él, Stage 01 usa su fallback de centrado
por pico.

## Verificacion

Desde la raiz del proyecto:

```bash
python -m unittest discover -s tests
python -m compileall musepipe tests stage08_full_spectrum_for_modeling.py
```

La suite actual contiene 45 pruebas. El entorno Python aun no esta fijado en
un `requirements.txt` o `environment.yml`; esa tarea permanece pendiente.

## Documentacion

- `docs/roxs12b_clean_spectrum_pipeline.md`: uso operativo y productos de la
  cadena local para ROXs12b.
- `docs/refactor_plan_far_objects.md`: fases, decisiones y criterios del
  refactor.
- `docs/plan_mejora_y_sugerencias.md`: diagnostico cientifico y prioridades
  posteriores.
- `docs/00_config_parameters.md`: referencia de configuracion existente.
