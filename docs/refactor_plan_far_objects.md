# Refactor plan - objetos lejanos y libreria comun

Fecha: 2026-06-17

Este documento fija el contexto antes de empezar el refactor. La meta es
extraer logica reutilizable desde notebooks hacia un paquete comun, sin cambiar
resultados cientificos en la misma pasada. Para objetos lejanos, el objetivo no
es crear un proyecto separado, sino convertir `Far_*` en un modo de analisis
configurable basado en sustraccion local.

## Objetivo del refactor

1. Mantener la arquitectura por etapas y los productos actuales en `runs/`.
2. Evitar ramas paralelas de notebooks que dupliquen logica.
3. Centralizar configuracion, paths, IO, estadistica, aperturas, espectro y
   sustraccion local en un paquete `musepipe/`.
4. Hacer que los notebooks queden como capa fina: cargar config, ejecutar una
   funcion de etapa y revisar figuras.
5. Reducir el riesgo de mezclar runs por `RUN_ID` hardcodeado.

## Fase 0 - estado y seguridad

### Resultado de verificacion

- Raiz usada: `/Users/ricardoramirez/Documents/GitHub/Muse/MUSE-accretion-pipeline`.
- Estado Git: esta carpeta no es un repositorio Git activo. `git rev-parse
  --show-toplevel` y `git status --short` fallan con `fatal: not a git
  repository`.
- No se encontro `.git/` dentro de los dos primeros niveles de esta carpeta.
- Se creo `.gitignore` conservador para evitar versionar `runs/`, caches de
  notebooks/Python, ruido local y auxiliares LaTeX.
- No se movieron, borraron ni regeneraron productos cientificos.

### Inventario actual

| Categoria | Estado |
|---|---:|
| Notebooks raiz | 29 |
| Modulos Python raiz | 2 |
| Archivos docs | 4 |
| Lineas de codigo en notebooks | 32734 |
| Definiciones `def` en notebooks | 671 |
| Notebooks con `RUN_ID` activo hardcodeado | 23 |
| Lineas de codigo en notebooks `Far_*` | 4591 |
| Definiciones `def` en notebooks `Far_*` | 77 |
| Tamano total aproximado | 92G |
| Tamano aproximado de `runs/` | 80G |

Modulos Python actuales:

- `muse_crop_checks.py`
- `stage08_full_spectrum_for_modeling.py`

Docs existentes:

- `docs/00_config_parameters.md`
- `docs/00_config_parameters.tex`
- `docs/plan_mejora_y_sugerencias.md`
- `docs/roxs12b_clean_spectrum_pipeline.md`

Runs detectados:

| Carpeta run | `cfg.run_id` | Target | `crop_npix` | Cubos | Comentario |
|---|---|---|---:|---:|---|
| `LkCa_15` | `LkCa_15` | `LkCa_15` | 160 | 32 | Run completo y pesado. |
| `ROXs12b` | `ROXs12b` | `ROX12b` | 170 | 1 | Run cientifico principal para objeto lejano. |
| `ROXs12b_HaInject_SNR5` | `ROXs12b` | `ROX12b` | 170 | 1 | Carpeta y config no coinciden en `run_id`; requiere cuidado. |
| `ROXs12b_short` | `ROXs12b_short` | `ROX12b` | 170 | 1 | Buen candidato para smoke tests. |
| `ROXs42Bb` | `ROXs42Bb` | `ROX42` | 120 | 1 | Objeto lejano adicional. |
| `YSES 2b` | `YSES 2b` | `YSES 2b` | 100 | 1 | Tiene espacio en el nombre; conviene corregir en fase posterior. |

Estado de notebooks `Far_*`:

| Notebook | `RUN_ID` activo | Lineas codigo | `def` | Riesgo |
|---|---|---:|---:|---|
| `Far_04b_local_surface_subtraction.ipynb` | `LkCa_15` | 1952 | 21 | Alto: notebook Far activo en run cercano. |
| `Far_06_inject_halpha_signal.ipynb` | `LkCa_15` | 775 | 17 | Alto: version vieja y run activo no Far. |
| `Far_07_accretion_line_spectra.ipynb` | `ROXs12b` | 1020 | 22 | Medio: run correcto para ROXs12b, pero hardcodeado. |
| `Far_07b_halpha_robustness_checks.ipynb` | `ROXs12b` | 716 | 15 | Medio: coordenadas/run hardcodeados. |
| `Far_08_full_spectrum_for_modeling.ipynb` | ninguno | 128 | 2 | Bajo: ya usa modulo Stage 8. |

## Reglas de seguridad

1. No borrar ni mover `runs/` durante el refactor.
2. No cambiar resultados cientificos junto con cambios estructurales, salvo bug
   confirmado.
3. Mantener nombres de productos existentes para no romper etapas posteriores.
4. Migrar por equivalencia: primero reproducir comportamiento previo, despues
   optimizar.
5. Preferir `ROXs12b_short` para pruebas rapidas y `ROXs12b` para validacion
   cientifica.
6. Registrar en QC cualquier validacion nueva que compare config, run y forma
   espacial.

## Runs de validacion

### Smoke test

Run recomendado: `ROXs12b_short`

Uso:

- Validar imports del paquete `musepipe`.
- Probar utilidades puras y rutas de lectura livianas.
- Probar Stage 8 o extracciones locales sin costo grande.

### Validacion principal de objetos lejanos

Run recomendado: `ROXs12b`

Uso:

- Comparar productos antes/despues para `stage04b`, `stage07b` y `stage08`.
- Mantener outputs esperados:
  - `cube_input_local_object.fits`
  - `cube_residual_local_object.fits`
  - `cube_localmodel_object.fits`
  - `stage04b_local_surface_cube_stack.fits`
  - `stage04b_qc.json`
  - `stage07b_halpha_robustness_summary.csv`
  - `stage08_full_spectrum_model_input.csv`

### Validacion secundaria

Runs candidatos: `ROXs42Bb`, `YSES 2b`

Uso:

- Confirmar que el refactor no queda amarrado a coordenadas especificas de
  ROXs12b.
- `YSES 2b` debe tratarse con cuidado por el espacio en el nombre de carpeta.

### Run de inyeccion

Run candidato: `ROXs12b_HaInject_SNR5`

Uso:

- No usar como primera validacion de RUN_ID, porque la carpeta declara
  `cfg.run_id = ROXs12b`.
- Resolver el desajuste antes de usarlo como prueba automatica estricta.

## Plan de fases

### Fase 1 - paquete base

Crear:

```text
musepipe/
  __init__.py
  config.py
  paths.py
  io.py
  stats.py
  spectral.py
  apertures.py
  localfit.py
  plotting.py
  stages/
    __init__.py
```

Criterio de salida:

- `python -c "import musepipe"` funciona desde la raiz.
- No cambia ningun notebook todavia, salvo imports de prueba si hace falta.

### Fase 2 - config y RUN_ID unico

Implementar:

- `get_run_id()`: `MUSE_RUN_ID` -> `active_run.txt` -> error claro.
- `load_run_config(run_id)`.
- Validacion `CFG["run_id"] == RUN_ID`, con excepciones documentadas si hay
  runs historicos inconsistentes.
- `RunPaths` para `config`, `stages`, `tables`, `plots`, `logs`.

Criterio de salida:

- Los notebooks criticos dejan de necesitar bloques de `RUN_ID` comentados.

### Fase 3 - utilidades comunes

Mover funciones duplicadas:

- `stats.py`: `robust_sigma`, `robust_sigma_axis0`, `finite_percentile`.
- `io.py`: `read_json`, `write_json`, `write_csv`, `get_cube_data`,
  `read_wavelengths_and_masks`.
- `spectral.py`: `nearest_channel_indices`, `nearest_channel_index`,
  `continuum_running_median`, mascaras de longitudes de onda.
- `apertures.py`: `aperture_weights`, `box_spectrum_sum`,
  `box_spectrum_mean`, `same_radius_control_positions`.

Criterio de salida:

- Stage 8 puede usar estas funciones sin cambiar productos.

### Fase 4 - tests minimos

Crear tests puros, sin FITS pesados:

```text
tests/test_stats.py
tests/test_spectral.py
tests/test_apertures.py
tests/test_localfit.py
```

Criterio de salida:

- Tests pasan localmente.
- Cubren outliers, aperturas, canales cercanos y plano sintetico.

### Fase 5 - sustraccion local comun

Crear `musepipe/localfit.py` con:

- `fit_local_surface_2d` robusto.
- `fit_fast_surface_coefficients`.
- `local_surface_spectra_fast`.
- Helpers de mascara, pesos y metadatos.

Criterio de salida:

- `stage08_full_spectrum_for_modeling.py` puede importar localfit.
- `Far_07b` puede reutilizar la misma extraccion local.

### Fase 6 - migrar Stage 8

Transformar `stage08_full_spectrum_for_modeling.py` en consumidor de
`musepipe`.

Criterio de salida:

- `Far_08_full_spectrum_for_modeling.ipynb` sigue funcionando.
- `stage08_full_spectrum_model_input.csv` mantiene columnas y valores
  esperados.

### Fase 7 - migrar Far_07b

Crear:

```text
musepipe/stages/stage07b_halpha_robustness.py
```

Responsabilidades:

- Leer Stage04b.
- Resolver objeto y estrella desde config/QC.
- Extraer objeto y controles.
- Calcular metricas Halpha.
- Guardar CSV, QC y plots con nombres actuales.

Criterio de salida:

- `Far_07b_halpha_robustness_checks.ipynb` queda como wrapper fino.

### Fase 8 - migrar Far_04b

Crear:

```text
musepipe/stages/stage04b_local_surface.py
```

Responsabilidades:

- Seleccionar entrada `stage02`, `stage03` o producto configurado.
- Resolver coordenadas de objeto lejano.
- Construir mascaras espectrales.
- Sustraer superficie local por cubo.
- Rankear cubos.
- Escribir productos compatibles actuales.

Criterio de salida:

- `Far_04b_local_surface_subtraction.ipynb` queda como wrapper fino.
- Productos principales de `ROXs12b_short` y `ROXs12b` se reproducen.

### Fase 9 - documentacion y nombres

Actualizar:

- `docs/roxs12b_clean_spectrum_pipeline.md`
- `docs/plan_mejora_y_sugerencias.md`
- `README.md` raiz

Decidir despues si `Far_*` se mantiene como alias historico o se reemplaza por
notebooks unicos parametrizados.

### Fase 10 - inyeccion local-surface

Despues de estabilizar `stage04b` y `stage07b`:

- Portar grilla moderna de `06_low_memory_inject_halpha_signal`.
- Medir recuperacion a traves de sustraccion local.
- No iniciar esta fase hasta que el refactor estructural este validado.

## Checklist Fase 0

- [x] Confirmar raiz de trabajo.
- [x] Confirmar ausencia de repo Git activo.
- [x] Crear `.gitignore` conservador.
- [x] Inventariar notebooks, modulos, docs y runs.
- [x] Identificar riesgos inmediatos en `Far_*`.
- [x] Definir runs de validacion.
- [ ] Inicializar o conectar Git antes de cambios grandes.
- [ ] Pasar a Fase 1: crear paquete `musepipe/`.

