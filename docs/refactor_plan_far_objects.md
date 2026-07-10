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
- Estado Git al iniciar Fase 0: esta carpeta no era un repositorio Git activo.
  Posteriormente se inicializo Git antes de comenzar el refactor.
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

Estado 2026-06-17: completado el esqueleto inicial. Se crearon los modulos
base con docstrings y sin migrar comportamiento todavia:

- `musepipe/__init__.py`
- `musepipe/config.py`
- `musepipe/paths.py`
- `musepipe/io.py`
- `musepipe/stats.py`
- `musepipe/spectral.py`
- `musepipe/apertures.py`
- `musepipe/localfit.py`
- `musepipe/plotting.py`
- `musepipe/stages/__init__.py`

### Fase 2 - config y RUN_ID unico

Implementar:

- `get_run_id()`: `MUSE_RUN_ID` -> `active_run.txt` -> error claro.
- `load_run_config(run_id)`.
- Validacion `CFG["run_id"] == RUN_ID`, con excepciones documentadas si hay
  runs historicos inconsistentes.
- `RunPaths` para `config`, `stages`, `tables`, `plots`, `logs`.

Criterio de salida:

- Los notebooks criticos dejan de necesitar bloques de `RUN_ID` comentados.

Estado 2026-06-17: infraestructura implementada, notebooks aun no migrados.

Nuevo en `musepipe/config.py`:

- `get_run_id()`
- `load_config_payload()`
- `validate_run_config()`
- `load_run_config()`
- `available_run_ids()`
- `ConfigError`
- `RunConfig`

Nuevo en `musepipe/paths.py`:

- `RunPaths.from_project_root()`
- `RunPaths.config_json`
- `RunPaths.plot_stage_dir()`
- `RunPaths.ensure_base_dirs()`

Verificaciones ejecutadas:

- `python -m compileall musepipe`
- `available_run_ids()` devuelve los seis runs con `config/config.json`.
- `load_run_config("ROXs12b_short")` carga y valida `cfg.run_id`.
- `MUSE_RUN_ID=ROXs12b load_run_config()` resuelve el run desde entorno.
- `load_run_config("ROXs12b_HaInject_SNR5")` falla por mismatch
  `cfg.run_id="ROXs12b"`, como se esperaba.
- `load_run_config("ROXs12b_HaInject_SNR5", allow_run_id_mismatch=True)`
  permite cargar el run historico de inyeccion de forma explicita.

Pendiente para cerrar Fase 2 completa:

- Migrar notebooks criticos para usar `load_run_config()` en vez de `RUN_ID`
  hardcodeado.

Decision 2026-06-17:

- Se creo `active_run.txt` local con `ROXs12b_short` como run por defecto para
  smoke tests.
- `active_run.txt` queda ignorado por Git. Para validacion cientifica se debe
  usar `MUSE_RUN_ID=ROXs12b` o editar temporalmente el archivo local.

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

Estado 2026-06-17: primera extraccion completada y Stage 8 migrado a las
utilidades comunes.

Implementado en `musepipe/stats.py`:

- `finite_values()`
- `robust_sigma()`
- `robust_sigma_axis0()`
- `finite_percentile()`
- `median_finite()`
- `empirical_z()`
- `robust_limits()`

Implementado en `musepipe/io.py`:

- `read_json()`
- `write_json()`
- `write_csv()`
- `get_cube_data()`
- `read_wavelengths_and_masks()`

Implementado en `musepipe/spectral.py`:

- `nearest_channel_index()`
- `nearest_channel_indices()`
- `continuum_running_median()`
- `make_wavelength_mask()`

Implementado en `musepipe/apertures.py`:

- `angular_separation_deg()`
- `same_radius_control_positions()`
- `aperture_weights()`
- `box_spectrum_sum()`
- `box_spectrum_mean()`

`stage08_full_spectrum_for_modeling.py` ahora importa esas utilidades desde
`musepipe`. La logica especifica de ajuste local rapido queda ahi hasta Fase 5.

Verificaciones ejecutadas:

- `python -m compileall musepipe stage08_full_spectrum_for_modeling.py`
- Import de `stage08_full_spectrum_for_modeling`
- Pruebas puras con arrays sinteticos para estadistica, canales espectrales,
  mascara de longitud de onda, aperturas y controles al mismo radio.

Nota: el import de Stage 8 emitio warnings de cache de Matplotlib por
directorio local no escribible, pero termino correctamente.

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

Estado 2026-06-17: tests minimos creados con `unittest`, sin FITS pesados.

Archivos creados:

- `tests/test_stats.py`
- `tests/test_spectral.py`
- `tests/test_apertures.py`
- `tests/test_localfit.py`

Cobertura actual:

- Estadistica robusta: valores finitos, MAD con fallback, percentiles,
  mediana finita, z empirico y limites robustos.
- Espectral: canal mas cercano ignorando NaN, errores en entradas invalidas,
  continuo por mediana movil y mascara con rango Na-LGS.
- Aperturas: separacion angular, controles al mismo radio, pesos de aperturas
  pixel/box/circle/gaussian, sumas/medias de caja con recorte de bordes.
- Localfit: inicialmente marcado como `skip`; activado en Fase 5.

Verificaciones ejecutadas:

- `python -m unittest discover -s tests`
- Resultado inicial: `Ran 16 tests ... OK (skipped=1)`
- `python -m compileall musepipe tests stage08_full_spectrum_for_modeling.py`

### Fase 5 - sustraccion local comun

Crear `musepipe/localfit.py` con:

- `fit_local_surface_2d` robusto.
- `fit_fast_surface_coefficients`.
- `local_surface_spectra_fast`.
- Helpers de mascara, pesos y metadatos.

Criterio de salida:

- `stage08_full_spectrum_for_modeling.py` puede importar localfit.
- `Far_07b` puede reutilizar la misma extraccion local.

Estado 2026-06-17: completado.

Implementado en `musepipe/localfit.py`:

- `fit_local_surface_2d()`: ajuste local robusto por canal para modelos
  `constant` y `plane`, con mascara central, exclusiones extra y sigma clipping.
- `subtract_local_surface_cube()`: aplica `fit_local_surface_2d()` canal a
  canal y devuelve residual, modelo y numero de pixeles usados.
- `fit_fast_surface_coefficients()`: ajuste vectorizado por longitud de onda
  para mascara fija.
- `local_surface_spectra_fast()`: extraccion rapida de espectros residuales
  usada por Stage 8.

Cambios asociados:

- `stage08_full_spectrum_for_modeling.py` ya importa
  `local_surface_spectra_fast()` desde `musepipe.localfit`.
- `tests/test_localfit.py` dejo de estar saltado y ahora prueba plano sintetico,
  modelo constante, fallback por pocos pixeles, sustraccion de cubo, coeficientes
  rapidos y extraccion de fuente puntual.

Verificaciones ejecutadas:

- `python -m unittest discover -s tests`
- Resultado: `Ran 21 tests ... OK`
- `python -m compileall musepipe tests stage08_full_spectrum_for_modeling.py`
- Import de `stage08_full_spectrum_for_modeling` y `musepipe.localfit`
  correcto. Persisten warnings externos de cache de Matplotlib durante import.

### Fase 6 - migrar Stage 8

Transformar `stage08_full_spectrum_for_modeling.py` en consumidor de
`musepipe`.

Criterio de salida:

- `Far_08_full_spectrum_for_modeling.ipynb` sigue funcionando.
- `stage08_full_spectrum_model_input.csv` mantiene columnas y valores
  esperados.

Estado 2026-06-17: completado a nivel de codigo sin regenerar productos.

Cambios en `stage08_full_spectrum_for_modeling.py`:

- Mantiene `CONFIG` legacy para consumidores antiguos, apuntando a `ROXs12b`
  con `(x, y) = (72, 152)` y estrella `(y, x) = (85, 85)`. El wrapper
  `Far_08` ya no lo usa desde Fase 9.
- Agrega `default_apertures()` y `STAGE08_DEFAULTS`.
- Agrega `stage08_config_from_run()`, que usa `load_run_config()` y
  `RunPaths`, lee `stage04b_qc.json` si existe y resuelve coordenadas desde:
  claves explicitas de config/QC, `target_yx`/`target_xy`, o
  `detected_peaks[target_object]` para QCs antiguos.
- `run_stage08()` sin argumentos ahora usa el run activo (`MUSE_RUN_ID` o
  `active_run.txt`), mientras `run_stage08(CONFIG)` conserva el flujo antiguo.
- `run_stage08()` usa `RunPaths.from_project_root()` para resolver `stages`,
  `tables` y `plots`.

Verificaciones ejecutadas sin correr cubos pesados:

- `python -m unittest discover -s tests`
- Resultado: `Ran 21 tests ... OK`
- `python -m compileall musepipe tests stage08_full_spectrum_for_modeling.py`
- `stage08_config_from_run()` con `active_run.txt = ROXs12b_short` resuelve
  `object_xy=(72, 152)`, `star_yx=(85, 85)`.
- `CONFIG` legacy sigue resolviendo `ROXs12b`.
- `stage08_config_from_run("ROXs12b")` resuelve `(72, 152)`.
- `stage08_config_from_run("ROXs42Bb")` resuelve `(108, 61)` desde
  `detected_peaks[target_object]`.

Nota: se mantiene el warning externo de Fontconfig/Matplotlib durante imports,
pero no afecta la suite ni la construccion de config.

### Fase 7 - migrar Far_07b

Crear:

```text
musepipe/stages/stage07b_halpha_robustness.py
```

Estado: completado a nivel de codigo sin regenerar productos reales.

Responsabilidades:

- Leer Stage04b desde `runs/<RUN_ID>/stages`.
- Resolver objeto y estrella desde config/QC, incluyendo `detected_peaks[target_object]`.
- Extraer objeto y controles en el mismo radio respecto a la estrella.
- Reutilizar aperturas comunes (`pixel`, cajas, circulares y gaussiana).
- Calcular metricas Halpha de objeto contra controles leave-one-out.
- Guardar CSV, QC y plots con nombres actuales:
  - `stage07b_halpha_robustness_summary.csv`
  - `stage07b_halpha_control_metrics.csv`
  - `stage07b_halpha_robustness_qc.json`
  - `plots/stage07b_halpha_robustness/*.png`

Verificacion hecha:

- `stage07b_config_from_run("ROXs12b_short")` resuelve `object_xy=(72, 152)` y `star_yx=(85, 85)`.
- Tests puros para mascaras Halpha, metricas, controles leave-one-out, resolucion de coordenadas y cubo sintetico.
- La suite de `tests/` pasa sin ejecutar cubos pesados ni regenerar productos reales.
- `Far_07b_halpha_robustness_checks.ipynb` quedo como wrapper fino sobre `run_stage07b()`.

Criterio de salida:

- `Far_07b_halpha_robustness_checks.ipynb` queda como wrapper fino. Cumplido.

### Fase 8 - migrar Far_04b

Crear:

```text
musepipe/stages/stage04b_local_surface.py
```

Estado: completado a nivel de codigo sin regenerar productos reales.

Responsabilidades:

- Seleccionar entrada `stage02` (`native_stage02`) o `stage03`
  (`fakecont_stage03`).
- Resolver coordenadas `b/c` desde `stage04_qc.json` o fallback manual.
- Validar coordenadas contra el tamano real del cubo.
- Construir mascaras espectrales buenas/malas desde rangos Na-LGS.
- Sustraer superficie local por cubo usando `musepipe.localfit`.
- Rankear cubos por S/N Halpha en el objeto objetivo.
- Escribir productos compatibles actuales:
  - `stage04b_local_surface_cube_stack.fits`
  - `cube_residual_best4.fits`
  - `cube_residual_local_object.fits`
  - `cube_localmodel_best4.fits`
  - `cube_localmodel_object.fits`
  - `cube_fakecont_best4.fits`
  - `cube_input_local_object.fits`
  - `Ha/Hb image/SNR best4`
  - `stage04b_good_wavelength_mask.npy`
  - `stage04b_bad_wavelength_mask.npy`
  - `stage04b_cube_ranking.csv`
  - `stage04b_qc.json`
  - alias compatible `stage04_qc.json`

Verificacion hecha:

- Tests sinteticos para mascaras de longitud de onda, resolucion de coordenadas,
  ranking por Halpha, escritura FITS/CSV/QC y contrato de extensiones FITS.
- `stage04b_config_from_run("ROXs12b_short")` conserva defaults esperados:
  `input_mode="native_stage02"`, `target_object="c"`, canales Halpha historicos.
- `Far_04b_local_surface_subtraction.ipynb` quedo como wrapper fino sobre
  `run_stage04b()`.
- No se ejecuto Stage 04b sobre cubos reales en esta fase.

Criterio de salida:

- `Far_04b_local_surface_subtraction.ipynb` queda como wrapper fino. Cumplido.
- `ROXs12b_short` validado en una copia aislada. Cumplido.
- Validacion cientifica final sobre `ROXs12b`: cumplida posteriormente sobre
  `ROXs12b_refactor_validation`.

Validacion controlada de `ROXs12b_short` (2026-06-17):

- Se creo `ROXs12b_short_refactor_validation` mediante copia APFS; el run
  historico permanecio intacto.
- Formas, coordenadas, canales Halpha/Hbeta, parametros, `best4=[0]` y
  `worst2=[]` coinciden.
- El producto historico del 15 de mayo no era un golden file valido para la
  receta actual: fue generado sin clipping y antes de guardar mascaras/QC
  modernos. Su score Halpha era `0.469346`; la receta documentada de junio con
  tres iteraciones de clipping produce `0.545108`.
- Reconstruccion de procedencia: 69/69 canales muestreados del baseline de mayo
  coinciden exactamente con el plano sin clipping (`max_abs=0`); los mismos
  69/69 canales del notebook inmediatamente anterior al refactor coinciden
  exactamente con `musepipe.localfit` (`max_abs=0`).
- Stage 07b produjo 5 aperturas, 35 filas de controles, 7 posiciones de control
  y 3 plots validos.
- Stage 08 produjo 3681 filas, mascaras complementarias de 3465 canales buenos
  y 216 malos, valores finitos en todos los canales buenos, y contratos
  CSV/FITS/NPZ/QC consistentes. Sus 3 plots fueron inspeccionados.
- La suite completa permanece en `Ran 33 tests ... OK`.

### Fase 9 - documentacion y nombres

Actualizar:

- `docs/roxs12b_clean_spectrum_pipeline.md`
- `docs/plan_mejora_y_sugerencias.md`
- `README.md` raiz

Estado 2026-06-17: completado.

Decision de nombres:

- `musepipe/stages/stage04b_local_surface.py` y
  `musepipe/stages/stage07b_halpha_robustness.py` son las implementaciones
  canonicas.
- `stage08_full_spectrum_for_modeling.py` sigue siendo la implementacion
  canonica de Stage 8 mientras no se mueva al paquete.
- Los notebooks migrados con prefijo `Far_` se conservan como aliases
  historicos y capas de inspeccion. No deben contener una implementacion
  paralela.
- `Far_07` todavia no esta migrado; su prefijo no implica que haya sido
  unificado. `Far_06` se migra en Fase 10.
- No se renombran productos en `runs/`: los nombres `stage04b_*`, `stage07b_*`
  y `stage08_*` forman el contrato estable entre etapas.

Documentacion actualizada:

- Se creo `README.md` en la raiz con seleccion de run, arquitectura, comandos,
  advertencias de escritura y estado de validacion.
- `docs/roxs12b_clean_spectrum_pipeline.md` ahora distingue implementacion
  canonica, wrappers `Far_*`, inspeccion sin escritura y ejecucion real.
- `docs/plan_mejora_y_sugerencias.md` registra O1/O2/O3/O7 y la migracion de
  04b/07b sin presentar el diagnostico historico como estado actual.
- `Far_08_full_spectrum_for_modeling.ipynb` usa el run activo en vez del
  `CONFIG` legacy hardcodeado.

Criterio de salida:

- Convencion de nombres documentada y consistente. Cumplido.
- README raiz disponible. Cumplido.
- Suite completa: `Ran 33 tests ... OK`.
- `python -m compileall musepipe tests stage08_full_spectrum_for_modeling.py`
  termina correctamente.
- Los wrappers `Far_04b`, `Far_07b` y `Far_08` tienen cinco celdas, cero
  outputs incrustados y todas sus celdas de codigo compilan.
- Smoke test de configuracion con `ROXs12b_short`: `native_stage02`,
  `object_xy=(72, 152)` y apertura Stage 8 `box3_sum`.
- Validacion numerica con cubos reales permanece separada de esta fase.

### Fase 10 - inyeccion local-surface

Despues de estabilizar `stage04b` y `stage07b`:

- Portar grilla moderna de `06_low_memory_inject_halpha_signal`.
- Medir recuperacion a traves de sustraccion local.
- No iniciar esta fase hasta que el refactor estructural este validado.

Estado 2026-06-17: implementado a nivel de codigo y pruebas sinteticas, sin
ejecutar la grilla sobre cubos reales.

Implementacion canonica:

```text
musepipe/stages/stage06_local_surface_injection.py
```

Responsabilidades:

- Cargar la misma entrada Stage 02/03 y los mismos parametros locales de Stage
  04b.
- Resolver estrella, objeto de referencia e inyeccion opuesta desde config/QC.
- Inyectar una PSF espacial y un perfil Halpha normalizados en unidades
  fisicas; para Stage 03 deshacer `MEDPIX` al insertar la senal.
- Reutilizar la seleccion `stage04_best4_indices.npy` cuando existe.
- Barrer una grilla configurable de S/N y recalcular la sustraccion local
  robusta dentro de una ventana diagnostica Halpha.
- Medir recuperacion realista contra controles al mismo radio y transferencia
  determinista con `injected - baseline`.
- Guardar productos con prefijo `stage06_local_surface_`, sin sobrescribir los
  productos PCA ni los productos historicos genericos.

Productos:

- `stage06_local_surface_halpha_injection_recovery.csv`
- `stage06_local_surface_halpha_injection_qc.json`
- `stage06_local_surface_halpha_injection_truth_summary.json`
- `stage06_local_surface_halpha_signal_template.fits`
- `stage06_local_surface_recovered_halpha_nominal_diag.fits`
- `stage06_local_surface_halpha_injection_summary.png`

Verificacion sintetica:

- La integral espacial/espectral de la plantilla es uno.
- La conversion mediante `MEDPIX` conserva el flujo fisico inyectado.
- La geometria coloca la inyeccion opuesta al objeto de referencia.
- La recuperacion es monotona y la transferencia matched-filter sintetica es
  aproximadamente `0.969`.
- FITS, CSV, QC y truth JSON mantienen el contrato esperado.
- `Far_06_inject_halpha_signal.ipynb` queda como wrapper de cinco celdas sin
  outputs incrustados.
- Suite completa: `Ran 38 tests ... OK`.

Validacion controlada:

- Primera ejecucion sobre `ROXs12b_short_refactor_validation`: cumplida en
  Fase 12.
- Validacion cientifica sobre `ROXs12b`: cumplida posteriormente sobre una
  copia aislada.

### Fase 11 - migrar Far_07

Estado 2026-06-17: completado a nivel de codigo y pruebas sinteticas, sin
regenerar productos reales.

Implementacion canonica:

```text
musepipe/stages/stage07_accretion_lines.py
```

Responsabilidades:

- Resolver run, estrella y objeto desde config/Stage 04b QC.
- Extraer pixel central y caja integrada para estrella y objeto.
- Recalcular controles al mismo radio con la sustraccion local robusta.
- Medir las 24 lineas candidatas del notebook historico y excluir candidatas
  dentro de rangos espectrales malos.
- Conservar los productos existentes:
  - `stage07_accretion_line_metrics.csv`
  - `stage07_accretion_line_qc.json`
  - cuatro figuras resumen y figuras individuales en
    `plots/stage07_accretion_lines/`.

Correccion confirmada:

- `Far_07` fijaba `STAR_YX=(20,20)` con un comentario obsoleto de crop `40x40`.
  El Stage 02 de ROXs12b es `170x170` y Stage 04b QC ubica la estrella en
  `(85,85)`. La implementacion nueva usa config/QC y registra por separado
  `star_yx` y `star_spectrum_yx`.

Verificacion:

- Cinco pruebas nuevas cubren config, filtros de lineas, metricas, controles y
  contrato CSV/QC.
- Cubo sintetico con plano, Halpha y siete controles produce recuperacion
  positiva en objeto y estrella.
- Las cinco celdas del wrapper compilan y no contienen outputs.
- Las cinco familias de figuras sinteticas se generaron y la figura resumen se
  inspecciono visualmente.
- Suite completa: `Ran 43 tests ... OK`.
- `ROXs12b`, `ROXs12b_short`, `ROXs42Bb`, `LkCa_15` y el run historico de
  inyeccion resuelven coordenadas sin leer cubos. `YSES 2b` falla temprano por
  falta de Stage 04b QC/coordenadas validas.

Validacion controlada:

- Primera ejecucion sobre `ROXs12b_short_refactor_validation`: cumplida en
  Fase 12.
- Validacion cientifica sobre `ROXs12b`: cumplida posteriormente sobre una
  copia aislada.

### Fase 12 - validacion integrada de Stage 06 local y Stage 07

Estado 2026-06-17: completado sobre la copia aislada
`ROXs12b_short_refactor_validation`. No se escribio en `ROXs12b`.

Stage 06 local-surface:

- Uso `native_stage02`, cubo seleccionado `[0]`, estrella `(85,85)`, objeto de
  referencia `(152,72)` e inyeccion opuesta `(18,98)`.
- Coloco ocho controles validos al mismo radio y produjo las siete filas de la
  grilla de S/N `[0,0.1,0.5,1,2,3,5]`.
- La plantilla integra `0.99999994` y los FITS de plantilla/diagnostico tienen
  las extensiones y formas esperadas.
- La transferencia matched-filter delta es `0.99823993`, con dispersion
  `7.8e-8`, y la recuperacion realista es estrictamente monotona.
- En esta posicion el residual baseline es `-3.954 sigma`. La relacion medida
  es `S/N_realista = 0.99824 * S/N_inyectado - 3.95403`; por eso la inyeccion
  nominal de entrada `3 sigma` queda en `-0.959 sigma` y una de `5 sigma` llega
  a `1.037 sigma`. Esto es un limite del fondo local, no perdida de throughput.
- La figura resumen fue inspeccionada y separa correctamente baseline,
  inyeccion y transferencia.

Stage 07:

- Resolvio estrella `(85,85)`, objeto `(152,72)` y siete controles.
- Produjo 23 filas: las 24 candidatas originales menos `He I 5876`, excluida
  correctamente por el rango malo `5780-6050 A`.
- Las 23 metricas `object_box_ctrlsub_line_peak_snr` son finitas; Halpha es la
  candidata de mayor pico con `S/N=4.22419` y flujo integrado historico
  `76.9676` en unidades nativas integradas en Angstrom.
- El pico Halpha coincide con Stage 07b `box3_sum` a `7.7e-7`. Los flujos no
  se comparan directamente: Stage 07 multiplica por `dlam=1.25 A`, mientras
  Stage 07b conserva la suma por canales.
- Se generaron 27 PNG validos: cuatro resumen y 23 ventanas individuales. Se
  inspeccionaron el espectro completo, la grilla y la ventana Halpha.

Criterio de salida:

- Contratos reales de Stage 06 local y Stage 07 aprobados en el run corto.
- Interpretacion cientifica registrada sin confundir transferencia delta con
  detectabilidad realista.
- Validacion final sobre `ROXs12b`: cumplida posteriormente sobre una copia
  aislada.

### Validacion final de ROXs12b

Estado 2026-06-17: completada sobre `ROXs12b_refactor_validation`, una copia
APFS de 5.1 GB. El run original `ROXs12b` no fue modificado.

Stage 04b:

- Recalculo completo en 15.6 s.
- Doce productos FITS, incluidos residual, modelo local, entrada y mapas
  Halpha/Hbeta, coinciden exactamente con el baseline: `max_abs=0`, `RMS=0` y
  cero pixeles diferentes.
- Mascaras buena/mala, ranking e indices best/worst son identicos.
- El QC solo cambia run/rutas, tiempo y metadatos nuevos esperados.

Stage 06 local-surface:

- Transferencia delta matched-filter `0.9982315`, recuperacion realista
  estrictamente monotona y plantilla integrada `1.00000009`.
- El baseline en la posicion opuesta es `-0.981 sigma`; una inyeccion de
  entrada `3 sigma` se recupera en `2.013 sigma` y una de `5 sigma` en
  `4.010 sigma`.
- Ocho controles, FITS/CSV/JSON y figura resumen aprobados.

Stage 07:

- Produce 23 candidatas validas y excluye correctamente `He I 5876` dentro de
  `5780-6050 A`.
- Las metricas coinciden con el baseline; Halpha es la candidata de mayor pico
  con `S/N=4.17829`.
- Se corrigio la limpieza de ventanas antiguas: el directorio ahora contiene
  exactamente los 23 plots declarados y 27 PNG totales.

Stage 07b:

- Cinco aperturas, siete posiciones y 35 filas de control.
- Diferencias frente al baseline entre `1e-14` y `4e-13`.
- Halpha alcanza pico local `S/N=4.161` en caja 3x3, pero el flujo no es
  excepcional frente a controles (`z_flux=-0.075`, percentil `42.9%`). No se
  interpreta como deteccion robusta aislada.

Stage 08:

- 3681 canales, con mascaras complementarias de 3465 buenos y 216 malos.
- Todos los productos son finitos en canales buenos y `NaN` en canales malos
  para las cinco aperturas.
- Diferencia maxima frente al baseline: `1.3e-11` en S/N y menos de `1e-12`
  en flujos; CSV, NPZ y FITS conservan formas y patrones de `NaN`.
- Se eliminaron warnings esperados de reducciones all-NaN exclusivamente en
  canales enmascarados, con una prueba de regresion dedicada.
- Las tres figuras fueron inspeccionadas y muestran correctamente la mascara,
  el espectro S/N y la comparacion de aperturas.

Cierre:

- Cadena `04b -> 06 local -> 07 -> 07b -> 08` aprobada para ROXs12b.
- Suite final: `Ran 45 tests ... OK`.
- Compilacion completa y `git diff --check` correctos.

## Checklist Fase 0

- [x] Confirmar raiz de trabajo.
- [x] Confirmar ausencia de repo Git activo.
- [x] Crear `.gitignore` conservador.
- [x] Inventariar notebooks, modulos, docs y runs.
- [x] Identificar riesgos inmediatos en `Far_*`.
- [x] Definir runs de validacion.
- [x] Inicializar o conectar Git antes de cambios grandes.
- [x] Pasar a Fase 1: crear paquete `musepipe/`.
