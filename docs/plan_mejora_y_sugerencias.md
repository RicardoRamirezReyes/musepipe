# Plan de mejora y sugerencias — Pipeline 15_MUSE

Fecha: 2026-06-10. Revisado: 2026-06-11, tras la reorganización (eliminación de `Untitled`/`-Copy1`, cadena `Far_*` para objetos lejanos, nuevo `06_inject_halpha_signal` con inyección-recuperación por PCA). Revisado: 2026-06-14, tras el cambio de `crop_npix` a 50 y la incorporación de chequeos de consistencia de crop. Revisado: 2026-06-17, tras las Fases 0-9 del refactor de objetos lejanos. Auditado: 2026-06-18 contra el árbol de trabajo y los QC reales disponibles. Actualizado: 2026-06-21 con los cierres de C2 y C4.

> Estado auditado 2026-06-18: el diagnóstico original se conserva como registro
> del punto de partida. La cadena lejana canónica `04b -> 06 local -> 07 -> 07b
> -> 08` fue validada sobre la copia aislada `ROXs12b_refactor_validation`; 04b
> coincide exactamente con el baseline y 07b/08 concuerdan hasta precisión
> numérica. La suite contiene 52 pruebas. También existen ejecuciones reales de
> 01b, 03b, 09, 10 y 11 sobre `LkCa_15`. C1 fue cerrado después sobre la copia
> aislada `ROXs12b_HaInject_SNR5_C1_validation`. Siguen pendientes C6, C7,
> la regresión real automatizada, logging/trazabilidad y la migración de los
> notebooks restantes. El checkpoint de O1 registra el estado validado y añade un
> `environment.yml` reproducible; el filtrado automático de outputs queda como
> mejora opcional.

### Resumen de completitud auditado

| Ítem | Estado | Evidencia o pendiente principal |
|---|---|---|
| C1 | Completo y ejecutado | Inyección nominal matched-filter de 5 sigma propagada por 04b/07b/08; `box3_sum` recupera S/N medio 8.80 frente a 8.97 esperado |
| C2 | Completo y ejecutado | Local-surface y PCA evaluados en 3 líneas × 3 posiciones × 9 niveles de S/N; existen throughput, completitud, QC y figura consolidados |
| C3 | Completo y ejecutado | Stage 11 ejecutado en `LkCa_15`; ruido empírico/propagado = 3.63 |
| C4 | Completo y ejecutado | 31 controles al mismo radio calibran máximos sobre 3465 canales; máximo del objeto FAP global 90.6% y Halpha FAP global 100% |
| C5 | Parcial | Stage 10 produce límites físicos para una posición/PA; falta cobertura espacial/espectral y cierre de caveats publicables |
| C6 | Pendiente | La equivalencia con el baseline no reemplaza la comparación rápida-vs-robusta por canal propuesta |
| C7 | Pendiente | No hay CCF con templates, corrección baricéntrica ni ajuste de RV |
| O1 | Completo para el baseline | Git, reglas de exclusión, entorno fijado y checkpoint del estado validado |
| O2 | Parcial | 04b/06/07/07b y utilidades comunes migradas; faltan stripes, plotting y notebooks diagnósticos |
| O3 | Parcial | Fuente única de `RUN_ID` en las etapas migradas, no en todos los notebooks |
| O4 | Parcial | Hay variantes low-memory e `IncrementalPCA`, pero no están consolidadas como ruta canónica |
| O5 | Pendiente | Los 04c y los dos 00_config siguen como copias separadas |
| O6 | Pendiente | No hay logging formal ni hashes/versiones de entradas en todos los QC |
| O7 | Parcial | 52 pruebas y validación manual real; falta regresión automatizada con producto real pequeño |

## 1. Diagnóstico general

Lo que está bien y conviene conservar:

- Arquitectura por etapas (00→08) con `config.json` como fuente única de verdad y QC JSON por etapa. Es el punto más fuerte del proyecto.
- Salidas organizadas por run en `runs/<RUN_ID>/{config,stages,tables,plots}`.
- Stage 8 ya extraído a un módulo `.py` importable y ejecutable por CLI: ese es exactamente el patrón a replicar.
- Uso de controles al mismo radio, comparación de aperturas y métricas empíricas (07b) es metodológicamente sólido.
- Nuevo helper `muse_crop_checks.py`: los notebooks que leen productos guardados comparan la forma espacial del FITS/QC contra `CFG["crop_npix"]` y avisan si se mezclan productos de un crop antiguo con la configuración actual. En etapas con coordenadas críticas, el chequeo se vuelve estricto.

Problemas principales encontrados:

| # | Problema | Evidencia | Riesgo |
|---|---|---|---|
| 1 | ~20.000 líneas de código viven en celdas de notebook, con funciones duplicadas 3–6 veces (`robust_sigma` ×6, `nearest_channel_indices` ×6, `find_centroid_peak` ×5, toda la maquinaria de stripes ×3…) | grep sobre los 19 notebooks | Una corrección de bug en un notebook no se propaga; divergencia silenciosa entre etapas |
| 2 | `RUN_ID` se cambia comentando/descomentando líneas en *cada* notebook (el patrón persiste en el nuevo 06 y en la cadena `Far_*`) | celda 1–2 de cada notebook | Mezclar etapas de objetos distintos en un mismo run sin ningún error visible |
| 3 | No hay control de versiones dentro de `15_MUSE` (ni `.gitignore`); 23 GB de FITS conviven con el código; quedan `.ipynb_checkpoints/`, `__pycache__/`, `.DS_Store`. ~~`Untitled.ipynb` y `-Copy1`~~ ya eliminados ✔ | inspección de carpeta | Pérdida de historia, imposible saber qué versión produjo cada resultado |
| 4 | Notebooks con outputs embebidos de hasta 10 MB (04c ×2, 04d, 07) | tamaños de archivo | Lentos de abrir, indiferenciables, imposibles de revisar |
| 5 | Los dos `04c` son ~75% idénticos entre sí; `00_config` y `00_config_ERIS` también divergen como copias | diff entre versiones extraídas | Mismo riesgo que #1 |
| 6 | Ruta absoluta de datos (`/Users/ricardoramirez/Downloads/...`) escrita dos veces en `00_config` | 00_config celdas 3–4 | No portable; al re-correr en otra máquina falla tarde |
| 7 | Nunca se usa la extensión STAT (varianza) de los cubos MUSE; todos los errores son empíricos (`snr_like`, `control_sigma`) | grep STAT en todo el código | Sin incertidumbres calibradas para modelado posterior |
| 8 | PCA (Stage 04): matriz `X` en float64 de `N·ny·nx × nz`. Con crop 170×170 y ~3700 canales serían ~25 GB por solución, ×4 candidatos de ncomp | `pca_subtract_cube_stack` | Límite duro de memoria al escalar ROXs12b a múltiples cubos |
| 9 | `LOG_DIR` previsto pero sin logging; no hay `requirements.txt`/`environment.yml`; no hay tests | estructura | Reproducibilidad y debugging difíciles |
| 10 | **(Nuevo, 2026-06-11)** La cadena `Far_*` (04b/07/07b/08) es una copia íntegra de la cadena original — el código es idéntico salvo una celda markdown. Ahora existen dos cadenas paralelas (cercana/PCA vs lejana/local-surface) que comparten ~95% de la maquinaria | diff Far_* vs versiones previas | Duplica el problema #1: cada bug habrá que arreglarlo en dos familias de notebooks |

## 2. Plan de mejora — objetivos científicos

Ordenado por prioridad; cada ítem es independiente.

### C1. Validar la receta con la inyección — *completo y ejecutado* ✔
Para la cadena lejana: correr las implementaciones canónicas
`stage04b→stage07b→stage08` sobre un run de inyección consistente y verificar
que `box3_sum` recupera la señal con el SNR esperado. Los notebooks `Far_*` son
ahora aliases de inspección para 04b/07b/08. `ROXs12b_HaInject_SNR5` mantiene un
desajuste histórico entre nombre de carpeta y `cfg.run_id`, que debe corregirse
antes de usar la validación estricta. Para la cadena cercana, el nuevo `06` ya
cierra el lazo inyección→PCA→recuperación dentro del mismo notebook.

**Cierre 2026-06-18**: se normalizó el identificador únicamente en la copia
aislada `ROXs12b_HaInject_SNR5_C1_validation` y se ejecutó la cadena
`04b -> 07b -> 08`; el run histórico original no fue modificado. La diferencia
contra `ROXs12b` está confinada a 918 voxeles, nueve canales
`6558.28-6568.28 A` y una caja espacial `13x13` centrada en `(108,61)`. Su flujo
integrado es `1088.285`, consistente con la verdad original: S/N matched-filter
nominal `5`, y S/N esperado `8.97` para la apertura 3x3.

Stage 04b preservó la coordenada `(108,61)` y los 3465 canales buenos. En
Stage 07b, `box3_sum` recuperó `line_mean_snr=8.803` (diferencia `-1.9%` frente
al valor esperado), `line_peak_snr=17.718` y `z_flux=8.536`; el objeto quedó por
encima de los siete controles. Las cifras `5`, `8.97` y `17.718` no son
intercambiables: corresponden respectivamente a matched filter integrado,
apertura 3x3 integrada y pico espectral dividido por la dispersión del continuo.
Stage 08 ubicó el máximo de Halpha en `6563.283 A` con `SNR-like=7.288`; el
baseline evaluado en la misma posición da `0.840`. Espectros, distribución de
controles y mapa residual fueron inspeccionados visualmente y son coherentes con
la inyección.

### C2. Grilla de inyección-recuperación, no un solo punto — *completo y ejecutado* ✔

Ambas cadenas tienen ahora la misma superficie experimental: Hbeta `4861.33 A`,
Halpha `6562.80 A` y O I `8446.36 A`; tres posiciones al mismo radio; y S/N de
entrada `{0, 0.1, 0.5, 1, 2, 3, 5, 7, 10}`. La completitud usa un umbral de
recuperación de `5 sigma` y representa la fracción de los nueve casos
línea/posición detectados, no una probabilidad Monte Carlo.

**Cadena local-surface (2026-06-18)**: la implementación vive en
`musepipe/stages/stage06_local_surface_sweep.py` y fue ejecutada sobre
`ROXs12b_refactor_validation`. La transferencia delta matched-filter
`p16/mediana/p84` fue `0.998225/0.998230/0.998232`. La completitud alcanzó 50%
en S/N de entrada `5.25` (matched) y `4.25` (apertura), y 90% en `7.30` y
`8.65`, respectivamente.

**Cadena PCA (2026-06-21)**: `06c_pca_c2_multiline_40.ipynb` repitió
IncrementalPCA después de cada inyección sobre una vista central `40x40` de
`LkCa_15`, en PA `45/135/225 deg`. Los nueve casos y 81 puntos fueron
consolidados por `musepipe/stages/stage06_pca_c2_summary.py`. La transferencia
delta matched-filter `p16/mediana/p84` es
`0.987974/0.988067/0.988209`; por tanto PCA conserva aproximadamente 98.8% del
template. La apertura 3x3 conserva `0.522/0.593/0.630`, porque captura solo una
parte del template espacial/espectral y no por una pérdida adicional de PCA.

Con S/N de entrada 10, la completitud global es `7/9 = 77.8%` por matched
filter y `6/9 = 66.7%` por apertura. La interpolación monotónica sitúa el 50%
en `8.93` y `9.10`; el 90% no se alcanza dentro de la grilla. Halpha es el caso
limitante por residuales basales negativos en dos PA, pese a mantener el mismo
throughput delta. Productos finales:

- `runs/LkCa_15/tables/stage06_pca_c2_40/stage06_pca_c2_40_grid.csv`
- `runs/LkCa_15/tables/stage06_pca_c2_40/stage06_pca_c2_40_case_summary.csv`
- `runs/LkCa_15/tables/stage06_pca_c2_40/stage06_pca_c2_40_completeness.csv`
- `runs/LkCa_15/stages/stage06_pca_c2_40_summary_qc.json`
- `runs/LkCa_15/plots/stage06_pca_c2_40/stage06_pca_c2_40_summary.png`

### C3. Usar la extensión STAT de MUSE — *implementado* ✔ (2026-06-11)
Implementado en `11_stat_variance_propagation.ipynb`. Lee la extensión STAT de los cubos crudos (autodetección por EXTNAME, override `stage11_stat_ext`) y replica sobre la varianza la cadena geométrica registrada en los QC: crop (exacto) → shift espacial subpixel (**exacto** vía kernel de respuesta a impulso al cuadrado — el spline cúbico de stage01 no es bilineal; validado por Monte Carlo con ratio MC/propagada = 0.998–1.001) → regrid espectral lineal (exacto, pesos²) → shifts de stripes de stage02 (κ exacto por kernel 1D, validado a 3 decimales vs MC; promedio por cubo) → keep_mask de stage03. PCA/sustracción local tratadas como operadores deterministas (estándar).

Productos: `stage11_variance_cube_stack.fits`, espectro de error por canal en la posición objetivo (`stage11_propagated_error_spectrum.csv`), columna `flux_err_propagated` fusionada al CSV de Stage 8 si existe, y el QC clave de C3: comparación σ_propagada vs σ_empírica (stage09) vs σ_asumida (stage06) en la posición de inyección — el ratio empírica/propagada cuantifica los sistemáticos residuales. Aproximaciones declaradas en QC: voxeles independientes en sumas de apertura (σ propagada es piso), κ por cubo y no por franja, cancelación fakecont/MEDPIX a primer orden. Nota: requiere acceso a los cubos crudos (DATA_DIR), por lo que debe ejecutarse en la máquina local.

**Ejecución real confirmada**: `runs/LkCa_15/stages/stage11_variance_qc.json`
contiene 32 cubos y una pila `(32, 3465, 40, 40)`. En la posición de inyección,
`sigma_propagated=22.65`, `sigma_empirical=82.24` y el cociente es `3.63`: los
sistemáticos residuales dominan sobre el piso STAT propagado.

### C4. Tasa de falsos positivos y look-elsewhere — *completo y ejecutado* ✔

`musepipe/stages/stage08c_look_elsewhere.py` implementa una busqueda unilateral
de lineas en emision sin reemplazar Stage 8. En
`ROXs12b_refactor_validation` extrajo 31 controles al mismo radio a partir de
36 angulos candidatos. Para cada control construyo un espectro nulo
leave-neighborhood-out, excluyendo la posicion objetivo y vecinos dentro de
`15 deg`; cada nulo uso 28-29 referencias. Los máximos se buscaron sobre los
mismos 3465 canales buenos y conservaron por construccion la correlacion
espectral y los residuos no gaussianos reales.

El maximo global del objeto fue `snr_like=4.433` en `8565.78 A`, mientras los
maximos nulos abarcaron `4.266-7.064`; su FAP global empirica es `29/32 =
90.625%`. Los umbrales globales son `5.816` para FAP 10% y `6.953` para FAP
5%. La resolucion finita es `1/(31+1)=3.125%`, por lo que FAP 1% se declara
explicitamente no resoluble y no se extrapola.

En la ventana Halpha `6562.8 +/- 5 A`, el maximo es `snr_like=2.049` en
`6564.53 A`. La FAP puntual del canal es `6.25%`, la FAP que incluye la busqueda
dentro de la ventana es `31.25%` y la FAP global sobre todo el espectro es
`100%`. Por tanto no hay evidencia de una linea Halpha significativa ni de
otro pico global en Stage 8. Productos:

- `runs/ROXs12b_refactor_validation/tables/stage08c_null_maxima.csv`
- `runs/ROXs12b_refactor_validation/tables/stage08c_object_candidates.csv`
- `runs/ROXs12b_refactor_validation/tables/stage08c_global_thresholds.csv`
- `runs/ROXs12b_refactor_validation/stages/stage08c_look_elsewhere_qc.json`
- `runs/ROXs12b_refactor_validation/plots/stage08c_look_elsewhere/stage08c_look_elsewhere_summary.png`

La limitacion restante es espacial: los controles densos al mismo radio pueden
estar correlacionados entre si. El QC lo declara y evita interpretar la grilla
como 31 ensayos perfectamente independientes.

### C5. Límite superior de acreción — *parcial, implementado para una posición*
Aunque Hα no sea robusta, el producto científico natural es un límite superior de F(Hα) → L(Hα) → Ṁ (escalas tipo Alcalá et al.). C2 ya aporta throughput y variación espacial/espectral; falta integrar esos productos con la calibración de flujo. Stage 10 hace la conversión para `LkCa_15`, pero todavía consume el caso histórico de una sola separación/PA y no constituye un mapa de límites.

### C6. Stage 8: cuantificar el costo del modo rápido — *pendiente*
El caveat documentado (Stage 8 no repite el sigma-clipping de 04b) es verificable: para ~50 canales aleatorios, comparar el ajuste rápido vs. el robusto y guardar la diferencia en el QC. Si es despreciable, el caveat desaparece; si no, sabes en qué canales desconfiar.

### C7. Identificación espectral con el espectro de Stage 8 — *pendiente*
Siguiente paso natural tras C1–C3: correlación cruzada del `flux_continuum_sub` contra templates (BT-Settl/atmósferas de baja gravedad) en el rango MUSE, con corrección baricéntrica y RV como parámetro libre. La detección por CCF de muchas líneas débiles es más sensible que cualquier línea individual — coherente con tu decisión de no apostar todo a Hα.

## 3. Plan de optimización del código

Ordenado para que cada paso facilite el siguiente.

### O1. Higiene y git (medio día) — *completado para el baseline* ✔
1. Git y `.gitignore` ya están activos. Se ignoran `runs/`, caches, ruido local y `active_run.txt`.
2. ~~Eliminar `Untitled.ipynb` y `02b_...-Copy1.ipynb`~~ Hecho ✔ (2026-06-11).
3. ~~Añadir `environment.yml` o `requirements.txt` con versiones fijadas~~ Hecho con `environment.yml` ✔ (2026-06-18).
4. Opcional: `nbstripout` como filtro git para no versionar outputs de notebooks.

**Cierre 2026-06-18**: el checkpoint de O1 incorpora notebooks, `musepipe/`,
tests, README, documentación e informe; excluye runs, caches, selección local
de run y configuración personal. `environment.yml` fija las versiones que
pasaron la suite y añade `maoppy`, requerido por el perfil activo de Stage 01.

### O2. Extraer la librería común (el cambio de mayor retorno, 2–4 días) — *parcialmente implementado* ✔
Se creó `musepipe/` y ya contiene:

```text
musepipe/
  config.py      # carga/validación de config.json, resolución de paths por RUN_ID
  io.py          # get_cube_data, read_wavelengths_and_masks, write_fits, write_csv, QC json
  stats.py       # robust_sigma, robust_limits, finite_percentile, empirical_z
  spectral.py    # nearest_channel_indices, make_wavelength_mask, continuum_running_median
  apertures.py   # aperture_weights, box_spectrum_sum, same_radius_control_positions
  localfit.py    # fit_local_surface_2d, local_surface_spectra_* (versión robusta y rápida)
  plotting.py    # esqueleto; extraccion de helpers aun pendiente
  stages/        # Stage 04b, 06 local-surface, 07 y 07b
```

Los notebooks `Far_04b`, `Far_06`, `Far_07` y `Far_07b` ya son capas finas: cargar config → llamar
funciones → mirar plots. `stage08_full_spectrum_for_modeling.py` consume las
utilidades comunes. Quedan pendientes `stripes.py`, parte de plotting y la
migración del resto de notebooks duplicados.

**Decisión 2026-06-17**: `Far_` se conserva temporalmente como alias histórico,
no como familia de implementaciones. El código canónico migrado vive en
`musepipe/stages/`; los productos mantienen sus nombres `stage04b_*`,
`stage07b_*` y `stage08_*` para no romper consumidores existentes.

### O3. RUN_ID único (medio día, gran reducción de riesgo) — *implementado en etapas migradas* ✔
Las etapas migradas usan una sola fuente:

```python
# musepipe/config.py
def get_run_id():
    return os.environ.get("MUSE_RUN_ID") or Path("active_run.txt").read_text().strip()
```

Un `run_id` pasado a la API es explícito; si se omite, la prioridad es
`MUSE_RUN_ID` → `active_run.txt`. La carga valida que `CFG["run_id"]` coincida
con la carpeta, salvo override deliberado. El patrón todavía debe extenderse a
los notebooks no migrados.

### O4. Memoria y velocidad (1–2 días, cuando escale ROXs12b)
- PCA: mantener `X` en float32 (sklearn lo acepta) y usar `svd_solver="randomized"` para el scan de ncomp ∈ {1..4}; solo la solución final con solver completo si se quiere. Reduce memoria ~2× y tiempo ~5–10×. Alternativa para crops grandes: `IncrementalPCA`.
- `continuum_running_median` es O(nz²) con búsqueda de máscara por canal (~3700² operaciones). Reemplazar por ventana deslizante sobre índices ordenados o `scipy.ndimage.median_filter` sobre la grilla regular: de minutos a segundos.
- Aprovechar `memmap=True` consistentemente y recortar (`cube[:, y1:y2, x1:x2]`) *antes* de `astype`, no después, en las etapas que cargan stacks de 400 MB–GB.
- Unificar paralelización: hoy conviven `ThreadPoolExecutor` ad-hoc y claves `*_n_jobs` por etapa. Una utilidad común `parallel_map(fn, items, n_jobs)` + una sola clave de config con overrides por etapa.

**Estado auditado**: existen variantes `01_lowMemory`, `02_low_memory`,
`03_low_memory` y `04_low_memory`; esta última usa `IncrementalPCA` y bloques
espaciales. Stage 09 usa PCA randomized en sus tests de escalado. Son avances
reales, pero siguen como rutas paralelas y no cierran la consolidación propuesta.

### O5. Fusionar duplicados estructurales (1 día)
- Los dos `04c` (~75% idénticos): un solo notebook con parámetro `INPUT_KIND ∈ {pca_residual, fakecont}`.
- `00_config` vs `00_config_ERIS`: un config base + diccionario de overrides por instrumento, en vez de copia divergente.

### O6. Logging y trazabilidad (medio día)
- Usar el `LOG_DIR` ya previsto: `logging` a archivo por etapa con timestamp, en lugar de `print`.
- Añadir a cada QC JSON: hash git del código, versiones de paquetes y hash/fecha de los archivos de entrada. Con eso cada resultado es trazable a código + datos exactos.

### O7. Tests mínimos (1 día, después de O2) — *implementado para el alcance migrado* ✔
La suite contiene 52 pruebas para estadística, espectro, aperturas, ajuste local,
Stage 04b, Stage 06 local-surface, Stage 07, Stage 07b, Stage 08 y Stage 08c. Incluye cubos
sintéticos, ranking Halpha, inyección-recuperación, canales enmascarados y
contratos de FITS/CSV/QC. `ROXs12b_short` y `ROXs12b` tienen regresiones reales
manuales documentadas; falta convertir una de ellas en un test automatizado y
pequeño.

## 4. Sugerencias adicionales

- **Orquestador ligero — pendiente**: con O2 hecho, un `run_pipeline.py --run-id ROXs12b --stages 01,02,03,04b,08` que ejecute etapas en orden, valide prerrequisitos y registre tiempos. Evita el error de correr etapas fuera de orden o sobre el run equivocado.
- **Manifiesto por run — pendiente**: `runs/<RUN_ID>/manifest.json` actualizado por cada etapa (etapa, fecha, entradas, salidas, hash de config). Responde de un vistazo "¿este run está completo y consistente?".
- **Gestión de disco — pendiente**: 23 GB y creciendo. Los stacks intermedios (stage01, stage03) son regenerables; un script `prune_run.py` que borre intermedios conservando config + QC + tablas + productos finales liberaría ~60–70% por run. `ROXs12b_ERIS` está vacío (0 bytes) — decidir si va o se borra.
- **Nombres de run sin espacios — hecho (2026-07-10)**: `YSES 2b` → `YSES_2b` (los espacios en rutas acaban rompiendo algo en shell/FITS headers).
- **Documentación**: completado en Fase 9. El README raíz describe arquitectura,
  selección de run, nombres canónicos, comandos y estado de validación.
- **Stage 8 como plantilla**: aplicado a `Far_04b`, `Far_07` y `Far_07b`; delegan en
  módulos importables y conservan el notebook para inspección.

## 5. Revisión de `07_integration_time_flux_projection` (2026-06-11) — IMPLEMENTADO ✔

**Estado 2026-06-11**: todas las sugerencias de esta sección fueron implementadas en `09_integration_time_flux_projection.ipynb` (reemplaza al notebook `07_`, eliminado; outputs ahora `stage09_*`). Detalles de la implementación al final de la sección.

Qué hace: suma tiempos de exposición desde headers, lee la recuperación de inyección PCA de Stage06, ajusta S/N vs flujo inyectado (recta por el origen) y proyecta límites de flujo a 3σ/5σ con escalado √t para un multiplicador de tiempo.

Lo correcto y a conservar: la distinción flujo-vs-incertidumbre al duplicar tiempo está bien planteada (el flujo recuperado no se duplica, σ baja como 1/√t); las suposiciones quedan explícitas en el QC; los productos (CSV/QC/figura) siguen la convención del pipeline; el fit por el origen reporta `slope_sigma` y `scatter`.

### Sugerencias científicas, por prioridad

**S1. Validar empíricamente el escalado √t — la mejora más importante y la tienes gratis.**
El √t solo vale si el ruido es aleatorio e independiente entre exposiciones. A pocas λ/D tras sustracción PSF/PCA, los speckles cuasi-estáticos suelen promediar más lento que √t. Con 32 cubos puedes medirlo: repetir la recuperación de Stage06 sobre subconjuntos de N = 4, 8, 16, 32 cubos y ajustar σ_F ∝ N^(−α). Si α ≈ 0.5, la proyección queda justificada; si α < 0.5, proyectar con el α medido. Esto convierte la suposición central del notebook en un resultado medido, y de paso responde si una segunda noche mejora la PCA (más diversidad de frames) o no.

**S2. La recta por el origen asume throughput PCA independiente del flujo — verificarlo.**
La autosustracción de PCA depende de la intensidad de la señal inyectada: a flujos altos la inyección perturba la base PCA y la relación S/N–flujo se aplana, sesgando la pendiente (y por tanto el límite) hacia abajo. Sugerencia: graficar η(flujo) punto a punto en lugar de solo la mediana; ajustar la pendiente usando solo la región S/N ≲ 5 (donde vive el límite); reportar un test de linealidad con los residuos que ya calculas.

**S3. Usar la fila de flujo cero como piso de sistemáticos.**
La grilla de Stage06 incluye SNR=0 pero `good_flux` la excluye (flux > 0). Esa fila es valiosa: el S/N del matched filter con inyección nula en esa posición es el piso de falsos positivos del método. Reportarlo en el QC y exigir que el límite a kσ esté por encima de ese piso; si el piso no es ~0, la recta por el origen tampoco es la forma funcional correcta.

**S4. Propagar la incertidumbre de la pendiente al límite.**
Calculas `slope_sigma` pero el límite se reporta sin error: `flux_limit = k/slope` debería llevar `σ_limit ≈ k·slope_sigma/slope²`. Sin eso, "límite proyectado = X" aparenta una precisión que no tiene.

**S5. Acotar el alcance: una posición, una línea.**
El límite vigente de Stage 10 todavía consume el producto histórico de Stage06 para una sola posición y Hα. C2 ya cuantifica tres PA y tres longitudes de onda, pero esos factores aún deben incorporarse al QC/figura de Stage 10 y, después, extenderse a un mapa radial.

**S6. Mediana de throughput sobre toda la grilla mezcla regímenes.**
A flujos bajos el throughput medido es ruido/ruido (mal definido); a flujos altos sufre la autosustracción de S2. Mejor: η de los puntos intermedios (S/N recuperado entre ~3 y ~7), o directamente la curva η(flujo).

**S7. Coherencia de tiempos: el t₁ relevante es el de los cubos seleccionados.**
La recuperación de Stage06 se mide sobre el stack de cubos seleccionados (stage04 best), así que el límite actual corresponde a `total_time_selected_s`, no al total. La proyección con multiplicador es consistente, pero el QC debería declarar cuál de los dos tiempos ancla el límite.

### Sugerencias de código

1. Reemplazar el parser FITS manual (~80 líneas) por `astropy.io.fits.getheader(path, 0)` — astropy ya es dependencia de todo el pipeline; el parser propio no maneja tarjetas CONTINUE ni todos los formatos HIERARCH.
2. El patrón `RUN_ID` hardcodeado reaparece ("Keep this synchronized with..."): mismo riesgo #2 del diagnóstico; candidato directo a O3.
3. Nombre de etapa: `stage07_*` colisiona conceptualmente con la familia 07 de líneas de acreción (`Far_07*`). Considerar `stage09_` o un prefijo propio (`proj_`) antes de que se fije en más outputs.
4. Las funciones `safe_float`, `json_clean`, `median_finite` son nuevas duplicaciones → `musepipe/` (O2).

### Implementación (2026-06-11) en `09_integration_time_flux_projection.ipynb`

- **S1**: bloque "Noise scaling test" — reajusta PCA sobre subconjuntos de N cubos (tamaños automáticos o `stage09_noise_scaling_subset_sizes`), en ventana espectral alrededor de Hα (default 6450–6700 Å), mide σ_F en aperturas de control al mismo radio (excluyendo PA de inyección y de referencia) y ajusta α con su incertidumbre. Proyecta el límite con α=0.5 *y* con el α medido (columnas separadas en CSV/QC, ambas curvas en la figura). Verificado contra LkCa_15: σ_F(N=30)=82.2 nativo coincide con el `aperture_line_noise`=79.9 de Stage06; α≈0.69 en el test rápido (1 draw) — el notebook usa 3 draws × 4 tamaños.
- **S2**: ajuste restringido a S/N recuperado ≤ `stage09_fit_recovered_snr_max` (default 6) + ratio de linealidad pendiente_total/pendiente_baja en QC y figura.
- **S3**: fila de flujo cero → `systematic_floor_zero_injection` en QC y `floor_fraction_of_limit` por fila de proyección.
- **S4**: `input_flux_limit_current_sigma_native` = k·σ_slope/slope² propagado también a los límites proyectados.
- **S5**: bloque `scope` en QC (separación/PA/línea de Stage06) + aviso en el suptítulo de la figura.
- **S6**: η de referencia desde puntos con S/N recuperado en `stage09_throughput_snr_range` (default 3–7) + panel η(flujo).
- **S7**: `time_anchor = "selected_cubes"` explícito en QC, CSV y resumen.
- Código: headers vía `astropy.io.fits.getheader` (parser manual eliminado); `MUSE_RUN_ID` por variable de entorno con validación contra `config.json`; outputs renombrados a `stage09_*` (los `stage07_integration_*` viejos quedan en `runs/` y son regenerables/borrables).

**Ejecución real vigente (`LkCa_15`)**: el QC confirma una grilla realista
(`recovery_grid_deterministic=false`), pendiente matched-filter
`0.0020758 +/- 0.0002008` S/N por flujo nativo y piso de inyección cero
`S/N=0.938`. El test profundo obtiene `alpha=0.375 +/- 0.037`, menor que
`0.5`; por tanto, las proyecciones finales usan también el escalado empírico y
no deben resumirse sólo con `sqrt(t)`.

**Hallazgo histórico durante la implementación — importante**: la primera grilla de recuperación de Stage06 era *determinista* (throughput 0.988 idéntico en todos los flujos, dispersión del fit ~10⁻¹⁸). Stage06 medía el delta (run inyectado − run base), de modo que el ruido real se cancelaba: la relación S/N–flujo era exacta por construcción y el límite quedaba anclado únicamente en la σ asumida (`pca_matched_flux_sigma_native`), no en un experimento de detección con ruido. El notebook 09 detecta automáticamente esa configuración mediante `recovery_grid_deterministic`; el estado vigente se describe a continuación.

**Resuelto (2026-06-12)**: Stage06 ahora tiene `stage06_pca_recovery_mode` ∈ {`realistic` (default), `deterministic`}. En modo realista, el flujo se mide directamente sobre el residual del stack inyectado (espectro de apertura menos mediana de continuo; matched filter con resta de continuo por spaxel) y el ruido sale de controles al mismo radio (excluyendo PA de inyección y de referencia) — un experimento de detección genuino con dispersión real en la grilla. Las columnas primarias del CSV (`pca_*`, las que consumen 09/10) reflejan el modo activo; el delta queda siempre como diagnóstico (`delta_*_signal_transfer`). Nomenclatura corregida en plots y QC: "signal transfer" para el delta (el ruido se cancela por construcción — por eso la imagen delta se ve ~0 fuera del objeto y la estrella) y "flux ratio (measured/expected)" en modo realista; la figura de imágenes ahora muestra el residual inyectado *con* ruido junto al delta. Validado con sintéticos: matched filter recupera 99% del flujo inyectado, controles al radio correcto. Las etapas 06, 09 y 10 fueron reejecutadas en `LkCa_15` y sus QC vigentes usan el modo realista.

**Correcciones post-primer-run realista (2026-06-12)**: (1) el matched filter realista ahora trunca el template en los bordes del campo y normaliza con la parte visible (umbral de cobertura 60%) — con el radio de 17.9 px en el campo de 40 px, los 8 controles pasaban de NaN a válidos con cobertura ≥99.9%; (2) en modo realista la σ de referencia de la grilla de inyección se calcula con la *misma* medición del sweep (dispersión control-a-control sobre el residual base), no con la dispersión espectral local — sin esto la grilla quedaba ~10× subcalibrada porque el ruido azimutal entre controles es mucho mayor que el espectral en una posición; (3) stage09 degrada con gracia si un método no tiene puntos válidos (skip con warning en vez de ValueError).

**Paneles granulares + convención de suavizado (2026-06-12)**: las figuras de imágenes de Stage06 ahora muestran cada residual en versión suavizada (`fig_convolve_sigma_px`, gaussiana σ=2 px) **y** sin suavizar (ruido granular pixel a pixel), lado a lado: la figura resumen ganó una tercera fila (3×3) y las figuras showcase 3×3 pasaron a 2×4 (fila superior: residual con ruido suavizado/granular, delta suavizado/granular; el panel de S/N de transferencia bajó a [1,3]). Escalado por panel (`±Nσ` robusto), con claves opcionales `with_noise_raw`/`delta_raw` en `stage06_3x3_image_manual_limits`. **Convención de pipeline**: todo panel de imagen declara en su título si está suavizado y con qué σ ("smoothed: Gaussian sigma=N px" vs "raw pixels (no smoothing)") — aplicado también a las 6 figuras con imágenes de `04_pca_and_cube_selection` y a las 4 de `Far_04b_local_surface_subtraction` (vía `FIG_SMOOTHING_NOTE` en suptitle/título). Los suavizados de `03_fakecont` son algorítmicos (no display) y los notebooks de canal único (diagnóstico Hα de 04b) plotean sin suavizar, por lo que no llevan etiqueta.

## 6. Notebook 10: límites de masa planetaria (2026-06-11)

`10_planet_mass_detection_limits.ipynb` convierte los límites de stage09 en límites físicos: F(Hα) [erg/s/cm²] → L(Hα) = 4πd²F → L_acc (Alcalá 2017 y Aoyama 2021) → M·Ṁ = L_acc·R_p/(0.8G) → masa mínima detectable por Ṁ asumido, con plano de detectabilidad (M_p, Ṁ). Cubre la parte de "límite de acreción" de C5 para la posición de la inyección.

Decisiones y validaciones:

- Calibración de flujo: asume BUNIT estándar MUSE (10⁻²⁰ erg/s/cm²/Å), verificable contra el cubo crudo, y se **valida empíricamente con ancla estelar**: curva de crecimiento del continuo de la estrella (MEDPIX) vs. flujo esperado de su magnitud R. En LkCa_15 el ratio converge 4.9 → 1.3 (r = 1.5 → 15 px), confirmando la calibración a ~30% (halo residual fuera del crop). El check corre en cada ejecución y avisa si el ratio sale de [0.3, 3].
- La ejecución realista vigente reemplaza el valor determinista anterior. Para
  `5 sigma` matched-filter en el tiempo actual: límite `2408.68` nativo,
  `F(Halpha)=3.01e-17 erg/s/cm2`, `L_Halpha=2.33e-8 L_sun` y
  `M*Mdot=2.05e-8 M_Jup2/yr` con Aoyama. Hereda `A_Halpha=0`, una sola
  separación/PA y los sistemáticos medidos por Stage 09/11; no es el antiguo
  límite optimista de la grilla determinista.
- Caveats explícitos en QC: solo restringe planetas *acretantes*; extinción circumplanetaria debilitaría los límites; coeficientes de las escalas L_Hα→L_acc deben verificarse contra los papers antes de publicar; válido solo en la separación/PA de la inyección.
- Parámetros en config (`stage10_*`): distancia, A_Hα, R_p, R_in/R_p, magnitud R de la estrella, grillas de Ṁ y masa.
- **Presupuesto de error (añadido 2026-06-11)**: cada límite M·Ṁ lleva incertidumbre total en dex combinando en cuadratura: dispersión intrínseca de la escala L_Hα→L_acc (0.19/0.30 dex), calibración de flujo del ancla estelar (0.12 dex, `stage10_flux_cal_dex`), σ de la pendiente de stage09 (×a), σ_α en el escenario proyectado con α medido (σ_α·log₁₀(mult), ~0.13 dex), distancia (2σ_d/d, configurable) y radio planetario (σ_R/R, ~0.11 dex). Totales actuales: ±0.26 dex (Alcalá) y ±0.34 dex (Aoyama). Componentes desglosados en CSV (`err_dex_*`) y QC (`error_budget_inputs`); el panel M·Ṁ muestra barras asimétricas en log y el plano de detectabilidad usa las bandas actualizadas.

## 6c. Notebook 03b: verificación de eficacia del fakecont (2026-06-12)

Creado `03b_fakecont_diagnostics.ipynb` ante dudas sobre la eficiencia de Stage03. Solo lee productos de stage02/03 y escribe diagnósticos propios (`stage03b_*` QC/CSV/plots). Seis tests:

- **T1 — Reproducibilidad**: re-corre fakecont con config actual y compara con el stack guardado (detecta stacks de corridas viejas).
- **T2 — Throughput de línea (self-subtraction)**: inyección de gaussianas (3 anchos × ~5 λ incl. Hα y pegada a la costura del notch × 3 posiciones) en stage02, re-corrida del fakecont y recuperación contra **referencia congelada** `Δyellow_ref = ΔL/(med_pix_base·black_base)`. Validado con sintéticos: throughput 0.85→0.99 para σ kernel 25→460 ch, comportamiento esperado. Este factor multiplica al throughput PCA de stage06 en cualquier límite publicado.
- **T3 — Trade-off del kernel**: estructura residual del continuo (anillo) vs déficit de throughput por kernel; barrido σ ∈ {0.25,0.5,1,2,4}×config + la **interpretación FWHM** (σ=97.7 ch) — ojo: `kernel_230_value=230` como σ son ~287 Å (FWHM≈677 Å); si el paper de referencia hablaba de FWHM, el valor actual difiere ×2.35. T2+T3 deciden empíricamente.
- **T4 — Bordes y costura**: el suavizado actúa en índice, cruza el hueco Na-LGS (5780–6050 Å) como si fuera contiguo y usa `mode="nearest"` en extremos; mide perfil residual por canal y cuenta canales sesgados (>5 MAD) a enmascarar río abajo.
- **T5 — Entorno Hα**: σ del residuo en ventana Hα vs ventanas de control (estrella y halo) — exceso = sistemático que 06/09 no modelan.
- **T6 — Elección de BLUE**: purple-median (actual) vs mediana global cruda, mismo subset.

Parámetros `stage03b_*` todos opcionales con default (n_test_cubes=3, kernel_grid auto, contraste 0.5, offsets ±12 px, etc.). Funciones de 03 copiadas verbatim (sincronizar hasta O2).

**Ejecución real confirmada en `LkCa_15` (2026-06-12)**: T1 reproduce
exactamente los cubos 0/16/31; el throughput angosto Halpha con el kernel
configurado es `0.986` (mínimo `0.9855`). Se detectan seis canales sesgados en
el borde rojo y un exceso de ruido Halpha de `2.16x` en la estrella y `2.90x`
en el halo respecto de las ventanas de control. El diagnóstico está completo;
la extracción de sus funciones duplicadas a `musepipe/` sigue pendiente.

## 6d. Notebook 01b: fidelidad del remuestreo de Stage01 (2026-06-13)

Creado `01b_align_resampling_tests.ipynb`, complementario al `01_align_test` existente. El test viejo audita el **centrado** (consistencia peak/gaussian2d/MAOPPY + estabilidad del centro tras alinear); 01b audita las **operaciones de remuestreo** que aquel no toca: ¿el shift subpixel (spline orden 3) y el regrid espectral (interp lineal) preservan fotometría, astrometría sub-pixel y la estadística del ruido? Solo lee productos + fuentes sintéticas con verdad conocida.

- **T1 — Exactitud sub-pixel**: estrella Moffat sintética, shift conocido, centro recuperado por error diferencial (cancela el sesgo del estimador). En smoke-test, RMS diferencial < 0.05 px.
- **T2/T3 — Round-trip y flujo**: shift(+s)+shift(−s) conserva el flujo sintético a ~`-0.02%`; en la white-light real el cambio medido es `-0.54%` y el RMS del round-trip equivale a `0.077` de la dispersión de la imagen.
- **T4 — Corrupción del ruido (hallazgo clave)**: ruido blanco → shift subpixel **reduce la varianza a 0.763 (shift 0.25) y 0.581 (shift 0.5)** e introduce **autocorrelación vecino-a-vecino de +0.136 y +0.264**. Implicación directa: los σ empíricos que miden 06/09/11 *después* de stage01 están sesgados hacia abajo y las sumas de apertura no escalan como √N. Conecta con la nota de C3 ("el spline cúbico de stage01 no es bilineal") — ahora cuantificado.
- **T5 — Regrid espectral**: línea gaussiana de área conocida con offset sub-canal conserva el área mejor que `0.00002%`, pero redondea el pico entre `-1.79%` y `-2.69%` en las pruebas desplazadas (afecta line-spread, no flujo integrado). Bordes → NaN al extrapolar.
- **T6 — Trade-off de modos**: none/integer/subpixel, error astrométrico vs corrupción de ruido; justifica `spatial_shift_mode=subpixel` del config.

Parámetros `stage01b_*` opcionales con default. Funciones `apply_spatial_alignment` y `regrid_cube_spectral_axis` copiadas verbatim de stage01 (sincronizar hasta O2). **Ejecución real confirmada en `LkCa_15` (2026-06-13)**: T1 pasa con peor RMS `0.00092 px`; T2–T6 produjeron el QC descrito arriba. **Sugerencia derivada de T4 para C3/C4**: aplicar un factor de corrección de correlación (o medir σ sobre el cubo ya remuestreado, no sobre ruido independiente) a las sumas de apertura de 06/09.

## 7. Orden sugerido de ejecución (actualizado 2026-06-21)

| Semana | Acción | Por qué primero |
|---|---|---|
| 1 | Integrar los productos C2/C4 en el mapa de límites C5 | Lleva throughput, varianza espacial y umbrales globales al producto científico publicable |
| 2 | Regresión real pequeña automatizada | Protege los cierres C1-C4 frente a cambios futuros |
| 3 | Consolidar O4 | Convierte las variantes low-memory en una ruta operativa reproducible |
| 4+ | C6–C7, O5–O6 y migración restante | Cierra ciencia espectral, duplicados y trazabilidad completa |
