# Guia de parametros de `00_config`

Esta guia documenta los parametros definidos por
[`00_config.ipynb`](../00_config.ipynb) y guardados en
[`runs/LkCa_15/config/config.json`](../runs/LkCa_15/config/config.json).

El objetivo de `00_config` es actuar como la fuente unica de verdad para el
resto del pipeline: rutas, lista de cubos, decisiones cientificas, geometria de
diagnostico y opciones de procesamiento. Los notebooks posteriores deberian leer
`config.json` en vez de repetir valores a mano.

Fecha de la documentacion: 2026-06-14.

> Nota: este archivo Markdown es una instantanea estatica. Cuando sea necesario
> mostrar valores vivos del run, los notebooks deben leer `config.json` y
> generar Markdown con `display(Markdown(...))`. Si `config.json` no esta
> disponible, la documentacion debe seguir funcionando como guia descriptiva.

## Como Usar Esta Guia

- `Valor actual` describe el run activo `LkCa_15`.
- `Cuando cambiarlo` indica la situacion tipica en que conviene modificar ese
  parametro.
- Las coordenadas espaciales internas del pipeline estan en pixeles, aunque los
  plots pueden mostrar ticks en arcosegundos.
- Los parametros derivados, como `drot_posang_deg` o `xcorr_stripe_angle_deg`,
  se calculan dentro de `00_config` y luego se guardan en `config.json`.

## Flujo General De `00_config`

1. Define identidad del run: `RUN_ID`, `TARGET_NAME`, carpetas de salida y
   lista de cubos.
2. Construye el diccionario `CFG` con los parametros base.
3. Agrega parametros derivados para xcorr, rotacion, paralelizacion y Stage01.
4. Valida consistencia: longitudes por cubo, unidades, radios, ejes y valores
   positivos.
5. Verifica rapidamente que los FITS existen.
6. Guarda `CFG` y metadata en `runs/<RUN_ID>/config/config.json`.

## Variables Del Notebook Que No Son `CFG`

Estas variables existen en `00_config`, pero no todas quedan como claves
directas dentro de `CFG`.

| Variable | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `RUN_ID` | `LkCa_15` | Nombre interno del experimento. Define `runs/LkCa_15/`. | Al iniciar un nuevo objeto, configuracion o version de prueba que deba tener salidas separadas. |
| `PROJECT_ROOT` | `Path(".").resolve()` | Carpeta raiz del proyecto, donde viven los notebooks. | Solo si ejecutas el notebook desde otro directorio. |
| `DATA_DIR` | `/Users/ricardoramirez/Downloads/LkCa15_raw_cubes` | Carpeta donde estan los cubos crudos. | Si cambias de dataset o mueves los FITS. |
| `RUNS_DIR` | `PROJECT_ROOT / "runs"` | Carpeta general para salidas por run. | Casi nunca; solo si quieres otro arbol de salidas. |
| `WORKDIR` | `RUNS_DIR / RUN_ID` | Carpeta de trabajo del run activo. | Se actualiza automaticamente al cambiar `RUN_ID`. |
| `CONFIG_DIR` | `WORKDIR / "config"` | Carpeta donde se guarda `config.json`. | No conviene cambiarlo sin ajustar notebooks posteriores. |
| `STAGE_DIR` | `WORKDIR / "stages"` | FITS, NPY y productos intermedios de cada etapa. | No conviene cambiarlo sin ajustar notebooks posteriores. |
| `PLOT_DIR` | `WORKDIR / "plots"` | Figuras de diagnostico. | Si quieres separar plots fuera del run. |
| `TABLE_DIR` | `WORKDIR / "tables"` | CSVs y tablas de diagnostico. | Si quieres separar tablas fuera del run. |
| `LOG_DIR` | `WORKDIR / "logs"` | Carpeta prevista para logs. | Si agregas logging formal. |
| `CUBE_FILES` | 32 archivos FITS | Lista explicita de cubos crudos MUSE. El orden importa porque varias etapas reportan indices de cubo. | Al agregar/quitar exposiciones o cambiar de objeto. |
| `TARGET_NAME` | `LkCa_15` | Nombre fisico del objeto. Se guarda en FITS/QC. | Al cambiar de objeto. |
| `DATA_EXT` | `1` | Extension FITS que contiene el cubo de ciencia. En estos datos, el header primario esta en 0 y los datos en 1. | Si otro producto FITS guarda el cubo en otra extension. |
| `STAGE01_PROFILE` | `maoppy_refined` | Perfil que define centering, shift espacial y grilla espectral para Stage01. | Para comparar un flujo rapido/interpolacion minima contra una reduccion mas fina. |

## Parametros Base Del Dataset

| Clave `CFG` | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `target_name` | `LkCa_15` | Nombre del objeto observado. Se copia a headers, QC y reportes. | Al procesar otro objetivo. |
| `run_id` | `LkCa_15` | Identificador del run. Define la carpeta `runs/<run_id>`. | Al crear una version independiente de procesamiento. |
| `data_ext` | `1` | Extension FITS usada para leer los cubos de ciencia. | Si el formato FITS del dataset cambia. |
| `cube_files` | 32 rutas | Lista completa de cubos crudos. | Al seleccionar otro conjunto de exposiciones. |
| `crop_npix` | `50` | Tamano del recorte espacial alrededor de la estrella: `50 x 50` pixeles. | Si necesitas incluir mas campo, objetos externos o diagnosticos de halo. Al cambiarlo, hay que recalcular productos desde Stage01. |

## Consistencia Del Crop Y Productos Guardados

Cambiar `crop_npix` no modifica automaticamente los FITS, NPY o QC ya escritos
en `runs/<RUN_ID>/stages`. Por ejemplo, si `config.json` dice `crop_npix=50`
pero `stage03_fakecont_cube_stack.fits` todavia fue creado con `40 x 40`, el
pipeline estaria mezclando coordenadas y centros incompatibles.

Para evitarlo, varios notebooks importan `muse_crop_checks.py` y comparan la
forma espacial de los productos cargados contra `CFG["crop_npix"]`. Si detectan
una diferencia, imprimen un aviso como:

```text
WARNING: Stage03 cube stack spatial shape is (40, 40), but current
CFG['crop_npix'] expects (50, 50). This product may come from an older crop.
```

En etapas donde una coordenada vieja seria peligrosa, el chequeo es estricto y
detiene la celda. Esto ocurre, por ejemplo, cuando `04c_fakecont` intenta usar
centros Moffat medidos con otro crop, cuando `06_inject_halpha_signal` lee la
posicion de referencia desde un `stage04_qc` antiguo, o cuando `05` usa imagenes
Stage04 para extraer aperturas.

## Rango Espectral Y Lineas

| Clave `CFG` | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `drop_wave_min_A` | `5780.0` | Inicio del rango contaminado por Na-LGS que se excluye. Unidad: Angstrom. | Si el rango contaminado cambia o si quieres diagnosticarlo. |
| `drop_wave_max_A` | `6050.0` | Fin del rango Na-LGS excluido. Unidad: Angstrom. | Igual que `drop_wave_min_A`. |
| `halpha_A` | `6562.8` | Longitud de onda de referencia de H alpha. Se usa para excluir lineas fuertes y centrar diagnosticos. | Si analizas otra linea o si quieres ajustar un reposo distinto. |
| `ha_channels_A` | `[6560.96, 6562.21, 6563.46]` | Canales nominales usados para construir imagen H alpha en Stage04. | Si el objeto tiene corrimiento radial o si quieres usar otros canales. |
| `hb_channels_A` | `[4860.96]` | Canal nominal usado para H beta. | Si quieres usar mas canales o ajustar por corrimiento. |
| `line_exclude_halfwidth_A` | `2.0` | Semiancho, en Angstrom, para excluir lineas fuertes durante xcorr. | Si xcorr se contamina por senales de linea o si quieres proteger ventanas mas anchas. |

## Cross-Correlation Y Correccion Por Stripes

| Clave `CFG` | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `xcorr_nstripes` | `6` | Numero de franjas por cubo para estimar/corregir shifts espectrales. | Si las variaciones espaciales son mas finas o mas suaves. |
| `xcorr_wmin_A` | `4650.0` | Longitud minima usada para xcorr. | Si quieres restringir xcorr a una zona espectral mas estable. |
| `xcorr_wmax_A` | `9300.0` | Longitud maxima usada para xcorr. | Igual que `xcorr_wmin_A`. |
| `xcorr_max_lag_ch` | `6` | Maximo desplazamiento permitido en canales espectrales. | Si esperas shifts mas grandes o quieres evitar soluciones espurias. |
| `xcorr_exclude_strong_line` | `True` | Excluye lineas fuertes, como H alpha, de la correlacion. | Normalmente debe quedar `True` si buscas senales de linea. |
| `xcorr_exclude_nalgs` | `True` | Excluye la region Na-LGS definida por `nalgs_wmin_A` y `nalgs_wmax_A`. | Para diagnosticar el impacto de esa region o si no existe contaminacion Na-LGS. |
| `nalgs_wmin_A` | `5780.0` | Inicio del rango Na-LGS excluido en xcorr. | Si cambia el rango instrumental contaminado. |
| `nalgs_wmax_A` | `6050.0` | Fin del rango Na-LGS excluido en xcorr. | Igual que `nalgs_wmin_A`. |
| `xcorr_shift_estimator` | `spaxel_map_region_median` | Metodo para resumir shifts por region. `stripe_median_spectrum` combina espectros; `spaxel_map_region_median` estima un mapa de dz y resume por franja. | Cambiarlo para comparar robustez o velocidad. |
| `xcorr_save_spaxel_shiftmaps` | `True` | Guarda mapas spaxel-wise de shifts cuando estan disponibles. | Apagar si quieres ahorrar disco. |
| `xcorr_n_jobs` | `8` | Numero de workers para partes paralelizables de xcorr. | Ajustar segun CPU/RAM. |
| `xcorr_diag_n_jobs` | `8` | Workers para diagnosticos de xcorr. | Igual que `xcorr_n_jobs`. |

## Rotacion DROT Y Geometria De Stripes

El header `HIERARCH ESO INS DROT POSANG` esta en la extension 0, aunque los
datos del cubo estan en la extension 1. `00_config` lee ese angulo y aplica un
offset de `90 deg`, porque se observo que la orientacion aplicada necesitaba ese
desfase.

Importante: el angulo aplicado se reduce modulo `180 deg` porque define el eje
de split/proyeccion. Sin embargo, los rangos manuales y los grupos de referencia
se separan por `DROT POSANG` completo modulo `360 deg`, ya que dos cubos con el
mismo split aplicado, por ejemplo `DROT=0` y `DROT=180`, pueden mostrar patrones
de stripes distintos.

| Clave `CFG` | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `drot_posang_header_ext` | `0` | Extension FITS donde se busca DROT POSANG. | Si otro dataset guarda DROT en otra extension. |
| `drot_posang_header_keys` | `["HIERARCH ESO INS DROT POSANG", "ESO INS DROT POSANG"]` | Alias aceptados para leer la rotacion instrumental. | Si cambia el nombre de keyword en el header. |
| `drot_posang_deg` | 32 valores | Rotacion leida por cubo. Actual: grupos en `0, 45, 90, 135, 180, 225, 270, 315 deg`. | Es derivado; no se edita normalmente. |
| `drot_posang_header_keys_found` | 32 entradas | Keyword realmente encontrada en cada cubo. | Es diagnostico. |
| `xcorr_use_drot_posang_for_stripes` | `True` | Usa DROT POSANG para definir la geometria por cubo. | Apagar solo para forzar una geometria generica/manual. |
| `xcorr_stripe_angle_offset_deg` | `90.0` | Offset aplicado: `angle = (DROT + 90) % 180`. | Si se verifica otro desfase entre header y geometria real. |
| `xcorr_stripe_angle_deg` | 32 valores | Angulo de split/proyeccion usado para stripes. `0` = stripes verticales, `90` = horizontales, otros = oblicuos. | Es derivado de DROT y offset; editar solo si fuerzas geometria manual. |
| `xcorr_stripe_angle_convention` | `split_axis` | Declara que el angulo representa el eje de split/proyeccion para dividir franjas. | Mantener salvo que se cambie la convencion del codigo de Stage02. |
| `xcorr_stripe_orientation` | 32 etiquetas | Etiquetas simplificadas: `vertical`, `horizontal` o `angled`. Sirven para compatibilidad y diagnostico. | Es derivado; no editar normalmente. |
| `xcorr_stripe_geometry_grouping` | `drot_posang_360` | Indica que las plantillas manuales se eligen por DROT completo, no solo por el split modulo 180. | Mantener para LkCa 15; cambiar solo si el patron realmente se repite cada 180 deg. |
| `xcorr_stripe_group_key` | 32 valores | Llave usada por Stage02 para seleccionar referencias y correr la correccion por grupos, por ejemplo `drot_180_split_90`. | Es derivado; revisarlo si se sospecha mezcla de rotaciones. |
| `xcorr_stripe_base_ranges_by_drot_posang` | 8 plantillas | Rangos editables por rotacion instrumental: `0, 45, ..., 315 deg`. Aqui se ajustan los cortes cuando el patron cambia al volver al mismo split. | Editar con ayuda de `02c_define_xcorr_stripe_ranges_template`. |
| `xcorr_stripe_base_ranges_by_cube` | 32 listas | Plantilla base ya expandida a cada cubo segun su DROT. | Es derivado de `xcorr_stripe_base_ranges_by_drot_posang`. |
| `xcorr_stripe_ranges_per_cube` | 32 listas | Rangos finales usados por Stage02. En stripes oblicuos, el ultimo rango puede extenderse al span proyectado completo para cubrir las esquinas. | Revisar si `02b` o `02c` muestran cobertura incompleta o limites incorrectos. |
| `xcorr_rotation_k90` | `None` | Clave antigua para rotaciones multiplos de 90 grados. Ahora esta deprecada. | No usar salvo compatibilidad con notebooks viejos. |

Plantillas base actuales para `crop_npix = 50`:

```python
0 deg:   [[0, 14], [14, 18], [18, 30], [30, 31], [31, 40], [40, 50]]
45 deg:  [[0, 14], [14, 15], [15, 27], [27, 28], [28, 40], [40, 50]]
90 deg:  [[0, 11], [11, 20], [20, 26], [26, 33], [33, 43], [43, 50]]
135 deg: [[0, 14], [14, 15], [15, 27], [27, 28], [28, 40], [40, 50]]
180 deg: [[0, 8],  [8, 19],  [19, 26], [26, 33], [33, 34], [34, 50]]
225 deg: [[0, 14], [14, 15], [15, 27], [27, 28], [28, 40], [40, 50]]
270 deg: [[0, 11], [11, 19], [19, 26], [26, 32], [32, 43], [43, 50]]
315 deg: [[0, 14], [14, 15], [15, 27], [27, 28], [28, 40], [40, 50]]
```

Estas plantillas fueron desplazadas desde las pruebas con crop `40 x 40`. Deben
revisarse visualmente con `02c_define_xcorr_stripe_ranges_template.ipynb` cuando
cambie el crop, el centrado de Stage01 o la estructura de stripes.

## Fake Continuum

| Clave `CFG` | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `kernel_230_is_sigma` | `True` | Interpreta `kernel_230_value` como sigma del suavizado/continuo, no necesariamente como ancho literal. | Si el codigo de fake continuum cambia la interpretacion del kernel. |
| `kernel_230_value` | `230` | Escala espectral del fake continuum, en canales bajo la convencion actual. | Si el continuo queda sub/sobre suavizado. |
| `stage03_n_jobs` | `8` | Workers para Stage03 fake continuum. | Ajustar segun CPU/RAM. |

## Productos De Imagen Y Geometria De Diagnostico

| Clave `CFG` | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `pixel_scale_mas` | `25.0` | Escala espacial MUSE NFM en milisegundos de arco por pixel. Equivale a `0.025 arcsec/pix`. | Si el instrumento/modo cambia. |
| `fig_convolve_sigma_px` | `2.0` | Sigma del suavizado gaussiano usado para mapas/figuras de linea. Unidad: pixeles. | Para plots mas/menos suavizados. |
| `apply_central_mask` | `True` | Enmascara visualmente el centro estelar en varios plots. | Apagar si quieres inspeccionar el centro. |
| `central_mask_radius_px` | `3` | Radio del mask central visual, en pixeles. | Si el nucleo saturado/artefactado es mas grande o mas pequeno. |
| `airy_ring_reference_radii_px` | `[9.0, 15.0]` | Radios de referencia para los anillos/annulus de diagnostico tipo Airy. Unidad: pixeles. | Ajustar si se redefine el annulus de comparacion con residuos PCA. |
| `reference_geometry_center_source` | `stage04c_fakecont_qc` | Fuente para el centro estelar usado por overlays. Opciones: `crop_center`, `stage04c_qc`, `stage04c_fakecont_qc`. | Cambiar si quieres centrar overlays en el centro geometrico o en otro QC. |
| `reference_geometry_center_yx` | `None` | Override manual del centro `[y, x]`. `None` usa `reference_geometry_center_source`. | Usar si quieres forzar un centro medido externo. |
| `reference_geometry_east_is_left` | `True` | Declara que, con el WCS actual, el eje x de imagen aumenta hacia oeste y este queda a la izquierda. Afecta conversion PA -> angulo de Matplotlib. | Cambiar si otro dataset tiene eje x orientado al este. |
| `reference_ring_color` | `black` | Color de los circulos/anillos de referencia en plots. | Ajuste visual. |

## Ejes Espaciales En Plots

Estos parametros cambian solo los ticks y labels de los plots; no cambian las
coordenadas internas, aperturas, circulos o elipses, que siguen en pixeles.

| Clave `CFG` | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `plot_spatial_axis_units` | `arcsec` | Unidad mostrada en ejes espaciales. Opciones: `pix`, `arcsec`, `deg`. | Usar `arcsec` para interpretacion astronomica; `pix` para debugging de aperturas. |
| `plot_spatial_axis_center_source` | `reference_geometry` | Centro cero de los ejes. `reference_geometry` usa el mismo centro de overlays; `crop_center` usa el centro geometrico del recorte. | Cambiar a `crop_center` para comparar contra pixeles puros del recorte. |
| `plot_spatial_axis_arcsec_decimals` | `2` | Numero de decimales al mostrar arcsec. | Subir a 3 si necesitas ver desplazamientos de pocos pixeles. |
| `plot_spatial_axis_arcsec_tick_step` | `0.25` | Separacion fija entre ticks cuando los ejes estan en arcsec. Con `0.25`, el centro queda marcado como `0.00` y los ticks son redondos. | Cambiar si quieres ticks mas densos (`0.10`) o mas espaciados (`0.50`). Usar `None` para volver a ticks automaticos. |
| `plot_spatial_axis_deg_decimals` | `5` | Numero de decimales si se muestran grados. | Normalmente no se usa para mapas tan pequenos; arcsec es mas legible. |
| `plot_spatial_axis_show_zero_lines` | `True` | Dibuja lineas punteadas finas en `x=0` e `y=0` para marcar el centro usado por los ejes. | Apagar si las lineas interfieren con una figura final. |

Conversion actual:

```text
25 mas/pix = 0.025 arcsec/pix = 6.944444e-06 deg/pix
10 pix = 0.25 arcsec
```

## Referencias De Disco LkCa 15

Estas referencias se dibujan como elipses proyectadas en plots de Stage04. Son
overlays de diagnostico: no alteran datos ni seleccion de cubos.

| Clave `CFG` | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `disk_reference_enabled` | `True` | Activa/desactiva globalmente todos los overlays de disco. | Poner `False` si quieres plots sin elipses de disco. |
| `disk_reference_color` | `black` | Color por defecto para elipses de disco. | Ajuste visual. |
| `disk_reference_source` | Gardner et al. 2025 | Fuente bibliografica de las referencias B43/B69. | Si actualizas parametros desde otra publicacion. |
| `disk_reference_ellipses` | B43 y B69 activas | Lista de elipses opcionales. Cada entrada tiene `label`, `enabled`, `semimajor_mas`, `inclination_deg`, `pa_deg`, `color`, `line_style`, `line_width`, `source`. | Para agregar/quitar anillos del disco. |
| `disk_reference_label` | `LkCa 15 B43 dust ring` | Clave antigua para una sola referencia, mantenida por compatibilidad. | Preferir `disk_reference_ellipses` para cambios nuevos. |
| `disk_reference_semimajor_mas` | `267.6` | Semieje mayor B43 en mas, clave antigua. | Preferir la entrada B43 dentro de `disk_reference_ellipses`. |
| `disk_reference_inclination_deg` | `49.1` | Inclinacion B43, clave antigua. | Preferir `disk_reference_ellipses`. |
| `disk_reference_pa_deg` | `63.0` | PA B43, clave antigua. | Preferir `disk_reference_ellipses`. |

Entradas activas actuales:

| Label | Enabled | Semieje mayor | Inclinacion | PA | Estilo |
|---|---:|---:|---:|---:|---|
| `LkCa 15 B43 dust ring` | `True` | `267.6 mas` | `49.1 deg` | `63.0 deg` | negro `-.`, ancho `1.5` |
| `LkCa 15 B69 dust ring` | `True` | `434.2 mas` | `50.5 deg` | `61.7 deg` | negro solido, ancho `1.3` |

Conversion a pixeles usada por Stage04:

```python
semimajor_px = semimajor_mas / pixel_scale_mas
semiminor_px = semimajor_px * cos(inclination_deg)
matplotlib_angle = 90 + PA   # porque east_is_left=True
```

Con `pixel_scale_mas = 25`:

```text
B43: a=10.704 px, b=7.008 px, angle=153.0 deg
B69: a=17.368 px, b=11.047 px, angle=151.7 deg
```

## Aperturas Y Seleccion De Cubos

| Clave `CFG` | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `box_aperture_size_px` | `3` | Tamano por defecto de aperturas cuadradas, normalmente `3 x 3`. | Si el seeing/PSF o el objetivo requiere aperturas mas grandes. |
| `stage04_n_best_cubes` | `30` | Numero de cubos seleccionados como mejores en Stage04. Con 32 cubos, se usan los mejores 30. | Ajustar al numero de exposiciones utiles. |
| `stage04_n_reject_cubes` | `2` | Fallback: numero de cubos rechazados si `stage04_n_best_cubes` es `None`. | Usar si prefieres definir por rechazo en vez de seleccion directa. |

## Stage01: Centrado, Alineamiento Y Grilla Espectral

| Clave `CFG` | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `stage01_profile` | `maoppy_refined` | Perfil activo de Stage01. `maoppy_refined` usa centro MAOPPY, shifts subpixel y grilla comun. | Para pruebas rapidas, cambiar a `native_nointerp`. |
| `stage01_initial_crop_npix` | `160` | Recorte preliminar `160 x 160` usado solo para calcular la imagen white-light y el centro estelar antes del crop final. `None`/`0` desactiva este atajo y usa campo completo. | Cambiar si el centro cae fuera del recorte inicial o si necesitas depurar contra el campo completo. |
| `stage01_initial_crop_edge_guard_pix` | `12` | Margen de seguridad: si el centro estimado cae a menos de este numero de pixeles del borde del recorte preliminar, Stage01 recalcula ese cubo con campo completo. | Subirlo si sospechas que el recorte inicial esta demasiado justo; bajarlo si quieres evitar fallbacks. |
| `centering_method` | `maoppy` | Metodo de centrado estelar. | Usar `peak` si MAOPPY falla o para una prueba rapida. |
| `spatial_shift_mode` | `subpixel` | Forma de aplicar shifts espaciales. | `integer` evita interpolacion; `subpixel` alinea mejor. |
| `spectral_grid_mode` | `common_grid` | Grilla espectral usada al cargar/alinear cubos. | Cambiar si quieres conservar grillas nativas. |
| `maoppy_stamp_half_size` | `10` | Semitamano del stamp alrededor de la estrella para el ajuste MAOPPY. | Si el ajuste se queda sin contexto o incluye demasiado fondo. |
| `maoppy_max_nfev` | `120` | Numero maximo de evaluaciones del ajuste MAOPPY. | Subir si no converge; bajar para velocidad. |

## Stage06: Inyeccion Y Recuperacion PCA De H Alpha

Stage06 inyecta una senal H alpha artificial en los cubos fake-continuum y la
recupera despues de repetir la PCA. Esto mide throughput, sensibilidad y
consistencia de la deteccion cerca de la estrella.

| Clave `CFG` | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `stage06_pca_nominal_snr` | `3.0` | S/N nominal de la inyeccion guardada como producto principal. | Para guardar una inyeccion mas debil/fuerte como caso de inspeccion. |
| `stage06_pca_injection_snr_reference` | `pca_aperture_3x3` | Define que escala se interpreta como `1 sigma` para convertir S/N de entrada a flujo inyectado. | Cambiar a `pca_matched_filter` o `pre_pca_matched_filter` para comparar definiciones de ruido. |
| `stage06_reference_object_yx` | `None` | Coordenada manual `[y, x]` del objeto usado como referencia de separacion. `None` usa `stage04_qc`. | Usar solo si quieres forzar una posicion manual en el crop actual. |
| `stage06_reference_object_qc_key` | `c` | Llave dentro de `stage04_qc["detected_peaks"]` usada cuando `stage06_reference_object_yx=None`. | Cambiar a `b` u otra llave si el objeto de referencia no es `c`. |
| `stage06_pca_recovery_mode` | `realistic` | `realistic` mide flujo/ruido sobre residuales con controles al mismo radio; `deterministic` usa solo la transferencia de senal `delta`. | Usar `deterministic` para estudiar throughput puro; `realistic` para detectabilidad. |
| `stage06_3x3_image_scale_mode` | `sigma` | Modo de escala para los dos primeros plots 3x3. Opciones: `sigma`, `percentile`, `manual`. | Cambiar si necesitas afinar visualmente el ruido de continuo. |
| `stage06_3x3_image_scale_sigma` | `3.0` | Limite visual en multiplos de sigma cuando el modo es `sigma`. | Ajustar para ver detalles mas debiles/fuertes. |
| `stage06_3x3_image_scale_percent` | `99.0` | Percentil usado cuando el modo es `percentile`. | Ajustar si hay outliers o si la imagen queda saturada. |
| `stage06_3x3_image_manual_limits` | `{"with_noise": None, "delta": [-1, 1]}` | Limites manuales para los plots 3x3. | Usar cuando quieres comparar plots con la misma escala fija. |

### Stage 06 local-surface para objetos lejanos

Estas claves son opcionales. Si no existen, la etapa hereda la geometria y el
modelo local desde Stage 04b y usa los defaults indicados.

| Clave `CFG` | Default | Explicacion |
|---|---:|---|
| `stage06_local_injection_yx` | `None` | Posicion manual `[y,x]`; si es `None`, se calcula desde el objeto de referencia. |
| `stage06_local_reference_object_yx` | objetivo de Stage 04b | Objeto cuya separacion se replica. |
| `stage06_local_star_yx` | centro/config Stage 07b | Centro respecto del cual se conserva el radio. |
| `stage06_local_match_reference_separation` | `true` | Coloca la inyeccion al mismo radio que el objeto real. |
| `stage06_local_injection_pa_offset_deg` | `180` | Offset angular respecto del objeto de referencia. |
| `stage06_local_injection_snr_grid` | `[0,0.1,0.5,1,2,3,5]` | Grilla de S/N inyectado. |
| `stage06_local_nominal_snr` | `3` | Punto guardado en el FITS diagnostico nominal. |
| `stage06_local_injection_snr_reference` | `local_surface_matched_filter` | Escala de ruido usada para convertir S/N a flujo. |
| `stage06_local_recovery_mode` | `realistic` | `realistic` usa controles; `deterministic` usa el delta de transferencia. |
| `stage06_local_n_control_angles` | `12` | Numero de angulos candidatos para controles al mismo radio. |
| `stage06_local_control_exclude_pa_deg` | `30` | Exclusion angular alrededor de inyeccion y objeto real. |
| `stage06_local_diagnostic_halfwidth_A` | `75` | Semiancho minimo de la ventana diagnostica alrededor de Halpha. |
| `stage06_local_save_diagnostic_fits` | `true` | Guarda baseline, inyeccion nominal y delta compactos. |

### Stage 07: espectros de lineas de acrecion

Estas claves son opcionales. Las coordenadas y el modelo local se heredan de
Stage 04b cuando no se especifican.

| Clave `CFG` | Default | Explicacion |
|---|---:|---|
| `stage07_integrated_box_size` | `3` | Lado impar de la caja integrada para estrella y objeto. |
| `stage07_star_spectrum_yx` | centro/config | Posicion `[y,x]` para extraer el espectro estelar. |
| `stage07_use_local_control_apertures` | `true` | Activa controles con la misma separacion estrella-objeto. |
| `stage07_control_apertures` | `8` | Numero de angulos candidatos para controles. |
| `stage07_control_exclude_angle_deg` | `25` | Exclusion angular alrededor del objeto real. |
| `stage07_full_continuum_filter_A` | `80` | Ancho del filtro de mediana para vistas de espectro completo. |
| `stage07_line_window_half_width_A` | `18` | Semiancho mostrado en cada ventana de linea. |
| `stage07_line_integration_half_width_A` | `2.5` | Semiancho integrado para flujo y pico. |
| `stage07_line_continuum_inner_A` | `6` | Borde interior de las bandas de continuo. |
| `stage07_line_continuum_outer_A` | `16` | Borde exterior de las bandas de continuo. |
| `stage07_exclude_bad_line_candidates` | `true` | Excluye lineas centrales dentro de rangos espectrales malos de 04b. |
| `stage07_accretion_lines` | lista canonica de 24 lineas | Reemplazo opcional del catalogo de lineas. |

## Paralelizacion

| Clave `CFG` | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `xcorr_n_jobs` | `8` | Workers para xcorr. | Reducir si falta RAM; subir si hay CPU disponible. |
| `stage03_n_jobs` | `8` | Workers para fake continuum. | Igual que arriba. |
| `xcorr_diag_n_jobs` | `8` | Workers para diagnosticos xcorr. | Igual que arriba. |
| `stage04b_n_jobs` | `4` | Workers para Stage04b. Se limita a 4 para controlar memoria. | Subir con cuidado si la maquina lo soporta. |

## Opciones De Debug Y Escritura

| Clave `CFG` | Valor actual | Explicacion | Cuando cambiarlo |
|---|---:|---|---|
| `display_cube_for_plots` | `-1` | Cubo mostrado en algunos diagnosticos. `-1` suele significar seleccion automatica o ultimo cubo, dependiendo del notebook. | Para forzar inspeccion de un cubo especifico. |
| `save_intermediate_plots` | `True` | Guarda figuras intermedias. | Apagar para correr mas limpio/rapido. |
| `overwrite_existing_stage_files` | `True` | Permite sobrescribir productos previos. | Poner `False` para proteger salidas existentes. |

## Metadata Guardada Por `00_config`

Ademas de `config`, el JSON guarda `meta`:

| Campo `meta` | Explicacion |
|---|---|
| `created_utc` | Momento UTC en que se guardo el config. |
| `python_version` | Version de Python usada al ejecutar `00_config`. |
| `platform` | Plataforma/sistema operativo. |
| `project_root` | Ruta absoluta del proyecto. |
| `data_dir` | Ruta absoluta a los datos crudos. |
| `workdir` | Ruta absoluta del run activo. |

## Validaciones Importantes

`00_config` falla temprano si detecta inconsistencias:

- `cube_files` no puede estar vacio.
- `crop_npix` debe ser positivo.
- `drop_wave_min_A < drop_wave_max_A`.
- `kernel_230_value` debe ser positivo.
- `stage04_n_best_cubes` debe ser positivo si no es `None`.
- `stage04_n_reject_cubes` debe ser `>= 0`.
- `airy_ring_reference_radii_px` debe tener dos radios positivos y ordenados.
- `plot_spatial_axis_units` debe ser `pix`, `arcsec` o `deg`.
- `plot_spatial_axis_arcsec_tick_step` debe ser `None` o positivo.
- Cada elipse en `disk_reference_ellipses` debe tener semieje positivo,
  inclinacion `0 <= i < 90 deg` y PA finito.
- Los parametros por cubo, como orientaciones o rangos, deben tener longitud 1
  o longitud igual al numero de cubos.
- Los notebooks posteriores comparan `CFG["crop_npix"]` contra las formas
  guardadas en FITS/QC y avisan si un producto parece venir de un crop antiguo.

## Checklist Rapido Para Cambios Comunes

Cambiar de objeto:

1. Editar `RUN_ID`.
2. Editar `DATA_DIR`.
3. Editar `CUBE_FILES`.
4. Editar `TARGET_NAME`.
5. Revisar `disk_reference_ellipses` o apagar `disk_reference_enabled`.

Cambiar el recorte:

1. Editar `crop_npix`.
2. Revisar `xcorr_stripe_base_ranges_by_drot_posang`; para LkCa 15, los rangos
   actuales estan definidos para `crop_npix=50`.
3. Rerunear desde `01_load_align_crop.ipynb`; los productos Stage01/02/03/04 y
   QC anteriores no son compatibles con el nuevo crop.
4. Rerunear `04c_airy_ring_moffat_diagnostics.ipynb` y luego
   `04c_fakecont_airy_ring_moffat_diagnostics.ipynb` si se usan centros Moffat
   fijos.
5. Revisar manual peaks/aperturas en notebooks posteriores.
6. Revisar `reference_geometry_center_source`.

Cambiar diagnosticos de anillos:

1. Editar `airy_ring_reference_radii_px`.
2. Mantener en pixeles.
3. Rerunear notebooks de diagnostico que dependen de esos radios.

Cambiar ejes de plots:

1. Usar `plot_spatial_axis_units = "arcsec"` para ciencia.
2. Usar `"pix"` para debugging.
3. Usar `"deg"` solo si se requiere comparacion directa en grados.

Cambiar seleccion Stage04:

1. Editar `stage04_n_best_cubes`.
2. Si lo dejas `None`, se usa `N - stage04_n_reject_cubes`.
3. Verificar que los titulos de plots reporten `ncomp` y seleccion.
