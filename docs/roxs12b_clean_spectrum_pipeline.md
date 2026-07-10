# ROXs12b clean spectrum pipeline

Estado de esta guia: actualizada el 2026-06-17 para el refactor de objetos
lejanos. Las etapas 04b, 06, 07 y 07b tienen implementacion canonica en `musepipe`; los
notebooks `Far_*` correspondientes se conservan como wrappers historicos.

Esta nota documenta la cadena actual para obtener un espectro lo mas limpio posible de `ROXs12b`, con foco en el objeto mas alejado dentro del crop.

## Objetivo

El objetivo cientifico es producir un espectro del objeto que pueda usarse luego con modelos de identificacion espectral. La cadena evita medir solo una linea aislada: primero valida si una senal puntual, como Halpha, es robusta, y despues genera un producto de espectro completo con mascara de longitudes de onda, controles locales, continuo estimado y una metrica tipo SNR.

## Coordenadas de ROXs12b

Para `ROXs12b`, el crop usado tiene tamano `170 x 170` pixeles.

La coordenada entregada del objeto es:

```text
(x, y) = (72, 152)
```

En Python/cubo, la convencion es:

```text
(y, x) = (152, 72)
```

La estrella central se usa como referencia aproximada en:

```text
(y, x) = (85, 85)
```

El otro objeto del campo se deja absorbido en la region central/estrella para esta etapa, porque el objetivo de ciencia es el objeto mas alejado.

## Cadena recomendada

La cadena local recomendada para este caso es:

```text
01_load_align_crop
02_xcorr_stripes
04b_local_surface_subtraction
07_accretion_line_spectra
07b_halpha_robustness_checks
08_full_spectrum_for_modeling
```

Con `input_mode = native_stage02`, Stage 04b lee directamente la salida de
Stage 02. Stage 03 es opcional cuando se selecciona `fakecont_stage03`. Stage
04/PCA no forma parte de la sustraccion local, aunque su QC historico puede
aportar coordenadas. `05_fluxes_and_linewidths` y
`07_accretion_line_spectra` sigue siendo util para inspeccion de lineas,
pero el producto recomendado para modelado posterior sale de Stage 8.

## Seleccion del run y seguridad

Las etapas migradas resuelven el run desde `MUSE_RUN_ID` y luego desde
`active_run.txt`. Para confirmar el run sin escribir productos:

```bash
MUSE_RUN_ID=ROXs12b python -c "from musepipe.config import load_run_config; print(load_run_config().run_id)"
```

Tambien se puede inspeccionar la configuracion efectiva de una etapa:

```bash
MUSE_RUN_ID=ROXs12b python -c "from musepipe.stages import stage04b_config_from_run; print(stage04b_config_from_run())"
```

Las funciones `run_stage04b`, `run_stage07`, `run_stage07b` y `run_stage08` escriben o
reemplazan productos en `runs/<RUN_ID>`. El refactor todavia no se ha ejecutado
sobre los cubos reales para comprobar equivalencia numerica.

## Stage 04b: sustraccion local de superficie

Implementacion canonica:

```text
musepipe/stages/stage04b_local_surface.py
```

Notebook wrapper historico:

```text
Far_04b_local_surface_subtraction.ipynb
```

Para `ROXs12b`, esta etapa fue ajustada para trabajar sobre el objeto en `(y, x) = (152, 72)`.

Cambios importantes:

- El run se resuelve con `MUSE_RUN_ID` o `active_run.txt`.
- El objetivo cientifico se resuelve desde config/QC como `(x, y) = (72, 152)`.
- La entrada por defecto usa `native_stage02`, es decir `stage02_xcorr_cube_stack.fits`.
- La region mala de longitudes de onda se enmascara:

```text
5780-6050 A
```

- La sustraccion local usa un modelo plano con clipping robusto:

```text
local_model_kind = plane
fit_radius_px = 12
mask_radius_px = 3
local_fit_sigma_clip = 3
local_fit_max_iter = 3
```

Ejecucion directa, que escribe productos:

```bash
MUSE_RUN_ID=ROXs12b python -c "from musepipe.stages import run_stage04b; run_stage04b()"
```

Productos principales:

```text
runs/ROXs12b/stages/cube_input_local_object.fits
runs/ROXs12b/stages/cube_residual_local_object.fits
runs/ROXs12b/stages/cube_localmodel_object.fits
runs/ROXs12b/stages/stage04b_local_surface_cube_stack.fits
runs/ROXs12b/stages/stage04b_good_wavelength_mask.npy
runs/ROXs12b/stages/stage04b_bad_wavelength_mask.npy
runs/ROXs12b/stages/stage04b_qc.json
```

Interpretacion:

- `cube_input_local_object.fits` es el cubo de entrada usado para la extraccion local.
- `cube_residual_local_object.fits` es el cubo residual centrado en la sustraccion local del objeto.
- `stage04b_qc.json` guarda coordenadas, modo de entrada, radios, modelo local y mascara espectral.

## Stage 06: inyeccion-recuperacion local

Implementacion canonica:

```text
musepipe/stages/stage06_local_surface_injection.py
```

Notebook wrapper historico:

```text
Far_06_inject_halpha_signal.ipynb
```

Esta etapa valida la sensibilidad de la receta local sin usar PCA. Para
`ROXs12b` resuelve por defecto:

```text
estrella (y, x) = (85, 85)
objeto de referencia (y, x) = (152, 72)
inyeccion opuesta (y, x) = (18, 98)
```

La fuente sintetica usa una PSF y un perfil espectral Halpha normalizados. La
grilla se procesa con los mismos radios, sigma clipping y modelo local de Stage
04b. Reporta por separado:

- `realistic`: flujo en el residual inyectado y ruido de controles al mismo
  radio.
- `delta`: transferencia de senal en `injected - baseline`, donde el ruido se
  cancela por construccion.

Ejecucion directa, que escribe productos nuevos sin reemplazar PCA ni Stage
04b:

```bash
MUSE_RUN_ID=ROXs12b python -c "from musepipe.stages import run_stage06_local; run_stage06_local()"
```

Productos principales:

```text
runs/ROXs12b/tables/stage06_local_surface_halpha_injection_recovery.csv
runs/ROXs12b/tables/stage06_local_surface_halpha_injection_truth_summary.json
runs/ROXs12b/stages/stage06_local_surface_halpha_injection_qc.json
runs/ROXs12b/stages/injections/stage06_local_surface_halpha_signal_template.fits
runs/ROXs12b/stages/injections/stage06_local_surface_recovered_halpha_nominal_diag.fits
runs/ROXs12b/plots/stage06_local_surface_injection/
```

Estado: implementado y validado con cubos sinteticos y sobre la copia aislada
`ROXs12b_short_refactor_validation`. La transferencia delta fue `0.99824`,
pero el punto de inyeccion tenia un baseline de `-3.95 sigma`; transferencia y
detectabilidad realista deben interpretarse por separado.

## Stage 07: espectros de lineas de acrecion

Implementacion canonica:

```text
musepipe/stages/stage07_accretion_lines.py
```

Notebook wrapper historico:

```text
Far_07_accretion_line_spectra.ipynb
```

Stage 07 combina el espectro estelar de Stage 02 con el residual local del
objeto de Stage 04b. Extrae pixel central y caja integrada, calcula metricas
para 24 lineas candidatas y, cuando existe `cube_input_local_object.fits`,
resta la mediana de controles al mismo radio.

La posicion estelar se resuelve desde config/QC. Esto corrige el valor
heredado `(20,20)` de `Far_07`, que no correspondia al cubo `170 x 170` de
ROXs12b; la posicion efectiva es `(85,85)`.

Ejecucion directa, que escribe tabla, QC y plots:

```bash
MUSE_RUN_ID=ROXs12b python -c "from musepipe.stages import run_stage07; run_stage07()"
```

Productos:

```text
runs/ROXs12b/tables/stage07_accretion_line_metrics.csv
runs/ROXs12b/stages/stage07_accretion_line_qc.json
runs/ROXs12b/plots/stage07_accretion_lines/
```

Estado: implementado y validado con cubos sinteticos y sobre la copia aislada
`ROXs12b_short_refactor_validation`. Produjo 23 lineas medibles y 27 figuras;
`He I 5876` fue excluida por la mascara mala. `YSES 2b` necesita coordenadas
Stage 04b validas antes de usar esta etapa.

## Stage 07b: robustez de Halpha

Implementacion canonica:

```text
musepipe/stages/stage07b_halpha_robustness.py
```

Notebook wrapper historico:

```text
Far_07b_halpha_robustness_checks.ipynb
```

Esta etapa no produce el espectro final. Su funcion es verificar si Halpha sobrevive decisiones razonables de extraccion.

Hace lo siguiente:

- Lee `cube_input_local_object.fits`.
- Rehace la sustraccion local alrededor del objeto y de controles al mismo radio.
- Compara varias aperturas:

```text
pixel
box3_sum
circle_r1p5_sum
circle_r2p0_sum
gauss_sig1p0_r3_sum
```

- Compara el objeto contra controles usando metricas empiricas:

```text
object_empirical_z_flux
object_empirical_z_peak
object_flux_percentile_vs_controls
```

Ejecucion directa, que escribe tablas, QC y plots:

```bash
MUSE_RUN_ID=ROXs12b python -c "from musepipe.stages import run_stage07b; run_stage07b()"
```

Productos:

```text
runs/ROXs12b/tables/stage07b_halpha_robustness_summary.csv
runs/ROXs12b/tables/stage07b_halpha_control_metrics.csv
runs/ROXs12b/stages/stage07b_halpha_robustness_qc.json
runs/ROXs12b/plots/stage07b_halpha_robustness/
```

Resultado actual:

Halpha puede alcanzar un `peakSNR` local cercano a `4` en algunas aperturas, pero no aparece como un outlier fuerte respecto de los controles. En particular, el `z_flux` empirico queda cerca de cero para la apertura recomendada `box3_sum`. Por ahora se interpreta como una senal no robusta o dominada por canales/picos locales, no como una linea integrada segura.

## Stage 08: espectro completo para modelado

Notebook wrapper historico:

```text
Far_08_full_spectrum_for_modeling.ipynb
```

Script asociado:

```text
stage08_full_spectrum_for_modeling.py
```

Esta es la etapa recomendada para obtener el espectro completo que luego puede entrar a modelos.

Corre asi desde terminal usando el run activo:

```bash
MUSE_RUN_ID=ROXs12b python stage08_full_spectrum_for_modeling.py
```

El notebook tambien puede ejecutarse directamente y llama a la misma funcion:

```python
from stage08_full_spectrum_for_modeling import run_stage08, stage08_config_from_run

config = stage08_config_from_run()
products = run_stage08(config, show_plots=True)
```

`CONFIG` sigue existiendo como compatibilidad legacy para ROXs12b, pero no es
la interfaz recomendada porque ignora la seleccion del run activo.

La apertura por defecto es:

```text
box3_sum
```

El producto principal para modelos es:

```text
runs/ROXs12b/tables/stage08_full_spectrum_model_input.csv
```

Columnas:

```text
wavelength_A
good_wave_mask
bad_wave_mask
object_local_residual
control_median_local_residual
flux_native
continuum
flux_continuum_sub
control_sigma
snr_like
```

Uso recomendado de columnas:

- `wavelength_A`: eje espectral en Angstrom.
- `good_wave_mask`: usar solo filas con valor `1` para modelado inicial.
- `bad_wave_mask`: marca canales descartados, especialmente `5780-6050 A`.
- `flux_native`: objeto menos mediana de controles locales.
- `continuum`: continuo amplio estimado con mediana movil.
- `flux_continuum_sub`: espectro recomendado para busqueda de lineas/modelos de emision.
- `control_sigma`: dispersion empirica de los controles al mismo radio.
- `snr_like`: `flux_continuum_sub / control_sigma`.

Productos adicionales:

```text
runs/ROXs12b/stages/stage08_full_spectrum_model_input.fits
runs/ROXs12b/tables/stage08_full_spectrum_all_apertures.csv
runs/ROXs12b/stages/stage08_full_spectrum_all_apertures.npz
runs/ROXs12b/stages/stage08_full_spectrum_qc.json
runs/ROXs12b/plots/stage08_full_spectrum/
```

El FITS contiene una tabla binaria `SPECTRUM` con las mismas columnas principales del CSV.

## Plots utiles

Stage 8 genera tres plots de control:

```text
runs/ROXs12b/plots/stage08_full_spectrum/stage08_full_spectrum_default.png
runs/ROXs12b/plots/stage08_full_spectrum/stage08_full_spectrum_snr_like.png
runs/ROXs12b/plots/stage08_full_spectrum/stage08_aperture_comparison_overview.png
```

Lectura rapida:

- `stage08_full_spectrum_default.png`: espectro objeto-controles, continuo y espectro sin continuo.
- `stage08_full_spectrum_snr_like.png`: canales con exceso relativo a la dispersion de controles.
- `stage08_aperture_comparison_overview.png`: estabilidad frente a aperturas alternativas.

## Caveats importantes

1. Stage 8 usa un ajuste local rapido de mascara fija para poder procesar todo el espectro completo de forma eficiente.

2. Stage 04b si usa sigma clipping robusto por canal en la sustraccion local. Stage 8 no repite todo ese clipping para cada control y longitud de onda; usa el mismo modelo local base, pero en modo rapido.

3. La region `5780-6050 A` esta enmascarada. Cualquier modelo que use esa region debe ignorarla o tratarla como datos faltantes.

4. `snr_like` no es una incertidumbre calibrada absoluta. Es una escala empirica basada en controles al mismo radio, util para ranking y deteccion preliminar.

5. En el run corto Halpha es la candidata mas fuerte para la caja integrada:
Stage 07 mide `peak S/N = 4.224` y coincide con Stage 07b. El pixel central es
mucho mas debil y la significancia cambia con la apertura; por eso debe
reportarse junto con la comparacion de controles de 07b y el espectro completo
de Stage 8, no como deteccion aislada.

6. Las 45 pruebas actuales cubren utilidades, arrays sinteticos y contratos de
productos. `ROXs12b_short` y `ROXs12b` pasaron validaciones controladas en
copias aisladas, sin reemplazar los productos originales.

## Siguiente paso sugerido

Antes de ajustar modelos fisicos, conviene corregir el `cfg.run_id` historico
del run de inyeccion y luego ejecutar la cadena tambien sobre:

```text
ROXs12b_HaInject_SNR5
```

Eso permitiria verificar si la receta recupera una senal inyectada conocida y calibrar sensibilidad, throughput y falsos positivos.

Despues de eso, el flujo para modelos seria:

```text
1. Leer stage08_full_spectrum_model_input.csv
2. Filtrar good_wave_mask == 1
3. Usar flux_continuum_sub para lineas o flux_native para modelos con continuo
4. Usar control_sigma/snr_like como peso empirico inicial
5. Comparar contra templates/modelos en el rango MUSE disponible
```
