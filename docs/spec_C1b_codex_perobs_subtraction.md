# Spec C1b — resta de la primaria por observación y combinado posterior

**Estado:** contrato de la etapa `C1b` (`musepipe/stages/stage_e01b_perobs_subtract.py`).
Complementa la spec C1 (alcance `psf_scope`) y sustituye, para la variante `psfsub`, la resta
que C3 hacía sobre el cubo combinado.

## 1 · Por qué

La cadena restaba el halo de la primaria sobre el cubo **combinado**, lo que obliga a describir
con un solo modelo la mezcla de 29–30 exposiciones con seeing y calidad de AO distintas. Con la
PSF ya ajustada por observación (C1 con `psf_scope=per_observation`), la resta puede hacerse
**donde el modelo vale** —en su propia exposición— y la combinación viene después.

El orden importa por dos motivos distintos:

1. **El modelo.** Ninguna Psfao ni Moffat puede tener a la vez el núcleo de la mejor noche y el
   halo de la peor; restar con un modelo ajustado a la media deja residuo estructurado.
2. **El recorte.** El combinado canónico usa `sigclip`, que **no** es lineal. Restar antes o
   después no da exactamente lo mismo, y la diferencia es medible (§5).

## 2 · Entradas

| entrada | de dónde | qué aporta |
|---|---|---|
| `stages/psf_model_mixture.json` (`form="mixture"`) | C1 (`psf_scope=per_observation`) | un documento de PSF por exposición, con su peso |
| `stages/observation_plan.json` | C1 | qué exposiciones, con qué ventana, desplazamiento y peso |
| `stream_combine_plan.json` | A1 | la geometría original del combinado (vía `musepipe.observations`) |
| `stages/stage01c_qc.json` | B3 | posición del compañero y de la fuente de campo |
| los cubos por exposición | `perexp_cubes` / `perexp_dir` o el propio plan | el dato |

Si lo que encuentra no es una mezcla, la etapa **falla**: restar un modelo del combinado ya lo hace
C3, y hacerlo aquí en silencio sería entregar dos veces la misma cosa con nombres distintos. C1
escribe la mezcla siempre en `psf_model_mixture.json`, aunque además sea lo que entrega como
`psf_model.json`, para que C1b no dependa de ese knob.

## 3 · Procedimiento

1. Se resuelve la vista por observación (`musepipe.observations.resolve_observation_plan`), que
   valida que todas las exposiciones del plan existen y re-mide el centroide de las que hubo que
   re-resolver.
2. Se recorre el combinado por **trozos de λ** con `stream_combine.combine_streaming`, y en cada
   trozo de cada exposición —ya recortado y alineado con la misma convención que produjo el cubo
   combinado— se aplica el gancho `transform`:
   * se evalúa **su** modelo de PSF, centrado en el centro del array (donde el alineado deja la
     primaria);
   * se ajusta por canal la amplitud y el fondo por mínimos cuadrados pesados por `STAT`
     (`extraction.optimal.fit_primary_psf_amplitudes`, la misma función que usa C3);
   * se resta `amp·PSF`. **El fondo ajustado no se resta**, igual que en C3: lo que se quita es el
     halo de la primaria, no el cielo.
3. El acumulador del combinado sigue igual: pesos del plan, `STAT` propagado, `sigclip` si el plan
   lo declara. **`STAT` no se toca**: restar un modelo determinista no cambia la varianza.

La memoria queda acotada por el trozo, no por el número de exposiciones.

## 4 · Productos

| producto | contenido |
|---|---|
| `stages/cube_psfsub_perobs.fits` | `DATA`, `STAT`, `WAVELENGTH`, `COUNT` del combinado de los residuos |
| `stages/stage_e01b_qc.json` | geometría usada, QC del combinado y, por exposición, la amplitud por canal resumida |

La amplitud por canal de cada exposición es **el flujo de la primaria en esa exposición**: su
dispersión entre exposiciones mide lo que el combinado promedia, y es la primera vez que la cadena
lo publica.

## 5 · Verificación

* **Equivalencia lineal (obligatoria).** Con `method="mean"`, restar-luego-combinar tiene que dar
  exactamente lo mismo que combinar-luego-restar la mezcla. Está fijado por test
  (`tests/test_stage_e01b_subtract.py`). Si deja de cumplirse, el alineado, los pesos o el orden
  de las operaciones han dejado de ser los que se creen.
* **Neutralidad del gancho.** Sin `transform`, `combine_streaming` reproduce bit a bit el cubo que
  hay en disco.
* **Recorte.** Con `sigclip`, la fracción de vóxeles recortados se publica en
  `combine.rejected_fraction`: es la única fuente de diferencia entre las dos rutas, y se mide en
  vez de suponerse despreciable.

## 6 · Qué NO decide esta etapa

1. **No elige la forma de PSF** ni la ajusta: eso es C1.
2. **No sustituye al cubo combinado.** C2, C4, C5 y C6 siguen sobre el cubo de B2; sólo la
   variante `psfsub` de C3 consume este producto.
3. **No cambia el presupuesto de ruido.** `STAT` viaja intacto y σ se sigue estimando
   empíricamente con controles procesados igual que el objeto (`docs/noise_model.md`).
