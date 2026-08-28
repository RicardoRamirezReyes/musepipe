# Spec E4b — inyección-recuperación **por exposición**

**Estado:** contrato propuesto. **Sin implementar**: este documento es el diseño, escrito antes del
código como pide la convención del repo (igual que se hizo con C7).

## 1 · Por qué, y por qué ahora

E4 **no tiene modo por exposición**: no hay una sola referencia a exposiciones en toda la etapa. Por
eso **todo** test de inyección que existe hoy —los falsos positivos de `psffit`, el barrido de
radios, la Figura 2— está hecho sobre el cubo combinado, y eso es una limitación estructural, no
una elección.

Y ya sabemos que el sustrato importa. Medido el 2026-08-27
(`docs/2026-08-27_modelo_psf_por_exposicion.md`): con **su propio** cubo y **su propio** modelo, las
exposiciones de la noche buena de ROXs 12 b subestiman el halo a la separación del compañero en
**+55 a +107 %**, contra el **+17 %** del combinado. El sustrato cambia el error del modelo por un
factor de 3 a 6. Cualquier completitud medida solo en el combinado desconoce eso.

## 2 · La decisión de arquitectura, y por qué NO es un knob de E4

La instrucción original era «actualizar E4». **Se propone una etapa aparte y opcional**, por la
misma razón que C7 no entró en `METHOD_ORDER`:

**E3 consume la tabla de throughput de E4.** Un knob dentro de E4 que cambiara el sustrato
cambiaría, en silencio, el límite de Ṁ que E3 publica — y en un repo cuya familia de bugs
recurrente es exactamente esa (`rc=0` y el fallback silencioso), un interruptor que mueve un
resultado congelado sin decirlo es lo que no se puede hacer. Con etapa aparte: QC propio, E3 no la
lee, la cadena no se mueve, y es un producto de **referencia y validación**, igual que C7.

Reutiliza, sin duplicar:

| de | qué |
|---|---|
| **C7** (`stage_x06_perexp`) | `resolve_observation_plan`, `extract_one_exposure` (misma ventana, mismo desplazamiento subpíxel y mismo recorte que produjeron el combinado), `group_exposures`, `combine_measurements` |
| **E4** (`stage_h04_injection`) | `build_h04_cases`, `_injection_sigma`, `_line_center_from_config`, `measure_recovery_with_h01_estimator`, `_row_for_method` |
| **C1** | `psf_model_mixture.json`: el modelo **de cada exposición**, que es lo que se inyecta y lo que corrige la apertura |

## 3 · Procedimiento

1. Resolver el plan de observación (la vista cacheada de C7, que valida que los cubos siguen ahí).
2. Rejilla de casos **idéntica a la de E4** (`build_h04_cases`): posiciones, `h04_snr_grid`,
   `h04_template_width_factors`, `h04_continuum_modes`. Las posiciones se declaran en el marco del
   plan, no en el del recorte — es el error que ya costó una medida el 2026-08-27.
3. **Por exposición**: inyectar la fuente sintética en el cubo de esa exposición **con el modelo de
   PSF de esa exposición**, y extraer con el mismo camino que C7. La σ de la inyección se escala a
   la de **esa** exposición, no a la del combinado: inyectar «SNR=1» con la σ del combinado en una
   exposición de 300 s no es SNR=1.
4. **Combinar las medidas** (`combine_measurements`, ley declarada, `invvar` por defecto) y agrupar
   (`none` / `night`), exactamente como C7.
5. Publicar, además de la completitud combinada, la **completitud por exposición**: es lo que esta
   etapa existe para poder ver.

## 4 · Productos

| producto | contenido |
|---|---|
| `tables/perexp_injection_by_exposure.csv` | una fila por (exposición, caso, método): la tabla larga |
| `tables/perexp_injection_combined.csv` | las medidas ya combinadas, con la ley y el grupo |
| `stages/stage_h04b_qc.json` | QC: rejilla, ley, agrupación, completitud, y `positions_per_point` / `injections_per_point` / `rows_per_point` como campos **distintos** |

`stage_registry` la publica con `qc_optional=True` y `exec_kind="module_main"`.

## 5 · Verificación

- **V1 — Ancla contra E4**: con una sola exposición y sin agrupar, la fila tiene que reproducir lo
  que E4 daría sobre ese mismo cubo. Es lo que hace honesto llamarlo «la misma medida, otro
  sustrato».
- **V2 — Señal nula es señal nula**: con `input_snr=0` el flujo inyectado tiene que ser
  **exactamente** `0.0` en todas las exposiciones. Es la definición que hace del punto SNR=0 una
  medida de falsos positivos y no un relleno.
- **V3 — La σ es de la exposición**, no del combinado. Falla si el escalado usa la σ del combinado.
- **V4 — El marco**: las posiciones inyectadas caen donde el plan dice, verificado contra el pico de
  brillo de cada exposición, no contra una clave declarada.
- **V5 — Agrupación**: la unión de los grupos contiene exactamente las exposiciones del plan.
- **Guardias que deben fallar**: un documento de PSF que no sea `form="mixture"`; una exposición del
  plan sin modelo; `x06b_combine` desconocida.

## 6 · Qué NO decide esta etapa

- **No sustituye a E4 ni cambia su salida.** E3 sigue leyendo a E4. Si algún día el límite de Ṁ
  debe salir de aquí, será con una medida y una decisión explícita, no por un knob.
- **No cambia dónde se combina la cadena.** `docs/2026-08-15_psf_por_observacion.md` sigue
  congelando «combinar cubos».
- **No resuelve el pedestal de `psffit`.** Da el sustrato para poder preguntarlo; la respuesta será
  lo que mida.

## 7 · Coste estimado

E4 sobre el combinado cuesta ~50 min con 192 inyecciones y 6 métodos. Por exposición son 29 cubos,
así que el coste crudo es ~29×. Mitigaciones ya validadas el 2026-08-27: restringir métodos
(`h04_methods`) y rejilla de SNR bajan el coste ~4× **reproduciendo el resultado al 0.1 %**. Un
barrido útil (3 métodos, rejilla corta) debería quedar en unas pocas horas, y admite el mismo
`--jobs` que C7.
