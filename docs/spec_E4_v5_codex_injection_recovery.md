# Especificación E4 v5 — la puerta V2 compara contra su propia población

Fecha: 2026-08-29. Revisión formal de `docs/spec_E4_v4_codex_injection_recovery.md`, que sigue
siendo la referencia para todo lo demás: v5 cambia **sólo la referencia de la puerta V2**. La v2 §1.5
exigía spec nueva para cualquier cambio en las reglas de gate; esto lo es.

## 0. Motivo

V2 pregunta si las inyecciones **nulas** salen limpias, comparando el flujo de cada fila contra una
distribución empírica. Hasta v4 esa distribución eran los **33 controles de producción**
(`build_empirical_null_reference`). **No son la misma población que las filas que juzga.**

Medido el 2026-08-29 sobre los productos en disco:

- Los controles de producción están repartidos por el campo; las inyecciones nulas están **al radio
  del compañero** y pasan por la maquinaria de inyección. En ROXs 12 b, `aperture`:
  **175.0 ± 20.4** en producción contra **60.3 ± 162.8** en inyección — un factor 8 en la escala.
- La consecuencia se ve directamente: **162 de 324** filas en ROXs 12 b y **107 de 210** en
  ROXs 42B b caen **por debajo de los 33 controles**, o sea en el suelo de la cola baja. La mitad.
- **La puerta es unilateral por diseño** (busca valores altos). Una puerta cuyas filas viven enteras
  en la otra cola **no puede disparar**: sus `pass` no dicen que las nulas estén limpias, dicen que
  se están comparando con otra cosa.
- Y explica los fallos históricos que la v3 §0 atribuía a la agregación y a la extracción
  (`excess_p` de 1e-4 a 1e-7): con las poblaciones cruzadas, lo que fallaba podía ser el cruce.

## 1. Qué cambia

Sólo la referencia:

- La FAP de rango de cada fila nula se calcula contra **las otras filas nulas de su estrato**
  (método × ancho de plantilla × modo de continuo), **dejándose fuera** (leave-one-out).
- `h04_null_gate_reference` declara cuál se usa: `injection_nulls` (nuevo, por defecto) o
  `production_controls` (histórico).
- El QC lo publica en `checks.v2_nulls_clean.reference`.

**No cambian**: la unidad de la puerta (la **posición** de control), el resumen por posición (la
**mediana** de sus filas), la FAP de rango `(1 + N(null ≥ observado))/(N + 1)`, los dos umbrales
congelados (`p_null = 0.0455`, `gate_alpha = 0.01`), el binomial sobre `n_positions`, ni el
diagnóstico de cola baja **sin voto**. La v3 sigue vigente en todo eso.

## 2. Por qué el leave-one-out no es un detalle

Sin él cada fila entra en su propia referencia y se rebaja sola: con `n` nulas su FAP no puede bajar
de `1/(n+1)` por construcción. Con LOO la FAP de una fila limpia es **uniforme** en `1/n … 1`, que
es lo que el umbral `p_null` supone.

Y de paso arregla la granularidad que la v3 §0 punto 3 denunciaba: con 33 controles la FAP valía
`k/34` y el único valor por debajo de 0.0455 era `1/34 = 0.0294`, así que la prueba individual era
«esta fila es el máximo de sus 34». Con 54 nulas en ROXs 12 b la FAP vale `k/54` y hay **dos**
niveles por debajo del umbral. En ROXs 42B b, con 35, sigue habiendo uno: **la granularidad depende
del número de posiciones**, y ese está limitado por el radio (spec v4 §7).

## 3. Qué dice sobre los datos de hoy

| | referencia de producción (v4) | población emparejada (v5) |
|---|---|---|
| ROXs 12 b | `pass`, 1/54 extremas, `excess_p` 0.919 | **`pass`, 0/54, `excess_p` 1.000** |
| ROXs 42B b | `pass`, 0/35, `excess_p` 1.000 | **`pass`, 0/35, `excess_p` 1.000** |
| filas en el suelo de la cola baja | **162/324 y 107/210** | **12 y 6** |
| FAP por posición | 1.000 o 0.029 según el método | mediana **0.519** y **0.471** |

**El veredicto no cambia: las dos pasan.** Lo que cambia es que ahora pasar significa algo. Una FAP
que se distribuye alrededor de 0.5 es la firma de un test bien planteado; una que sólo vale 1.000 o
0.029 según el método es la de uno roto.

Que el cambio sea **neutral en el veredicto hoy** es la razón de hacerlo hoy: la puerta se arregla
mientras no está decidiendo nada.

## 4. QC

`checks.v2_nulls_clean.reference` declara la población. `derived_finalize.v2_reference` y
`v2_status` registran el recálculo cuando la puerta se recomputa sin re-inyectar (§5).

## 5. Cómo se aplica

V2 es **derivada**: sus filas son las nulas de la tabla y su referencia se reconstruye. Cambiarla
**no exige repetir la rejilla de inyecciones**:

```bash
python -m musepipe.stages.stage_h04_injection --run-id <RUN> --finalize
```

`finalize` recomputa V2, `complete`, la completitud y la declaración de sustrato, y deja intactas las
medidas (flujo, sigma, SNR, throughput).

## 6. Tests mínimos

`tests/test_h04_snr_standardization.py::TestReferenciaDeV2`: una nula no entra en su propia
referencia; no se cruzan estratos; se declara contra qué se comparó; la posición real sigue sin
entrar; un modo desconocido para. `tests/test_h04_v2_policy.py` conserva la regla de agregación de
v3 y pasa `reference_mode="production_controls"` explícito, porque no va de esto.

## 7. Protocolo de parada

- **La referencia emparejada no arregla la potencia, sólo la validez.** Con 35 posiciones en
  ROXs 42B b, la FAP mínima es 1/35 = 0.029 y el umbral 0.0455: la puerta sólo puede señalar «esta
  posición es la más extrema de las suyas». Para resolver más hacen falta más posiciones, y el radio
  las limita.
- **El diagnóstico de cola baja sigue sin votar**, y ahora es interpretable: con la población
  emparejada, un exceso en la cola baja sí es sobre-sustracción y no un artefacto de comparar dos
  poblaciones.
- **La referencia de producción no se borra.** Sigue disponible en el knob y sigue siendo la que
  usan E1/E2/G2 para otras cosas; lo que se retira es su uso como referencia de esta puerta.
