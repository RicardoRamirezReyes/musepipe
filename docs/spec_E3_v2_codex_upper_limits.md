# Especificación E3 v2 · `H03_upper_limits` — qué es el límite citable

Fecha: 2026-10-20. Decisión del autor: «adelante con A al 99 %», tras la auditoría del paper (`~/muse_sesion_20261020/auditoria/INFORME.md`, §1).

**Sustituye** a la v1 (`spec_E3_codex_upper_limits.md`) en la definición del límite (§3.1, §3.2 y el punto 2 de §4). El resto de la v1 sigue vigente: límites duros, entradas, conversión física, figura de contexto, esquema de QC y verificaciones.

---

## 1. Dos cantidades distintas

| | qué es | fórmula | se publica como |
|---|---|---|---|
| **Umbral de detección del 99 %** | el flujo con el que E1 declararía detección (FAP = 0.01) | `z_umbral · σ / throughput` | **sensibilidad** de cada esquema |
| **Límite superior del 99 %** | lo que se puede afirmar del flujo verdadero, dado lo medido | `(max(F_medido, 0) + z_CL · σ) / throughput` | **límite citable** |

- `z_umbral` es el cuantil del 99 % de la Gumbel ajustada a los máximos nulos. Se usa el mismo ajuste que E1 (`stage08c_look_elsewhere.fit_gumbel_tail`, máxima verosimilitud; `docs/2026-10-20_e3_misma_cola_que_e1.md`).
- `σ` es la dispersión del estadístico del filtro adaptado en los controles, la misma que usa E1.
- `F_medido` es el flujo del filtro adaptado que E1 midió en la posición del compañero (`matched_flux` de `halpha_detection_by_method.csv`). Es el máximo del barrido, el número cuya z decide E1. Como entrada del límite es conservador, porque un máximo está sesgado hacia arriba.
- `z_CL = Φ⁻¹(1 − 0.01) = 2.326`. El nivel sale de `h03_tail_fap_99`, el mismo 1 % del criterio de detección.
- El suelo en 0 impide que una fluctuación negativa dé un límite más estrecho que la propia sensibilidad.

**Por qué el umbral no es un límite superior.** Una fuente con el flujo del umbral se detecta la mitad de las veces. En ROXs 42B b, el flujo medido (z = 2.97) quedaba casi igual al umbral (z = 3.00): el «límite» publicado no acotaba el flujo verdadero.

## 2. Qué se publica según el veredicto de E1

- `non_detection` y `candidate` (`product_kind = upper_limit`): el límite superior del 99 % es la cifra citable. El umbral se publica aparte (`f_stat_threshold_99`), como sensibilidad.
- `detection` (`product_kind = sensitivity`, solo con `h03_allow_detection`): una detección no tiene límite superior. Se publica el umbral, como hasta ahora, y el QC dice que no es un límite.

## 3. El knob

`h03_limit_definition` puede valer `measured_plus_margin` (por defecto para `upper_limit`) o `detection_threshold`, que reproduce el comportamiento de la v1. El QC registra la definición, su fuente, el nivel, `z_CL` y la fórmula en `limit_definition`. Cada fila lleva `f_measured`, `z_measured`, `f_stat_threshold_99`, `f_stat_99` (el estadístico del límite) y `z_limit`.

## 4. Otras consecuencias

- **Throughput:** se interpola al S/N del propio límite (`z_limit`), no al del umbral.
- **5σ extrapolado:** sigue siendo un umbral, la sensibilidad que citan los sondeos como «límite a 5σ». Se publica con su advertencia de extrapolación.
- **Dispersión entre métodos de los límites:** ahora incluye la del flujo medido por cada esquema y puede superar el 30 % aunque los umbrales coincidan. En ROXs 42B b: umbrales 16.4 %, límites 32.5 %. E3 lo reporta como issue y F1 lo recoge como `major`.
- **Tests:**
  - `tests/test_h03_chain.py`: `H03LimitDefinitionTests` y la aserción del límite en `test_stage_h03_writes_upper_limit_table_qc_and_context_plot`.
  - `tests/test_h03_tail.py`: la cola compartida con E1.
