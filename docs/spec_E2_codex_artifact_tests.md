# Especificación E2 · `H02_artifact_tests` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa E2 (ESENCIAL) del plan
`docs/plan_roxs12_reduccion_multimetodo.md`. Prerrequisitos: E1 (veredicto y,
si lo hay, la señal candidata); B2 (métrica de stripes); cubos residuales de
C2/C4.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: ejecutar una batería FIJA de pruebas de artefactos sobre
el resultado de E1 — tanto si hubo señal (¿es real?) como si no (¿la
no-detección es robusta?). La batería se corre completa siempre: no se elige
qué tests correr según el resultado de E1.

**Definición de terminado**: los 5 tests de §2 ejecutados con veredicto
individual en QC; figuras por test; `stage_h02_qc.json` según §4;
`musepipe/stages/stage_h02_artifacts.py` + tests sintéticos; reporte.

---

## 1. Límites duros

1. **Batería congelada**: los 5 tests de §2, con sus umbrales. Ni más ni menos;
   tests adicionales que parezcan buena idea → `open_issues` para revisión.
2. Solo lectura de productos/cubos; sin re-extracciones fuera de lo que la
   batería define.
3. Un test "no aplicable" (p. ej. §2.3 con una sola exposición) se declara
   `unavailable` con su razón — nunca se rellena con un sustituto improvisado.
4. Git: rama `stage-e2-artifacts`.

---

## 2. La batería (congelada)

### T1 — Coincidencia instrumental

¿El canal de la señal (o de los máximos más altos en no-detección) coincide
con: canales sucios de stripes (lista de B2), skylines del catálogo de A4, o
bordes del gap del láser? Métrica: distancia en canales al artefacto más
cercano de cada lista; umbral: coincidencia = |Δcanal| ≤ 2. Veredicto por
lista.

### T2 — Coherencia espacial

En el cubo residual (C4 preferente; C2 como contraste), stamp del canal de la
señal: ajuste de la PSF de C1 al stamp. Una señal real tiene la forma de la
PSF centrada en la posición de B3; un artefacto es alargado, desplazado o
multi-pico. Métricas: razón de χ² (PSF vs plano), desplazamiento del centroide
vs B3 (umbral: < 1 px), elongación vs PSF (< 1.5×). En no-detección se aplica
al máximo global del mapa Hα como caracterización del ruido dominante.

### T3 — Estabilidad temporal

Si el run tiene N ≥ 2 exposiciones/cubos: repetir la medición de E1 (método
canónico) en sub-stacks independientes; una señal real escala como √N, un
artefacto de una exposición no. Umbral: presencia con z > z_total/√2 en ambas
mitades. **Para ROXs12b con UNA exposición: `unavailable`, documentado** — y
el reporte debe decir explícitamente que este eje de robustez no está
disponible para este dataset.

### T4 — Estabilidad frente a parámetros

Repetir E1 (método canónico) variando UNA perilla a la vez, valores fijados
aquí: radio de ajuste local 04b {10, 12, 14 px}; máscara del compañero en C1
{sin cambio, +2 px}; ancho de plantilla (ya barrido en E1, se reutiliza);
continuo {runmed, poly}. Métrica: rango de z del máximo entre variantes;
umbral: Δz < 1 (señal estable) — tornado plot.

### T5 — Placebos espectrales

La cadena E1 completa (mismas plantillas, mismos controles, mismo
look-elsewhere) centrada en líneas placebo FIJAS: 6300 Å descartada (skyline),
usar 6200, 6400, 6700, 7100 Å. Criterio: ningún placebo debe superar el
umbral de detección de E1; si alguno lo supera, la FAP de E1 está mal
calibrada → hallazgo mayor.

## 3. Puntos de atención

1. **No convertir E2 en búsqueda**: los placebos y variantes generan muchos
   números; ninguno se reporta como "posible señal". E2 responde UNA pregunta:
   ¿el resultado de E1 sobrevive?
2. **T4 toca etapas aguas arriba** (re-corre 04b/C1 con perillas): usar copias
   de validación aisladas (patrón del repo), jamás sobre el run principal.
3. **La lista de artefactos de T1 se toma de QC existentes** (B2, A4), no se
   re-deriva aquí — si falta un QC, `unavailable`, no re-medición.

## 4. Esquema de `stage_h02_qc.json`

```json
{
  "stage": "h02_artifact_tests",
  "run_id": "...",
  "input_verdict_e1": "detection|candidate|non_detection",
  "t1": {"nearest_stripe_dch": 0, "nearest_skyline_dch": 0, "coincidence": false, "status": "pass|fail"},
  "t2": {"chi2_ratio_psf_vs_plane": 0.0, "centroid_offset_px": 0.0,
          "elongation_vs_psf": 0.0, "status": "pass|fail"},
  "t3": {"status": "pass|fail|unavailable", "reason": "single_exposure"},
  "t4": {"z_range": 0.0, "worst_knob": "...", "status": "pass|fail"},
  "t5": {"placebo_max_fap_global": 0.0, "any_above_threshold": false, "status": "pass|fail"},
  "overall": "survives|fails|mixed",
  "open_issues": []
}
```

`overall = survives` requiere pass en todos los aplicables; `fails` si T1, T2
o T5 fallan; `mixed` en el resto (decisión al usuario).

## 5. Verificaciones y tests

- Figuras por test (stamp+PSF, tornado, placebos vs umbral, mapa de canales
  con artefactos marcados).
- Tests sintéticos de potencia:

```text
tests/test_h02_t2_shape.py     # fuente PSF sintética → pass; blob alargado 2× → fail (T2 discrimina)
tests/test_h02_t5_calibration.py # ruido puro por la cadena de placebos → ninguno supera umbral (tasa de falsos positivos correcta)
tests/test_h02_t1_lists.py     # señal colocada sobre un canal sucio sintético → coincidencia detectada
```

## 6. Protocolo de parada y reporte

Preguntar cuando: T5 falle (FAP mal calibrada — bloquea E1 y E3); `overall =
mixed`; T4 revele una perilla dominante inesperada. Reporte: tabla de los 5
tests, veredicto global, e implicación directa: detección que sobrevive →
reportar con E4/throughput; candidato que falla → se degrada a no-detección
con causa; no-detección robusta → luz verde a E3.
