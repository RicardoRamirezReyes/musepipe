# Especificación E3 · `H03_upper_limits` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa E3 (ESENCIAL si E1+E2 concluyen no-detección) del
plan `docs/plan_roxs12_reduccion_multimetodo.md`. Prerrequisitos: E1
(no-detección), E2 (`overall: survives` para la no-detección), E4 (throughput
con incertidumbre — insumo OBLIGATORIO).

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: convertir la no-detección en un límite superior
publicable de flujo y luminosidad de Hα (y tasa de acreción), con cada paso de
la conversión trazable y cada supuesto declarado.

**Cadena de conversión**: límite estadístico (controles, E1) → corregido por
throughput medido (E4, nunca asumido) → flujo físico (calibración D2) →
L_Hα (distancia, extinción) → Ṁ (relación L_acc–L_Hα de literatura, con su
scatter como error dominante declarado).

**Definición de terminado**:

1. `tables/halpha_upper_limits.csv`: límite al 5σ por método y el combinado
   final, en flujo observado, flujo desenrojecido, L_Hα y Ṁ.
2. Cada fila con su cadena de conversión completa (columnas de factores).
3. Figura de contexto: límite vs separación contra la literatura MUSE (puntos
   de Xie et al. 2020 como referencia, escalados con honestidad — ver §3.4).
4. `stage_h03_qc.json` (§5); verificaciones §6; módulo + tests.

---

## 1. Límites duros

1. **Sin throughput de E4 no hay límite**: si E4 no corrió o su V1 falló, E3
   se bloquea. Prohibido publicar el límite estadístico crudo como si fuera
   físico.
2. Parámetros físicos SOLO de config, con cita: distancia (con error), A_V
   (con error y fuente), RV sistémica, relación L_acc–L_Hα elegida (cita +
   scatter). Ningún número astrofísico hardcodeado en el módulo.
3. El límite estadístico viene de la MISMA distribución de controles y el
   mismo estadístico de E1 (funciones importadas, no reimplementadas).
4. Reportar SIEMPRE ambos: flujo observado y desenrojecido — la extinción es
   el supuesto más incierto y el lector debe poder deshacerlo.
5. Git: rama `stage-e3-limits`.

---

## 2. Entradas

QC de E1 (distribución de máximos de controles por método), E4 (throughput ±
error al nivel del límite), D2 (escala de flujo + sistemáticos), config
físico (§1.2), tabla de límites de literatura para la figura (CSV con cita
por punto, en `musepipe/qc/data/`).

## 3. Operaciones

### 3.1 Límite estadístico

Por método: percentil 5σ equivalente de la distribución de máximos
look-elsewhere de los controles (E1/08c), convertido a flujo de línea
integrado con la plantilla de ancho LSF (y 2×LSF como variante reportada).

### 3.2 Corrección de throughput

`F_lim = F_stat / throughput(método, S/N≈límite)` con propagación del error
del throughput (incluida la componente ±PSF de E4 §3.4). Punto de atención:
el throughput a S/N bajo tiene error grande — usar el valor interpolado de la
curva de E4 con su banda, no el punto más cercano.

### 3.3 Conversión física

- Flujo → L_Hα: `4π d²` con el error de distancia propagado.
- Desenrojecimiento: ley estándar (config: R_V, ley citada) con A_V ± σ.
- Ṁ: relación L_acc–L_Hα de config; el scatter intrínseco de la relación
  (~0.3–0.5 dex típico) se propaga y DOMINA — decirlo explícitamente en la
  tabla y el reporte, no esconderlo en una nota.

### 3.4 Figura de contexto

Límite propio vs separación sobre los límites de Xie et al. 2020. Honestidad
del escalado: los límites de literatura son para SUS datos (tiempo de
integración, modo, estrella); anotar en la figura las diferencias (texp, modo)
y NO re-escalar los puntos de literatura — se comparan como están, con
leyenda clara.

### 3.5 Combinado

El límite final citable es el del método canónico (D1); los demás aparecen
como consistencia. NO promediar límites entre métodos correlacionados (mismos
fotones): el "combinado" es el canónico con la dispersión inter-método como
sistemático adicional si supera el 30%.

## 4. Puntos de atención

1. **La cadena entera en una tabla**: cada límite debe poder auditarse leyendo
   una fila (valor estadístico → ×1/throughput → ×4πd² → ×10^(0.4·A_Hα) → Ṁ).
   Si un revisor no puede rehacer el número con la fila, la tabla está mal.
2. **No mezclar σ y percentiles**: la distribución de máximos no es gaussiana;
   "5σ" aquí significa el cuantil equivalente (p ≈ 2.9×10⁻⁷) EXTRAPOLADO —
   con 31 controles no se puebla ese cuantil. Usar el ajuste de cola que 08c
   ya emplea (o Gumbel ajustada a los máximos, documentada) y reportar
   TAMBIÉN el límite al 99% (p=0.01), que sí está poblado empíricamente. El
   límite citable por default es el del 99%; el "5σ" va con su caveat de
   extrapolación.
3. **Consistencia inter-método** > 30% → investigar antes de publicar (¿throughput
   mal interpolado? ¿control contaminado?).

## 5. Esquema de `stage_h03_qc.json`

```json
{
  "stage": "h03_upper_limits",
  "run_id": "...",
  "prerequisites": {"e1_verdict": "non_detection", "e2_overall": "survives", "e4_v1": "pass"},
  "physical_inputs": {"distance_pc": 0.0, "distance_err": 0.0, "av": 0.0, "av_err": 0.0,
                       "av_source": "cita", "lacc_lha_relation": "cita", "relation_scatter_dex": 0.0},
  "limits": [{"method": "psffit", "template_width": "lsf",
               "f_stat_99": 0.0, "f_stat_5sigma_extrap": 0.0,
               "throughput": 0.0, "throughput_err": 0.0,
               "f_lim_observed": 0.0, "f_lim_dereddened": 0.0,
               "l_halpha": 0.0, "mdot": 0.0}],
  "intermethod_scatter_pct": 0.0,
  "open_issues": []
}
```

## 6. Verificaciones y tests

- **V1 — Auditoría de fila**: recomputar un límite a mano (script separado)
  desde la fila de la tabla → mismo número.
- **V2 — Consistencia inter-método** < 30% tras throughput.
- **V3 — Cola bien ajustada**: QQ-plot del ajuste de cola sobre los máximos de
  controles; el límite al 99% empírico vs el del ajuste deben coincidir.

```text
tests/test_h03_chain.py       # cadena de conversión con valores sintéticos → resultado analítico exacto
tests/test_h03_tail.py        # muestras Gumbel sintéticas → cuantiles recuperados; y aviso de extrapolación activo
tests/test_h03_blocks.py      # sin E4 válido → la etapa se niega a correr (el límite duro §1.1 funciona)
```

## 7. Protocolo de parada y reporte

Preguntar cuando: V2 falle; A_V de literatura tenga fuentes discrepantes
(presentar opciones con citas); la relación L_acc–L_Hα no esté decidida en
config. Reporte: tabla completa, figura de contexto, los dos límites (99% y
5σ-extrapolado con caveat), supuestos dominantes en orden de impacto, y
párrafo listo-para-paper con los números del método canónico.
