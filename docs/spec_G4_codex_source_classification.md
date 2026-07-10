# Especificación G4 · `source_classification` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-06. Fase G4 del plan
`docs/00_plan_caracterizacion_espectroscopica.md`. Prerrequisitos: G2 cerrada;
G3 cerrada (la parte astrométrica/morfológica puede desarrollarse con
G0+G1). Aplica §7 del índice.

**Propósito científico**: evaluar de forma transparente y cuantitativa las
hipótesis sobre la naturaleza de la fuente — planeta en formación, compañera
subestelar/enana marrón, estrella M asociada, estrella M de fondo, fuente
contaminante (p. ej. galaxia), artefacto residual — mostrando la evidencia a
favor y en contra de cada una. **Prohibido un clasificador opaco que entregue
solo una etiqueta.**

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: autor de `musepipe/stages/stage_g4_classify.py` y del
marco de evidencia: una matriz hipótesis × pruebas donde cada celda es un
número interpretable con su procedencia.

**Definición de terminado**:

1. Conjunto de pruebas de §3 implementado; cada prueba produce, por
   hipótesis: una verosimilitud o un veredicto discreto
   (`supports | neutral | disfavors | excludes`) + el dato numérico que lo
   sustenta + su fuente (QC/tabla de fases previas).
2. Matriz de evidencia completa (`tables/g4_evidence_matrix.csv`) y resumen
   por hipótesis con las pruebas dominantes identificadas.
3. Probabilidad de contaminación de fondo calculada (§3.4) con densidad
   estelar citada.
4. Clasificación final = ranking de hipótesis con su evidencia, etiquetada
   por robustez (`secure | probable | ambiguous`), criterios congelados de
   §6 definidos ANTES de mirar la matriz real.
5. Tests §7 verdes; figuras §6.

---

## 1. Límites duros

1. **Transparencia**: el producto principal es la matriz de evidencia, no la
   etiqueta. Toda celda tiene `source` (archivo/clave de QC de origen) y es
   re-derivable. Nada de pesos ocultos: los pesos/umbrales de combinación
   viven en config, versionados y justificados con comentario.
2. **Anti-sesgo (patrón E1)**: los umbrales de decisión de §6 se congelan en
   esta spec/config ANTES de computar la matriz con datos reales; cambiarlos
   después de ver el resultado exige issue + autorización humana.
3. G4 consume productos de G0–G3 y catálogos externos citados; no re-mide
   nada. Si una prueba necesita una medición inexistente, se marca
   `not_available`, no se improvisa.
4. Cada hipótesis se evalúa contra TODAS las pruebas disponibles; prohibido
   descartar una hipótesis por una sola prueba sin marcar la celda
   `excludes` con su justificación numérica explícita.
5. Git: rama `phase-g4-classify`.

---

## 2. Precondiciones e inspección obligatoria

1. Leer: auditoría, índice G, QC de B3 (astrometría), H02 (artefactos, su
   veredicto se REUTILIZA como prueba T7, no se re-implementa), tabla G2
   (líneas, RV), tabla G3 (propiedades, distancias fotométricas implícitas),
   presupuesto G1.
2. Inputs externos a config con cita: parallax/distancia y cinemática del
   sistema primario (Gaia), edad de la región, densidad estelar de fondo
   (modelo de población tipo Besançon/TRILEGAL o conteos de catálogo — el
   checkpoint §8.1 decide), astrometría de épocas previas del compañero
   (Bowler et al. u otras) si existe.
3. `pytest` base registrado.

## 3. Pruebas (cada una: dato → verosimilitud/veredicto por hipótesis)

- **T1 Astrometría relativa**: separación/PA de B3 vs predicción de compañera
  ligada (órbita/época previa) y vs escenario de fondo estático (si hay ≥ 2
  épocas: test de movimiento propio común con χ² de ambas hipótesis; con 1
  época: `neutral` entre ligada/fondo, solo consistencia).
- **T2 Morfología/PSF**: ¿fuente puntual con la PSF de E01? (χ² del stamp vs
  PSF; extendida → favorece galaxia/artefacto; reutilizar la maquinaria de
  stamps de H02).
- **T3 Continuo y SpT (G3)**: ¿la forma espectral es de M/L juvenil, de M de
  campo, o no estelar? Δχ² entre las mejores plantillas de cada clase de
  gravedad; `excludes` para clases con Δχ² > umbral congelado.
- **T4 Luminosidad/distancia**: escala Ω de G3 + distancia del sistema → R
  físico plausible a esa distancia vs R implicado si estuviera a distancia de
  campo (¿qué distancia haría autoconsistente la luminosidad?); compara con
  la fotometría esperada para cada hipótesis.
- **T5 RV**: RV de G2 vs RV sistémica (config, cita) vs expectativa de fondo
  (dispersión galáctica en esa línea de visión); nota: σ_RV de MUSE limita el
  poder — la prueba declara su potencia, no finge discriminar lo que no puede.
- **T6 Extinción**: A_V de G3 vs A_V del sistema/nube (cita); A_V mucho mayor
  → favorece fondo tras la nube.
- **T7 Artefacto**: veredicto de H02 (batería completa) importado tal cual.
- **T8 Líneas de emisión**: presencia/ausencia y límites de acreción (G2/G3)
  interpretados por hipótesis (acreción activa favorece objeto joven; su
  ausencia es débilmente informativa — codificar esa asimetría en el mapeo).
- **T9 Contaminación de fondo (§3.4)**: probabilidad a priori de encontrar
  una fuente de las propiedades observadas en el área de búsqueda.

### 3.4 Probabilidad de contaminación

P(al menos un contaminante en el área de tolerancia de B3 con magnitud ≤ la
observada y color/SpT compatible), con densidad del modelo/catálogo citado.
Reportar también la versión condicionada a lo que T3/T6 ya excluyen. Área de
búsqueda = la región REALMENTE explorada (honestidad look-elsewhere espacial).

## 4. Combinación y puntos de atención

1. Combinación: verosimilitudes log-aditivas donde las pruebas sean
   aproximadamente independientes; donde no lo sean (T3/T4/T6 comparten el
   espectro y A_V), agrupar en una sola verosimilitud conjunta derivada del
   ajuste G3 — NO multiplicar evidencia correlacionada como si fuera
   independiente.
2. Hipótesis compuestas: "planeta en formación" vs "compañera subestelar" se
   distinguen esencialmente por masa (G3, dependiente de modelos) y acreción;
   si sus verosimilitudes son indistinguibles, el resultado honesto es la
   clase combinada `companion_substellar_or_planetary` — está permitida y
   prevista.
3. Pruebas `not_available` no puntúan: la robustez (`secure/probable/
   ambiguous`) depende de CUÁNTAS pruebas independientes soportan al líder.
4. Sensibilidad: repetir el ranking excluyendo una prueba a la vez
   (leave-one-out); si el líder cambia, la clasificación es `ambiguous` por
   definición (regla congelada).

## 5. Esquema de salidas

`tables/g4_evidence_matrix.csv`: filas = pruebas T1–T9, columnas = hipótesis;
celdas `verdict:value±err@source`. `stage_g4_qc.json`:

```json
{
  "stage": "g4_classification",
  "run_id": "...",
  "hypotheses": ["planet_forming", "substellar_companion", "brown_dwarf",
                  "m_star_associated", "m_star_background", "contaminant",
                  "artifact"],
  "tests_available": ["T1", "..."], "tests_unavailable": [],
  "combined_ranking": [{"hypothesis": "...", "log_l_rel": 0.0,
                          "dominant_tests": []}],
  "background_probability": {"raw": 0.0, "conditioned": 0.0, "source": "..."},
  "leave_one_out_stable": true,
  "final_class": {"label": "...", "robustness": "secure|probable|ambiguous"},
  "frozen_thresholds_hash": "...",
  "open_issues": []
}
```

## 6. Verificaciones y criterios congelados

Umbrales congelados (en config, hash en QC, definidos antes de correr con
datos reales): Δχ² de exclusión de T3; umbral de `excludes` por prueba;
`secure` = líder soportado por ≥ 3 pruebas independientes, sin `excludes` en
contra, leave-one-out estable y P(fondo) < 1%; `probable` = líder estable
pero < 3 pruebas o P(fondo) 1–10%; resto = `ambiguous`.

- **V1 — Matriz completa**: toda celda con valor+fuente o `not_available`.
- **V2 — Figura resumen**: heatmap hipótesis × pruebas con veredictos.
- **V3 — Leave-one-out**: tabla de estabilidad del ranking.
- **V4 — Casos de control sintéticos**: matrices fabricadas para un fondo
  M obvio, un artefacto obvio y una compañera clara → la combinación devuelve
  la clase correcta con la robustez esperada (test §7).
- **V5 — Anti-sesgo**: hash de umbrales congelados anterior (en git) a la
  primera ejecución con datos reales.

## 7. Tests

```text
tests/test_g4_matrix_contract.py     # contrato: celdas, fuentes, hipótesis, QC
tests/test_g4_combination.py         # log-verosimilitudes con casos analíticos; agrupación de pruebas correlacionadas
tests/test_g4_synthetic_cases.py     # V4: tres escenarios fabricados → clase y robustez correctas
tests/test_g4_background_prob.py     # P(fondo) con densidad sintética conocida (analítico)
tests/test_g4_loo_rule.py            # cambio de líder en leave-one-out → ambiguous forzado
```

## 8. Protocolo de parada

1. **Checkpoint antes de implementar T9**: proponer la fuente de densidad de
   fondo (modelo vs catálogo) al humano.
2. Además, preguntar cuando: dos hipótesis lideren dentro del margen (reportar
   ambas, no forzar una); alguna prueba contradiga frontalmente a G3 (p. ej.
   T4 exige una distancia incompatible con T1); haya que ajustar umbrales
   congelados tras ver datos reales (§1.2); T7 (H02) esté en rojo — la
   clasificación no procede sobre una señal no depurada de artefactos.

## 9. Riesgos

- **Científicos**: falsa seguridad por evidencia correlacionada (§4.1);
  clasificación con una sola época astrométrica (T1 débil — la robustez lo
  refleja); asimetría de información de T8 mal codificada (ausencia de Hα no
  prueba fondo).
- **Técnicos**: dependencia de catálogos externos (citas y versiones en
  config); acoplamiento a G3 (usar solo su tabla publicada, no internos).

## 10. Reporte final de Codex

Matriz de evidencia comentada prueba a prueba, ranking con pruebas
dominantes, P(fondo) con supuestos, leave-one-out, clase final con robustez y
TODA la evidencia en contra listada (obligatorio aunque el líder sea claro),
V1–V5, checklist §7 del índice, issues para G5, comando de reproducción.
