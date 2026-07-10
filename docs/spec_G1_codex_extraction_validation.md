# Especificación G1 · `extraction_validation` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-06. Fase G1 del plan
`docs/00_plan_caracterizacion_espectroscopica.md`. Prerrequisito: G0 cerrada
(run real completo con QCs). Aplica §7 del índice.

**Propósito científico**: convertir los productos de G0 en un espectro con
exactitud demostrada. La concordancia entre X01/X02/X03 NO basta (comparten
la PSF de E01): el árbitro de exactitud es la señal inyectada conocida. G1
entrega el presupuesto de sesgos y la covarianza espectral que G2/G3
necesitan para que sus χ² y errores sean defendibles.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: auditor cuantitativo de la cadena espacial y espectral —
localización, PSF, tres extractores — usando inyección-recuperación como
patrón de referencia y los controles al mismo radio como población nula.

**Definición de terminado**:

1. `tables/g1_bias_budget.csv`: sesgo de flujo (continuo y línea), pérdida de
   apertura residual, contaminación del halo y sesgo de centroide, POR
   método, con incertidumbre, derivados de inyecciones (H04 ampliado según
   §3.2, dentro de la grilla congelada + perturbaciones autorizadas por E4).
2. Matriz de covarianza espectral empírica por bandas
   (`stages/g1_channel_covariance.npz`) + factor de covarianza espacial del
   resampleo (§3.4), consumibles por X10/X11/G2 (formato §5).
3. Análisis de sensibilidad (§3.3): flujo del compañero vs radio de apertura,
   modelo de fondo, perturbación de PSF y desplazamiento de posición, con
   derivadas registradas.
4. Veredicto de validación por método (`validated` / `validated_with_bias` /
   `rejected`) con reglas de §6, escrito en `stage_g1_qc.json`.
5. Módulo nuevo mínimo `musepipe/covariance.py` + tests; sin tocar los
   extractores (si un extractor necesita cambio → issue, no parche aquí).

---

## 1. Límites duros

1. Los extractores se ejecutan con su config de producción EXACTA (la de sus
   QC de G0); prohibido afinarlos para que pasen la validación.
2. Inyecciones solo en clones (patrón E4); la grilla congelada de E4 no se
   amplía sin preguntar — las corridas extra de §3.2/§3.3 se limitan a lo
   presupuestado ahí.
3. La covarianza se estima de datos (controles, residuos), no de un modelo
   teórico del instrumento.
4. Sin cambios de interfaces públicas de `musepipe/extraction/`;
   `covariance.py` es aditivo.
5. Ningún resultado de Hα se reinterpreta aquí; G1 valida la maquinaria, el
   veredicto científico de Hα queda como lo dejó G0 + specs E.
6. Git: rama `phase-g1-validation`.

---

## 2. Precondiciones e inspección obligatoria

1. Leer: auditoría, índice G, bitácora G0 (`docs/g0_execution_log.md`) e
   issues abiertos, specs C1–C4, D1, E4 y sus QC del run G0.
2. Verificar precondiciones: QCs de B3, E01, X01–X03, X10, X11, H04 del run
   G0 existentes y en verde (o con issues aceptados); clones de inyección de
   H04 disponibles.
3. Reutilizar SIEMPRE: `musepipe/injection.py` (inyección), `stats.py`
   (robust sigma), los 31 controles al mismo radio (patrón 07b/08c),
   `extraction/product.py` (formato estándar). No duplicar nada de esto.
4. `pytest` base registrado.

## 3. Operaciones

### 3.1 Validación de localización y PSF (consume QCs de G0, no re-corre)

- Contraste B3: separación/PA vs literatura (ya en QC) + estabilidad del
  centroide entre las imágenes colapsadas de bandas distintas; si
  `chromatic_centroid_needed=true`, verificar que X02/X03 lo consumieron.
- Contraste E01: residuo del halo en el radio del compañero vs λ (métrica de
  C1) tabulado como término del presupuesto de sesgos; verificar suavidad de
  parámetros PSF (sin saltos > 3σ entre bins).

### 3.2 Exactitud por inyección (el árbitro)

- Con los resultados de H04 (grilla congelada): curvas throughput, sesgo de
  flujo vs S/N, sesgo de centroide, completitud — POR método.
- Añadir (dentro del presupuesto E4): inyección de **continuo + línea**
  simultáneos al nivel medido del compañero en G0, en la posición real y en
  3 controles, para medir sesgo de continuo (no solo de línea) y crosstalk
  línea/continuo en X02/X03.
- Perturbación de PSF ±10% FWHM (prevista en E4): la variación del flujo
  recuperado es el término sistemático de PSF del presupuesto.
- Regla central: sesgo aceptable si |sesgo| < 5% para S/N ≥ 5 (criterio E4);
  todo sesgo medido se REGISTRA y corrige en el presupuesto, no se oculta.

### 3.3 Sensibilidad a elecciones de análisis

Tornado plot por método: flujo en las bandas congeladas de D1 al variar
(uno a la vez, rangos en config): radio/box de apertura, modelo de fondo
(plane vs mediana azimutal), ±10% FWHM de PSF, ±0.5 px de posición.
Derivada finita y rango de variación en tabla; cualquier sensibilidad que
supere el error estadístico de la banda → issue para G2/G3.

### 3.4 Covarianza

- **Espectral**: matriz empírica de correlación canal-canal de los 31
  espectros de control (por bloques de λ, p. ej. 200 canales, promediando la
  estructura por distancia entre canales); longitud de correlación efectiva
  por bloque. Guardar densa por bloques + resumen (`rho_1`, `n_eff/n`).
- **Espacial del resampleo** (plan B1-R): en regiones vacías del cubo G0,
  razón varianza(suma en caja N×N) / (N²·varianza(píxel)) para N=1..5 →
  factor de inflación por tamaño de apertura, tabulado.
- Documentar el dominio de validez: la covarianza de controles NO captura
  sistemáticos de la posición del compañero (halo); eso lo cubre §3.2.

## 4. Puntos de atención

1. **Circularidad PSF**: inyectar con la MISMA PSF con la que se extrae
   subestima el sesgo de modelo; por eso la perturbación ±10% es obligatoria
   y su resultado es el sistemático, no el caso nominal.
2. **Controles contaminados**: un control que caiga sobre el segundo objeto
   del campo o un residuo de stripe sesga la covarianza — reutilizar la
   selección validada de 07b/08c, no regenerar posiciones.
3. **N pequeño**: con 31 controles la matriz densa por bloques es ruidosa;
   reportar el error de la propia covarianza (bootstrap sobre controles) y
   preferir el resumen paramétrico si la densa no converge.
4. **No degradar X10/X11**: la covarianza se ENTREGA como producto para
   consumo posterior; recalcular los χ² de X10 con ella es un análisis de
   impacto (tabla comparativa), no una reescritura de la etapa.

## 5. Salidas y esquema de QC

```json
{
  "stage": "g1_extraction_validation",
  "run_id": "...",
  "inputs": {"g0_qc_hashes": {}},
  "method_verdicts": {"x01": "validated", "x02": "validated_with_bias",
                        "x03": "validated"},
  "bias_budget": [{"method": "x03", "term": "psf_model", "value_frac": 0.0,
                    "err_frac": 0.0, "source": "injection_psf_perturbation"}],
  "covariance": {"spectral_file": "g1_channel_covariance.npz",
                  "corr_length_channels_median": 0.0,
                  "spatial_inflation_by_box": {"3": 1.0, "5": 1.0},
                  "bootstrap_err": 0.0},
  "sensitivity": {"worst_term_by_method": {}},
  "impact_on_x10_chi2": {"before": 0.0, "after": 0.0},
  "open_issues": []
}
```

Productos: `tables/g1_bias_budget.csv`, `tables/g1_sensitivity.csv`,
`stages/g1_channel_covariance.npz`, figuras §6.

## 6. Verificaciones y reglas de veredicto

- **V1 — Regresión de inyección**: caso nominal histórico reproducido (gate
  E4 ya exigido en G0; aquí se re-verifica tras cualquier fix).
- **V2 — Curvas 1:1**: flujo recuperado vs inyectado por método con banda de
  sesgo; figura obligatoria.
- **V3 — Covarianza**: figura matriz de correlación por bloque + longitud de
  correlación vs λ; factor espacial vs N con el valor 1.0 marcado.
- **V4 — Tornado**: figura por método; términos > error estadístico
  señalados.
- **V5 — Impacto**: tabla χ² de X10 con/sin covarianza; el veredicto de X10
  no cambia de categoría o, si cambia, issue bloqueante.

Reglas de veredicto por método: `validated` si |sesgo| < 5% (S/N≥5) y
sensibilidad §3.3 < 1σ estadístico; `validated_with_bias` si el sesgo es
estable y queda corregido/propagado en el presupuesto; `rejected` si el sesgo
depende de parámetros de forma no acotable — un método `rejected` no puede
ser el canónico de X11 (issue bloqueante si lo era).

## 7. Tests

```text
tests/test_covariance_synthetic.py   # ruido correlacionado sintético con rho conocido → recuperado a <10%
tests/test_covariance_blocks.py      # contrato: formas, bloques, claves del npz y del QC
tests/test_g1_budget_math.py         # suma en cuadratura y propagación del presupuesto con casos analíticos
tests/test_g1_verdict_rules.py       # reglas de §6 con presupuestos fabricados (frontera 5%, casos rejected)
```

## 8. Protocolo de parada

Preguntar cuando: algún método resulte `rejected`; el sesgo nominal supere 5%
en S/N≥5; la perturbación de PSF cambie el flujo > 15%; la covarianza
implique n_eff/n < 0.5 (canales fuertemente correlacionados — invalida FAPs
previos: issue bloqueante); el impacto V5 cambie el veredicto de X10.

## 9. Riesgos

- **Científicos**: presupuesto de sesgos incompleto por circularidad de PSF
  (§4.1); covarianza subestimada por pocos controles (§4.3) → errores finales
  optimistas.
- **Técnicos**: costo de las inyecciones extra (presupuestar antes, límite
  4 h de E4); manejo de matrices grandes (bloques, no densa completa).

## 10. Reporte final de Codex

Presupuesto de sesgos comentado término a término, veredictos por método con
su regla aplicada, covarianza (figuras V3) y su dominio de validez, tornado
plots, impacto en X10, checklist §7 del índice, issues para G2/G3 (qué factor
usar y cuándo), comando de reproducción.
