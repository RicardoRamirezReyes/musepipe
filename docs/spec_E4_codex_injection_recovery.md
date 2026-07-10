# Especificación E4 · `H04_injection_recovery` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa E4 (ESENCIAL) del plan
`docs/plan_roxs12_reduccion_multimetodo.md`. Prerrequisitos: C1 (PSF), C2–C4
(extractores), B2/B3. Es el árbitro de throughput de D1 y el insumo
obligatorio de E3. Se especifica ANTES que E3 porque E3 la consume.

Generaliza la maquinaria de inyección ya validada
(`stage06_local_surface_injection`, validaciones C1/C2 del historial: grilla
3 líneas × 3 posiciones × 9 S/N; caso nominal S/N 5 → `box3_sum` 8.80 vs 8.97
esperado) a TODOS los extractores nuevos.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: medir sensibilidad, throughput y sesgos de la cadena
completa para señales Hα débiles, POR MÉTODO de extracción, inyectando
compañeros sintéticos con la PSF medida (no gaussianas ad hoc).

**Definición de terminado**:

1. `musepipe/injection.py`: interfaz `inject(cube, catalog) → cube` con
   catálogo de fuentes sintéticas (posición, espectro, PSF de C1),
   independiente del extractor; los extractores se pasan como lista de
   callables con la firma común de `musepipe/extraction/`.
2. Regresión histórica: la inyección nominal (S/N 5, posición (108,61),
   9 canales 6558–6568 Å) reproducida por la vía nueva con `box3_sum` →
   mismos números del QC histórico dentro de precisión numérica. Es el gate
   que conecta lo nuevo con lo validado.
3. Grilla §3 ejecutada y propagada por C2, C3 (ambas variantes) y C4.
4. `tables/injection_throughput_by_method.csv` + curvas; perturbación de PSF
   ejecutada (§3.4); QC según §5; verificaciones §6.

---

## 1. Límites duros

1. **Inyecciones solo en clones**: el patrón del repo (copias aisladas tipo
   `ROXs12b_HaInject_*`, subregión mínima — el caso histórico confinó la
   diferencia a 918 vóxeles). El run científico jamás recibe una inyección.
2. La PSF de inyección es `psf_model` de C1 tal cual (con híbrido si lo hay).
   Gaussianas solo en tests unitarios sintéticos, nunca en el cubo real.
3. **La grilla de §3 es la grilla**: compatible con la histórica para poder
   comparar. Ampliaciones → preguntar (costo computacional).
4. Los extractores se llaman con su configuración de producción EXACTA (la de
   sus QC); prohibido afinarlos para las inyecciones.
5. Presupuesto: estimar tiempo total (nº inyecciones × cadena) ANTES de
   lanzar y presentarlo (checkpoint humano si > 4 h de cómputo).
6. Git: rama `stage-e4-injection`.

---

## 2. Entradas

Cubo de B2 del run científico (base de los clones); `psf_model.json`;
posiciones B3; extractores C2/C3/C4 importables; LSF(6563) de A4; QC
históricos de stage06 para la regresión.

## 3. La grilla

- **Línea**: Hα a la RV sistémica; anchos {LSF, 2×LSF}.
- **Amplitud**: S/N matched-filter de entrada {0, 1, 2, 3, 5, 7, 10}
  (definición idéntica a la histórica para comparabilidad).
- **Continuo del sintético**: {sin continuo, continuo plano al nivel medido
  del compañero} — el segundo caso es crítico para C5/HRSDI futura y para el
  sesgo de auto-sustracción del fondo local.
- **Posiciones**: la real (clon) + las 3 posiciones de control históricas al
  mismo radio.
- Total: 7 amplitudes × 2 anchos × 2 continuos × 4 posiciones = 112
  inyecciones por método (el S/N 0 sirve de nulo). Reducciones por costo →
  checkpoint §1.5.

### 3.4 Perturbación de PSF

Repetir un subconjunto (S/N {3, 5}, ancho LSF, posición real) con PSF de
inyección perturbada ±10% en FWHM manteniendo la extracción con la PSF
nominal: mide la sensibilidad del throughput al error del modelo C1. Va a QC
como incertidumbre del throughput.

## 4. Puntos de atención

1. **No inyectar y extraer con "la misma verdad" sin darse cuenta de qué se
   mide**: inyectar con PSF C1 y extraer con PSF C1 mide el throughput de la
   CADENA, no el error de la PSF — para eso existe §3.4. Ambos números son
   necesarios y NO deben mezclarse en una sola cifra.
2. **Los clones deben pasar por TODA la cadena** (04b/C1-fondo → extractor),
   no solo por el extractor: el throughput incluye lo que el fondo local o el
   modelo de halo se comen de la señal.
3. **Contaminación entre inyecciones**: una inyección por clon en la posición
   real; las 3 posiciones de control pueden compartir clon si su separación
   > 4×FWHM (verificado numéricamente, no asumido).
4. **El estimador de recuperación es el de E1** (matched filter + z empírico),
   no uno propio: si E4 midiera con otro estadístico, su throughput no
   aplicaría a los resultados de E1/E3.

## 5. Esquema de QC y salidas

```json
{
  "stage": "h04_injection_recovery",
  "run_id_base": "...",
  "clones_created": [],
  "regression_historic": {"expected_snr": 8.97, "recovered": 0.0, "verdict": "pass|fail"},
  "grid": {"n_injections": 112, "methods": ["aperture", "optimal_ls", "optimal_psfsub", "psffit"]},
  "throughput": {"per_method_at_snr5": {}, "psf_perturbation_pct": 0.0},
  "bias": {"flux_pct_at_snr5": {}, "centroid_A": {}, "fwhm_pct": {}},
  "completeness_at_5sigma": {},
  "open_issues": []
}
```

`tables/injection_throughput_by_method.csv` (fila por inyección×método) y
figuras: throughput vs S/N por método; flujo recuperado vs inyectado (1:1);
completitud vs flujo; sesgo de centroide vs S/N.

## 6. Verificaciones

- **V1 — Regresión histórica**: §0.2 (gate).
- **V2 — Nulos limpios**: las inyecciones S/N 0 no producen recuperaciones
  sobre el umbral (tasa de falsos positivos consistente con E1).
- **V3 — Monotonía**: throughput y completitud monótonos con S/N por método;
  cualquier no-monotonía = bug o estadística insuficiente → investigar antes
  de reportar.
- **V4 — Jerarquía esperada**: C3/C4 ≥ C2 en throughput con sesgo < 5% a
  S/N ≥ 5 (criterio del plan §D1/E4); si C2 supera a C4, algo está mal en el
  PSF-fitting (cruzar con C4/V-serie).
- **V5 — Continuo**: el caso "con continuo" no degrada el throughput de línea
  más de lo esperable por la sustracción de fondo (número reportado; es el
  dato que la decisión sobre C5 usará).

## 7. Tests

```text
tests/test_injection_unit.py        # inyección analítica en mini-cubo: flujo del catálogo == flujo integrado inyectado
tests/test_injection_clone_safety.py# el run base queda intacto (hash antes/después) tras crear clones
tests/test_injection_regression.py  # V1 automatizado sobre el clon histórico
tests/test_injection_estimator.py   # la recuperación usa el estimador de E1 (misma función importada, no copia)
```

## 8. Protocolo de parada y reporte

Preguntar cuando: V1 falle; el presupuesto §1.5 exceda lo razonable; V3/V4
revelen anomalías sin causa. Reporte: curvas por método, throughput con su
incertidumbre (±PSF), tabla de sesgos, y las dos cifras que E3 necesita
explícitas: throughput al nivel del límite y su error.
