# Especificación G2 · `spectral_measurements` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-06. Fase G2 del plan
`docs/2026-07-08_plan_caracterizacion_espectroscopica.md`. El desarrollo del módulo
puede correr en paralelo con G0/G1 (validación sintética); la aplicación al
espectro real exige el canónico de X11 (G0) y los factores de G1. Aplica §7
del índice.

**Propósito científico**: infraestructura ÚNICA y reutilizable de medición de
líneas y continuo — flujo integrado, densidad de continuo, ancho equivalente,
centroide, FWHM, asimetría, ajuste de perfil, velocidad radial, límites
superiores y significancia — aplicable a Hα y a cualquier línea del rango
MUSE, con errores propagados y etiquetas `direct_measurement`/`upper_limit`.
Cierra la brecha: hoy `line_metrics` (stage07) no mide EW ni FWHM y H01 solo
se usa para Hα.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: autor de `musepipe/lines.py` (módulo puro) +
`musepipe/stages/stage_g2_measure_lines.py` (etapa), reutilizando lo
existente sin duplicar.

**Definición de terminado**:

1. `musepipe/lines.py`: API de §4 con perfil gaussiano⊗LSF (y plantilla para
   otros perfiles), medición por momentos, EW, RV, límites y significancia;
   errores por Monte Carlo con covarianza opcional de G1.
2. `stage_g2_measure_lines`: aplica la API al espectro estándar §2.3 del plan
   maestro para el catálogo de líneas de config (default: las ~25 de
   `default_accretion_lines()` de stage07) → tabla maestra + QC + figuras.
3. H01 NO duplicado: G2 usa su propio estimador para líneas medidas y DELEGA
   la significancia tipo matched-filter a la maquinaria H01 parametrizada por
   `rest_A` (es genérica; verificar y, si hay fricción de interfaz, issue).
4. Análisis de sensibilidad a ventanas (§3.5) integrado en la tabla.
5. Tests §7 verdes; aplicación al espectro real del run G0 completada (si G0
   cerró) o marcada `pending_g0` en QC.

---

## 1. Límites duros

1. **Cero lógica específica de Hα en el módulo**: toda línea se define por
   catálogo en config (`name, wave_A, family, kind` — el esquema de stage07).
   Hα es una fila más.
2. No modificar `stage07` ni H01–H03 (siguen siendo la referencia histórica y
   la cadena de detección congelada). G2 es aditiva; la conciliación con
   stage07 es una verificación (§6), no una fusión.
3. Unidades: λ en Å (aire, como el resto del repo — verificar la convención
   en A4/QC antes de asumir), flujo en las unidades del producto estándar
   (BUNIT del FITS), EW en Å con signo NEGATIVO para emisión (documentar la
   convención en el docstring y en cada tabla), velocidades en km/s.
4. Sin constantes ocultas: c, factores FWHM/σ, etc. desde un único lugar
   (`musepipe/constants.py` nuevo si no existe — grep antes).
5. Errores: si hay covarianza de G1 disponible se usa; si no, MC con errores
   independientes + nota `covariance: none` en QC. Nunca silenciosamente.
6. Git: rama `phase-g2-lines`.

---

## 2. Precondiciones e inspección obligatoria

1. Leer: auditoría, índice G, specs E1/E3 (para no duplicar su estadística),
   `stage07_accretion_lines.py` (catálogo y `line_metrics`),
   `stage_h01_detect.py` (matched filter, momentos), `spectral.py`
   (continuos), `extraction/product.py` (formato), QC de X11 del run G0.
2. Confirmar qué ya existe y se reutiliza: continuo running-median y polyfit
   log-λ (`spectral.py`), `robust_sigma` (`stats.py`), plantilla gaussiana y
   momentos (H01). Lo que se reutiliza se IMPORTA, no se copia.
3. LSF: leer la LSF(λ) tabulada del QC de A4; si A4 no la produjo en G0,
   protocolo de parada (no inventar una LSF).
4. `pytest` base registrado.

## 3. Requisitos funcionales y científicos

### 3.1 Ventanas

Por línea: ventana de línea (± half-width en km/s o Å, config con default
por familia) y dos ventanas de continuo (azul/rojo) con exclusión automática
de: otras líneas del catálogo, ventanas malas del run (`bad_ranges`), canales
con flag de skyline/stripe (QC de B2/A4). Si el continuo útil queda con menos
de `min_continuum_pixels`, la línea se marca `not_measurable`, jamás se mide
con lo que quede.

### 3.2 Mediciones por línea (etiqueta `direct_measurement`)

- Continuo local: mediana robusta + pendiente lineal en las dos ventanas;
  densidad de continuo en el centro de línea con error.
- Flujo integrado (suma directa sobre la ventana − continuo) y por ajuste de
  perfil; ambos en la tabla (su diferencia es diagnóstico).
- EW con error (integración de (1 − F/Fc)).
- Perfil: gaussiana ⊗ LSF(λ local) por defecto; parámetros libres amplitud,
  centro, σ intrínseca (≥ 0, permitiendo no resuelta); χ² con errores del
  espectro. Arquitectura de perfil enchufable (lorentziana/voigt después; NO
  implementarlas ahora, solo la interfaz).
- Centroide (momentos y ajuste), FWHM observada e intrínseca (deconvuelta de
  LSF con error; si σ_int consistente con 0 → `unresolved`), asimetría
  (momento tercero normalizado o diferencia de percentiles 10/90 del perfil
  acumulado — elegir uno, documentar).
- RV por línea = c·(centro − λ_rest·(1+v_sys/c))/λ_rest con error del ajuste
  ⊕ error de calibración λ (QC X11); RV promedio ponderada de líneas
  `detected` como producto separado.

### 3.3 Detección, límites y significancia

- Significancia por línea: delega en el matched filter de H01 con `rest_A`
  de la línea y los controles disponibles; donde no haya controles (espectro
  ya 1D), z-score contra el ruido del continuo local con look-elsewhere
  DENTRO de la ventana de búsqueda (Bonferroni sobre canales efectivos,
  usando n_eff de la covarianza G1).
- Estado por línea: `detected` (según umbral en config, default z ≥ 5),
  `marginal` (3–5), `upper_limit` (< 3 → límite 5σ integrado con el ancho de
  plantilla = LSF y variante 2×LSF, etiqueta `upper_limit`),
  `not_measurable`.
- Los límites usan el throughput del método canónico (G1); campo obligatorio
  `throughput_applied`.

### 3.4 Propagación de errores

Monte Carlo (default 1000 realizaciones, config): perturbar el espectro con
`flux_err` (+ covarianza por bloques de G1 si existe) y repetir TODA la
medición; el error de cada cantidad = percentiles 16/84. Semilla en config y
QC. Errores analíticos solo como cross-check en tests.

### 3.5 Sensibilidad a ventanas

Repetir §3.2 con ±25% y ±50% del half-width de línea y con las ventanas de
continuo desplazadas; columna `window_sensitivity_frac` por cantidad. Si
supera el error estadístico → flag `window_dominated`.

## 4. Contrato de la API (`musepipe/lines.py`)

```python
measure_line(wave_A, flux, flux_err, line_def, *, lsf_fwhm_A, cov=None,
             windows=None, n_mc=1000, seed=...) -> LineMeasurement
measure_catalog(spectrum_product, catalog, qc_inputs, cfg) -> list[LineMeasurement]
```

`LineMeasurement` (dataclass → fila de tabla): identificación de línea,
ventanas efectivas, todas las cantidades de §3.2–§3.3 con `_err`, estado,
etiqueta, flags (`unresolved`, `window_dominated`, `sky_contaminated`),
`n_mc`, `seed`, `covariance_used`. La tabla se escribe en
`tables/g2_line_measurements.csv` y el espejo JSON en QC.

Casos límite obligatorios: línea en borde del rango o dentro de `bad_ranges`
(→ `not_measurable` con razón), NaN en ventana (se excluyen; si > 30% →
flag), perfil no convergente (→ momentos con flag `fit_failed`), EW con
continuo ≤ 0 (→ `not_measurable`, típico de flujo negativo residual).

## 5. Esquema de QC

```json
{
  "stage": "g2_measure_lines",
  "run_id": "...",
  "input_spectrum": {"file": "...", "sha256": "...", "method": "x03"},
  "lsf_source": "stage00q_qc",
  "covariance_used": "g1_channel_covariance.npz|none",
  "catalog_n": 25,
  "n_detected": 0, "n_marginal": 0, "n_upper_limit": 0, "n_not_measurable": 0,
  "rv_weighted_kms": {"value": 0.0, "err": 0.0, "n_lines": 0},
  "mc": {"n": 1000, "seed": 0},
  "stage07_reconciliation": {"max_flux_discrepancy_sigma": 0.0},
  "open_issues": []
}
```

## 6. Verificaciones

- **V1 — Sintético**: espectro fabricado con 5 líneas de parámetros conocidos
  (una no resuelta, una en emisión fuerte, una absorción, una en borde, una
  bajo el ruido) → tabla comparativa verdad/medido; todo dentro de 1σ MC.
- **V2 — Conciliación stage07**: sobre el mismo espectro de entrada, flujos
  integrados de G2 vs `line_metrics` históricos; discrepancias > 2σ
  explicadas (definiciones de ventana) o issue.
- **V3 — Hα**: el estado de Hα según G2 coincide en categoría con H01/H03 del
  run G0 (detección/límite); si no, issue bloqueante (dos estadísticas del
  mismo dato no pueden contradecirse sin explicación).
- **V4 — Figuras**: grid de ventanas por línea (dato, continuo, perfil,
  residuo) reutilizando el patrón visual de stage07; figura RV por línea vs
  λ con la ponderada.
- **V5 — MC**: histograma de una cantidad clave con percentiles marcados;
  semilla reproducible (dos corridas → misma tabla).

## 7. Tests

```text
tests/test_lines_synthetic.py      # V1 automatizado (verdad conocida, tolerancias 1σ)
tests/test_lines_ew_sign.py        # convención de signo EW (emisión negativa) y continuo≤0 → not_measurable
tests/test_lines_lsf_deconv.py     # FWHM intrínseca: casos resuelto/no-resuelto analíticos
tests/test_lines_edges.py          # casos límite de §4 (borde, NaN, fit_failed)
tests/test_lines_mc_seed.py        # reproducibilidad con semilla; error MC ≈ analítico en caso gaussiano simple
tests/test_lines_windows.py        # exclusión de líneas vecinas y bad_ranges en ventanas de continuo
tests/test_stage_g2_contract.py    # contrato: columnas de la tabla, claves de QC, etiquetas válidas
```

## 8. Protocolo de parada

Preguntar cuando: falte LSF de A4; V3 contradiga H01; la covarianza de G1
cambie algún estado detección↔límite; > 20% del catálogo resulte
`window_dominated` (las ventanas por familia están mal elegidas — decisión
humana); se necesite tocar stage07/H01 (prohibido en §1.2).

## 9. Riesgos

- **Científicos**: EW y flujo dominados por la elección de continuo en un
  espectro con S/N bajo (mitigación: §3.5 y flags); deconvolución de LSF
  inestable cerca del límite de resolución (reportar `unresolved`, no σ
  espurias); RV con precisión ilusoria (siempre ⊕ error de calibración λ).
- **Técnicos**: MC×25 líneas×sensibilidad = costo (vectorizar; presupuestar);
  duplicación accidental de H01 (prohibida, §0.3).

## 10. Reporte final de Codex

Tabla maestra comentada, V1–V5 con figuras, conciliación stage07 y H01,
decisiones de convenciones (EW, asimetría) con docstrings citados, checklist
§7 del índice, issues para G3 (qué líneas son utilizables para acreción y
cuáles quedan `not_measurable`), comando de reproducción.
