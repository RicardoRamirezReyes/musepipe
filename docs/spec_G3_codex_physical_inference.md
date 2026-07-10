# Especificación G3 · `physical_inference` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-06. Fase G3 del plan
`docs/00_plan_caracterizacion_espectroscopica.md`. Prerrequisitos: G1 cerrada
(presupuesto de sesgos + covarianza), G2 cerrada (tabla de líneas). Aplica §7
del índice.

**Propósito científico**: estimar las propiedades físicas de la fuente que
los datos realmente soportan — tipo espectral, Teff, extinción, luminosidad,
radio, masa condicionada por modelos, Lacc y Mdot — comparando el espectro
canónico con plantillas empíricas y modelos atmosféricos, y declarando cada
parámetro con su etiqueta de dependencia. El objetivo NO es producir el mayor
número de parámetros sino separar lo medido de lo asumido.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: autor del paquete `musepipe/models/` (adaptadores de
bibliotecas + ajuste) y de las etapas `stage_g3_template_fit`,
`stage_g3_atmo_fit`, `stage_g3_accretion`.

**Definición de terminado**:

1. Arquitectura de adaptadores (§3.1): interfaz única para plantillas
   empíricas, rejillas atmosféricas, leyes de extinción y tracks evolutivos;
   al menos UNA implementación concreta de cada tipo, elegida con el humano
   en el checkpoint de §8.1, con cita y versión en config.
2. Ajuste de plantillas (§3.2) y de rejilla atmosférica (§3.3) con χ²
   usando errores de X11 + covarianza G1; mapas de degeneración Teff–A_V
   obligatorios.
3. Cadena de derivación (§3.4): escala → R (con distancia), L_bol, y
   masa/edad por tracks — cada paso con su etiqueta y propagación MC.
4. Acreción multilínea (§3.5): Lacc por CADA línea `detected`/`upper_limit`
   de G2 con relaciones citadas, comparación entre relaciones publicadas como
   sistemático, Mdot con M y R heredados de §3.4.
5. `tables/g3_physical_properties.csv` con el esquema de §5 (una fila por
   propiedad, etiqueta obligatoria); tests §7 verdes.

---

## 1. Límites duros

1. **Etiquetado obligatorio**: toda cantidad lleva
   `direct_measurement | empirical_inference | atmospheric_model_dependent |
   evolutionary_model_dependent | upper_limit | not_constrained`. Masa,
   radio, edad, log g y Mdot JAMÁS como `direct_measurement`. Si el ajuste no
   restringe un parámetro (Δχ²<1 en todo el rango), el resultado ES
   `not_constrained` — prohibido reportar el mejor punto sin intervalo.
2. Patrón "cita obligatoria" de H03 extendido: toda relación empírica, ley de
   extinción, biblioteca de modelos y edad asumida requiere
   `*_citation`/`*_version` en config; su ausencia lanza `RuntimeError`.
3. Los archivos de bibliotecas (BT-Settl u otra, plantillas, isócronas) NO
   entran al repo (tamaño/licencia): directorio externo referenciado en
   config + manifiesto sha256; los tests usan mini-rejillas sintéticas.
4. Distancia, RV sistémica, edad de la región: inputs de config con error y
   cita — nunca constantes en código.
5. G3 no re-mide líneas (consume G2) ni re-extrae (consume X11). Sin cambios
   a módulos previos.
6. Git: rama `phase-g3-inference`.

---

## 2. Precondiciones e inspección obligatoria

1. Leer: auditoría (§4 — clasificación de propiedades), índice G, spec/QC de
   H03 (patrón Lacc–Mdot y citas), QC de X11 (presupuesto de errores,
   unidades absolutas), tabla y QC de G2, presupuesto/covarianza de G1.
2. Verificar la calibración absoluta: factor de flujo de A4/X11 y su error
   (~10% por variabilidad de la primaria) — entra como sistemático en todo
   flujo absoluto de G3.
3. Reutilizar: conversión flujo→L y extinción CCM de `stage_h03_limits.py`
   (extraer a `musepipe/models/extinction.py` SOLO si es sin cambios de
   comportamiento y H03 pasa sus tests idénticos; si no, importar de H03).
4. `pytest` base registrado.

## 3. Operaciones

### 3.1 Arquitectura de adaptadores (`musepipe/models/`)

```python
class SpectralLibrary(Protocol):     # plantillas y rejillas
    def get(self, **params) -> TemplateSpectrum   # wave_A, flux, meta
    def grid(self) -> dict[str, np.ndarray]        # ejes disponibles
class ExtinctionLaw(Protocol):
    def a_lambda_over_av(self, wave_A) -> np.ndarray
class EvolutionaryModel(Protocol):
    def lookup(self, l_bol, age, *, teff=None) -> dict  # masa, radio, logg + errores de interpolación
```

Preparación común: degradar plantilla/modelo a la LSF(λ) de A4, remuestrear a
la malla del espectro (conservando flujo), aplicar extinción, escalar. El
pipeline no puede quedar acoplado a una familia: nada fuera de los
adaptadores conoce el nombre de la biblioteca.

### 3.2 Plantillas empíricas → tipo espectral (`empirical_inference`)

- Biblioteca de M/L jóvenes (elegir con el humano: p. ej. compilaciones tipo
  Luhman/Bonnefoy/Manjavacas; lo que importa es que cubra M0–L5 con clases de
  gravedad juvenil).
- χ² vs plantilla con (A_V, escala) libres por plantilla; ranking; SpT del
  mínimo ± rango de plantillas dentro de Δχ² de confianza.
- Índices espectrales (TiO/VO/Na I en el rango MUSE, definidos por cita) como
  estimador independiente; ambos SpT en la tabla — su discrepancia es un
  sistemático, no se promedia en silencio.
- Enmascarar en el χ² las líneas de emisión detectadas por G2 (la acreción no
  es fotosfera) y las `bad_ranges`.

### 3.3 Rejilla atmosférica → Teff, A_V, escala (`atmospheric_model_dependent`)

- Ejes mínimos: Teff, log g (aunque quede no restringido), A_V, factor de
  escala Ω=(R/d)². χ² con covarianza por bloques de G1.
- Superficies Δχ²(Teff, A_V) y (Teff, log g) OBLIGATORIAS como figuras; los
  intervalos se leen de Δχ² marginalizado con los sistemáticos de X11
  añadidos en cuadratura.
- Si log g no queda restringido (esperable, auditoría §4.4): reportar
  `not_constrained` y fijar el prior de juventud SOLO como variante
  etiquetada.

### 3.4 Cadena derivada

| Cantidad | Vía | Etiqueta |
|---|---|---|
| R | Ω y distancia (Gaia, config+cita) | `atmospheric_model_dependent` |
| L_bol | integración del modelo escalado (+BC del modelo fuera de rango) | `atmospheric_model_dependent` |
| masa, edad, log g(evol) | tracks (elegir: p. ej. familia BHAC/ATMO/BEX) con L_bol (y/o Teff) + edad de la región con error | `evolutionary_model_dependent` |

Propagación: MC end-to-end (semilla en config) muestreando errores de
espectro (con covarianza), distancia, A_V, edad y el sistemático de
calibración absoluta; percentiles 16/50/84 por cantidad. Comparar al menos
DOS familias de tracks; la dispersión entre familias = sistemático de modelo.

### 3.5 Acreción multilínea

- Para cada línea `detected` o `upper_limit` de G2 con relación publicada
  Lacc–L_línea (compilación tipo Alcalá et al. 2017; config con cita por
  línea): L_línea desextinguida (A_V de §3.3 con su error) → Lacc con la
  dispersión de la relación incluida.
- Detecciones: ¿son consistentes entre líneas? (χ² de compatibilidad);
  combinada solo si compatibles, si no `discrepant` con las individuales.
- Límites: el límite combinado es el MÁS RESTRICTIVO individual (no una
  combinación estadística de límites — regla explícita).
- Mdot = 1.25·Lacc·R/(G·M) con R, M de §3.4 y su MC; etiqueta
  `empirical_inference` + dependencias listadas
  (`depends_on: [atmospheric_model, evolutionary_model, lacc_relation]`).
- Consistencia con H03 (que solo usa Hα): misma entrada → mismo resultado
  dentro de redondeo; verificación V5.

## 4. Puntos de atención

1. **Degeneración Teff–A_V**: es EL resultado a comunicar honestamente en el
   óptico; los mapas Δχ² son productos de primera clase, no diagnósticos
   internos.
2. **Continuo contaminado por acreción**: si hay acreción fuerte, el veiling
   sesga SpT/Teff al azul; incluir en §3.2/§3.3 la variante con exceso azul
   libre (potencia/recta) y reportar su efecto como sistemático.
3. **Errores de biblioteca**: paso de rejilla finito → error de interpolación
   (incluir medio paso de rejilla como término); versiones de biblioteca en
   QC (manifiesto sha256).
4. **No sobre-ajustar**: con S/N bajo, plantilla con 2 parámetros libres
   puede "ganar" a física mejor; el ranking reporta Δχ² y n_eff de G1, no
   solo el orden.

## 5. Esquema de salida (`tables/g3_physical_properties.csv` + espejo en QC)

Columnas obligatorias por propiedad:

```text
property, value, err_stat_lo, err_stat_hi, err_sys, unit, label,
data_used, method, assumptions, calibrations_citations, validity_range,
limitations, depends_on, mc_seed
```

`stage_g3_qc.json` añade: bibliotecas usadas (nombre, versión, sha256 del
manifiesto), ejes y rangos de rejilla, mapas de degeneración generados,
resultado de V1–V6, `open_issues`.

## 6. Verificaciones

- **V1 — Recuperación sintética**: espectro sintético generado DESDE la
  propia rejilla (Teff/A_V/escala conocidos) + ruido realista del run →
  parámetros recuperados dentro de 1σ; ídem con plantilla empírica.
- **V2 — Degeneración**: mapas Δχ²(Teff,A_V) y (Teff,log g) con contornos
  1/2/3σ; si el 3σ toca el borde de la rejilla → ampliar rango o declarar
  `not_constrained` en esa dirección.
- **V3 — SpT doble vía**: plantillas vs índices; discrepancia > 2 subtipos →
  issue con hipótesis (veiling, extinción, biblioteca).
- **V4 — Tracks**: masa/edad con DOS familias; figura HRD con la fuente y las
  isócronas de ambas.
- **V5 — Consistencia H03**: Lacc/Mdot vía Hα de G3 == H03 dentro de
  redondeo, con las mismas entradas.
- **V6 — Etiquetas**: test automático — ninguna fila sin etiqueta válida;
  masa/radio/edad/logg/Mdot nunca `direct_measurement`.

## 7. Tests

```text
tests/test_models_adapter_contract.py  # los 3 protocolos con mini-bibliotecas sintéticas incluidas en tests/data
tests/test_models_resample_lsf.py      # degradación LSF + remuestreo conservan flujo (analítico)
tests/test_models_fit_synthetic.py     # V1 automatizado con mini-rejilla 3×3×3
tests/test_models_extinction.py        # ley de extinción vs valores tabulados publicados; cita obligatoria (RuntimeError sin ella)
tests/test_g3_accretion_multiline.py   # Lacc multilínea: consistencia, límite combinado = más restrictivo, Mdot con MC
tests/test_g3_labels.py                # V6: etiquetas válidas y prohibiciones
tests/test_g3_h03_consistency.py       # V5 con entradas fabricadas
```

## 8. Protocolo de parada

1. **Checkpoint obligatorio ANTES de implementar §3.2/§3.3**: proponer al
   humano las bibliotecas concretas (plantillas, rejilla, tracks, relación de
   acreción) con pros/contras y cobertura; no descargar ni acoplar nada sin
   ese OK.
2. Además, preguntar cuando: el mejor χ²_red > 3 (ni modelos ni plantillas
   describen el espectro — posible problema de calibración o de naturaleza);
   V3 discrepe > 2 subtipos; V5 falle; las detecciones de acreción entre
   líneas sean incompatibles (> 3σ); la distancia Gaia sea ambigua para la
   fuente (¡podría no ser compañera! → insumo directo de G4).

## 9. Riesgos

- **Científicos**: degeneración Teff–A_V no comunicada (V2 la fuerza);
  veiling sesgando SpT (§4.2); dependencia de edad de la región dominando la
  masa (el MC la incluye; la tabla la declara); relación Lacc–L_línea
  extrapolada fuera de su rango de calibración (campo `validity_range`
  obligatorio).
- **Técnicos**: bibliotecas pesadas fuera del repo (§1.3, manifiesto);
  interpolación de rejilla (término de error §4.3); costo del MC end-to-end
  (vectorizar, presupuestar).

## 10. Reporte final de Codex

Tabla de propiedades completa comentada fila a fila, mapas de degeneración,
comparación de familias de tracks, Lacc multilínea con figura por línea,
decisiones de biblioteca (checkpoint §8.1 documentado), V1–V6, checklist §7
del índice, issues para G4 (parámetros y ambigüedades que alimentan la
clasificación), comando de reproducción.
