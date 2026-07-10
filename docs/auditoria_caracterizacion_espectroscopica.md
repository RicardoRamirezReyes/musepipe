# Auditoría: capacidades para la caracterización espectroscópica completa de la fuente tenue

Fecha: 2026-07-06. Base: código en `musepipe/` (23.4k líneas, 82 archivos de test),
plan `plan_roxs12_reduccion_multimetodo.md`, 14 specs Codex en `docs/`, y productos
en `runs/`.

Hallazgo central: **el pipeline de extracción y detección de Hα está implementado
y testeado, pero ninguna etapa nueva (A2–A4, B3, C1–C4, D1–D2, E1–E4) se ha
ejecutado aún sobre el cubo real** — en `runs/` solo hay productos de la cadena
histórica local-surface (stage01–08). Además, **no existe ningún módulo de
inferencia física** (comparación con modelos atmosféricos, tipo espectral, Teff,
masa): el diseño asume que el modelado ocurre fuera del pipeline
(`stage08_full_spectrum_for_modeling` solo exporta el espectro).

---

## 1. Capacidades ya implementadas correctamente (código + tests)

| Capacidad pedida | Módulo | Estado |
|---|---|---|
| Localización precisa en el cubo | `stages/stage01c_localize.py` (B3): centroide de la primaria por canal, matched filter para el compañero, deriva cromática | Implementada, 4 tests, **no ejecutada en datos reales** |
| Posición y separación respecto de la estrella | B3: sep/PA vs predicción orbital (Bowler+17), coordenadas canónicas en QC | Implementada |
| Extracción por apertura | `extraction/aperture.py` + `stage_x01` con corrección de apertura por curva de crecimiento | Implementada, tests de apcorr |
| Extracción óptima (Horne) | `extraction/optimal.py` + `stage_x02`: pesos P²/V, clipping, varianza analítica | Implementada, tests de sesgo de PSF y normalización |
| Ajuste de PSF (deblending estrella+compañero) | `extraction/psffit.py` + `stage_x03`: ajuste lineal por canal con fondo, χ², condición del sistema | Implementada, tests de crosstalk/conditioning |
| Modelo de PSF cromática | `psf.py` + `stage_e01`: Moffat elíptico por bin, suavizado en λ, modo híbrido (Moffat + residuo empírico) | Implementada, tests sintéticos y de seguridad del híbrido |
| Separación del halo estelar | Doble vía: superficie local (04b, validada) y deblending PSF (x03); métrica de residuo en el radio del compañero | Implementada |
| Comparación entre métodos | `stage_x10_compare`: χ² par a par, bandas de continuo y línea, veredicto (`consistent`/`divergent_*`), gate con controles | Implementada, 5 tests |
| Calibración espectral y flujo | `stage_x11_calibrate`: offset λ (skylines A4), marco baricéntrico verificado, factor de flujo (fotometría sintética), apcorr sin doble aplicación, presupuesto de errores (PSF, cielo, telúricas) | Implementada, 6 tests |
| Corrección telúrica y de cielo | `reduction/telluric.py` (molecfit-like, A3), `reduction/sky_zap.py` (ZAP, A2), con decisión automática y verificación | Implementadas |
| QC de calibración del cubo | `qc/cube_qc.py` (A4): λ vs skylines, LSF(λ), factor STAT, fotometría sintética vs literatura (passbands Gaia incluidos) | Implementada |
| Detección de Hα y significancia | `stage_h01_detect`: matched filter en velocidad (±500 km/s), z-score empírico contra controles, criterio predefinido con consistencia de RV; look-elsewhere en `stage08c` | Implementada, validación histórica (C4: FAP calibrado con 31 controles) |
| Pruebas de artefactos | `stage_h02_artifacts`: skylines/stripes, coherencia espacial, sub-stacks, líneas placebo | Implementada |
| Límites superiores y Lacc/Mdot (vía Hα) | `stage_h03_limits`: límite 5σ desde controles corregido por throughput medido, A_V con error, ley CCM con cita obligatoria, relación Lacc–L_Hα con cita obligatoria, `mdot_err_dex` propagado | Implementada |
| Inyección-recuperación multi-método | `injection.py` + `stage_h04`: inyección con PSF medida, propagación por cualquier extractor | Implementada |
| Productos reproducibles | `report.py` + `scripts/build_report.py`: hashes encadenados, determinismo testeado, trazabilidad | Implementada |

Fortaleza notable: el diseño estadístico es conservador y anti-p-hacking
(criterio de detección fijado antes de mirar, controles al mismo radio,
placebos, regla del máximo entre error STAT y empírico).

## 2. Capacidades parciales o exploratorias

- **Medición de líneas (stage07)**: mide flujo integrado, pico, S/N y continuo
  para ~25 líneas de acreción (Balmer, He I, Ca II IRT, Paschen, prohibidas),
  pero **no calcula ancho equivalente ni FWHM por línea** — FWHM, centroide y
  velocidad solo existen en H01 (parametrizado por `rest_A`, así que es
  generalizable, pero hoy solo se usa para Hα).
- **Velocidad radial**: solo centroide de línea individual (H01). El CCF con
  templates ("resto de C7") está explícitamente diferido en el plan §4.
- **Covarianza espectral**: el formato `spec_*_cov.npz` está definido en el
  contrato §2.3 del plan, pero ningún módulo la calcula; los canales se tratan
  como independientes (supuesto reconocido en plan §1.2.4).
- **Incertidumbres**: propagación STAT + validación empírica implementadas,
  pero la covarianza espacial inducida por el resampleo de alineación
  (subestimación ~10–30% en aperturas pequeñas, plan B1-R) está anotada, no
  cuantificada.
- **Combinación de exposiciones**: existe dentro de un run (apilado stage01,
  selección best-4). **No existe** combinación de espectros entre épocas, runs
  o métodos (más allá del veredicto de X10, que compara pero no combina).
- **HRSDI/PCA + forward modeling (C5/C6)**: solo la maquinaria PCA histórica de
  los notebooks 04/06c; los módulos `hrsdi.py` y `forward_model.py` del plan no
  existen (son opcionales por diseño).
- **Reducción desde raws (A1)**: `esorex_driver.py` existe con tests, pero la
  operación actual usa el cubo ADP (`entry_point=eso_cube`); A1 diferida — sin
  pixtables, ZAP es la única herramienta de cielo.

## 3. Herramientas que faltan por completo

1. **Comparación con modelos atmosféricos y plantillas empíricas** (BT-Settl,
   plantillas M/L jóvenes, índices espectrales): no hay ningún módulo, spec ni
   test. Es la brecha mayor frente al objetivo de caracterización.
2. **Estimación de tipo espectral, Teff, log g, A_V, L, R**: inexistente; hoy
   A_V y distancia son *inputs* de config tomados de literatura, no productos.
3. **Masa** (modelos evolutivos: DUSTY, BEX, ATMO...): inexistente.
4. **Lacc multilínea**: H03 solo convierte Hα; stage07 mide las demás líneas de
   acreción pero nada las convierte a Lacc (relaciones Alcalá+17 por línea) ni
   combina estimadores.
5. **Ancho equivalente** y ajuste de perfil (gaussiano/lorentziano) por línea.
6. **Clasificación de la naturaleza de la fuente** (planeta / enana marrón /
   M de fondo / contaminante): no hay herramienta formal. B3 aporta la
   consistencia con la órbita y H02 descarta artefactos, pero faltan: test de
   movimiento propio común (con épocas/astrometría de archivo), colores
   sintéticos vs secuencias de campo, y densidad de contaminantes esperada
   (modelo tipo Besançon/TRILEGAL) para la probabilidad de fondo.
7. **Combinador de espectros** inter-época/inter-método con pesos por varianza.
8. **RV por CCF** contra templates.

## 4. Propiedades físicas: qué se puede calcular hoy y con qué respaldo

### 4.1 Directamente medidas de los datos (una vez ejecutada la cadena x/h)
- Posición, separación y PA del compañero (+ deriva cromática).
- Espectro 5000–9300 Å con errores calibrados y presupuesto de sistemáticos.
- Flujos de línea y de continuo; S/N; límites superiores 5σ con throughput medido.
- Centroide, FWHM y velocidad de líneas detectadas (resolución MUSE ~120 km/s:
  la RV será útil a nivel de decenas de km/s, suficiente para descartar fondo
  cinemático extremo, no para órbitas).

### 4.2 Inferidas por relaciones empíricas (requieren módulos nuevos)
- L_Hα → Lacc (implementado solo para Hα, con cita obligatoria) → **Mdot**
  (requiere M y R, hoy inputs de config: dependencia oculta de modelos).
- Tipo espectral por índices/plantillas (falta módulo).
- Lacc multilínea (falta módulo).

### 4.3 Dependientes de modelos atmosféricos/evolutivos (faltan por completo)
- Teff, log g, A_V simultáneos (ajuste de rejilla; fuerte degeneración
  Teff–A_V en el óptico para objetos rojos jóvenes).
- Radio (escala del flujo con distancia Gaia), luminosidad bolométrica (BC).
- Masa (tracks evolutivos; sensible a edad asumida de Ophiuchus).

### 4.4 No determinables de forma confiable con estos datos
- log g con precisión mejor que ~0.5 dex (resolución y S/N de MUSE en un
  compañero tenue); v·sin i (muy por debajo de la resolución); campos
  magnéticos; variabilidad (una sola época); masa dinámica.

## 5. Supuestos que podrían sesgar resultados

1. **PSF AO no-Moffat** (riesgo dominante declarado, plan §5.1): un residuo
   estructurado en el radio del compañero sesga b(λ) del PSF-fitting y el
   throughput de la extracción óptima. Mitigación existente: modo híbrido +
   perturbaciones ±10% en E4 — hay que ejecutarla, no solo tenerla.
2. **Independencia de canales**: sin covarianza espectral, todo χ² entre
   métodos y toda significancia integrada en banda están algo sobreestimados.
3. **A_V, distancia y elección de la relación Lacc–L_Hα** dominan el error de
   Mdot (reconocido en H03 vía `mdot_err_dex`, pero la elección de relación es
   única, sin comparación entre relaciones publicadas).
4. **Calibración absoluta anclada a la primaria**, estrella joven variable
   (~10%): sesgo sistemático de flujo y por tanto de Lacc.
5. **Cubo ADP sin STAT verificado y sin pixtables** (A1 diferida): si A4
   encuentra STAT inutilizable, todo cae al plan B empírico (previsto, pero
   degrada el presupuesto de errores) y no hay re-reducción de cielo posible.
6. **Coordenadas heredadas** de la cadena histórica hasta que B3 corra sobre el
   cubo real (el plan lo identifica: hoy hardcodeadas en config).
7. **Concordancia entre métodos no garantiza exactitud**: los tres extractores
   comparten el mismo modelo de PSF (sesgo común); el árbitro es la inyección.

## 6. Validaciones necesarias antes de interpretar

- Ejecutar **E4 (inyección-recuperación) a través de x01/x02/x03 en el cubo
  real**, incluida la regresión contra la validación histórica C1 (−1.9% a S/N 5).
- Perturbaciones de PSF (±10% FWHM) para acotar el sistemático de modelo.
- Líneas placebo y sub-stacks (H02) sobre los espectros nuevos.
- Verificación STAT vs varianza empírica (A4) en el cubo ADP.
- Inyección de una fuente sintética con espectro completo (continuo + líneas)
  para validar el pipeline de caracterización de extremo a extremo, no solo Hα.
- Cuantificar (una vez) el factor de covarianza espacial del resampleo (plan B1-R).

## 7. Plan de trabajo propuesto

### Fase 0 — Ejecutar lo ya construido (bloqueante; nada de interpretación antes)
1. A4 (QC del cubo ADP: λ, LSF, STAT, flujo) → A2 condicional → A3 opcional.
2. B3 (localización reproducible; retira coordenadas hardcodeadas).
3. E01 (PSF cromática) → X01, X02, X03 → X10 (veredicto inter-método) → X11
   (espectro canónico calibrado con presupuesto de errores).
4. H04 (inyección multi-método) → H01 (detección Hα con criterio prefijado) →
   H02 (artefactos) → H03 (límites o flujos) → F1 (reporte).

### Fase 1 — Completar medición de líneas y continuo
5. Extender `line_metrics`/nuevo `musepipe/lines.py`: ancho equivalente,
   ajuste de perfil (gauss + LSF), FWHM y velocidad por línea con errores por
   Monte Carlo usando `flux_err`; aplicar a las ~25 líneas de stage07 sobre el
   espectro canónico. Tests sintéticos con verdad conocida.
6. Generalizar H01 a las demás líneas de acreción (ya es paramétrico en
   `rest_A`): detecciones/límites por línea con el mismo control de FAP.

### Fase 2 — Incertidumbres y combinación
7. Covarianza espectral empírica desde controles (matriz por bandas) y factor
   de covarianza espacial del resampleo; integrarlos en X10/X11 y en los χ².
8. Módulo combinador de espectros (pesos 1/σ², chequeo de consistencia previo)
   para exposiciones/épocas/métodos concordantes.

### Fase 3 — Inferencia física (módulos nuevos, specs tipo Codex)
9. `musepipe/models/`: ajuste de rejilla BT-Settl (+ plantillas empíricas
   M/L jóvenes, p.ej. Luhman/Bonnefoy) con (Teff, A_V, escala R²/d²) por
   χ² con covarianza; degeneraciones mapeadas explícitamente; tipo espectral
   por índices + mejor plantilla.
10. Lacc multilínea (relaciones Alcalá+17 por línea, comparación entre
    relaciones publicadas como sistemático) y Mdot con M, R heredados de la
    fase 3.9 + tracks evolutivos (con edad y su incertidumbre como input
    explícito y citado, mismo patrón de "cita obligatoria" de H03).
11. RV por CCF contra la mejor plantilla (cierra C7).

### Fase 4 — Clasificación de la naturaleza
12. Módulo de clasificación: consistencia orbital (B3 multi-época si hay datos
    de archivo), RV vs sistema, gravedad/juventud espectral (índices sensibles
    a log g), densidad esperada de contaminantes de fondo → verosimilitudes
    relativas planeta/BD/M de fondo, con el mismo criterio prefijado
    anti-sesgo que E1.

### Fase 5 — Cierre
13. Inyección end-to-end de un compañero sintético con espectro físico
    (validación de fases 1–4) y extensión de F1 con las tablas/figuras de
    caracterización.
14. (Cuando se instale esorex) A1 + A4 comparativo ADP vs re-reducido.

Regla transversal: cada módulo nuevo sigue el contrato de etapa (§2.2 del
plan: QC JSON, hashes, test sintético + test de contrato) y toda propiedad
publicada declara su categoría (§4 de este documento) y sus citas.
