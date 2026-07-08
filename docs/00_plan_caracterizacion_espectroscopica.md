# Plan de caracterización espectroscópica y física — índice de fases G0–G5

Fecha: 2026-07-06. Fuente: `docs/auditoria_caracterizacion_espectroscopica.md`
(2026-07-06). Complementa (no sustituye) el plan maestro
`docs/plan_roxs12_reduccion_multimetodo.md` y sus specs A1–F1. La serie usa la
letra **G** porque A–F ya están asignadas; `spec_F1_codex_final_products.md`
ya existe y NO debe confundirse con la fase 1 de este plan.

---

## 1. Resumen de la auditoría

- Toda la cadena nueva de extracción y detección (A2–A4, B3, E01, X01–X03,
  X10–X11, H01–H04) está **implementada y testeada (82 archivos de test) pero
  jamás ejecutada sobre el cubo ADP real**. `runs/` solo contiene productos de
  la cadena histórica local-surface (stage01–08).
- **No existe ningún módulo de inferencia física**: sin comparación con
  modelos atmosféricos ni plantillas empíricas; sin tipo espectral, Teff,
  log g, A_V ajustada, L, R, masa; sin EW ni ajuste de perfil por línea; sin
  CCF/RV; sin covarianza espectral; sin combinación de épocas; sin Lacc
  multilínea; sin clasificador de naturaleza. `stage08` solo exporta el
  espectro asumiendo modelado externo.
- Riesgos dominantes: PSF AO compartida por los tres extractores (sesgo
  común — la concordancia inter-método NO prueba exactitud), independencia de
  canales asumida, STAT del cubo ADP sin verificar, y A_V/distancia/relación
  Lacc–L_Hα dominando el error de Mdot.

## 2. Estado actual del proyecto

- `entry_point = eso_cube` con el cubo ADP `ADP.2022-09-12T17_17_39.371.fits`;
  A1 (reducción raw) diferida; sin pixtables → ZAP es la única vía de cielo.
- Resultado histórico: no-detección de Hα (FAP global 100%) con local-surface;
  es hipótesis a re-testear con los métodos nuevos, no conclusión.
- Criterios congelados que NINGUNA fase puede alterar sin autorización:
  criterio de detección de E1/H01, bandas de comparación de D1, grilla de E4,
  reglas de veredicto de X10.

### Actualización de estado — 2026-07-07 (post-ejecución real)

La premisa de §1 ("implementada pero jamás ejecutada sobre el cubo real") quedó
**obsoleta**: la cadena completa se ejecutó de facto — bloque A (A1–A4) sobre
crudos en `runs/ROXs12b_raw` (con el blocker de alineación `muse_exp_align`
pendiente) y A4→B3→E01→X01–X03→X10→X11→H01→H02→H04→H03→F1 sobre el ADP en
`runs/ROXs12b_B_adp` (ramas `stage-*`, no `phase-g0-real-cube`). También queda
obsoleto "A1 diferida" en §2. **G0 está cumplido en sustancia pero NO cerrado
formalmente**: faltan sus entregables propios — `docs/g0_execution_log.md`
(bitácora), `stage_g0_qc.json`, la comparación legacy §5.3
(`tables/g0_legacy_comparison.csv`) y los tests `test_g0_*`. El rol de resumen
lo cubre parcialmente F1 (`report/run_summary.json`, overall **red**, 42 issues,
15 bloqueantes = la lista priorizada que G0 §10 pedía para G1).

Desviaciones a criterios congelados, todas con autorización humana explícita y
registradas en QC (`open_issues`): (a) grilla E4 **reducida** (S/N {0,3,5,7} ×
LSF × continuo=none × 4 posiciones; la grilla completa 7×2×2×4 queda para el
redo paper-válido); (b) `overall` de H02 reinterpretado a `survives` para el
caso no-detección (`overall_raw=fails` preservado); (c) wiring de H04 nuevo
(`musepipe/stages/stage_h04_extractors.py`, in-memory, supera las ~30 líneas de
G0 §1.1) con auto-calibración de la escala de inyección y resta de continuo
E1-consistente en el estimador; (d) H03 con `h03_flux_unit_cgs=1e-20` (unidad
nativa MUSE; sin cross-check absoluto M3) y bypass documentado del gate V1 de E4
(`h03_allow_unvalidated_throughput`).

Insumos reales que G1 hereda: throughput psffit **0.80±0.20** al nivel del
límite (caveats: perturbación ±PSF no-op por normalización de flujo, V3/V4
internos de H04 fallan en la posición de borde para aperture/optimal_ls, solo
7 controles → FAP mínima 0.125); X10 `uninterpretable` (escala de flujo
inter-método, blocker #9); WFRAME inconsistente entre productos; resultado
provisional H03: Ṁ ≲ 5×10⁻¹³ M☉/yr (99%, psffit). **Recomendación**: cerrar G0
formalmente (bitácora retroactiva + QC de fase + comparación legacy + tests)
antes de arrancar G1; G2 puede desarrollarse en paralelo como estaba previsto.
Notebook de inspección de la cadena: `A1_F1_run_inspection.ipynb` (solo
lectura, cumple §7/G0 §1.3).

**G1 arrancada (provisional) — 2026-07-08.** Módulo nuevo `musepipe/covariance.py`
(covarianza espectral por lag/bloques desde controles, factor espacial de
resampleo, presupuesto en cuadratura, reglas de veredicto) + 4 tests (16 casos).
Corrida sobre `ROXs12b_B_adp`: `stages/g1_channel_covariance.npz`,
`tables/g1_bias_budget.csv`, `tables/g1_sensitivity.csv`, `stage_g1_qc.json`.
Veredictos: **psffit y optimal_psfsub = `validated_with_bias`** (pérdida de
throughput −27%/−32%, estable y corregida en E3); **aperture y optimal_ls =
`rejected`** (insensibles en la posición de borde del compañero — no bloquea
porque el canónico X11 es psffit). Covarianza espectral: longitud 1.45±0.08
canales, **n_eff/n=0.69** (>0.5, no dispara el gate §8). Inflación espacial del
resampleo: box3≈6.5×, box5≈17× — confirma el M5 rojo (blocker #7) y valida el
plan-B de ruido empírico. Pendientes G1 (provisional): grilla E4 completa (sesgo
de continuo §3.2, perturbación PSF real), 31 controles (§4.3), figuras V2–V4,
impacto en χ² de X10 (§V5, diferido: D1 ya `uninterpretable`). Rama de facto
`stage-f1-report`, no `phase-g1-validation`.

**G2 implementada + corrida (provisional) — 2026-07-08.** Módulos nuevos
`musepipe/constants.py` (c, factores FWHM/σ) y `musepipe/lines.py` (medición
genérica: continuo local, flujo directo + por ajuste gaussiana⊗LSF, EW con signo
emisión-negativa, centroide, FWHM observada e intrínseca deconvuelta, asimetría,
RV, estado detección/límite, errores MC) + stage `stage_g2_measure_lines.py`.
7 archivos de test (22 casos). Cero lógica específica de Hα (catálogo en config;
default = las 24 líneas de stage07). Corrida sobre el espectro canónico X11
`spec_final_object.fits`: **0 detectadas, 23 límites superiores, 1 no medible**
— coherente con la no-detección. **V3 (reconciliación Hα): G2 `upper_limit` vs
H01 `non_detection` = consistente** (dos estadísticas del mismo dato concuerdan).
Throughput 0.768 (E4). Provisional: LSF de config (`h01_lsf_fwhm_A`=2.5, A4 M2
no medida — spec §2.3 pararía; usada con flag), error de calibración λ ausente
(A4 M1), covarianza G1 no aplicada al MC (errores independientes). Pendientes:
covarianza por bloques en el MC, sensibilidad a ventanas §3.5, figuras V4,
conciliación cuantitativa con stage07 (V2).

## 3. Tabla de fases

| Fase | Spec | Contenido | Depende de | Produce para |
|---|---|---|---|---|
| G0 | `spec_G0_codex_real_cube_execution.md` | Ejecutar A4→B3→E01→X01–X03→X10→X11→H04→H01→H02→H03 sobre el cubo ADP real; registrar fallos; NO reescribir | specs A–F implementadas | G1 (productos reales), G2 (espectro canónico) |
| G1 | `spec_G1_codex_extraction_validation.md` | Validación científica de localización, PSF y extractores: inyección-recuperación como árbitro, sesgos, covarianza | G0 | G2, G3 (factores de error), G5 |
| G2 | `spec_G2_codex_spectral_measurements.md` | Infraestructura genérica de medición de líneas y continuo (EW, FWHM, RV, límites) multilínea | G0 (espectro canónico); código en paralelo con G1 | G3, G4 |
| G3 | `spec_G3_codex_physical_inference.md` | Plantillas empíricas, modelos atmosféricos y evolutivos, extinción, acreción multilínea → parámetros físicos etiquetados | G1 + G2 | G4, G5 |
| G4 | `spec_G4_codex_source_classification.md` | Evaluación transparente de hipótesis (planeta / BD / M asociada / M de fondo / contaminante / artefacto) | G2 + G3 (parcial: puede empezar con G2) | G5 |
| G5 | `spec_G5_codex_final_synthesis.md` | Síntesis: tablas finales, covarianzas, trazabilidad completa, reporte reproducible; extiende F1 | Todas | Paquete publicable |

## 4. Orden de ejecución y paralelismo

```text
G0 ──► G1 ──► G3 ──► G4 ──► G5
        ▲      ▲       ▲
G2 (código en paralelo desde el día 1; sus tests son sintéticos)
```

- **G0 es bloqueante** para toda interpretación: ninguna cifra de los
  productos nuevos se interpreta científicamente antes de cerrar G0.
- **G2 puede desarrollarse en paralelo** con G0/G1 porque su validación
  primaria es sintética; su aplicación al espectro real espera al canónico de
  G0 y a los factores de error de G1.
- G3 no arranca hasta que G1 haya emitido el presupuesto de sesgos y
  covarianza (sus χ² lo consumen).
- G4 puede iniciar la parte astrométrica/artefactos con G0+G1, pero el
  veredicto espectral requiere G3.
- G5 es estrictamente final.

**Gate entre fases**: una fase se considera cerrada cuando su spec tiene todas
las verificaciones V* en verde (o issues abiertos aceptados explícitamente por
el humano), sus tests pasan, y su reporte final está entregado. La fase
siguiente lista en su §2 (precondiciones) los productos exactos que exige.

## 5. Criterios globales de aceptación

1. Cero números específicos del target en módulos de `musepipe/`; todo lo
   target-specific vive en `runs/<RUN_ID>/config/config.json`.
2. Todo producto espectral respeta el formato estándar §2.3 del plan maestro
   (`wave_A, flux, flux_err, npix_eff, flags` + `_cov.npz` opcional).
3. Toda etapa nueva cumple el contrato §2.2: QC JSON con parámetros efectivos,
   hashes de entradas, versiones; figuras de verificación; test sintético +
   test de contrato.
4. Toda propiedad física publicada lleva UNA de estas etiquetas:

   ```text
   direct_measurement | empirical_inference | atmospheric_model_dependent |
   evolutionary_model_dependent | upper_limit | not_constrained
   ```

   y declara: datos usados, método, supuestos, calibraciones (con cita),
   incertidumbre estadística, incertidumbre sistemática, rango de validez y
   limitaciones. **Masa, radio, edad, log g y Mdot jamás se reportan como
   `direct_measurement`.**
5. La validación de exactitud es SIEMPRE inyección-recuperación, nunca la
   concordancia entre extractores (comparten PSF).

## 6. Riesgos abiertos (heredados por todas las fases)

1. PSF AO no-Moffat → sesgo común de los tres extractores (mitigación: modelo
   híbrido de C1 + perturbaciones ±10% FWHM en inyecciones).
2. STAT del ADP sin verificar → si A4 la invalida, plan B empírico en toda la
   cadena (previsto; degrada el presupuesto de errores, no bloquea).
3. Covarianza espectral ignorada → χ² y significancias en banda optimistas
   hasta que G1 entregue la matriz empírica.
4. Calibración absoluta anclada a la primaria (joven, variable ~10%).
5. Degeneración Teff–A_V en el óptico para objetos rojos jóvenes (G3 debe
   mapearla, no ocultarla).
6. Posible no-detección del compañero en continuo (B3-§4.3): cambiaría el
   plan de caracterización a búsqueda de línea; G0 debe parar y reportar.

## 7. Reglas obligatorias para Codex (aplican a TODAS las specs G)

Cada spec las incorpora por referencia a esta sección; el reporte final de
cada fase incluye la checklist marcada.

1. Inspeccionar antes de editar: leer la auditoría completa, esta página, la
   spec de la fase y las specs A–F relacionadas ANTES de tocar código.
2. No duplicar funciones existentes (buscar en `musepipe/` primero); no
   reemplazar código funcional sin demostrar necesidad; no cambiar interfaces
   públicas sin justificación escrita en el reporte.
3. No mezclar refactorizaciones amplias con cambios científicos en un mismo
   commit/rama.
4. Sin rutas absolutas; sin constantes científicas ocultas (toda constante con
   nombre, unidad y cita en config o módulo de constantes); sin silenciar
   errores con defaults (patrón del repo: `RuntimeError` con mensaje claro).
5. No suponer que STAT representa la varianza sin la verificación de A4; no
   suponer independencia entre canales donde hubo interpolación o sustracción
   de PSF; no aceptar la concordancia entre extractores como prueba de
   exactitud — la inyección-recuperación es la validación principal.
6. Registrar toda configuración efectiva junto a los resultados (QC JSON);
   mantener la estructura `runs/<RUN_ID>/{config,stages,tables,plots,logs}`.
7. `pytest` completo antes y después de cada cambio; documentar los tests que
   YA fallaban al inicio (con su traza) sin arreglarlos silenciosamente.
8. Cambios pequeños y verificables; cada fase queda en estado ejecutable antes
   de avanzar; protocolo de parada de cada spec es vinculante.

## 8. Instrucciones de entrega a Codex

- Entregar UNA spec por sesión de trabajo, en orden de dependencias (§4),
  junto con: esta página, la auditoría, y el plan maestro.
- Prompt mínimo: "Implementa/ejecuta la spec `spec_GX_codex_*.md` del repo.
  Respeta §7 de `00_plan_caracterizacion_espectroscopica.md`. Trabaja en la
  rama indicada en sus Límites duros. Detente en las condiciones de su
  protocolo de parada."
- El humano revisa el reporte final de la fase y las verificaciones V* antes
  de autorizar la fase siguiente (gate de §4).
- Si Codex propone desviarse de una spec, la desviación se documenta como
  issue en el QC y se pregunta; jamás se implementa primero.
