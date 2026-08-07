# Especificación G5 · `final_synthesis` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-06. Fase G5 (cierre) del plan
`docs/2026-07-08_plan_caracterizacion_espectroscopica.md`. Prerrequisitos: G0–G4
cerradas (o con issues aceptados explícitamente por el humano). Aplica §7 del
índice. **Extiende la maquinaria de F1 (`musepipe/report.py`,
`scripts/build_report.py`, spec_F1) — no la reemplaza ni la duplica.**

**Propósito científico**: consolidar la caracterización completa en un
paquete final reproducible, legible por máquina y con trazabilidad total:
espectros, tablas de líneas y propiedades, covarianzas, límites, comparación
de modelos, clasificación, y el resumen honesto de supuestos y limitaciones.
Como en F1: **G5 no calcula ciencia nueva** — si algo falta, es un hueco de
la fase correspondiente y se reporta como issue, no se parcha aquí.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: ensamblador y auditor de trazabilidad. Amplía el paquete
`report/` de F1 con la sección de caracterización (G1–G4).

**Definición de terminado**:

1. `runs/<RUN_ID>/report/characterization/` con las tablas de §4, figuras de
   §3, `characterization_summary.json` y `characterization.md` generado desde
   plantilla (prohibido editarlo a mano — patrón F1).
2. Checklist de trazabilidad §2 en verde: cadena de hashes desde el cubo ADP
   hasta cada número publicado, sin rupturas.
3. Determinismo demostrado: dos ejecuciones → mismos archivos (patrón F1).
4. `musepipe/report.py` extendido (funciones nuevas, interfaces existentes
   intactas) + `scripts/build_characterization.py`; tests §7.
5. Los tests de determinismo/trazabilidad de F1 existentes siguen verdes.

---

## 1. Límites duros

1. Solo lectura de productos y QCs de G0–G4 y de las etapas A–H; G5 no
   re-ejecuta etapas ni recalcula cantidades (la única aritmética permitida:
   agregación/formateo y verificación de consistencia).
2. El set de figuras y tablas es el de §3–§4, fijo; extras van a
   `report/characterization/extra/`, nunca mezclados.
3. Sin embellecimiento: los datos se muestran con sus flags y etiquetas tal
   como los produjeron G2–G4; una propiedad `not_constrained` aparece como
   tal, no se omite.
4. Compatibilidad total con `run_summary.json` y el `report.md` de F1: G5
   añade claves/secciones, no renombra ni reestructura las existentes.
5. Archivos legibles por máquina en formatos del repo: CSV para tablas, JSON
   para metadatos, FITS BinTable para espectros, NPZ para covarianzas.
6. Git: rama `phase-g5-synthesis`.

---

## 2. Precondiciones, inspección obligatoria y checklist de trazabilidad

1. Leer: auditoría, índice G, spec_F1 completa y `musepipe/report.py` (qué
   recolecta ya y cómo encadena hashes), QCs y reportes finales de G0–G4.
2. Verificar precondiciones: QCs de fase `stage_g0_qc.json` …
   `stage_g4_qc.json` presentes; paquete F1 del run generado y verde.
3. Checklist de trazabilidad (gate, extiende la de F1): para CADA número de
   las tablas de §4 debe poder reconstruirse la cadena

   ```text
   cubo ADP (sha256) → config del run (hash) → versión de código (git) →
   método de extracción → calibraciones (QC A4/X11) → máscaras y ventanas →
   covarianza usada (G1) → modelos/bibliotecas (nombre, versión, sha256 del
   manifiesto, citas) → semillas MC → parámetros adoptados (distancia, edad,
   A_V, RV con citas) → valor final
   ```

   implementada como verificación automática que recorre los QCs (los campos
   ya existen por contrato de fases previas; si alguno falta → issue
   bloqueante contra esa fase, no rellenar aquí).

## 3. Figuras (formato publicación, generadas desde productos)

Añadidas a las 7 de F1, sin tocarlas:

- **G5-1**: espectro final calibrado con banda de error total y flags,
  con los espectros de los tres métodos superpuestos en panel inferior.
- **G5-2**: grid de líneas medidas (G2) — dato, perfil, estado por línea.
- **G5-3**: mejor plantilla y mejor modelo atmosférico sobre el espectro,
  con residuos; recuadro con mapa de degeneración Teff–A_V (G3-V2).
- **G5-4**: HRD/isócronas con la fuente y las dos familias de tracks (G3-V4).
- **G5-5**: Lacc por línea (detecciones y límites) con la combinada (G3).
- **G5-6**: throughput/completitud de inyección por método (G1/H04).
- **G5-7**: heatmap de la matriz de evidencia de clasificación (G4-V2).

## 4. Tablas maestras (CSV + espejo en `characterization_summary.json`)

1. `final_spectra_index.csv`: espectros publicados (final + por método) con
   archivo, sha256, método, calibraciones aplicadas, covarianza asociada.
2. `final_line_table.csv`: la tabla G2 completa (todas las líneas, todos los
   estados, límites con throughput aplicado).
3. `final_physical_properties.csv`: la tabla G3 íntegra — columnas de esquema
   G3-§5 incluyendo `label`, `assumptions`, `citations`, `err_stat`,
   `err_sys`, `validity_range`, `limitations`. Ninguna fila filtrada.
4. `adopted_parameters.csv`: parámetros adoptados de entrada (distancia,
   edad, A_V del sistema, RV sistémica, relaciones y bibliotecas elegidas)
   con valor, error, cita y fase que los consumió.
5. `final_classification.csv`: ranking G4 con robustez, pruebas dominantes,
   P(fondo) y la evidencia en contra del líder.
6. `uncertainty_budget.csv`: presupuesto consolidado estadístico vs
   sistemático por resultado principal (espectro, Hα, Lacc, Mdot, SpT/Teff),
   agregando X11 + G1 + G3.
7. `assumptions_and_limitations.md` (generado): lista numerada de supuestos
   activos (STAT, independencia residual de canales donde aplique, PSF,
   variabilidad de la primaria, edad de la región…) con la fase que los
   introdujo y su efecto estimado.

## 5. Esquema de `stage_g5_qc.json`

```json
{
  "stage": "g5_final_synthesis",
  "run_id": "...",
  "inputs": {"phase_qc_hashes": {"g0": "...", "g1": "...", "g2": "...",
                                    "g3": "...", "g4": "..."}},
  "traceability": {"complete": true, "broken_chains": []},
  "tables_generated": [], "figures_generated": [],
  "f1_compatibility": {"run_summary_extended": true, "f1_tests_green": true},
  "determinism_hash": "...",
  "open_issues": []
}
```

## 6. Verificaciones

- **V1 — Trazabilidad**: la verificación automática §2.3 recorre todas las
  cadenas sin ruptura; tabla de cadenas en el QC.
- **V2 — Consistencia interna**: los números repetidos entre productos
  coinciden (Hα en `final_line_table` vs H01/H03 vs G3; SpT en propiedades vs
  clasificación) — discrepancia = issue bloqueante contra la fase de origen.
- **V3 — Etiquetas**: ninguna propiedad sin etiqueta válida; masa/radio/
  edad/log g/Mdot nunca `direct_measurement` (re-verificación final).
- **V4 — Determinismo**: dos ejecuciones → hashes idénticos (salvo el campo
  único de timestamp, patrón F1).
- **V5 — Legibilidad por máquina**: todos los CSV parsean con esquema
  declarado; JSON validado contra su esquema; FITS con headers completos.
- **V6 — F1 intacto**: paquete F1 regenerado idéntico tras integrar G5.

## 7. Tests

```text
tests/test_g5_traceability.py     # cadena de hashes sintética completa y con ruptura → detectada
tests/test_g5_consistency.py      # V2 con QCs fabricados coherentes/incoherentes
tests/test_g5_labels_final.py     # V3 sobre tabla fabricada
tests/test_g5_determinism.py      # V4 (patrón del test de F1, reutilizar utilidades)
tests/test_g5_f1_compat.py        # V6: extensión no rompe claves/secciones de F1
```

## 8. Protocolo de parada

Preguntar cuando: una cadena de trazabilidad esté rota (falta un hash/cita en
una fase previa); V2 encuentre inconsistencias entre fases; alguna fase tenga
issues bloqueantes abiertos no aceptados; el determinismo falle por fuentes
de aleatoriedad no sembradas aguas arriba.

## 9. Riesgos

- **Científicos**: presentar como cerrado lo que tiene issues abiertos (la
  checklist y `assumptions_and_limitations.md` lo impiden); pérdida de matiz
  al resumir (por eso las tablas van íntegras, sin filtrar).
- **Técnicos**: acoplamiento frágil a claves de QCs de cinco fases (tests de
  contrato §7); romper F1 al extender `report.py` (V6).

## 10. Reporte final de Codex

Inventario del paquete completo, resultado de la verificación de
trazabilidad, V1–V6 con evidencia, lista de issues heredados y su estado
(aceptado/pendiente), checklist §7 del índice, y el comando único que
regenera todo el paquete desde los productos.
