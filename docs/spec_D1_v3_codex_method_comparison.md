# Especificación D1 v3 · `X10_method_comparison` — 6 métodos (familia espectral)

Fecha: 2026-07-14. Revisión formal de
`docs/spec_D1_v2_codex_method_comparison.md` (2026-07-09), que queda como
histórico vigente para los QC con `spec_version="D1_v2"`. Motivada por el plan
`docs/2026-07-15_plan_integracion_halosub.md` (WP-H3): incorporar los métodos de
diversidad espectral C5 (`sgf`) y C6 (`lpm`) de Julo et al. 2025 a la
comparación inter-método.

**Protocolo anti cherry-picking (heredado, intacto):** esta v3 se escribe y
commitea ANTES de re-ejecutar D1 sobre datos reales con 6 métodos. El QC
registra `spec_version="D1_v3"`. Cualquier cambio posterior de umbrales,
bandas o reglas = v4 con justificación escrita.

## 0. Qué cambia respecto a v2 (y nada más)

1. **Set de métodos**: `METHOD_ORDER = (aperture, optimal_ls, optimal_psfsub,
   psffit, sgf, lpm)`. D1 v3 exige los 6 productos congelados; runs
   históricos sin C5/C6 conservan su QC v2 sin re-interpretación.
2. **Árbol de recomendación extendido** (§2): criterios de régimen del paper
   para `recommended_method`, con salvaguardas para `sgf`.
3. **Insumos nuevos del QC** (§3): predictor de auto-sustracción del QC C5,
   flag `companion_continuum_is_science`, caveats por método.

**Re-congelado idéntico a v2, sin excepciones:** bandas B1–B6/LHα/LHβ/LOI
(§2 v2), convención de escala y scale-check (§3.1 v2), insumos G1 y pares
primarios = validados por G1 (§3.2 v2), corrección de throughput en memoria
(§3.3 v2), estadístico t centrado con controles (§3.4 v2), gates por par
(§4.2 v2), reglas de veredicto y umbrales p_div=0.0455 / p_strong=0.0027
(§5 v2), salidas (§6 v2), verificaciones V1–V7 (§7 v2). Los métodos sgf/lpm
entran en pie de igualdad: primarios si y solo si G1 los valida
(`validated`/`validated_with_bias`); su throughput E4 se corrige en memoria
como el resto. El veredicto `absent` de G1 (métodos sin filas E4) cuenta como
no-validado (secundario), nunca como `rejected`.

## 1. Entradas nuevas

- `stages/spec_sgf_object.fits` + `spec_sgf_controls.npz` (C5).
- `stages/spec_lpm_object.fits` + `spec_lpm_controls.npz` (C6).
- `stages/spec_sgf_qc.json` (predictor Ec. 1 por línea; opcional: si falta,
  caveat + open_issue, sin veto).
- Config: `companion_continuum_is_science` (bool, default `true`),
  `x10_sgf_predictor_max` (default 0.10, congelado).

Ambos productos deben cumplir el contrato congelado (misma malla λ del cubo
madre, headers `BKGMODE`/`SCALEREF`, controles calibrados) — la validación de
productos y el gate de controles v2 aplican sin cambios.

## 2. Árbol de recomendación (congelado)

`recommended_method` se emite SOLO con veredicto `consistent` (igual que v2);
en cualquier otro veredicto es `null` y decide el usuario en el checkpoint con
la matriz completa + caveats. La elección del canónico sigue siendo del
usuario (límite duro v2 §1.5, intacto).

Reglas, en orden:

1. **Candidatos** = métodos con verdict G1 ∈ {validated, validated_with_bias}.
   Sin candidatos → `recommended_method = null` + open_issue.
2. **Exclusión de `sgf`** del candidato a recomendación si:
   (a) `companion_continuum_is_science = true` (pérdida irrecuperable de
   información de continuo, paper §2.1: L_S ≠ L̂_S + C̃_S), o
   (b) el predictor Ec. 1 de alguna línea de ciencia en rango satisface
   `|C̃_P/L̂_P| > x10_sgf_predictor_max` (auto-sustracción no despreciable).
   La exclusión se registra en `method_caveats.sgf` (no es rechazo G1: sigue
   siendo primario para el veredicto si G1 lo validó).
3. **Preferencia congelada** entre los candidatos restantes:
   `psffit > lpm > optimal_psfsub > sgf > aperture > optimal_ls`.
   Racional registrado: psffit ataca el halo con modelo espacial simultáneo y
   preserva continuo (canónico vigente); lpm es el mejor de la familia
   espectral (preserva líneas y continuo vecino, paper §3.4–3.5) y aplica
   donde el modelo espacial es débil; optimal_psfsub validado; sgf solo como
   línea base; aperture/optimal_ls históricamente insensibles en el borde.

`method_caveats` (QC, informativo, siempre presente):

- `sgf`: predictores Ec. 1 por línea (o "unavailable"), exclusión y causa.
- `lpm`: nota de colinealidad (componente del espectro planetario colineal
  con ŝ se absorbe; óptimo para compañeras dominadas por líneas, §4.1).
- `companion_continuum_is_science`: valor usado.

## 3. QC (delta sobre v2 §6)

`stage_x10_qc.json` añade: `method_caveats`, `recommendation_rules`
(candidatos, exclusiones aplicadas, orden de preferencia) y los 6 métodos en
`products`/`throughput_correction.by_method`. El CSV maestro no cambia de
esquema (más filas por los pares nuevos: 15 pares con 6 métodos).

## 4. Divergencias esperadas y su tratamiento

- La sistemática B6 (rojo lejano) presupuestada en la decisión 2026-07-10
  se hereda: si los pares nuevos (p.ej. psffit vs lpm) reproducen la misma
  divergencia B6, NO reabre la iteración C1 — se documenta contra la misma
  decisión. Solo la contradicción (familia espectral consistente donde la
  espacial diverge por encima de p_strong, o viceversa) escala al checkpoint.
- `sgf` divergente en bandas de línea contra lpm/psffit es el RESULTADO
  ESPERADO del paper (auto-sustracción), no un fallo del gate: se reporta en
  `verdict_by_pair` como cualquier par, y el veredicto global sigue gobernado
  por las reglas v2 §5 sobre pares primarios limpios.

## 5. D2 y F1 (cambios mecánicos)

- `stage_x11_calibrate` calibra los 6 métodos (`also_calibrated` incluye
  sgf/lpm) y acepta `sgf`/`lpm` como canónico si el usuario lo fija.
- `musepipe/report.py` (F1) enumera los 6 productos calibrados; los ausentes
  se omiten con placeholder (comportamiento existente).

## 6. Tests

Adaptados (mismos caminos v2, ahora con 6 métodos): `test_compare_stage`
(escritura de 6 productos/controles), `test_calibrate_stage` (calibra 6).
Nuevos: recomendación v3 (consistent + todos validados → psffit; psffit no
validado y lpm validado → lpm; sgf único validado con continuo-ciencia →
null + caveat), exclusión de sgf por predictor sobre umbral.

## 7. Protocolo de parada y reporte

Idéntico a v2 §9. El checkpoint final presenta: veredicto, matriz t/p de los
15 pares (primarios y secundarios), caveats por método, y espera la elección
del usuario para D2.
