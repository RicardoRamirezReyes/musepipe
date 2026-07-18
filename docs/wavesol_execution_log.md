# Bitácora de ejecución — Track S (wavesol / stripes)

Cierre de `docs/plan_wavesol_stripes_2026-07-17.md` (pasos S0–S8). Ejecutado en
la rama `wavesol-s0b` (PR #1). Árbitro de estado: la **decisión G1**
(`docs/decision_g1_wavesol_2026-07-17.md`), CERRADA y confirmada.

## Resumen en una línea

La solución de longitud de onda del cubo combinado realineado es **estable
espacial y temporalmente**: sin *stripes* de slicer (S0 por-exposición 0/7),
deriva temporal por exposición ~0.04 Å (S3), y el scatter por-spaxel de 0.32 Å
del combinado es **ruido de S/N**, no un sistemático de λ. Ninguna re-reducción
(Fase 2 S4/S5) compraría mejora medible.

## Cronología

| Fecha | Paso | Resultado | Commit |
|---|---|---|---|
| 07-17 | S0a/b | núcleo `wavesol_map.py` + CLI + normalización vectorizada + corte S/N; 14 tests | 398d595 |
| 07-17 | S0c | full-res realineado (p95 0.324 Å, stripe_sig 0.92× vs transv 1.00×) + ADP (0.310, 0.76/0.76); notebook S0 | 398d595 |
| 07-17 | **G1** | **decisión humana:** cerrar como sistemático acotado (temporal); Fase 2 NO disparada | 398d595 |
| 07-18 | S1a/b | `halpha_map.py` + 8 tests; integrado en E2 (`halpha_map_correlation`); a/σ NO alineados con slicers (confundido radial) | 4d137ef / ce70f31 |
| 07-18 | S2a/b/c | 7 cubos por-exposición regenerados (`regen_perexp_cubes.py`, ~130 min medidos) + SKY_SPECTRUM | 1a44d3a |
| 07-18 | S0-perexp | **0/7** exposiciones con estructura de slicer ⇒ stripes DESCARTADOS (rama espacial) | 1ce11a0 |
| 07-18 | S3a/b | deriva temporal ~0.04 Å std (airglow y xcorr estelar consistentes, gate agregado PASS) | 3d6c984 |
| 07-18 | S6a | descomposición de ruido: α=0.78, **Hα a 2.50× el límite fotónico** en r_B (Xie Fig.8) | 2b7fe44 |
| 07-18 | S7a/b/c | nota ILLUM (Xie §3.1); tabla Ṁ vs A_V (×8 de A_V=0 a A_Hα=2); nota Hβ/Hα en el informe | 0e408e1 |
| 07-19 | S2a | plan de regeneración retroactivo (tiempos medidos) | (docs) |
| 07-19 | S8 | cierre: G1 marcado `closed`, notebooks S0/S1, bitácora, F1 refresh, memoria | (este) |

## Desviaciones documentadas (honestidad)

1. **S3b** no corrió el B1/B2 completo (per-stripe-group) — desproporcionado una
   vez que S0-por-exposición descartó los stripes; se hizo xcorr estelar a nivel
   de campo entero (segundo estimador para el gate S3a↔S3b). El matching fino
   por-exposición queda limitado por ruido (señal ~ precisión); el agregado
   (spread ~0.04 Å) es consistente y robusto.
2. **S6a** ajusta el término fotónico sobre el continuo LIMPIO ANCHO (5100–8800,
   F* ×18.7), no la ventana estrecha 6510–6825 del brief (F* ×2 ⇒ α degenerado).
   El QC lleva `regime`/`corr_logF_logS` para exponer el régimen.
3. **S7b** relajación real A_V=0→A_Hα=2 = ×8.0 (no ×6 del brief; la pendiente
   Alcala 1.13 da Ṁ∝ext^1.13).
4. **S4/S5 NO ejecutados**: sin stripes ni deriva grande, la combinación propia
   no compra mejora medible; Fase 2 no disparada por G1. Pieza residual con valor
   independiente: STAT_EMP (varianza empírica inter-exposición) para el M5 rojo —
   los 7 cubos ya existen si se quiere hacer como mini-paso.
5. **Orquestador**: `scratchpad/run_cascade.py` se perdió (scratchpad efímero);
   reconstruido y COMMITEADO como `scripts/regen_perexp_cubes.py`.

## Productos

- Código: `musepipe/qc/wavesol_map.py`, `halpha_map.py`, `noise_decomposition.py`;
  `scripts/{regen_perexp_cubes,s0_perexp,s3_perexp_offsets,s1b_integrate_e2,s7b_mdot_vs_extinction}.py`.
- Notebooks: `notebooks/S0_wavesol_map.ipynb`, `notebooks/S1_halpha_map.ipynb`,
  `notebooks/E2_artifacts.ipynb` (Plot 3).
- QC/tablas/figuras/cubos por-exposición: en `/mnt/2TB` (detrás del symlink del
  run; no versionados) — `runs/ROXs12b_realigned/stages/stageS0..S6*`,
  `tables/{s0_perexp_summary,perexp_m1_offsets,mdot_limit_vs_extinction}.csv`,
  `/mnt/2TB/MUSE_work/ROXs12b_perexp/exp{1..7}/`.
- Registro de decisión: `docs/decision_g1_wavesol_2026-07-17.md` (CERRADO).
