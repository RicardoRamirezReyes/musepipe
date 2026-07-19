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
| 07-18 | **R1** | límite Ṁ dual: Alcalá (headline, ×0 cambio) + **Aoyama+21 planetario ×10.5 más débil**; coefs cross-check vs arXiv:2108.01277 | 0c155cd |
| 07-18 | R2 | caveat de variabilidad (1 época 2022-09-01, 7 exp ~87 min) en el informe (Cody&Hillenbrand14, Hashimoto+20 §5.4) | ef04d9b |
| 07-18 | R3 | censo de ghosts en B: **line_ghost=none, fringing=none** (tras restar halo radial AO; el naïve daba +9σ falso) | d8247f0 |
| 07-18 | R5 | frame-QC de las 7 exp: **7/7 utilizables, 0 descartadas** (FWHM 11-13px, fondo 33-50) | 45a1861 |
| 07-18 | **R4** | STAT_EMP inter-exposición: DRS subestima la varianza **×2.9 (fondo, bias-corr ×3.3) a ×8.6 (ponderado)** ⇒ M5 acotado | 98a805a |

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

## Lote R — re-auditoría 2026-07-18 (post-merge PR #1)

Cinco brechas restantes tras comparar Xie+20 / Hashimoto+20 contra el estado
post-cierre. Todas ejecutadas (ramas `wavesol-r1`..`r5`, suite 584 verde).
Detalle por paso en `docs/plan_wavesol_stripes_pasos_agente.md` (sección Lote R).

- **R1** (sustantivo, cambia el número del paper): límite de Ṁ bajo DOS
  calibraciones — Alcalá+2017 (estelar, headline, INTACTO 8.19e-13) y **Aoyama+21
  planetario 8.64e-12 (×10.5 más débil)**. Coefs (0.95, 1.61) cross-checkeados
  contra el full-text de arXiv:2108.01277 (species no instalado). Claves E3
  `mdot_aoyama21_*` aditivas, columnas S7b, sección "Accretion Relation" en el
  informe.
- **R2**: caveat de variabilidad (1 época) en el informe.
- **R3**: censo de ghosts en B ⇒ **limpio**. `ghost_census.py` + 9 tests.
- **R4** (ataca M5): STAT_EMP inter-exposición. El DRS subestima la varianza
  por-vóxel **×2.9 (fondo) a ×8.6 (ponderado)** ⇒ M5 "~4-6× estimado" **medido y
  acotado**. Nota M5 del informe actualizada.
- **R5**: frame-QC ⇒ **7/7 utilizables, 0 descartadas**.

### Desviaciones documentadas del Lote R

6. **R3 (2 desviaciones)**: (a) resta del halo radial AO de la primaria antes del
   test de tira — el método naïve fila/columna daba +9σ falso (la columna de B
   cruza la primaria); (b) fringing detectado por fracción-de-varianza del modo
   dominante (>0.15), no por "pico>5×mediana" (piso de ruido blanco ~1.4·ln M ≈ 8).
7. **R4 (crítica)**: los 7 `DATACUBE_FINAL` **NO están en grid vóxel-común**
   (NAXIS distintos, dither ~20-26px), contra la premisa del brief. Verificado que
   son North-up (scipost absorbió ABSROT); separación = pura traslación ⇒ alineo
   por **shift entero** (sin interpolación → sin nueva covarianza; ≤0.4px residual
   ⇒ límite superior). Gate WCS relajado a sub-canal en CRVAL3 (deriva λ de S3).
   Dos estadísticos: mediana por-canal (vóxel típico) y pooled (ponderado, límite
   superior).

### Productos del Lote R

- Código: `musepipe/qc/{ghost_census,frame_qc,stat_emp}.py` (+ tests
  `test_{ghost_census,frame_qc,stat_emp}.py`, +22); ampliaciones en
  `stage_h03_limits.py` (dual Aoyama), `report.py` (3 notas + nota M5 medida),
  `scripts/s7b_mdot_vs_extinction.py`.
- QC/tablas/figuras (no versionados, tras el symlink del run):
  `stageR3_ghost_census_qc.json`, `stageR4_stat_emp_qc.json`,
  `stageR5_frame_qc.json`, `tables/{perexp_frame_qc,stat_emp_ratio_by_channel}.csv`,
  `plots/{r3_ghost,r4_stat_emp}/`, y `STAT_EMP.fits` (285×290×3681) en
  `/mnt/2TB/MUSE_work/ROXs12b_perexp/`.
- Config: `g3_lacc_relations.halpha_aoyama21` en el run realineado.
