# Definición del límite superior de Ṁ — E3 (h03) vs G3 (g3_accretion)

> WP-7 del plan `docs/plan_agente_correcciones_revision_2026-07-10.md`.
> Estado (2026-07-10): **ambas definiciones documentadas; NO se elige canónica
> todavía** (decisión del usuario diferida). No se ha tocado código ni re-corrido
> ninguna etapa; solo se añadió una nota de referencia cruzada en cada QC.

Las etapas E3 (`stage_h03_upper_limits`) y G3 (`stage_g3_accretion`) reportan un
límite superior de la tasa de acreción Ṁ para ROXs 12 B, pero con **definiciones
distintas**. La cadena física (relación L_acc→Ṁ, Alcalá et al. 2017 Hα, mismos
inputs físicos) es **idéntica para entradas idénticas**; la diferencia es
puramente de definición.

## Comparación (valores del run `ROXs12b_realigned`, método canónico `psffit`)

| Aspecto | E3 (`stage_h03_qc.json`) | G3 (`stage_g3_qc.json`) |
|---|---|---|
| Entrada estadística | **Gumbel 99 %** de los máximos de ruido de los controles (matched filter, `f_stat_99`) | **5σ** del flujo de línea Hα de G2 |
| Factor R_in (1.25) | **NO aplicado** | **Aplicado** — Ṁ = L_acc·R/(G·M)·(1−R/R_in) |
| Relación L_acc→Ṁ | Alcalá 2017 Hα | Alcalá 2017 Hα (idéntica) |
| Valor Ṁ (M☉/yr) | **8.2×10⁻¹³** | **1.3×10⁻¹²** |
| También calcula | `f_stat_5sigma_extrap` = 3.07×10⁻¹⁶ (variante 5σ) | MC n=2000, línea más restrictiva |
| Fuente | `runs/ROXs12b_realigned/stages/stage_h03_qc.json` | `runs/ROXs12b_realigned/stages/stage_g3_qc.json` |

El desfase ~1.6× es la combinación de (a) umbral estadístico más estricto en E3
(Gumbel 99 % vs 5σ) y (b) el factor de truncamiento de disco 1.25·R_in que solo
G3 aplica. Cada definición es internamente coherente.

## Estado y próximo paso

- **No hay definición canónica declarada.** Cuando el usuario decida, la etapa
  no elegida debe (i) mantener/actualizar su `limit_definition_note` apuntando a
  la canónica y, si el usuario lo pide, (ii) alinear su código (mismo umbral +
  factor R_in) y re-correrse para reportar el mismo número.
- Referencia cruzada añadida (2026-07-10) en el campo `limit_definition_note` de
  `stage_h03_qc.json` y `stage_g3_qc.json` de `ROXs12b_realigned` y
  `ROXs12b_B_adp`. **Ese campo se regenera al re-correr la etapa**; este documento
  es el registro durable.
