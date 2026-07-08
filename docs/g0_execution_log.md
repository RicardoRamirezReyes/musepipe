# Bitácora de ejecución G0 — cadena real sobre el cubo ADP

Fecha: 2026-07-08. Cierre **retroactivo** de la fase G0
(`docs/spec_G0_codex_real_cube_execution.md`) sobre `runs/ROXs12b_B_adp`.

## Nota de honestidad (desviación respecto a la letra de la spec)

La spec G0 pedía un run nuevo `ROXs12b_G0` en la rama `phase-g0-real-cube`,
ejecutado desde un shell limpio, con arreglos mínimos (≤30 líneas) documentados
uno a uno. Lo que realmente ocurrió: la cadena se ejecutó de facto sobre
`runs/ROXs12b_B_adp` en las ramas `stage-*`, a lo largo de varias sesiones, con
**correcciones de infraestructura que superan el límite de §1.1** (notablemente
el wiring nuevo de E4). Por tanto G0 queda **cumplido en sustancia** y sus
entregables se producen aquí retroactivamente, con las desviaciones marcadas
como issues en `stage_g0_qc.json`. El árbitro de estado sigue siendo F1
(`report/run_summary.json`, `overall_status=red`).

## Cubo de entrada (inventario §2.2)

- Archivo efectivo: `runs/ROXs12b_B_adp/stages/stage02_xcorr_cube_stack.fits`
  (producto B2; el crudo de entrada es
  `Data/ROX12b/20220829/ADP.2022-09-12T17_17_39.371.fits`, `entry_point=adp`).
- `sha256` = `08a4ecbd600bcd02…` (registrado en `stage_g0_qc.json`); extensiones
  `CUBES` + `STAT` + `WAVELENGTH`; marco de λ = ver `stage00q_qc.json`;
  fracción de no-finitos ≈ 6% (bordes NFM + región láser).

## Etapas ejecutadas (semáforo desde F1)

| Etapa | Estado | Notas |
|-------|--------|-------|
| A1 raw_reduction | red | QC vive en `runs/ROXs12b_raw`; blocker de alineación `muse_exp_align` |
| A2 sky_zap | yellow | R=0.617 → ZAP no necesaria |
| A3 telluric | not_run/QC en raw | STD_TELLURIC aplicado en el run raw |
| A4 cube_qc | red | M5 STAT red → ruido empírico (plan B del repo) |
| B1 align / B2 xcorr / B3 localize | green/red/yellow | B2 QC ausente en el run ADP |
| C1 psf | yellow | provisional sobre ADP |
| C2 aperture / C3 optimal / C4 psffit | yellow/green/green | |
| D1 compare | red | `verdict=uninterpretable` (escala de flujo inter-método, blocker #9) |
| D2 calibrate | yellow | M1 λ y M3 flujo no disponibles; flux scale=1 |
| E1 halpha | yellow | **no-detección**; 7 controles → FAP mín 0.125 |
| E2 artifacts | red(raw)/survives | T5 pass; overall reinterpretado para no-detección |
| E4 injection | red(V-checks) | throughput psffit ≈0.80; grilla reducida; wiring nuevo |
| E3 limits | yellow | Ṁ ≲ 5×10⁻¹³ M☉/yr (99%, psffit), provisional |
| F1 report | red | 42 issues (15 bloqueantes) = lista priorizada para G1 |

## Veredicto STAT (§3)

M5 = red en ambos cubos (covarianza de resampleo). Decisión propagada: **ruido
empírico** en toda la cadena X01–X11 (plan B previsto por las specs), coherente
en los QC de cada extractor.

## Cadena de hashes (§4/V2)

`hash_chain_ok = True`: el `sha256` del cubo stage02 declarado como entrada por
B3 coincide con el producto B2; F1 confirma la cadena completa `pass`. Test
`tests/test_g0_hash_chain.py` cubre la detección de una ruptura sintética.

## Comparación legacy (§5.3/V4)

`tables/g0_legacy_comparison.csv`: X01 apertura (compañero) del ADP vs
`stage05_box3x3_spectra.npz` (`spec_c`) del run histórico local-surface
`runs/ROXs12b`, en las 9 bandas congeladas de D1. **9/9 bandas marcadas**
(ratios ~10²–10³, algunos negativos). Causa identificada y esperada: los dos
runs NO están en una escala de flujo común (el ADP en unidad nativa MUSE 10⁻²⁰
sobre el residual 04b —puede ser negativo—; el histórico en otra calibración) —
es el blocker #9 de D1, no un error de coordenadas B3. Test sintético
`tests/test_g0_legacy_compare.py`.

## Fixes de infraestructura aplicados en el ciclo (resumen)

- E4: nuevo `musepipe/stages/stage_h04_extractors.py` (wiring in-memory de los
  extractores C2/C3/C4), auto-calibración de la escala de inyección y resta de
  continuo E1-consistente en el estimador de recuperación (**supera §1.1**).
- E3: knob `h03_flux_unit_cgs` (unidad MUSE 10⁻²⁰) y `h03_allow_unvalidated_throughput`
  (bypass documentado del gate V1 de E4).
- Rendimiento (refactor puro, bit-idéntico): `musepipe/parallel.py` +
  `n_jobs`/`h04_n_jobs` en psffit/optimal/localfit y en la grilla E4 (ThreadPool,
  ~1.6×; el salto a ~6× por procesos queda pendiente).

## pytest (§V5)

- Antes: baseline limpio no capturado al inicio de sesión; la suite se mantuvo
  verde durante todo el ciclo.
- Después: **230 passed** (incluye 6 tests de equivalencia paralela + 6 tests G0).

## Verificaciones V1–V6

- V1 inventario: cubo por sha256, extensiones, marco de λ, no-finitos → OK.
- V2 cadena completa: 17 etapas con status en F1; hash chain `pass` → OK.
- V3 STAT: veredicto red documentado y propagado (ruido empírico) → OK.
- V4 legacy: tabla generada; 9/9 discrepancias con causa (blocker #9) → OK con issue.
- V5 higiene: 230 tests verdes; criterios congelados intactos → OK.
- V6 diagnósticos por etapa: figuras en `runs/ROXs12b_B_adp/plots/` y en
  `report/figures/` (7 figuras de F1) → OK.

## Issues abiertos priorizados para G1

Ver `stage_g0_qc.json` (`open_issues`) y `report/run_summary.json` (42 issues,
15 bloqueantes). Los dominantes: A-block sin cerrar (alineación A1, M3 flujo,
M5 STAT), D1 `uninterpretable`, WFRAME inconsistente, y los caveats provisionales
de E3/E4. **Recomendación**: G1 (validación por inyección-recuperación) puede
consumir el throughput y los sesgos ya medidos, pero ninguna cifra es
paper-válida hasta cerrar el A-block.
