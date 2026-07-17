# Checkpoint D1 → D2 · Elección del método canónico

Fecha: 2026-07-09. Run: `runs/ROXs12b_B_adp` (provisional; A-block abierto).
Decisión tomada por el usuario en el checkpoint final de D1 v2
(`docs/spec_D1_v2_codex_method_comparison.md` §9). Esta es la elección humana
que la spec reserva al usuario: D1 emite `recommended_method`, nunca fija el
canónico.

## Decisión

**Método canónico = `psffit`.**

Registrado en `runs/ROXs12b_B_adp/config/config.json` como
`x11_canonical_method: "psffit"` — que es lo que D2
(`stage_x11_calibrate._resolve_canonical_method`) consume con prioridad sobre
el QC de D1.

## Contexto del veredicto D1 v2

Ejecución sobre 33 controles (`stages/stage_x10_qc.json`,
`spec_version=D1_v2`):

- **Veredicto = `divergent_continuum`**, par primario `psffit vs optimal_psfsub`
  (los dos métodos validados por G1). Acción del árbol congelado:
  `iterate_C1_refine_PSF_before_PCA`.
- La divergencia es **real, no de convención**: tras la reconciliación de
  escala (blocker #9 cerrado), el scale-check da `level_ratio = 1.0015` y el
  gate de controles queda limpio (1.7% de filas malas vs 4.6% esperado). El
  patrón espectral: B6 rojo lejano (8600–9100 Å) con t = +4.5 (p < 0.0027,
  fuerte) y B2 marginal (t = −2.8); todas las líneas (Hα, Hβ, O I) consistentes.

## Justificación de elegir psffit pese a `divergent_continuum`

1. **Validado por G1** (`validated_with_bias`, throughput T = 0.726 medido y
   corregido aguas abajo en E3), igual que optimal_psfsub; aperture y
   optimal_ls fueron `rejected` (insensibles en el borde del compañero).
2. **Positivo y físico en el borde**: ajusta simultáneamente estrella y
   compañera sobre el modelo de PSF, manejando el gradiente de halo sin la
   sobre-sustracción que arrastra la rama local-surface (optimal_ls).
3. **La divergencia B6 apunta a C1, no al método**: el compañero es muy rojo y
   el modelo Moffat de C1 no ajusta bien el halo AO justo en las bandas rojas
   (blocker #8, trabajo Psfao pendiente de integrar en C1). La acción correcta
   del árbol es iterar C1 con la PSF física, no descartar psffit ni saltar a
   PCA.

## Estatus y trabajo pendiente

- La divergencia de continuo B6 queda como **incertidumbre sistemática
  abierta** hasta iterar C1 (integrar `stage_e01_psfao` / `maoppy.Psfao` en la
  ruta de C1 con comparación Moffat-vs-Psfao). Cuando C1 se refine, re-ejecutar
  C2/C3/C4 → D1 v2 y re-evaluar; si B6 persiste consistente, el veredicto podría
  bajar a `consistent`.
- Todo es **provisional**: ninguna cifra es paper-válida hasta cerrar el
  A-block (ver `docs/plan_roxs12_reduccion_multimetodo.md` y el inventario de
  blockers). psffit se adopta como canónico de trabajo para que D2/E1–E3
  puedan seguir, con la reserva anterior explícita.

## Actualización 2026-07-10 — C1 Psfao consolidado; B6 aceptado como sistemática presupuestada

El blocker #8 (integrar Psfao en la ruta canónica de C1) se **cerró** en WP-5:
`stage_e01_psf.py` ahora ajusta Moffat **y** Psfao por bin, aplica la métrica
canónica del anillo del compañero a ambas formas, y selecciona por regla de la
spec (menor residuo mediano; empate→Moffat). Bloque `model_comparison` en
`stage_e01_qc.json`.

- **Resultado en el cubo ADP (auto):** forma elegida = **psfao** (residuo
  mediano del anillo 4.99 % vs 18.99 % de Moffat). El `psf_model.json`
  consolidado es **byte-idéntico** al que producía la etapa lateral
  `stage_e01_psfao` (param_table Δrel = 0.00e+00): la consolidación es **neutra**.
- **D1 v2 re-ejecutado** (C1→C2→C3→C4→D1 completo, 2026-07-10): veredicto
  **sigue `divergent_continuum`** (B6 rojo lejano), idéntico campo a campo al
  previo. Es decir: **refinar la PSF con el modelo físico Psfao NO cierra la
  divergencia B6**; el residuo de continuo del bin rojo lejano persiste con la
  PSF física.

**Decisión del usuario (2026-07-10):** dado que la ruta física ya está agotada
en C1, la divergencia de continuo B6 se **acepta como error sistemático
presupuestado** (no se itera C1 con híbrido más agresivo ni se salta a PCA por
ahora). `psffit` permanece como canónico de trabajo, con la reserva explícita y
provisional (nada es paper-válido hasta cerrar el A-block). Opciones abiertas si
en el futuro se decide reducir esta sistemática: (a) híbrido azimutal más
agresivo en C1, (b) sustracción de residuos por PCA. El
`action=iterate_C1_refine_PSF_before_PCA` que emite el QC de D1 queda
**reconocido pero no accionado** por esta decisión.

## Actualización 2026-07-15 — D1 v3 (6 métodos): psffit se mantiene como canónico

Con la integración Julo et al. 2025 (plan
`docs/plan_integracion_halosub_julo2025.md`), D1 v3 corrió sobre
`ROXs12b_realigned` con 6 métodos. Hechos del checkpoint:

- **G1 a 6 métodos**: validados = {psffit (T=0.66), sgf (0.42), lpm (0.44)};
  `optimal_psfsub` pasó a **rejected** (T=0.25 < 0.4). Causa verificada: su
  T=0.667 histórico venía de un E4 anterior a la consolidación Psfao de C1
  (2026-07-10 18:48); con el modelo C1 vigente no es bias-bounded en el borde
  del compañero. Verificado con worktree HEAD que el cambio NO proviene del
  código nuevo (caso de control byte-comparable).
- **Veredicto D1 v3 = `divergent_continuum`** con 3 pares primarios limpios
  (psffit–sgf, psffit–lpm, sgf–lpm; 33 controles, df=32). Estructura B6:
  psffit vs familia espectral t≈+14, lpm vs sgf t=−3.1 — la familia espectral
  es ~consistente donde la espacial diverge. Afila el diagnóstico de B6 hacia
  el residuo de halo rojo de psffit, con la salvedad de que sgf/lpm comparten
  ŝ (sesgo común posible). `recommended_method=None` (regla congelada);
  elegibles registrados: {psffit, lpm}.

**Decisión del usuario (2026-07-15): `psffit` SE MANTIENE como canónico de
trabajo.** B6 sigue como sistemática presupuestada (decisión 2026-07-10,
heredada por la v3). D2 re-corrido con el comparador de continuo corregido
(lpm, no sgf — errata spec D2 v1.1): acuerdo control-referenciado 0.683,
ratio banda roja −5.15x, mismo carácter de limitación aceptada que en F1.
Todo sigue provisional hasta cerrar el A-block.

## Reproducción (era D1 v3, run realineado)

```bash
conda activate MUSE
# C5/C6 (sustracción de halo espectral; productos + cubos residuales):
python -m musepipe.stages.stage_x04_sgf --run-id ROXs12b_realigned
python -m musepipe.stages.stage_x05_lpm --run-id ROXs12b_realigned
# E4 (throughput 6 métodos) y G1 (veredictos por método):
bash scripts/stage_h04_injection.sh --run-id ROXs12b_realigned --allow-long-run
python scripts/run_g1.py --run-id ROXs12b_realigned
# D1 v3 (veredicto + recommended_method + caveats):
python -m musepipe.stages.stage_x10_compare --run-id ROXs12b_realigned
# D2 consume el canónico fijado en config (x11_canonical_method=psffit):
python -m musepipe.stages.stage_x11_calibrate --run-id ROXs12b_realigned
```
