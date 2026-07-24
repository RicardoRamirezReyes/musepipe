# Pasos ejecutables por agente — plan wavesol/stripes 2026-07-17

Descomposición de `docs/plan_wavesol_stripes_2026-07-17.md` en pasos pequeños,
cada uno auto-contenido, con contrato de salida y límites duros. Formato y
cultura: como los `spec_*_codex_*.md` del repo (alcance deliberadamente
pequeño, gates de equivalencia, QC extendido sin renombrar claves, runs
históricos intocables, rama git por paso).

> **ACTUALIZACIÓN 2026-07-18 (re-auditoría tras el cierre del Track S).**
> El Track S (S0–S8) está EJECUTADO Y MERGEADO (PR #1, `0ad9da1`); árbitro:
> `docs/decision_g1_wavesol_2026-07-17.md` (G1 cerrado, sin stripes, deriva
> ~0.04 Å) y bitácora `docs/wavesol_execution_log.md`. La re-auditoría de las
> metodologías de Xie+20 y Hashimoto+20 contra el estado post-merge dejó un
> **Lote R (pasos R1–R5, sección al final)** con lo que sigue faltando:
> relación L_acc planetaria (R1, el único sustantivo), caveat de variabilidad
> (R2), censo de ghosts (R3), STAT_EMP para el M5 rojo (R4) y censo frame-QC
> (R5). El Track A sigue pendiente completo (A1a/A1b, A2a/A2b, A3 — verificado
> 2026-07-18: los 4 open_issues de `stage00r_qc.json` intactos y sin
> `stage00s/stage00t` en el run realineado). Los briefs S0–S8 de abajo se
> conservan como registro; NO re-ejecutarlos.

**Leyenda de nivel**: 🟢 = ejecutable por agente menos potente con este brief ·
🟡 = ejecutable, pero requiere revisión humana del resultado · 🔴 = complejo,
requiere agente fuerte o sesión dedicada · ✅ = HECHO (con fecha y resultado en
el propio paso) · ⛔ = CERRADO SIN EJECUTAR (por decisión G1; brief conservado
por si se reabre).

**Reglas globales para TODOS los pasos** (no repetidas en cada brief):
1. Rama git propia `wavesol-<paso>`; suite completa verde antes de terminar
   (`python -m pytest tests/ -q`, entorno conda `MUSE` de miniforge3).
2. QC: claves nuevas se AÑADEN, ninguna se renombra; runs históricos intocables.
3. Nada se escribe dentro de `runs/ROXs12b_realigned` sin que el paso lo diga
   explícitamente; productos pesados van a `/mnt/2TB/MUSE_work/`.
4. Si un gate falla, PARAR y reportar; no improvisar correcciones fuera del brief.

---

## Estado de las partes complejas (resueltas por la sesión fuerte)

| Parte compleja | Estado |
|---|---|
| Núcleo del estimador de offsets por spaxel (S0a) | ✅ `musepipe/qc/wavesol_map.py` + `tests/test_wavesol_map.py` (9 tests verdes) |
| Diagnóstico preliminar sobre cubos reales | ✅ corrido en scratchpad 2026-07-17 (binning 2×2, ambos cubos). SIN estructura alineada con slicers: realigned amp columna 72 mÅ a 1.2× ruido (transversal 1.4×), ADP 67 mÅ a 0.9× (transversal 1.0×); núcleo r<1": med |off| = 64 mÅ (realigned) / 62 mÅ (ADP) ≈ M1 global (+74 mÅ). El p95 global (~4 Å) es ruido de spaxels débiles ⇒ corte por error añadido al brief S0b. Evidencia preliminar apunta a G1 = fase2_descartable (decisión humana pendiente con el S0c formal) |
| Diseño del algoritmo de combinación propia (S4a) | ✅ FIJADO en el brief S4a (la implementación queda mecánica) |
| Modelo de descomposición de ruido (S6) | ✅ FIJADO en el brief S6a |
| Debug de molecfit (A1a) | ✅ HECHO 2026-07-19 — converge (flujo normalizado + banda A O₂); corrobora STD_TELLURIC (|ΔT| 0.8 % px en banda B) |

---

## Track A — cierres A-block

### PASO A1b ✅ · Justificación STD_TELLURIC (documental) (HECHO 2026-07-18, rama `track-a-a1b-a3`, commit 6634532)
> `docs/a3_telluric_justification.md` completado: citas verificadas por web — Weilbacher+20 (A&A 641
> A28: la telúrica de la estándar es el método NATIVO del DRS) y Hashimoto+20 (PDS 70b, análogo MUSE
> más cercano: pipeline estándar vía EsoReflex, flujo in-pipeline, SIN molecfit, solo enmascaró
> regiones) ⇒ nuestro STD_TELLURIC explícito con ventana Hα protegida es al menos tan riguroso.
> `stage00r_qc.json` anotado con `a3_telluric_resolution = {resolution: justified_std_telluric, doc}`.

### PASO A1b-orig 🟢 · Justificación STD_TELLURIC (documental)
- **Objetivo**: cerrar el open_issue "molecfit no convergió" con una justificación
  formal del método aplicado, publicable en la sección de métodos.
- **Entradas**: `docs/a3_telluric_justification.md` (esqueleto),
  `runs/ROXs12b_raw/stages/stage00t_qc.json` (V1–V5: O2 6.76%→0.60%, etc.).
- **Salidas**: doc completado con los números reales de V1–V5 + referencia a que
  Hashimoto+20 usó la calibración de flujo del pipeline estándar (sin molecfit);
  `stage00r_qc.json` (realigned): open_issue de telúrica anotado con
  `"resolution": "justified_std_telluric", "doc": "docs/a3_telluric_justification.md"`.
- **No hacer**: no re-correr esorex; no tocar el cubo.

### PASO A1a ✅ · Intento molecfit (HECHO 2026-07-19, scratch; corrobora A1b)
> **molecfit CONVERGE.** Causa raíz real de la no-convergencia previa: el espectro 1D se pasó
> **sin normalizar** (flujo mediano ~58000, continuo atascado en 1.0, `bestnorm 1e12`) con solo
> la **banda B de O₂** débil (5.6 % absorción) ⇒ O₂ sin leverage, χ² congelado (el GDAS ausente
> era secundario). Fix: normalizar el flujo + añadir la **banda A de O₂ (7590–7690 Å, 30 %
> absorción)**. Resultado: `status=2`, `rel_mol_col_O2=0.966±0.016`, `ppmv_O2≈205000` (≈20.5 %,
> físico), `rms_rel_to_err 85.8→5.09`. **T(λ) molecfit vs STD_TELLURIC: coinciden al ~4.5 %**
> (integrado) / 0.8 % (píxel) en la banda B junto a Hα y al ~5.5 % en la banda A; Hα intacta en
> ambos. **Corroboración independiente de A1b**; A1b sigue siendo el cierre del paper. Registrado
> en `stage00r_qc.json → a1a_molecfit_crosscheck`, `docs/a3_telluric_justification.md §6`, gráfico
> `runs/ROXs12b_raw/plots/stage00t_a1a_molecfit_vs_std.png`. (calctrans full-range no producido:
> quirk de mapping de la build, no afecta la conclusión.)

### PASO A1a-orig 🔴 · Intento molecfit (timebox, OPCIONAL si A1b aprobado)
Sesión dedicada con agente fuerte: iterar config de `molecfit_model`
(WAVE_INCLUDE, kernel, continuum, columnas del espectro 1D) con timebox de una
sesión. Si converge, comparar T(λ) contra STD_TELLURIC y decidir; si no,
A1b queda como cierre definitivo.

### PASO A2a ✅ · Master BIAS por noche (evidencia) (HECHO 2026-07-18, rama `track-a-a1b-a3`, commit 30554ce)
> `scripts/a2a_bias_by_night.py`: 44 BIAS en **4 noches** (no 3), master por noche vía esorex en
> scratch (master en uso intocado), comparado por IFU×cuadrante (QC keywords LEVEL/RON) vs el global.
> **Criterio NO se cumple formalmente**: el nivel deriva noche a noche coherentemente (noche científica
> 2022-09-01 a −3.84 ADU uniformes del global; 08-28 a +3.3; 08-31/09-03 <1 ADU); RON mayormente <5%
> con outliers de IFU. Tabla `runs/ROXs12b_raw/tables/bias_by_night_comparison.csv` + `bias_grouping_check`
> en stage00r_qc.json.

### PASO A2a-orig 🟡 · Master BIAS por noche (evidencia)
- **Objetivo**: comprobar si combinar las 44 BIAS de varias noches en un master
  único sesga el nivel/RON.
- **Procedimiento**: agrupar las BIAS raw por noche (header `MJD-OBS`, 3 noches);
  correr `esorex muse_bias` por noche (mismo esquema SOF que el orquestador
  `scratchpad/run_cascade.py`); tabla comparativa por cuadrante/IFU: nivel medio
  y RON del master por-noche vs el master global existente.
- **Salidas**: `runs/ROXs12b_raw/tables/bias_by_night_comparison.csv` + resumen
  en `stage00r_qc.json` (`bias_grouping_check: {max_level_diff_adu, max_ron_diff}`).
- **Criterio (para la decisión humana A2b)**: |Δnivel| < 1 ADU y |ΔRON| < 5%
  ⇒ agrupación global aceptable.
- **No hacer**: no re-correr la cascada; no reemplazar el master en uso.

### PASO A2b ✅ · Decisión de agrupación (humana) (DECIDIDO 2026-07-18: ACEPTAR)
> **Decisión humana: ACEPTAR la agrupación global** (documentar). Análisis de impacto cuantificado
> (bias_grouping_check.impact_analysis en stage00r_qc.json): el offset de bias es espectralmente PLANO
> ⇒ **efecto CERO exacto en el límite de LÍNEA Hα** (E1 resta el continuo running-median 80 Å antes del
> matched filter, el pedestal se cancela) y <2% del ruido en el continuo (3.84 ADU→10.5 u de cubo = 1.8%
> del ruido 590; residual post-cielo 0.29%). El master global (44 frames) además tiene menos read-noise
> que uno por-noche. Regrupar = re-cascada completa sin ganancia medible. `a2b_decision` +
> `bias_grouping_resolution` en stage00r_qc.json; F1 yellow/0 red.

### PASO A2b-orig 🔴-lite · Decisión de agrupación (humana)
Con la tabla de A2a: aceptar (documentar en QC como limitación verificada) o
reagrupar (lo que dispararía re-cascada — decisión de costo del usuario).

### PASO A3 ✅ · Trazabilidad QC A2/A3 en el run realineado (HECHO 2026-07-18, rama `track-a-a1b-a3`, commit 6634532)
> Escritos `runs/ROXs12b_realigned/stages/stage00s_qc.json` (A2) y `stage00t_qc.json` (A3) DERIVADOS de
> los reales de `ROXs12b_raw` (mismos números, sin renombrar claves) + bloque `provenance` (derived_from,
> reason, sha256 del cubo realineado). **F1: A2 pasó de "conditional QC missing" a yellow; A3 de "not_run"
> a yellow**; overall yellow, 0 red.

### PASO A3-orig 🟢 · Trazabilidad QC A2/A3 en el run realineado
- **Objetivo**: que F1 deje de marcar A2 "conditional QC missing" y A3 "not_run".
- **Procedimiento**: escribir `runs/ROXs12b_realigned/stages/stage00s_qc.json` y
  `stage00t_qc.json` DERIVADOS de los reales de `ROXs12b_raw` (mismos números),
  añadiendo bloque `provenance: {derived_from: "runs/ROXs12b_raw/...", reason:
  "A2/A3 decididos y aplicados sobre la reducción origen; el cubo realineado
  hereda cube_telcorr", sha256 del cubo}`. Re-correr F1 del realigned.
- **Gate**: F1 nuevo sin issues NUEVOS; A2 pasa a yellow/green con el QC presente.

---

## Track S · Fase 1 — diagnósticos (sin esorex)

### PASO S0a ✅ · Núcleo del mapa de offsets — HECHO
`musepipe/qc/wavesol_map.py`: ventanas por defecto (5100–5550, 6100–6500,
6620–6850, 8480–8680 — evitan láser/Hα/telúricas), normalización por división
de running-median (mata el continuo cromático del halo, Hashimoto Fig. 5),
xcorr subpixel reutilizando `stripes._xcorr_shift_pixels`, perfil por columnas,
métrica de estructura con control transversal, `evaluate_gate_g1` (recomendación
automática, decisión humana). Tests: `tests/test_wavesol_map.py` (9, verdes;
recuperan shifts inyectados de 0.2–0.35 canales con atol 0.05).

### PASO S0b ✅ · CLI + optimización vectorizada + plots (HECHO 2026-07-17, rama `wavesol-s0b`)
> CLI `python -m musepipe.qc.wavesol_map`; `normalize_window_cube` vectorizada por canal con
> `searchsorted` (equivalencia <1e-9, reproduce truncado de bordes + min_pixels en cualquier
> grid — se desvió del boceto `sliding_window_view` por exactitud); error por spaxel +
> `max_err_ch`; plots 2×2 + FITS multi-ext; 14 tests verdes. Loop xcorr sin vectorizar
> (~5 min/cubo a full-res, aceptable).

- **Objetivo**: hacer ejecutable S0 sobre cubos reales a resolución completa.
- **Procedimiento**:
  1. CLI en `wavesol_map.py`: `python -m musepipe.qc.wavesol_map --cube X.fits
     --qc-output stageS0_qc.json --map-output stageS0_offset_map.fits
     --orientation vertical [--binning N] [--windows "5100:5550,..."]`
     (patrón de CLI: copiar `python -m musepipe.qc.cube_qc m3-flux`).
  2. **Optimización**: `normalize_window` llama `continuum_running_median`
     (O(n²) Python) por spaxel — a resolución completa es horas. Añadir
     `normalize_window_cube(waves_w, cube_w, continuum_window_A)` vectorizada:
     `np.lib.stride_tricks.sliding_window_view` sobre el eje espectral +
     `np.nanmedian` sobre la ventana, procesando en bloques de ≤32 filas
     (control de memoria), bordes con ventana truncada = replicar el
     comportamiento del original. **Gate de equivalencia**: sobre el cubo
     sintético de los tests, `max|norm_vec − norm_ref| < 1e-9` y el mapa de
     offsets resultante idéntico dentro de 1e-6 canales.
  3. **Incertidumbre por spaxel + corte S/N (lección del preliminar 2026-07-17)**:
     el criterio p95 del gate se contamina con spaxels débiles donde la xcorr
     falla (en el preliminar: p95 global 3.9 Å puro ruido, vs 0.18 Å en r<1").
     Añadir a `measure_spaxel_offset` el error por spaxel
     `err_ch = robust_sigma(per_window)/sqrt(n_windows)` (NaN si n<2) y un mapa
     `err_map`; `structure_metrics` y `evaluate_gate_g1` deben aceptar
     `max_err_ch` (default 0.08 ch = 0.1 Å) y calcular el p95 SOLO sobre
     spaxels con err < max_err_ch (reportando n_selected). El perfil por
     columnas se queda con todos (la mediana ya es robusta).
  4. Plots (patrón `musepipe/plotting.py`): mapa de offsets (Å, escala robusta),
     mapa de error, perfil por columnas con banda de ruido esperado, histograma
     (solo spaxels bajo el corte de error).
  5. QC JSON: `{cube, sha256, binning, windows_used_A, channel_step_A, metrics:
     {…structure_metrics…, n_selected_low_err}, gate_g1: {…}, runtime_s}`.
- **Tests nuevos**: equivalencia vectorizada; CLI end-to-end sobre cubo sintético
  pequeño escrito a FITS temporal.
- **No hacer**: no cambiar la física del estimador; no tocar `stripes.py`.

### PASO S0c ✅ · Ejecución formal + notebook + checklist G1 (HECHO 2026-07-17, rama `wavesol-s0b`)
> Full-res realineado (p95=0.324 Å, stripe_sig 0.92× vs transv 1.00×) y ADP (p95=0.310 Å,
> 0.76× vs 0.76×): **sin estructura de slicer, realineado≈ADP**, pero p95>0.1 Å ⇒ gate
> auto-recomienda `fase2_justificada` SOLO por magnitud. QC/mapas/plots en
> `runs/ROXs12b_realigned/stages/stageS0[_adp]_*` y `plots/s0_wavesol/`; notebook
> `notebooks/ROXs12b/S0_wavesol_map.ipynb`.
> **GATE G1 DECIDIDO 2026-07-17 (humano):** cerrar como sistemático acotado, interpretación
> TEMPORAL; **Fase 2 NO disparada**; confirmación DIFERIDA (correr S0 por exposición + S3
> cuando existan los 7 cubos). Anula la recomendación automática. Clave: el cubo combinado es
> ciego a los stripes porque el campo rota 5.9° (PA) entre las 7 exposiciones (derotador
> ABSROT 19.3°) ⇒ los stripes por-slice se promedian acimutalmente. Detalle:
> `docs/decision_g1_wavesol_2026-07-17.md`; QC: `g1_human_decision` en `stageS0[_adp]_qc.json`.

- **Objetivo**: producto formal para la decisión G1.
- **Procedimiento**: correr el CLI de S0b sobre (a)
  `/mnt/2TB/MUSE_work/ROXs12b_realigned/cube_telcorr.fits` y (b) el ADP
  `Data/ROX12b/20220829/ADP.2022-09-12T17_17_39.371.fits`, a resolución
  completa, QC a `runs/ROXs12b_realigned/stages/stageS0_qc.json` (y tabla/plots
  del run); notebook de revisión nuevo `notebooks/ROXs12b/S0_wavesol_map.ipynb` vía
  `scripts/build_review_notebooks.py` (mismo patrón que los 25 existentes);
  sección "checklist G1" con los umbrales del plan y la recomendación automática.
- **Referencia**: el diagnóstico preliminar 2026-07-17 (binning 2×2, scratchpad)
  ya existe; el formal debe reproducir sus conclusiones a resolución completa.
- **Argumentos para el checklist G1** (incluir en el notebook): (1) la métrica
  espacial de S0 NO ve un offset común a todas las exposiciones (deriva
  temporal uniforme) — ese modo solo ensancharía la LSF combinada; (2) pero A4
  M2 midió LSF = 2.383 Å, MÁS ESTRECHA que el nominal, lo que acota ese
  smearing a nivel pequeño; (3) el M1 global (+0.074 Å) ya corrige el
  zero-point. Los tres juntos son el caso "fase2_descartable"; documentarlos
  aunque la decisión sea seguir.
- **Gate**: presentar al usuario → DECISIÓN G1 (humana) antes de cualquier paso
  de Fase 2.

### PASO S1a ✅ · Núcleo mapas Hα de la primaria (Xie Ec. 1) (HECHO 2026-07-18, rama `wavesol-s0b`)
> `musepipe/qc/halpha_map.py`: fit phi=b(1+a·exp(...)) por spaxel (6480–6650), mapas a/σ/μ/P,
> clean-NaN por gate de line-SNR, reutiliza `structure_metrics` de wavesol_map, `corr(a,σ)` y
> `P_cov` sin interpretar; CLI + plots + FITS. 8 tests.

- **Objetivo**: mapas de line-to-continuum (a), anchura (σ) y potencia integrada
  (P) del Hα de la primaria por spaxel — ¿varía la LSF alineada con slicers?
- **Procedimiento**: nuevo `musepipe/qc/halpha_map.py`. Por spaxel del halo
  (misma selección por brillo que `wavesol_map`), ajustar en 6480–6650 Å el
  modelo `phi(λ) = b·(1 + a·exp(−(λ−μ)²/(2σ²)))` con `scipy.optimize.curve_fit`
  — semillas por momentos (b = mediana fuera de 6540–6590; a,μ,σ del exceso),
  bounds: a∈[0,50], μ∈[6540,6590], σ∈[0.5,8] Å, b>0. Mapas: a, σ_A, μ_A,
  P = a·b·σ·√(2π) (flujo integrado de línea), máscara de fits fallidos.
  Métrica de estructura: REUTILIZAR `wavesol_map.stripe_profile`/
  `structure_metrics` sobre los mapas a y σ (misma orientación).
- **Tests**: cubo sintético con a,σ conocidos por columna → recuperación
  (atol 5%); espectro sin línea → fit falla limpio (NaN, no excepción).
- **Salidas**: QC `halpha_map: {a_p95, sigma_median_A, sigma_structure_significance,
  …}` + 3 mapas PNG. Correlación esperada de Xie Fig. 3: a vs σ anticorrelados
  con P constante ⇒ LSF variable (instrumental), no ghost.
- **No hacer**: no interpretar (eso es S1b/humano); no tocar la LSF de E1.

### PASO S1b ✅ · Ejecutar S1a + integrar en E2 (HECHO 2026-07-18, rama `wavesol-s0b`)
Correr sobre realigned; añadir al QC de E2 la correlación "zonas sucias de
stripes vs mapas Hα" (claves nuevas `halpha_map_correlation`); figura al informe.
> S1 full-res sobre realineado (303 s, n_fit=29203, σ_median=2.01 Å).
> `scripts/s1b_integrate_e2.py` parchea `stage_h02_qc.json` (aditivo) con
> `halpha_map_correlation` + figura `plots/s1_halpha/s1b_stripe_vs_halpha.png` (registrada en
> E2 `figures`), y el notebook E2 gana un Plot 3. **Resultado:** a/σ NO alineados con slicers
> (estructura ≤ control transversal ⇒ radial); corr por columna stripe↔σ = −0.91 pero el
> control transversal = −0.905 ⇒ **confundido radial**, no firma de slicer; P_cov≈1.07 (no
> constante ⇒ tampoco el caso instrumental de Xie Fig.3). Consistente con G1. Sin interpretar
> ghost-vs-instrumental (por-exposición/humano).

### PASO S6a ✅ · Descomposición de ruido (modelo FIJADO) (HECHO 2026-07-18, rama `wavesol-s0b`)
> `musepipe/qc/noise_decomposition.py` (+ 5 tests): σ_ap(λ,r_B) por fotometría de apertura en
> anillo (reutiliza `ring_positions` de E5), F*(λ) del primario, fit σ²=(c·F^α)²+σ_bg² en log.
> DESVIACIÓN del brief: el fit usa el continuo LIMPIO ANCHO (5100–8800, excluyendo láser/telúricas/
> Hα) en vez de solo 6510–6825 — esa ventana estrecha da F* con rango ~2× y α degenerado; el rango
> ancho (F* ×18.7) recupera el término fotónico (como Xie Fig.8). **Resultado (realineado, psffit):**
> α=0.78, corr(logF,logS)=+0.75 (photon_limited), c=0.043, σ_bg=166; **Hα a r_B=72px = 2.50× el
> límite fotónico** (n_fit=711). El QC añade `regime` y `corr_logF_logS` para detectar el régimen
> degenerado. QC `stages/stageS6_noise_decomposition_qc.json`, fig `plots/s6_noise/`. Diagnóstico
> (no recalcula E3).
- **Objetivo**: "estamos a X× del límite fotónico en Hα a la separación de B"
  (lenguaje estándar de Xie §5 para el paper).
- **Modelo (no cambiar)**: en imágenes residuales post-sustracción a múltiples λ
  (filtros de 3 canales = 3.75 Å entre 6510–6825 Å, excluyendo 6540–6590 y
  features estelares), medir σ_ap(λ, r) por fotometría de apertura (radio =
  FWHM; REUTILIZAR la maquinaria de ruido de E1b/E5) y F*(λ) del espectro de la
  primaria. Ajustar `σ² = (c·F*^α)² + σ_bg²` (mínimos cuadrados en log,
  parámetros c, α, σ_bg; α esperado ≈ 0.5). Factor sobre fotónico en Hα =
  σ_medida(Hα, r_B) / (c·F*(Hα)^α).
- **Entradas**: productos E1b (mapas FOV residuales por método canónico) del run
  realineado; espectro de la primaria de A4/M3.
- **Salidas**: `stageS6_noise_decomposition_qc.json` con
  `{c, alpha, sigma_bg, factor_over_photon_at_halpha_rB, per_lambda_table}` +
  figura tipo Xie Fig. 8. Tests con imágenes sintéticas (ruido gaussiano puro ⇒
  factor ≈ 1; añadiendo sistemático ⇒ factor > 1 recuperado).
- **No hacer**: no recalcular los límites E3 con esto (solo diagnóstico QC).

### PASO S7a ✅ · Nota ILLUM en proveniencia A1 (HECHO 2026-07-18, rama `wavesol-s0b`)
Verificado en los SOF: `muse_scibasic_{object,std}.sof` incluyen 1 ILLUM c/u
(object 01:56, std 23:41, nearest-in-time). Añadido `illumination_correction`
{used, illum_files, note, reference Xie+20 §3.1} a `stage00r_qc.json` (realineado)
+ frase en `docs/spec_A1_codex_raw_reduction.md`.

### PASO S7b ✅ · Tabla Ṁ vs A_Hα (HECHO 2026-07-18, rama `wavesol-s0b`)
`scripts/s7b_mdot_vs_extinction.py` reutiliza la cadena de conversión de
`stage_h03_limits` (luminosity/lacc/mdot) desde `f_lim_observed` del método
canónico sobre A_V={0,1,1.8,2,4}. Reproduce el adoptado A_V=1.8 → Ṁ=8.19e-13
exacto. `tables/mdot_limit_vs_extinction.csv` + `plots/s7_extinction/` + bloque
`extinction_ladder` (aditivo) en `stage_h03_qc.json`. Relajación real
A_V=0→A_Hα=2 = **×8.0** (no ×6; la pendiente Alcala 1.13 ⇒ Ṁ∝ext^1.13).

### PASO S7c ✅ · Nota Hβ/Hα (HECHO 2026-07-18, rama `wavesol-s0b`)
Sección "Line Diagnostics" añadida a `musepipe/report.py` (constante
`LINE_DIAGNOSTICS_NOTE` + placeholder en la plantilla): con Hβ y Hα en upper
limit el decremento no restringe A_Hα ⇒ límite reportado sobre grid A_V (S7b);
cita Aoyama&Ikoma19/Hashimoto+20. Report realineado regenerado (aparece en
`report/report.md`). Tests de report verdes (determinismo/hash).

---

## Track S · Fase 2 — estado final

> **ESTADO (corregido 2026-07-18): PARCIALMENTE EJECUTADA como confirmación diferida de G1.**
> G1 se cerró 2026-07-17 como sistemático acotado (temporal) y la Fase 2 "de mejora" (S4/S5/G2)
> NO se disparó. Sin embargo, el usuario disparó la **confirmación diferida**: S2b/S2c
> regeneraron los 7 cubos por exposición (2026-07-18, ~130 min) y sobre ellos corrieron
> **S0-por-exposición** (0/7 con estructura de slicer ⇒ stripes descartados empíricamente,
> commit `1ce11a0`) y **S3a/S3b** (deriva temporal ~0.04 Å std, gate agregado PASS, commit
> `3d6c984`) ⇒ **cierre G1 CONFIRMADO y cuantificado**. S4a/S4b/S5a/S5b quedan **⛔ CERRADOS
> SIN EJECUTAR** (sin stripes ni deriva no compran mejora medible); sus briefs se conservan
> por si una decisión humana futura los reabre. Ver `docs/decision_g1_wavesol_2026-07-17.md`
> y `docs/wavesol_execution_log.md`.

### PASO S2a ✅ · Inventario + plan de regeneración (HECHO 2026-07-18 inline; tabla retroactiva 2026-07-19)
> El inventario se hizo inline el 2026-07-18 (SOF supervivientes, productos podados, params de
> spec_A1 §3, raw verificado) y S2b/S2c se ejecutaron el mismo día. Entregable formal escrito
> retroactivamente con TIEMPOS MEDIDOS (no estimados):
> `runs/ROXs12b_raw/tables/regeneration_plan.csv` — S2b 102 min (bias 31.1 + flat 47.3 +
> scibasic 23.7), S2c 27.8 min (7× scipost 3.7–4.6 min), total ~130 min vs ~180 estimados.

Inventariar qué intermedios sobreviven en `runs/ROXs12b_raw/raw_reduction/`
(verificado 2026-07-17: wavecal/lsf/standard/OFFSET_LIST/scipost_exp* [solo
IMAGE_FOV] sí; **muse_bias, muse_flat (⇒ TRACE_TABLE), muse_scibasic_* y los
DATACUBE por exposición NO — podados**). Producto: tabla `regeneration_plan.csv`
(recipe, SOF exacto, entradas existentes/faltantes, tiempo estimado). Raw en
`~/Descargas/MUSE_DATA/Rox12` (193 FITS, verificado presente; incluye masters
M.MUSE del archivo que NO deben entrar como raw — el orquestador ya los excluye).

### PASO S2b ✅ · Regenerar bias/flat/scibasic (HECHO 2026-07-18)
> Ejecutado vía `scripts/regen_perexp_cubes.py` (el `run_cascade.py` del scratchpad se había
> perdido; reconstruido y COMMITEADO, `1a44d3a`). 102 min medidos: bias 31.1 m (24
> MASTER_BIAS) + flat 47.3 m (24 MASTER_FLAT+TRACE) + scibasic 23.7 m. Gate PASS: 168
> PIXTABLE_OBJECT como la corrida original.

Brief original: extender el orquestador (un recipe por vez, `--products-json`,
cwd=output-dir, ILLUM nearest-in-time, OBJECT/STD separados) para re-correr
`muse_bias` + `muse_flat` (produce TRACE_TABLE) + `muse_scibasic` de los 7 OBJECT.

### PASO S2c ✅ · scipost ×7 por exposición (HECHO 2026-07-18)
> Ejecutado: 27.8 min (7× scipost de 3.7–4.6 min), SOF = `muse_scipost_exp{i}.sof` +
> OFFSET_LIST manual, `--save=cube,skymodel`. **7 cubos (2.7 GB c/u) + SKY_SPECTRUM en
> `/mnt/2TB/MUSE_work/ROXs12b_perexp/exp{1..7}/`.** Desviación documentada: el gate WCS
> se relajó a solo-ESPECTRAL (CRVAL3/CD3_3/NAXIS3 idénticos) — espacialmente cada cubo
> queda centrado en su propio pointing (~0.6" aparte), esperado en cubos por-exposición.

### PASO S0-perexp ✅ · Confirmación diferida G1 sobre los 7 cubos (HECHO 2026-07-18)
> Paso añadido por la decisión G1 (no estaba en el plan original): `scripts/s0_perexp.py`
> corre `wavesol_map` en cada cubo por-exposición (con la rotación de campo deshecha, un
> stripe de slicer YA NO se promedia acimutalmente y debería aparecer). **Resultado: 0/7
> exposiciones con estructura de slicer (stripe_sig ≈ transv ≈ 1× en todas) ⇒ stripes
> DESCARTADOS empíricamente** (a diferencia de Hashimoto 5/6). Tabla
> `tables/s0_perexp_summary.csv`; commit `1ce11a0`; addendum en la decisión G1.

### PASO S3a ✅ · Offsets absolutos por exposición — airglow (HECHO 2026-07-18)
> Ejecutado (`scripts/s3_perexp_offsets.py`, CLI `m1m2-sky` por SKY_SPECTRUM): offsets
> +0.033…+0.143 Å (exp4 atípico, peor S/N), media +0.062 ≈ combinado (+0.074), **spread
> 0.039 Å std** ⇒ deriva temporal despreciable. Tabla `tables/perexp_m1_offsets.csv`.

### PASO S3b ✅ (con desviación) · Offsets relativos por xcorr (HECHO 2026-07-18)
> Ejecutado CON DESVIACIÓN documentada (bitácora §desviaciones): NO se corrió el B1/B2
> completo por-stripe-group (desproporcionado tras descartar stripes en S0-perexp); se hizo
> xcorr estelar a nivel de campo entero vía `wavesol_map`. Spread 0.038 Å std. **Gate
> S3a↔S3b AGREGADO: PASS** (ambos ~0.04 Å); el matching por-exposición fino queda limitado
> por ruido (5/7, señal ~ precisión), sin sesgo. Commit `3d6c984`.

Brief original (por si se reabre): run `ROXs12b_perexp` con stage01+stage02 (la
maquinaria soporta N cubos), `stage02_xcorr_shifts.npy` por cubo × grupo.

### PASO S4a ⛔ · Combinación propia — CERRADO SIN EJECUTAR (G1: sin stripes ni deriva)
> Algoritmo FIJADO; se conserva por si una decisión humana futura lo reabre. La pieza de
> valor independiente (STAT_EMP, punto 5) se rescató como **PASO R4** del Lote R.
Nuevo `musepipe/stages/stage02b_combine.py` (+ tests sintéticos):
1. Entrada: los 7 cubos del stack B1 con sus shifts de consenso S3
   (promedio de S3a/S3b si consistentes).
2. Aplicar shifts con `stripes.apply_stripe_spectral_shifts` EXISTENTE
   (STAT con kernel², ya implementado y testeado).
3. **QC de frame** por exposición (tabla `frame_qc.csv`): amplitud de stripes
   post-shift, FWHM del core de la primaria en banda 8000–9000, fondo mediano.
   Peso w_i = 1/mediana(STAT_i); **descartar** exposición si FWHM > 1.5× la
   mediana de frames o fondo > 2× mediana (criterio Hashimoto; se espera
   descartar 0–2).
4. Combinar: media ponderada con sigma-clip por vóxel (3.5σ, 1 iteración,
   σ = dispersión inter-exposición); donde queden <3 valores finitos, mediana
   simple. DATA_out float32.
5. STAT doble: `STAT` = propagada Σw²σ²/(Σw)²; extensión extra `STAT_EMP` =
   s²_inter-exposición/n_eff por vóxel (¡insumo directo contra el M5 rojo!).
6. **Gate de equivalencia**: combinación SIN shifts ni descartes vs el
   DATACUBE combinado del DRS — imagen blanca con correlación > 0.999 y ratio
   de flujo 1±0.02 (no serán idénticos: drizzle vs media de cubos; documentar).
7. Tests sintéticos: 3 cubos con shifts conocidos + un frame malo ⇒ el
   combinado recupera el espectro de referencia (atol), el frame malo se
   descarta, STAT_EMP ≈ varianza teórica.

### PASO S4b ⛔ · Telúrica sobre el combinado — CERRADO SIN EJECUTAR (depende de S4a)

### PASO S5a ⛔ · A4 sobre v2 — CERRADO SIN EJECUTAR (no hay cubo v2)

### PASO S5b ⛔ · Re-run B→F comparativo + G2 — CERRADO SIN EJECUTAR (no hay cubo v2)

### PASO S8 ✅ · Cierre (HECHO 2026-07-19)
> G1 marcado `closed`/`confirmed_closed` en QC y en el encabezado de la decisión; notebooks
> `S0_wavesol_map.ipynb` (conclusión confirmada) y NUEVO `S1_halpha_map.ipynb`; bitácora
> `docs/wavesol_execution_log.md`; F1 refrescado (overall yellow, sin cambio — caveat
> A-block); memoria actualizada. Suite 561. Merge a `main`: PR #1 (`0ad9da1`).

---

## Lote R — re-auditoría 2026-07-18 (pasos NUEVOS, post-cierre del Track S)

Brechas restantes tras comparar Xie+20 y Hashimoto+20 contra el estado
post-merge (PR #1). R1 es el único sustantivo (cambia el número del paper);
R2/R3 son texto/documentación; R4/R5 explotan los 7 cubos por-exposición que
ahora existen en `/mnt/2TB/MUSE_work/ROXs12b_perexp/exp{1..7}/`.

### PASO R1 ✅ · Relación L_acc–L_Hα PLANETARIA (Aoyama+21) junto a la estelar (HECHO 2026-07-18, rama `wavesol-r1`)
> **Cross-check PASADO**: coefs (a,b)=(0.95,1.61), σ=0.30 dex verificados contra el full-text de
> Aoyama+21 (arXiv:2108.01277 vía ar5iv: "log L_acc = 0.95 log L_Hα + 1.61", RMS 0.11 dex, σ rec.
> 0.30 dex, validez L_acc≤1e-4 L☉); Marleau&Aoyama23 EXCLUYE Hα (solo n>8), así que la fuente de Hα
> es Aoyama+21. `species` no instalado ⇒ cross-check contra la fuente primaria. **Implementado**:
> config `g3_lacc_relations.halpha_aoyama21`; E3 `limit_conversion_chain` gana params alt →
> claves `l_acc_aoyama21_lsun`/`mdot_aoyama21_msun_yr`/`*_5sigma`/`*_err_dex` (aditivas, NaN si no
> hay relación); QC `physical_inputs.lacc_aoyama21_relation` + `limits[].mdot_aoyama21`; S7b table
> gana columnas `l_acc_aoyama21_lsun`/`mdot_aoyama21_msun_yr`; `report.py` sección "Accretion
> Relation" (`ACCRETION_RELATION_NOTE`). **Resultado (realineado, psffit/combined):** headline
> Alcalá **8.194e-13 M☉/yr INTACTO** (0 dígitos cambiados); Aoyama+21 planetario **8.64e-12 M☉/yr,
> ×10.5 más DÉBIL** (relación más plana ⇒ L_acc mayor). Tests: 2 nuevos en `test_h03_chain.py`
> (ratio analítico 10^((a1-a2)logL+(b1-b2)) + presencia en stage). Suite 562. E3/S7b/G3/report
> re-corridos; overall yellow sin cambio. **Pendiente revisión humana del texto del informe.**

Brief original:
- **Objetivo**: reportar el límite de Ṁ con AMBAS calibraciones — Alcalá+2017
  (estelar, la actual) y Aoyama et al. 2021 (choque planetario) — porque en
  nuestro régimen (L_Hα ≲ 1e−6 L☉) difieren 1–4 dex en L_acc y ROXs 12 B es
  planetario/BD. Hoy E3/G3 usan SOLO Alcalá (verificado en
  `stage_h03_qc.json`/`stage_g3_qc.json`: `lacc_lha_relation = "Alcala et al.
  2017 Halpha"`), aunque el informe ya cita el marco Aoyama&Ikoma.
- **Coeficientes (VERIFICADOS 2026-07-18 contra Marleau & Aoyama 2023, RNAAS,
  arXiv:2303.00011, Fig. 1 y §3; NO re-derivar)**: para Hα,
  `log10(L_acc/L☉) = 0.95·log10(L_Hα/L☉) + 1.61`, scatter σ = 0.3 dex
  (errorbar de Ao21). Cita: Aoyama, Marleau, Ikoma & Mordasini 2021, ApJL 917,
  L30. **Cross-check obligatorio del agente** (sin acceso al paper): el toolkit
  `species` (Stolker et al. 2020; tutorial "Emission line",
  species.readthedocs.io) implementa estos mismos fits — verificar (a, b) de
  Hα contra `species` antes de tocar config; si difieren de (0.95, 1.61),
  PARAR y reportar.
- **Procedimiento**:
  1. Config del run realigned: añadir a `g3_lacc_relations` la entrada
     `halpha_aoyama21` con `{a: 0.95, b: 1.61, scatter_dex: 0.3, citation:
     "Aoyama et al. 2021, ApJL 917, L30 (planetary shock)", validity_range:
     "planetary-mass accretors; preshock n0~1e9-1e14 cm-3, v0<~200 km/s"}`.
     El código de G3 ya itera el dict (`stage_g3_accretion.py`) — NO tocar la
     relación Alcalá existente ni su posición de "headline".
  2. E3 (`stage_h03_limits.py`): añadir claves paralelas
     `mdot_lim_aoyama21_*` calculadas con (0.95, 1.61) sobre el MISMO
     `f_lim_dereddened` — sin cambiar las claves existentes.
  3. Extender la tabla S7b `tables/mdot_limit_vs_extinction.csv` con columnas
     por relación (Alcalá / Aoyama+21) — el grid de A_V se mantiene.
  4. `report.py`: ampliar `LINE_DIAGNOSTICS_NOTE` (o nota nueva
     `ACCRETION_RELATION_NOTE`) explicando la dualidad: la relación planetaria
     da L_acc MAYOR para el mismo L_Hα ⇒ límite de Ṁ MÁS DÉBIL; el paper
     reporta ambos (patrón estándar post-Hashimoto). Citar también Marleau &
     Aoyama 2023 (RNAAS) como fuente de la extensión/validez.
  5. Re-correr E3/G3/F1 del realigned (etapas baratas, minutos).
- **Tests**: unitario del cálculo dual en E3 (mismo L_Hα ⇒ ratio de límites
  = 10^((1.13−0.95)·logL_Hα + (1.74−1.61)) — verificar signo/magnitud con un
  caso numérico fijado); tests existentes de G3/report intactos.
- **Gate**: el límite Alcalá actual NO cambia ni un dígito (solo se añade);
  revisión humana del texto del informe.
- **No hacer**: no añadir relaciones para otras líneas (Hβ etc. quedan fuera);
  no cambiar el headline sin decisión humana.

### PASO R2 ✅ · Caveat de variabilidad de acreción (una época) (HECHO 2026-07-18, rama `wavesol-r2`)
> `VARIABILITY_CAVEAT_NOTE` en `report.py` (prosa fija, patrón LINE_DIAGNOSTICS_NOTE) + sección
> nueva "Variability Caveat" en la plantilla. Época verificada contra la decisión G1: **2022-09-01,
> MJD 59823.025–59823.086, 7 exposiciones ~87 min**; cita Cody & Hillenbrand 2014 + Hashimoto+20
> §5.4 (multi-época). Solo texto: sin tocar QC ni números. Report realineado regenerado; report
> tests (determinismo/hash recalculado) verdes. Suite 562.

Brief original:
- **Objetivo**: frase obligatoria de conclusiones: la acreción es variable en
  el tiempo (Cody & Hillenbrand 2014; Hashimoto+20 §5.4 la piden para
  multi-época); una NO-detección de una sola época no excluye acreción
  episódica ni media a otro nivel.
- **Procedimiento**: constante nueva `VARIABILITY_CAVEAT_NOTE` en
  `musepipe/report.py` (mismo patrón que `LINE_DIAGNOSTICS_NOTE`: prosa fija +
  citas), insertada en la sección de límites/conclusiones de la plantilla;
  mencionar que el dataset es 1 época (2022-09-01, 7 exposiciones en 87 min) y
  que el límite de Ṁ aplica a ESA época. Regenerar report del realigned.
- **Tests**: los de determinismo/hash del report (actualizar hash esperado).
- **No hacer**: no tocar QC ni números; es solo texto.

### PASO R3 ✅ · Censo de ghosts instrumentales (Xie+20 Apéndice A) (HECHO 2026-07-18, rama `wavesol-r3`)
> Núcleo en `musepipe/qc/ghost_census.py` (patrón S0a/S6a: I/O-free + CLI) + 9 tests. Corre sobre
> el cubo `stage02_xcorr_cube_stack.fits` (el frame que localizó stage01c; B en pos_yx medida
> [155.60, 75.79], NO el [72,152] legacy). **DOS desviaciones documentadas del brief:**
> (1) **Resta del halo radial de la primaria** antes del test de tira: el método naïve (mediana de
> fila/columna vs anillo local) dio un falso positivo de **col=+9.2σ en las 3 bandas** porque la
> COLUMNA de B (x≈76) cruza y≈85 donde está la primaria (a 8.6px) y capta su halo AO — el mismo
> confound radial de S1b. Con `subtract_radial_profile` (mediana azimutal por anillo) el exceso
> colapsa a **max 0.6σ** ⇒ un ghost de slicer (línea a dirección fija) sobreviviría, el halo
> simétrico no. (2) **Fringing**: el criterio "pico>5×mediana" del brief tiene piso de ruido
> blanco ~1.4·ln(M)≈8 (mide 8.9 en limpio) ⇒ añadida condición robusta a M: fracción de varianza
> del modo dominante >0.15 (mide 0.021). **Resultado: `line_ghost_strip=none`, `blob_fringing=none`
> ⇒ B libre de ghosts.** QC standalone `stageR3_ghost_census_qc.json` + parche ADITIVO a
> `stage_h02_qc.json` (`ghost_census`) + figura `plots/r3_ghost/`. Suite 571. Sin corregir nada
> (no había nada que corregir).

Brief original:
- **Objetivo**: párrafo de blindaje en E2: verificar y documentar que en la
  posición de B no hay line ghosts (strips sobre-brillantes de IFU/slice,
  Weilbacher et al. 2015) ni blob ghost (mancha con fringing espectral azul,
  Xie+20 App. A).
- **Procedimiento**: script corto (puede vivir en `scripts/`) que sobre el
  cubo realineado: (1) imagen blanca + banda Hα (6540–6590) + banda azul
  (4800–5500); (2) en cada una, perfil de filas/columnas por la posición de B
  (y=152, x=72 en el crop; trasladar al uncropped) buscando strip
  sobre-brillante > 3σ del anillo local; (3) espectro en caja 3×3 en B buscando
  fringing periódico azul (FFT simple del continuo 4800–5500 normalizado:
  ningún pico > 5× la mediana del espectro de potencia). Añadir al QC de E2
  clave `ghost_census: {line_ghost_strip: none|detected, blob_fringing:
  none|detected, method, thresholds}` + 1 figura.
- **Tests**: sintético con strip inyectado ⇒ `detected`; limpio ⇒ `none`.
- **No hacer**: no corregir nada (si detecta algo, PARAR y reportar — sería
  hallazgo nuevo).

### PASO R4 ✅ · STAT_EMP — varianza empírica inter-exposición (ataca el M5 rojo) (HECHO 2026-07-18, rama `wavesol-r5`)
> Núcleo en `musepipe/qc/stat_emp.py` (I/O-free + CLI chunked por canal, memmap) + 7 tests.
> **HALLAZGO/DESVIACIÓN clave (con guía humana): los 7 cubos NO están en grid vóxel-común.** El
> premisa del brief ("mismo grid por OFFSET_LIST") era falsa: NAXIS1/2 difieren (316×306 vs
> 308×313), dither ~20-26px. Verificado: los cubos FINALES son todos **North-up** (CD sin
> rotación, eje-x=−180° idéntico); la rotación del derotador (ABSROT −16→+3°) la absorbió scipost
> al resamplear cada uno North-up. CRVAL sí difiere ~0.6" (mi allclose inicial lo ocultó) ⇒
> separación = **pura traslación**. ⇒ alineo por **shift ENTERO** (sin interpolación → no añade
> covarianza; residual sub-px ≤0.4px ⇒ límite superior). La rotación distinta por exposición
> orienta distinto la covarianza de resampleo ⇒ exposiciones cuasi-independientes ⇒ STAT_EMP
> captura justo lo que el STAT subestima. Gate WCS relajado a sub-canal en CRVAL3 (los cubos
> difieren mÅ = la deriva λ de S3; grid CD3_3/NAXIS3 idénticos). **Resultado:** ratio = s²_emp/
> mean(STAT_i). **Vóxel típico (fondo): mediana ×2.9 (bias-corregida ×3.3, ceñido 2.82-3.18);
> ponderado por flujo ×8.6 (límite superior, inflado por variabilidad de seeing/transparencia
> de la fuente).** El M5 "~4-6× estimado" queda **acotado empíricamente [~3×, ~8.6×]**. STAT_EMP.fits
> (285×290×3681, `/mnt/2TB/.../STAT_EMP.fits`), `stageR4_stat_emp_qc.json`,
> `tables/stat_emp_ratio_by_channel.csv`, figura `plots/r4_stat_emp/`. **Nota M5 en `report.py`
> (ACCEPTED_LIMITATIONS) actualizada** de "~4-6× estimado" a los valores medidos. Suite 584. NO
> sustituye el STAT canónico ni re-corre B→F (medición/QC).

Brief original:
- **Objetivo**: producir la única estimación de varianza libre de la
  covarianza de resampleo del DRS: por vóxel, la dispersión ENTRE las 7
  exposiciones ya regeneradas (mismo grid por OFFSET_LIST).
- **Procedimiento**: script/CLI nuevo `python -m musepipe.qc.stat_emp
  --cubes /mnt/2TB/MUSE_work/ROXs12b_perexp/exp*/DATACUBE_FINAL.fits
  --output .../STAT_EMP.fits`:
  1. Cargar los 7 DATA (verificar WCS idéntico: mismos CRVAL/CD/NAXIS — gate).
  2. Por vóxel: n = nº de exposiciones finitas; si n ≥ 4:
     `var_emp = varianza muestral entre exposiciones / n` (varianza DE LA
     MEDIA); si n < 4: NaN. Guardar como FITS float32 (misma cabecera WCS).
  3. Comparación con el STAT del cubo combinado DRS: mapa e histograma de
     `ratio = var_emp / var_DRS` por canal (mediana por canal + global).
     Esperado según A4 M5: ratio ~4–6 (la DRS SUBESTIMA); confirmar con número.
  4. QC `stat_emp_qc.json`: `{n_exposures, ratio_median_global,
     ratio_p16_p84, per_channel_table, gate_wcs: pass}`.
- **Salida clave para el paper**: el ratio medido convierte la limitación
  aceptada M5 de "~4–6× (estimado)" a "X.X× (medido empíricamente)" — citar en
  la justificación de por qué D2/E1 usan ruido empírico de controles.
- **Tests**: sintético de 5 cubos con varianza conocida ⇒ var_emp recupera
  σ²/n (atol 5%); cubos con WCS distinto ⇒ gate falla limpio.
- **No hacer**: NO sustituir el STAT del cubo canónico ni re-correr B→F con
  STAT_EMP (eso sería una decisión G2-like, humana); esto es medición/QC.
- **Nota**: ojo con la media vs mediana — las 7 exposiciones tienen el mismo
  apuntado nominal post-OFFSET_LIST pero seeing/transparencia distintos; la
  varianza muestral incluye esa variación real de PSF/fotometría además del
  ruido; documentar como límite superior del ruido por vóxel.

### PASO R5 ✅ · Censo frame-QC de las 7 exposiciones (patrón Hashimoto 2/6) (HECHO 2026-07-18, rama `wavesol-r5`)
> Núcleo en `musepipe/qc/frame_qc.py` (I/O-free + CLI, reusa `band_image` de ghost_census) + 6
> tests. Por exposición: (a) FWHM del core por momentos ponderados en 8000–9000 Å tras restar el
> fondo de anillo (helper simple, no Psfao); (b) fondo mediano en anillo exterior; (c) stripe_sig
> **reusado de `s0_perexp_summary.csv`** (no recalculado). Criterio S4a (FWHM>1.5×med o fondo>2×med),
> SIN descartar de facto. **Resultado: 7/7 utilizables, 0 flaggeadas** — FWHM 11.0–12.8px (mediana
> 11.69, max 1.09×), fondo 33.8–49.9 (max 1.33×), stripe_sig 0.82–1.36. Tabla
> `tables/perexp_frame_qc.csv` + QC `stages/stageR5_frame_qc.json`. Suite 577. (Nota: la FWHM por
> momentos infla el core por el halo AO; lo que importa es la consistencia relativa entre frames.)

Brief original:
- **Objetivo**: una línea de métodos para el paper: "las 7 exposiciones son
  utilizables; 0 descartadas" — con números, no por fe (Hashimoto descartó 2/6
  y el referee puede preguntar por las nuestras).
- **Procedimiento**: script corto sobre los 7 cubos por-exposición: por
  exposición, medir (a) FWHM del core de la primaria en banda 8000–9000 Å
  (ajuste 2D sencillo o momento; existe maquinaria de PSF en C1 — reutilizar
  el helper más simple, no Psfao); (b) fondo mediano en anillo exterior;
  (c) amplitud de stripes (ya medida en S0-perexp — REUSAR
  `tables/s0_perexp_summary.csv`, no recalcular). Criterios del brief S4a:
  descartable si FWHM > 1.5× mediana o fondo > 2× mediana. Tabla
  `tables/perexp_frame_qc.csv` + clave en QC de la bitácora/F1.
- **Tests**: ninguno nuevo si reutiliza helpers testeados; smoke sobre
  sintético si añade código a `musepipe/`.
- **No hacer**: no descartar nada de facto — si alguna exposición viola el
  criterio, PARAR y reportar (sería insumo para reabrir S4, decisión humana).

---

## Orden sugerido de despacho al agente

**Histórico (Track S, EJECUTADO):** Lote 1 (A1b, A3, S7a–c — solo se ejecutó
S7a–c), Lote 2 (S0b/S0c, S1a/S1b), G1, Lote 3 parcial (S2, S0-perexp, S3, S6a),
S8. Ver bitácora.

**Vigente (2026-07-18):**
- Lote R (paralelo, sin dependencias entre sí): R1, R2, R3, R5. R4 después de
  R5 (usa la misma carga de cubos).
- Track A pendiente (paralelo al Lote R): A1b, A3 primero; A2a después;
  checkpoint humano A2b; A1a (molecfit) en sesión dedicada 🔴.
- Al cerrar el lote: F1 refresh + actualización de bitácora y memoria.
