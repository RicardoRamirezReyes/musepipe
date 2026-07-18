# Pasos ejecutables por agente — plan wavesol/stripes 2026-07-17

Descomposición de `docs/plan_wavesol_stripes_2026-07-17.md` en pasos pequeños,
cada uno auto-contenido, con contrato de salida y límites duros. Formato y
cultura: como los `spec_*_codex_*.md` del repo (alcance deliberadamente
pequeño, gates de equivalencia, QC extendido sin renombrar claves, runs
históricos intocables, rama git por paso).

**Leyenda de nivel**: 🟢 = ejecutable por agente menos potente con este brief ·
🟡 = ejecutable, pero requiere revisión humana del resultado · 🔴 = complejo,
requiere agente fuerte o sesión dedicada · ✅ = HECHO (2026-07-17, sesión fuerte).

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
| Debug de molecfit (A1a) | 🔴 pendiente — interactivo, sesión dedicada con agente fuerte |

---

## Track A — cierres A-block

### PASO A1b 🟢 · Justificación STD_TELLURIC (documental)
- **Objetivo**: cerrar el open_issue "molecfit no convergió" con una justificación
  formal del método aplicado, publicable en la sección de métodos.
- **Entradas**: `docs/a3_telluric_justification.md` (esqueleto),
  `runs/ROXs12b_raw/stages/stage00t_qc.json` (V1–V5: O2 6.76%→0.60%, etc.).
- **Salidas**: doc completado con los números reales de V1–V5 + referencia a que
  Hashimoto+20 usó la calibración de flujo del pipeline estándar (sin molecfit);
  `stage00r_qc.json` (realigned): open_issue de telúrica anotado con
  `"resolution": "justified_std_telluric", "doc": "docs/a3_telluric_justification.md"`.
- **No hacer**: no re-correr esorex; no tocar el cubo.

### PASO A1a 🔴 · Intento molecfit (timebox, OPCIONAL si A1b aprobado)
Sesión dedicada con agente fuerte: iterar config de `molecfit_model`
(WAVE_INCLUDE, kernel, continuum, columnas del espectro 1D) con timebox de una
sesión. Si converge, comparar T(λ) contra STD_TELLURIC y decidir; si no,
A1b queda como cierre definitivo.

### PASO A2a 🟡 · Master BIAS por noche (evidencia)
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

### PASO A2b 🔴-lite · Decisión de agrupación (humana)
Con la tabla de A2a: aceptar (documentar en QC como limitación verificada) o
reagrupar (lo que dispararía re-cascada — decisión de costo del usuario).

### PASO A3 🟢 · Trazabilidad QC A2/A3 en el run realineado
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
> `notebooks/S0_wavesol_map.ipynb`.
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
  del run); notebook de revisión nuevo `notebooks/S0_wavesol_map.ipynb` vía
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

## Track S · Fase 2 — SOLO tras G1 aprobado (humano)

> **ESTADO: NO disparada.** G1 se cerró 2026-07-17 como sistemático acotado (temporal); la
> Fase 2 no se ejecuta. **Confirmación diferida:** si estos cubos por exposición se regeneran
> alguna vez (S2b/S2c), correr **S0 por exposición + S3** como verificación obligatoria del
> cierre G1 (con la rotación deshecha, los stripes deben verse si existen; separa temporal vs
> per-slice). Ver `docs/decision_g1_wavesol_2026-07-17.md`.

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

### PASO S2b 🟡 · Regenerar bias/flat/scibasic
Extender `scratchpad/run_cascade.py` (patrón existente: un recipe por vez,
`--products-json`, cwd=output-dir, ILLUM nearest-in-time, OBJECT/STD separados)
para re-correr `muse_bias` (~33 m) + `muse_flat` (~52 m, produce TRACE_TABLE) +
`muse_scibasic` de los 7 OBJECT (~24 m). Gate: 168 PIXTABLE_OBJECT como la
corrida original. Total S2b+S2c ≈ 3 h de esorex.

### PASO S2c 🟡 · scipost ×7 por exposición
Por exposición i=1..7: `muse_scipost` con los 24 PIXTABLE de esa exposición,
`OFFSET_LIST.fits` manual del plan B (para que las 7 salidas compartan grid WCS),
**save=cube,skymodel** (⇒ DATACUBE_FINAL + SKY_SPECTRUM por exposición — el
SKY_SPECTRUM es imprescindible para S3a). Salida:
`/mnt/2TB/MUSE_work/ROXs12b_perexp/expN/` (~25 GB). Gate por exposición: shape
espectral idéntica, WCS idéntico al cubo combinado (mismos CRVAL/CD).

### PASO S3a 🟢 · Offsets absolutos por exposición (airglow)
CLI existente `python -m musepipe.qc.cube_qc m1m2-sky --sky-spectrum
expN/SKY_SPECTRUM_0001.fits` en loop ⇒ tabla
`tables/perexp_m1_offsets.csv` (exposición, offset_A, scatter, n_lines).
Contexto: M1 combinado = +0.074 Å; aquí interesa el SPREAD entre exposiciones.

### PASO S3b 🟡 · Offsets relativos por xcorr (B1+B2 con N=7)
Nuevo run `ROXs12b_perexp` (config copiada de realigned, input = los 7 cubos):
correr stage01 (B1, stack común) y stage02 (B2) — la maquinaria YA soporta N
cubos (fue diseñada para esto; con el realigned recibía N=1). Salida natural:
`stage02_xcorr_shifts.npy` (por cubo × grupo de stripes) + stripe_metric.
**Gate de consistencia**: shifts B2 (relativos) vs S3a (absolutos, restada la
media) coinciden dentro de la suma en cuadratura de sus errores; si no,
PARAR y reportar (posible sistemática del estimador).

### PASO S4a 🟡 · Combinación propia (algoritmo FIJADO — implementar tal cual)
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

### PASO S4b 🟢 · Telúrica sobre el combinado
Aplicar `TELLURIC_TRANS.fits` existente (DATA/T, STAT/T², helpers de
`telluric.py`) ⇒ `cube_telcorr_v2.fits`. Gate: V1–V5 de A3 (mismos checks).

### PASO S5a 🟢 · A4 sobre v2
CLIs existentes de `cube_qc` (M1–M5 + m3-flux con growth curve) + stripe
metric sobre v2. Comparativa v1 vs v2 en tabla. Esperable si había smearing:
M2 LSF más estrecha; M5 mejor con STAT_EMP.

### PASO S5b 🟡→🔴 · Re-run B→F comparativo + G2
Run nuevo `ROXs12b_realigned_v2` con la automatización existente del realigned;
deltas D1/E1/E3/F1 en tabla única. **DECISIÓN G2 (humana)**: adoptar v2 como
canónico o documentar mejora nula.

### PASO S8 🟢 · Cierre
Notebooks de revisión nuevos/extendidos (S0, S1, B2 con N=7 si Fase 2 corrió),
bitácora, informe, actualización de memoria del proyecto, F1 refresh.

---

## Orden sugerido de despacho al agente

Lote 1 (paralelo, sin dependencias): A1b, A3, S7a, S7b, S7c.
Lote 2: S0b → S0c (formal) y S1a → S1b; A2a.
Checkpoint humano: G1 (con S0c) + decisión A2b.
Lote 3 (si G1 aprueba): S2a → S2b → S2c → S3a/S3b → S4a → S4b → S5a → S5b(G2).
Siempre al final: S6a (independiente, cualquier momento), S8.
