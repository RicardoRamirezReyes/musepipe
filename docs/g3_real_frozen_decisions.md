# G3 real · Decisiones congeladas (checkpoint WP-G3R-1)

Fecha: 2026-07-16. Plan: `docs/plan_g3_real_2026-07-16.md`. Spec: G3 §8.1 +
patrón anti-sesgo E1 / G4 §1.2.

**Aprobación del usuario (verbatim, sesión 2026-07-16):** «Apruebo D1–D14,
ejecuta WP-G3R-0 y WP-G3R-1». Las catorce decisiones se aprobaron TAL COMO
estaban propuestas en el plan, sin modificaciones. La reorganización pendiente
de `reports/` se commiteó por separado con autorización explícita del usuario
en la misma sesión (opción «Commitéala tú ahora», commit `cc82098` en
`stage-d1-v2`).

**Dónde vive la congelación**: los valores están escritos en
`runs/ROXs12b_B_adp/config/config.json` (bloque `config`). Ese archivo está
IGNORADO por git (todo `runs/` lo está), así que el registro commiteado y
citable de la congelación es ESTE documento: lista cada clave con su valor
exacto. El commit de esta fase es el hash de congelación que los QC de
WP-G3R-11/12 deben referenciar (`frozen_decisions_commit`). Si en el futuro
config y este documento discrepan, manda este documento y hay que investigar.

Cambiar cualquier valor de aquí después de WP-G3R-11 exige issue +
autorización humana (plan §0.5.7).

---

## Decisiones y claves de config

### D1 · Plantillas empíricas jóvenes — CORRIGE la elección previa

La config traía `g3_template_citation = "Bonnefoy et al. 2014"` (biblioteca
NIR 1–2.5 µm, NO cubre el rango MUSE). Reemplazada con el OK del usuario:

- `g3_template_family = "X-shooter Class III young M-L templates"`
- `g3_template_citation = "Manara et al. 2013 (A&A 551, A107); Manara et al. 2017 (A&A 605, A86)"`
- `g3_template_version`: se añade en WP-G3R-2 al registrar la descarga
  (versión/URL/fecha en el manifiesto).

### D2 · Plantillas de gravedad de campo

- `g3_template_field_family = "SDSS empirical templates O5-L3"`
- `g3_template_field_citation = "Kesseli et al. 2017 (ApJS 230, 16)"`
- `g3_template_resolution_rule`: si FWHM_plantilla > LSF MUSE (2.383 Å) no se
  degrada; se registra `resolution_mismatch` como limitación (válido porque el
  χ² se computa sobre bins ≥ 25 Å).

### D3 · Alternativa no estelar para T3

- `g3_nonstellar_alternative = "powerlaw"` (F_λ ∝ λ^α, α libre). Sin plantilla
  de galaxia (T2/morfología cubre ese caso); anotado como limitación de T3.

### D4 · Rejilla atmosférica BT-Settl CIFIST (Allard et al. 2012)

Formato de eje: `[min, max, paso]`.

- `g3_atmo_teff_axis_k = [2000.0, 4500.0, 100.0]`
- `g3_atmo_logg_axis = [3.5, 5.5, 0.5]`
- `g3_atmo_av_axis = [0.0, 5.0, 0.1]` (CCM89 R_V=3.1 ya en config, aplicada
  externamente al modelo)

### D5 · Tracks evolutivos (sin cambio)

`g3_tracks_families = ["BHAC15", "ATMO2020"]`, `g3_tracks_citations` (Baraffe
et al. 2015; Phillips et al. 2020) — ya estaban en config. Nota de cobertura
vigente: ATMO2020 ≤ ~0.075 M☉ / Teff ≲ 3000 K; BHAC15 ≥ ~0.01 M☉; si el
posterior cae fuera de una familia → parada WP-G3R-11.

### D6 · Escala SpT↔Teff para jóvenes

- `g3_spt_teff_scale_citation = "Herczeg & Hillenbrand 2014 (ApJ 786, 97)"`
- Refinamiento documentado: la TABLA numérica se transcribe del paper en
  WP-G3R-9/11 (donde se usa), verificada contra la fuente, con parada si el
  paper no está accesible. La cita queda congelada aquí; la transcripción no
  introduce libertad de ajuste (es una constante publicada).

### D7 · Índices espectrales de SpT

- `g3_spt_indices_citations`:
  - Riddick, Roche & Lucas 2007 (MNRAS 381, 1067) — índices ópticos M jóvenes
    dentro de 6800–9349 Å
  - Martín et al. 1999 (AJ 118, 2466) — PC3
  - Slesnick, Hillenbrand & Carpenter 2004 (ApJ 610, 1045) — Na I 8183/8195
    (gravedad)
- Mismo refinamiento que D6: las ventanas numéricas (`g3_spt_indices`) se
  transcriben del paper en WP-G3R-7, NUNCA de memoria; parada si no hay acceso.

### D8 · Rango de ajuste y binning

- `g3_fit_wave_range_A = [6300.0, 9349.5]` (rango primario)
- `g3_fit_wave_range_full_A = [4749.5, 9349.5]` (variante de sensibilidad;
  se reporta como sistemático, no se promedia)
- `g3_fit_bin_channels = 20` (≈25 Å/bin)
- `g3_fit_bin_max_masked_frac = 0.5` (bins con >50% enmascarado se descartan)
- `g3_fit_err_column = "flux_err_total"`
- Varianza por bin: Var = (Σσ²)/N² × (N/N_eff_bloque) con
  `n_eff_over_n_by_block` de `g1_channel_covariance.npz`; correlación ENTRE
  bins despreciada (corr_len 1.45 canales ≪ 20) — se anota en QC.

### D9 · Máscaras congeladas (`g3_fit_masks`)

- `use_stage04b_bad_mask = true` — nota WP-G3R-6: en `stages/` existen
  `stage04b_bad_wavelength_mask.npy` (216 canales True) y
  `stage04b_good_wavelength_mask.npy`; WP-G3R-6 fija la semántica leyendo el
  código de stage04b y lo documenta en el QC.
- `line_window_kms = 300.0` alrededor de las 24 líneas del catálogo G2
  (detectadas o no).
- `telluric_bands_A = [[6865.0, 6950.0], [7590.0, 7700.0]]` (O2 B y A,
  rangos conservadores aprobados en el checkpoint).

### D10 · Veiling / exceso azul

- `g3_veiling_variant = true`: variante con componente aditiva ley de
  potencias (a·λ^α, a ≥ 0); efecto sobre SpT/Teff → `err_sys`. Si no converge
  (S/N azul ≈ 0.15), veiling `not_constrained`.

### D11 · Estadística congelada

- `g3_dchi2_1sigma_per_param = 1.0` (intervalos por ΔΧ² marginalizado)
- `g3_chi2red_inflate_threshold = 1.5` (χ²_red > 1.5 → inflar errores √χ²_red
  y anotar)
- `g3_chi2red_stop = 3.0` (χ²_red > 3 → PARADA, spec G3 §8.2)
- `g3_n_mc = 4000`, `g3_seed = 0` (MC end-to-end; nota: `stage_g3_accretion`
  usaba defaults 2000/0 vía `cfg.get` — ahora quedan explícitos y el MC de
  acreción pasa a n=4000, decisión incluida en la congelación)
- Error de interpolación de rejilla: medio paso de eje en cuadratura.

### D12 · Edad de la región

- `g3_age_myr = 6.0`, `g3_age_err_myr = [2.0, 4.0]` (formato `[lo, hi]`)
- `g3_age_citation = "Bowler et al. 2017 (AJ 154, 165; arXiv:1708.07611)"`
- Transcripción verificada contra la fuente el 2026-07-16 (abstract de
  arXiv:1708.07611): «the age of the host star (6^{+4}_{-2} Myr)» — la misma
  edad de la que Bowler+2017 deriva 17.5 ± 1.5 MJup (consistente con
  `h03_mass_source` ya en config). MC de WP-G3R-9: gaussiana asimétrica
  (dos medias-normales σ_lo=2, σ_hi=4) truncada > 0.

### D13 · Umbrales G4 congelados

- T3: `g4_t3_dchi2_excl = 9.0` (ΔΧ² > 9 → `excludes`),
  `g4_t3_dchi2_disfavor = 4.0` (4–9 → `disfavors`; < 4 → `neutral`; clase
  ganadora → `supports`).
- T4: `g4_t4_radius_rjup_range = [0.5, 3.0]` (R a d=138.6 pc dentro del rango
  → `supports` ligadas); `g4_t4_distance_factor = 3.0` (distancia
  autoconsistente de campo a < 3× o > 1/3× la del sistema → `disfavors` la
  hipótesis correspondiente).
- T6: `g4_t6_av_sigma_support = 2.0` (< 2σ → `supports` asociada),
  `g4_t6_av_sigma_background = 3.0` (> 3σ por exceso → `supports` fondo;
  2–3σ → `neutral`).
- Fronteras de masa: `g4_mass_boundaries_mjup = [13.0, 75.0]` con
  `g4_mass_boundaries_citations` (Spiegel, Burrows & Milsom 2011, ApJ 727,
  57; Chabrier & Baraffe 2000, ARA&A 38, 337).
- Veredicto por P(m ∈ rango): `g4_mass_prob_verdicts = {supports: >0.6,
  neutral: 0.2–0.6, disfavors: 0.05–0.2}`; < 0.05 → `excludes`.
- Clase combinada: `g4_combined_class_margin = 0.5` (|ΔlogL| top-2 < 0.5 y
  ambas subestelares → `companion_substellar_or_planetary`).

### D14 · Directorio de bibliotecas externas

- `g3_libraries_root = "../Data/external_libraries"` (relativo a la raíz del
  repo). Subdirectorios `bt-settl-cifist/`, `templates_young/`,
  `templates_field/`, `tracks_bhac15/`, `tracks_atmo2020/`, cada uno con
  `MANIFEST.sha256`.

---

## Estado tras el checkpoint

- 37 claves escritas en la config del run el 2026-07-16; solo
  `g3_template_family` y `g3_template_citation` sobrescribieron valores
  previos (corrección D1). El resto son claves nuevas.
- Pendientes de transcripción con cita ya congelada (parada si no hay acceso
  al paper en su fase): tabla SpT–Teff HH14 (WP-G3R-9/11), ventanas de
  índices (WP-G3R-7), coeficientes Lacc–L_line de Alcalá+2017 Tabla B.1
  (WP-G3R-10).
- Siguiente fase: WP-G3R-2 (infraestructura de bibliotecas externas).
