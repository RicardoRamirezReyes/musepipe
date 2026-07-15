# Especificación C5 · `X04_sgf` — sustracción de halo por diversidad espectral (SGF)

Fecha: 2026-07-14. Etapa C5 del plan
`docs/plan_integracion_halosub_julo2025.md` (WP-H1). Método de referencia de
literatura: Haffert et al. 2019 / Xie et al. 2020, formalizado y analizado en
Julo et al. 2025 (arXiv:2509.09878, App. A.3). Infraestructura: WP-H0
(`musepipe/halosub.py`, oráculos en `tests/test_halosub_toy.py`).

Protocolo spec-first: esta spec se commitea ANTES de ejecutar la etapa sobre
datos reales. El QC registra `spec_version="C5_v1"`. Cambiar los parámetros
congelados de §1 mirando estadísticos de un run = revisión C5_v2 con
justificación escrita.

## 0. Rol, objetivo y definición de terminado

C5 produce el quinto método de extracción del pipeline: espectros congelados
(`SpectrumProduct`) del compañero y sus controles tras sustraer el halo
estelar por **diversidad espectral con filtrado Savitzky-Golay** (SGF). Es la
línea base de literatura, implementada FIEL al paper — deliberadamente sin
enmascarado de líneas y sin PCA — para que sus sesgos conocidos
(auto-sustracción, continuo negativo; Ec. 1) queden medidos y comparables en
D1 v3, no corregidos en silencio.

Terminado = productos `spec_sgf_object.fits` + controles npz + cubo residual
mediano + `spec_sgf_qc.json` con los checks de §6 evaluados.

## 1. Límites duros (parámetros congelados)

1. Filtro: Savitzky-Golay `sgf_degree = 1`, `sgf_window = 101` canales
   (Haffert et al. 2019; Tabla 1 del paper). Prohibido ajustarlos por run sin
   revisión de spec.
2. Sin enmascarado de líneas en el filtrado y sin PCA sobre residuos. La
   mitigación de la auto-sustracción del SGF es la corrección de throughput
   E4/E3 (como Jorquera et al. 2024), nunca un parche dentro de C5.
3. Espectro de referencia: mediana de spaxels con flujo integrado en
   (`halosub_flux_mask_lo`, `halosub_flux_mask_hi`) × Fmax = (0.01, 0.1),
   excluyendo un disco de `halosub_exclude_radius_px` = 3.0 px alrededor del
   compañero. Por exposición, nunca global al stack.
4. Sustracción por exposición; los residuos se combinan con el MISMO
   combinador que usa el cubo madre de C2/C3/C4 (`nanmean` sobre los índices
   best de stage04b). `halosub_combine = "median"` (lo que hace el paper)
   existe como opción no-default: cambia la convención de ruido frente a los
   otros métodos y solo se usa para reproducciones de literatura.
5. Convención de escala D1 v2 §3.1 ("control = objeto"): controles procesados
   exactamente como el objeto, `SCALEREF = "normrad_total_flux"`,
   `BKGMODE = "sgf_residual+annulus"`. El `INCUBE` registrado es el cubo madre
   stage02 (mismo principio que C3).
6. C5 no re-alinea, no re-calibra y no toca el A/B-block.

## 2. Entradas

- `stages/stage02_xcorr_cube_stack.fits` (CUBES 4D o 3D + WAVELENGTH; STAT si
  existe) — mismo insumo que C2/C3/C4.
- `stages/stage04b_qc.json` (índices best para el combinador) y máscaras de
  ventana mala/skyline de config (`x02_bad_windows_A` y equivalentes).
- `stages/stage01c_qc.json` (posiciones companion/primary).
- `stages/psf_model.json` (C1) — solo para la corrección de apertura del
  producto, NO para la sustracción (SGF no usa modelo espacial).
- Config del run (claves `halosub_*`, `sgf_*`; defaults en
  `docs/00_config_parameters.md`).

## 3. Operaciones

### 3.1 Sustracción (por exposición i del stack)

1. ŝ_i = mediana de spaxels seleccionados (§1.3) del cubo i
   (`halosub.select_reference_spaxels` + `reference_spectrum`).
2. α̂ = SG(d_xy / ŝ_i) por spaxel a lo largo del eje espectral
   (`halosub.sgf_subtract`, NaN-safe por interpolación lineal).
3. residual_i = d − ŝ_i · α̂.

### 3.2 Combinación y extracción

4. Cubo residual combinado según §1.4. Se persiste en float32
   (`stage_x04_sgf_residual_cube.fits`) porque E1b (mapa FoV) lo consume.
5. Producto: `make_aperture_product` sobre el cubo residual combinado,
   apertura `box3`, corrección de apertura `auto` (curva de crecimiento de la
   PSF C1 — el "compensation ratio" cromático del paper §3.4), fondo
   `annulus` con `x01_annulus_bkg_px`, `n_controls = 8` al mismo radio
   (convención x01/x02).

### 3.3 QC físico del método

6. Predictor de auto-sustracción por línea de ciencia (Ec. 1, exacta en el
   modelo de juguete): para cada línea del catálogo estándar, medir C_S/L_S
   en ŝ (altura de línea vs continuo lateral) y R = FWHM de línea esperada
   (LSF de QC/config) / `sgf_window`; registrar
   `self_subtraction_predictor = -(R/(1-R))·(C_S/L_S)`
   (`halosub.sgf_self_subtraction_ratio`).
7. Continuo negativo: fracción de canales con flujo < −2σ en las ventanas
   laterales (±40 Å, excluida la línea) de cada línea de ciencia del producto
   del compañero.

## 4. Puntos de máxima atención

- La división d/ŝ es inestable en canales de bajo flujo (paper Fig. 2a): los
  canales donde |ŝ| < 1e-6 × mediana(|ŝ|) van a NaN ANTES del filtro y se
  interpolan, nunca se filtran con valores espurios.
- El stack 4D se procesa por exposición; prohibido promediar primero y
  sustraer después (cambia el ruido del ratio y no es lo que hace el paper).
- ŝ se calcula EXCLUYENDO al compañero; si la exclusión deja
  `n_spaxels_kept` < 50, abortar con error claro (campo demasiado pequeño
  para diversidad espectral).
- Los controles usan la MISMA ŝ y el MISMO cubo residual que el objeto (no se
  re-estima por control).

## 5. Esquema de `spec_sgf_qc.json`

```
stage, spec_version="C5_v1", run_id
sgf: {window, degree, combine, n_exposures, n_spaxels_kept: [por exp],
      ref_exclude_radius_px}
self_subtraction_predictor: [{line, rest_A, cs_over_ls, R, predictor}]
negative_continuum: [{line, frac_below_minus2sigma}]
errors: {mode, stat_vs_empirical_median_ratio}
products: {object, controls_npz, residual_cube, reference_spectra_npy}
checks: {v1..v5 de §6}
open_issues: [...]
```

## 6. Verificaciones

- v1_reference_ok: `n_spaxels_kept ≥ 50` en todas las exposiciones.
- v2_far_continuum_ok: mediana del residuo en los controles, en bandas de
  continuo lejos de líneas, compatible con 0 (|mediana| < 2·σ/√n_chan).
- v3_predictor_written: predictor Ec. 1 registrado para toda línea estándar
  (Hα, Hβ, O I) que caiga dentro de la cobertura espectral del cubo; al menos
  una debe estar en rango.
- v4_scale_convention_ok: headers `BKGMODE`/`SCALEREF` presentes y
  `control_spectra_cal` persistidos (insumo D1 v3).
- v5_no_pca: el QC declara explícitamente `pca_applied = false`.

## 7. Tests

- Sintético estilo `test_stage01c_synthetic`: cubo con halo cromático + línea
  del compañero; el producto C5 debe mostrar la auto-sustracción predicha por
  Ec. 1 dentro del 25% (la referencia ya no es perfecta) y el flujo lejano de
  continuo de los controles compatible con 0.
- Contrato de producto: headers obligatorios, malla λ idéntica al cubo madre,
  controles en escala calibrada.
- Los oráculos del núcleo ya están cubiertos por `tests/test_halosub_toy.py`.

## 8. Protocolo de parada y reporte

Si el stack no tiene STAT, seguir con errores empíricos (mismo criterio que
x02) y registrar `open_issue`. Si `n_spaxels_kept < 50` o la malla λ difiere
del cubo madre: abortar, nunca degradar en silencio. El reporte final imprime
la ruta del QC y los checks; el notebook de revisión C5 (generado por
`scripts/build_review_notebooks.py`) audita este QC.
