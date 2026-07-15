# Especificación C6 · `X05_lpm` — sustracción de halo por modulación polinomial de Legendre (LPM)

Fecha: 2026-07-14. Etapa C6 del plan
`docs/plan_integracion_halosub_julo2025.md` (WP-H1). Método propuesto por
Julo et al. 2025 (arXiv:2509.09878, App. A.4). Infraestructura: WP-H0
(`musepipe/halosub.py`, oráculos en `tests/test_halosub_toy.py`).

Protocolo spec-first: esta spec se commitea ANTES de ejecutar la etapa sobre
datos reales. El QC registra `spec_version="C6_v1"`. Cambiar el grado u otras
elecciones de §1 mirando estadísticos de un run = revisión C6_v2 con
justificación escrita.

## 0b. Erratas v1.1 (2026-07-15, tras la primera ejecución real — corregidas ANTES de re-ejecutar)

La primera ejecución sobre `ROXs12b_realigned` reveló dos defectos de
ESPECIFICACIÓN (no ajustes de umbral mirando estadísticos de ciencia):

1. **Higiene de canales del ajuste (§3.1)**: el cubo real tiene 215 canales
   NaN en >50% de los spaxels (pero no en todos), para los que `nanmedian`
   produce una ŝ finita → quedaban dentro de la máscara común y empujaban al
   97% de los spaxels al camino lento (v4 falso espurio). Corrección: la
   máscara del ajuste excluye además los canales con fracción finita entre
   los spaxels < `lpm_min_channel_finite_frac` = 0.75 (congelado; es higiene
   de datos, no umbral científico). Con ello el camino lento vuelve a medir
   lo que la spec pretendía (NaNs dispersos propios del spaxel).
2. **Curva MSE (§3.3.6)**: v1 usaba ŝ como "estrella", que pertenece al
   espacio columna del modelo POR CONSTRUCCIÓN (columna de grado 0) →
   underfit ≡ 0 y argmin degenerado al grado máximo (warning espurio).
   Corrección: la estrella del oráculo es un SPAXEL REAL seleccionado
   determinísticamente (el de flujo integrado mediano dentro de la selección
   de referencia), SUAVIZADO con mediana móvil de 301 canales — la deformación
   cromática es suave por hipótesis (premisa del LPM) y sin el suavizado el
   ruido propio del spaxel infla el underfit con una pendiente trivial de
   −σ² por grado. Además, `mse_check_warn` solo se emite si la curva expresa
   preferencia real: profundidad relativa (total_max − total_min)/total_min
   ≥ 1%; con curva plana el argmin es ruido y el warning se anula (None,
   registrado como `mse_curve_flat`).

El QC re-ejecutado registra `spec_version="C6_v1.1"`. El QC v1 previo queda
superseded.

## 0. Rol, objetivo y definición de terminado

C6 produce el sexto método de extracción: espectros congelados del compañero
y controles tras sustraer el halo estelar modelando cada spaxel como una
**modulación polinomial de Legendre del espectro estelar de referencia**,
estimada por mínimos cuadrados (proyección ortogonal) con las líneas de
ciencia ENMASCARADAS del ajuste. Es la alternativa del paper al SGF: preserva
flujo y perfil de las líneas (sin auto-sustracción estructural) y conserva
mejor el continuo vecino.

Terminado = productos `spec_lpm_object.fits` + controles npz + cubo residual
mediano + mapas de coeficientes + `spec_lpm_qc.json` con los checks de §6.

## 1. Límites duros (parámetros congelados)

1. Grado de modulación `lpm_degree = 4` (consenso del paper §3.2 por tres
   vías independientes: MSE analítico, mapas de coeficientes, energy-share).
   El QC `lpm_degree_check` (§3.3) puede emitir warning, NUNCA cambiar el
   grado en el mismo run.
2. Base: polinomios de Legendre sobre λ escalada a [−1, 1]
   (`halosub.lpm_design_matrix`); prohibida la base monomial canónica (Gram
   mal condicionada, paper App. A.4).
3. Máscara del ajuste (`halosub.lpm_fit_mask`): líneas de ciencia
   `lpm_masked_lines_A` (default: ventanas estándar Hα/Hβ/O I de
   `STANDARD_LINE_WINDOWS_A`; por target puede EXTENDERSE vía config
   congelada en el checkpoint del run — p.ej. CaII/HeI para objetos tipo
   YSES1 b) + canales de ventana mala/skyline. El modelo SIEMPRE se evalúa
   sobre todos los canales (interpola a través de las líneas).
4. Espectro de referencia, exclusión del compañero, procesamiento por
   exposición y combinación de residuos: idénticos a C5 §1.3–§1.4 (misma
   claves `halosub_*`).
5. Convención de escala D1 v2 §3.1: controles = objeto,
   `SCALEREF = "normrad_total_flux"`, `BKGMODE = "lpm_residual+annulus"`,
   `INCUBE` = cubo madre stage02.
6. Sin PCA sobre residuos. Sin pesos por varianza en v1 (OLS plano, como el
   paper; la ponderación por matrices de precisión es trabajo futuro
   explícito, paper §4.2).

## 2. Entradas

Las mismas de C5 §2 (stage02 stack, stage04b best, stage01c posiciones,
psf_model.json de C1 para apcorr del producto, config `halosub_*`/`lpm_*`).

## 3. Operaciones

### 3.1 Sustracción (por exposición i)

1. ŝ_i como en C5 §3.1.
2. M_i = matriz de diseño Legendre×ŝ_i de grado 4; una sola pseudo-inversa
   sobre los canales de la máscara común, aplicada a todos los spaxels
   (`halosub.lpm_subtract`). Spaxels con NaNs propios dentro de la máscara →
   lstsq por spaxel (camino lento; su conteo va al QC).
3. residual_i = d − M_i β̂; los coeficientes β̂ (grado+1 planos) se guardan
   por exposición para el QC de §3.3.

### 3.2 Combinación y extracción

4. Idéntico a C5 §3.2: cubo residual combinado persistido en float32
   (`stage_x05_lpm_residual_cube.fits`), producto `make_aperture_product`
   box3 + apcorr auto + annulus + 8 controles.

### 3.3 QC de tuning del grado (diagnóstico, nunca ajuste al vuelo)

5. **Energy-share** (paper Fig. 8): ajuste diagnóstico adicional con grado 9
   sobre la PRIMERA exposición; energía por grado
   (`halosub.lpm_coefficient_energy_share`). Regla congelada
   `lpm_degree_check`: warning si share(grado 5) >
   `lpm_energy_share_warn_factor` (= 2.0) × mediana(shares grados 6–9) — es
   decir, si queda estructura física por encima del grado 4.
6. **Curva MSE analítica** (paper Fig. 6 / Ec. B.5): con ŝ de la primera
   exposición como estrella, σ = sigma robusto del residual en los controles,
   y línea planetaria gaussiana (FWHM = LSF) a Hα con altura = 5σ, evaluar
   `halosub.lpm_mse_terms` para grados 1..9 y registrar la curva y su argmin.
   Diagnóstico: si argmin ∉ [3, 7], warning `lpm_mse_check`.
7. **Mapas de coeficientes** (paper Fig. 7): planos β̂_k de la primera
   exposición persistidos (`stage_x05_lpm_coeff_maps.fits`) para el notebook
   de revisión (estructura de PSF: radio AO, spikes, anillos).

## 4. Puntos de máxima atención

- El número de condición de M enmascarada va al QC; si > 1e8, abortar (base
  degenerada — señal de ŝ patológica o máscara que vacía el rango).
- La máscara de líneas debe cubrir TODAS las líneas de ciencia del target; una
  línea de ciencia sin enmascarar reintroduce auto-sustracción parcial (peor:
  silenciosa). El QC lista `masked_lines_A` y el notebook la audita contra el
  catálogo G2 del run.
- Camino lento (lstsq por spaxel): si supera el 20% de los spaxels, warning —
  el stack trae demasiados NaNs espectrales y conviene revisar B/A2.
- Igual que C5: ŝ por exposición excluyendo al compañero, abortar si
  `n_spaxels_kept < 50`.

## 5. Esquema de `spec_lpm_qc.json`

```
stage, spec_version="C6_v1", run_id
lpm: {degree, masked_lines_A, combine, n_exposures, n_spaxels_kept: [por exp],
      condition_number: [por exp], n_slow_spaxels: [por exp],
      slow_fraction_max}
degree_diagnostics: {energy_share: [g1..g9], degree_check_warn: bool,
                     mse_curve: [{degree, star_underfit, planet_overfit,
                                  noise_overfit, total}], mse_argmin,
                     mse_check_warn: bool}
errors: {mode, stat_vs_empirical_median_ratio}
products: {object, controls_npz, residual_cube, coeff_maps,
           reference_spectra_npy}
checks: {v1..v6 de §6}
open_issues: [...]
```

## 6. Verificaciones

- v1_reference_ok: `n_spaxels_kept ≥ 50` en todas las exposiciones.
- v2_line_preservation_ok: en un test de inyección interna barata (línea
  gaussiana sintética de 5σ inyectada en UN control, sustraída con la misma
  M), recuperación del flujo de línea ≥ 90% (contrasta con el déficit
  esperado del SGF; es un smoke interno, la calibración formal es E4).
- v3_condition_ok: número de condición < 1e8 en todas las exposiciones.
- v4_slow_path_ok: fracción de spaxels por camino lento ≤ 20%.
- v5_scale_convention_ok: headers y `control_spectra_cal` como en C5 v4.
- v6_degree_diagnostics_written: energy-share, curva MSE y mapas de
  coeficientes persistidos (warnings NO bloquean; se revisan en el notebook).

## 7. Tests

- Sintético: cubo con halo cromático suave (modulación cuadrática del
  espectro estelar) + línea del compañero → producto C6 recupera el flujo de
  línea dentro del 5% y residuos de controles compatibles con 0.
- Grado insuficiente a propósito (modulación cúbica, `lpm_degree=1`) → el
  residuo estelar sube y la curva MSE lo refleja (star_underfit decreciente
  con el grado): protege la semántica del diagnóstico.
- Contrato de producto: como C5 §7.
- Núcleo ya cubierto por `tests/test_halosub_toy.py` (Fig. 2d, Ec. 1, B.5).

## 8. Protocolo de parada y reporte

Idéntico a C5 §8 (STAT ausente → errores empíricos + open_issue; malla λ
distinta o referencia insuficiente → abortar). El notebook de revisión C6
audita el QC, los mapas de coeficientes y los dos warnings de grado.
