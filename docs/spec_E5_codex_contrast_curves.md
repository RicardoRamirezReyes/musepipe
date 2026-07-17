# Especificación E5 · `H05_contrast` — curvas de contraste por inyección en anillos

Fecha: 2026-07-14. Etapa E5 del plan
`docs/plan_integracion_halosub_julo2025.md` (WP-H5). Método: Julo et al. 2025
§3.3.2 (Fig. 10), inyecciones en anillos concéntricos (Andres 1994).

Protocolo spec-first: commiteada antes de ejecutar sobre datos reales. QC con
`spec_version="E5_v1"`. Grillas y umbrales de §2 congelados.

## 0. Rol, objetivo y definición de terminado

E5 mide el **contraste mínimo detectable vs separación** para los métodos de
sustracción de halo con cubo residual (sgf, lpm; opcionalmente
optimal_psfsub), inyectando líneas falsas en una grilla de anillos y
validando la detección sobre el mapa matched-filter de E1b. Produce la curva
de contraste al 50% de detección con bandas 25/75% (Fig. 10 del paper). Sus
filas de inyección alimentan E6 (ROC).

Terminado = `tables/contrast_curve_by_method.csv` +
`tables/contrast_injections.csv` + `stage_h05_qc.json` + plot.

## 1. Límites duros

1. E5 usa la MISMA cadena de detección que E1b (plantilla espectral, kernel
   C1, normalización por anillos): la curva mide la sensibilidad del pipeline
   real, no la de un detector ad-hoc.
2. **Linealidad**: para ŝ fija, la sustracción sgf/lpm es lineal → el residual
   de (base + inyección) = residual(base) + kernel(inyección). E5 precalcula
   el residual base, su mapa de correlación y las estadísticas de anillos UNA
   vez por método; por inyección solo evalúa la contribución del delta en su
   posición. Exacto (misma propiedad verificada en WP-H2 contra fuerza
   bruta). `optimal_psfsub` no es por-spaxel independiente: solo se incluye
   con `h05_include_psfsub=true` (re-sustracción completa por caso, coste
   documentado; default false).
3. La compañera real (si existe) queda excluida: no se inyecta a < 2×FWHM de
   su posición y sus píxeles no entran en las estadísticas de anillos
   (herencia E1b).
4. E5 no emite límites físicos (eso es E3): su producto es la curva de
   contraste instrumental por método.

## 2. Operaciones y parámetros congelados

1. **Referencia estelar**: F_star = flujo del cubo madre integrado en
   apertura de radio `psf_norm_radius_px` (NORMRAD, 25 px) alrededor de la
   estrella y en la ventana de línea (±FWHM de LSF alrededor de
   `h01b_line_rest_A` corregida por rv), × dλ. Contraste c → flujo de línea
   inyectado = c · F_star.
2. **Grilla** (congelada; overrides = revisión de spec):
   - separaciones: desde `h05_r_min_px` = 4.0 hasta el radio útil
     (borde − kernel) en pasos de `h05_r_step_px` = 4.0 (~1 FWHM);
   - ángulos: `h05_n_angles` = 8 (paso 45°, offset 0°);
   - contrastes: grilla logarítmica `h05_contrasts` = 9 puntos en
     [1e-5, 1e-2] (paper §3.3.2).
3. **Inyección**: línea gaussiana FWHM = LSF × PSF cromática C1 en la
   posición (misma rutina E4, `musepipe/injection.py`).
4. **Detección**: z = (corr(pos) − μ_r)/σ_r ≥ `h05_threshold_sigma` = 5.0,
   con corr(pos) = corr_base(pos) + corr_delta(pos) y μ_r/σ_r de los anillos
   del mapa base (E1b §2.2). corr_base(pos) usa la realización de ruido REAL
   de esa posición (como el paper, que valida contra μ̂+5σ̂ con las
   realizaciones reales).
5. **Curva**: por (método, separación): contraste_50 = mínimo c de la grilla
   con fracción de detección ≥ 0.5 sobre los ángulos; ídem 0.25 y 0.75 para
   la banda. Sin interpolación entre puntos de la grilla en v1 (se reporta la
   grilla completa para E6/notebook).

## 3. QC y verificaciones

```
stage="h05_contrast", spec_version="E5_v1", run_id
params: {grid, threshold_sigma, f_star_line, line_center_A, lsf_fwhm_A}
methods: {<m>: {n_injections, curve: [{separation_px, contrast_50,
                contrast_25, contrast_75, n_angles}]}}
checks: {v1_curves_written, v2_monotonic_trend, v3_grid_saturation}
```

- v1: curva escrita para todo método disponible.
- v2 (sanidad, no bloqueante): la mediana de contrast_50 en el tercio externo
  ≤ mediana en el tercio interno (la sensibilidad mejora hacia afuera).
- v3: fracción de separaciones donde contrast_50 quedó en el borde de la
  grilla (saturada) — si > 50%, warning (grilla mal dimensionada para el
  target).

## 4. Tests

- Sintético con ruido plano + PSF constante: la curva recuperada escala como
  σ_ring (contraste_50 ∝ σ/F_star) y es ~plana; con ruido radial decreciente,
  contrast_50 decrece hacia afuera (v2).
- Exactitud del camino delta: detección de una inyección puntual coincide con
  la de la re-sustracción completa (misma decisión) en una muestra de casos.
- Contrato: CSVs y QC escritos, checks evaluados.

## 5. Protocolo de parada y reporte

Sin cubos residuales → abortar (correr C5/C6). El notebook E5 grafica la
curva tipo Fig. 10 (contraste vs separación en arcsec con la escala 25
mas/px) con las bandas 25/75%, y marca la compañera real si existe.
