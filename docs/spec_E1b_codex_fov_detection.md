# Especificación E1b · `H01b_fovmap` — detección ciega por matched filter espacio-espectral

Fecha: 2026-07-14. Etapa E1b del plan
`docs/2026-07-15_plan_integracion_halosub.md` (WP-H4). Método: Julo et al. 2025
§3.3.1 (Fig. 9) + estadística de ruido por anillos (App. G).

Protocolo spec-first: commiteada antes de ejecutar sobre datos reales. QC con
`spec_version="E1b_v1"`. Umbrales de §3 congelados.

## 0. Rol, objetivo y definición de terminado

E1b produce **mapas de detección sobre todo el campo** a la longitud de onda
de una línea objetivo, desde los cubos residuales de los métodos que los
generan. Es la etapa de detección CIEGA del pipeline multi-target: para
targets nuevos corre ANTES de fijar `companion` en B3 (propone candidatos);
para targets con compañera conocida actúa como test de consistencia con E1.
E6 (ROC) consume sus mapas.

Terminado = `stage_h01b_fovmap_<method>.fits` (mapas por método) +
`stage_h01b_qc.json` (candidatos, estadística de ruido por anillos, checks) +
plot de mapas.

## 1. Límites duros

1. E1b **no re-sustrae**: consume cubos residuales existentes
   (`stage_x04_sgf_residual_cube.fits`, `stage_x05_lpm_residual_cube.fits`).
   Excepción declarada: `optimal_psfsub` no persiste su cubo (x02); E1b lo
   reconstruye con la MISMA rutina de producción
   (`fit_primary_psf_model_cube`, config x02), activable con
   `h01b_include_psfsub` (default true).
2. E1b no emite veredictos científicos: propone `candidates` con su z; la
   promoción de un candidato a `companion` (B3) es SIEMPRE decisión del
   usuario en checkpoint.
3. La posición conocida de la compañera (si existe en B3) se reporta aparte
   (`known_source`), nunca como candidato nuevo.
4. Sin PCA y sin whitening adicional: el mapa es matched filter + normalización
   por anillos, nada más (v1).

## 2. Operaciones

### 2.1 Matched filter (por método, paper §3.3.1)

1. **Modelo espectral** f: gaussiana con FWHM = LSF (QC/config), centrada en
   `h01b_line_rest_A` (default Hα 6562.8) corregida por `rv_sys_kms` del run;
   normalizada ‖f‖=1 sobre canales buenos (sin flags, dentro de la ventana de
   trabajo del cubo residual).
2. **Mapa espectral** M(y,x) = Σ_λ f(λ)·residual(λ,y,x) (NaN-safe; píxel con
   < 50% de canales buenos del soporte de f → NaN).
3. **Kernel espacial**: PSF cromática C1 evaluada en λ de la línea, recortada
   a `h01b_kernel_halfsize_px` (default 7) y normalizada ‖K‖=1. Fallback si no
   hay `psf_model.json`: kernel empírico = cutout del mapa Σ f·cubo_madre en
   la estrella (registrado en QC como `kernel_source`).
4. **Mapa de correlación** C(y,x) = (K ⋆ M)(y,x) (correlación cruzada,
   NaN-safe).

### 2.2 Normalización por anillos y candidatos (Andres 1994 / App. G)

5. Anillos enteros de radio r alrededor de la estrella (agrupados de a
   `h01b_ring_width_px` = 2): μ_r y σ_r robustos (mediana / sigma robusto)
   de C en cada anillo, EXCLUYENDO un disco de 1.5×FWHM alrededor de la
   compañera conocida si existe. z(y,x) = (C − μ_r)/σ_r.
6. **Candidatos**: máximos locales de z con z ≥ `h01b_threshold_sigma` (= 5.0,
   congelado) y r ≥ `h01b_r_min_px` (= 3.0, núcleo AO). Se registra (y, x, r,
   z, método). La compañera conocida se reporta con su z en `known_source`.
7. **QC de gaussianidad por anillos** (App. G): por grupo de anillos,
   sigma robusto, asimetría, curtosis en exceso y fracción de |z|>3 con su
   valor esperado gaussiano (0.0027) — el notebook los grafica (histograma +
   QQ). Ningún check bloquea por no-gaussianidad: se REGISTRA (es la física
   del residuo cerca del núcleo) y motiva que E3/E6 usen estadística empírica.

## 3. Esquema de `stage_h01b_qc.json` y verificaciones

```
stage="h01b_fovmap", spec_version="E1b_v1", run_id
params: {line_rest_A, line_center_A, lsf_fwhm_A, threshold_sigma, r_min_px,
         kernel_halfsize_px, ring_width_px, kernel_source}
methods: {<method>: {map_fits, candidates: [...], known_source: {...} | null,
                     ring_noise: [{r_lo, r_hi, sigma, skew, kurtosis_excess,
                                   frac_abs_z_gt3}], n_pixels_valid}}
checks: {v1_maps_written, v2_known_source_reported, v3_ring_noise_written,
         v4_no_candidate_at_star_core}
open_issues: [...]
```

- v1: un mapa FITS por método disponible (mínimo 1 método).
- v2: si B3 define compañera, su z se reporta en todos los métodos.
- v3: estadística de anillos escrita para todos los métodos.
- v4: ningún candidato con r < r_min (comprobación de la máscara).

## 4. Tests

- Sintético: cubo residual = ruido gaussiano + fuente puntual con línea
  gaussiana (5–8σ) a radio conocido → candidato recuperado a ≤1 px y z ≥ 5;
  cubo de solo ruido → 0 candidatos (con umbral 5σ y campo pequeño).
- Anillos: ruido con σ dependiente del radio → `ring_noise` lo refleja y la
  normalización por anillos mantiene la calibración de z (fracción |z|>3
  compatible con gaussiana por anillo).
- Contrato: QC schema, mapas escritos, v1–v4.

## 5. Protocolo de parada y reporte

Sin cubos residuales disponibles → abortar con mensaje accionable (correr
C5/C6). El notebook E1b muestra los mapas (Fig. 9 del paper), la tabla de
candidatos y el QQ de ruido por anillos (App. G).
