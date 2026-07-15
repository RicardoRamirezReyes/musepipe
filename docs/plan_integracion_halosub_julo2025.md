# Plan · Integración de los métodos de sustracción de halo de Julo et al. 2025

Fecha: 2026-07-14. Paper de referencia: Julo, Bonnefoy, Chatelain et al. 2025,
arXiv:2509.09878 ("Stellar halo subtraction alternative for accreting
companions' characterization with integral field spectroscopy").
Rama de partida: `stage-d1-v2`.

## 0. Decisiones de alcance (usuario, 2026-07-14)

Registradas en el checkpoint de planificación de esta fecha:

1. **Métodos de primera clase**: se implementan **SGF** (Savitzky-Golay
   filtering, el "estado del arte" de Haffert et al. 2019 / Xie et al. 2020,
   fiel al paper: SG de orden d=1, ventana W̆=101 canales por defecto) **y
   LPM** (Legendre Polynomial Modulation, grado ∂=4 por defecto, línea
   enmascarada en el ajuste). Ambos entran a musepipe como métodos de
   extracción congelados, al mismo nivel que aperture/optimal/psffit.
2. **Comparación**: se escribe **spec D1 v3** con `METHOD_ORDER` extendido a
   **6 métodos** (`aperture, optimal_ls, optimal_psfsub, psffit, sgf, lpm`).
   Un solo veredicto y un solo `recommended_method`, como hasta ahora.
3. **Diagnósticos de detección**: se implementan los **tres** del paper:
   mapas matched-filter espacio-espectrales sobre el FoV (Fig. 9), curvas de
   contraste vs separación con inyecciones en anillos (Fig. 10) y curvas ROC
   (Fig. 11). Los diagnósticos de tuning del LPM (descomposición del MSE,
   mapas de coeficientes, energy-share; Figs. 6–8) van incluidos con el LPM.
4. **Selección de método por caso**: se mantiene el principio de la spec D1
   ("la elección del canónico es del usuario"); D1 v3 emite
   `recommended_method` con un **árbol de decisión extendido** con los
   criterios de régimen del paper (contraste continuo/línea, C_S/L_S,
   separación, si el continuo de la compañera es ciencia). No hay selección
   automática.

Motivación: el pipeline es multi-target (ya existen runs de LkCa_15 y specs
pensadas para reuso); los targets nuevos pueden caer en el régimen donde el
psffit espacial no basta (compañera sin continuo detectable, separaciones
pequeñas, necesidad de detección ciega) y donde los sesgos del filtrado
espectral (auto-sustracción, continuo negativo) que cuantifica el paper son
relevantes.

## 1. Estado actual vs. paper (diagnóstico 2026-07-14)

Ya implementado (misma idea, a veces otra técnica):

- Inyecciones falsas con línea Hα gaussiana × PSF cromática (E4,
  `stage_h04_injection` + `musepipe/injection.py`), corrección de throughput
  consumida por E3 — es el "tratamiento correctivo" que el paper §3.4 acepta
  como alternativa (Jorquera et al. 2024).
- Matched filter espectral con plantilla LSF a posición fija (E1,
  `matched_filter_point/scan`), FAP empírica de controles, look-elsewhere en λ
  (stage08c).
- Ruido a igual separación/distinto ángulo (`same_radius_control_positions`,
  filosofía de anillos Andres 1994) y colas no gaussianas vía cuantiles
  empíricos + Gumbel (E3).
- Corrección de apertura dependiente de λ (`aperture_correction_from_psf` +
  PSF cromática C1) — el "compensation ratio" del paper.
- Enmascarado de línea en la estimación de continuo 1D (`lines.py`, G2) y
  Ṁ desde Hα vía L_acc–L_Hα (G3/E3).

No implementado (huecos que cubre este plan):

- El método **LPM** (contribución central del paper) y el **SGF** formalizado
  (la cadena legacy `03_fakecont` usa kernel gaussiano σ=230 canales, sin
  enmascarar Hα, y vive fuera de musepipe).
- Diagnósticos de tuning del LPM: descomposición del MSE (App. B.2.2), mapas
  de coeficientes (Fig. 7), energy-share (Fig. 8).
- Mapas de detección matched-filter sobre todo el FoV (detección ciega).
- Curvas de contraste vs separación con grilla de anillos.
- Curvas ROC.

## 2. Principios de integración (límites duros)

Heredados de las specs vigentes; este plan NO los modifica:

1. **Spec-first / anti cherry-picking**: cada etapa nueva se especifica y se
   commitea ANTES de ejecutarse sobre datos reales (protocolo D1 v2 §0). Los
   hiperparámetros por defecto (W̆=101, d=1, ∂=4, máscaras) quedan congelados
   en la spec; cambiarlos mirando estadísticos del run exige nueva versión.
2. **Convención de escala "control = objeto"** (D1 v2 §3.1): los productos
   sgf/lpm procesan los controles EXACTAMENTE como el objeto y persisten en la
   escala física común (`SCALEREF = "normrad_total_flux"`, header `BKGMODE`
   propio). Sin esto, D1 v3 los degradaría en el gate de controles.
3. **D1 solo lee productos**; no re-extrae ni re-normaliza. Los métodos nuevos
   llegan a D1 como `SpectrumProduct` congelados de su etapa C.
4. **Pares primarios = métodos validados por G1**; la validación (covarianza,
   bias budget con `throughput_loss`, umbral `bias_bounded` ≥ 0.4) aplica a
   sgf/lpm igual que al resto.
5. **El canónico lo fija el usuario** en el checkpoint final; D1 v3 escribe
   `recommended_method`, nunca `canonical_method`.
6. Suite de tests verde en cada fase; notebooks de revisión regenerados por
   `scripts/build_review_notebooks.py` para cada spec nueva.

## 3. Arquitectura de la integración

Los dos métodos nuevos son de **familia espectral** (diversidad espectral por
spaxel), a diferencia de los cuatro existentes (familia espacial). Encajan en
el C-block como dos etapas nuevas que producen (a) un **cubo residual** por
exposición + mediana, y (b) **SpectrumProducts congelados** extraídos del cubo
residual reutilizando la maquinaria existente (apertura box3 + apcorr
cromática + fondo de anillo, como hace `optimal_psfsub` sobre
`stage02 − modelo PSF`).

```
A (reducción) → B (align/localize) → C1 (PSF cromática)
                                      ├─ C2 aperture      (x01)
                                      ├─ C3 optimal ls/psfsub (x02)
                                      ├─ C4 psffit        (x03)
                                      ├─ C5 sgf           (x04)  ← NUEVO
                                      └─ C6 lpm           (x05)  ← NUEVO
G1 (validación 6 métodos) → E4 (throughput 6 métodos)
        → D1 v3 (x10, 6 métodos, árbol extendido) → checkpoint usuario
        → D2 (x11, canónico) → E1/E2/E3 → F1 → G2..G5
Detección (nuevo sub-bloque):
  E1b mapa matched-filter FoV (h01b) · E5 curvas de contraste (h05)
  · E6 ROC (h06)
```

Nomenclatura interna consistente con la existente: `stage_x04_sgf.py`,
`stage_x05_lpm.py`, `stage_h01b_fovmap.py`, `stage_h05_contrast.py`,
`stage_h06_roc.py`. Specs: `spec_C5_codex_sgf_subtraction.md`,
`spec_C6_codex_lpm_subtraction.md`, `spec_D1_v3_codex_method_comparison.md`,
`spec_E1b_codex_fov_detection.md`, `spec_E5_codex_contrast_curves.md`,
`spec_E6_codex_roc_curves.md`. Notebooks de revisión: C5, C6, D1v3, E1b, E5,
E6.

## 4. Fases y paquetes de trabajo

### Fase 0 · Infraestructura común (`musepipe/halosub.py`) — WP-H0

Módulo compartido por C5/C6:

- **Estimación del espectro estelar de referencia** ŝ por exposición:
  mediana de spaxels seleccionados por máscara de flujo (excluir
  F < 0.01·Fmax y F > 0.1·Fmax; Tabla 1 del paper), claves de config
  `halosub_flux_mask_lo/hi`.
- **Ventana espectral de trabajo**: por defecto la banda completa del cubo con
  el notch Na-LGS ya descartado (B-block); opción `halosub_wave_range_A` para
  reproducir la ventana 6051–7075 Å del paper. Máscaras de líneas
  configurables (`halosub_masked_lines`, default `["halpha"]`, extensible a
  Hβ/CaII para targets tipo YSES1) reutilizando `musepipe/constants.py`.
- **Oráculos analíticos del paper como tests**: el paper da fórmulas cerradas
  que sirven de verdad-terreno para tests unitarios sintéticos:
  - profundidad de auto-sustracción del SGF: C̃_P/L̂_P = −(R/(1−R))·(C_S/L_S)
    (Ec. 1) y las expresiones de pico SG de App. B.1;
  - descomposición del MSE del LPM en underfit-estrella / overfit-planeta /
    overfit-ruido con término σ²·(ℓ−(∂+1)) (Ec. B.5);
  - distribución de residuos post-LPM (Ec. B.3).
  Tests: cubo de juguete (modelo rectangular Fig. 1 + versión realista tipo
  App. E reducida) donde SGF debe reproducir la auto-sustracción predicha y
  LPM debe recuperar el flujo de línea dentro de tolerancia.

Entregables: `musepipe/halosub.py`, `tests/test_halosub_toy.py`, claves de
config documentadas en `docs/00_config_parameters.md`.

### Fase 1 · C5 (SGF) y C6 (LPM) — WP-H1

**C5 `stage_x04_sgf.py`** (spec fiel a App. A.3):

1. Por exposición del stack de stage02: ratio spaxel/ŝ → filtrado
   Savitzky-Golay (`sgf_degree=1`, `sgf_window=101` congelados en spec) →
   estimación estelar = ŝ · ratio_suavizado → cubo residual.
2. Mediana de los cubos residuales de las exposiciones (como el paper).
3. **Sin PCA sobre residuos** (el paper la desaconseja; queda como opción
   futura explícitamente NO activada, coherente con la opción (b) abierta en
   `d1_canonical_method_decision.md`).
4. Producto: extracción sobre el cubo residual en la posición del compañero +
   3 controles al mismo radio, reutilizando `make_aperture_product` (box3,
   apcorr cromática de C1, fondo `annulus`) para cumplir la convención de
   escala. Headers `BKGMODE="sgf_residual+annulus"`, `SCALEREF` común.
5. QC propio: predictor de auto-sustracción por línea (Ec. 1 evaluada con
   C_S/L_S medido de ŝ y R = ancho de línea/W̆), fracción de canales con
   continuo negativo alrededor de las líneas de ciencia, nº de spaxels usados
   en ŝ.

**C6 `stage_x05_lpm.py`** (spec fiel a App. A.4):

1. Matriz de diseño M ∈ ℝ^(ℓ×(∂+1)) con columnas = polinomios de Legendre
   λ̃^k modulando ŝ; **una sola pseudo-inversa** con la máscara de canales
   común (líneas enmascaradas + flags de ventana mala/skyline) aplicada a
   todos los spaxels — el coste es O(ℓ·∂) por spaxel, despreciable. Spaxels
   con NaNs propios caen a un ajuste por-spaxel (camino lento, contado en QC).
2. ŝ_xy = Π_M d_xy (proyección ortogonal), residual = d_xy − ŝ_xy; mediana
   entre exposiciones; mismo esquema de producto que C5.
3. Grado `lpm_degree=4` congelado; `lpm_masked_lines` por target.
4. **QC de tuning (Figs. 6–8 del paper)**:
   - descomposición del MSE sobre la simulación de juguete apareada al run
     (curva MSE vs ∂, mínimo esperado en 4–7);
   - mapas de coeficientes por grado (0..9) sobre el cubo real (diagnóstico
     de PSF: radio AO, spikes, anillos de Airy);
   - curva de energy-share por grado (mediana del campo); si el codo no está
     en ∂≤5, el QC emite warning `lpm_degree_check` (NO cambia el grado; eso
     exigiría revisión de spec).

Entregables por etapa: spec + stage + script de invocación + tests (sintéticos
y de contrato de producto) + notebook de revisión + entrada en
`build_review_notebooks.py`.

### Fase 2 · Validación G1 y throughput E4 para sgf/lpm — WP-H2

1. **E4**: añadir `sgf_extractor` y `lpm_extractor` al registro de
   `build_production_extractors` (stage_h04_extractors). Flujo por caso:
   inyectar en el stack de stage02 → correr la sustracción de halo → extraer.
   - **Optimización necesaria** (el coste ingenuo es re-sustraer el cubo
     completo por cada caso de la grilla E4): la sustracción SGF/LPM es
     independiente por spaxel y ŝ excluye los spaxels brillantes, así que la
     inyección solo altera el soporte espacial de la PSF inyectada →
     re-sustraer únicamente los spaxels afectados y verificar en QC que la
     perturbación de ŝ es < tolerancia (‖Δŝ‖/‖ŝ‖ registrada). Fallback
     configurable a re-sustracción completa.
2. **G1**: extender `scripts/run_g1.py` a 6 métodos (veredicto por método con
   el mismo bias budget: `throughput_loss` corregible en E3, `bias_bounded`
   ≥ 0.4, sensibilidad posicional y PSF ±10%).
3. **E3**: sin cambio de código esperado (la interpolación de throughput ya es
   genérica por método); verificación explícita con los 6.

### Fase 3 · D1 v3 — WP-H3

`spec_D1_v3_codex_method_comparison.md` (commiteada antes de ejecutar):

1. `METHOD_ORDER = (aperture, optimal_ls, optimal_psfsub, psffit, sgf, lpm)`.
   Bandas B1–B6/LHα/LHβ/LOI re-congeladas sin cambio. Pares primarios =
   validados por G1 (regla v2 intacta). Scale-check y gate por par intactos.
2. **Árbol de decisión extendido** (guía del `recommended_method`, checkpoint
   humano intacto). Insumos nuevos en el QC:
   - régimen de contraste: C_S/L_S de ŝ en cada línea de ciencia, contraste
     compañera/estrella en línea y en continuo (de los productos);
   - predictor de auto-sustracción SGF por línea (Ec. 1) y su versión medida
     (SGF vs LPM en las bandas de línea);
   - flag `companion_continuum_is_science` (config por run).
   Reglas congeladas en la spec (resumen): compañera dominada por líneas y
   continuo no-ciencia → psffit y lpm equivalentes, sgf aceptable solo si el
   predictor de auto-sustracción < umbral; continuo de la compañera es ciencia
   → excluir sgf de la recomendación (pérdida irrecuperable de información,
   §2.1 del paper) y preferir psffit/lpm; divergencia entre familias espacial
   y espectral en bandas de continuo → tratarla como sistemática presupuestada
   (mismo criterio que B6) salvo que supere umbral congelado.
3. `stage_x11_calibrate` no cambia (importa `METHOD_ORDER`); acepta `sgf`/`lpm`
   como canónico si el usuario lo fija.
4. Los runs históricos con QC `spec_version="D1_v2"` quedan como están; v3 no
   los re-interpreta.

### Fase 4 · Diagnósticos de detección — WP-H4 (E1b), WP-H5 (E5), WP-H6 (E6)

**E1b `stage_h01b_fovmap.py` — mapa matched-filter FoV** (Fig. 9):

- Insumo: cubos residuales medianos de los métodos que los producen
  (`optimal_psfsub`, `sgf`, `lpm`).
- Modelo espectral: gaussiana FWHM = LSF (~3 canales), centrada en la línea
  objetivo; modelo espacial: producto punto de los spaxels del cubo
  PRE-sustracción con el modelo espectral (hipótesis de campo pequeño, §3.3.1
  del paper); ambos normalizados.
- Salida: mapa de correlación normalizado por método + lista de candidatos
  sobre umbral (excluida la posición conocida si existe) + QC de estadística
  de ruido por anillos (histogramas y QQ-plot tipo App. G; registra desviación
  de gaussianidad vs separación).
- Para ROXs12b actúa como test de consistencia con E1; para targets nuevos es
  la etapa de detección ciega previa a fijar `companion` en B3.

**E5 `stage_h05_contrast.py` — curvas de contraste** (Fig. 10):

- Inyecciones en grilla de anillos concéntricos (Andres 1994): claves
  `h05_separations_px` (default: paso ~1 FWHM desde el radio AO interno hasta
  el borde útil), `h05_angles_deg` (default 8), `h05_contrast_grid`
  (logarítmica 1e-5→1e-2 como el paper), reutilizando `musepipe/injection.py`
  y el registro de extractores de E4 (+ la optimización por-spaxel de WP-H2).
- Validación de detección: flujo integrado > μ̂ + 5σ̂ del anillo (μ, σ de las
  imágenes de banda estrecha en λ de la línea); curva = contraste mínimo con
  >50% de inyecciones detectadas, bandas al 25/75%.
- Presupuesto de cómputo dimensionado en la spec para una noche de ejecución
  con el process-pool de E4; los defaults de grilla se congelan ahí.
- Salida: `contrast_curve_by_method.csv` + plot; alimenta el notebook 10
  (límites de masa) que se migra a consumidor de esta tabla.

**E6 `stage_h06_roc.py` — curvas ROC** (Fig. 11):

- Reutiliza las inyecciones de E5 en 2–3 separaciones representativas y un
  contraste tipo-detección-límite; clasificación por umbral variable sobre el
  mapa E1b; realizaciones de ruido = exposiciones/cubos disponibles × canales
  espectrales fuera de las ventanas de línea (§3.3.3 del paper).
- Salida: curvas DP vs FAP + AUC por método y separación, con banda 1σ;
  QC compara familias (espacial vs espectral) y lo registra como insumo
  informativo del árbol D1 v3 (no cambia veredictos).

Orden interno: E1b primero (E6 depende de su mapa); E5 antes que E6 (comparten
inyecciones).

### Fase 5 · Validación end-to-end y multi-target — WP-H7

1. **Regresión ROXs12b**: re-run completo C1→C6→G1→E4→D1v3→D2→E1–E3 sobre
   `ROXs12b_realigned`. Expectativas registradas ANTES de ejecutar (protocolo):
   canónico recomendado sigue siendo psffit (o lpm si el árbol lo justifica —
   decisión del usuario), límite E3 compatible dentro de errores con el
   vigente, y diferencia SGF−LPM en LHα documentada (el paper predice
   subestimación SGF ~30% en régimen de alto contraste de continuo).
2. **Smoke multi-target**: `runs/LkCa_15` por la cadena moderna B→C5/C6→E1b
   (requiere localización B3 en ese target). Comparación cualitativa contra la
   cadena legacy fakecont+PCA de los notebooks raíz; discrepancias se
   documentan, no se "corrigen" ad hoc.
3. **Backlog opcional** (no bloquea el cierre del plan): reproducción externa
   con datos de archivo de PDS70 (programa 60.A-9100) para validar contra las
   cifras publicadas del paper.

## 5. Qué NO hace este plan

- No toca el A-block ni sus blockers (siguen gobernando la validez paper de
  cualquier cifra).
- No activa PCA sobre residuos (sigue siendo opción futura explícita).
- No migra la cadena legacy fakecont+PCA a musepipe (decisión de alcance:
  LkCa_15 y compañía se atienden corriendo la cadena moderna, no portando la
  vieja). Los notebooks raíz quedan como histórico.
- No automatiza la elección del canónico.

## 6. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Coste E4/E5 con métodos de cubo (re-sustracción por inyección) | Optimización por-spaxel (WP-H2) + verificación ‖Δŝ‖ en QC + grillas congeladas dimensionadas en spec |
| ∂ óptimo del LPM depende del dataset (paper §3.2.1) | ∂=4 congelado + QC energy-share con warning; cambiar ∂ = revisión de spec, nunca ajuste al vuelo |
| Auto-sustracción SGF contamina G2/G3 si sgf fuese canónico | El árbol D1 v3 excluye sgf cuando el continuo es ciencia; predictor Ec. 1 en QC de C5 |
| Continuo negativo post-SGF rompe rutinas aguas abajo (paper §3.5) | Los estimadores E1/G2 ya operan sobre residuos con signo; test explícito con producto sgf sintético negativo |
| Veredicto D1 v3 reabre la sistemática B6 | v3 hereda la decisión "B6 = sistemática presupuestada" (2026-07-10); solo se reabre si la familia espectral la contradice sobre umbral congelado |
| Runs históricos sin productos sgf/lpm | v3 exige los 6 productos; los runs viejos conservan su QC v2 como histórico, sin re-interpretación |

## 7. Orden de ejecución y dependencias

```
WP-H0 (halosub + oráculos analíticos)
  → WP-H1 (C5 SGF ∥ C6 LPM)
    → WP-H2 (E4 extractores + G1 a 6 métodos)
      → WP-H3 (spec D1 v3 + re-run comparación)
      → WP-H4 (E1b FoV map) ∥ WP-H5 (E5 contraste) → WP-H6 (E6 ROC)
        → WP-H7 (regresión ROXs12b + smoke LkCa_15)
```

Tamaños estimados (sesiones de trabajo): H0 ≈ 1, H1 ≈ 2–3, H2 ≈ 1–2, H3 ≈ 1–2,
H4 ≈ 1, H5 ≈ 1–2 (más la ejecución nocturna), H6 ≈ 1, H7 ≈ 1–2. Total ≈ 9–14.

Checkpoints de usuario: (a) aprobación de cada spec antes de ejecutar sobre
datos reales (H1, H3, H4–H6); (b) elección del canónico tras D1 v3 (H3);
(c) aceptación de la regresión (H7).
