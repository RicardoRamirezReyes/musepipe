# Notebooks de revisión A1 → G5

Un notebook por spec de la cadena canónica ROXs 12 (A→G). **Reemplazan** a los
notebooks históricos de la raíz (`00_*`…`Far_*`), que quedaron como interfaz de
inspección y **no reflejan las correcciones recientes**. Son interfaces delgadas
sobre `musepipe/` + `scripts/`: no reimplementan lógica.

Las 30 están **enriquecidas**: cada una explica su etapa, carga la evidencia real
del QC y trae 1–2 figuras — no son esqueletos.

Generados por [`scripts/build_review_notebooks.py`](../scripts/build_review_notebooks.py)
(regenerar: `python scripts/build_review_notebooks.py`, o `... A1 C3` para algunas).
Utilidades comunes en [`_nbcommon.py`](_nbcommon.py).

## Estructura de cada notebook

1. **Spec & rol** — qué hace, input←/output→, link a `docs/spec_*.md`.
2. **Cómo ejecutar de forma independiente** — `conda activate MUSE` + el comando
   canónico exacto (script `.sh`, `python -m …`, `run_*()` o `scripts/*.py`).
3. **Ejecutar o auditar** — guardada por `RUN`: por defecto **audita**; `RUN=True`
   re-ejecuta.
4. **QC / resultados** — carga el JSON del QC.
5. **Qué es / evidencia** — narrativa de la etapa + celda que imprime la evidencia
   real que sostiene la decisión.
6. **Plots** — 1–2 figuras (ver abajo).
7. **Decisiones** y **Conclusión (registrada)** — con fecha y datos.

## Cómo se usan

- **Por defecto AUDITAN** el QC existente del run `ROXs12b_realigned` (el run
  científico: F1 `yellow`, 0 rojos bloqueantes). Las celdas de auditoría solo leen
  JSON — se abren sin la pila científica.
- **Los plots necesitan el kernel MUSE** (`conda activate MUSE`: astropy, pandas,
  matplotlib) y, algunos, los cubos en `/mnt/2TB`. Fallan con un mensaje claro si
  falta algo, sin romper el notebook. Se guardan en `runs/<run>/plots/<etapa>/`.
- Para **re-ejecutar** una etapa, pon `RUN = True` en su celda "Ejecutar o
  auditar", o corre el comando de la celda "Cómo ejecutar de forma independiente".
- Cambiar de run: `export MUSE_RUN_ID=ROXs12b_B_adp` antes de abrir Jupyter, o
  edita `RUN_ID` en la celda de setup.

> ⚠️ Nada es paper-válido hasta cerrar el A-block. El bloque A **no** es
> re-ejecutable desde raw en este repo (poda WP-10): esos notebooks solo auditan y
> documentan el comando histórico.

## Orden de ejecución y qué muestra cada plot

| # | Notebook | Etapa | Ejec. | Plots |
|---|----------|-------|:---:|-------|
| 1 | [A1_raw_reduction](A1_raw_reduction.ipynb) | Reducción raw (esorex) | auditoría | verificaciones V1–V6 + estimador de runtime de esorex |
| 2 | [A2_sky_zap](A2_sky_zap.ipynb) | Decisión ZAP (cielo) | auditoría | RMS vs λ (ventanas skyline/continuo) → R; zona de cielo |
| 3 | [A3_telluric](A3_telluric.ipynb) | Corrección telúrica | auditoría | transmisión telúrica + ventanas protegidas |
| 4 | [A4_cube_qc](A4_cube_qc.ipynb) | QC del cubo (M1–M5) | híbrido | M2 LSF + M3 growth-curve; M5 covarianza |
| 5 | [B1_load_align_crop](B1_load_align_crop.ipynb) | Carga/alineación/crop | sí | centrado + crop sobre luz-blanca |
| 6 | [B2_xcorr_stripes](B2_xcorr_stripes.ipynb) | Xcorr / franjas | sí | amplitud de franjas vs λ |
| 7 | [B3_localize](B3_localize.ipynb) | Localización del compañero | sí | detección banda-roja (primaria + compañero) |
| 8 | [C1_chromatic_psf](C1_chromatic_psf.ipynb) | PSF cromática (Moffat/Psfao) | sí | Moffat vs Psfao; r0 (Fried) vs λ |
| 9 | [C_04b_local_surface](C_04b_local_surface.ipynb) | Fondo local (superficie) | sí | before/after del fondo local |
| 10 | [C2_aperture](C2_aperture.ipynb) | Extracción por apertura | sí | espectro + banda de ruido; apcorr cromática |
| 11 | [C3_optimal](C3_optimal.ipynb) | Extracción óptima | sí | ls vs psfsub (sobre-sustracción) |
| 12 | [C4_psffit](C4_psffit.ipynb) | Ajuste de PSF (psffit) | sí | cubo residual; espectro canónico |
| 12b | [C5_sgf](C5_sgf.ipynb) | Sustracción de halo SGF (Julo+25) | sí (pendiente 1ª ejec.) | residual + espectro; predictor Ec. 1 |
| 12c | [C6_lpm](C6_lpm.ipynb) | Sustracción de halo LPM (Julo+25) | sí (pendiente 1ª ejec.) | mapas de coeficientes; SGF vs LPM en Hα |
| 13 | [D1_method_compare](D1_method_compare.ipynb) | Comparación inter-método | sí | heatmap t-matrix; par primario |
| 14 | [D2_calibrate](D2_calibrate.ipynb) | Calibración espectral | sí | referenciación antes/después; presupuesto de error |
| 15 | [E1_halpha_detect](E1_halpha_detect.ipynb) | Detección Hα | sí | pico vs distribución nula; región de Hα |
| 15b | [E1b_fov_detection](E1b_fov_detection.ipynb) | Detección ciega FoV (Julo+25) | sí (pendiente 1ª ejec.) | mapas z por método; gaussianidad por anillos |
| 16 | [E2_artifacts](E2_artifacts.ipynb) | Batería de artefactos | sí | T5 placebos; batería a golpe de vista |
| 17 | [E3_upper_limits](E3_upper_limits.ipynb) | Límites superiores (Ṁ) | sí | Ṁ por método; cadena física |
| 18 | [E4_injection](E4_injection.ipynb) | Inyección-recuperación | sí (pesado) | throughput por método; curva de recuperación |
| 18b | [E5_contrast_curves](E5_contrast_curves.ipynb) | Curvas de contraste (Julo+25) | sí (pendiente 1ª ejec.) | contraste 5σ vs separación con banda 25/75% |
| 18c | [E6_roc_curves](E6_roc_curves.ipynb) | Curvas ROC (Julo+25) | sí (pendiente 1ª ejec.) | DP vs FAP por método/separación; AUC |
| 19 | [F1_final_report](F1_final_report.ipynb) | Paquete final + gate | sí | semáforo del gate; 4 limitaciones aceptadas |
| 20 | [G0_real_cube](G0_real_cube.ipynb) | Ejecución cubo real | sí | comparación con legacy ADP |
| 21 | [G1_extraction_validation](G1_extraction_validation.ipynb) | Validación de extracción | sí | covarianza del remuestreo; veredictos |
| 22 | [G2_measure_lines](G2_measure_lines.ipynb) | Medición de líneas | sí | forest de z_score; límites de flujo |
| 23 | [G3_accretion](G3_accretion.ipynb) | Inferencia física | sí | E3 vs G3; estado de caracterización |
| 24 | [G4_classify](G4_classify.ipynb) | Clasificación de fuente | sí | ranking de hipótesis; tests disponibles |
| 25 | [G5_final_synthesis](G5_final_synthesis.ipynb) | Síntesis final | sí | tarjeta de síntesis; integridad del paquete |

## Endpoint científico (run realineado)

No-detección de Hα (E1) → **Ṁ ≲ 8×10⁻¹³ M☉/yr** (E3, Gumbel 99%) / 1.3×10⁻¹² (G3,
5σ + R_in) → compañero subestelar **real ligado** pero clasificación **ambigua**
(G4, falta el tipado espectral diferido en G3). F1 `yellow`, 4 limitaciones
aceptadas; **provisional hasta cerrar el A-block**. Cada notebook enlaza la nota de
decisión correspondiente en `docs/`.
