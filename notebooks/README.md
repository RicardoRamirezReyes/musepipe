# Notebooks de revisión (cadena A1 → G5, por objeto)

Un notebook por spec de la cadena canónica (A→B→C→D→E→F→G, +S). Son interfaces
**delgadas** sobre `musepipe/` + `scripts/`: no reimplementan lógica.

## Un set por objeto

```
notebooks/
  _nbcommon.py            utilidades compartidas (solo stdlib)
  ROXs12b/    32 .ipynb   primer objeto
    debug/                notebooks de análisis (constructor opcional)
  ROXs42Bb/   32 .ipynb   segundo objeto
    debug/
```

Los de `<objeto>/` **auditan** la cadena: llaman a `musepipe` y enseñan lo que
decidió cada etapa. Los de `<objeto>/debug/` hacen lo contrario — rehacen el
proceso **dentro del notebook**, con las funciones copiadas del código, para
probar y ajustar sin tocar la cadena; los genera
`scripts/build_debug_notebooks.py` y llevan una celda que compara su resultado
con el producto real de la etapa. Ver el README raíz.

Los dos sets son **idénticos en estructura**: ningún objeto es "el canónico".
Cada set audita la cadena de SU objeto, declarada en la clave `chain` de
`runs/<run>/config/config.json` (qué etapa vive en qué run). Ver el plan
[`docs/plan_multiobjeto_notebooks_2026-07-24.md`](../docs/plan_multiobjeto_notebooks_2026-07-24.md).

## Generar / regenerar

```bash
python scripts/build_review_notebooks.py --target ROXs12b        # -> notebooks/ROXs12b/
python scripts/build_review_notebooks.py --target ROXs42Bb --check  # + verifica que cada QC resuelve
python scripts/build_review_notebooks.py --target ROXs12b A1 C3   # solo algunas etapas
```

`--check` sale ≠0 si algún `load_qc` resuelve a un run de **otro objeto**
(contaminación cross-object); las etapas aún no ejecutadas para el objeto se
reportan como pendientes, no como error. Regenerar es idempotente.

**Añadir un objeto nuevo** = crear su run + config (con `chain`), su
`targets/<slug>.json`, y `--target <slug>`. Cero código.

## Estructura de cada notebook

1. **Spec & rol** — qué hace, input←/output→, link a `docs/spec_*.md`.
2. **Mapa de la cadena** (solo A1) — `nb.show_chain()`: cada etapa, en qué run
   vive su QC, estado (ok / otro run / pendiente / cross-object) y fecha.
3. **Cómo ejecutar de forma independiente** — el comando canónico exacto.
4. **Ejecutar o auditar** — guardada por `RUN`; por defecto audita.
5. **QC / resultados** — `nb.load_qc_optional`: si la etapa no corrió para este
   objeto, avisa y sigue (no rompe el notebook).
6. **Evidencia / plots / decisiones / conclusión.**

## Cómo se usan

- **La celda de setup fija el objeto**: `RUN_ID = nb.resolve_run_id('<run>')`. No
  hay run por defecto — sin run se lanza un error explicativo (un default
  apuntaba al primer objeto en silencio).
- **Procedencia visible**: cada notebook imprime de qué run sale su QC. Si sale de
  otro objeto, avisa en rojo (cross-object).
- **Etapas de reducción (A1–A4, S0/S1)**: la celda "Ejecutar o auditar" resuelve
  el comando real para ESTE objeto desde su config y puede lanzarlo (background +
  log). Si falta un dato en el config, lo dice y no lanza.
- **Los plots necesitan el kernel MUSE** (astropy/pandas/matplotlib) y, algunos,
  los cubos en `/mnt/2TB`. Fallan con un mensaje claro sin romper el notebook.
- **Cambiar de objeto**: abre el set del otro (`notebooks/<obj>/`), o
  `export MUSE_RUN_ID=<run>`.

> ⚠️ Nada es paper-válido hasta cerrar el A-block del objeto. Las etapas de
> reducción se pueden lanzar (son trabajos largos: revisa el coste en el notebook).

## Orden de ejecución y qué muestra cada plot

| # | Notebook | Etapa | Ejec. | Plots |
|---|----------|-------|:---:|-------|
| 1 | [A1_raw_reduction](ROXs12b/A1_raw_reduction.ipynb) | Reducción raw (esorex) | auditoría | verificaciones V1–V6 + estimador de runtime de esorex |
| 2 | [A2_sky_zap](ROXs12b/A2_sky_zap.ipynb) | Decisión ZAP (cielo) | auditoría | RMS vs λ (ventanas skyline/continuo) → R; zona de cielo |
| 3 | [A3_telluric](ROXs12b/A3_telluric.ipynb) | Corrección telúrica | auditoría | transmisión telúrica + ventanas protegidas |
| 4 | [A4_cube_qc](ROXs12b/A4_cube_qc.ipynb) | QC del cubo (M1–M5) | híbrido | M2 LSF + M3 growth-curve; M5 covarianza |
| 5 | [B1_load_align_crop](ROXs12b/B1_load_align_crop.ipynb) | Carga/alineación/crop | sí | centrado + crop sobre luz-blanca |
| 6 | [B2_xcorr_stripes](ROXs12b/B2_xcorr_stripes.ipynb) | Xcorr / franjas | sí | amplitud de franjas vs λ |
| 7 | [B3_localize](ROXs12b/B3_localize.ipynb) | Localización del compañero | sí | detección banda-roja (primaria + compañero) |
| 8 | [C1_chromatic_psf](ROXs12b/C1_chromatic_psf.ipynb) | PSF cromática (Moffat/Psfao) | sí | Moffat vs Psfao; r0 (Fried) vs λ |
| 9 | [C_04b_local_surface](ROXs12b/C_04b_local_surface.ipynb) | Fondo local (superficie) | sí | before/after del fondo local |
| 10 | [C2_aperture](ROXs12b/C2_aperture.ipynb) | Extracción por apertura | sí | espectro + banda de ruido; apcorr cromática |
| 11 | [C3_optimal](ROXs12b/C3_optimal.ipynb) | Extracción óptima — **2 variantes → 2 métodos** | sí | ls vs psfsub (sobre-sustracción) |
| 12 | [C4_psffit](ROXs12b/C4_psffit.ipynb) | Ajuste de PSF (psffit) | sí | cubo residual; espectro canónico |
| 12b | [C5_sgf](ROXs12b/C5_sgf.ipynb) | Sustracción de halo SGF (Julo+25) | sí (pendiente 1ª ejec.) | residual + espectro; predictor Ec. 1 |
| 12c | [C6_lpm](ROXs12b/C6_lpm.ipynb) | Sustracción de halo LPM (Julo+25) | sí (pendiente 1ª ejec.) | mapas de coeficientes; SGF vs LPM en Hα |
| 13 | [D1_method_compare](ROXs12b/D1_method_compare.ipynb) | Comparación inter-método | sí | heatmap t-matrix; par primario |
| 14 | [D2_calibrate](ROXs12b/D2_calibrate.ipynb) | Calibración espectral | sí | referenciación antes/después; presupuesto de error; **los espectros definitivos** (6 métodos + primaria) |
| 15 | [E1_halpha_detect](ROXs12b/E1_halpha_detect.ipynb) | Detección Hα | sí | pico vs distribución nula; región de Hα |
| 15b | [E1b_fov_detection](ROXs12b/E1b_fov_detection.ipynb) | Detección ciega FoV (Julo+25) | sí (pendiente 1ª ejec.) | mapas z por método; gaussianidad por anillos |
| 16 | [E2_artifacts](ROXs12b/E2_artifacts.ipynb) | Batería de artefactos | sí | T5 placebos; batería a golpe de vista |
| 17 | [E3_upper_limits](ROXs12b/E3_upper_limits.ipynb) | Límites superiores (Ṁ) | sí | Ṁ por método; cadena física |
| 18 | [E4_injection](ROXs12b/E4_injection.ipynb) | Inyección-recuperación | sí (pesado) | throughput por método; curva de recuperación |
| 18b | [E5_contrast_curves](ROXs12b/E5_contrast_curves.ipynb) | Curvas de contraste (Julo+25) | sí (pendiente 1ª ejec.) | contraste 5σ vs separación con banda 25/75% |
| 18c | [E6_roc_curves](ROXs12b/E6_roc_curves.ipynb) | Curvas ROC (Julo+25) | sí (pendiente 1ª ejec.) | DP vs FAP por método/separación; AUC |
| 19 | [F1_final_report](ROXs12b/F1_final_report.ipynb) | Paquete final + gate | sí | semáforo del gate; 4 limitaciones aceptadas |
| 20 | [G0_real_cube](ROXs12b/G0_real_cube.ipynb) | Ejecución cubo real | sí | comparación con legacy ADP |
| 21 | [G1_extraction_validation](ROXs12b/G1_extraction_validation.ipynb) | Validación de extracción | sí | covarianza del remuestreo; veredictos |
| 22 | [G2_measure_lines](ROXs12b/G2_measure_lines.ipynb) | Medición de líneas | sí | forest de z_score; límites de flujo |
| 23 | [G3_accretion](ROXs12b/G3_accretion.ipynb) | Inferencia física | sí | E3 vs G3; estado de caracterización |
| 24 | [G4_classify](ROXs12b/G4_classify.ipynb) | Clasificación de fuente | sí | ranking de hipótesis; tests disponibles |
| 25 | [G5_final_synthesis](ROXs12b/G5_final_synthesis.ipynb) | Síntesis final | sí | tarjeta de síntesis; integridad del paquete |

## Endpoint científico (run realineado)

No-detección de Hα (E1) → **Ṁ ≲ 8×10⁻¹³ M☉/yr** (E3, Gumbel 99%) / 1.3×10⁻¹² (G3,
5σ + R_in) → compañero subestelar **real ligado** pero clasificación **ambigua**
(G4, falta el tipado espectral diferido en G3). F1 `yellow`, 4 limitaciones
aceptadas; **provisional hasta cerrar el A-block**. Cada notebook enlaza la nota de
decisión correspondiente en `docs/`.
