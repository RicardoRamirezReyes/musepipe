# Notebooks de revisión A1 → G5

Un notebook por spec de la cadena canónica ROXs 12 (A→G). **Reemplazan** a los
notebooks históricos de la raíz (`00_*`…`Far_*`), que quedaron como interfaz de
inspección y **no reflejan las correcciones recientes**. Estos son interfaces
delgadas sobre `musepipe/` + `scripts/`: no reimplementan lógica.

Generados por [`scripts/build_review_notebooks.py`](../scripts/build_review_notebooks.py)
(regenerar: `python scripts/build_review_notebooks.py`). Utilidades comunes en
[`_nbcommon.py`](_nbcommon.py).

## Cómo se usan

- **Por defecto AUDITAN** el QC existente del run `ROXs12b_realigned` (el run
  científico: F1 `yellow`, 0 rojos bloqueantes). Solo leen JSON — se abren sin
  la pila científica.
- Para **re-ejecutar** una etapa, pon `RUN = True` en su celda "Ejecutar o
  auditar" (necesita `conda activate MUSE`), o corre el comando de la celda
  "Cómo ejecutar de forma independiente".
- Cambiar de run: `export MUSE_RUN_ID=ROXs12b_B_adp` antes de abrir Jupyter, o
  edita `RUN_ID` en la celda de setup.

> ⚠️ Nada es paper-válido hasta cerrar el A-block (directiva del usuario). El
> bloque A **no** es re-ejecutable desde raw en este repo (poda WP-10): esos
> notebooks solo auditan y documentan el comando histórico.

## Orden de ejecución

| # | Notebook | Etapa | Ejecutable | QC |
|---|----------|-------|:---:|----|
| 1 | [A1_raw_reduction](A1_raw_reduction.ipynb) | Reducción raw (esorex) | auditoría | `stage00r_qc.json` |
| 2 | [A2_sky_zap](A2_sky_zap.ipynb) | Decisión ZAP (cielo) | auditoría | — |
| 3 | [A3_telluric](A3_telluric.ipynb) | Corrección telúrica | auditoría | — |
| 4 | [A4_cube_qc](A4_cube_qc.ipynb) | QC del cubo (M1–M5) | híbrido | `stage00q_qc.json` |
| 5 | [B1_load_align_crop](B1_load_align_crop.ipynb) | Carga/alineación/crop | sí | `stage01_qc.json` |
| 6 | [B2_xcorr_stripes](B2_xcorr_stripes.ipynb) | Xcorr / franjas | sí | `stage02_xcorr_qc.json` |
| 7 | [B3_localize](B3_localize.ipynb) | Localización del compañero | sí | `stage01c_qc.json` |
| 8 | [C1_chromatic_psf](C1_chromatic_psf.ipynb) | PSF cromática (Moffat/Psfao) | sí | `stage_e01_qc.json` |
| 9 | [C_04b_local_surface](C_04b_local_surface.ipynb) | Fondo local (superficie) | sí | `stage04b_qc.json` |
| 10 | [C2_aperture](C2_aperture.ipynb) | Extracción por apertura | sí | `spec_aperture_qc.json` |
| 11 | [C3_optimal](C3_optimal.ipynb) | Extracción óptima | sí | `spec_optimal_qc.json` |
| 12 | [C4_psffit](C4_psffit.ipynb) | Ajuste de PSF (psffit) | sí | `spec_psffit_qc.json` |
| 13 | [D1_method_compare](D1_method_compare.ipynb) | Comparación inter-método | sí | `stage_x10_qc.json` |
| 14 | [D2_calibrate](D2_calibrate.ipynb) | Calibración espectral | sí | `stage_x11_qc.json` |
| 15 | [E1_halpha_detect](E1_halpha_detect.ipynb) | Detección Hα | sí | `stage_h01_qc.json` |
| 16 | [E2_artifacts](E2_artifacts.ipynb) | Batería de artefactos | sí | `stage_h02_qc.json` |
| 17 | [E3_upper_limits](E3_upper_limits.ipynb) | Límites superiores (Ṁ) | sí | `stage_h03_qc.json` |
| 18 | [E4_injection](E4_injection.ipynb) | Inyección-recuperación | sí (pesado) | `stage_h04_qc.json` |
| 19 | [F1_final_report](F1_final_report.ipynb) | Paquete final + gate | sí | `report/run_summary.json` |
| 20 | [G0_real_cube](G0_real_cube.ipynb) | Ejecución cubo real | sí | `stage_g0_qc.json` |
| 21 | [G1_extraction_validation](G1_extraction_validation.ipynb) | Validación de extracción | sí | `stage_g1_qc.json` |
| 22 | [G2_measure_lines](G2_measure_lines.ipynb) | Medición de líneas | sí | `stage_g2_qc.json` |
| 23 | [G3_accretion](G3_accretion.ipynb) | Inferencia física | sí | `stage_g3_qc.json` |
| 24 | [G4_classify](G4_classify.ipynb) | Clasificación de fuente | sí | `stage_g4_classification.json` |
| 25 | [G5_final_synthesis](G5_final_synthesis.ipynb) | Síntesis final | sí | `report/characterization/` |

## Endpoint científico (run realineado)

No-detección de Hα (E1) → **Ṁ ≲ 8×10⁻¹³ M☉/yr** (E3, Gumbel 99%) → compañero
subestelar real pero clasificación **ambigua** (G4, falta tipado espectral).
Cada notebook enlaza la nota de decisión correspondiente en `docs/`.
