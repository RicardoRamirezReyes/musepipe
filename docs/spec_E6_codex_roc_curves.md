# Especificación E6 · `H06_roc` — curvas ROC del detector matched-filter

Fecha: 2026-07-15. Etapa E6 del plan
`docs/2026-07-15_plan_integracion_halosub.md` (WP-H6). Método: Julo et al. 2025
§3.3.3 (Fig. 11): DP vs FAP variando el umbral del matched filter, con
inyecciones a contraste tipo límite-de-detección y realizaciones de ruido de
los cubos residuales y de canales fuera de las líneas.

Protocolo spec-first: commiteada antes de ejecutar sobre datos reales. QC con
`spec_version="E6_v1"`. Parámetros de §2 congelados.

## 0. Rol y definición de terminado

E6 caracteriza la **robustez** del detector (trade-off precisión-retorno) por
método (sgf/lpm), complementando la curva de contraste de E5 (que fija
FAP y varía el contraste). Produce ROC (DP vs FAP) y AUC por método y
separación. Insumo informativo del checkpoint D1 v3 (nunca cambia veredictos).

Terminado = `tables/roc_curves.csv` + `stage_h06_qc.json` + plot.

## 1. Límites duros

1. Misma cadena de detección que E1b/E5 (plantilla, kernel C1, anillos);
   mismos estados de método (linealidad del delta).
2. E6 **depende de E5**: los escenarios usan su grilla y (si existe) su
   `contrast_50` como contraste tipo límite; sin E5 QC se usa `h06_contrast`
   (default 1.7e-3, el valor ilustrativo del paper) + open_issue.
3. El nulo es EMPÍRICO: píxeles del anillo del mapa z base + mapas z
   construidos en longitudes de onda libres de línea (paper §3.3.3). Sin
   supuestos gaussianos.
4. La compañera conocida queda excluida del nulo y de las inyecciones
   (herencia E1b/E5).

## 2. Operaciones y parámetros congelados

1. **Escenarios**: separaciones = cuantiles {25%, 50%, 75%} de la grilla E5
   (`h06_separations_px` para override = revisión de spec); contraste por
   (método, separación) = `contrast_50` de E5 (fallback `h06_contrast`).
2. **Inyecciones**: `h06_n_angles` = 16 posiciones por anillo (offset 11.25°
   para no repetir las de E5); z de cada inyección por el camino delta de E5.
3. **Nulo por separación**: banda de anillo [r − 2, r + 2] px del mapa z base
   + los mismos píxeles de mapas z re-construidos con la plantilla centrada
   en canales nulos: cada `h06_null_step_channels` = 25 canales dentro de las
   regiones libres de líneas estándar (y a > 4×FWHM de la línea objetivo).
4. **Curva**: umbrales = todos los valores únicos de la muestra combinada
   (inyecciones + nulo); DP(t) = frac(z_iny ≥ t), FAP(t) = frac(z_nulo ≥ t);
   AUC por integración trapezoidal de DP(FAP).

## 3. QC y verificaciones

```
stage="h06_roc", spec_version="E6_v1", run_id
params: {separations_px, contrasts_used, n_angles, null_step_channels}
methods: {<m>: {scenarios: [{separation_px, contrast, auc, n_injections,
                             n_null, roc_csv_rows}]}}
checks: {v1_curves_written, v2_auc_above_random, v3_null_sample_ok}
```

- v1: ROC escrita para todo método × escenario.
- v2 (sanidad): AUC ≥ 0.5 − 2/√n_iny en todos los escenarios (un detector
  peor que aleatorio indica bug de signo/normalización).
- v3: n_nulo ≥ 500 por escenario (si no, warning: subir mapas nulos).

## 4. Tests

- Sintético (escena E5): contraste alto → AUC > 0.95; contraste ~0 → AUC ≈
  0.5 (±0.15); DP y FAP monótonas no-crecientes con el umbral.
- Contrato: CSV/QC/plot escritos, checks evaluados.

## 5. Protocolo de parada y reporte

Sin cubos residuales → abortar. El notebook E6 grafica las ROC por método y
separación con el clasificador aleatorio de referencia (Fig. 11 del paper) y
tabula AUC.
