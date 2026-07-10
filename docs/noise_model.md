# Modelo de ruido canónico (MUSE ROXs 12)

> Nota canónica del modelo de ruido para la cadena MUSE. Consolida en un solo lugar el
> conocimiento hoy repartido en varios QC, para que **toda etapa futura lo cite en lugar
> de re-derivarlo**. Cada cifra lleva al lado la ruta del QC/notebook de la que se tomó
> (verificadas en disco el 2026-07-10).

---

## Resumen operativo (la regla que importa)

**Ninguna etapa usa la extensión `STAT` del cubo directamente para σ, ni asume √N entre
píxeles vecinos.** El STAT subestima el ruido y el remuestreo introduce correlación
espacial/espectral. Por tanto **σ SIEMPRE se estima de forma empírica**, a partir de
controles procesados exactamente como el objeto (principio *"control = objeto"* de D1 v2,
ver `docs/spec_D1_v2_codex_method_comparison.md`).

---

## 1. El STAT subestima el ruido de apertura (A4 / M5)

- **Factor de subestimación por spaxel (mediana): 4.26×** — el STAT subestima el ruido de
  apertura ~4× (`stage_status = red`).
  Fuente: `runs/<RUN>/stages/stage00q_qc.json` → `m5_stat.factor_spaxel_median`
  (idéntico en `ROXs12b_B_adp` y `ROXs12b_realigned`; nota del QC:
  *"STAT underestimates aperture noise ~4x (blocker #7); D2 uses empirical noise."*).
- Causa: la covarianza introducida por el remuestreo del cubo (ver §3–§4). El STAT propaga
  varianza por píxel independiente, hipótesis rota tras el regrid.

## 2. Inflación de ruido espacial integrada en apertura (G1)

La varianza real integrada en una caja N×N respecto a la ingenua (√N) crece con el tamaño
de la apertura. Fuente: `runs/ROXs12b_B_adp/stages/stage_g1_qc.json` →
`covariance.spatial_inflation_by_box`:

| Caja | Factor de inflación |
|---|---|
| 1×1 | 1.00 |
| 2×2 | 2.98 |
| **3×3** | **6.45** (≈ 6.5×) |
| 4×4 | 11.31 |
| **5×5** | **16.96** (≈ 17×) |

Es decir: sumar señal en una caja 3×3 infla la varianza ~6.5× frente a la suma ingenua, y
~17× en 5×5. Cualquier σ de apertura calculado como √(Σ STAT) está mal por estos factores.

## 3. Correlación espectral (G1)

Fuente: `runs/ROXs12b_B_adp/stages/stage_g1_qc.json` → `covariance`
(`spectral_file = stages/g1_channel_covariance.npz`):

- **Longitud de correlación: 1.45 ± 0.08 canales** (`corr_length_channels_median = 1.4508`,
  `corr_length_bootstrap_err = 0.0833`).
- **Canales efectivos: n_eff / n = 0.69** (`n_eff_over_n_median = 0.6918`). Solo el ~69 %
  de los canales cuentan como independientes: promediar en λ NO gana √N.

## 4. Origen físico cuantificado: el shift subpixel de stage01 (notebook 01b, T4)

El shift subpixel (spline orden 3) de la alineación de stage01 **reduce la varianza del
ruido blanco e introduce autocorrelación vecino-a-vecino** — la fuente física de §1–§2.
Fuente: `01b_align_resampling_tests.ipynb`, test **T4** (productos:
`tables/stage01b_noise_corruption.csv`, QC de stage01b):

| Shift (px) | Varianza residual | Autocorr. vecino | σ subestimado |
|---|---|---|---|
| 0.25 | 0.763 | +0.136 | +12.7 % |
| 0.50 | 0.581 | +0.264 | +23.8 % |

(El peor caso T4 registrado: var 0.581, autocorr +0.264 @ shift 0.50.) T6 confirma que el
modo `subpixel` baja la varianza a 0.570 e introduce autocorr +0.248 frente al ruido blanco
(`var≈1.0`) de los modos `none`/`integer`. Esto explica por qué el STAT (que asume ruido
blanco por píxel) subestima: tras el remuestreo el ruido está correlacionado.

## 5. Regla operativa (consecuencia)

1. **Nunca** usar `STAT` directamente como σ ni asumir √N entre píxeles/canales vecinos.
2. Estimar σ **empíricamente** de controles procesados idénticamente al objeto, al **mismo
   radio** y con la **misma cadena** de extracción/calibración (principio *control = objeto*
   de D1 v2). Así el σ hereda automáticamente la inflación de §2 y la correlación de §3.
3. Reportar FAP/significancia con esa σ empírica (p. ej. E1 con ≥ 31 controles).

---

### Referencias de QC (rutas)

- M5: `runs/<RUN>/stages/stage00q_qc.json` → `m5_stat`
- G1 (inflación espacial + correlación espectral): `runs/ROXs12b_B_adp/stages/stage_g1_qc.json` → `covariance`
- Origen físico (shift subpixel): `01b_align_resampling_tests.ipynb` (T4)
- Principio control = objeto: `docs/spec_D1_v2_codex_method_comparison.md`
