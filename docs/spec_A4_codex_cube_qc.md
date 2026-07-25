# Especificación A4 · `00q_cube_qc` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa A4 (ESENCIAL, todos los entry points) del plan
`docs/plan_roxs12_reduccion_multimetodo.md`. Es la **puerta de entrada común**:
todo cubo — re-reducido (A1–A3), ADP del archivo, o histórico — pasa por aquí
antes de la cadena B–F.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: A4 es una etapa de **medición, no de corrección**. Mide la
calidad de la calibración del cubo (longitud de onda, LSF, flujo, cielo, STAT)
y entrega factores y tablas que las etapas D2/E1 consumirán. No modifica el
cubo; no "arregla" nada. El éxito no es que las métricas den bonitas: es que
todas las mediciones existan, con incertidumbre, y con semáforo honesto.

**Objetivo**: producir `stage00q_qc.json` con: offset de longitud de onda (y su
dependencia con λ), LSF(λ) tabulada, factor de escala de flujo vs literatura,
factor de corrección de STAT, y estadística de cielo — cada uno con su
incertidumbre y su semáforo verde/amarillo/rojo según los umbrales de §6.

**Definición de terminado**:

1. Las 5 mediciones de §4 ejecutadas sobre el cubo objetivo, con incertidumbres.
2. `stage00q_qc.json` completo (§5) y figuras de §4 en `plots/`.
3. Semáforo por medición asignado por los umbrales de §6 (no por juicio).
4. `musepipe/qc/cube_qc.py` + tests versionados; **cero números de ROXs 12
   dentro del módulo** — todo target-specific vive en config (el módulo se
   reutiliza tal cual para cualquier target futuro).
5. Ejecución demostrada sobre DOS cubos: el de A1/A2/A3 y el ADP histórico
   (`ADP.2022-09-12T17_17_39.371.fits`), con tabla comparativa. Esto valida a
   la vez el módulo y la re-reducción.
6. Reporte final (§9).

---

## 1. Límites duros

1. Solo lectura sobre todos los cubos; A4 no escribe FITS, solo QC, tablas y
   figuras.
2. Solo archivos nuevos: `musepipe/qc/` (paquete nuevo), tests, notebook fino
   `00q_cube_qc.ipynb`. Nada existente se modifica.
3. **Sin umbrales inventados**: los semáforos usan los de §6. Si una medición
   requiere un umbral no definido aquí → preguntar.
4. Si una medición es imposible (p. ej. no hay skylines porque el cubo es
   post-ZAP y no existe el pre-ZAP), la clave correspondiente queda en
   `"unavailable"` con la razón — prohibido rellenar con un valor estimado sin
   marcarlo.
5. Git: rama `stage-a4-cube-qc`.

---

## 2. Entradas

- Cubo(s) objetivo (config: lista de rutas + etiqueta de procedencia
  `raw_reduction | adp | historic`).
- Catálogo de skylines (Osterbrock/UVES sky atlas; incluirlo como CSV pequeño
  en `musepipe/qc/data/skylines.csv` con cita en el header del archivo).
- Fotometría de referencia de la primaria: Gaia DR3 (G, BP, RP) — las únicas
  bandas bien cubiertas por el rango MUSE. Config: IDs y magnitudes con
  errores, más curvas de transmisión de los filtros (archivo estático citado).
- Posiciones de primaria/compañero (del header ADP o config del run).
- Si A2 aplicó ZAP: ruta del cubo PRE-ZAP (ver §3, punto crítico).

## 3. Puntos donde mantener máxima atención (leer antes de implementar)

1. **Corrección baricéntrica vs skylines — el error clásico de esta etapa.**
   Las skylines están en reposo en el marco topocéntrico. `muse_scipost`
   aplica por default la corrección baricéntrica a la malla de λ del cubo
   (keyword del header, p. ej. `ESO DRS MUSE ... RVCORR` / historia del
   pipeline). En un cubo corregido, las skylines aparecen desplazadas
   −v_bary. El módulo DEBE leer del header si el cubo está en marco
   baricéntrico y cuánto vale la corrección, y comparar los centroides de
   skylines contra `λ_lab · (1 − v_bary/c)` cuando corresponda. Registrar en
   QC el marco detectado y el valor usado. Si el header no lo deja claro:
   preguntar, no asumir.
2. **Cubo post-ZAP no tiene skylines**: la validación de λ y la LSF por
   skylines se hacen sobre el cubo PRE-ZAP (o pre-skysubtraction si se guardó
   el pixtable), y el resultado se hereda al cubo limpio (la malla de λ es la
   misma). Documentar la procedencia en QC.
3. **La primaria puede ser variable** (estrella joven): una discrepancia de
   flujo vs Gaia no implica mala calibración. El factor de flujo se reporta
   con esa reserva explícita; nunca se "corrige" el cubo aquí.
4. **STAT y correlación espacial**: el resampleo del cubo correlaciona píxeles
   vecinos; la varianza empírica spaxel a spaxel subestima el ruido de
   aperturas. Medir el factor STAT en DOS escalas: por spaxel y por apertura
   3×3 (la razón entre ambas estima la covarianza espacial efectiva). Ambos
   números van a QC; D2 y E1 los necesitan.
5. **Bandas excluidas de toda métrica**: 5780–6050 Å (láser) y bordes del
   rango (< 4800, > 9300 Å).

## 4. Las cinco mediciones

### M1 — Solución de longitud de onda

- Seleccionar ~15–30 skylines aisladas y con S/N > 10 del catálogo, repartidas
  en 4800–9300 Å; medir centroide por ajuste gaussiano en el espectro de cielo
  (mediana de spaxels vacíos), globalmente y en 3×3 sub-regiones del campo.
- Producto: `Δλ(λ)` global (mediana ± error) y mapa de Δλ por sub-región;
  ajuste lineal `Δλ = a + b·λ` con incertidumbres.
- Figura: Δλ vs λ con las líneas usadas, banda de ±0.1 Å marcada, y el marco
  (topo/bary) anotado.

### M2 — LSF empírica

- FWHM de las mismas skylines (deconvolución trivial asumiendo línea
  intrínseca delta); tabla LSF(λ) con ~8 puntos + ajuste polinómico grado 2.
- Comparar contra la **LSF de referencia publicada de MUSE**:
  `FWHM(λ) = 5.866e-8 λ² − 9.187e-4 λ + 6.040` Å (Bacon et al. 2017, A&A 608,
  A1, Ec. 8 — mediana de la LSF medida en los cubos del MUSE UDF, dispersión
  1–3%). Desviaciones > 15% → amarillo, > 30% → rojo.
  - Implementación: `musepipe.qc.cube_qc.nominal_muse_fwhm_A`; la cita y los
    coeficientes se registran en el QC (`m2_lsf.nominal_reference`,
    `m2_lsf.nominal_poly_coeffs`).
  - Es una referencia de WFM y actúa solo como **patrón de comparación**: aguas
    abajo (E1/E3/G2) se usa siempre la LSF **medida**.
  - Antes de 2026-07-24 la referencia era una interpolación lineal en R
    (R~1770 en 4800 Å → R~3590 en 9300 Å) sin origen publicado; los QC escritos
    con ella no llevan `nominal_reference`.
- Figura: FWHM vs λ, medida y referencia (la figura debe citar el paper).

### M3 — Escala de flujo

- Fotometría sintética de la primaria: espectro por apertura grande con
  corrección de apertura (curva de crecimiento en la imagen blanca), integrado
  contra las curvas Gaia G/BP/RP → magnitudes sintéticas vs catálogo.
- Producto: `flux_factor` por banda (sintética/catálogo, en flujo lineal) con
  error combinado (fotometría + apertura + error de catálogo).
- Figura: SED de 3 puntos, sintética vs Gaia, con barras.

### M4 — Estadística de cielo

- En ≥ 10 aperturas vacías (reutilizar lógica de `musepipe/apertures.py`):
  rms por canal, mediana por canal (¿sesgo de sobre/sub-sustracción de
  cielo?), y la métrica `R` de A2 §3 para trazabilidad.
- Figura: rms y mediana vs λ con ventanas de skylines marcadas.

### M5 — Validación de STAT

- En las mismas aperturas vacías: razón `var_empirica / mediana(STAT)` por
  canal, en escala spaxel y en apertura 3×3 (ver §3.4).
- Producto: `stat_factor_spaxel(λ)` y `stat_factor_box3(λ)` (mediana global +
  tendencia con λ).
- Figura: ambas razones vs λ.

## 5. Esquema de `stage00q_qc.json`

```json
{
  "stage": "00q_cube_qc",
  "run_id": "...",
  "timestamp_utc": "...",
  "cube": {"file": "...", "sha256": "...", "provenance": "raw_reduction|adp|historic",
            "wavelength_frame": "barycentric|topocentric", "vbary_kms": 0.0,
            "skyline_source_cube": "same|pre_zap|unavailable"},
  "m1_wavelength": {"n_lines": 0, "offset_median_A": 0.0, "offset_err_A": 0.0,
                     "linear_a_A": 0.0, "linear_b": 0.0, "spatial_scatter_A": 0.0,
                     "status": "green|yellow|red|unavailable"},
  "m2_lsf": {"table_A_fwhm": [], "poly2_coeffs": [], "max_dev_vs_nominal_pct": 0.0,
              "nominal_reference": "Bacon et al. 2017, A&A 608, A1, Eq. 8",
              "nominal_poly_coeffs": [5.866e-8, -9.187e-4, 6.040],
              "status": "..."},
  "m3_flux": {"factor_by_band": {"G": 0.0, "BP": 0.0, "RP": 0.0},
               "err_by_band": {}, "aperture_correction": 0.0,
               "variability_caveat": true, "status": "..."},
  "m4_sky": {"rms_continuum": 0.0, "rms_skylines": 0.0, "R": 0.0,
              "median_bias": 0.0, "n_apertures": 0, "status": "..."},
  "m5_stat": {"factor_spaxel_median": 0.0, "factor_box3_median": 0.0,
               "trend_with_lambda": "flat|rising|falling", "status": "..."},
  "excluded_windows_A": [[5780, 6050]],
  "comparison_adp": {"ran": true, "table": "tables/stage00q_adp_vs_raw.csv"},
  "open_issues": []
}
```

## 6. Umbrales de semáforo (fijos; cambiarlos requiere preguntar)

| Medición | Verde | Amarillo | Rojo |
|---|---|---|---|
| M1 offset λ | \|Δλ\| < 0.1 Å y scatter espacial < 0.1 Å | corregible con término lineal (residuo < 0.1 Å) | residuo > 0.1 Å tras lineal |
| M2 LSF | desv. < 15% de la referencia (Bacon+2017, Ec. 8) | 15–30% | > 30% o no medible |
| M3 flujo | factor en [0.9, 1.1] en las 3 bandas | [0.8, 1.25] (con caveat de variabilidad) | fuera, o pendiente fuerte entre bandas |
| M4 cielo | \|mediana\| < 0.2×rms y R < 1.5 | R 1.5–2.0 | R > 2 (y A2 no corrió) o sesgo de mediana |
| M5 STAT | factores en [0.8, 1.5], tendencia plana | [0.5, 2.0] o tendencia suave | fuera de [0.5, 2.0] → STAT inutilizable, plan B empírico |

Interpretación aguas abajo (informativa, no acción de A4): amarillo en M1 →
D2 aplica el término lineal; rojo en M5 → D2/E1 usan solo ruido empírico
(camino ya validado en el repo); rojo en M4 con A2 saltada → volver a A2.

## 7. Organización del código

```text
musepipe/qc/__init__.py
musepipe/qc/cube_qc.py           # las 5 mediciones, cada una función pura testeable
musepipe/qc/data/skylines.csv    # catálogo con cita
musepipe/qc/data/gaia_passbands/ # curvas de filtro con cita
scripts/cube_qc.sh               # fachada CLI reproducible para A4
00q_cube_qc.ipynb                # notebook fino: config → run → figuras
tests/test_cube_qc_m1.py         # cubo sintético con skylines desplazadas +0.05 Å → M1 lo recupera
tests/test_cube_qc_m5.py         # cubo con ruido conocido y STAT fabricado ×1.3 → M5 recupera 1.3
tests/test_cube_qc_frames.py     # el caso baricéntrico: v_bary=25 km/s simulada → M1 corrige el marco
```

Los tres tests son de potencia: cada uno fabrica el error que la medición debe
detectar. `test_cube_qc_frames` es obligatorio por el punto §3.1.

## 8. Protocolo de parada

Preguntar cuando: el marco de λ no se pueda determinar del header (§3.1); haya
< 8 skylines medibles; no existan magnitudes Gaia confiables de la primaria;
cualquier medición dé rojo (se reporta el semáforo y se pregunta cómo proceder
antes de que la cadena B use el cubo).

## 9. Reporte final

- Tabla de las 5 mediciones con valores, incertidumbres y semáforo, para ambos
  cubos (re-reducido y ADP) lado a lado.
- Diferencias relevantes entre ambos cubos y su implicación (¿la re-reducción
  es al menos tan buena como el ADP?).
- Factores que D2 debe consumir (`Δλ`, `flux_factor`, `stat_factor_*`) listados
  explícitamente.
- Checklist de límites §1 y comando de reproducción.

## Errata (2026-07-25) — M3 y la unidad del cubo

Aditivo; ni un umbral ni el semáforo de §6 cambian. M3 mide en **una** banda
(`m3_recommended_band`, RP por defecto) y publica `flux_factor` **sin barra de
error**, no el trío `factor_by_band`/`err_by_band` de §5: con una sola banda no
hay dispersión entre bandas de la que derivarla. La consecuencia aguas abajo es
que `sys_fluxcal` de D2 salía idénticamente cero; desde 2026-07-25 D2 lo
**declara sin plegarlo** (`sys_fluxcal_declared`, ver errata v1.2 de la spec D2).
Si M3 llega a publicar su incertidumbre, esa manda y el término declarado
desaparece.

La comparación con el catálogo es **en cgs**, así que M3 necesita la unidad del
cubo. Antes la suponía en silencio (`m3_flux_unit_cgs`, default `1e-20`): un
cubo en otras unidades daba un `flux_factor` mal por 1e20 con semáforo verde.
Ahora la resuelve con `musepipe.io.resolve_flux_unit` — knob
`m3_flux_unit_cgs` → `BUNIT` del cubo que mide → cubo de entrada del run — y,
si no hay ninguna, devuelve `status: unavailable` con
`reason: flux_unit_unknown` en vez de suponerla. El QC de M3 anota además
`flux_unit_cgs`, `flux_unit_source` y `bunit`; esa unidad es la **tercera
fuente** que E3/G3/D2 usan cuando un producto viejo perdió su `BUNIT`.
