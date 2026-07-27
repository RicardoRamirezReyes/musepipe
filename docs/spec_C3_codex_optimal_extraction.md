# Especificación C3 · `X02_optimal` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa C3 (ESENCIAL, nueva) del plan
`docs/plan_roxs12_reduccion_multimetodo.md`. Prerrequisitos: C1 (`psf_model`),
C2 (formato `SpectrumProduct` congelado), B3 (posiciones), A4/B1 (factores de
varianza).

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: implementar extracción óptima tipo Horne (1986) por canal:
ponderar los píxeles por el perfil espacial esperado y la varianza. Es un
algoritmo corto y cerrado — el valor está en implementarlo exactamente bien y
en demostrar con tests analíticos que flujo y varianza son correctos. Nada de
creatividad algorítmica: la fórmula es la fórmula.

**Estimador** (por canal, sobre la ventana de extracción):

```text
f = Σᵢ Mᵢ·Pᵢ·Dᵢ/Vᵢ / Σᵢ Mᵢ·Pᵢ²/Vᵢ        var(f) = 1 / Σᵢ Mᵢ·Pᵢ²/Vᵢ
```

con `P` = psf_model de C1 evaluada en la posición del compañero (normalizada
según §3.2), `D` = dato con fondo restado, `V` = varianza, `M` = máscara de
píxeles válidos/clipeados.

**Definición de terminado**:

1. `musepipe/extraction/optimal.py` implementado, con las DOS variantes de
   fondo (§3.1) produciendo `spec_optimal_object.fits` y
   `spec_optimal_psfsub_object.fits` en formato `SpectrumProduct`.
2. Varianza analítica correcta, demostrada con test sintético (§7).
3. Verificaciones §6 en verde, incluida la comparación contra C2.
4. QC según §5; suite del repo intacta.

---

## 1. Límites duros

1. Formato `SpectrumProduct` de C2: se consume tal cual (`FORMATV=1`), sin
   extensiones de esquema.
2. `psf_model` de C1 se usa como caja negra a través de su API; prohibido
   re-ajustar PSF dentro de esta etapa.
3. Posiciones desde `stage01c_qc.json` (con seguimiento cromático si B3 lo
   dictaminó); nunca hardcodeadas ni re-medidas aquí.
4. El sigma-clipping es ESPACIAL por canal (outliers dentro de la ventana,
   p. ej. cósmicos), jamás espectral: prohibido cualquier mecanismo que pueda
   recortar un canal entero con línea real (ver test §7.3).
5. Sin barrido de hiperparámetros guiado por el S/N final. Ventana, clipping y
   variantes son los de esta spec; cambios → preguntar.
6. Git: rama `stage-c3-optimal`.

---

## 2. Entradas

- Cubo de B2 + STAT (con factores de A4/B1 para escalar V).
- Fondos: (a) residual local-surface (productos 04b vía C2), (b) cubo −
  modelo de primaria de C1.
- `psf_model.json` (+ `norm_radius_px`), posiciones B3, máscaras de ventanas
  malas y flags de C2.
- Config: `optimal_window_px` (default: radio 8 px), `clip_sigma` (default 4),
  `clip_max_iter` (2).

## 3. Operaciones

### 3.1 Las dos variantes de fondo (ambas obligatorias)

- **Variante LS** (`spec_optimal_object.fits`): D = cubo − superficie local
  (04b). Comparable 1:1 con C2 — aísla la ganancia del ponderado óptimo.
- **Variante PSFSUB** (`spec_optimal_psfsub_object.fits`): D = cubo − modelo
  de primaria (C1, con término híbrido si lo hay). Anticipa el fondo que C4
  usa — la comparación LS vs PSFSUB en D1 diagnostica el modelo de halo.

### 3.2 Normalización de P — punto de máxima atención

`P` se evalúa en la ventana de extracción y se renormaliza para que
`Σ P = 1` DENTRO de la ventana; el estimador entonces devuelve el flujo de la
ventana, y la fracción de PSF fuera de la ventana se corrige con la curva de
crecimiento de C1 (columna `apcorr`, igual que C2). Las dos convenciones
posibles (renormalizar vs no) difieren en un factor suave en λ ≈ 1/(fracción
encerrada); mezclarlas es el bug clásico de Horne en IFU. Elegida y grabada:
**renormalización en ventana + apcorr explícito**. El test §7.1 verifica el
flujo total de punta a punta.

### 3.3 Clipping

- Por canal: residuo `(D − f·P)/√V`; rechazar |z| > `clip_sigma`; máx. 2
  iteraciones; registrar máscara M en el cubo de diagnóstico.
- Guardar fracción clipeada por canal (columna `flags` bit 8 si > 5% del
  canal) y el mapa espacial agregado de rechazo.

### 3.4 Varianza y errores

- `V = STAT × stat_factor_spaxel(λ)`; var(f) analítica del estimador ×
  `covariance_factor` interpolado a la escala efectiva de la ventana
  (documentar la interpolación entre el factor spaxel y el box3 de B1/A4).
- `flux_err_emp`: mismo estimador aplicado en las posiciones de control al
  mismo radio (reutilizar `same_radius_control_positions`) → dispersión por
  canal. Ambas columnas pobladas, como en C2.

## 4. Puntos donde mantener máxima atención

1. **Normalización de P** (§3.2) — el error factor-suave que ningún test de
   S/N detecta; solo el test de flujo absoluto lo caza.
2. **Centrado**: un descentrado de 0.3 px sesga el flujo ~1–2%; usar el
   centroide (cromático si aplica) de B3 y NO recentrar sobre el dato canal a
   canal (a S/N bajo, recentrar sobre ruido sesga al alza — prohibido).
3. **P demasiado ancha/estrecha**: sesgo directo de flujo. No se "corrige"
   aquí: se CUANTIFICA con el test de sensibilidad §7.2 (±10% FWHM) y el
   número queda en QC como sistemático heredado de C1.
4. **NaN y ventanas malas**: píxeles NaN salen de M (no imputar); canales en
   ventanas malas se extraen igual pero con flag 1 — la decisión de usarlos
   es de las etapas E, no de C3.
5. **Ruido correlacionado**: var(f) analítica asume V diagonal; el factor de
   covarianza de B1 lo compensa en promedio. La validación real es
   `flux_err` vs `flux_err_emp` (V2) — si difieren > 40%, no maquillar:
   reportar.

## 5. Esquema de `spec_optimal_qc.json`

```json
{
  "stage": "x02_optimal",
  "run_id": "...",
  "variants": ["ls", "psfsub"],
  "window_px": 8,
  "p_normalization": "window_renorm_plus_apcorr",
  "clip": {"sigma": 4, "max_iter": 2, "frac_clipped_median": 0.0,
            "channels_flagged": 0},
  "errors": {"mode": "stat|empirical", "stat_vs_empirical_median_ratio": 0.0},
  "snr_gain_vs_aperture": {"median": 0.0, "p10": 0.0, "p90": 0.0},
  "continuum_bias_vs_aperture_pct": 0.0,
  "psf_sensitivity": {"fwhm_pm10pct_flux_bias_pct": 0.0},
  "open_issues": []
}
```

## 6. Verificaciones

- **V1 — Ganancia de S/N**: distribución por canal de S/N(óptima)/S/N(C2 box3),
  variante LS; mediana ≥ 1.0 (esperable 1.1–1.3). Figura histograma + vs λ.
- **V2 — Errores**: `flux_err` vs `flux_err_emp` vs λ; razón mediana en
  [0.7, 1.4].
- **V3 — Sin sesgo de continuo**: (óptima − C2)/C2 en bandas de continuo
  < 2–3%; si es mayor, primero sospechar §3.2/§4.2, no de C2.
- **V4 — Clipping sano**: mapa espacial de rechazo sin concentración en la
  posición del compañero (< 2× la tasa media en su ventana).
- **V5 — LS vs PSFSUB**: espectros superpuestos; diferencias estructuradas se
  reportan como diagnóstico del modelo de halo (insumo para D1), no se
  resuelven aquí.

## 7. Tests

```text
tests/test_optimal_analytic.py     # fuente sintética con PSF y ruido conocidos: flujo exacto y var(f) == analítica (el test central)
tests/test_optimal_psf_bias.py     # P con FWHM ±10%: sesgo medido y monótono; documenta la sensibilidad (test de potencia)
tests/test_optimal_no_line_clip.py # línea de emisión sintética fuerte en un canal → el clipping espacial NO reduce su flujo (>99% sobrevive)
tests/test_optimal_norm.py         # las dos convenciones de §3.2 mezcladas a propósito → el test de flujo total falla (demuestra que el diseño caza el bug)
```

## 8. Protocolo de parada y reporte

Preguntar cuando: V3 falle tras revisar normalización y centrado; V1 < 1.0
(óptima peor que apertura = síntoma de P o V mal escaladas); STAT vs empírico
fuera de rango sin explicación. Reporte: tabla V1–V5, sensibilidad a PSF,
sistemático heredado de C1, checklist de límites, comando de reproducción.

## Errata (2026-07-26) — LS extrae del cubo crudo, no del residual de 04b

§3.1 define LS como «D = cubo − superficie local (04b). Comparable 1:1 con C2»,
y esa comparabilidad es su razón de ser: aislar la ganancia del ponderado
óptimo. Dejó de cumplirse en `d688a64`, que en el mismo commit movió **C2** al
cubo crudo con fondo de anillo (*wings-intact*, porque el residual de 04b se
come las alas del compañero) y añadió a **LS** ese mismo anillo **encima** del
residual de 04b.

El resultado eran dos tratamientos de fondo sobre un cubo que **no es
homogéneo**: 04b ajusta una superficie local solo alrededor del objeto, así que
el anillo (8–14 px) medía

| | en el compañero | en los controles |
|---|---:|---:|
| ROXs 12 b | 2.95 /px | 12.77 /px |
| ROXs 42B b | 3.67 /px | 14.80 /px |

es decir, al objeto se le quitaba el fondo **dos veces** y a los controles una.
Medido con la extracción óptima sobre el cubo de LS, esa segunda resta producía
el **93%** y el **96%** del continuo negativo del método (mediana de la banda
roja: −2045 con anillo, −140 sin él, en ROXs 12 b).

Consecuencia importante: el sesgo de continuo de **−373%** que hace fallar
`v3_continuum_bias` y por el que **G1 rechaza `optimal_ls`** estaba dominado por
el tratamiento de fondo, no por el estimador de Horne.

**Corregido**: cuando hay anillo configurado, LS extrae del cubo crudo de B2 con
ese anillo — el mismo tratamiento que C2 —, con `x02_wings_intact_ls=false` para
volver al comportamiento histórico. Tras el cambio, el sesgo de continuo pasa de
**−373% a −58%**. Lo que queda es de la misma familia que V4(b) de C2 y que el
contraste círculo/caja: la curva de crecimiento de C1 no es del todo consistente
entre aperturas.
