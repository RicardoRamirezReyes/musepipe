# Especificación C1 · `E01_chromatic_psf` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa C1 (ESENCIAL, nueva) del plan
`docs/plan_roxs12_reduccion_multimetodo.md`. Prerrequisitos: B2 (cubo), B3
(posiciones y veredicto cromático), A4 (LSF, factores).

Es la etapa de mayor riesgo técnico de todo el plan: la extracción óptima (C3)
y el PSF-fitting (C4) heredan directamente sus errores. Leer §4 (puntos de
atención) antes de escribir una sola línea.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: construir un modelo de PSF dependiente de la longitud de
onda a partir de la primaria, empaquetado como función evaluable
`psf_model(lam, dy, dx)`, con su presupuesto de error documentado. El
entregable NO es "un ajuste bonito de la estrella": es un modelo cuyo residuo
**en el radio del compañero** esté cuantificado, porque ese residuo es el error
sistemático que C3/C4 propagarán al espectro.

**La métrica que gobierna la etapa**: residuo relativo
`|dato − modelo| / halo` evaluado en un anillo en la separación del compañero
(de B3), por bin de λ. Todo lo demás (χ², suavidad de parámetros) está al
servicio de esa métrica.

**Definición de terminado**:

1. Parámetros de PSF ajustados por bin de λ, suavizados, con incertidumbres:
   `stage_e01_psf_params.csv` + `psf_model.json` (coeficientes).
2. Función de evaluación implementada, normalizada (§4.6), con test de
   round-trip.
3. Residuo en el radio del compañero < 5% por bin (o modelo híbrido aplicado y
   re-medido; o issue abierto con el número real — nunca silencio).
4. `stage_e01_qc.json` (§6); verificaciones §7 en verde.
5. `musepipe/psf.py` + `musepipe/stages/stage_e01_psf.py` + tests; los
   notebooks 04c/04d quedan como referencia histórica, ya no como fuente de
   verdad (anotarlo en el README de docs).

---

## 1. Límites duros

1. Solo lectura sobre cubos; salidas nuevas únicamente.
2. Solo archivos nuevos + registro en `stages/__init__.py`. Los notebooks
   04c/04d NO se borran ni se editan en esta etapa.
3. **El modelo no puede ver al compañero** (§4.1): máscara obligatoria, con
   test que lo demuestre (§8). Ídem el segundo objeto del campo.
4. Formas funcionales permitidas: Moffat elíptica (base) y el perfil de
   `maoppy` (alternativa, ya en `environment.yml`). Cualquier otra forma o un
   modelo híbrido distinto del especificado en §5 → preguntar antes.
5. Sin ajustes iterativos guiados por la métrica final ("tunear hasta que el
   anillo pase"): la secuencia es fija — ajustar, suavizar, medir, y si falla,
   aplicar el híbrido §5 UNA vez y re-medir. Si sigue fallando: parar y
   reportar.
6. Git: rama `stage-c1-psf`.

---

## 2. Entradas

- Cubo de B2; posiciones y errores de B3 (primaria, compañero, segundo
  objeto); veredicto `chromatic_centroid_needed` de B3.
- LSF(λ) y semáforos de A4 (informativo).
- Config: `psf_bin_A` (default 100 Å), `psf_fit_radius_px`, radios de máscara
  de fuentes, `companion_ring_width_px` (default 3).
- Material de referencia: los notebooks `04c_airy_ring_moffat_diagnostics` y
  `04d_pca_residual_moffat_diagnostics` documentan la estructura ya conocida
  de esta PSF (anillos tipo Airy del AO). Leerlos para saber qué esperar; no
  copiar código sin entenderlo — lo útil se reescribe en `musepipe/psf.py`.

## 3. Operaciones (secuencia fija)

### 3.1 Preparación

- Bins de λ de ~100 Å excluyendo 5780–6050 Å y ventanas de skylines fuertes;
  el gap del láser se puentea por interpolación de parámetros, marcado en QC.
- Máscaras: círculos sobre compañero y segundo objeto (radio ≥ 3× FWHM
  preliminar); chequeo de saturación/no-linealidad del core de la primaria
  (¿canales con pico > umbral del detector según header, o perfiles con tope
  plano?) → si hay, el core se enmascara en el ajuste (`core_mask_px` en QC).

### 3.2 Ajuste por bin

- Imagen mediana del bin → ajuste de Moffat elíptica: (y0, x0, FWHM_maj,
  FWHM_min, θ, β, amplitud) + fondo. **El fondo NO se ajusta libre** (§4.5):
  se fija a la mediana de las esquinas/regiones vacías del bin y se registra.
- Pesos por STAT si está disponible; sigma-clipping de residuos (3σ, máx. 3
  iteraciones — los anillos AO no deben cliparse: verificar que la fracción
  clipeada sea < 5% y anotarla).
- Guardar por bin: parámetros, errores formales, χ²ᵣ, fracción clipeada.
- En paralelo (mismo bin, misma máscara): ajuste con `maoppy` como
  contraste. No se elige aún; se comparan en §3.4.

### 3.3 Suavizado en λ

- Cada parámetro vs λ: polinomio de grado ≤ 2 (FWHM puede requerir la forma
  física ~λ^(-1/5) o lineal; elegir por AIC entre lineal, cuadrático y ley de
  potencia, registrado). Rechazo de bins outlier (>3σ del ajuste) con causa
  anotada (¿skyline residual? ¿bin del borde?).
- El centro (y0, x0) vs λ se compara con la deriva medida en B3: deben contar
  la misma historia; discrepancia > 0.3 px → issue.

### 3.4 Elección de forma base y medición del residuo

- Para Moffat y maoppy: reconstruir el modelo por bin, calcular la métrica del
  anillo del compañero. Elegir la forma con menor residuo mediano (empate →
  Moffat, por simplicidad); decisión y números en QC.
- Producir el mapa de residuo relativo por bin y el perfil residual azimutal.

### 3.5 Modelo híbrido (solo si la métrica falla)

- Si el residuo en el anillo del compañero > 5% en más del 20% de los bins:
  añadir al modelo la **mediana azimutal del residuo** (perfil radial residual
  por bin, suavizado en λ), que captura anillos AO simétricos.
- Restricción de seguridad (test §8): el término híbrido es azimutalmente
  simétrico y suavizado con escala radial ≥ 2×FWHM, de modo que NO PUEDE
  absorber una fuente puntual como el compañero.
- Re-medir la métrica. Este paso se aplica una sola vez.

## 4. Puntos donde mantener máxima atención

1. **El compañero dentro del modelo** — el fallo más dañino y silencioso: si
   la máscara no lo cubre en todos los bins (recordar la deriva cromática), el
   modelo lo absorbe parcialmente y C4 lo restará como si fuera halo → flujo
   sesgado a la baja de forma indetectable aguas abajo. Máscara con margen,
   centrada por bin si `chromatic_centroid_needed`, y test obligatorio §8.2.
2. **PSF AO ≠ Moffat**: anillos y wings cromáticas ya documentados en 04c/04d.
   No perseguirlos con β exótico ni con clipping: para eso está el híbrido §3.5.
3. **Saturación del core**: un core saturado ajustado sin máscara sesga FWHM y
   β en TODO el modelo. El chequeo de §3.1 no es opcional.
4. **El gap del láser**: parámetros interpolados, no extrapolados; los bins
   adyacentes al gap se marcan y D1/E1 sabrán que ahí el modelo es interpolación.
5. **Fondo degenerado con wings**: β bajo y fondo alto se compensan. Por eso
   el fondo se fija externamente (§3.2); si el χ²ᵣ exige fondo libre, eso es
   un hallazgo para reportar, no un parámetro para liberar en silencio.
6. **Normalización**: `psf_model` debe integrar a 1 dentro de un radio de
   normalización FIJO y documentado (`norm_radius_px`, default 25 px) en todo
   λ. C2 (corrección de apertura), C3 (perfil P) y C4 (fotometría) dependen de
   que esta convención sea única. Un cambio de convención a mitad de camino
   invalida los tres extractores: el radio queda grabado en `psf_model.json` y
   los consumidores lo leen de ahí.

## 5. Productos

```text
runs/<RUN_ID>/stages/stage_e01_psf_params.csv    # por bin: params, errores, chi2, clip_frac
runs/<RUN_ID>/stages/psf_model.json              # forma elegida, coeficientes suavizados, norm_radius_px, hybrid: true/false
runs/<RUN_ID>/stages/psf_hybrid_residual.fits    # solo si híbrido: perfil radial residual por bin
runs/<RUN_ID>/plots/stage_e01_*.png
```

## 6. Esquema de `stage_e01_qc.json`

```json
{
  "stage": "e01_chromatic_psf",
  "run_id": "...",
  "input": {"cube": "...", "sha256": "...", "positions_from": "stage01c_qc.json"},
  "binning": {"bin_A": 100, "n_bins": 0, "excluded_windows_A": [[5780, 6050]],
               "bins_interpolated": []},
  "masks": {"companion_radius_px": 0, "chromatic_tracking": false,
             "core_mask_px": 0, "saturation_detected": false},
  "fit": {"form_chosen": "moffat|maoppy", "background_mode": "fixed_external",
           "clip_frac_max": 0.0, "chi2r_median": 0.0},
  "smoothing": {"per_param_model": {}, "outlier_bins": [],
                 "centroid_vs_b3_max_diff_px": 0.0},
  "companion_ring_metric": {"radius_px": 0.0, "width_px": 3,
                              "residual_pct_median": 0.0, "residual_pct_p90": 0.0,
                              "bins_above_5pct": 0, "after_hybrid": false},
  "hybrid": {"applied": false, "smoothing_scale_px": 0.0},
  "normalization": {"norm_radius_px": 25, "roundtrip_error": 0.0},
  "open_issues": []
}
```

## 7. Verificaciones

- **V1 — Suavidad**: FWHM(λ), β(λ), elipticidad(λ), θ(λ) con su ajuste y
  outliers marcados; sin saltos no físicos (outliers ≤ 10% de bins).
- **V2 — La métrica**: residuo del anillo del compañero vs λ, con la línea del
  5%; mediana y p90 en QC. Es la figura principal de la etapa.
- **V3 — Mapas de residuo**: (dato−modelo)/halo en 6 bandas; sin estructura
  puntual en la posición del compañero (eso sería §4.1 fallando).
- **V4 — Energía encapsulada**: curva de crecimiento modelo vs dato en 3
  bandas; acuerdo < 3% hasta `norm_radius_px`.
- **V5 — Round-trip de normalización**: integrar `psf_model` numéricamente en
  una malla fina → 1 ± 0.5% en 10 valores de λ.
- **V6 — Consistencia con B3**: centroide del modelo vs deriva de B3 (§3.3).

## 8. Tests

```text
tests/test_psf_synthetic.py      # cubo sintético Moffat con params(λ) conocidos → recuperados dentro de errores
tests/test_psf_mask_companion.py # escena con compañero brillante: modelo con máscara vs sin compañero → idénticos a <0.5%; y SIN máscara → test detecta la absorción (test de potencia)
tests/test_psf_hybrid_safety.py  # inyectar fuente puntual en el residuo → el término híbrido (azimutal+suavizado) NO la absorbe (>90% del flujo sobrevive)
tests/test_psf_normalization.py  # round-trip de V5 como test unitario
```

`test_psf_mask_companion.py` es el test más importante del bloque C: fabrica
el fallo silencioso de §4.1 y demuestra que el diseño lo previene.

## 9. Protocolo de parada

Preguntar cuando: la métrica del anillo siga > 5% tras el híbrido (traer el
mapa de residuo y opciones: subir bin, forma nueva, o aceptar el número como
sistemático documentado — decisión del usuario); saturación ambigua; χ²ᵣ
sistemáticamente ≫ 1 con STAT en verde de A4 (contradicción a investigar);
Moffat y maoppy discrepen fuerte en FWHM (>10%) sin ganador claro.

## 10. Reporte final

Forma elegida y por qué; parámetros suavizados (figura V1); la métrica del
anillo antes/después del híbrido; presupuesto de error sistemático heredable
por C3/C4 (número explícito); verificaciones; checklist de límites; comando de
reproducción.
