# Especificación C1 · `E01_chromatic_psf` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa C1 (ESENCIAL, nueva) del plan
`docs/2026-07-10_plan_roxs12_reduccion_multimetodo.md`. Prerrequisitos: B2 (cubo), B3
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
  de fuentes, `companion_ring_width_px` (default 3), `psfao_wave_bin_A`
  (default: el propio `psf_bin_A` — ver §5.1), `psfao_grid_reach_px`
  (default 140 px; el alcance de la rejilla con que se construye la PSF de
  psfao cuando C4 la evalúa sobre toda la imagen), `psf_fit_weighting`
  (default `stat` — ver §5.2).
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

### 5.1 `psfao_wave_bin_A` — la rejilla de evaluación viaja en el documento

Solo aplica a la forma `psfao`. Es la rejilla a la que `_evaluate_psfao`
(`musepipe/psf.py`) **redondea λ** antes de construir la PSF, y cumple dos
funciones: canales consecutivos comparten una sola FFT cacheada (3681 → ~45
construcciones), y —lo que la hace científica y no de rendimiento— **ningún
canal recibe sus parámetros del PSD interpolados entre dos bins**. Los siete
parámetros de Psfao son degenerados entre sí: la recta que une dos ajustes se
sale del valle, la PSF sale ~1 % mal en el halo y el diseño casi degenerado del
psffit (PSF + PSF + plano) lo amplifica a una onda cuadrada del 13 % en el
espectro. Medido en `apcorr_debug` §§14–17; historia en
`docs/2026-08-12_handoff.md`.

Reglas:

- C1 **siempre** la escribe en `psf_model.json` (con su clave hermana
  `psfao_wave_bin_A_note`), tomándola del config o, si no está declarada, de
  `psf_bin_A`. Nunca debe ser **menor** que el ancho de los bins ajustados.
- Los consumidores la leen del documento, jamás de un literal. Si un documento
  antiguo no la trae, `psf.py` la deriva de la separación mínima de su propio
  `param_table` (los huecos son múltiplos del ancho, así que la mediana
  mentiría); un documento sin `param_table` se evalúa con λ exacta.
- **No hay default numérico**: el histórico era 50 Å, la mitad del ancho de los
  bins, y es exactamente el fallo descrito arriba. No reintroducirlo.

### 5.2 · `psf_fit_weighting` — qué parte de la imagen decide el ajuste

Solo aplica a la forma `psfao`. Con `stat` —1/STAT, lo que se usó siempre— el χ² lo
**domina el núcleo** por varios órdenes de magnitud y el halo no llega a tener voz. Eso no
es un detalle: medido en `C1_chromatic_psf_debug` §13.f sobre ROXs 12 b, así el modelo
reproduce el **28 %** del cromatismo del halo y deja el residuo de anillo en **32 %**.

| valor | pesos | efecto medido (ROXs 12 b, 43 bins) |
|---|---|---|
| `stat` | `1/STAT` | croma 28 %, anillo 32.05 % — el histórico, y el **default** |
| `relative` | `1/STAT ÷ clip(\|imagen\|, piso, techo)²`, con `psf_fit_weight_cap` | ver el barrido de abajo |
| `halo` | `1/STAT`, núcleo a cero | croma **101 %**, anillo 15.84 % |

Reglas:

- **El default es `stat`**: ningún run cambia si no lo declara.
- La elección **viaja en `psf_model.json`** (`psfao_fit_weighting`) y en el QC
  (`fit.weighting`), por el mismo motivo que la rejilla: quien lea el producto tiene que
  poder saber qué decidió el ajuste, no suponerlo.
- `halo` es **diagnóstico, no default**: recupera todo el cromatismo pero deja el nivel del
  anillo un 4–14 % bajo y empuja `beta` contra su cota.
- Con `relative`, `beta` se apila cerca de su tope (5), pero **el resultado no depende de
  eso**: fijándolo en 1.35 —la mediana histórica, lejos del borde— salen los mismos
  números (croma 45 %, anillo 4.66 %) con 0 de 43 bins en la cota. Es la degeneración
  `alpha`–`beta` (r = +0.98), no un ajuste apoyado en el límite.
- La rama Moffat **no** pasa por aquí: `fit_moffat_image` nunca usó `STAT`, hace mínimos
  cuadrados con recorte sigma. Las dos formas ya pesaban distinto antes de este knob.

> **`relative` NO se puede usar tal cual, y está medido.** Se probó en la cadena entera el
> 2026-08-14 y se revirtió. Arregla el halo —el residuo de anillo cae de 24.30 % a **4.64 %**,
> por primera vez dentro del objetivo de esta spec, y el híbrido deja de hacer falta— pero
> **rompe el núcleo**: el dato dice `F(r≤25)/F(box3) = 4.96` y el modelo pasa de decir 5.71
> a decir **13.94**. Como la corrección de apertura es justo ese cociente, se infla ×2.7
> (mediana 9.51 → 26.05) y arrastra a todo lo que cuelga: dispersión entre métodos
> 27.9 % → 63.9 %, pares de continuo divergentes en D1 5 → 6, `open_issues` de F1 38 → 49 y
> Ṁ un 51 % más alto (1.83e-13 → 2.76e-13 M☉/año) por un motivo equivocado.
>
> El diagnóstico es simétrico al problema original: con `stat` el núcleo se lo lleva todo;
> con `relative` sin tope cada anillo pesa igual y el núcleo —9 píxeles de un disco de
> 78 px de radio— deja de contar.

**El tope: `psf_fit_weight_cap`.** Es la razón máxima entre el peso mayor y el menor que la
imagen puede introducir (`cap = 1` → equivale a `stat`; `cap = None` → el `relative` sin
tope de arriba). Barrido sobre ROXs 12 b, 15 bins, un solo vector de arranque —los absolutos
son peores que los de la cadena, que usa arranque en caliente; lo que vale es la forma:

| `cap` | anillo del compañero | `F(r≤25)/F(box3)` | error vs el dato (4.87) |
|---|---|---|---|
| 1 (= `stat`) | 32.80 % | 5.05 | +4 % |
| **5** | **11.76 %** | **5.11** | **+5 %** |
| 10 | 9.94 % | 5.16 | +6 % |
| 20 | 8.77 % | 5.29 | +9 % |
| 50 | 7.28 % | 5.80 | +19 % |
| 100 | 6.34 % | 7.12 | +46 % |
| 1000 | 4.84 % | 22.21 | +357 % |
| ∞ (sin tope) | ~4.6 % | 13.94 | +185 % |

El codo está en **5**: el residuo de anillo cae **a un tercio** y el núcleo se mueve **un
punto**. A partir de ahí cada mejora del anillo se paga cara, y por encima de 100 el núcleo
se dispara. **`cap = 5` es el default** (`PSFAO_DEFAULT_WEIGHT_CAP`), y el valor viaja en el
documento (`psfao_fit_weight_cap`) y en el QC (`fit.weight_cap`) junto al esquema.

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
