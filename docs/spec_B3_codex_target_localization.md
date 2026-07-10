# Especificación B3 · `01c_target_localization` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa B3 (ESENCIAL, nueva) del plan
`docs/plan_roxs12_reduccion_multimetodo.md`. Prerrequisito: B2 (cubo apilado y
corregido de stripes). Sustituye las coordenadas hardcodeadas del target por
coordenadas medidas, validadas contra la astrometría publicada.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: medir de forma reproducible las posiciones de la primaria
y del compañero en el cubo de trabajo, validarlas contra la separación/PA
esperadas de la literatura, medir la deriva cromática del centroide, y
escribirlas como coordenadas canónicas que TODAS las etapas posteriores
(C1–C4, E4) leerán de QC — nunca más de un número escrito a mano.

**Valores de referencia actuales (para validar, NO para asumir)**: el repo usa
`(x, y) = (72, 152)` para el compañero y `(y, x) ≈ (85, 85)` para la primaria
en el crop 170×170 de ROXs12b, y la literatura (Bowler et al. 2017) da
separación ~1.75″ (~1.78″) con PA ~240°. La medición nueva debe caer cerca de
estos valores; si no cae, eso es un hallazgo para reportar, no para silenciar.

**Definición de terminado**:

1. Posiciones de primaria, compañero y del segundo objeto del campo medidas
   con incertidumbre, en píxeles del crop Y en separación/PA en arcsec/grados.
2. Comparación contra la predicción de literatura y contra las coordenadas
   hardcodeadas históricas, ambas en QC.
3. Deriva cromática del centroide del compañero medida (§3.4) y veredicto
   sobre si los extractores deben usar centroide dependiente de λ.
4. `stage01c_qc.json` según §5; verificaciones §6 en verde.
5. `musepipe/stages/stage01c_localize.py` + tests; cero números de ROXs 12 en
   el módulo (target-specific solo en config).

---

## 1. Límites duros

1. Solo lectura sobre cubos; B3 escribe únicamente QC, tablas y figuras.
2. Solo archivos nuevos (etapa, tests, notebook fino); registro en
   `musepipe/stages/__init__.py` permitido.
3. **No sobreescribir las coordenadas históricas en configs de runs
   existentes.** Las coordenadas medidas viven en `stage01c_qc.json`; la regla
   de consumo (nuevas etapas leen QC de B3; etapas históricas siguen leyendo
   su config) queda documentada en el docstring y el reporte.
4. Convención de coordenadas: seguir EXACTAMENTE la del repo — `(y, x)` en
   arrays, `(x, y)` solo al reportar donde los docs ya lo hacen, y cada clave
   de QC lleva el sufijo `_yx` o `_xy` explícito. La confusión x↔y es el bug
   número uno de este tipo de etapa; el esquema de §5 la hace imposible de
   escribir ambigua.
5. Git: rama `stage-b3-localize`.

---

## 2. Entradas

- Cubo de B2 (`stage02_xcorr_cube_stack.fits`) — y modo degradado documentado
  si solo existe el de B1.
- Config del run: separación y PA esperadas con incertidumbre (nuevas claves
  `expected_sep_arcsec`, `expected_pa_deg`, `expected_sep_err`,
  `expected_pa_err`), escala de píxel (del header WCS; si el crop perdió WCS,
  0.2″/px WFM o 0.025″/px NFM según modo registrado en A1/A4 — verificar, no
  asumir).
- PSF preliminar: FWHM medida en la imagen colapsada roja (no requiere C1; C1
  refinará después).

## 3. Operaciones

### 3.1 Primaria

- Centroide robusto en imágenes colapsadas por bandas (4 bandas a lo largo del
  rango, excluyendo ventanas malas), método consistente con `stage01_profile`
  (maoppy si disponible, si no ajuste 2D del pico). Posición final = mediana
  entre bandas; incertidumbre = dispersión.

### 3.2 Compañero

- Imagen colapsada en banda roja (donde el compañero es más brillante, p. ej.
  7500–9000 Å excluyendo skylines fuertes), tras restar un modelo radial
  azimutal mediano del halo de la primaria (suficiente a 1.75″; no requiere C1).
- Detección por matched filter con kernel gaussiano del FWHM medido; mapa de
  S/N sobre la imagen filtrada usando ruido robusto en anillos.
- **Sin búsqueda ciega global**: buscar el máximo dentro de un radio de
  tolerancia (config, default 10 px) alrededor de la posición PREDICHA por
  separación/PA de literatura aplicada a la primaria medida. Esto evita
  engancharse al segundo objeto del campo o a un residuo del halo.
- Centroide fino por ajuste 2D en un stamp alrededor del máximo; incertidumbre
  del ajuste registrada.

### 3.3 Segundo objeto del campo

- Mismo procedimiento con su propia región de tolerancia (posición aproximada
  en config). Su única función es estar identificado para que nadie lo
  confunda con el compañero y para las máscaras de A2/C1.

### 3.4 Deriva cromática

- Centroide del compañero (y de la primaria como control) en 6–8 bins de λ;
  ajuste lineal deriva vs λ.
- Veredicto en QC: `chromatic_centroid_needed = (deriva pico a pico > 0.5 px)`.
  La deriva de la PRIMARIA mide la refracción diferencial residual del
  instrumento/reducción; si primaria y compañero derivan igual, es global (y
  los extractores pueden corregirla con un solo modelo); si difieren, reportar.

## 4. Puntos donde mantener máxima atención

1. **Confusión de fuentes**: la búsqueda restringida de §3.2 es la defensa
   principal. Si aparecen DOS máximos con S/N > 5 dentro de la tolerancia:
   detenerse y preguntar con la figura, jamás elegir por cercanía.
2. **x↔y**: sufijos obligatorios en QC (§1.4) y un test dedicado (§7).
3. **El compañero podría ser débil en continuo**: si el matched filter no da
   S/N ≥ 5 en la banda roja, probar la banda 6000–7000 Å y el colapso total
   antes de declarar no-detección; si sigue sin aparecer, ES un resultado
   mayor (cambia el plan científico: de caracterización a búsqueda de línea)
   → parar y reportar de inmediato.
4. **PA y orientación del WCS**: el signo del PA depende de la orientación
   del crop (¿norte arriba?). Derivar la matriz de rotación del WCS del
   header, no asumir norte-arriba/este-izquierda. Si el crop no conserva WCS,
   validar la orientación con la posición RELATIVA conocida del segundo objeto.

## 5. Esquema de `stage01c_qc.json`

```json
{
  "stage": "01c_target_localization",
  "run_id": "...",
  "input_cube": {"file": "...", "sha256": "..."},
  "pixel_scale_arcsec": 0.2,
  "wcs_orientation": {"north_angle_deg": 0.0, "source": "header|relative_source_check"},
  "primary": {"pos_yx": [0.0, 0.0], "err_px": 0.0, "per_band_scatter_px": 0.0},
  "companion": {"pos_yx": [0.0, 0.0], "err_px": 0.0, "snr_detection": 0.0,
                 "band_used_A": [7500, 9000]},
  "field_source": {"pos_yx": [0.0, 0.0], "snr_detection": 0.0},
  "astrometry": {"sep_arcsec": 0.0, "sep_err": 0.0, "pa_deg": 0.0, "pa_err": 0.0,
                  "expected_sep_arcsec": 0.0, "expected_pa_deg": 0.0,
                  "sep_deviation_sigma": 0.0, "pa_deviation_sigma": 0.0},
  "legacy_check": {"hardcoded_companion_xy": [72, 152], "distance_px": 0.0},
  "chromatic": {"companion_drift_px_peak_to_peak": 0.0,
                 "primary_drift_px_peak_to_peak": 0.0,
                 "chromatic_centroid_needed": false},
  "open_issues": []
}
```

## 6. Verificaciones

- **V1 — Detección**: compañero con S/N ≥ 5 en el mapa filtrado; figura stamp
  con isofotas, cruz del centroide y círculo de tolerancia.
- **V2 — Astrometría**: separación y PA dentro de 2σ combinadas de la
  predicción; figura campo completo con las tres fuentes y el vector esperado.
- **V3 — Consistencia con lo histórico**: distancia entre posición medida y la
  hardcodeada < 2 px; si es mayor, issue abierto con implicación explícita
  (¡todas las etapas históricas usaron la posición vieja!).
- **V4 — Deriva cromática**: figura centroide vs λ para compañero y primaria
  con el umbral de 0.5 px marcado.
- **V5 — Unicidad**: mapa de S/N de la región de tolerancia mostrando UN solo
  máximo significativo.

## 7. Tests

```text
tests/test_stage01c_synthetic.py    # escena sintética (2 fuentes + halo) con posiciones conocidas → recuperadas a <0.2 px
tests/test_stage01c_confusion.py    # dos fuentes dentro de la tolerancia → la etapa LANZA la excepción de ambigüedad (no elige)
tests/test_stage01c_yx.py           # inyectar fuente en (y=30, x=90) asimétrico → QC reporta exactamente eso en pos_yx (mata el bug x↔y)
tests/test_stage01c_drift.py        # deriva lineal sintética de 1 px → detectada y chromatic_centroid_needed=true
```

## 8. Protocolo de parada

Preguntar cuando: ambigüedad de fuentes (V5); compañero no detectado tras
§4.3; |desviación astrométrica| > 2σ; discrepancia > 2 px con lo hardcodeado;
WCS ausente y orientación no verificable con el segundo objeto.

## 9. Reporte final

Posiciones medidas con errores (tabla), astrometría vs literatura, veredicto
cromático, comparación con coordenadas históricas y su implicación,
verificaciones V1–V5 con figuras, checklist de límites, comando de reproducción.
