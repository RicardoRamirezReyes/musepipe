# Especificación E1 · `H01_halpha_detection` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa E1 (ESENCIAL) del plan
`docs/plan_roxs12_reduccion_multimetodo.md`. Prerrequisitos: D2 (espectros
calibrados de todos los métodos), E4 al menos planificada (el criterio usa su
throughput a posteriori para flujos, no para la significancia).

Reutiliza el núcleo estadístico YA VALIDADO del repo: z empírico contra
controles (`stage07b`) y calibración look-elsewhere (`stage08c`, 31 controles,
3465 canales, patrón C4 del historial). El trabajo nuevo es de interfaz y de
protocolo, no de estadística.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: cuantificar la evidencia de emisión Hα en el espectro de
cada método con un criterio fijado ANTES de mirar los espectros nuevos.

**El criterio de detección (congelado aquí)**: se declara detección si y solo
si (a) FAP global look-elsewhere < 1% en ≥ 2 métodos con extracción
independiente del fondo (p. ej. psffit y aperture), Y (b) el centroide de la
señal es consistente con la RV sistémica de config dentro de la LSF. Todo lo
demás es "no-detección" o "candidato a investigar en E2". El agente no puede
relajar ni endurecer este criterio según lo que vea.

**Definición de terminado**:

1. Interfaz nueva: `stage07b`/`stage08c` aceptan un `SpectrumProduct` (+ sus
   controles en el mismo formato) como entrada, SIN tocar su núcleo
   estadístico (funciones de z, máximos y FAP intactas — regresión §6/V1).
2. Matched filter en velocidad ejecutado por método y por plantilla (§3).
3. `tables/halpha_detection_by_method.csv` con flujo, σ, FAP local y global,
   centroide y FWHM (si hay señal); figuras por método.
4. Veredicto emitido por el código según el criterio; verificaciones §6.

---

## 1. Límites duros

1. **Criterio congelado** (§0). Cambios = revisión de spec con el usuario.
2. El núcleo estadístico de 07b/08c no se modifica; solo se envuelve. La
   equivalencia numérica sobre el caso histórico es el gate de la interfaz.
3. **El nº de pruebas se declara todo**: plantillas × ventanas × métodos
   entran en la corrección look-elsewhere como corresponda (§3.3); prohibido
   añadir plantillas después de ver resultados.
4. Los espectros de entrada son los calibrados por D2 (todos los métodos, no
   solo el canónico). Sin re-extracciones ad hoc.
5. Git: rama `stage-e1-halpha`.

---

## 2. Entradas

- `SpectrumProduct` calibrados (D2) de: aperture, optimal-ls, optimal-psfsub,
  psffit; controles al mismo radio por método (31, los mismos índices
  espaciales para todos).
- LSF(λ=6563) de A4; RV sistémica y error de config; máscara de canales
  buenos (los ~3465 históricos o su equivalente del run nuevo).
- Resultado histórico como referencia: Hα FAP global 100% con local-surface
  (contexto, no prior).

## 3. Operaciones

### 3.1 Plantillas (fijadas)

Gaussianas centradas en `6562.8 Å × (1 + RV_sys/c)` con anchos
{LSF, 2×LSF, 4×LSF} (acreción puede ser ancha), evaluadas en una ventana de
búsqueda de ±500 km/s alrededor del centro esperado. Tres plantillas, una
ventana: seis grados de libertad declarados por método (3 anchos × búsqueda
en velocidad, calibrada empíricamente en §3.3).

### 3.2 Estadístico

Por método y plantilla: matched filter sobre el espectro con continuo
(`cont_runmed` de D2) restado; z empírico del máximo en la ventana usando la
distribución del MISMO estadístico sobre los 31 controles (maquinaria 07b).

### 3.3 Look-elsewhere

FAP global por método: distribución de máximos del matched filter sobre los
controles barriendo TODOS los canales buenos y las 3 plantillas (maquinaria
08c extendida a la dimensión plantilla). Así la búsqueda en velocidad y en
ancho queda auto-calibrada empíricamente — sin fórmulas analíticas de trials.

### 3.4 Si hay señal

Centroide y FWHM por ajuste gaussiano; consistencia con RV sistémica (criterio
b); flujo integrado con error (y nota: el flujo publicable espera la
corrección de throughput de E4).

## 4. Puntos de atención

1. **Marco de velocidad**: los espectros D2 están en marco baricéntrico; el
   centro esperado usa la RV sistémica de config. Un error de marco aquí
   desplaza la ventana ~10–30 km/s — verificar contra `WFRAME` del header (ya
   auditado en D2, pero re-leerlo, no asumirlo).
2. **Los métodos no son independientes** (mismos fotones): por eso el criterio
   pide ≥ 2 métodos con FONDO independiente (local-surface vs modelo PSF), no
   dos métodos cualesquiera. La lista de pares admisibles va en QC.
3. **Continuo restado**: usar `cont_runmed`; la columna `sys_continuum` de D2
   dice cuánto importa la elección — si en Hα es grande, correr también con
   `cont_poly` y reportar ambos (la diferencia es sistemático, no opción).
4. **No mirar el objeto hasta que los controles pasen**: orden de ejecución
   obligatorio — primero la distribución de controles (V2), después el objeto.

## 5. Salidas

`tables/halpha_detection_by_method.csv`; `stage_h01_qc.json` (criterio, pares
admisibles, plantillas, FAPs, veredicto, orden de ejecución respetado);
figuras: espectro ±100 Å por método con plantillas, distribución de máximos de
controles vs objeto (como 08c), señal en velocidad.

## 6. Verificaciones

- **V1 — Regresión de interfaz**: el caso histórico (run de validación C1 con
  inyección S/N 5) reproducido vía la interfaz nueva: mismos números que el
  QC histórico (`line_mean_snr=8.803`, FAP conocidas) dentro de precisión
  numérica.
- **V2 — Controles sanos**: distribución de máximos de controles razonable
  (sin bimodalidades ni outliers extremos que delaten un control contaminado
  — si uno lo está, se excluye CON registro y se recalcula, máx. 2).
- **V3 — Placebo rápido**: la misma cadena en dos líneas placebo (6400 Å,
  6700 Å) no supera el umbral (la batería completa de placebos es de E2).
- **V4 — Coherencia multi-método**: los FAP por método contados y comparados;
  discrepancias grandes entre métodos → insumo directo para E2.

## 7. Tests

```text
tests/test_h01_interface_regression.py  # V1 como test automatizado (con productos del run de validación)
tests/test_h01_template_dimension.py    # la FAP con 3 plantillas > FAP con 1 plantilla sobre ruido sintético (los trials cuentan)
tests/test_h01_criterion.py             # veredictos sintéticos: señal fuerte en 2 métodos admisibles → detección; en 1 → candidato; nada → no-detección
```

## 8. Protocolo de parada y reporte

Preguntar cuando: V1 no reproduzca el histórico; un control contaminado no sea
identificable; el veredicto sea "candidato" (decidir juntos el orden de E2).
Reporte: veredicto con sus números, tabla por método, figuras, y el paso
siguiente que corresponde (E2 si candidato/detección; E3 si no-detección).
