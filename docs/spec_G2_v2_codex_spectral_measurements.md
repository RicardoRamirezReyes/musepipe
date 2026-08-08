# Especificación G2 v2 · `g2_measure_lines` — la escala de detección es empírica

Fecha: 2026-08-07. Revisión de `docs/spec_G2_codex_spectral_measurements.md`
(v1), que queda como histórico. Solo cambia **§3.3 (detección, límites y
significancia)** y lo que arrastra en el QC y la tabla; el resto de la v1 sigue
vigente palabra por palabra.

## 0. Por qué existe una v2

La v1 §3.3 ya decía lo correcto:

> «Significancia por línea: **delega en el matched filter de H01 con `rest_A` de
> la línea y los controles disponibles**; donde no haya controles (espectro ya
> 1D), z-score contra el ruido del continuo local […]»

La implementación se quedó **permanentemente en la rama de excepción**. Nunca
cargó los controles, aunque D2 los escribe al lado del objeto
(`spec_calibrated_<método>_controls.npz`, 33 posiciones), y etiquetaba contra el
σ **propagado canal a canal** — su propio QC lo declaraba con
`covariance_used: "none"`.

Lo que eso produce, medido en `ROXs12b_realigned` el 2026-08-07 tras re-correr
la cadena sobre el cubo de 29 exposiciones:

| | |
|---|---|
| Hα del compañero, flujo | 5.63e-17 erg s⁻¹ cm⁻² |
| σ propagado (G2) | 5.08e-18 → **z = 11.1 → `detected`** |
| controles a la misma separación (E1) | mediana z = 22.6, p99 = 41.4 |
| z del objeto en esas unidades | 15.9 → **por debajo de la mediana de los controles** |
| veredicto de E1 | `non_detection`, FAP = 0.97 |

El chequeo **V3** de la propia v1 lo detectó y lo marcó `blocking` («two
statistics on the same data disagree»). Esta v2 elimina la causa en vez de
seguir reportando el síntoma.

## 1. §3.3 revisada

### 1.1 La distribución nula

Para **cada línea del catálogo** se repite la medida completa —las mismas
ventanas de línea y continuo, el mismo continuo lineal, el mismo filtro
adaptado— sobre los **N espectros de control** del método canónico, y con el
**mismo `flux_err`** que el objeto, para que los `z` salgan en las mismas
unidades. Las ventanas se construyen **una sola vez por línea** y se pasan a
objeto y controles: si cada uno eligiera las suyas, dejaría de ser la misma
medida.

Es el mismo procedimiento que E1 (`matched_filter_scan`) y usa el mismo
estimador de cuantil que E3 (`empirical_upper_quantile`), a propósito: dos
etapas no deben medir lo mismo de dos maneras.

### 1.2 El factor de inflación

```
sigma_inflation = cuantil_(1−fap)(z_controles) / z_gauss_(1−fap)      (fap = 0.01)
```

Es **cuánto subestima** el σ propagado la dispersión real de esta medida. 1.0 =
lo describe bien; 14 = se queda catorce veces corto.

**Suelo en 1.0** (`sigma_inflation_applied = max(sigma_inflation, 1)`). Los
controles solo pueden **destapar** dispersión de más, nunca quitar la que el
presupuesto ya contabiliza: comparten con el objeto los sistemáticos globales
(calibración de flujo, throughput, PSF), que se cancelan en la comparación y por
tanto la nula no los ve. Sin el suelo, una línea cuyos 33 controles salgan
tranquilos por azar se divide por <1 y **se fabrica significancia** — pasó en la
primera implementación con `Ca II 8662`, que con z nominal 1.40 e inflación 0.27
salía `detected`. Cuando el suelo actúa se emite la bandera
`null_quieter_than_budget`: se ve, no se silencia.

### 1.3 Etiquetas

**Los umbrales de la v1 no cambian: 5 y 3.** Cambia la unidad en la que se
miden.

```
z_emp  = z_score / sigma_inflation_applied
detected     z_emp >= 5
marginal     3 <= z_emp < 5
upper_limit  z_emp < 3
```

El límite superior 5σ se calcula con el σ **inflado por el mismo factor**: uno
calculado con el σ propagado sería optimista exactamente en esa proporción.

### 1.4 Sin controles

Si no hay `npz` de controles para el método canónico, o su malla no coincide con
la del objeto, G2 **no adivina**: etiqueta con el σ propagado, pone
`null_source` al motivo concreto, y emite un `open_issue` **`blocking`**
diciendo que sus `detected` no son comparables con el veredicto de E1. La
prohibición de fallbacks silenciosos de este repositorio aplica aquí como en
todas partes.

## 2. Campos nuevos

Por línea, en `tables/g2_line_measurements.csv` y en `LineMeasurement`:

| campo | qué es |
|---|---|
| `z_emp` | el z en σ medidos — el que decide la etiqueta |
| `sigma_inflation` | **medido**, sin suelo: la métrica de diagnóstico |
| `sigma_inflation_applied` | `max(medido, 1)`: el que entra en la etiqueta |
| `fap_empirical` | fracción de controles que iguala o supera al objeto |
| `n_controls`, `min_resolvable_fap` | tamaño de la nula y el suelo `1/(n+1)` que impone |
| `null_source` | de qué fichero salió la nula, o por qué no hay |

Y en el QC, el bloque `empirical_scale` con la procedencia, la mediana, el rango
y los percentiles 16/84 de la inflación entre líneas.

## 3. Para qué sirve `sigma_inflation` además de para etiquetar

Es una **métrica de salud de la reducción**, y por eso se publica por línea y sin
suelo. Se calcula sobre los mismos datos en cada paso y para cada método, así
que su **estabilidad** es la señal: entre líneas vecinas, entre métodos de
extracción y entre reducciones sucesivas debería moverse poco. Un salto que no
se explique por algo que se haya cambiado a propósito —más exposiciones, otra
apertura, otro fondo— apunta a la reducción y no a la física.

El caso que motivó esta v2 es justo ese: al pasar de 7 a 29 exposiciones, el σ
propagado cayó ×4–10 en los seis métodos mientras la dispersión real entre
controles no bajaba con él (`z_99_empirical` ×5–12 arriba). El límite final
apenas se movió un 5 %, pero la inflación se disparó — y es lo que convirtió una
no detección en una «detección» en la etapa que no miraba los controles.

## 4. Lo que esta v2 NO cambia

- Ni el catálogo, ni las ventanas, ni el continuo, ni el ajuste de perfil, ni el
  Monte Carlo, ni las convenciones de EW/RV: todo eso es la v1.
- Ni los umbrales 5/3.
- Ni el chequeo **V3** contra H01, que se conserva como guardia: si vuelve a
  saltar, es que las dos escalas se han separado otra vez.
- No toca E1 ni E3, que ya trabajaban con la nula empírica.
