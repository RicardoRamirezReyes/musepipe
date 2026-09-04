# Especificación E1 v2 · `H01_halpha_detection` — el estimador de FAP se declara

Fecha: 2026-09-02. Sustituye a `docs/spec_E1_codex_halpha_detection.md`, que
sigue vigente en todo lo que esta versión no cambia. **Lo único que cambia es
de dónde sale la FAP que decide el veredicto.** El estadístico, las plantillas,
los máximos, los controles y la condición de RV no se tocan.

## 1 · Por qué

El criterio de v1 pide **FAP global look-elsewhere < 1 %**. La FAP se estima
contando excedencias sobre los controles:

```
FAP = (excedencias + 1) / (n_controles + 1)
```

El «+1» es correcto —impide afirmar probabilidad cero con muestra finita— pero
impone un **suelo**: con 33 controles el mejor valor posible es 1/34 = 0.029.
**0.029 > 0.01, así que el criterio de v1 no lo puede pasar ningún dato.** El
veredicto `non_detection` de v1 no distingue «no hay señal» de «no se puede
certificar».

Y no se arregla con más controles. Los controles van en un anillo al radio de la
compañera, y `docs/noise_model.md` mide correlación espacial hasta 5×5 px:

| separación mínima | ROXs 12 b (r=71.2 px) | ROXs 42B b (r=46.6 px) |
|---|---|---|
| 5 px (independientes) | 76 ctrl → 0.0130 | 49 ctrl → 0.0200 |
| 3 px (aperturas tocándose) | 128 ctrl → 0.0078 | **84 ctrl → 0.0118** |

En ROXs 42B b **ni violando la independencia** se baja de 0.01. Es geometría del
campo, no una limitación del run. Medido en
`docs/2026-09-02_criterio_fap_opciones.md`.

## 2 · Qué cambia

`classify_h01_verdict` lee la FAP de la columna que indica
**`h01_fap_estimator`**, declarado en el config:

| valor | columna | qué hace |
|---|---|---|
| `empirical` (defecto) | `global_empirical_fap` | cuenta excedencias; suelo `1/(n+1)` |
| `parametric` | `global_parametric_fap` | ajusta una cola a la MISMA nula |

**El defecto sigue siendo `empirical`**: cambiar el estimador cambia el
veredicto, así que no puede ocurrir por omisión. Un run que no lo declare se
comporta exactamente como en v1.

La familia es **Gumbel de dos parámetros** (`h01_fap_family`), por MLE sobre los
mismos `null_maxima` que ya se calculan. Es la familia correcta por
construcción: la nula son máximos de bloque.

**No se admite GEV ni GPD.** Con n≈33 el parámetro de forma no está constreñido
y, cuando sale negativo, la distribución tiene extremo superior finito y
devuelve `p = 0` exacto. Eso no es evidencia infinita: es exceso de confianza
del ajuste. Medido el 2026-09-01 en los dos objetos.

## 3 · El número no viaja solo

`parametric_fap` devuelve siempre el diagnóstico, y E1 lo publica en
`stage_h01_qc.json → parametric_fap.by_method`:

- **`ks_p`** — bondad de ajuste. Si `< 0.05`, la cola no describe la nula y su
  FAP **no vale**: E1 levanta una `open_issue`.
- **`extrapolation_sd`** — cuánto se extrapola por encima del mayor control, en
  unidades de la σ de la nula. Si `> 3`, `open_issue`: **el orden de magnitud es
  defendible, la cifra no.**
- **`ci95`** — intervalo por bootstrap (`h01_fap_n_boot`, semilla
  `h01_fap_seed`). **Se publica siempre y se cita con el valor**: un 8.3e-03 con
  IC [2.0e-03, 1.9e-02] cruza el umbral y no es una detección.
- `loc`, `scale`, `n_null`, `family`, `reason`.

Casos en que **se niega a dar un número** (`fap = nan` con `reason`):
`insufficient_null_sample` (< 8 controles), `degenerate_null_zero_spread` (la
nula no tiene dispersión), `fit_failed`, `degenerate_fit`.

## 4 · Con `empirical`, el suelo se denuncia

Cuando el estimador es `empirical` y `1/(n+1) ≥ h01_detection_fap`, E1 levanta
una `open_issue` por método diciendo que **el criterio es inalcanzable** y que
el `non_detection` puede no significar ausencia de señal. En v1 esto no se
decía, y es la razón de que el problema tardara en verse.

## 5 · Lo que NO cambia

- El estadístico, las plantillas y los máximos: `matched_filter_scan`,
  `_scan_maximum` y `analyze_halpha_method` calculan lo mismo.
- `global_empirical_fap` **se sigue calculando y publicando siempre**, sea cual
  sea el estimador. Es la medida cruda y no se pierde.
- La condición (b) de RV dentro de la LSF, y la exigencia de un par admisible.
- El resto de la spec v1.

## 6 · Verificaciones

`tests/test_h01_parametric_fap.py`:

- la cola baja del suelo del contador donde el contador no puede;
- el diagnóstico viaja con el número, y el bootstrap contiene el valor;
- nula degenerada, muestra corta y familia desconocida **no** producen un
  número: producen `nan` con motivo, o `ValueError`;
- **con las mismas filas, `empirical` da `non_detection` y `parametric` da
  `detection`** — el test que fija que el estimador es una decisión, no un
  detalle.

## 7 · Lo que esta spec NO resuelve

Que el exceso sea astrofísico. La cola dice que **no es compatible con el ruido
tal como lo muestrean los controles**. Un sistemático que afecte a la posición
de la compañera y no a los controles —todos al mismo radio, distinto azimut—
produce la misma firma. Ver `docs/2026-09-02_mitades_noche_buena.md`: las 28
mitades escalan igual para una línea real que para un residuo constante.
