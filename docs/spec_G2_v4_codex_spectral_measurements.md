# Especificación G2 v4 · `g2_measure_lines` — continuo de plantilla en el rojo y nula espectral

Fecha: 2026-10-26. Revisión de `docs/spec_G2_v3_codex_spectral_measurements.md` (v3), que revisa la v2 y la
v1. Decisión del autor, 2026-10-26: «adelante con G2 v4», tras el informe del agente `lineas` sobre las
preguntas de Claudio (`~/muse_sesion_20261026/claudio/lineas/INFORME.md`, §6 y §8). Cambian el **continuo de
las líneas del rojo**, la **escala empírica** (que añade una segunda nula) y se añade un **límite superior al
99 %**. El resto de la v1, la v2 y la v3 sigue vigente palabra por palabra.

## 0. Por qué existe una v4

Por encima de ~7000 Å el espectro de un compañero M tardío lo dominan las bandas de TiO y VO. G2 mide cada
línea contra una **recta** ajustada en dos ventanas de 40 Å a ±8 Å (v1 §3.2), y una recta no sigue esas
bandas. Lo que G2 llama flujo de línea en el rojo es, en gran parte, la forma de la banda. Así está el
G2 v3 de los dos runs del paper (2026-10-26):

| línea | ROXs 12 b: `z` / inflación | ROXs 42B b: `z` / inflación |
|---|---:|---:|
| O I 8446 | −5.5 / 0.24 | −9.3 / 0.48 |
| Pa 17 | −6.2 / 0.28 | −4.1 / 0.37 |
| Pa 11 | −4.7 / 0.08 | −4.7 / 0.24 |
| Pa 9 | −1.4 / 0.07 | −3.4 / 0.14 |
| He I 7065 | −4.8 / 1.08 | 0.2 / 0.53 |

Los `z` negativos de varios sigmas son bandas en absorción, no líneas. La escala empírica de la v2 no lo ve:
los 33 controles **no tienen fotosfera**, así que su dispersión no contiene el error de la recta, y en el
rojo salen más tranquilos que el presupuesto (inflación 0.07–0.48, casi todas con la bandera
`null_quieter_than_budget` y el suelo en 1). La nula espacial mide bien el ruido del halo; el error del
continuo fotosférico necesita otra nula.

El agente `lineas` midió la alternativa (2026-10-26): una plantilla joven de Clase III del tipo del
compañero, multiplicada por un polinomio de grado 2, y una nula **espectral** (la misma medida en
longitudes de onda sin líneas del propio espectro). Con plantilla, la nula espectral es 1.6–5 veces la
espacial en ROXs 12 b y 1–2.7 en ROXs 42B b. En el azul las dos coinciden.

## 1. Continuo de plantilla en el rojo

**Qué líneas.** Las de `rest_A ≥ g2_template_continuum_min_A` (por defecto **7000 Å**). Las demás, Hα
incluida, siguen con la recta de la v1 sin ningún cambio.

**Qué plantilla.** La declara el run, sin valor por defecto:

```json
"g2_continuum_template": {"library": "templates_young", "object": "LM601"}
```

`library` es el `name` de una biblioteca de `g3_template_libraries` y `object` el `object` de una de sus
entradas. Se prepara **como la prepara G3**: `declared_prep_kwargs` de la plantilla, marco del dato
(`g3_data_wave_frame`), degradada a la LSF de G2 y remuestreada a la malla del espectro. Sin extinción ni
escala: el polinomio las absorbe.

**El modelo.** `continuo(λ) = T(λ) · (c0 + c1 x + c2 x²)`, `x = (λ − λc)/H`, con
`H = g2_template_fit_half_A` (por defecto 60 Å), ajustado por mínimos cuadrados en los canales con
`|λ − λc| ≤ H` que:

- quedan fuera de la ventana de línea (`|λ − λc| > 6 Å`, la de la v1);
- quedan a más de 6 Å de cualquier otra línea del catálogo (la exclusión de vecinas de la v1);
- no llevan bandera (`flags != 0`);
- no caen en una banda telúrica **fuerte** (`musepipe.telluric_lines`, `severity = strong`);
- tienen la plantilla y el flujo finitos.

Con menos de `g2_template_min_fit_pixels` (por defecto 30) la línea es `not_measurable` con motivo
`template_fit_pixels=<n>`; jamás se mide con lo que quede.

**El mismo modelo para todo.** El objeto, los 33 controles y los placebos de §2 llevan **el mismo** modelo
de continuo, con la misma plantilla y la misma máscara de ajuste centrada donde toque. Si cada uno usara
otro continuo, la nula dejaría de ser la misma medida (v2 §1.1).

**Sin plantilla declarada.** Las líneas del rojo se miden con la recta, como en la v3, llevan la bandera
`linear_continuum_over_molecular_bands` y G2 levanta un issue `major`. No hay continuo de plantilla por
defecto: elegirla es elegir el tipo espectral, y eso lo declara el run.

## 2. La nula espectral

Para cada línea se repite la medida —el mismo continuo, el mismo filtro adaptado, el mismo `flux_err`— en
**longitudes de onda sin líneas del propio espectro del objeto** («placebos»):

- centros `λc + k Δ`, con `Δ = g2_placebo_step_A` (12.5 Å, 10 canales: placebos casi independientes) y
  `g2_placebo_min_A ≤ |k Δ| ≤ g2_placebo_max_A` (20 y 300 Å);
- se descarta el centro que cae a menos de 10 Å de una línea del catálogo, a menos de 6 Å de una banda
  telúrica fuerte, a menos de 50 Å del hueco del láser (`AO_LASER_WINDOW_A`), a menos de 50 Å (recta) o de
  `H` (plantilla) del borde del espectro, o que tiene un canal con bandera en su ventana de línea o la
  plantilla no finita en su ventana de ajuste.

Un placebo de recta usa ventanas de continuo construidas alrededor de su centro excluyendo **todas** las
líneas del catálogo. Cada placebo da un `z` en las mismas unidades que el objeto y los controles.

`sigma_inflation_spectral` = cuantil 99 de esos `z` / `Z_GAUSS_99`, con el mismo estimador de cuantil que
la v2 (`empirical_upper_quantile` de E3). Se publica **medido**, sin suelo, con `n_placebos`. Con menos de
`g2_min_placebos` (por defecto 10) no se aplica, y la línea lleva `spectral_null_unavailable`.

La nula espacial ve el halo y no la fotosfera; la espectral ve el error del continuo y la estructura
fotosférica, pero no el halo (es el mismo espectro). Ninguna de las dos basta sola.

## 3. La escala aplicada

```
sigma_inflation_applied = max(sigma_inflation, sigma_inflation_spectral, 1)
z_emp = z_score / sigma_inflation_applied
```

`sigma_inflation` **conserva** su significado de la v2 (la nula espacial, la métrica de salud de la
reducción); la espectral va en su propia columna. Los umbrales 5/3 no cambian. Cuando la espectral manda,
la línea lleva la bandera `spectral_null_dominates`.

## 4. Límite superior al 99 %

Por línea, en el estimador que decide la etiqueta (el filtro adaptado a λ fija):

| campo | qué es |
|---|---|
| `flux_mf` | flujo del filtro adaptado del objeto |
| `flux_mf_sigma_emp` | `σ_mf,propagado × sigma_inflation_applied`: la σ empírica de `flux_mf` |
| `flux_upper_limit_99` | `(max(flux_mf, 0) + 2.326 · flux_mf_sigma_emp) / throughput` |

Es la regla de E3 v2 (`measured_plus_margin`): lo que se puede afirmar del flujo verdadero dado lo medido.
Como `2.326 · σ_mf,propagado · inflación` es el cuantil 99 de la nula en flujo, el límite es el flujo
medido más el cuantil 99 de la nula que manda. El límite a 5σ de la v2 se conserva, ahora con la escala
aplicada de §3.

## 5. Hα y la reconciliación V3

Hα (6562.8 Å) queda por debajo de 7000 Å: su continuo, su `flux_direct`, su `z_score`, su ajuste de perfil
y su Monte Carlo son **idénticos bit a bit** a los de la v3. Solo su `z_emp` puede bajar, si su nula
espectral es más ancha que la espacial; nunca subir. V3 (spec G2 v3) no cambia. G3 solo lee Hα de G2 (sus
relaciones son `Halpha` y `halpha_aoyama21` en los dos runs del paper), así que G3 no se mueve.

## 6. QC

- `empirical_scale` añade `spectral`: mediana, rango y percentiles 16/84 de `sigma_inflation_spectral`
  entre líneas, número de placebos, y las líneas donde manda la espectral.
- Bloque nuevo `continuum_model`: `template_min_A`, la plantilla declarada (biblioteca, objeto, tipo,
  cita, `source_npz`) o el motivo por el que no hay, `fit_half_A`, `poly_degree = 2`, las bandas telúricas
  excluidas y las líneas medidas con cada modelo.
- `rule: "spec_G2_v4"` en la raíz del QC.
- Columnas nuevas de la tabla: `continuum_model`, `sigma_inflation_spectral`, `n_placebos`, `flux_mf`,
  `flux_mf_sigma_emp`, `flux_upper_limit_99`.

## 7. Lo que NO cambia

- El catálogo, las ventanas de línea, la LSF (un valor, el de Hα, como en la v3: una LSF(λ) es otro cambio),
  los umbrales 5/3, el suelo en 1, la nula espacial y V3.
- Las líneas del azul y Hα, salvo su `z_emp` (§5).
- E1, E3 y G3.

## 8. Implementación y pruebas

- `musepipe/lines.py`: `template_fit_mask`, `placebo_centers`, el continuo de plantilla en `measure_line`
  (argumento `continuum_template`, opcional) y la nula espectral en `measure_catalog`
  (`spectral_null=True`, opcional). Los dos son **opcionales en la librería**, para que quien la llame sin
  pedirlos obtenga la v3 exacta; la etapa los activa.
- `musepipe/stages/stage_g2_measure_lines.py`: resuelve la plantilla del knob, construye las máscaras y
  publica el QC.
- Tests nuevos en `tests/test_g2_v4_template_and_spectral_null.py`.
