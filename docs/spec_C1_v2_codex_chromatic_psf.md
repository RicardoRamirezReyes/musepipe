# Especificación C1 v2 · `E01_chromatic_psf` — la PSF se ajusta POR OBSERVACIÓN

Fecha: 2026-08-15. Sustituye a `spec_C1_codex_chromatic_psf.md` en lo que aquí se dice y **hereda
todo lo demás sin cambios** (§3.1–§3.5 del ajuste por bin, la métrica del anillo, el híbrido, la
normalización, el `psfao_wave_bin_A`, los pesos del ajuste al combinado). Léase la v1 primero: esto
es el delta, no un documento independiente.

Decisión y medidas que la motivan: `docs/2026-08-15_psf_por_observacion.md`.
Etapa hermana: `docs/spec_C1b_codex_perobs_subtraction.md`.

---

## 1 · Qué cambia y por qué

La v1 ajusta **una** PSF analítica al cubo **combinado**. Ese cubo es una media pesada de 29–30
exposiciones con seeing y calidad de AO distintas, y **la mezcla de N PSF no es una PSF**: ninguna
Psfao ni Moffat puede tener a la vez el núcleo de la mejor noche y el halo de la peor. El sesgo cae
entero sobre la razón núcleo/total, que **es** la corrección de apertura que aplican C2/C3 — el
mismo mecanismo que en ROXs 42B b dejó la `apcorr` 4-5× de más.

**A partir de v2, el alcance por defecto es `per_observation`**: se ajusta la PSF en cada
exposición, y ese modelo es el que se usa **donde vale** — en su propia exposición, para la resta de
C1b. Qué modelo describe al cubo **combinado** es otra pregunta, y la §4 la contesta con una medida.

## 2 · El knob `psf_scope`

| valor | qué hace |
|---|---|
| `per_observation` (**default**) | ajusta cada exposición **y** el combinado, y escribe los dos modelos; lo que se entrega como `psf_model.json` lo decide §4 |
| `combined` | el camino de la v1, bit a bit |

Un run que **no declare exposiciones** (sin `stream_combine_plan.json`: reducción monolítica, run
histórico) cae al camino `combined` **diciéndolo** en `psf_scope`, `psf_scope_requested`,
`per_observation.status = "unavailable"` y en `open_issues`. Nunca en silencio.

## 3 · De dónde sale la geometría

Del `stream_combine_plan.json` con el que se construyó el combinado (`musepipe.observations`), no
de una convención de nombres sobre un disco: ventana, desplazamiento subpíxel, peso y `EXPTIME` por
exposición. Cada exposición se lee con las mismas funciones que la combinaron, así que la primaria
cae en el centro del array — que es donde `fit_bin` la da por hecha.

Si una ruta del plan ya no existe, se re-resuelve por `exposure_id` contra `perexp_cubes` /
`perexp_dir` **y se vuelve a medir el centroide**, porque un cubo re-reducido no tiene por qué caer
en el mismo píxel; la diferencia viaja en la procedencia. Si algo no se resuelve, la etapa falla con
la lista.

## 4 · La mezcla: `form = "mixture"` — y qué se entrega

**La aritmética del dato es exacta; la de la mezcla, no.** La razón núcleo/total del cubo combinado
es `Σw·F25 / Σw·Fbox3` —un **cociente de sumas**, no una media de cocientes—. Medido en ROXs 12 b a
7000 Å sobre las 29 exposiciones: la razón individual va de **3.24 a 81.88** (mediana 4.53), la
media de los cocientes sale 21.28 y no significa nada, el **cociente de las sumas sale 5.40**, y el
cubo combinado en disco da **5.48**. El 1.5 % que separa a los dos es el `sigclip` y el paso por B2.

Lo que **no** se sigue de ahí, y el 2026-08-15 se escribió como si se siguiera, es que la mezcla
herede esa exactitud: la mezcla se monta con **modelos**, y cada uno llega con su propio error de
V4 (mediana 4.2 %, hasta 9.8 % en ROXs 12 b). Esos errores no se cancelan. **Medido el 2026-08-17
sobre la misma vara** (44 bins, mismas imágenes, fondos y máscara, los dos documentos evaluados
como los consume C2/C3):

| | V4 mediana | azul <6000 Å | rojo >7500 Å |
|---|---|---|---|
| ajuste analítico al combinado | +8.83 % | +17.32 % | +5.90 % |
| **mezcla** | **+6.77 %** | **+0.34 %** | +8.55 % |

La mezcla gana, y **por eso se entrega** (`psf_mixture_as_combined_model`, por defecto `true`) —
pero por medida, no por construcción, y **las dos siguen fallando la tolerancia del 3 %** de la §7.
Lo que decide es el reparto en λ: la mezcla clava el azul, que es el problema abierto de C1, y paga
en el rojo. El ajuste analítico tampoco tiene garantía por el otro lado: es el que en ROXs 42B b
llegó a **+356 %**.

Las dos V4 tienen que medirse **con la misma vara**, y hasta el 2026-08-17 no lo estaban: la del
ajuste salía de la rama ganadora en sus propios bins, con sus reconstrucciones y el `bck` de su
ajuste, y así la mezcla parecía empeorar (+6.77 contra +6.04 %). El QC publica ahora las dos sobre
la misma rejilla (`combined_fit_encircled_energy` y `mixture_encircled_energy`, mismo `n_bins`), y
la de la rama queda como `combined_fit_v4_pct_own_grid`, etiquetada como no comparable.

> El rango 3.24–81.88 no es ruido de la medida: hay exposiciones con el núcleo prácticamente
> destruido que hoy entran en el combinado con todo su peso. Es una pregunta de **rechazo de
> exposiciones** que la cadena nunca se ha hecho, y ahora hay con qué hacérsela.
>
> Pero **no por re-pesado de la mezcla**, como se escribió el 2026-08-15. Medido el 2026-08-17:
> quitar de la mezcla las 5 peores lleva su V4 a **−12.55 %** y las 10 peores a **−20.64 %**, peor
> que dejarlas. El cubo combinado sigue conteniéndolas, así que una mezcla que las excluye ya no
> describe al cubo que C2–C6 miden. **Rechazar exige re-combinar.**

### 4.a · Cómo se construye la mezcla

```
PSF_comb(λ) = Σ wᵢ · Fᵢ(λ) · PSFᵢ(λ)  /  Σ wᵢ · Fᵢ(λ)
```

* `wᵢ` es el peso del combinado (`EXPTIME`) y `Fᵢ(λ)` el **flujo de la estrella dentro de
  `norm_radius_px` medido sobre el dato** de esa exposición. Los pesos son cromáticos a propósito:
  el combinado es una media pesada de brillo, y la transmisión y la masa de aire cambian entre
  exposiciones.
* Cada componente está normalizada a 1 dentro de `norm_radius_px`, así que la mezcla también lo
  está: `psf_roundtrip_error` sigue valiendo exactamente y el contrato de `evaluate_psf_model` no
  cambia, así que cualquier consumidor de `psf_model.json` podría evaluarla sin cambios — lo que
  decide si la ve es §4, no una limitación técnica.
* `scaled_psf_model` escala la mezcla entera (una dilatación conmuta con la suma pesada), no
  componente a componente.

Se exige que todas las componentes compartan `norm_radius_px`; si no, la suma no está normalizada
en la misma región y el cociente núcleo/total deja de significar nada.

## 5 · Las dos formas, y quién elige

Las dos formas se ajustan **en cada exposición**, como en la §3.4 de la v1, y sus residuos de
anillo por exposición se publican. Pero **el anillo de una exposición no elige la forma**: se mide
a la separación del compañero (71 px en ROXs 12 b), donde una exposición de 300 s no tiene señal.
Medido: 30–310 % contra el 8–20 % del combinado, y con esos números la comparación es una moneda al
aire.

Elige quien tiene S/N: `e01_psf_form` si el objeto la congela, y si es `auto`, la forma que ganó en
el cubo combinado. Queda escrito en `per_observation.form_source` y `form_used`.

## 6 · El peso del ajuste por exposición

Knob propio, `psf_perobs_fit_weighting` (default **`stat`**), independiente del
`psf_fit_weighting` del combinado. El `relative` con tope 5 se eligió midiendo sobre un halo
promediado sobre 29 exposiciones; en una sola ese halo está dominado por el moteado de la AO
residual. Medido (una exposición de ROXs 12 b, 6 bins, residuo del anillo): `stat` gana en toda la
banda y por mucho en el azul (133 % contra 310 % a 4800 Å). La V4 sale bien con los tres pesos
(−4 % a +1 %): lo que el peso decide es el halo, no el núcleo.

## 7 · Productos nuevos

| producto | contenido |
|---|---|
| `stages/psf_model.json` | la **mezcla** (lo que consumen C2–E5), salvo que `psf_mixture_as_combined_model=false` |
| `stages/psf_model_mixture.json` | la mezcla otra vez, siempre en su propio fichero: así C1b no depende del knob |
| `stages/psf_model_combined.json` | copia del ajuste al combinado, para poder comparar sin ambigüedad |
| `stages/psf_model_perobs.json` | índice: forma, anillo, V4, flujo y cabecera de cada exposición |
| `stages/psf_perobs/params_<exposure_id>_{psfao,moffat}.csv` | las filas por bin de cada exposición |
| `stages/observation_plan.json` | la vista por observación resuelta, con su procedencia |

Los documentos de PSF por exposición **no** se duplican fuera de la mezcla: tenerlos dos veces
invita a editar la copia que no se evalúa.

## 8 · QC: el bloque `per_observation`

Además de lo de la v1, `stage_e01_qc.json` gana:

* `psf_scope`, `psf_scope_requested`;
* `per_observation.n_exposures_used` / `n_exposures_failed` / `failures` (lo que no se pudo ajustar
  se cuenta, nunca se descarta en silencio);
* `per_observation.forms_chosen`, `form_source`, `form_used`, `form_stable_across_exposures`;
* `per_observation.v4_per_exposure_pct` (mediana, mínimo y máximo);
* `per_observation.exposures[]` con el resumen de cada una;
* **`per_observation.delivered_vs_combined_fit`** — la V4 del ajuste analítico al combinado contra
  la V4 de la mezcla, las dos medidas sobre el **mismo** dato combinado y con la misma máscara, más
  el residuo de anillo de la mezcla y cuál de los dos se entregó (`delivered`). La mezcla es exacta
  por construcción para ese cociente, así que lo que quede mide el `sigclip` del combinado y el paso
  por B2 — no un defecto del modelo.

## 9 · Verificaciones que se añaden

1. Una mezcla de **una** componente es esa componente, bit a bit.
2. La mezcla está normalizada a 1 dentro de `norm_radius_px` a cualquier λ.
3. Los pesos son `wᵢ·Fᵢ(λ)`: una componente que se apaga hacia el rojo pesa menos allí.
4. La suma se construye **una vez por bin de λ** (si no, con 29 componentes el ajuste por canal de
   C4 pasaría de minutos a horas).
5. Lo que rompería la mezcla en silencio —radios de normalización distintos, una componente sin
   flujo— levanta.
6. `psf_scope=combined` reproduce el `psf_model.json` de la v1.

## 10 · Coste

~9–15 min por exposición (43 bins × 2 formas), ~70–110 min por objeto con 4 hilos. La resolución de
la vista por observación cuesta 104 s la primera vez y luego se lee de `observation_plan.json`.
Nada de esto corre dentro de los tests.
