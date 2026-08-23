# Especificación A3 v3 · `00t_telluric` — las dos vías se miden, y el empate lo gana molecfit

> **v3, 2026-08-22.** Sustituye a `spec_A3_v2_codex_telluric.md` (2026-07-31), que se conserva
> intacta: es el esquema bajo el que se emitieron los QC congelados. La v2 describía una etapa
> que **medía la profundidad y decidía si hacía falta corregir**; la v3 describe una que
> además **ajusta las dos vías, las puntúa con la misma métrica y se queda con la mejor**.
>
> El cambio de fondo es que la razón por la que la cadena no usaba molecfit era **falsa**.
> «molecfit no converge» (2026-07-06) resultó ser `WLC_CONST = −0.05`, el defecto de la
> receta. De ahí la regla que ordena toda esta spec: **una no-convergencia nunca es un motivo
> para descartar molecfit**; es un defecto de parámetros o de configuración del dato.

---

## 0. Qué cambia respecto a la v2

| | v2 | v3 |
|---|---|---|
| vías | STD_TELLURIC, aplicada a mano | molecfit **y** STD_TELLURIC, las dos en código |
| granularidad | una | por exposición **y** sobre el combinado, arbitradas |
| elección | escrita en un campo de texto (`decision.method`) | **medida**, en `arbiter` |
| no-convergencia | motivo para usar STD | **error de la etapa**, con su diagnóstico |
| fases | `check-env`, `decision` | `+ measure`, `+ apply` |
| reanudación | ninguna | bitácora por unidad (`logs/a3_progress.json`) |

`decision` **no se toca**: es el comando bajo el que se emitieron los tres QC congelados y
sigue siendo reproducible. Lo que se lanza hoy (`stage_registry`) es `measure`.

## 1. La regla dura sobre molecfit

Una no-convergencia se clasifica por su **firma en el propio producto** y se responde subiendo
un peldaño de `LADDER` (`musepipe/reduction/molecfit.py`):

| firma | qué la produce | remedio |
|---|---|---|
| `chi2_congelado` | `best_chi2 == initial_chi2` exacto, `status=4` | `WLC_CONST=0` (ya es el defecto) |
| `sin_leverage` | `rel_mol_col_O2` clavado en 1.0000, incertidumbre 0 | meter una banda profunda (O₂ A) |
| `continuo_atascado` | `rms_rel_to_err > 50` (A1a fallido: 85.8) | normalizar el flujo |
| `cabecera_incompleta` | `status=14`, «Unable to find keyword» | completar la cabecera (§4) |
| `residuo_alto` | `rms_rel_to_err > 20` | `WLC_N`, `CONTINUUM_N`, ventanas, agua, núcleo |
| `status_error` | `status <= 0` de MPFIT | entrada mal formada; se sube peldaño |

**Los códigos de MPFIT no son binarios**, y confundirlos cuesta peldaños para nada:
`<= 0` es error, **1–4 es convergencia limpia** (χ², parámetros, ambos, ortogonalidad) y
**5–8 también terminan bien** — 5 es «máximo de iteraciones» y 6–8 son «ya no se puede
mejorar». Medido el 2026-08-22 en ROXs 42B b: `status = 5` con 151 iteraciones y el χ²
cayendo de 130876 a **2876**, o sea 45×. Tratarlo como fallo hacía subir la escalera sobre un
ajuste perfectamente bueno. Se acepta, y se declara aparte como `converge_por_iteraciones`
para que no pase por limpio sin serlo.

Si la escalera se agota, `fit_molecfit` levanta `MolecfitNotConverged` y **la etapa para**.
STD_TELLURIC solo puede aplicarse si **gana la métrica**, o por un `--method` explícito con
`--override-reason`, que queda escrito en `decision.override`.

## 2. La métrica, y por qué no es la que se publicó

Por banda, sobre el espectro de apertura de la primaria ya corregido:

1. `rms` = `rms(flujo/continuo − 1)`, con `local_continuum_linear` (el continuo de la etapa).
2. `scatter` = desviación **sobre la media** del mismo cociente.
3. `floor` = `scatter` en la ventana limpia **7750–7860 Å**.
4. **`fom` = `(rms/floor) / sqrt(<1/T²>)`** — el exceso sobre lo inevitable.

Tres cosas que hay que leer juntas:

- **`rms` y `scatter` no son lo mismo, y la diferencia es el sesgo**: `rms² = scatter² +
  offset²`. La comparación publicada el 2026-08-06 tabula el **scatter** (verificado: esta
  implementación reproduce sus 0.0720 y 0.0304 del combinado al cuarto decimal), y el scatter
  es **ciego a una sobre- o sub-corrección constante**, que es justo el modo de fallo de una
  transmisión con la masa de aire equivocada. **Decide el `rms`**; el `scatter` se publica al
  lado para poder contrastar con aquella tabla.
- **El denominador `sqrt(<1/T²>)` no es cosmético**: dividir por T amplifica el ruido canal a
  canal, y en O₂ A eso solo explica 1.79× de los 3.79× medidos. Sin descontarlo se penaliza a
  la vía que corrige más hondo.
- **`depth_pct` va sin recortar el signo** (`clip_negative=False`) para que una
  sobre-corrección se vea negativa en vez de aplastarse contra 0.

**Selección, en dos pasos** (decisión del usuario 2026-08-22, afinada con lo medido ese mismo
día):

1. **Entre las dos vías de corrección** (`molecfit` en su granularidad ganadora y
   `std_telluric`): gana la `fom` media más baja, y la banda de empate es **1σ de la `fom`
   entre exposiciones del propio run**. Dentro de ella gana **molecfit**. Sin σ medida no se
   declara empate.
2. **Contra no corregir**: la ganadora del paso 1 tiene que ganar **por medida**, sin banda de
   empate. `sin_corregir` no es una tercera vía, es el nulo, y aplicar una corrección que solo
   *empata* con no hacer nada no es lo que la regla autorizaba.

El segundo paso existe porque el primero, aplicado también al nulo, autorizaba daño. Medido en
`ROXs42Bb_realigned`: σ = **1.5035** (la dispersión entre sus 30 exposiciones es enorme), así
que el empate se tragaba un margen real de 0.52 a favor de no corregir — y la regla acababa
aplicando una corrección que convierte una absorción de **+3.86 %** en una joroba en emisión
de **−7.40 %**.

## 3. Los cubos ya vienen corregidos, y eso cambia la pregunta

**Medido el 2026-08-22 sobre `ROXs12b_OB3444577`** (combinado de 7 exposiciones):

| banda | profundidad del cubo tal cual |
|---|---|
| O₂ B | **0.00 %** |
| O₂ A | **5.09 %** |
| H₂O 7200 | **0.00 %** |
| H₂O 8200 | **0.00 %** |

Los cubos por exposición salen de `muse_scipost` **consumiendo `STD_TELLURIC`**
(tarjeta `ESO PRO REC2 CAL6`), así que el DRS ya corrigió. **A3 corrige un residuo**, no una
absorción cruda, y aplicar una segunda corrección completa hunde la banda: STD_TELLURIC
escalada a X = 1.30 deja O₂ A en **−25.6 %** (una joroba en emisión del 25 %) y puntúa
`fom = 2.32` frente a `0.92` de no tocar nada. Por eso `sin_corregir` es un candidato del
árbitro y no una rama de escape.

**Las ventanas de ajuste se eligen por la profundidad medida** (`select_fit_windows`): solo
entra al ajuste la banda que supera el umbral de A3 (3 %). No es un refinamiento, es lo que
se midió:

| ventanas de H₂O en el ajuste | `rel_mol_col_H2O` | H₂O 7200 | H₂O 8200 |
|---|---|---|---|
| congelada a 1.0 | 1.0 (fija) | −6.6 % | −7.4 % |
| libre (`FIT_MOLEC=1,1`) | **5.88** | −16.1 % | −17.4 % |
| **fuera del ajuste (v3)** | — | **sin tocar** | **sin tocar** |

Con el agua libre el ajuste pide **casi seis veces** la columna de vapor. La primaria es una
**M0** y esas dos ventanas caen sobre bandas moleculares **fotosféricas** (TiO, VO): molecfit
las lee como agua. Es exactamente el riesgo que la v2 §4.1 llamaba «punto de máxima atención
de la etapa», ahora con número. Como fuera de las ventanas ajustadas **T ≡ 1**, no ajustar
una banda plana es exactamente dejarla en paz.

### 3.1 Y a veces no hay nada que corregir

**`not_needed_shallow` es un final exitoso**, no un error — la v2 §0 ya lo decía— y **tiene
que dejar QC**: un run sin QC no se distingue de un run que nadie ha corrido. Cuando ninguna
banda llega al umbral, `measure` escribe el QC con el veredicto, las profundidades y
`telluric_applied: false`, y sale con 0. No se ajusta nada, ni molecfit ni STD.

Medido el 2026-08-22, y es un resultado sobre el dato, no sobre el método:

| run | O₂ B | O₂ A | H₂O 7200 | veredicto |
|---|---|---|---|---|
| `ROXs12b_OB3445598` (22 exp, noche del 29) | 0.48 % | **0.00 %** | 0.00 % | **`not_needed_shallow`** |
| `ROXs12b_OB3444577` (7 exp, noche del 31) | 0.00 % | **5.09 %** | 0.00 % | `needed` |
| `ROXs42Bb_realigned` (30 exp) | 2.52 % | **3.86 %** | 0.65 % | `needed` |

Las dos épocas de ROXs 12 b **también difieren en esto**: la noche del 29 no dejó residuo
telúrico medible y la del 31 sí, lo que encaja con todo lo demás que las separa. Y el
**canónico de 29 exposiciones sale a 0.00 % en las cuatro bandas**: apilar las dos noches
borra el residuo que la del 31 tenía sola.

### 3.2 A veces hay residuo y aun así no hay que corregir

`ROXs42Bb_realigned` es el caso duro: O₂ A tiene **+3.86 %**, por encima del umbral, y **las
tres vías lo empeoran**.

| vía | O₂ A tras corregir | `fom` media |
|---|---|---|
| **sin corregir** | **+3.86 %** | **2.1946** |
| molecfit combinado | −12.75 % | 2.7017 |
| molecfit por exposición | −7.40 % | 2.7161 |
| STD_TELLURIC | −27.24 % | 7.9008 |

El veredicto es `sin_corregir` con `reason: no_correction_measures_better`. Que una banda pase
el umbral significa que **hay** algo que corregir, no que exista una corrección que lo mejore:
son dos preguntas distintas y la etapa las contesta por separado.

## 4. La cabecera del combinado, que es lo que mataba a molecfit

`molecfit_model` aborta con `status = 14` **antes de ajustar** si al `science.fits` le faltan
`MJD-OBS`, `UTC`, presión y temperatura ambiente, temperatura de M1 y las tres tarjetas del
emplazamiento. El combine por voxel escribe cabecera propia (`RUNID`, `NEXP`, `EXPTOT`…) y no
propaga ninguna de las ocho. Visto desde fuera, eso parece otra vez «molecfit no converge».

`merge_science_header` las completa **desde las exposiciones que hay dentro del cubo**, con el
mismo criterio que la masa de aire: media ponderada por EXPTIME para el estado de la
atmósfera, suma para el tiempo de integración, copia para las constantes del sitio. `UTC` y
`DATE-OBS` se derivan del MJD ya promediado, porque promediar segundos-desde-medianoche se
rompe al cruzar las 00:00 — y estas observaciones la cruzan (2022-08-31 → 2022-09-01).

**La masa de aire va aparte y es efectiva**: `effective_airmass` la pondera por EXPTIME sobre
las exposiciones (X = 1.3018 para este OB) en vez de heredar la de la primera exposición
(1.14 para luz que llega hasta 1.54), que es el sesgo estructural medido en las indicaciones
de 2026-08-06 §5.2b.

## 5. Aplicación, y la trampa del STAT

- **Sobre el combinado** (`apply`): `apply_transmission_streaming` recorre λ por trozos,
  `DATA/T` y `STAT/T²`, anotando cada trozo en la bitácora. Los datos se leen **siempre del
  cubo original**, nunca de la salida a medio corregir: leer de la salida al reanudar
  dividiría dos veces los trozos ya hechos.
- **Por exposición**: la T viaja en `StreamCombinePlan.transmission_by_exposure`, aplicada
  dentro del bucle del combine. **No** por el hook `transform`, que devuelve solo DATA y
  deliberadamente no toca STAT — correcto para restar un modelo determinista, falso para
  dividir por T. La clave del diccionario es el **fichero** de la exposición y no
  `exposure_id`, porque ese ID sale del **directorio padre** y los cubos por exposición se
  llaman todos `DATACUBE_FINAL.fits`: dos que compartan carpeta comparten ID, y la T acabaría
  en la exposición equivocada sin que nada fallara.
- Ventanas protegidas intactas: `T ≡ 1` en 6540–6590 Å (Hα) y 5780–6050 Å (láser AO), forzado
  en `enforce_protected_transmission` y verificado en `v3_halpha_maxabs`.

## 6. Reanudación y tiempos

`runs/<RUN>/logs/a3_progress.json`, con la disciplina de `scripts/rerun_chain.py`. Una unidad
se salta **solo si** las cuatro: la bitácora la da con `rc=0`, el producto existe, su mtime no
ha retrocedido, y el **hash de los parámetros coincide**. El hash es lo que distingue «ya está
hecho» de «está hecho con otra receta». Un fichero `PARAR` junto a la bitácora detiene entre
unidades.

Unidades: un ajuste de molecfit (`measure`), un trozo de λ (`apply`, combine). El QC publica
`timing` con la forma del `runtime_budget` de E4.

**Coste medido (2026-08-22, `ROXs12b_OB3444577`)**: 70–120 s por ajuste de molecfit sobre este
hardware, con una sola ventana. La cifra de 35–75 s de las indicaciones de 2026-08-06
corresponde a dos ventanas sobre otra reducción.

## 7. Esquema del QC (lo que añade la v3)

```json
{
  "spec_version": "A3_v3",
  "decision": {"depth_pct_by_band": {}, "telluric_applied": true, "applied_to_cube": false,
               "method": "molecfit|std_telluric|sin_corregir",
               "granularity": "molecfit_combined|molecfit_perexp",
               "override": {"method": "...", "reason": "...", "arbiter_winner": "..."}},
  "fit": {"regions_A": [], "regions_bands": [], "regions_excluded_shallow": [],
          "airmass_effective": 1.3018, "airmass_source": "media ponderada por EXPTIME de N",
          "molecfit": {"combined": {"rungs_tried": [], "fit": {}}}, "std_telluric": {}},
  "arbiter": {"fom_by_method": {}, "scores": {}, "tie_sigma": null,
              "granularity": {}, "method": {"winner": "", "reason": "", "margin": 0.0}},
  "timing": {"n_unidades_hechas": 0, "segundos_por_unidad": 0.0, "por_unidad": {}}
}
```

## 7.1 Estado en disco (2026-08-22)

`measure` corrió en los cuatro runs y **`apply` no se corrió en ninguno** (decisión del
usuario): los QC de medida son el producto, `applied_to_cube` es `false` en los cuatro y no se
ha escrito ni un `cube_telcorr.fits` ni una `TELLURIC_TRANS.fits`. **La corrección telúrica
que llevan los datos sigue siendo la del DRS** (`muse_scipost` consumiendo `STD_TELLURIC`), y
para `ROXs12b_realigned` además la de A3 de 2026-07-08 descrita en
`runs/ROXs12b_raw/stages/stage00t_realigned_qc.json`.

> **Aviso de procedencia.** Al escribir los QC de medida se **sobrescribieron** dos QC de A3
> anteriores: `runs/ROXs12b_realigned/stages/stage00t_qc.json` y
> `runs/ROXs42Bb_realigned/stages/stage00t_qc.json` (`runs/` está gitignorado, no hay copia).
> Lo que se perdió es el registro de la **decisión** de julio, no de ninguna aplicación: el de
> ROXs 42B b ya tenía `applied_to_cube: false` —ese objeto nunca tuvo la corrección aplicada al
> cubo— y sus `depth_pct_by_band` los reproduce el QC nuevo cifra a cifra (2.52 / 3.86 / 0.65 /
> 0.00). Para ROXs 12 b el registro de lo aplicado sobrevive en
> `stage00t_realigned_qc.json`, que no se tocó.

## 8. Lo que queda abierto

1. **`molecfit_calctrans` sigue bloqueado** en esta build (rechaza `MAPPING_ATMOSPHERIC` para
   una ciencia de una sola BINTABLE). Por eso T solo está definida dentro de las ventanas
   ajustadas. Con la selección por profundidad de §3 esto deja de ser una limitación práctica
   —lo que no se ajusta es justo lo que no hay que corregir— pero sigue siendo la pieza que
   habría que desbloquear para una transmisión de rango completo.
2. **`a1a_molecfit_crosscheck` vive en el QC del run REALINEADO**
   (`runs/ROXs12b_realigned/stages/stage00r_qc.json`), no en el del crudo. La justificación
   §6 y el README de A1a lo citan sin decir de qué run, y buscarlo en `ROXs12b_raw` no lo
   encuentra. No es una cita rota, es una cita incompleta — y una cadena que no tenga ese A1
   (ROXs 42B b) sigue sin traerlo, que es por lo que el notebook lo lee dentro de un `try`.
3. **El `cube_telcorr_qc.json` que escribe el combine no es un producto de A3** — es el nombre
   por defecto de la salida de `run_stream_combine.py`, y el combine no aplica ninguna
   transmisión. El nombre se deja como está (es el alias del QC de A1-cascade en
   `stage_registry` y en `_nbcommon`, y renombrarlo rompería los runs en disco), pero conviene
   saberlo al leerlo.
