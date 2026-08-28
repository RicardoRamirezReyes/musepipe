# Spec C7 — extracción por exposición y combinación de las medidas

**Estado:** contrato de la etapa `C7` (`musepipe/stages/stage_x06_perexp.py`). **Sin implementar**:
este documento es el diseño aprobado el 2026-08-26, escrito antes del código como pide la
convención del repo.

Complementa —no sustituye— a C2–C6. **No entra en `METHOD_ORDER`**: un método nuevo ahí se vuelve
obligatorio para todos los runs al instante (`stage_x10_compare.py:1660-1668` lanza
`FileNotFoundError` sin su producto) y arrastra D1, D2, E1, E3, E4, G1 y F1.

## 1 · Por qué

La cadena combina los **cubos** y extrae una vez. Eso obliga a describir con un solo modelo de PSF
la mezcla de 29 exposiciones, y el error de esa mezcla está medido: su corrección de apertura
—`F(≤25)/F(box3)`, que es la V4 de la spec C1— falla la tolerancia del 3 % con **+6.77 %**, y el
fallo es cromático: +0.34 % en el azul, **+8.55 % en el rojo**
(`docs/2026-08-25_...`, `docs/2026-08-26_perexp_medido_y_la_noche_mala.md`).

Dos vías se cerraron **con medida** antes de llegar aquí: el término híbrido de C1 (arregla el
anillo y lleva la V4 de +6.04 % a **+35.08 %**) y cambiar la ley de combinación de la mezcla
(media +4.28 %, `sigclip` +4.72 %, mediana por píxel −12.38 %).

Lo que queda, y lo que esta etapa hace, es **extraer en cada exposición con su propia PSF y
combinar las medidas**. Medido el 2026-08-26:

- **S/N: empata** con el cubo combinado (0.97–1.09× con pesos independientes), no lo mejora.
- **La apcorr por exposición quita la deriva cromática**: `apcorr_perexp/apcorr_mezcla` deriva
  **−8.52 %** del azul al rojo, contra el **+8.2 %** que deriva la V4 de la mezcla.
- **Lo que de verdad se gana es poder pesar por calidad.** El combinado pesa por `exptime` y con
  eso le da **el 38.1 %** del peso a la noche del 2022-08-31, que tiene σ de controles **4.2×
  peor** y menos de la mitad de señal. La inversa de la varianza le da **2.6 %**.

## 2 · Entradas

| entrada | de dónde | qué aporta |
|---|---|---|
| `stages/psf_model_mixture.json` (`form="mixture"`) | C1 con `psf_scope=per_observation` | un documento de PSF **por exposición**, con su peso |
| `stages/observation_plan.json` | C1 | qué exposiciones, con qué ventana, desplazamiento y peso |
| `stream_combine_plan.json` | A1, vía `musepipe.observations` | la geometría del combinado |
| `stages/stage01c_qc.json` | B3 | posición del compañero y de la fuente de campo |
| `stages/stage01_qc.json` | B1 | la ventana de recorte, para entregar en el marco de la cadena |
| los cubos por exposición | `perexp_cubes` / `perexp_dir` o el propio plan | el dato |

Si lo que encuentra no es una mezcla, la etapa **falla**: extraer del combinado ya lo hacen C2–C6.

## 3 · Procedimiento

1. `observations.resolve_observation_plan(..., plan_json=...)` — la vista cacheada, que valida que
   los cubos siguen ahí y re-mide el centroide de los sustituidos.
2. Por exposición, con `stage_e01_perobs.load_aligned_exposure` (**la misma ventana, el mismo
   desplazamiento subpíxel y el mismo recorte que produjeron el combinado**):
   * fondo de anillo (`extraction/aperture.annulus_background_spectrum`);
   * apertura (`aperture_spectrum`) y/o ajuste de PSF (`extraction/psffit.fit_psffit_cube`);
   * **corrección de apertura con el modelo de ESA exposición**
     (`aperture_correction_from_psf`) — es justo lo que la PSF por observación cambia;
   * las **mismas** posiciones de control (`apertures.same_radius_control_positions`), procesadas
     idénticas al objeto.
3. **Agrupación** (`x06_group_by`): `none` (un solo producto con las N) o `night` (un producto por
   noche, por la MJD del plan). La agrupación por noche no es cosmética: las dos noches de
   ROXs 12 b difieren en `apcorr` **4.84 contra 44.01**, y mezclarlas es lo que esta etapa existe
   para poder no hacer. **Medido el 2026-08-26** (S/N, `box3`, pesos `invvar` de banda
   independiente):

   | grupo | continuo | Hα | rojo |
   |---|---|---|---|
   | las 29 juntas | 19.33 | 8.35 | 388.75 |
   | **solo 2022-08-29 (22)** | **20.60** | **9.24** | **401.36** |
   | solo 2022-08-31 (7) | −1.87 | −4.12 | 36.83 |

   Es decir: **incluso pesando por calidad, añadir la noche mala empeora las tres bandas**. La
   inversa de la varianza le deja un 2.6 % del peso y aun así resta. Por eso `night` no es una
   comodidad: es la forma de no pagar por una noche que no aporta.
4. **Combinación** (`x06_combine`), sobre las medidas, no sobre los cubos.
5. Marco de B1 (`stage_e01b_perobs_subtract.b1_window`) y factor de flujo total de A2
   (`growth_curve.resolve_flux_convention`) aplicados **una sola vez, al final**: son comunes a
   las N y aplicarlos por exposición sería contarlos N veces.

### 3.1 · Las leyes de combinación, y lo que cada una vale

| `x06_combine` | qué hace | medido |
|---|---|---|
| `invvar` (por defecto) | media pesada por `1/σᵢ²`, con σᵢ de los controles de esa exposición | S/N 19.33 en el continuo; `n_eff` 16 de 29 |
| `exptime` | los pesos del plan | **0.11** en el continuo — la peor, y es la del cubo |
| `equal` | media sin pesos | 2.26 |
| `sum` | **suma** de las medidas | equivalente a `equal` en S/N salvo un factor de escala global (ver abajo) |

**`sum` se declara porque se ha pedido para pruebas posteriores**, y con su consecuencia
**medida**: combinando los controles igual que el objeto, `sum` y `equal` dan la **misma S/N al
decimal** (2.26 / −1.45 / 129.87 en continuo, Hα y rojo), y difieren solo en el factor de escala N.
Es decir, la suma es útil para conservar flujo total y comparar contra un apilado, pero **no da
mejor límite que la media sin pesos**, y las dos quedan muy por debajo de `invvar` (19.33 en el
continuo). Nunca debe ser el valor por defecto.

> Cuidado al medirlo: comparar `sum` con σ propagada (`√Σσᵢ²`) contra `equal` con σ de controles
> combinados da 3.55 contra 2.26 y hace creer que la suma gana. Es la trampa nº 1 del
> `docs/2026-08-26_perexp_medido_y_la_noche_mala.md`: dos estimadores distintos no se comparan.

**Los pesos de `invvar` se calculan en una banda declarada (`x06_weight_band_A`), no en la banda
que se está midiendo.** Medido: pesar con la σ de la propia banda infla la S/N un 40–60 % por
auto-selección (27.94 contra 19.33 en el continuo).

### 3.2 · La σ del producto

De **controles combinados con los mismos pesos**, nunca de `STAT` (que subestima ~4× el ruido de
apertura, `docs/noise_model.md`) ni de `σᵢ/√n` (que trata como ruido la estructura del halo, que es
la misma en las N y **no promedia** — `docs/2026-08-21_heterogeneidad_y_limite.md`).

## 4 · Productos

| producto | contenido |
|---|---|
| `stages/spec_perexp_<grupo>_object.fits` | `SpectrumProduct`; `<grupo>` es `all` o la noche (`20220829`) |
| `stages/spec_perexp_<grupo>_controls.npz` | los controles combinados, para poder re-derivar σ |
| `stages/spec_perexp_qc.json` | la QC |

`METHOD` **sigue siendo el del extractor** (`aperture` o `psffit`): `SpectrumProduct.validate`
tiene lista blanca (`extraction/product.py:106`) y un nombre nuevo la rompería. Lo que distingue a
este producto va en cabeceras libres, que el formato ya admite (`product.py:71-80`):
`NEXP`, `COMBMODE`, `GROUP`, `WBAND`, y `EXPIDS` en una extensión aparte si no cabe.

## 5 · Verificación

- **V1 — Ancla contra el diagnóstico**: con una sola exposición y sin agrupar, el producto tiene
  que reproducir su fila de `perobs_spectra.npz` al bit. `tests/test_x06_perexp.py`.
- **V2 — La σ es la de los controles combinados**, y el QC publica las tres leyes para poder
  compararlas. Falla si `STAT` acaba usado como σ.
- **V3 — Pesos de banda independiente**: el QC declara `x06_weight_band_A` y falla si coincide con
  una banda de medida.
- **V4 — Conservación en `sum`**: la suma de las N medidas tiene que dar, dentro del error, N veces
  la media sin pesos. Es el test que hace honesto declarar `sum`.
- **V5 — Agrupación**: con `group_by=night` la unión de los grupos tiene que contener exactamente
  las exposiciones del plan, sin repetir ni perder.
- **Guardias que deben fallar, con `assertRaises`**: `exposure_id` repetido (no es único por
  construcción, `reduction/stream_combine.py:110-117`); una exposición del plan sin modelo en la
  mezcla; un documento que no sea `form="mixture"`; `x06_combine` desconocido.

## 6 · Qué NO decide esta etapa

- **No cambia dónde se combina la cadena.** `docs/2026-08-15_psf_por_observacion.md` congela
  «combinar cubos», y los seis métodos de C2–C6 siguen igual. Esta etapa **mide la alternativa**.
- **No calibra**: al no entrar en `METHOD_ORDER`, D2 no la toca y no entra en el bloque E. Nace
  como producto de **referencia y validación**, no como sustituto.
- **No decide qué hacer con la noche mala.** Publica el peso que cada ley le da (38.1 % con
  `exptime`, 2.6 % con `invvar`) y deja la decisión escrita en el QC, no tomada.
- **No rechaza exposiciones por umbral.** El rechazo se midió y no es obvio: quitar las siete de
  la noche mala empeoró el test de forma. Si algún día se rechaza, será con una medida detrás.
