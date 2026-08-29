# Especificación E4 v4 — la SNR se estandariza contra la nula de su propia población

Fecha: 2026-08-28. Revisión formal de
`docs/spec_E4_v3_codex_injection_recovery.md`, que sigue siendo la referencia
para todo lo demás: v4 cambia **solo la escala en la que se decide `complete`**,
y separa una perilla que hacía dos cosas. El QC declara `spec_version="E4_v4"`
**únicamente** cuando la estandarización está encendida; con
`h04_snr_standardization="none"` sigue declarando `E4_v3` y el producto es el de
siempre.

## 0. Motivo

`h04_detection_threshold_snr` es un solo 5σ para los seis métodos. Medido el
2026-08-28 sobre productos ya en disco
(`docs/2026-08-28_umbral_por_metodo_medido.md`): con señal **nula** en posiciones
de control, la SNR que E4 usa para decidir no tiene media 0 ni sigma 1 en ningún
método, y falla de **dos formas distintas**:

- **pedestal** en el cubo combinado —la media se desplaza y la sigma se queda en
  ~1—: `psffit` +5.55 en ROXs 12 b, `optimal_ls` −3.66 y −2.25 en los dos
  objetos;
- **cola** por exposición —la media es 0 y la sigma se infla—: `psffit` ×4.86 y
  ×4.43, `aperture` ×1.82 y ×1.98.

Un umbral único no puede corregir ninguna de las dos, y el 5σ actual falla **en
las dos direcciones a la vez**: deja pasar el 60 % de falsos positivos de
`psffit` en el combinado de ROXs 12 b, y le cuesta a `optimal_ls` de ROXs 42B b
la completitud entera (0 % contra 100 % a SNR inyectada 2) porque su cero no está
donde el umbral supone.

El defecto no está en el umbral: está en la **escala** sobre la que se aplica.

## 1. Qué cambia

Solo esto:

- Cada fila lleva `recovered_snr_std`, la SNR estandarizada contra la nula
  empírica de su estrato. **Se emite siempre**, también con
  `h04_snr_standardization="none"`: es el diagnóstico que dice si la SNR formal
  está en escala, y medirlo no puede depender de haber decidido ya que no lo
  está.
- Con `h04_snr_standardization="injection_nulls"`, `complete` se decide sobre
  `recovered_snr_std` en vez de sobre `recovered_snr`. El umbral no cambia:
  sigue siendo `h04_detection_threshold_snr`, y con la escala arreglada vuelve a
  significar «sigmas de la distribución que este método tiene sin señal». Su
  **valor** es una decisión aparte, y desde el 2026-08-28 es **3.5**: cambiarlo no
  pide re-inyectar, porque `complete` es derivada — ver §4 y `--finalize`.
- `h04_completeness_input_snr` deja de ser el umbral. Hasta v3 `_completeness`
  usaba `h04_detection_threshold_snr` para **dos cosas**: el corte de detección
  —sobre la SNR recuperada— y el nivel de SNR **inyectada** al que se reporta.
  Son dos conceptos; el valor por defecto (5.0) conserva el número que producía
  el acoplamiento, así que nada se mueve mientras no se declare.

No cambian: la rejilla de S/N, anchos, modos de continuo, posiciones, métodos,
extractores, la definición de `throughput` —que **no** pasa por el umbral y por
tanto no mueve E3 ni el límite de Ṁ—, la regla de agregación de V2 ni sus dos
umbrales congelados.

## 2. La regla, entera

Para cada fila se toma su **estrato**: método × ancho de plantilla × modo de
continuo. Cruzar estratos sería comparar escalas distintas, porque el ancho
cambia la escala del filtro adaptado y el modo de continuo cambia lo que se resta
antes.

La **referencia** del estrato son sus inyecciones nulas: filas nominales con
`input_snr = 0` en posiciones de **control**. La posición real queda fuera —puede
llevar señal del compañero, y meterla en la referencia es poner la detección en
el cero.

    recovered_snr_std = (recovered_flux − mediana(referencia)) / sigma_robusta(referencia)

con `sigma_robusta = 1.4826 · MAD` (`musepipe.stats.empirical_z`, la misma pareja
que el resto de la cadena). Se estandariza `recovered_flux`, no
`recovered_flux_net`, porque es la cantidad que la referencia mide.

**Una fila nula se estandariza dejándose fuera** (leave-one-out). Si no, cada
nula entra en su propio centro y su propia escala: con 15 controles eso encoge su
z y cualquier tasa de falsos positivos medida con ellas sale optimista por
construcción.

**Un estrato sin al menos dos nulas de control para la etapa.** No hay escala, y
heredar la SNR formal sería volver en silencio justo a la escala que se acaba de
declarar mala. El mensaje dice qué estratos faltan y que la rejilla tiene que
incluir `0.0` para cada ancho y modo de continuo.

## 3. Por qué NO la referencia de V2

E4 ya construye una nula empírica —`build_empirical_null_reference`, 33 controles
de **producción** repartidos por el campo— y alimenta con ella la puerta V2. Era
la candidata obvia y **está medido que no sirve para esto**: las inyecciones
nulas están al radio del compañero y pasan por la maquinaria de inyección, así
que son otra población. En ROXs 12 b, `aperture`: **175.0 ± 20.4** en producción
contra **60.3 ± 162.8** en inyección, un factor **8** en la escala.

Estandarizar contra la de producción no arregla la escala, la empeora: la nula de
`aperture` pasa de 0.34 ± 1.20 a **−6.55 ± 6.98**, y `sgf` y `lpm` pasan de 0 % a
33–40 % de falsos positivos al 5σ. Contra la propia población, los **doce** casos
(seis métodos × dos objetos) caen en media −0.44…+0.31 y sigma 0.95…2.21.

**Esto también afecta a V2, que no cambia en v4 pero queda señalado.** V2 compara
las filas nulas de inyección contra la distribución de producción: dos
poblaciones. Se ve en el QC actual — **33 de 90** filas en ROXs 12 b y **47 de
90** en ROXs 42B b están en el suelo de la cola baja (1/34), es decir por debajo
de los 33 controles de producción; en `aperture` y `optimal_ls` de ROXs 42B b son
**15 de 15**. Una puerta unilateral cuyas filas viven enteras en la otra cola no
puede disparar. Por eso v4 publica
`snr_standardization.reference_population_check`, que compara las dos escalas
método a método. **Informa, no vota.**

## 4. QC

Bloque nuevo `snr_standardization`:

- `mode`, `column` (cuál decidió `complete`), `threshold_snr`, `reference`,
  `leave_one_out`;
- `scale_by_stratum`: centro, sigma y `n` de cada estrato;
- `reference_population_check`: por método, la escala de inyección y la de
  producción, el desplazamiento del centro en sigmas de producción y el cociente
  de sigmas;
- `null_distribution`: media y sigma de la SNR nula, **cruda y estandarizada**.
  Es la verificación de §5 publicada en el propio producto.

Y, fuera del bloque, `completeness_input_snr` y `completeness_threshold_snr`
declarados por separado, más `derived_finalize` cuando lo derivado se ha
recomputado después de la corrida: cuándo, con qué umbral, sobre qué columna y
cuántas filas cambiaron. Un producto que ya no sale de una sola corrida tiene que
decirlo. `completeness_at_5sigma` conserva el nombre —lo leen
otros— aunque el `5sigma` fuera el umbral y no el nivel.

## 5. Verificaciones

Corridas el 2026-08-28 con la rejilla de calibración (54 controles en ROXs 12 b,
35 en ROXs 42B b; ver §7):

- Con `injection_nulls`, `null_distribution.standardized` tiene que dar media ≈ 0
  y sigma ≈ 1 en todos los métodos. **Medido: media −0.19…+0.08 y sigma
  0.77…1.14 en ROXs 12 b; media −0.42…+0.14 y sigma 0.94…1.66 en ROXs 42B b.**
  El pedestal de `psffit` pasa de +5.16 ± 1.58 a **+0.04 ± 0.81**, y el de
  `optimal_ls` de −3.70 a −0.19. El caso peor es `optimal_ls` en ROXs 42B b
  (sigma 1.66), que es también el método con el sesgo de flujo más grande.
- **0 falsos positivos de 54 y de 35** en los seis métodos de los dos objetos, al
  5σ con el que se corrió y al **3.5σ** que se declaró después
  (`docs/2026-08-28_umbral_bajado_a_3p5.md`): el máximo de las 534 nulas es 2.84.
  Con la SNR formal, `psffit` daba **55.6 %** en ROXs 12 b.
  **La cota de FPR que se puede citar depende del método**, porque el `n`
  independiente lo fija el radio de cada extractor y no el de la rejilla (§7):
  < 1.9 % / 2.9 % para `aperture`, `sgf` y `lpm`; < 3.8 % / 5.9 % para
  `optimal_*`; < 5.9 % / 9.1 % para `psffit`.
- `completeness_at_5sigma` no se mueve: a SNR inyectada 5 los seis métodos están
  al 100 % con las dos escalas, en los dos objetos. **El `throughput` a SNR 5
  tampoco**: cambia entre −0.3 % y +0.1 % en once de las doce parejas
  (`optimal_ls` de ROXs 42B b, +1.1 %), o sea que E3 y el límite de Ṁ se quedan
  donde estaban.
- Con `none`, el producto es idéntico al de v3 salvo la columna nueva.

## 6. Tests mínimos

`tests/test_h04_snr_standardization.py`: el estrato separa método/ancho/continuo;
la posición real no entra en la referencia; el leave-one-out da el z calculado a
mano y es mayor que sin él; un pedestal deja de contar como detección y una señal
de verdad sigue contando; `none` no revisa la decisión pero publica la columna;
un estrato sin nulas para la etapa; la columna viaja en `TABLE_FIELDS`.

## 7. Protocolo de parada

- **El radio del compañero pone un techo al número de controles.** Van todos a
  ese radio repartidos en ángulo, así que la separación entre vecinos es
  `2πr/(n+1)`. Con `n = 100` cae a 4.4 px en ROXs 12 b y **2.9 px** en
  ROXs 42B b —por debajo de la apertura `box3` y de la FWHM—, y en ROXs 42B b un
  control aterrizaría dentro de la PSF del propio compañero. El 2026-08-28 se fijó
  el máximo que mantiene **≥ 8 px** (≈ 2 FWHM, la escala de correlación que
  implica `docs/noise_model.md`): **54 en ROXs 12 b y 35 en ROXs 42B b**.
  Comprobado a posteriori sobre las nulas, la correlación a desfase 1 (8.13 px)
  va de −0.29 a +0.32 en los doce casos. **Eso NO demuestra independencia**: con
  n = 54/35 el test sólo excluye |r| ≳ 0.3.
- **El `n` independiente lo fija el radio de cada extractor, no el de la rejilla.**
  Corregido el 2026-08-28 (`docs/2026-08-28_fig2_con_escala_v4.md` §5): el criterio
  de ≥ 8 px vale para `aperture` (`box3`, necesita 3 px) y para `sgf`/`lpm`
  (exclusión de 3 px, necesitan 6), pero **se queda corto** para `optimal_*`
  (`x02_window_radius_px` = 8 → necesitan 16 px) y para `psffit`
  (`x03_comp_radius_px` = 12 → necesita 24 px). Los máximos por método son 147/96,
  73/47, 26/17 y 17/11. **La escala no se resiente** —mediana y sigma robusta de
  muestras correlacionadas pero idénticamente distribuidas siguen siendo válidas,
  sólo pierden precisión, y la prueba es que la nula estandarizada sale ~(0,1)—;
  lo que hay que rebajar es la **cota de falsos positivos**, que se cita con el `n`
  del método (§5). El pre-registro de la Figura 2 se quedó en 15 controles porque
  es casi el máximo que `psffit` admite, no por prudencia.
- Bajar de ahí exige controles a **otros radios**, y eso reintroduce el desajuste
  de poblaciones de la §3.
- La estandarización arregla la **escala de detección**. **No** arregla el sesgo
  de flujo que la causa —el pedestal de `psffit` es el halo, y sigue ahí en
  `throughput` y en `bias_flux_pct`—. Quien cite completitud con v4 y sesgo de
  flujo en la misma frase tiene que decir que son cosas distintas.
- La referencia se calibra sobre las mismas posiciones en las que después se mide
  la completitud de las filas con señal. Las filas con señal no entran en la
  referencia, pero comparten posición con las nulas; una afirmación sobre la
  **tasa** de falsos positivos con esta muestra es circular y no debe hacerse sin
  validación cruzada por posiciones.
