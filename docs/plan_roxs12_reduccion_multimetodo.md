# Plan de implementación: reducción y análisis multi-método de ROXs 12 con MUSE

Fecha: 2026-07-01. Estado: propuesta para revisión, sin código escrito.

Decisiones de alcance acordadas:

1. El flujo empieza desde los datos crudos de ESO (pipeline oficial + ZAP +
   molecfit), pero con puntos de entrada configurables para poder empezar
   también desde un cubo ya reducido o desde un cubo recortado, como hoy.
2. Jerarquía de métodos según el documento de investigación: apertura como
   control, extracción óptima tipo Horne y PSF-fitting cromático
   estrella+compañero como métodos primarios; HRSDI/PCA con forward modeling
   como segunda línea; ANDROMEDA y deconvolución como validación/opcional.
3. La cadena local-surface existente (04b→07→07b→08→08c) se conserva como una
   rama más de comparación, no se descarta.

Objetivo científico global: obtener un espectro fiel de ROXs 12 B (continuo y
líneas) con incertidumbres calibradas, y decidir con significancia cuantificada
si existe emisión Hα detectable; si no, publicar un límite superior defendible.

---

## 1. Evaluación del pipeline actual

### 1.1 Qué hay y qué está validado

- Paquete `musepipe/` con módulos puros (`config`, `paths`, `io`, `stats`,
  `spectral`, `apertures`, `localfit`, `stripes`) y siete etapas migradas
  (`stage04b`, `stage06_local`, `stage06_sweep`, `stage06_pca_c2_summary`,
  `stage07`, `stage07b`, `stage08c`) más `stage08_full_spectrum_for_modeling.py`.
- Suite de 52 pruebas, entorno fijado en `environment.yml`, runs organizados en
  `runs/<RUN_ID>/{config,stages,tables,plots,logs}` con QC JSON por etapa.
- Validaciones ya ejecutadas: C1 (inyección nominal S/N 5 recuperada a −1.9%
  del valor esperado), C2 (grilla 3 líneas × 3 posiciones × 9 niveles de S/N
  con throughput y completitud), C4 (calibración look-elsewhere con 31
  controles al mismo radio: máximo del objeto FAP global 90.6%, Hα FAP 100%).
- Resultado científico actual: con la cadena local-surface, **no hay evidencia
  de Hα en ROXs 12 B**; el plan debe tratar esa no-detección como hipótesis a
  confirmar o refutar con métodos independientes, no como conclusión cerrada.

### 1.2 Supuestos implícitos de la cadena actual

1. El cubo de entrada (producto de `01_load_align_crop`) está bien calibrado en
   longitud de onda y flujo; nunca se re-verifica contra skylines ni contra
   fotometría externa.
2. El fondo bajo el compañero es una superficie local suave (`plane`, radio de
   ajuste 12 px, máscara 3 px); no se modela el halo cromático de la primaria
   como PSF.
3. El ruido es puramente empírico (`robust_sigma` sobre continuo y controles al
   mismo radio); la extensión `STAT` del cubo MUSE no se usa en ninguna etapa
   (problema #7 de `plan_mejora_y_sugerencias.md`).
4. La covarianza espectral inducida por el procesado se ignora; los canales se
   tratan como independientes.
5. Coordenadas del objeto fijas en config/QC: `(y, x) = (152, 72)` en crop
   170×170, estrella en ~`(85, 85)`; no hay re-localización automática ni
   astrometría relativa contra la órbita conocida (Bowler et al. 2017).
6. Región 5780–6050 Å enmascarada (láser AO); se asume suficiente y estática.

### 1.3 Limitaciones principales

- **Un solo método de extracción real**: apertura sobre residual local-surface.
  Sin PSF-fitting ni extracción óptima, un sesgo del modelo de fondo no es
  detectable internamente (el control al mismo radio detecta falsos positivos,
  no pérdida de flujo ni sesgo cromático).
- **Sin varianza propagada**: imposible dar barras de error calibradas al
  espectro final para modelado atmosférico.
- **Sin verificación de calibración**: longitud de onda, LSF y flujo absoluto
  se heredan sin control, y la significancia de un centroide de Hα depende de
  ello.
- **PCA (Stage 04) con problema de memoria** (float64, ~25 GB por solución en
  crop 170) — relevante si se activa la rama HRSDI/PCA.
- Deuda técnica conocida: `RUN_ID` hardcodeado en notebooks no migrados,
  duplicación en `04c`/`00_config`, sin logging formal (O3, O5, O6).

### 1.4 Componentes reutilizables vs a reescribir/generalizar

| Componente | Veredicto | Nota |
|---|---|---|
| `musepipe/config.py`, `paths.py`, `io.py` | Reutilizar | Extender config con `entry_point` y bloque `extraction` |
| `musepipe/stats.py`, `spectral.py`, `apertures.py` | Reutilizar | Añadir soporte de varianza (`STAT`) en las funciones de espectro |
| `musepipe/localfit.py` + `stage04b` | Reutilizar tal cual | Se convierte en el método "A: local-surface" de la comparación |
| `stage06_local_surface_injection` + sweep | Generalizar | La inyección debe poder propagarse por *cualquier* extractor, no solo 04b |
| `stage07`, `stage07b`, `stage08`, `stage08c` | Reutilizar con interfaz nueva | Hoy asumen el residual de 04b; deben aceptar un espectro 1D + varianza de cualquier método |
| `stripes.py` + etapas 02/02b/02c | Reutilizar | QC instrumental previo a toda extracción |
| Diagnósticos Moffat `04c`/`04d` | Reescribir como módulo | Son la semilla del modelo de PSF cromática; hoy viven duplicados en notebooks |
| `00_config.ipynb` | Reescribir | Migrar a `musepipe/config` + un YAML/JSON por run; el notebook queda como editor/visor |
| Etapas nuevas (reducción raw, ZAP, molecfit, PSF, Horne, PSF-fitting, comparación) | Escribir desde cero | Ver bloques A, C y D |

---

## 2. Arquitectura general propuesta

### 2.0 Estado operativo (actualizado 2026-07-01)

Ejecución en curso con `entry_point = eso_cube`: el cubo de trabajo es el ADP
histórico (`ADP.2022-09-12T17_17_39.371.fits`), sin descarga de raws ni
pipeline ESO instalado. A2 y A3 corren con `provenance="adp"` (specs ya
actualizadas). A1 queda diferida como fase posterior; cuando corra, A4 debe
re-ejecutarse en modo comparativo (re-reducido vs ADP) y las verificaciones
V4/V5 de A1 cierran el lazo. Consecuencia operativa: sin pixtables, ZAP (A2)
es la única herramienta disponible para residuales de cielo.

### 2.1 Puntos de entrada configurables

Nuevo campo en `config.json`:

```json
"entry_point": "raw" | "eso_cube" | "cropped_cube"
```

- `raw`: ejecuta bloque A completo (esorex/Reflex → ZAP → molecfit → QC).
- `eso_cube`: parte de `DATACUBE_FINAL.fits` ya reducido; ejecuta solo QC del
  cubo (A4) y sigue en bloque B.
- `cropped_cube`: comportamiento actual (compatibilidad con runs existentes).

Cada etapa declara sus productos de entrada por nombre lógico, no por ruta, de
modo que la misma cadena B–F funciona con cualquier punto de entrada.

### 2.2 Contrato de etapa (obligatorio para todo lo nuevo)

Cada etapa nueva sigue el patrón ya validado en el refactor:

- módulo `musepipe/stages/stageXX_nombre.py` con `run_stageXX(run_id=None, ...)`;
- productos con nombres estables en `runs/<RUN_ID>/stages/`;
- `stageXX_qc.json` con parámetros efectivos, hashes de entradas, versiones y
  métricas de éxito;
- figuras de verificación en `runs/<RUN_ID>/plots/stageXX_*.png`;
- notebook fino de inspección (carga config → ejecuta función → muestra figuras);
- tests: uno sintético (dato fabricado con verdad conocida) y uno de contrato
  (nombres/formas/claves de QC).

### 2.3 Formato estándar de espectro extraído

Todos los extractores (A, B, C, D) escriben el mismo producto:

```text
spec_<metodo>_<objeto>.fits  (BinTable: wave_A, flux, flux_err, npix_eff, flags)
spec_<metodo>_<objeto>_cov.npz (opcional: covarianza espectral, solo PCA/HRSDI)
spec_<metodo>_qc.json
```

Esto es lo que hace comparables los métodos y lo que consume el bloque E de Hα.

---

## 3. Etapas

Convención de cada ficha: **O** objetivo, **I** inputs, **P** operaciones,
**S** salidas, **V** verificación (plots y estadísticas), **C** criterio de
éxito, **R** riesgos/supuestos/fallos, **K** organización del código.
Etiqueta `[ESENCIAL]` u `[OPCIONAL]` al inicio.

### Bloque A — Reducción desde datos crudos (nuevo)

#### A1 · `00r_raw_reduction` — Reducción ESO [ESENCIAL para entry_point=raw]

- **O**: producir `DATACUBE_FINAL.fits` con extensiones DATA y STAT trazables,
  a partir de raw + calibraciones (bias, flat, arc, geometría, estándar).
- **I**: raws de ESO archive (ciencia + calibraciones asociadas + estándar
  espectrofotométrica), esorex/Reflex instalados (correr fuera de conda MUSE o
  en contenedor; documentar versión del pipeline).
- **P**: reducción estándar con esorex (scibasic → scipost), sin sky
  subtraction agresiva inicial (elegir `skymethod` conservador), guardando
  también el `PIXTABLE` post-calibración para poder re-hacer cielo sin repetir
  todo. Registrar semillas/parámetros en QC.
- **S**: `raw/DATACUBE_FINAL.fits`, `raw/IMAGE_FOV*.fits`, pixtables
  intermedios, `stage00r_qc.json` con versiones y parámetros esorex.
- **V**: imagen de banda blanca; espectro integrado del campo; mapa de máscara
  de slices; comparación del flujo de la estándar reducida contra su tabla de
  referencia (residual espectrofotométrico vs λ).
- **C**: cubo con STAT válido (sin NaN masivos fuera de bordes), residual de la
  estándar < 5% rms en 4800–9300 Å, header WCS completo.
- **R**: versiones distintas del pipeline dan cubos ligeramente distintos
  (registrar versión); calibraciones del archivo pueden no ser las óptimas de
  la noche; NFM vs WFM cambia parámetros. Fallo típico: astrometría del header
  desplazada — se corrige en B1, no aquí.
- **K**: script `musepipe/reduction/esorex_driver.py` + shell reproducible
  `scripts/reduce_raw.sh`; no notebook (proceso batch), solo notebook de QC.

#### A2 · `00s_sky_zap` — Residuales de cielo con ZAP [ESENCIAL, condicional]

- **O**: eliminar residuales de skylines que contaminan el rojo (y la región
  Hα si hay líneas cercanas) sin distorsionar líneas astrofísicas.
- **I**: cubo de A1; máscara de fuentes (estrella + compañero + campo).
- **P**: correr ZAP con máscara que excluya explícitamente un cilindro
  generoso alrededor de la primaria y del compañero; comparar cubo pre/post.
- **S**: `cube_zap.fits`, `stage00s_qc.json` (nº de componentes, máscara usada).
- **V**: espectro de cielo residual en aperturas vacías pre/post ZAP; rms por
  canal en ventanas de skylines conocidas (5577, 6300, 6864, bandas OH);
  espectro del compañero pre/post ZAP superpuesto (debe cambiar solo en
  skylines, no en continuo).
- **C**: rms en ventanas de skylines reducido ≥ 2× sin cambio sistemático del
  continuo del compañero (> 1%) fuera de esas ventanas.
- **R**: ZAP puede absorber señal astrofísica si la máscara de fuentes es
  pobre; sobre-limpieza si se usan demasiadas componentes. Si el QC de A4
  muestra cielo ya limpio, esta etapa se salta (por eso "condicional").
- **K**: `musepipe/reduction/sky_zap.py`; parámetros en config; test sintético
  con skyline inyectada.

#### A3 · `00t_telluric` — Corrección telúrica con molecfit [OPCIONAL]

- **O**: corregir absorción telúrica en el rojo (>6800 Å) si el objetivo de
  modelado del continuo lo requiere. Hα (6563 Å) casi no está afectada, por lo
  que esta etapa **no bloquea** el análisis de Hα.
- **I**: cubo de A2; espectro de la primaria como referencia de alta señal.
- **P**: ajustar molecfit sobre el espectro de la primaria; aplicar la
  transmisión al cubo completo.
- **S**: `cube_telcorr.fits`, `stage00t_qc.json`.
- **V**: espectro primaria pre/post en bandas O₂/H₂O (6870, 7600, 8200 Å);
  residual relativo en líneas telúricas no saturadas.
- **C**: residuales < 3–5% del continuo en bandas corregidas.
- **R**: sobre-corrección introduce artefactos en el continuo rojo; documentar
  y propagar como término de error sistemático.
- **K**: `musepipe/reduction/telluric.py`.

#### A4 · `00q_cube_qc` — QC del cubo y calibraciones [ESENCIAL, todos los entry points]

- **O**: verificar longitud de onda, LSF, flujo y cielo del cubo antes de
  cualquier extracción; es la puerta de entrada común de `raw`, `eso_cube` y
  (versión reducida) `cropped_cube`.
- **I**: cubo (A1–A3 o externo), catálogo de skylines, fotometría de referencia
  de la primaria (literatura/simbad) y de la estándar si existe.
- **P**: (i) medir centroides de skylines vs valores de laboratorio → offset y
  dispersión de la solución de λ por región del campo; (ii) medir FWHM de
  skylines → LSF(λ) empírica; (iii) fotometría sintética de la primaria en
  bandas anchas vs literatura → escala de flujo; (iv) estadística del cielo en
  aperturas vacías; (v) verificación de la extensión STAT: ¿var(DATA) en
  regiones vacías ≈ mediana de STAT? (factor de corrección si no).
- **S**: `stage00q_qc.json` (offset λ, LSF(λ) tabulada, factor de flujo, factor
  STAT), `plots/stage00q_*.png`.
- **V**: offset de λ vs λ con barras de error; LSF vs λ; razón
  var-empírica/STAT vs λ; SED sintética vs fotometría publicada.
- **C**: |offset λ| < 0.1 Å (≈5 km/s en Hα) o corregible con término lineal;
  factor STAT en [0.8, 1.5] y estable; flujo dentro de ~10% de la literatura
  (o factor documentado).
- **R**: la primaria puede ser variable (joven) → discrepancias de flujo no
  implican mala calibración; skylines débiles en el azul → validar λ solo donde
  se pueda y extrapolar con cautela. Si STAT no es confiable, todo el bloque D2
  cae al plan B (ruido empírico), como hoy.
- **K**: `musepipe/qc/cube_qc.py` + notebook `00q_cube_qc.ipynb`. Este módulo
  es reutilizable para cualquier target futuro sin cambios.

### Bloque B — Preparación y localización (adapta lo existente)

#### B1 · `01_load_align_crop` [ESENCIAL — reutilizar]

- **O**: alinear exposiciones/cubos, recortar al campo de trabajo, apilar.
- **I**: cubo(s) de A4 o entrada directa (`cropped_cube` salta esta etapa).
- **P**: como hoy (perfil `maoppy_refined` o centrado por pico), con **dos
  cambios**: propagar la extensión STAT por las mismas transformaciones
  (resampleo suma en cuadratura con los mismos pesos) y registrar en QC el
  shift aplicado por exposición.
- **S**: `stage01_cube_stack.fits` (DATA+STAT), QC con shifts.
- **V**: ya existen (01b); añadir: mapa de STAT apilado y razón STAT/varianza
  empírica post-apilado.
- **C**: residuo de alineación < 0.1 px entre exposiciones; STAT propagado sin
  NaN nuevos.
- **R**: interpolación al alinear correlaciona píxeles vecinos → anotar en QC
  el kernel usado; esta covarianza espacial afecta el ruido de aperturas
  pequeñas (subestimación ~10–30%): cuantificar una vez y guardar el factor.
- **K**: migrar la lógica del notebook a `musepipe/stages/stage01_align.py`
  (pendiente O2); notebook queda fino.

#### B2 · `02_xcorr_stripes` [ESENCIAL — reutilizar]

- **O**: corregir el patrón de stripes/offsets espectrales entre slices.
- **I/P/S/V/C/R/K**: como hoy (`musepipe/stripes.py` ya centraliza la
  maquinaria; 02b/02c dan diagnósticos). Único añadido: escribir en QC una
  métrica escalar de amplitud de stripes pre/post para que la etapa de
  comparación (D1) pueda correlacionar artefactos con métodos.

#### B3 · `01c_target_localization` — Localización y astrometría [ESENCIAL, nuevo]

- **O**: determinar (y validar) las coordenadas del compañero y de la primaria
  en el crop de forma reproducible, en lugar de heredarlas hardcodeadas.
- **I**: cubo apilado de B2; separación/PA esperadas de la literatura (Bowler
  et al. 2017: ~1.75″); escala de píxel del header.
- **P**: (i) centroide de la primaria por canal (robusto, mediana en bandas);
  (ii) imagen colapsada en bandas rojas (donde el compañero es más brillante)
  y detección del compañero por matched filter con PSF preliminar; (iii)
  conversión a separación/PA y comparación con la predicción; (iv) escritura
  de coordenadas canónicas en config/QC.
- **S**: `stage01c_qc.json` con `(y,x)` de estrella y compañero, separación,
  PA, y desviación vs literatura; stamp de verificación.
- **V**: stamp del compañero con isofotas y cruz en el centroide; separación
  medida vs esperada; centroide del compañero vs λ (deriva cromática —
  diagnóstico de refracción diferencial residual).
- **C**: compañero detectado a >5σ en la imagen colapsada; separación dentro de
  ~2σ de la predicción orbital; deriva cromática < 0.5 px entre 5000 y 9000 Å
  (si es mayor, los extractores deben usar centroide dependiente de λ).
- **R**: confusión con otra fuente del campo (hay un segundo objeto conocido en
  el crop); si el compañero no se detecta en continuo, todo el plan cambia de
  "caracterización" a "búsqueda de línea" y la rama D toma prioridad.
- **K**: `musepipe/stages/stage01c_localize.py`; reutiliza `apertures.py` y el
  centrado de 01.

### Bloque C — PSF cromática y extractores

#### C1 · `E01_chromatic_psf` — Modelo de PSF cromática [ESENCIAL, nuevo]

- **O**: construir el modelo de PSF(λ) de la primaria que necesitan la
  extracción óptima (C3) y el PSF-fitting (C4); es la pieza central nueva.
- **I**: cubo de B2, coordenadas de B3, LSF/QC de A4; `maoppy` disponible.
- **P**: ajustar por canal (o en bins de ~50–100 Å con interpolación suave) un
  perfil Moffat elíptico (y opcionalmente el perfil `maoppy` para NFM/AO) a la
  primaria; suavizar parámetros (FWHM, β, elipticidad, ángulo) con polinomios
  de bajo orden en λ; generar función `psf_model(lam, dy, dx)` evaluable en
  cualquier posición.
- **S**: `stage_e01_psf_params.csv` (parámetros por bin), `psf_model.json`
  (coeficientes suavizados), stamps del modelo, `stage_e01_qc.json`.
- **V**: FWHM(λ), β(λ), elipticidad(λ) con ajuste suave superpuesto; mapa de
  residuo (dato − modelo)/modelo de la primaria en 4–6 bandas; residuo en el
  radio del compañero vs λ (la métrica que importa); curva de energía
  encapsulada del modelo vs dato.
- **C**: residuo relativo del halo en el radio del compañero < ~5% por banda;
  parámetros suaves sin saltos no físicos; el modelo reproduce los anillos de
  Airy/estructura AO al nivel que muestran los diagnósticos 04c/04d actuales.
- **R**: la PSF AO no es Moffat pura (anillos, "wings" cromáticas): si el
  residuo estructurado domina en el radio del compañero, escalar a un modelo
  híbrido (Moffat + residual empírico azimutal mediano). Saturación de la
  primaria en el core → ajustar con core enmascarado. Este es el mayor riesgo
  técnico del plan.
- **K**: nuevo módulo `musepipe/psf.py` (ajuste, suavizado, evaluación) +
  `musepipe/stages/stage_e01_psf.py`. Reescribir aquí lo útil de los notebooks
  04c/04d y eliminarlos como fuente de verdad.

#### C2 · `X01_aperture` — Extracción por apertura + fondo local [ESENCIAL — envuelve lo existente]

- **O**: espectro de control transparente y auditable (benchmark interno).
- **I**: cubo de B2 (modo `native_stage02`, como 04b hoy), coordenadas B3.
- **P**: exactamente la cadena 04b actual (superficie local `plane`, clipping
  robusto) + suma en apertura 3×3/5×5; añadir varianza: STAT sumado en
  cuadratura × factor de A4/B1.
- **S**: `spec_aperture_object.fits` en el formato estándar (§2.3).
- **V**: las figuras de Far_04b/07 actuales + espectro con banda de error
  STAT vs error empírico.
- **C**: reproduce el baseline validado del refactor (equivalencia numérica ya
  demostrada); errores STAT y empíricos consistentes dentro de ~30%.
- **R**: pérdida de flujo por apertura finita (corregir con curva de
  crecimiento del modelo C1: factor de apertura por λ); contaminación del halo
  si el fondo local no la captura — precisamente lo que C4 debe superar.
- **K**: wrapper delgado `musepipe/extraction/aperture.py` sobre `stage04b` +
  `apertures.py`; casi todo ya existe.

#### C3 · `X02_optimal` — Extracción óptima tipo Horne [ESENCIAL, nuevo]

- **O**: maximizar S/N del espectro del compañero ponderando por perfil PSF y
  varianza, sin postproceso agresivo.
- **I**: cubo residual tras restar el fondo (superficie local de C2 **o** el
  modelo de primaria de C1 — ambas variantes, comparables), modelo PSF de C1,
  varianza STAT.
- **P**: por canal: `f = Σ w·P·D/V / Σ w·P²/V` con perfil `P` del modelo C1
  centrado en el compañero (centroide fijo o cromático según B3), pesos `w`
  con sigma-clipping de outliers/cósmicos; varianza `1/Σ(P²/V)`.
- **S**: `spec_optimal_object.fits` (dos variantes de fondo si aplica).
- **V**: ganancia de S/N vs apertura por canal (histograma y mediana); mapa de
  píxeles rechazados por clipping (no debe concentrarse en el objeto); espectro
  óptimo vs apertura superpuestos.
- **C**: S/N mediana ≥ la de apertura (típicamente ganancia 10–30%); sin sesgo
  sistemático de continuo vs apertura (>2–3%) — si lo hay, el perfil P está mal
  centrado o mal normalizado.
- **R**: sensible a errores del modelo de PSF (un P demasiado ancho diluye, uno
  estrecho pierde flujo); Horne asume perfil correcto — el test de inyección
  (E4) mide el throughput real.
- **K**: `musepipe/extraction/optimal.py`, ~200 líneas, con test sintético
  (fuente gaussiana + ruido conocido → recuperación de flujo y varianza).

#### C4 · `X03_psf_fitting` — Ajuste simultáneo estrella+compañero [ESENCIAL, nuevo]

- **O**: método primario recomendado por la literatura para separación tipo
  ROXs 12: deblending directo del par por canal, atacando la contaminación del
  halo sin sustracción agresiva.
- **I**: cubo de B2 (sin sustracción previa de fondo), modelo PSF C1,
  posiciones B3, STAT.
- **P**: por canal (o bin), ajuste por mínimos cuadrados lineales de
  `D = a·PSF(x_s) + b·PSF(x_c) + fondo(1, y, x)` con posiciones fijas (o
  refinadas suavemente en λ); los flujos `a(λ)`, `b(λ)` salen del ajuste;
  evaluar también con PampelMuse como implementación de referencia externa y
  comparar (decidir después cuál es la canónica).
- **S**: `spec_psffit_object.fits`, `spec_psffit_star.fits`, cubo residual del
  ajuste, `stage_x03_qc.json` (χ² por canal, condición del sistema).
- **V**: χ²ᵣ(λ); mapas de residuo en bandas (no debe quedar estructura en las
  posiciones de las fuentes); espectro de la primaria ajustada vs su espectro
  por apertura grande (validación de escala); correlación a(λ)·b(λ) (si es
  fuerte, el sistema está mal condicionado y hay que fijar más parámetros).
- **C**: χ²ᵣ ~ 1 con STAT calibrado; residuo en posición del compañero
  consistente con ruido; espectro del compañero sin "crosstalk" evidente de
  líneas de la primaria (test: ¿aparecen las líneas fotosféricas de la primaria
  en emisión/absorción espuria en b(λ)?).
- **R**: degeneración flujo-fondo si el término de fondo es demasiado flexible;
  errores del modelo PSF se propagan directamente a b(λ) — el residuo de C1 en
  el radio del compañero es el presupuesto de error sistemático; a 1.75″ el
  problema está bien condicionado, pero verificarlo con la matriz de
  covarianza del ajuste.
- **K**: `musepipe/extraction/psffit.py` (núcleo propio, lineal y testeable) +
  adaptador opcional `musepipe/extraction/pampelmuse_adapter.py`. El núcleo
  propio es preferible como canónico por trazabilidad; PampelMuse queda como
  verificación cruzada.

#### C5 · `X04_hrsdi` — HRSDI / sustracción espectral de halo [OPCIONAL — segunda línea]

- **O**: buscar Hα débil bajo halo residual: normalizar el espectro estelar
  local y restarlo (high-resolution SDI, tipo Xie et al. 2020), maximizando
  sensibilidad a línea en emisión.
- **I**: cubo de B2, continuo estelar de C4, máscara de líneas.
- **P**: por spaxel: ajustar/escalar el espectro estelar de referencia al
  continuo local y restar; opcionalmente PCA de bajo orden sobre los residuos
  (reutilizando la maquinaria de 04/06c con `IncrementalPCA` para memoria).
- **S**: cubo residual HRSDI, mapa de Hα integrado, espectro residual en el
  compañero.
- **V**: residuo por canal en controles al mismo radio; mapa Hα con controles;
  curva de throughput de línea (de E4) **obligatoria** antes de interpretar.
- **C**: ruido residual en Hα menor que el de C2–C4 en la misma banda (si no,
  la etapa no aporta); throughput de línea medido, no asumido.
- **R**: auto-sustracción de la línea si el compañero tiene continuo propio;
  covarianza espectral fuerte → cualquier flujo citado desde esta rama debe ir
  con corrección de throughput e incertidumbre de la corrección.
- **K**: `musepipe/extraction/hrsdi.py`; reutiliza PCA existente con modo
  low-memory como ruta canónica (cierra O4).

#### C6 · `X05_pca_fm` — PCA/KLIP + forward modeling [OPCIONAL — solo si C5 se usa para flujo]

- **O**: si un flujo de línea de la rama HRSDI/PCA va a inferencia física,
  corregir throughput/auto-sustracción con forward modeling y estimar
  covarianza espectral.
- **I**: residuales de C5, modelo PSF C1, espectro sintético de inyección.
- **P**: proyección del modelo de compañero sobre la base PCA (KLIP-FM estilo
  Pueyo 2016) o, plan B más simple y robusto: matriz de throughput empírica
  por inyección-recuperación densa (de E4) + covarianza empírica de controles.
- **S**: espectro corregido + matriz de covarianza (`_cov.npz`).
- **V**: sesgo de inyección-recuperación post-corrección (~0 dentro de errores);
  estructura de la matriz de covarianza (visualizada).
- **C**: sesgo residual < 10% del error estadístico en la banda de Hα.
- **R**: complejidad alta; solo se activa si C5 produce un candidato o si el
  límite superior de la rama conservadora no basta científicamente.
- **K**: `musepipe/extraction/forward_model.py`; la variante empírica primero.

### Bloque D — Comparación de métodos y calibración del espectro

#### D1 · `X10_method_comparison` — Comparación inter-método [ESENCIAL, nuevo]

- **O**: decidir con datos qué método es el canónico para ROXs 12 B y detectar
  sesgos que ningún método puede ver por sí solo. Implementa el árbol de
  decisión del documento de investigación (¿concuerdan PSF-fitting y apertura?
  → no hace falta PCA; ¿divergen? → refinar PSF antes de escalar).
- **I**: todos los `spec_<metodo>_object.fits` disponibles (mínimo C2, C3, C4).
- **P**: remuestreo a la malla común (ya lo están), razones y diferencias por
  canal; comparación en bandas integradas (5 bandas de continuo + bandas de
  línea Hα/Hβ/OI); χ² de consistencia par a par usando errores propagados;
  misma comparación sobre los espectros de *controles* al mismo radio (los
  métodos deben concordar en que los controles son ruido).
- **S**: `tables/method_comparison.csv`, `stage_x10_qc.json` con veredicto
  (`consistent` / `divergent_continuum` / `divergent_lines`), figura resumen.
- **V**: panel espectros superpuestos con bandas de error; razón
  método/método vs λ con banda de ±1σ; tabla de flujos en bandas con
  discrepancias en σ; S/N por método vs λ.
- **C**: **éxito del plan conservador** = C3 y C4 concuerdan con C2 dentro de
  2σ en continuo y bandas de línea. Si divergen: criterio explícito para
  iterar C1 (refinar PSF) antes de activar C5/C6.
- **R**: concordancia no garantiza exactitud (sesgos comunes, p.ej. mismo
  modelo PSF); por eso E4 (inyección) es el árbitro final de throughput.
- **K**: `musepipe/stages/stage_x10_compare.py` + notebook de revisión. Es
  genérico por construcción (opera sobre el formato estándar §2.3).

#### D2 · `X11_spectral_calibration` — λ, flujo, continuo e incertidumbres [ESENCIAL, nuevo]

- **O**: dejar el espectro canónico calibrado y con errores defendibles.
- **I**: espectro canónico elegido en D1, QC de A4 (offset λ, LSF, factor
  flujo, factor STAT), corrección baricéntrica (header).
- **P**: (i) aplicar offset de λ validado y corrección baricéntrica (cierra
  C7 parcialmente); (ii) aplicar factor de flujo y factor de apertura (curva
  de crecimiento C1); (iii) continuo por `continuum_running_median` existente
  + variante por ajuste polinómico con máscara de líneas — ambas guardadas;
  (iv) errores: STAT propagado × factores de A4/B1, validado contra el ruido
  empírico de controles (el mayor de los dos como error final conservador);
  (v) presupuesto de sistemáticos documentado (flujo abs., telúricas, PSF).
- **S**: `spec_final_object.fits` (con columnas de continuo y error total),
  `stage_x11_qc.json` con el presupuesto de errores.
- **V**: espectro final con continuo superpuesto; error STAT vs empírico vs λ;
  posición de skylines residuales tras corrección de λ (deben caer en 0);
  comparación del continuo con la fotometría/tipo espectral esperado de un
  M8–L0 joven (sanity check de forma).
- **C**: errores empírico y propagado consistentes (razón en [0.7, 1.4]);
  continuo estable frente a la elección de método de continuo (< 1σ).
- **R**: si STAT resultó inutilizable (A4), se documenta y se usa solo
  empírico — el plan no se bloquea; variabilidad de la primaria limita la
  calibración absoluta (~10%, anotar).
- **K**: `musepipe/stages/stage_x11_calibrate.py`; amplía `spectral.py` y
  `stats.py` con las variantes de continuo y el merge de errores.

### Bloque E — Análisis de Hα

#### E1 · `H01_halpha_detection` — Detección y significancia [ESENCIAL — reutiliza 07/07b/08/08c]

- **O**: cuantificar la evidencia de emisión Hα en el espectro de cada método
  con estadística calibrada empíricamente.
- **I**: espectros del formato estándar (todos los métodos), cubos residuales
  correspondientes, controles al mismo radio (31, como en C4).
- **P**: matched filter en velocidad alrededor de 6563 Å (± ~500 km/s, ancho
  de plantilla = LSF de A4 y variantes anchas por acreción); z-score empírico
  contra controles (maquinaria de 07b); calibración look-elsewhere sobre los
  3465 canales buenos (maquinaria de 08c) **por método**; centroide y FWHM si
  hay señal.
- **S**: `tables/halpha_detection_by_method.csv` (flujo, σ, FAP local y
  global, centroide, FWHM), figuras por método.
- **V**: espectro en ±100 Å de Hα por método con plantilla superpuesta;
  distribución de máximos de controles vs valor del objeto (como 08c);
  mapa espacial del canal Hα con posiciones de controles; señal en velocidad
  (¿centrada en la RV del sistema ± incertidumbre?).
- **C**: criterio de detección predefinido (evita p-hacking): FAP global < 1%
  en ≥ 2 métodos independientes **y** centroide consistente con la RV del
  sistema dentro de la LSF. Cualquier otra cosa = no-detección o "candidato a
  investigar".
- **R**: el resultado actual (FAP 100% con local-surface) sugiere
  no-detección; el riesgo principal es el sesgo del analista al "buscar hasta
  encontrar" — por eso el criterio se fija aquí, antes de mirar.
- **K**: refactor de interfaz: `stage07b`/`stage08c` aceptan un espectro
  estándar §2.3 como entrada además del residual 04b. Núcleo estadístico
  intacto (ya validado).

#### E2 · `H02_artifact_tests` — Artefactos y contaminación [ESENCIAL, nuevo como etapa formal]

- **O**: someter cualquier señal (o su ausencia) a pruebas de artefactos.
- **I**: cubos residuales, QC de stripes (B2), catálogo de skylines, señal E1.
- **P**: batería fija: (i) ¿coincide el canal de la señal con skyline/stripe
  residual? (correlación con métrica B2 por canal); (ii) coherencia espacial:
  ¿la "señal" tiene la forma de la PSF o de un artefacto alargado? (ajuste de
  stamp); (iii) estabilidad temporal: señal presente en sub-stacks
  independientes (mitades de exposiciones); (iv) estabilidad frente a
  parámetros (radio de ajuste, máscara, nº de componentes PCA); (v) test de
  desplazamiento espectral: repetir E1 en líneas "placebo" (p.ej. 6400, 6700 Å).
- **S**: `stage_h02_qc.json` con veredicto por test, figuras.
- **V**: stamp de la señal vs stamp de PSF esperada; señal vs sub-stack;
  sensibilidad del flujo a parámetros (tornado plot).
- **C**: una detección solo sobrevive si pasa (i)–(iv); una no-detección es
  robusta si ningún placebo supera el umbral de E1 (consistencia del FAP).
- **R**: pocas exposiciones limitan (iii); documentar potencia de cada test.
- **K**: `musepipe/stages/stage_h02_artifacts.py`; reutiliza stamps y
  controles existentes.

#### E3 · `H03_upper_limits` — Límite superior [ESENCIAL si no-detección]

- **O**: convertir la no-detección en límite superior publicable de flujo y
  luminosidad de Hα (y tasa de acreción vía relaciones L_acc–L_Hα).
- **I**: resultados E1/E4, distancia y extinción del sistema (literatura),
  throughput por método.
- **P**: límite al 5σ desde la distribución de controles corregido por
  throughput medido (no asumido); conversión a L_Hα con distancia y A_V (con
  su incertidumbre); comparación con los límites de la literatura MUSE (Xie
  et al. 2020) escalados a la separación de ROXs 12 B; conexión con la cadena
  09/10 existente (proyección de tiempos y masas) si se quiere.
- **S**: `tables/halpha_upper_limits.csv` (por método y combinado), figura de
  límite vs separación con la literatura.
- **V**: límite vs método (deben ser consistentes tras throughput); límite vs
  los puntos de la literatura.
- **C**: límites inter-método consistentes dentro de ~30%; documento claro de
  qué método define el límite final y por qué.
- **R**: extinción incierta domina el error en L_Hα → reportar ambos (flujo
  observado y desextinguido).
- **K**: `musepipe/stages/stage_h03_limits.py`; reutiliza piezas de 09/10.

#### E4 · `H04_injection_recovery` — Inyección-recuperación multi-método [ESENCIAL — generaliza stage06]

- **O**: medir sensibilidad y throughput de **toda** la cadena para señales Hα
  débiles, por método de extracción; es el árbitro de D1 y el insumo de E3.
- **I**: cubo de B2 pre-extracción, modelo PSF C1, malla de inyección.
- **P**: inyectar compañeros sintéticos (PSF de C1, no gaussiana ad hoc) con
  línea Hα de flujo/ancho variables sobre continuo variable, en la posición
  real (runs clonados, como `ROXs12b_HaInject_*`) y en las 3 posiciones de
  control; propagar por C2, C3, C4 (y C5 si está activa); malla mínima:
  S/N ∈ {0, 1, 2, 3, 5, 7, 10} × anchos {LSF, 2×LSF} × 3 posiciones —
  compatible con la grilla C2 ya ejecutada para poder comparar.
- **S**: `tables/injection_throughput_by_method.csv` (throughput, sesgo de
  flujo/centroide/FWHM, completitud al umbral de E1), figuras.
- **V**: throughput vs S/N inyectado por método; flujo recuperado vs inyectado
  (línea 1:1); completitud vs flujo; sesgo de centroide vs S/N.
- **C**: throughput de C2 reproduce la validación C1 histórica (regresión);
  throughput de C3/C4 ≥ C2 con sesgo < 5% para S/N ≥ 5; curvas monótonas y sin
  sorpresas — cualquier anomalía bloquea la publicación del límite.
- **R**: costo computacional (mitigar con runs clonados solo de la subregión,
  patrón ya usado en C1: 918 vóxeles); la inyección con PSF equivocada
  sesgaría el throughput — usar la PSF medida con perturbaciones (±10% FWHM)
  para estimar sensibilidad del resultado al modelo.
- **K**: generalizar `stage06_local_surface_injection` → `musepipe/injection.py`
  con interfaz `inject(cube, catalog) → cube` independiente del extractor; los
  extractores se pasan como lista de callables.

### Bloque F — Productos finales

#### F1 · `R01_final_products` — Productos, tablas y reporte [ESENCIAL]

- **O**: consolidar un paquete estándar por run: espectro final, límites o
  flujos de Hα, comparación de métodos, y QC global; reproducible con un
  comando.
- **I**: salidas de D1, D2, E1–E4.
- **P**: recolectar QC de todas las etapas en un `run_summary.json`; generar
  el set fijo de figuras "de papel": (1) campo + posiciones, (2) PSF(λ),
  (3) espectros por método superpuestos, (4) espectro final con errores,
  (5) zoom Hα por método con controles, (6) throughput/completitud,
  (7) límite superior vs literatura; tabla maestra por línea (Hα, Hβ, [OI]).
- **S**: `runs/<RUN_ID>/report/` con figuras, tablas y `report.md` generado.
- **V**: checklist automática: toda etapa ESENCIAL con QC en verde; hashes de
  entradas consistentes entre etapas (sin mezcla de runs).
- **C**: `run_summary.json` sin etapas en rojo; el reporte se regenera
  idénticamente desde los productos (determinismo).
- **K**: `musepipe/report.py` + `scripts/build_report.py`.

---

## 4. Esencial vs opcional

**Primera reducción confiable (ruta mínima):**

```text
A1 → A2(cond.) → A4 → B1 → B2 → B3 → C1 → C2 → C3 → C4 → D1 → D2 → E1 → E2 → E4 → E3(si no-detección) → F1
```

**Opcional / segunda fase:** A3 (telúrica), C5 (HRSDI), C6 (forward modeling),
adaptador PampelMuse, covarianza espectral completa, ANDROMEDA como validador
de fotometría de línea, deconvolución (solo si aparece emisión extendida),
CCF/RV con templates (resto de C7), conexión completa con 09/10.

Orden de implementación sugerido (cada hito deja algo usable):

1. **Hito 1 — infraestructura**: A4 (QC del cubo, incluye STAT), B3
   (localización), formato estándar §2.3, C2 como wrapper. Con esto la cadena
   actual ya gana varianza y coordenadas reproducibles.
2. **Hito 2 — PSF y extractores**: C1, C3, C4, D1. Primer veredicto
   multi-método sobre el continuo.
3. **Hito 3 — Hα**: interfaz nueva de 07b/08c (E1), E2, E4 generalizada, E3.
   Resultado científico principal.
4. **Hito 4 — raw**: A1/A2/A3 (puede correr en paralelo con hitos 2–3, porque
   los entry points lo desacoplan), D2 completo, F1.
5. **Fase 2**: C5/C6 solo si D1/E1 lo justifican.

---

## 5. Riesgos globales y supuestos del plan

1. **PSF AO no-Moffat** (C1) es el riesgo técnico dominante; el plan lo mitiga
   con el modelo híbrido y con perturbaciones de PSF en E4.
2. **STAT inutilizable o ausente** en los productos históricos: A4 lo detecta;
   plan B empírico ya validado.
3. **Memoria** en PCA/inyección con crop 170: rutas low-memory como canónicas
   (cierra O4); las inyecciones usan clones de subregión.
4. **Mezcla de runs**: el patrón `MUSE_RUN_ID`/hashes en QC se extiende a toda
   etapa nueva (cierra parte de O6).
5. **Sesgo de confirmación** en Hα: criterio de detección fijado en E1 antes
   de mirar los espectros nuevos; placebos y controles obligatorios.
6. **esorex/Reflex fuera del entorno conda**: documentar en `docs/` la
   instalación (o contenedor) y fijar versión del pipeline en QC.

## 6. Modularización para futuros targets

- Todo lo específico de ROXs 12 vive en `runs/<RUN_ID>/config/config.json`
  (coordenadas esperadas, separación/PA de literatura, bandas malas, RV);
  ningún módulo nuevo puede contener números del target.
- Nuevos paquetes: `musepipe/reduction/` (A1–A3), `musepipe/qc/` (A4),
  `musepipe/psf.py` (C1), `musepipe/extraction/` (C2–C6, un módulo por método
  con la misma firma `extract(cube, var, psf, coords, cfg) → SpectrumProduct`),
  `musepipe/injection.py` (E4), `musepipe/report.py` (F1).
- El formato estándar de espectro (§2.3) es el contrato entre bloques: para un
  target nuevo (ROXs 42B b, YSES 2b ya presentes en `runs/`) basta un config
  nuevo y correr la misma cadena.
- Tests por módulo: sintético + contrato, siguiendo el patrón de las 52
  pruebas existentes; añadir una regresión automatizada sobre
  `ROXs12b_short` (cierra O7).
- Los notebooks quedan solo como capa de inspección; ninguna lógica nueva en
  celdas (regla del refactor, se mantiene).
