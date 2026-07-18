# Especificación A1 · `00r_raw_reduction` — brief de ejecución para Codex (ChatGPT 5.5)

Fecha: 2026-07-01. Etapa A1 del plan `docs/plan_roxs12_reduccion_multimetodo.md`.
Este documento es la instrucción completa para el agente. Léelo entero antes de
ejecutar nada.

---

## 0. Rol, objetivo y definición de terminado

**Rol del agente**: implementar y ejecutar la reducción de datos crudos MUSE de
ROXs 12 con el pipeline oficial de ESO (`esorex`), dejando código, logs y QC
reproducibles dentro del contrato de etapas de este repositorio. No eres
responsable de decisiones científicas: eres responsable de que la reducción sea
trazable, verificable y detenerte cuando algo no cuadre.

**Objetivo**: producir `DATACUBE_FINAL.fits` (extensiones DATA + STAT) desde los
raws asociados a la observación del **2022-08-29** de ROXs 12 (el producto de
archivo actual es `ADP.2022-09-12T17_17_39.371.fits`, en
`~/Documents/GitHub/Muse/Data/ROX12b/20220829/`), junto con los pixtables
post-calibración, QC JSON y figuras de verificación.

**Definición de terminado (todas obligatorias)**:

1. `runs/ROXs12b_raw/raw_reduction/DATACUBE_FINAL.fits` existe, con DATA y STAT
   sin NaN masivos fuera de bordes/slice gaps.
2. `PIXTABLE_REDUCED_*.fits` guardados (permiten re-hacer cielo sin repetir todo).
3. `stage00r_qc.json` completo según el esquema de §7.
4. Las 6 verificaciones de §8 ejecutadas y en verde, incluida la comparación
   contra el cubo ADP de referencia.
5. `scripts/reduce_raw.sh` + `musepipe/reduction/esorex_driver.py` versionados,
   con los que la reducción se repite con un solo comando.
6. Reporte final (§10) entregado al usuario.

---

## 1. Límites duros (violarlos = fallo de la tarea)

1. **No modificar ni borrar nada** dentro de los runs históricos existentes:
   `runs/ROXs12b*`, `runs/LkCa_15`, `runs/ROXs42Bb`, `runs/YSES_2b`. La única
   excepción permitida es crear y usar el run nuevo `runs/ROXs12b_raw/` para A1.
2. **No modificar** módulos existentes de `musepipe/` (config, stats, stages…).
   Solo se crean archivos nuevos: `musepipe/reduction/` (nuevo paquete),
   `scripts/reduce_raw.sh`, tests nuevos en `tests/`, y este QC.
3. **No tocar los datos originales**: `~/Documents/GitHub/Muse/Data/` es de solo
   lectura. Los raws descargados van a un directorio nuevo declarado en config
   (`raw_data_dir`), nunca mezclados con los ADP.
4. **No inventar parámetros de esorex**. Todo parámetro no-default debe venir de
   esta especificación o de la documentación oficial del pipeline MUSE citada en
   el QC. Si un parámetro parece necesario y no está aquí: **detenerse y
   preguntar** (§9).
5. **Nada de fallbacks silenciosos**: si falta una calibración, si esorex
   devuelve código ≠ 0, o si un producto esperado no aparece, la fase falla y se
   reporta. Prohibido "continuar con lo que hay".
6. **No usar credenciales ESO ni descargar datos sin checkpoint humano** (§9).
7. Git: trabajar en rama `stage-a1-raw-reduction`; commits pequeños por fase;
   versionar solo código, tests y documentación. Los artefactos dependientes del
   run (`raw_data/`, SOF, QC, logs, FITS y `runs/ROXs12b_raw/`) quedan locales y
   no se commitean; respetar `.gitignore`.
8. Presupuestos: disco ≤ 120 GB adicionales, RAM pico esperada 16–32 GB en
   `muse_scipost`. Verificar antes de lanzar (§3). Si la máquina no lo cumple:
   detenerse y proponer alternativas, no degradar parámetros por tu cuenta.

---

## 2. Contexto técnico que debes asumir

- Target: ROXs 12 (primaria + compañero subestelar B a ~1.75″). Ciencia final:
  espectro fiel del compañero y búsqueda de Hα. Implicación para A1: **la
  fidelidad espectrofotométrica y la extensión STAT importan más que la
  estética del cubo**.
- La región 5780–6050 Å está contaminada por el láser de AO (NaLGS): es
  esperable y se enmascara aguas abajo. No intentes "arreglarla" en A1.
- El modo instrumental (WFM-AO o NFM) **no se asume**: se lee de los headers
  (`HIERARCH ESO INS MODE`) en la Fase 1 y se registra en QC. Toda la
  asociación de calibraciones depende de ese valor.
- El repositorio ya tiene un contrato de etapas (ver README y
  `docs/plan_roxs12_reduccion_multimetodo.md` §2.2): módulo + QC JSON + figuras
  + notebook fino. A1 lo cumple igual que las etapas migradas.

---

## 3. Fase 0 — Verificación del entorno (gate: no continuar sin pasar)

Comprobar y registrar en `stage00r_qc.json` (sección `environment`):

1. `esorex --version` y versión del pipeline MUSE (`esorex --recipes | grep muse`).
   Si `esorex` no está instalado, o si existe pero no lista recetas `muse_*`,
   lanzar un error de entorno y proponer al usuario UNA de estas rutas, esperando
   confirmación: (a) instalador oficial ESO (macOS: `install_esoreflex` o kit
   MacPorts de ESO), (b) contenedor (imagen ESO con pipeline MUSE fijado).
   **No instalar software de sistema sin confirmación.**
2. Versión del pipeline MUSE ≥ 2.8.x (soporte completo de modos AO). Registrar
   la versión exacta; será parte del hash de reproducibilidad.
3. Espacio libre en disco del volumen de trabajo (≥ 120 GB) y RAM total.
4. Python del repo funciona: `python -m unittest discover -s tests` pasa (52+
   pruebas) ANTES de tocar nada — es la línea base de regresión.
5. Crear `runs/ROXs12b_raw/{config,raw_reduction,stages,logs,plots}` y un
   `config.json` mínimo con `entry_point: "raw"`, `raw_data_dir`, versiones.
   El QC de la etapa vive explícitamente en
   `runs/ROXs12b_raw/stages/stage00r_qc.json`.

Criterio de éxito: todo lo anterior registrado; ningún valor "desconocido".

## 4. Fase 1 — Inventario y descarga de datos (checkpoint humano)

1. Leer el header del ADP existente (`ADP.2022-09-12T17_17_39.371.fits`) y
   extraer: `PROG.ID`, `OBS.ID`, fecha/hora, modo INS, tiempo de exposición,
   número de exposiciones combinadas (keywords `PROV*`). Esto identifica los
   raws exactos sin ambigüedad.
2. Construir la lista de datasets a pedir al ESO Archive con **calselector**
   (asociación "raw calibrations"): ciencia (`OBJECT`), y por cada noche los
   BIAS, FLAT (lamp), ARC (wavelength), ILLUM, TWILIGHT/SKYFLAT si existen,
   STD (estándar espectrofotométrica) y las tablas estáticas del modo
   (GEOMETRY_TABLE, ASTROMETRY_WCS, line catalogs, extinction table, filter
   list). Nota de atención: **ILLUM se toma contigua a la ciencia** y la
   GEOMETRY/ASTROMETRY correcta depende del modo y de la época — usar la que
   calselector asocie, no la "más reciente". La corrección ILLUM (flat de
   iluminación adjunto) en `muse_scibasic` sigue la recomendación de
   **Xie et al. 2020 (2011.08043) §3.1** para eliminar derivas de throughput
   entre IFUs; se usa exactamente una ILLUM por tipo de raw (ciencia y STD),
   la más contigua en tiempo (verificable en `illumination_correction` del
   `stage00r_qc.json`).
3. **Checkpoint humano**: presentar al usuario la tabla de archivos a descargar
   (nombre, tipo, tamaño total estimado) y esperar aprobación + credenciales/
   descarga manual si el portal lo requiere.
4. Tras la descarga: verificar checksums (MD5 del archivo ESO si está
   disponible; si no, registrar SHA256 propio de cada archivo en QC), y
   clasificar por `DPR TYPE`/`DPR CATG` leyendo headers, **no por nombre de
   archivo**. Escribir `raw_inventory.csv` (archivo, tipo, fecha, modo, binning,
   exptime, sha256).

Gate: la ciencia tiene su set completo de calibraciones del mismo modo y
binning, dentro de las ventanas de validez. Cualquier hueco → reportar y
preguntar (típico: falta TWILIGHT; documentar la consecuencia y pedir decisión).

## 5. Fase 2 — Construcción de SOFs y plan de ejecución (sin ejecutar aún)

Generar programáticamente (en `esorex_driver.py`, no a mano) los SOF de la
cascada estándar MUSE, en este orden:

```text
muse_bias → muse_flat → muse_wavecal → muse_lsf → muse_twilight (si hay)
→ muse_scibasic (ciencia, STD e ILLUM) → muse_standard → muse_scipost
```

Reglas:

- Un SOF por recipe y por grupo de calibración, escrito a
  `runs/ROXs12b_raw/raw_reduction/sof/`, legible y reproducible. Los SOF son
  productos locales dependientes del run y no se versionan.
- Validación previa ("dry-run"): antes de ejecutar cada recipe, comprobar que
  cada archivo del SOF existe, que su tag coincide con su `DPR TYPE` real y
  que los conteos son plausibles (p. ej. ≥ 5 BIAS, ≥ 3 FLAT por grupo). Esta
  validación es código, no una revisión visual.
- **Decisión del usuario (2026-07-01): usar la configuración por defecto del
  pipeline.** Todos los recipes corren con sus defaults (en particular
  `skymethod` queda en su valor default de `muse_scipost`). La ÚNICA
  desviación permitida es `--save=cube,individual` en `muse_scipost`, para
  conservar los `PIXTABLE_REDUCED` que A2/A4 necesitan; queda registrada en QC
  como única no-default.
  - Registrar en QC el valor efectivo de TODOS los parámetros relevantes de
    `muse_scipost` (`skymethod`, `filter`, `crsigma`, `rvcorr`…), aunque sean
    default: los defaults cambian entre versiones del pipeline.
  - `--nifu=-1` o paralelización por IFU si la RAM lo exige — decisión
    registrada en QC, mismo resultado.
- Cualquier otro parámetro no-default que consideres necesario: preguntar antes.

Gate: revisión de los SOF por el usuario (pegar resumen en el reporte de fase:
recipe → nº de inputs por tag). No ejecutar `muse_scipost` sin este OK.

## 6. Fase 3 — Ejecución con control de errores

1. Ejecutar la cascada con el driver, un recipe a la vez. Por recipe:
   - log completo de esorex a `runs/ROXs12b_raw/logs/<recipe>_<ts>.log`;
   - código de retorno ≠ 0 → abortar la cascada, no los siguientes recipes;
   - parsear el log en busca de `[ WARNING ]` y `[ ERROR ]`; los warnings se
     recopilan en QC (`warnings_by_recipe`) y los que mencionen
     `saturat|missing|extrapolat|bad pixels > X%` se elevan a revisión;
   - verificar que los productos declarados del recipe existen y tienen
     tamaño > 0 antes de dar el paso a `completed`.
2. Orden de prueba: correr primero **una sola** cadena de calibración +
   `muse_scibasic` de UNA exposición de ciencia de punta a punta antes de
   procesar el resto (si hay varias exposiciones). Esto detecta problemas de
   asociación con el mínimo costo.
3. Registrar por recipe: duración, RAM pico si es medible, versión, sha256 de
   productos principales.
4. Reintentos: **cero reintentos automáticos con parámetros distintos**. Un
   fallo se diagnostica y se reporta; solo se reintenta idéntico si la causa
   fue externa (disco lleno, kill por memoria) y tras corregirla.

Puntos donde mantener máxima atención durante esta fase:

- **Asociación temporal**: que cada ciencia use la ILLUM contigua y el grupo de
  calibraciones de su propia noche/modo. El error clásico es mezclar flats de
  otro binning u otro modo — el dry-run de Fase 2 lo debe impedir, y aquí se
  re-verifica leyendo los headers de los productos maestros.
- **La estándar espectrofotométrica**: si `muse_standard` produce una curva de
  respuesta con oscilaciones fuertes o el log avisa de pocas cuentas, la
  fotometría de todo el cubo queda comprometida → parar y reportar.
- **skymethod**: el campo es pequeño y con dos fuentes puntuales; si el modelo
  de cielo se construye del propio campo, verificar que la máscara de cielo no
  incluya la primaria ni el compañero (inspeccionar `SKY_MASK` producido). Si
  no es controlable con máscara o el producto/procedencia de máscara no permite
  completar V6, lanzar error y reportar: la alternativa (`--skymethod=none` +
  ZAP en A2) es decisión del usuario, no tuya.
- **STAT**: comprobar tras `muse_scipost` que la extensión STAT existe, es
  positiva, y su mediana en regiones vacías es del orden de la varianza
  empírica (chequeo grueso; el fino es A4).

## 7. Esquema de `stage00r_qc.json` (obligatorio, claves estables)

```json
{
  "stage": "00r_raw_reduction",
  "run_id": "ROXs12b_raw",
  "timestamp_utc": "...",
  "environment": {"esorex": "...", "muse_pipeline": "...", "platform": "...", "container_or_native": "..."},
  "inputs": {"adp_reference": {"file": "...", "sha256": "..."},
              "prog_id": "...", "obs_id": "...", "ins_mode": "...",
              "n_science_exposures": 0, "raw_inventory_csv": "..."},
  "recipes": [{"name": "muse_bias", "sof": "...", "status": "ok",
                "duration_s": 0, "n_warnings": 0, "products": [{"file": "...", "sha256": "..."}]}],
  "scipost_params": {"skymethod": "...", "save": "...", "filter": "..."},
  "warnings_by_recipe": {},
  "products": {"datacube": "...", "pixtables_reduced": ["..."], "whitelight": "..."},
  "verification": {"v1_stat_present": true, "v2_std_residual_rms": 0.0,
                    "v3_wcs_ok": true, "v4_adp_whitelight_corr": 0.0,
                    "v5_adp_star_spec_ratio_rms": 0.0, "v6_sky_mask_clean": true},
  "gates_passed": ["fase0", "fase1", "fase2", "fase3"],
  "open_issues": []
}
```

## 8. Verificaciones finales (las 6 de la definición de terminado)

Implementarlas como funciones en `musepipe/reduction/verify.py` con un test
sintético mínimo cada una, y ejecutarlas sobre el cubo real:

- **V1 — STAT**: existe, > 0, fracción de NaN en DATA/STAT fuera de bordes
  < 5%; mapa de NaN guardado en `plots/`.
- **V2 — Estándar**: residual rms de la respuesta espectrofotométrica < 5% en
  4800–9300 Å (excluyendo 5780–6050 Å y bandas telúricas). Figura: respuesta y
  residuo vs λ.
- **V3 — WCS/headers**: WCS espacial y espectral completos; `CD3_3`/`CRVAL3`
  consistentes con el ADP; keywords de modo y exptime coherentes con Fase 1.
- **V4 — Imagen blanca vs ADP**: correlación espacial (tras registro entero de
  píxeles) entre la imagen de banda blanca propia y la del ADP > 0.95. No se
  exige igualdad: versiones distintas del pipeline difieren; se exige la misma
  escena. Figura lado a lado + diferencia.
- **V5 — Espectro de la primaria vs ADP**: extraer con apertura idéntica en
  ambos cubos; la razón propia/ADP debe ser suave y estar en [0.9, 1.1] en el
  80% de los canales buenos. Desviaciones estructuradas (escalones, pendiente
  fuerte) → issue abierto, no silencio. Esta es **la verificación más
  importante de A1**: conecta la re-reducción con todo lo ya validado.
- **V6 — Máscara de cielo**: la máscara usada por scipost no solapa con las
  posiciones de la primaria ni del compañero (separación ~1.75″). Figura con
  máscara + posiciones. Si no existe una máscara/procedencia verificable para
  completar esta prueba, la verificación falla con error duro; A1 no queda en
  verde y se pregunta al usuario cómo proceder.

Cada V produce figura en `runs/ROXs12b_raw/plots/stage00r_*.png` y su número en
el QC. Una V en rojo no se "aprueba con nota": queda en `open_issues` y se
reporta.

## 9. Protocolo de parada y preguntas al usuario

Detenerse y preguntar (con contexto y opciones concretas) cuando:

1. Falten calibraciones asociadas o haya ambigüedad de asociación (dos grupos
   candidatos de flats, ILLUM ausente, etc.).
2. Un recipe falle o emita warnings elevados (§6.1) que no entiendas del log.
3. Cualquier V de §8 salga en rojo.
4. Haya que instalar software, descargar datos, o exceder los presupuestos de
   disco/RAM.
5. Un parámetro no-default parezca necesario (p. ej. cambiar `skymethod`,
   `crsigma`, offsets de exposiciones).

Formato de la pregunta: qué pasó (con la línea exacta del log), qué opciones
hay, qué recomiendas y por qué, y qué queda bloqueado mientras tanto. Una sola
pregunta consolidada por bloqueo, no un goteo.

## 10. Reporte final requerido

Al terminar (o al bloquearse definitivamente), entregar en el PR/respuesta:

- tabla recipe → estado, duración, nº warnings;
- las 6 verificaciones con sus números y miniaturas de figuras;
- diferencias conocidas contra el ADP (versión de pipeline, skymethod) y su
  impacto esperado en las etapas B–F;
- `open_issues` priorizados;
- confirmación explícita de que los límites de §1 se respetaron (checklist);
- comandos exactos para reproducir todo desde `raw_inventory.csv`.

## 11. Organización del código a crear

```text
musepipe/reduction/__init__.py
musepipe/reduction/esorex_driver.py   # inventario, SOFs, ejecución, parsing de logs
musepipe/reduction/verify.py          # V1–V6
scripts/reduce_raw.sh                 # entrada única: env → fases 0–3 → verify
tests/test_reduction_driver.py        # SOF building y clasificación por DPR con headers sintéticos
tests/test_reduction_verify.py        # V1–V6 sobre mini-cubos fabricados
```

Estilo: funciones puras donde sea posible, sin estado global, type hints,
docstrings con el "porqué" de cada parámetro no-default, y el mismo patrón de
QC/paths que las etapas migradas (`musepipe/paths.py` como referencia). El
driver no debe depender de rutas absolutas: todo desde el `config.json` del run.
