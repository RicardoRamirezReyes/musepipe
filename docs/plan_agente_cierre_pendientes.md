# Plan de ejecución para agente — cierre de pendientes (A1 primero)

> **Audiencia:** un agente ejecutor SIN contexto previo de este proyecto. Sigue los pasos
> literalmente, en orden. Cada paquete de trabajo (WP) tiene criterios de aceptación
> ("Hecho cuando…") y condiciones de parada ("PARA y reporta si…"). No improvises fuera
> de lo escrito: si un paso no se puede completar como está descrito, PARA y reporta.
>
> Escrito: 2026-07-08. Fuente de verdad de los pendientes: `runs/ROXs12b_raw/PAPER_BLOCKERS.md`.

---

## 0. Contexto mínimo

Este repositorio reduce y analiza datos MUSE/VLT (modo NFM) de la estrella ROXs 12 para
buscar acreción (línea Hα) en su compañera subestelar ROXs 12 B. La cadena A→F ya corrió
completa, pero el **cubo re-reducido propio está desalineado** (bloqueador #0): la cascada
combinó las 7 exposiciones de ciencia sin correr `muse_exp_align`, lo que creó una fuente
espuria (S/N≈31) y difuminó a la compañera. Mientras eso no se corrija, ningún resultado
es válido para el paper y todo el análisis B–F corre provisionalmente sobre el cubo ADP
de ESO.

### Rutas clave (todas existen; verifica antes de empezar)

| Qué | Ruta |
|---|---|
| Repo | `/home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline` |
| Run de la re-reducción | `runs/ROXs12b_raw/` |
| Productos de la cascada | `runs/ROXs12b_raw/raw_reduction/products/` |
| SOFs originales | `runs/ROXs12b_raw/raw_reduction/sof/` |
| Cubo desalineado (NO usar para ciencia) | `runs/ROXs12b_raw/raw_reduction/products/muse_scipost/DATACUBE_FINAL.fits` |
| Cubo ADP de referencia (bien alineado) | `Data/ROX12b/20220829/ADP.2022-09-12T17_17_39.371.fits` |
| Datos crudos | `/home/ricardo-ramirez/Descargas/MUSE_DATA/Rox12/` (171 FITS — SOLO LECTURA) |
| Lista de bloqueadores | `runs/ROXs12b_raw/PAPER_BLOCKERS.md` |
| Driver esorex | `musepipe/reduction/esorex_driver.py` |
| Verificaciones A1 | `musepipe/reduction/verify.py` |

### Entorno

```bash
conda activate MUSE                    # python del proyecto (pytest, astropy, pandas)
esorex --version                       # 3.13.10, en /home/linuxbrew/.linuxbrew/bin
esorex --man-page muse_scipost | head  # plugin MUSE 2.10.16 debe estar disponible
```

### Convención de nombres de pixtables

`runs/ROXs12b_raw/raw_reduction/products/muse_scibasic_object/PIXTABLE_OBJECT_000N-II.fits`
donde `N` = exposición 1…7 e `II` = IFU 01…24. Hay 168 pixtables (7×24).

---

## 1. Reglas globales (OBLIGATORIAS)

1. **Nunca borres ni modifiques**: los datos crudos en `Descargas/MUSE_DATA/`, los
   productos existentes bajo `runs/*/`, ni el cubo ADP. Los productos nuevos van SIEMPRE
   a directorios nuevos (indicados en cada WP).
2. **No hagas `git commit` ni `git push`** salvo que el usuario lo pida explícitamente.
   (Además el filtro nbstripout está roto en esta máquina para notebooks; no toques `.ipynb`.)
3. **Peculiaridades conocidas de esorex** (bugs ya diagnosticados — no los "redescubras"):
   - esorex **ignora `--output-dir`**: los productos caen en el directorio de trabajo.
     SIEMPRE haz `cd` al directorio de salida antes de invocar esorex.
   - Los parámetros de receta van **después** del nombre de la receta, no antes.
   - Un log que contiene la palabra "error" NO implica fallo: el criterio es el
     **código de salida** de esorex (0 = OK) y que los productos existan.
4. **Trabajos largos** (>10 min): lánzalos desanclados con `setsid nohup … > log 2>&1 &`
   y monitorea el log; no los dejes como job normal de shell.
5. **Registra todo**: cada esorex debe tener `--log-file` propio bajo `runs/ROXs12b_raw/logs/`.
6. Al terminar cada WP escribe un resumen (plantilla en §12) antes de pasar al siguiente.
7. **PARA y reporta** (no intentes rutas alternativas no descritas aquí) si: falta un
   archivo listado como existente, un esorex sale con código ≠ 0 dos veces seguidas, o
   un criterio de aceptación no se cumple.

---

## 2. WP-1 — A1: realinear las 7 exposiciones (CRÍTICO, hazlo primero)

**Objetivo:** producir un `DATACUBE_FINAL` alineado en
`runs/ROXs12b_raw/raw_reduction/products/muse_scipost_aligned/`, donde la compañera se vea
con S/N comparable al ADP (18.8) y sin la fuente espuria.

**Estrategia:** (a) correr `muse_scipost` una vez POR exposición solo para obtener 7
`IMAGE_FOV`; (b) `muse_exp_align` sobre esas 7 imágenes → `OFFSET_LIST.fits`; (c) re-correr
el `muse_scipost` combinado original AÑADIENDO el `OFFSET_LIST` al SOF. Las calibraciones
maestras (bias, flat, wavecal, LSF, respuesta) **ya existen y se reutilizan; NO las repitas**.

### Fase 1.0 — comprobaciones previas

```bash
cd /home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline
ls runs/ROXs12b_raw/raw_reduction/products/muse_scibasic_object/PIXTABLE_OBJECT_000?-*.fits | wc -l  # debe dar 168
wc -l runs/ROXs12b_raw/raw_reduction/sof/muse_scipost.sof   # debe dar 196
df -h .   # necesitas >= 80 GB libres (7 cubos ~3.5 GB + pixtables + cubo final)
```

PARA si algo no coincide.

### Fase 1.1 — SOFs por exposición

Las 28 líneas de calibración del SOF original (todo lo que NO es `PIXTABLE_OBJECT`) se
comparten; solo cambian los pixtables. Ejecuta literalmente:

```bash
cd /home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline
RR=$PWD/runs/ROXs12b_raw/raw_reduction
grep -v PIXTABLE_OBJECT $RR/sof/muse_scipost.sof > /tmp/scipost_calibs.sof
for N in 1 2 3 4 5 6 7; do
  SOF=$RR/sof/muse_scipost_exp${N}.sof
  grep "PIXTABLE_OBJECT_000${N}-" $RR/sof/muse_scipost.sof > $SOF
  cat /tmp/scipost_calibs.sof >> $SOF
  echo "exp${N}: $(grep -c PIXTABLE_OBJECT $SOF) pixtables (debe ser 24), total $(wc -l < $SOF) (debe ser 52)"
done
```

PARA si algún SOF no tiene 24 pixtables y 52 líneas.

### Fase 1.2 — scipost por exposición (solo para IMAGE_FOV)

Usa `save=cube` (NO `cube,individual`: el guardado individual tiene un bug conocido de
clave FITS no única y aquí no lo necesitamos). Cada corrida tarda ~10–20 min.

```bash
RR=/home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline/runs/ROXs12b_raw/raw_reduction
LOGS=/home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline/runs/ROXs12b_raw/logs
for N in 1 2 3 4 5 6 7; do
  OUT=$RR/products/muse_scipost_exp${N}
  mkdir -p $OUT && cd $OUT
  esorex --log-file=$LOGS/muse_scipost_exp${N}.log muse_scipost --save=cube $RR/sof/muse_scipost_exp${N}.sof
  echo "exp${N} exit=$?"
done
```

Criterio por exposición: código de salida 0 y existen `$OUT/DATACUBE_FINAL.fits` e
`$OUT/IMAGE_FOV_0001.fits`. Nota: es esperado el aviso de no-convergencia del ajuste de
líneas de cielo (campo NFM pequeño); NO es un fallo.

### Fase 1.3 — muse_exp_align

```bash
RR=/home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline/runs/ROXs12b_raw/raw_reduction
LOGS=/home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline/runs/ROXs12b_raw/logs
SOF=$RR/sof/muse_exp_align.sof
: > $SOF
for N in 1 2 3 4 5 6 7; do
  echo "$RR/products/muse_scipost_exp${N}/IMAGE_FOV_0001.fits IMAGE_FOV" >> $SOF
done
OUT=$RR/products/muse_exp_align
mkdir -p $OUT && cd $OUT
esorex --log-file=$LOGS/muse_exp_align.log muse_exp_align $SOF
echo "exit=$?"; ls -la $OUT
```

Criterio: código 0 y existe `$OUT/OFFSET_LIST.fits`. Inspecciona los offsets:

```bash
python - <<'EOF'
from astropy.io import fits
t = fits.open("/home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline/runs/ROXs12b_raw/raw_reduction/products/muse_exp_align/OFFSET_LIST.fits")
print(t[1].columns.names)
for row in t[1].data:
    print(row)
EOF
```

Sensatez: los offsets deben ser del orden de fracciones de arcosegundo (campo NFM ~7.5",
escala 0.0253"/px). **PARA y reporta si** `muse_exp_align` no encuentra fuentes (la
estrella es muy brillante, debería encontrarla siempre) o si algún offset supera 1" —
en ese caso adjunta el log y los 7 IMAGE_FOV al reporte; hay un plan B (offsets manuales
por centroide de la estrella) que decidirá el usuario.

### Fase 1.4 — scipost combinado con OFFSET_LIST

```bash
RR=/home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline/runs/ROXs12b_raw/raw_reduction
LOGS=/home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline/runs/ROXs12b_raw/logs
SOF=$RR/sof/muse_scipost_aligned.sof
cp $RR/sof/muse_scipost.sof $SOF
echo "$RR/products/muse_exp_align/OFFSET_LIST.fits OFFSET_LIST" >> $SOF
OUT=$RR/products/muse_scipost_aligned
mkdir -p $OUT && cd $OUT
setsid nohup esorex --log-file=$LOGS/muse_scipost_aligned.log muse_scipost --save=cube,individual $SOF > $LOGS/muse_scipost_aligned.console.log 2>&1 &
```

Tarda ~75 min. Monitorea con `tail -f` del log. Criterio: proceso termina, código 0
(míralo en el console.log) y existe `$OUT/DATACUBE_FINAL.fits` (~3.5 GB, extensiones
DATA y STAT). Si el guardado `individual` vuelve a fallar por la clave FITS no única
("File cannot be created"), re-lanza con `--save=cube` solamente y anótalo en el reporte
(es el bloqueador #2, no lo bloquees tú).

### Fase 1.5 — validación (aceptación de WP-1)

Ejecuta este script tal cual y guarda su salida:

```bash
python - <<'EOF'
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline"
CUBE = BASE + "/runs/ROXs12b_raw/raw_reduction/products/muse_scipost_aligned/DATACUBE_FINAL.fits"

hdul = fits.open(CUBE, memmap=True)
data = hdul[1].data                     # (nlam, ny, nx)
hdr  = hdul[1].header
lam = hdr["CRVAL3"] + (np.arange(data.shape[0]) - (hdr["CRPIX3"]-1)) * hdr["CD3_3"]
# banda roja 8800-9350 A: donde la companera (muy roja) es mas brillante
sel = (lam > 8800) & (lam < 9350)
img = np.nanmedian(data[sel], axis=0)

# estrella = pixel mas brillante
ys, xs = np.unravel_index(np.nanargmax(img), img.shape)
print(f"estrella en (y={ys}, x={xs}), pico={img[ys, xs]:.1f}")

# companera esperada a (dy=+71, dx=-9) de la estrella (sep 1.81", PA conocida del ADP)
yc, xc = ys + 71, xs - 9
box = img[yc-2:yc+3, xc-2:xc+3]
# fondo: anillo de radios 8-14 px alrededor de la posicion esperada
yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
r = np.hypot(yy - yc, xx - xc)
bg = img[(r > 8) & (r < 14)]
snr = (np.nanmax(box) - np.nanmedian(bg)) / np.nanstd(bg)
print(f"companera esperada en (y={yc}, x={xc}): S/N = {snr:.1f}  [criterio: >= 10]")

# fuente espuria: ningun otro pico > 10 sigma a >20 px de estrella y companera
med, std = np.nanmedian(img), np.nanstd(img[(np.hypot(yy-ys, xx-xs) > 30)])
det = (img - med) / std
det[np.hypot(yy - ys, xx - xs) < 20] = 0
det[np.hypot(yy - yc, xx - xc) < 10] = 0
n_spur = int(np.sum(det > 10))
print(f"pixeles espurios >10 sigma fuera de estrella/companera: {n_spur}  [criterio: ~0, tolera <5]")

plt.figure(figsize=(7, 7))
plt.imshow(np.arcsinh(img / np.nanstd(img)), origin="lower", cmap="magma")
plt.scatter([xs, xc], [ys, yc], s=120, facecolors="none", edgecolors=["cyan", "lime"])
plt.title("banda 8800-9350 A — cian: estrella, verde: companera esperada")
plt.savefig(BASE + "/runs/ROXs12b_raw/plots/aligned_cube_check.png", dpi=130)
print("figura: runs/ROXs12b_raw/plots/aligned_cube_check.png")
EOF
```

**Hecho cuando:** S/N de la compañera ≥ 10 en la posición esperada, píxeles espurios < 5,
y la figura muestra dos fuentes puntuales limpias. **PARA y reporta** (con la figura) si
la compañera no aparece o la fuente espuria persiste.

---

## 3. WP-2 — verify.py: corregir V1/V4 y correr V1–V6 (bloqueador #1)

**Requiere:** WP-1 hecho. **Archivo a modificar:** `musepipe/reduction/verify.py`
(léelo completo antes de tocar nada; hay tests en `tests/` que lo cubren — encuéntralos
con `grep -rl "verify" tests/`).

1. **V1 (fracción de NaN):** hoy aplica un tope plano de 5% de NaN y falla porque el cubo
   NFM tiene 11.7% de NaN **estructural**: (a) la región del láser NaLGS 5780–6050 Å está
   enmascarada en TODOS los spaxels, y (b) los bordes del campo NFM son NaN en TODAS las
   longitudes de onda. Cambia V1 para calcular la fracción de NaN EXCLUYENDO: los canales
   con λ ∈ [5780, 6050] Å, y los spaxels donde TODO el eje espectral es NaN (máscara de
   borde). Con eso la fracción restante debe quedar < 5% y V1 debe pasar.
2. **V4 (comparación con el ADP):** hoy exige forma idéntica y falla (323×365 vs 330×338).
   Cambia V4 para reproyectar/recortar ambos cubos a la región de solape usando el WCS
   (vale interpolación al vecino más cercano canal a canal, o comparar métricas globales
   —flujo de la estrella, FWHM— en vez de píxel a píxel; elige lo más simple que dé una
   comparación cuantitativa y documenta la elección en el docstring).
3. Añade/ajusta tests unitarios sintéticos para ambos cambios (sigue el estilo de los
   tests existentes del módulo) y corre **toda** la suite: `python -m pytest -q`. Debe
   quedar todo en verde (la referencia previa es 336 passed).
4. Corre las seis verificaciones V1–V6 contra el cubo alineado de WP-1 y el ADP, y guarda
   el resultado como QC JSON en `runs/ROXs12b_raw/stages/` (sigue el patrón de los
   `stage00*_qc.json` existentes).

**Hecho cuando:** pytest en verde, y V1–V6 corren todas con PASS (o con un fallo
justificado y documentado en el QC — p. ej. V2/V5/V6 nunca se habían corrido; si alguna
falla por una razón física real, PARA y reporta en vez de relajar el umbral).

---

## 4. WP-3 — re-correr A2, A3 y A4 sobre el cubo alineado

**Requiere:** WP-1. Los tres pasos tienen scripts en `scripts/` y stages ya probados;
apúntalos al cubo nuevo (`products/muse_scipost_aligned/DATACUBE_FINAL.fits`). Antes de
correr, lee cada script para ver cómo recibe el cubo de entrada (config del run o flag).

1. **A2 (decisión ZAP):** `scripts/sky_zap.sh` — se espera de nuevo `zap_no_necesario`
   (la vez anterior R=0.617 ≤ 1.5). Si ahora diera `needed`, PARA y reporta (cambiaría el
   flujo: habría que instalar `pip install zap` y aplicarlo).
2. **A3 (telúrico):** `scripts/telluric.sh` — método STD_TELLURIC escalado por masa de
   aire (molecfit NO converge, es el bloqueador #4; no pierdas tiempo con molecfit aquí).
   Produce el `cube_telcorr.fits` nuevo. Verifica: profundidad de la banda O2 pasa de
   ~6.8% a <1%, Hα [6540–6590] intacta.
3. **A4 (QC de cubo):** `scripts/cube_qc.sh` — corre M1–M5 sobre el cubo nuevo Y el ADP,
   regenera la tabla comparativa. M1/M2 seguirán poco fiables (cubo con cielo restado) y
   M3 no disponible (falta Gaia, WP-7): son bloqueadores conocidos (#5, #6), NO los
   resuelvas aquí, solo deja constancia en el QC.

**Hecho cuando:** existe el nuevo `cube_telcorr.fits` (este pasa a ser LA entrada de la
cadena B–F), y los QC JSON de A2/A3/A4 están escritos con sus métricas.

---

## 5. WP-4 — B1–B3 sobre el cubo propio + comparación con el ADP

**Requiere:** WP-3. Crea un run NUEVO (p. ej. `runs/ROXs12b_realigned/`) copiando el
patrón de config de `runs/ROXs12b_B_adp/config/` pero apuntando al `cube_telcorr.fits`
nuevo. NO reutilices `runs/ROXs12b_B` (contiene la evidencia del cubo roto) ni
sobrescribas `runs/ROXs12b_B_adp`.

1. Corre las etapas B (carga/recorte/alineación, franjas): `scripts/stage01c_localize.sh`,
   `scripts/stage02_xcorr.sh` con `--run-id` del run nuevo.
2. Criterio central: la localización de la compañera en el crop debe caer en ~(156, 76)
   ±3 px (la posición histórica equivalente es (152, 72) en el run macOS antiguo), con
   S/N de detección comparable al ADP.
3. Escribe una tabla corta comparando (posición, S/N, FWHM de la estrella) entre el run
   nuevo y `ROXs12b_B_adp`.

**Hecho cuando:** B1–B3 en verde en el run nuevo y la tabla comparativa escrita. A partir
de aquí la cadena C–F sobre el cubo propio es decisión del usuario (repórtalo y espera).

---

## 6. WP-5 — ≥31 controles y wframe (bloqueador #10) [independiente de WP-1; corre sobre ROXs12b_B_adp]

1. En la config del run `ROXs12b_B_adp` sube `x0N_control_apertures` a ≥ 31 (busca la
   clave exacta con `grep -rn "control_apertures" musepipe/`).
2. Corrige la etiqueta de frame: `x0N_wframe = barycentric` (los datos SON baricéntricos,
   vbary = −29.64 km/s; hoy están mal etiquetados como topocéntricos).
3. Re-corre C2/C3/C4 (`scripts/stage_x01_aperture.sh`, `stage_x02_optimal.sh`,
   `stage_x03_psffit.sh`) y luego D2 + E1 (`stage_x11_calibrate.sh`, `stage_h01_detect.sh`).
   psffit es lento; usa los knobs de paralelismo existentes (`h04_process_pool` es solo de
   E4; para las etapas por canal existe `n_jobs`, ganancia ~1.6×).
4. Verifica en el QC de E1 que `min_resolvable_fap ≤ 1/31 ≈ 0.032` y reporta el FAP del
   pico más alto.

**Hecho cuando:** E1 re-corrido con ≥31 controles, FAP<1% ya es medible, y el veredicto
(se espera que siga `non_detection`) está en el QC. Si el veredicto CAMBIA a detección,
PARA y reporta de inmediato.

---

## 7. WP-6 — D1: escala de flujo común entre métodos (bloqueador #9) [requiere criterio; hazlo DESPUÉS de WP-5]

Problema real (no bug): los espectros de control de los 4 métodos difieren ×20 en
magnitud (optimal-LS/04b ~2243 vs aperture 111, optimal-psfsub 108, psffit 325), así que
la comparación D1 es ininterpretable.

1. La variante optimal-LS hereda la sobre-sustracción del stage04b (fondo no nulo en los
   controles). Acción: desactívala de D1 o rebásala (opción de config preferida — busca
   cómo D1 selecciona métodos con `grep -rn "optimal_ls" musepipe/stages/stage_x10*`).
2. Asegura que aperture, optimal-psfsub y psffit reporten **flujo total dentro del mismo
   NORMRAD** y con la misma convención de fondo (revisa `musepipe/extraction/aperture.py`
   y los stages x01–x03; la corrección de apertura por curva de crecimiento solo es válida
   en extracciones wings-intact — no la apliques al residual de 04b).
3. Re-corre D1 (`scripts/stage_x10_compare.sh`).

**Hecho cuando:** el veredicto de D1 deja de ser `uninterpretable` y los z de control
entre métodos quedan en un rango razonable (|z| ≲ 3, no cientos). Este WP tiene decisiones
de diseño: si en el paso 2 encuentras más de una convención defendible, PARA y presenta
las opciones al usuario en vez de elegir tú.

---

## 8. WP-7 — M3: calibración absoluta con Gaia (bloqueador #6) [independiente]

1. Descarga los passbands de Gaia DR3 (G, BP, RP) desde la página oficial de la misión
   (ESA/Gaia DPAC, "Gaia DR3 passbands") y colócalos en `gaia_passbands/` (hoy solo hay
   un README ahí — léelo por si fija formato esperado).
2. Obtén las magnitudes Gaia DR3 de ROXs 12 (la primaria) vía VizieR/archivo Gaia y
   añádelas a la config donde A4/M3 las espera (busca con `grep -rn "gaia" musepipe/qc/`).
3. Re-corre A4-M3: sintetiza la fotometría del cubo en los passbands y compárala con las
   magnitudes de catálogo → factor de escala absoluta de flujo.
4. Propaga: D2 (`stage_x11_calibrate.sh`) debe dejar de usar `flux_scale = 1`.

**Hecho cuando:** M3 produce un factor de escala con incertidumbre, documentado en el QC
de A4, y D2 lo aplica. Si el factor difiere de 1 en más de ~30%, PARA y reporta antes de
propagarlo (cambiaría los límites de E3 en la misma proporción).

---

## 9. WP-8 — molecfit o justificación (bloqueador #4) [baja prioridad, opcional]

molecfit_model no convergió ni en la primaria ni en la estándar (transmisión→0, χ² sin
cambio: problema de configuración, no de los datos). Dos salidas aceptables:
(a) diagnosticar la config de molecfit (empieza por WAVE_INCLUDE y los parámetros
atmosféricos iniciales del driver en `runs/ROXs12b_raw/raw_reduction/molecfit/`) hasta que
converja en la estándar; o (b) redactar en `docs/` una justificación método-STD_TELLURIC
(escalado de masa de aire T^(X_sci/X_std), T=1 forzada en Hα y región láser, verificada
O2 6.8%→0.6%) para citarla en el paper. Si (a) no converge en ~2 horas de trabajo, cambia
a (b). **Hecho cuando:** una de las dos salidas está completa.

---

## 10. WP-9 — G3: librerías externas y cierre de la clasificación [independiente; requiere red]

La clasificación G4 quedó `substellar_companion / AMBIGUOUS` porque faltan los ajustes
espectrales reales. Descarga (y documenta URL + versión + fecha de cada una):

1. **BT-Settl** (grid CIFIST): espectros sintéticos Teff 1500–4000 K, log g 3.5–5.0.
2. **BHAC15** (isócronas Baraffe+2015) y **ATMO2020** (Phillips+2020): tracks evolutivos.
3. **Plantillas empíricas** de enanas M/L jóvenes (Luhman / Bonnefoy): las familias ya
   están citadas en la config del run (`runs/ROXs12b_B_adp/config/`).

Colócalas donde `musepipe/models/` las espera (lee los protocolos en
`musepipe/models/protocols*.py` o equivalente y el `pending_libraries` en el QC de G3
para el formato). Implementa los adaptadores mínimos que pidan los protocolos, corre
`stage_g3_accretion.py` real y luego re-corre G4:

```bash
python -c "from musepipe.stages.stage_g4_classify import run_stage_g4; run_stage_g4('ROXs12b_B_adp')"
```

**Hecho cuando:** G3 produce SpT/Teff con intervalos (o `not_constrained` justificado con
las librerías ya presentes) y G4 se re-evalúa; reporta si la ambigüedad
planeta/BD/M-asociada se resuelve o qué dato falta (p. ej. segunda época astrométrica).

---

## 11. Orden de ejecución y dependencias

```
WP-1 (A1 realinear)  ──► WP-2 (verify V1–V6) ──► WP-3 (A2–A4) ──► WP-4 (B1–B3)
      │                                                                │
      └── crítico, primero                        (C–F sobre cubo propio: decide el usuario)

En paralelo (no dependen de WP-1):
  WP-5 (≥31 controles, sobre ROXs12b_B_adp) ──► WP-6 (D1 escala de flujo)
  WP-7 (Gaia M3)      WP-8 (molecfit, opcional)      WP-9 (librerías G3)
```

Si solo hay tiempo para una cosa: **WP-1**.

## 12. Plantilla de reporte por WP (obligatoria)

```
WP-N <nombre> — estado: COMPLETO / PARCIAL / BLOQUEADO
- Qué se hizo: (comandos clave y rutas de productos nuevos)
- Criterios de aceptación: (cada uno con PASS/FAIL y el número medido)
- Desviaciones del plan: (qué y por qué; "ninguna" si no hubo)
- Archivos modificados: (lista exacta; confirmar que NO se commiteó)
- Pendiente / siguiente paso sugerido:
```
