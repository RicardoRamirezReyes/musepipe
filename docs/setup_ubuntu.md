# Puesta en marcha en Ubuntu — MUSE-accretion-pipeline

Guía para replicar el entorno de trabajo en una máquina Ubuntu con conda.
Fecha: 2026-07-03.

---

## 1. Qué copiar (git no lo trae todo)

El repositorio Git contiene código, docs y specs, pero **no** los datos:

| Qué | Dónde vive en la máquina original | Tamaño aprox. | En git |
|---|---|---:|---|
| Código + docs | `MUSE-accretion-pipeline/` | MBs | Sí |
| Productos de runs | `MUSE-accretion-pipeline/runs/` | ~80 GB | **No** (gitignored) |
| Cubo(s) ADP | `~/Documents/GitHub/Muse/Data/ROX12b/20220829/` | ~5 GB/cubo | **No** (fuera del repo) |
| `active_run.txt` | raíz del repo | bytes | **No** (gitignored) |

Copiar con rsync preservando estructura (ejemplo desde la Mac):

```bash
rsync -avh --progress MUSE-accretion-pipeline/ usuario@ubuntu:/ruta/MUSE-accretion-pipeline/
rsync -avh --progress ~/Documents/GitHub/Muse/Data/ usuario@ubuntu:/ruta/Data/
```

Si copias la carpeta completa (no `git clone`), `runs/` y `active_run.txt`
viajan solos. Verifica al llegar: `du -sh runs/` debe dar decenas de GB.

## 2. Trampa #1 — rutas absolutas de macOS en los configs

Los `config.json` de los runs históricos referencian el cubo con ruta absoluta
de macOS:

```text
/Users/ricardoramirez/Documents/GitHub/Muse/Data/ROX12b/20220829/ADP...fits
```

Esa ruta no existe en Ubuntu. Dos opciones:

- **Opción A (recomendada, no toca los configs):** recrear la ruta por symlink:

```bash
sudo mkdir -p /Users/ricardoramirez/Documents/GitHub/Muse
sudo ln -s /ruta/Data /Users/ricardoramirez/Documents/GitHub/Muse/Data
```

- **Opción B:** editar `cube_files` en los `config.json` de las COPIAS en
  Ubuntu (la máquina nueva es una copia; el original queda intacto). Anotar el
  cambio si se hace.

Las etapas que solo leen productos ya generados en `runs/` no necesitan el
cubo original; las que lo re-leen (01, y A2/A3/A4 con `provenance=adp`) sí.

## 3. Trampa #2 — filtro nbstripout en git

`.gitattributes` aplica `nbstripout` a los notebooks, y el filtro quedó
configurado con una ruta absoluta de conda de la Mac
(`/opt/miniconda3/envs/MUSE/bin/python3.10`). En Ubuntu, **cualquier
`git status`/`checkout` que toque notebooks fallará** hasta arreglarlo:

```bash
conda activate MUSE
pip install nbstripout
nbstripout --install          # reescribe el filtro con el python local
git config filter.nbstripout.clean   # verificar que apunta al env local
```

## 4. Entorno conda

```bash
cd /ruta/MUSE-accretion-pipeline
conda env create --file environment.yml     # crea el env "MUSE" (python 3.10, versiones fijadas)
conda activate MUSE
```

Paquetes extra NO incluidos en `environment.yml` (instalar según la etapa):

```bash
pip install zap          # etapa A2 (verificar: python -c "import zap; print(zap.__version__, zap.process)")
pip install nbstripout   # sección 3
```

Nada del bloque B–F requiere más que `environment.yml` (+ `maoppy`, que ya
está en el yml). No se necesita GPU; para PCA/inyecciones con crop 170 conviene
RAM ≥ 32 GB (las rutas low-memory funcionan con menos).

## 5. Software de sistema — solo para A1 y A3

El pipeline ESO **no** se instala con conda ni apt genérico:

- **esorex + pipeline MUSE (etapa A1)**: instalador oficial de ESO para Linux
  (`install_esoreflex` o los kits de pipelines de ESO). El kit MUSE incluye las
  calibraciones estáticas: **descarga de varios GB** (ver §7).
- **molecfit (etapa A3)**: kit de ESO aparte (recipes `molecfit_model`,
  `molecfit_calctrans`, `molecfit_correct`), cientos de MB.

Si por ahora sigues con el cubo ADP (`entry_point=eso_cube`), puedes postergar
ambos: A2/A4 y toda la cadena B–F no los usan.

## 6. Verificación antes de correr nada (en orden)

```bash
conda activate MUSE
cd /ruta/MUSE-accretion-pipeline

# 1. Suite de tests (52+; todos deben pasar)
python -m unittest discover -s tests

# 2. Compilación de módulos
python -m compileall musepipe tests stage08_full_spectrum_for_modeling.py

# 3. Resolución de run y configuración (no escribe nada)
MUSE_RUN_ID=ROXs12b_short python -c "from musepipe.config import load_run_config; print(load_run_config().run_id)"
MUSE_RUN_ID=ROXs12b_short python -c "from musepipe.stages import stage04b_config_from_run; print(stage04b_config_from_run())"

# 4. Acceso al cubo (prueba la sección 2)
python - <<'EOF'
from astropy.io import fits
import json
cfg = json.load(open('runs/ROXs12b/config/config.json'))['config']
f = cfg['cube_files'][0]
with fits.open(f) as h:
    print(f, '->', h[1].data.shape)
EOF
```

Regla del repo: `ROXs12b_short` para pruebas rápidas, `ROXs12b` para ciencia.
Preferir `MUSE_RUN_ID=...` explícito sobre editar `active_run.txt`.

## 7. Descargas grandes — dónde y cuánto

Respuesta directa a "¿hay fases con bases de datos importantes?":

| Fase | Descarga | Tamaño | Cuándo |
|---|---|---:|---|
| **A1** | Raws de ciencia + calibraciones del ESO Archive (calselector) | **decenas de GB** | Solo al activar la re-reducción; checkpoint humano antes de descargar |
| **A1** | Kit del pipeline MUSE + calibraciones estáticas de ESO | ~4–6 GB | Al instalar esorex |
| **A3** | Perfiles atmosféricos GDAS para molecfit | MBs (necesita red al correr) | Solo si A3 decide aplicar corrección |
| **A4** | Fotometría Gaia DR3 de la primaria | KBs (consulta puntual) | Una vez; se guarda en config |
| — | `runs/` históricos (~80 GB) | copia local, no descarga | En la migración (§1) |

Nada en B–F descarga datos externos.

## 8. Orden de trabajo sugerido en la máquina nueva

1. §1–§4 (copia, symlink de datos, nbstripout, env) → §6 completo en verde.
2. Continuar exactamente donde quedó el trabajo con Codex: ramas
   `stage-a2-sky-zap` / `stage-a3-telluric` ya contienen
   `musepipe/reduction/` y `scripts/`; la secuencia completa de etapas y
   comandos está en `docs/manual_flujo_roxs12.pdf`.
3. Instalar esorex/molecfit (§5) solo cuando se decida activar A1/A3.
