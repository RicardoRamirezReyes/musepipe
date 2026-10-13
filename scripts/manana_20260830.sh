#!/bin/bash
# Segunda tanda del 2026-08-30 (~2 h 20). Re-corre TRES configuraciones en los
# dos objetos volcando las FILAS CRUDAS, que la tanda de la noche no guardo.
#
# Por que estas tres: `20x12` es produccion (la referencia), `20x8` gano en las
# dos mitades de ROXs 42B b y `16x12` fue el elegido en la mitad A de ROXs 12 b
# —y el que perdio fuera de muestra—. Con las filas en disco, remuestrear `f50`,
# probar otras particiones A/B y mover la FPR salen del CSV sin recomputar.
#
# No escribe nada en runs/.
set -u

REPO=/home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline
BASE=/mnt/2TB/MUSE_work/psffit_radius_sweep/manana_20260830
mkdir -p "$BASE"
cd "$REPO" || exit 1

echo "=== inicio $(date -Is) ===" | tee -a "$BASE/manana.log"
echo "commit: $(git rev-parse --short HEAD)" | tee -a "$BASE/manana.log"

python -u scripts/psffit_radius_sweep.py \
    --runs ROXs12b_realigned,ROXs42Bb_realigned \
    --pares 20x12,20x8,16x12 \
    --salida "$BASE" > "$BASE/sweep.log" 2>&1
echo "barrido rc=$? $(date -Is)" | tee -a "$BASE/manana.log"
echo "=== fin $(date -Is) ===" | tee -a "$BASE/manana.log"
