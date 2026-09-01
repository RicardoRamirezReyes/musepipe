#!/bin/bash
# Tercera tanda del 2026-08-30 (~55 min). Completa ROXs 12 b con las dos
# configuraciones que la segunda tanda no volco: `20x16` y `28x12`.
#
# Por que solo ROXs 12 b: el remuestreo de 200 particiones dejo a `16x12` ganando
# a produccion en los DOS modos de umbral (70 % y 90 %), asi que es el unico
# candidato vivo de los dos objetos — y para recomendarlo hay que verlo contra
# las cinco, no contra dos. En ROXs 42B b el candidato `20x8` no sobrevivio al
# remuestreo, asi que ampliarlo ahi no compra nada.
set -u

REPO=/home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline
BASE=/mnt/2TB/MUSE_work/psffit_radius_sweep/tarde_20260830
mkdir -p "$BASE"
cd "$REPO" || exit 1

echo "=== inicio $(date -Is) ===" | tee -a "$BASE/tarde.log"
python -u scripts/psffit_radius_sweep.py \
    --runs ROXs12b_realigned \
    --pares 20x16,28x12 \
    --salida "$BASE" > "$BASE/sweep.log" 2>&1
echo "barrido rc=$? $(date -Is)" | tee -a "$BASE/tarde.log"
