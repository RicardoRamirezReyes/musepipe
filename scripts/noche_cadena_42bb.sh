#!/bin/bash
# Cadena completa de ROXs 42B b, desatendida. ESCRITO Y SIN LANZAR.
#
# Existe para el dia en que el ajuste de dos componentes de C1 este implementado
# y revisado (ver docs/2026-08-30_binaria_42b_sesga_la_apcorr.md §4). Lanzarlo
# HOY solo reproduciria los productos actuales: no hay ningun cambio de codigo
# que ejercitar.
#
# Tres guardas, en este orden:
#   1. foto previa del run -- los productos se sobreescriben y el tag
#      fondecyt-fig2-v3 depende de la tabla de E4.
#   2. la suite COMPLETA en verde. No se escriben productos cientificos con el
#      arbol en rojo.
#   3. la cadena, abortando en la primera etapa con rc != 0.
set -u

REPO=/home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline
RUN=ROXs42Bb_realigned
SELLO=$(date +%Y%m%d)
FOTO=/mnt/2TB/MUSE_work/${RUN}_pre_binaria_${SELLO}
LOG=/mnt/2TB/MUSE_work/cadena_42bb_${SELLO}
mkdir -p "$LOG"
cd "$REPO" || exit 1

echo "=== inicio $(date -Is) · commit $(git rev-parse --short HEAD) ===" | tee -a "$LOG/noche.log"

# 1 · foto previa
if [ ! -d "$FOTO" ]; then
    cp -a "/mnt/2TB/MUSE_work/$RUN" "$FOTO" || exit 1
    echo "foto previa en $FOTO" | tee -a "$LOG/noche.log"
fi

# 2 · la suite entera, con los notebooks lentos
python -m pytest tests/ -q > "$LOG/suite.log" 2>&1
RC=$?
tail -3 "$LOG/suite.log" | tee -a "$LOG/noche.log"
if [ $RC -ne 0 ]; then
    echo "SUITE EN ROJO (rc=$RC): la cadena NO se lanza." | tee -a "$LOG/noche.log"
    exit 1
fi

# 3 · la cadena. C1 y C1b van primero: todo lo demas cuelga del modelo de PSF.
python -u scripts/rerun_chain.py --run-id "$RUN" --desde C1 > "$LOG/cadena.log" 2>&1
echo "cadena rc=$? $(date -Is)" | tee -a "$LOG/noche.log"
tail -25 "$LOG/cadena.log" | tee -a "$LOG/noche.log"

echo "--- git status ---" | tee -a "$LOG/noche.log"
git status --short | tee -a "$LOG/noche.log"
echo "=== fin $(date -Is) ===" | tee -a "$LOG/noche.log"
