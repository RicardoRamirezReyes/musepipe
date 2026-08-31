#!/bin/bash
# Cadena completa de ROXs 42B b con la binaria en C1. DESATENDIDA.
#
# Resistencia a caida de red: se lanza con `setsid nohup ... </dev/null`, asi que
# el proceso queda huerfano de la terminal y sobrevive a que se caiga la sesion.
# Ninguna etapa usa red -verificado: no hay requests/urllib/astroquery en
# musepipe ni en los scripts que corren aqui-, y todos los datos son locales.
# No hay ningun punto que pida entrada por teclado.
#
# ORDEN, y por que este:
#   1. arbol limpio + espacio + foto previa.
#   2. suite RAPIDA como puerta. La lenta NO puede ir aqui: ejecuta el notebook
#      debug de C1, que recomputa la etapa con el codigo NUEVO y la compara con
#      el producto VIEJO -que aun no se ha regenerado-, asi que fallaria por
#      construccion y se perderia la noche sin motivo.
#   3. C1 -> C1b -> 04b..G5. C1b va explicito porque NO esta en la cadena de
#      `rerun_chain.py` y este run es `psf_scope=per_observation`: C3 saca
#      `optimal_psfsub` de `cube_psfsub_perobs.fits`, que produce C1b. Sin
#      re-correrlo, C3 extraeria de un cubo hecho con la PSF vieja.
#   4. suite COMPLETA al final, cuando los productos ya son nuevos: ahi si tiene
#      sentido que los notebooks debug comparen contra ellos.
set -u

REPO=/home/ricardo-ramirez/Offline_MUSE/MusePipeline/MUSE-accretion-pipeline
RUN=ROXs42Bb_realigned
SELLO=$(date +%Y%m%dT%H%M%S)
FOTO=/mnt/2TB/MUSE_work/${RUN}_pre_binaria_${SELLO}
LOG=/mnt/2TB/MUSE_work/cadena_42bb_${SELLO}
PY=$(command -v python)
mkdir -p "$LOG"
cd "$REPO" || exit 1

registra() { echo "$*" | tee -a "$LOG/noche.log"; }

registra "=== inicio $(date -Is) ==="
registra "commit: $(git rev-parse HEAD)"
registra "logs:   $LOG"

# --- 1 · guardas de arranque ------------------------------------------------
if [ -n "$(git status --porcelain)" ]; then
    registra "ABORTADO: el arbol tiene cambios sin commitear. Los productos de una"
    registra "          corrida tienen que ser trazables a un commit."
    exit 1
fi
LIBRE=$(df --output=avail -BG /mnt/2TB | tail -1 | tr -dc '0-9')
if [ "$LIBRE" -lt 40 ]; then
    registra "ABORTADO: solo ${LIBRE}G libres en /mnt/2TB; la foto previa son ~12G."
    exit 1
fi
cp -a "/mnt/2TB/MUSE_work/$RUN" "$FOTO" || { registra "ABORTADO: fallo la foto previa"; exit 1; }
registra "foto previa: $FOTO ($(du -sh "$FOTO" | cut -f1))"

# --- 2 · puerta: suite rapida ----------------------------------------------
"$PY" -u -m pytest tests/ -q -m "not slow" > "$LOG/suite_rapida.log" 2>&1
RC=$?
registra "suite rapida rc=$RC · $(tail -2 "$LOG/suite_rapida.log" | head -1)"
[ $RC -ne 0 ] && { registra "ABORTADO: suite en rojo, no se escriben productos."; exit 1; }

# --- 3 · la cadena ----------------------------------------------------------
etapa() {  # etapa <nombre> <comando...>
    local nombre="$1"; shift
    registra "--- $nombre  $(date -Is)"
    "$@" >> "$LOG/cadena.log" 2>&1
    local rc=$?
    registra "    $nombre rc=$rc"
    [ $rc -ne 0 ] && { registra "ABORTADO en $nombre. Foto previa intacta en $FOTO"; exit 1; }
    return 0
}

etapa C1  "$PY" -u -m musepipe.stages.stage_e01_psf --run-id "$RUN"
etapa C1b "$PY" -u -m musepipe.stages.stage_e01b_perobs_subtract --run-id "$RUN"
etapa "04b..G5" "$PY" -u scripts/rerun_chain.py --run-id "$RUN" --desde 04b

# --- 4 · suite completa, ya contra los productos nuevos ---------------------
"$PY" -u -m pytest tests/ -q > "$LOG/suite_completa.log" 2>&1
registra "suite completa rc=$? · $(tail -2 "$LOG/suite_completa.log" | head -1)"

registra "--- git status (los notebooks se commitean SIN salidas) ---"
git status --short | tee -a "$LOG/noche.log"
registra "=== fin $(date -Is) ==="
