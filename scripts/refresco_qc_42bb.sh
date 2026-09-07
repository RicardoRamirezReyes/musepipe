#!/bin/bash
# Refresco del QC de C1 en ROXs 42B b tras el arreglo del centroide. DESATENDIDO.
#
# Solo C1 y F1. El arreglo toca un campo DERIVADO del QC (`centroid_vs_b3`, que
# ahora reconstruye el fotocentro) y una columna del CSV (`flux_ratio`): el
# AJUSTE no cambia, asi que `psf_model.json` tiene que salir con el MISMO
# sha256. Si no, hay algo no determinista y eso es lo que habria que mirar --
# el script lo comprueba y lo deja escrito.
#
# Por eso C1b y el bloque C-G NO se re-corren: consumen el modelo, y el modelo
# no cambia.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MUSE_WORK:?define MUSE_WORK: el directorio de trabajo externo, fuera del repo}"
RUN=ROXs42Bb_realigned
SELLO=$(date +%Y%m%dT%H%M%S)
LOG="$MUSE_WORK/refresco_qc_${SELLO}"
PY=$(command -v python)
mkdir -p "$LOG"
cd "$REPO" || exit 1
registra() { echo "$*" | tee -a "$LOG/refresco.log"; }

registra "=== inicio $(date -Is) · commit $(git rev-parse --short HEAD) ==="
if [ -n "$(git status --porcelain)" ]; then
    registra "ABORTADO: arbol sucio; los productos tienen que ser trazables a un commit."; exit 1
fi

ANTES=$("$PY" -c "import hashlib;print(hashlib.sha256(open('runs/$RUN/stages/psf_model.json','rb').read()).hexdigest())")
registra "sha256 de psf_model.json ANTES: $ANTES"

"$PY" -u -m pytest tests/ -q -m "not slow" > "$LOG/suite_rapida.log" 2>&1
RC=$?; registra "suite rapida rc=$RC · $(tail -2 "$LOG/suite_rapida.log" | head -1)"
[ $RC -ne 0 ] && { registra "ABORTADO: suite en rojo."; exit 1; }

registra "--- C1  $(date -Is)"
"$PY" -u -m musepipe.stages.stage_e01_psf --run-id "$RUN" > "$LOG/c1.log" 2>&1
RC=$?; registra "    C1 rc=$RC  $(date -Is)"
[ $RC -ne 0 ] && { registra "ABORTADO en C1."; exit 1; }

DESPUES=$("$PY" -c "import hashlib;print(hashlib.sha256(open('runs/$RUN/stages/psf_model.json','rb').read()).hexdigest())")
registra "sha256 de psf_model.json DESPUES: $DESPUES"
if [ "$ANTES" = "$DESPUES" ]; then
    registra "  IDENTICO -- el ajuste no cambio, como debia."
else
    registra "  *** DISTINTO *** El modelo cambio sin que el ajuste lo hiciera:"
    registra "  algo NO ES DETERMINISTA. Mirar esto ANTES de creerse el resto."
fi

registra "--- F1  $(date -Is)"
"$PY" -u scripts/build_report.py --run-id "$RUN" > "$LOG/f1.log" 2>&1
registra "    F1 rc=$?  $(date -Is)"

"$PY" - <<'PY' 2>&1 | tee -a "$LOG/refresco.log"
import json, collections
q=json.load(open("runs/ROXs42Bb_realigned/stages/stage_e01_qc.json"))
print(f"  centroid_vs_b3_max_diff_px = {q['smoothing'].get('centroid_vs_b3_max_diff_px')}  (limite 0.3; antes 0.3233; esperado ~0.28)")
s=json.load(open("runs/ROXs42Bb_realigned/report/run_summary.json"))
c=collections.Counter(i.get("priority") for i in s.get("open_issues",[]))
print(f"  F1: {dict(c)}   (antes 6 blocking; esperado 5)")
PY
registra "=== fin $(date -Is) ==="
