#!/bin/bash
# Trabajo desatendido de la noche 2026-08-30. Dos jobs en secuencia:
#   1. barrido de radios de psffit con completitud inyectada (~4 h)
#   2. suite completa CON `slow`, los 20 notebooks debug incluidos (~1 h 16)
# El barrido va primero: es lo que no existe todavia, y si algo se pasa de
# tiempo prefiero perder la suite (que ya paso anoche) y no la medida.
# No escribe nada en runs/ ni en el repo: solo CSV y logs en $BASE.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MUSE_WORK:?define MUSE_WORK: el directorio de trabajo externo, fuera del repo}"
BASE="$MUSE_WORK/psffit_radius_sweep/noche_20260830"
mkdir -p "$BASE"
cd "$REPO" || exit 1

echo "=== inicio $(date -Is) ===" | tee -a "$BASE/noche.log"
echo "commit: $(git rev-parse --short HEAD)" | tee -a "$BASE/noche.log"

# -u obligatorio: sin el, el log queda vacio hasta el final y no se puede
# estimar avance (leccion del traspaso del 08-29).
# Cinco configuraciones, no una rejilla simetrica: `psffit` cuesta ~35 s por caso
# en serie (medido en la prueba de humo), asi que lo que cabe hay que gastarlo en
# los puntos donde el barrido del 08-29 DISCREPABA entre objetos.
#   20x12  produccion, la referencia. Va primera por si la noche se corta.
#   20x8   optimo de ROXs 12 b (+5.09) y lo PEOR de ROXs 42B b (-0.01)
#   16x12  optimo de ROXs 42B b (+1.49) y de los peores de ROXs 12 b
#   20x16  segundo de ROXs 42B b (+1.37) y el peor de ROXs 12 b (+1.91)
#   28x12  segundo de ROXs 12 b (+4.26) y de los peores de ROXs 42B b (+0.37)
python -u scripts/psffit_radius_sweep.py \
    --runs ROXs12b_realigned,ROXs42Bb_realigned \
    --pares 20x12,20x8,16x12,20x16,28x12 \
    --salida "$BASE" > "$BASE/sweep.log" 2>&1
echo "barrido rc=$? $(date -Is)" | tee -a "$BASE/noche.log"

python -m pytest tests/ -q > "$BASE/suite.log" 2>&1
echo "suite rc=$? $(date -Is)" | tee -a "$BASE/noche.log"
tail -3 "$BASE/suite.log" | tee -a "$BASE/noche.log"

# El test lento ejecuta los notebooks. El 08-29 uno aparecio despues con 1.87 MB
# de figuras embebidas y el mecanismo no esta fijado, asi que el estado del arbol
# se deja registrado en vez de descubrirlo al commitear.
echo "--- git status tras la suite ---" | tee -a "$BASE/noche.log"
git status --short | tee -a "$BASE/noche.log"
echo "=== fin $(date -Is) ===" | tee -a "$BASE/noche.log"
