#!/usr/bin/env bash
#
# validate_o2b.sh — validation helper for the O2a/O2b musepipe refactor.
#
# The O2b change is an import-only refactor: local copies of already-tested
# primitives (robust_sigma, nearest_channel_indices, nearest_channel_index)
# were replaced by imports from musepipe. The functions were verified
# numerically identical in isolation, so validation mainly confirms that the
# notebooks still import and run, and (optionally) that their deterministic
# products are byte-identical before vs after.
#
# IMPORTANT — RAM: these notebooks load the full 32-cube stack, whose size is
#   crop^2 * 3681 * 32 * 4 bytes  (~13.6 GB per stack at crop=170, ~1.2 GB at
#   crop=50). Notebooks like 02b load TWO stacks, and 04/06 build PCA matrices
#   on top. Validate on a SMALL-crop run, and run heavy notebooks ONE AT A TIME.
#   The crop=50 you set in 00_config only applies to runs you (re)generate with
#   00_config -> 01 -> 02; existing runs keep their frozen crop.
#
# Usage:
#   ./validate_o2b.sh imports                     # Level 0: import smoke test (no RAM)
#   ./validate_o2b.sh crop                         # list runs and their frozen crop_npix
#   ./validate_o2b.sh smoke <notebook> [RUN_ID]    # Level 1: headless-run ONE notebook
#   ./validate_o2b.sh diff  <notebook> [RUN_ID]    # Level 2: before/after product diff
#
# Run from the repo root, inside the MUSE conda env:
#   conda activate MUSE && ./validate_o2b.sh imports
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# Notebooks touched by O2a (02b/02c) and O2b (the rest), lightest first.
CHANGED_NBS=(
  00_config.ipynb                                  # no cube load — always safe
  02c_define_xcorr_stripe_ranges_template.ipynb
  05_fluxes_and_linewidths.ipynb
  09_integration_time_flux_projection.ipynb
  01b_align_test.ipynb
  02b_xcorr_stripe_diagnostics.ipynb               # loads TWO stacks
  04c_airy_ring_moffat_diagnostics.ipynb
  04c_fakecont_airy_ring_moffat_diagnostics.ipynb
  04d_pca_residual_moffat_diagnostics.ipynb
  04_pca_and_cube_selection.ipynb                  # heavy PCA
  04_low_memory_pca_and_cube_selection.ipynb
  06_inject_halpha_signal.ipynb                    # heavy PCA
  06_low_memory_inject_halpha_signal.ipynb
  06c_pca_c2_multiline_40.ipynb
)

green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }

cmd_imports() {
  echo "== Level 0: import smoke test =="
  python - <<'PY'
import importlib, sys
mods = {
    "musepipe": [],
    "musepipe.stats": ["robust_sigma", "robust_limits"],
    "musepipe.spectral": ["nearest_channel_indices", "nearest_channel_index",
                           "make_wavelength_mask"],
    "musepipe.apertures": ["box_spectrum_sum", "box_spectrum_mean"],
    "musepipe.stripes": ["xcorr_shift_map", "_xcorr_shift_pixels",
                         "_build_stripe_geometry"],
    "musepipe.io": ["write_csv"],
}
ok = True
for m, names in mods.items():
    try:
        mod = importlib.import_module(m)
        for n in names:
            getattr(mod, n)
    except Exception as e:
        ok = False
        print(f"  FAIL  {m}: {e}")
    else:
        print(f"  ok    {m}" + (f"  ({', '.join(names)})" if names else ""))
import musepipe
print("  musepipe ->", musepipe.__file__)
sys.exit(0 if ok else 1)
PY
  rc=$?
  if [ $rc -eq 0 ]; then green "Level 0 PASSED — musepipe imports resolve."; else red "Level 0 FAILED — launch Jupyter from the repo root / check the MUSE env."; fi
  return $rc
}

cmd_crop() {
  echo "== per run: config crop_npix vs ACTUAL stage01 cube on disk =="
  echo "   (the on-disk stack is what gets loaded into RAM — trust that column)"
  python - <<'PY'
import json, glob, os, struct

def cube_shape(path):
    """Return the shape (all NAXISk) of the first HDU with NAXIS>=3, no astropy."""
    try:
        with open(path, "rb") as fh:
            naxis = None; axes = {}
            while True:
                block = fh.read(2880)
                if not block:
                    break
                for i in range(0, len(block), 80):
                    card = block[i:i+80].decode("latin1")
                    key = card[:8].strip()
                    if key == "NAXIS": naxis = int(card.split("=")[1].split("/")[0])
                    elif key.startswith("NAXIS") and key[5:].isdigit():
                        axes[int(key[5:])] = int(card.split("=")[1].split("/")[0])
                    if card.startswith("END"):
                        if naxis and naxis >= 3:
                            return [axes[k] for k in sorted(axes)]
                        naxis = None; axes = {}  # next HDU
    except Exception:
        pass
    return None

for cfgp in sorted(glob.glob("runs/*/config/config.json")):
    run = cfgp.split("/")[1]
    try:
        crop = int(json.load(open(cfgp))["config"].get("crop_npix", 0))
    except Exception:
        crop = 0
    cube = f"runs/{run}/stages/stage01_cropped_cube_stack.fits"
    if os.path.exists(cube):
        disk = os.path.getsize(cube) / 1e9
        shp = cube_shape(cube)
        if shp:
            elems = 1
            for a in shp: elems *= a
            ram = elems * 4 / 1e9  # loaded as float32
            px = shp[0]  # NAXIS1 = x (spatial)
            flag = "  <- config MISMATCH" if (crop and px != crop) else ""
            dims = "x".join(str(a) for a in reversed(shp))
            print(f"  {run:34s} cfg={crop:4d} | cube {dims:>16s} | ~{ram:5.1f} GB in RAM (disk {disk:4.1f} GB){flag}")
        else:
            print(f"  {run:34s} cfg={crop:4d} | (unreadable cube)")
    else:
        print(f"  {run:34s} cfg={crop:4d} | (no stage01 cube)")
print("\n  'in RAM' is what matters (compressed on disk expands when loaded). 02b loads TWO; 04/06 add PCA.")
print("  Cheapest validation: a single-cube run (ncubes=1). Point a notebook at one with")
print("  './validate_o2b.sh smoke <nb> <RUN_ID>' (overrides the notebook's RUN_ID on a temp copy).")
PY
}

# Write a temp copy of a notebook, optionally overriding its RUN_ID assignment.
# Echoes the path of the copy to use.
prepare_nb() {
  local src="$1" dst="$2" run_id="${3:-}"
  python - "$src" "$dst" "$run_id" <<'PY'
import json, re, sys
src, dst, run_id = sys.argv[1], sys.argv[2], sys.argv[3]
nb = json.load(open(src))
if run_id:
    pat = re.compile(r'^(\s*RUN_ID\s*=\s*)["\'][^"\']*["\']')
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        c["source"] = [pat.sub(rf'\g<1>"{run_id}"', ln) for ln in c["source"]]
json.dump(nb, open(dst, "w"), indent=1, ensure_ascii=False)
PY
}

# Headless-execute a notebook copy so the original .ipynb on disk is untouched.
# Optional 3rd arg overrides the notebook's RUN_ID (to force a small run).
# The copy is executed FROM THE REPO ROOT so that `import musepipe` and the
# relative `runs/<RUN_ID>/...` paths resolve (nbconvert runs the kernel in the
# notebook's own directory).
run_nb() {
  local src="$1" out="$2" run_id="${3:-}"
  local prepped="$REPO/.o2b_run_$$.ipynb"
  prepare_nb "$src" "$prepped" "$run_id" || { rm -f "$prepped"; return 1; }
  ( cd "$REPO" && jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=-1 \
      --output "$(basename "$out")" --output-dir "$(dirname "$out")" \
      "$prepped" ) >"${out}.log" 2>&1
  local rc=$?
  rm -f "$prepped"
  return $rc
}

# Echo "<gb> <dims>" for a run's ACTUAL stage01 cube (float32 in-RAM size),
# e.g. "12.1 32x3681x160x160". Empty if the cube is missing/unreadable.
stack_info() {
  python - "$1" <<'PY' 2>/dev/null
import sys, os
run = sys.argv[1]
path = f"runs/{run}/stages/stage01_cropped_cube_stack.fits"
if not os.path.exists(path):
    sys.exit(0)
naxis=None; axes={}
with open(path,"rb") as fh:
    while True:
        block=fh.read(2880)
        if not block: break
        for i in range(0,len(block),80):
            card=block[i:i+80].decode("latin1"); key=card[:8].strip()
            if key=="NAXIS": naxis=int(card.split("=")[1].split("/")[0])
            elif key.startswith("NAXIS") and key[5:].isdigit(): axes[int(key[5:])]=int(card.split("=")[1].split("/")[0])
            if card.startswith("END"):
                if naxis and naxis>=3:
                    shp=[axes[k] for k in sorted(axes)]
                    elems=1
                    for a in shp: elems*=a
                    print(f"{elems*4/1e9:.1f} " + "x".join(str(a) for a in reversed(shp)))
                    sys.exit(0)
                naxis=None; axes={}
PY
}

cmd_smoke() {
  local nb="${1:-}" run_id="${2:-}"
  command -v jupyter >/dev/null || { red "jupyter not found — activate the MUSE env."; return 1; }
  if [ -z "$nb" ]; then
    red "Run ONE notebook at a time to control RAM. Usage:"
    echo "  ./validate_o2b.sh smoke <notebook> [RUN_ID]"
    echo
    echo "Changed notebooks (lightest first):"
    printf '  %s\n' "${CHANGED_NBS[@]}"
    echo
    yellow "Start with 00_config.ipynb (loads no cubes), then the light ones."
    return 2
  fi
  [ -f "$nb" ] || { red "notebook not found: $nb"; return 2; }

  # Warn about RAM based on the run this notebook points at.
  local detected; detected="$(grep -oE 'RUN_ID\s*=\s*"[^"]+"' "$nb" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
  local rid="${run_id:-$detected}"
  if [ -n "$rid" ]; then
    local info; info="$(stack_info "$rid")"
    if [ -n "$info" ]; then
      local gb="${info%% *}" dims="${info##* }"
      yellow "RUN_ID='$rid' -> stage01 cube ${dims}, ~${gb} GB/stack in RAM (02b loads two)."
      if awk "BEGIN{exit !($gb>=4)}"; then red "  Heavy (>=4 GB/stack). Prefer a single-cube run (./validate_o2b.sh crop)."; fi
    fi
  fi

  [ -n "$run_id" ] && yellow "Overriding RUN_ID -> '$run_id' on the temp copy."
  local tmp=/tmp/o2b_smoke; mkdir -p "$tmp"
  printf 'running %-52s ' "$nb"
  if run_nb "$nb" "$tmp/${nb%.ipynb}.exec.ipynb" "$run_id"; then
    green "OK"; return 0
  else
    red "FAIL  (see $tmp/${nb%.ipynb}.exec.ipynb.log)"
    tail -n 20 "$tmp/${nb%.ipynb}.exec.ipynb.log" 2>/dev/null
    return 1
  fi
}

snapshot() {
  local run_id="$1" dest="$2"
  rm -rf "$dest"; mkdir -p "$dest"
  find "runs/$run_id" \( -name '*.csv' -o -name '*qc*.json' -o -name '*_qc.json' \) -type f 2>/dev/null | while read -r f; do
    local rel="${f#runs/$run_id/}"
    mkdir -p "$dest/$(dirname "$rel")"
    cp "$f" "$dest/$rel"
  done
}

cmd_diff() {
  local nb="${1:-}" run_id="${2:-}"
  [ -n "$nb" ] || { red "usage: ./validate_o2b.sh diff <notebook> [RUN_ID]"; return 2; }
  [ -f "$nb" ] || { red "notebook not found: $nb"; return 2; }
  command -v jupyter >/dev/null || { red "jupyter not found — activate the MUSE env."; return 1; }
  if [ -z "$run_id" ]; then
    run_id="$(grep -oE 'RUN_ID\s*=\s*"[^"]+"' "$nb" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
    [ -n "$run_id" ] && yellow "Detected RUN_ID='$run_id' from the notebook."
  fi
  [ -n "$run_id" ] || { red "could not detect RUN_ID; pass it explicitly."; return 2; }
  [ -d "runs/$run_id" ] || { red "runs/$run_id not found."; return 2; }
  local info; info="$(stack_info "$run_id")"
  if [ -n "$info" ]; then
    local gb="${info%% *}"
    awk "BEGIN{exit !($gb>=4)}" && red "Warning: ~${gb} GB/stack — heavy RAM. A single-cube run is strongly recommended."
  fi

  local tmp=/tmp/o2b_diff; mkdir -p "$tmp"
  local old_nb="$tmp/OLD_${nb}"
  yellow "Extracting pre-refactor baseline from git HEAD..."
  if ! git show "HEAD:$nb" > "$old_nb" 2>/dev/null; then
    red "git show HEAD:$nb failed (is $nb tracked and committed at baseline?)."; return 2
  fi

  yellow "This overwrites runs/$run_id deterministic products (regenerable). Ctrl-C to abort."; sleep 3

  echo "== Running OLD (git HEAD) =="
  run_nb "$old_nb" "$tmp/old.exec.ipynb" "$run_id" || { red "OLD run failed — see $tmp/old.exec.ipynb.log"; return 1; }
  snapshot "$run_id" "$tmp/old_products"

  echo "== Running NEW (working tree) =="
  run_nb "$nb" "$tmp/new.exec.ipynb" "$run_id" || { red "NEW run failed — see $tmp/new.exec.ipynb.log"; return 1; }
  snapshot "$run_id" "$tmp/new_products"

  echo "== diff (old vs new deterministic products) =="
  if diff -r "$tmp/old_products" "$tmp/new_products"; then
    green "IDENTICAL — products are byte-for-byte equal before vs after the refactor."
    return 0
  else
    red "DIFFERENCES found (see above). Investigate before trusting the migration."
    return 1
  fi
}

case "${1:-}" in
  imports) cmd_imports ;;
  crop)    cmd_crop ;;
  smoke)   shift; cmd_smoke "${1:-}" "${2:-}" ;;
  diff)    shift; cmd_diff "${1:-}" "${2:-}" ;;
  *) echo "usage: $0 {imports | crop | smoke <notebook> [RUN_ID] | diff <notebook> [RUN_ID]}"; exit 2 ;;
esac
