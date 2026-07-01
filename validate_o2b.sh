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
# Usage:
#   ./validate_o2b.sh imports              # Level 0: import smoke test (fast)
#   ./validate_o2b.sh smoke  [RUN_ID]      # Level 1: headless-run changed notebooks
#   ./validate_o2b.sh diff  <notebook> [RUN_ID]   # Level 2: before/after product diff
#
# Run from the repository root, inside the MUSE conda env:
#   conda activate MUSE && ./validate_o2b.sh imports
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# Notebooks touched by O2a (02b/02c) and O2b (the rest).
CHANGED_NBS=(
  00_config.ipynb
  02b_xcorr_stripe_diagnostics.ipynb
  02c_define_xcorr_stripe_ranges_template.ipynb
  01b_align_test.ipynb
  04_pca_and_cube_selection.ipynb
  04_low_memory_pca_and_cube_selection.ipynb
  04c_airy_ring_moffat_diagnostics.ipynb
  04c_fakecont_airy_ring_moffat_diagnostics.ipynb
  04d_pca_residual_moffat_diagnostics.ipynb
  05_fluxes_and_linewidths.ipynb
  06_inject_halpha_signal.ipynb
  06_low_memory_inject_halpha_signal.ipynb
  06c_pca_c2_multiline_40.ipynb
  09_integration_time_flux_projection.ipynb
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

# Headless-execute a notebook copy so the original .ipynb on disk is untouched.
# Returns 0 on success. Writes the executed copy + log under /tmp/o2b_smoke.
run_nb() {
  local nb="$1" src="$2" out="$3"
  jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output "$out" --output-dir "$(dirname "$out")" \
    "$src" >"${out}.log" 2>&1
}

cmd_smoke() {
  local run_id="${1:-}"
  command -v jupyter >/dev/null || { red "jupyter not found — activate the MUSE env."; return 1; }
  local tmp=/tmp/o2b_smoke; mkdir -p "$tmp"
  echo "== Level 1: headless run of ${#CHANGED_NBS[@]} changed notebook(s) =="
  [ -n "$run_id" ] && yellow "Note: notebooks use their own hard-coded RUN_ID; '$run_id' is only a reminder for you."
  local fails=0
  for nb in "${CHANGED_NBS[@]}"; do
    [ -f "$nb" ] || { yellow "  skip (missing): $nb"; continue; }
    printf '  running %-52s ' "$nb"
    if run_nb "$nb" "$nb" "$tmp/${nb%.ipynb}.exec.ipynb"; then
      green "OK"
    else
      red "FAIL  (see $tmp/${nb%.ipynb}.exec.ipynb.log)"
      fails=$((fails+1))
    fi
  done
  echo
  if [ $fails -eq 0 ]; then green "Level 1 PASSED — all notebooks executed without error."
  else red "Level 1: $fails notebook(s) failed — inspect the .log files in $tmp"; fi
  return $fails
}

# Snapshot small deterministic products (csv + qc json) under a run into a dir.
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

  local tmp=/tmp/o2b_diff; mkdir -p "$tmp"
  local old_nb="$tmp/OLD_${nb}"
  yellow "Extracting pre-refactor baseline from git HEAD..."
  if ! git show "HEAD:$nb" > "$old_nb" 2>/dev/null; then
    red "git show HEAD:$nb failed (is $nb tracked and committed at baseline?)."; return 2
  fi

  yellow "This overwrites runs/$run_id deterministic products (regenerable). Ctrl-C to abort."; sleep 3

  echo "== Running OLD (git HEAD) =="
  run_nb "$nb" "$old_nb" "$tmp/old.exec.ipynb" || { red "OLD run failed — see $tmp/old.exec.ipynb.log"; return 1; }
  snapshot "$run_id" "$tmp/old_products"

  echo "== Running NEW (working tree) =="
  run_nb "$nb" "$nb" "$tmp/new.exec.ipynb" || { red "NEW run failed — see $tmp/new.exec.ipynb.log"; return 1; }
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
  smoke)   shift; cmd_smoke "${1:-}" ;;
  diff)    shift; cmd_diff "${1:-}" "${2:-}" ;;
  *) echo "usage: $0 {imports | smoke [RUN_ID] | diff <notebook> [RUN_ID]}"; exit 2 ;;
esac
