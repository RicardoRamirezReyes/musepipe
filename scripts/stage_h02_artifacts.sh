#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if python -c "import astropy" >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1 && python3 -c "import astropy" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif [[ -x /opt/local/bin/python3.11 ]] && /opt/local/bin/python3.11 -c "import astropy" >/dev/null 2>&1; then
    PYTHON_BIN="/opt/local/bin/python3.11"
  else
    PYTHON_BIN="python"
  fi
fi

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -c "from musepipe.stages.stage_h02_artifacts import main; main()" "$@"
