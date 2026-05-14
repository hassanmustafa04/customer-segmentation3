#!/usr/bin/env bash
# Create a virtual environment, install dependencies, and run the pipeline.
#
# By default the venv is ``.venv`` in the project root. If that path is not
# writable (some CI/sandbox setups), set ``CUSTOMER_SEG_VENV`` to a directory
# you can write to, e.g.:
#   export CUSTOMER_SEG_VENV="$HOME/.venvs/customer_segmentation"
#   bash scripts/setup_and_run.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_VENV="$ROOT/.venv"
VENVDIR="${CUSTOMER_SEG_VENV:-$DEFAULT_VENV}"

if [[ ! -d "$VENVDIR" ]]; then
  if ! python3 -m venv "$VENVDIR" 2>/dev/null; then
    _tmp="${TMPDIR:-/tmp}"
    _tmp="${_tmp%/}"
    FALLBACK="$_tmp/customer_segmentation_venv"
    echo "Note: could not create venv at $VENVDIR — using $FALLBACK instead."
    VENVDIR="$FALLBACK"
    python3 -m venv "$VENVDIR"
  fi
fi

echo "Using virtual environment: $VENVDIR"
"$VENVDIR/bin/python" -m pip install --upgrade pip
"$VENVDIR/bin/pip" install -r requirements.txt
"$VENVDIR/bin/python" main.py
echo ""
echo "Activate later with: source \"$VENVDIR/bin/activate\""
