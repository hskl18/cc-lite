#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 CONFIG CHECKPOINT ENGINE_COMMAND [DEPTH] [OUTPUT_DIR]" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-}:src"
python -m eval.evaluate \
  --config "$1" \
  --checkpoint "$2" \
  --opponent engine \
  --engine-command "$3" \
  --engine-depth "${4:-4}" \
  --output-dir "${5:-runs/engine-evaluation}"
