#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 CONFIG CHECKPOINT ENGINE_COMMAND EVAL_FILE [DEPTH] [OUTPUT_DIR]" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-}:src"
python -m eval.evaluate \
  --config "$1" \
  --checkpoint "$2" \
  --opponent engine \
  --engine-command "$3" \
  --engine-eval-file "$4" \
  --engine-depth "${5:-4}" \
  --output-dir "${6:-runs/engine-evaluation}"
