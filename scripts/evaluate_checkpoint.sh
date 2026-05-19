#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CONFIG CHECKPOINT [random|material]" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-}:src"
python -m eval.evaluate \
  --config "$1" \
  --checkpoint "$2" \
  --opponent "${3:-random}"

