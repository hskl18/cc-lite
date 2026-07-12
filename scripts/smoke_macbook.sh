#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
mkdir -p runs/research_debug/checkpoints

python -m train.self_play \
  --config configs/research_debug.yaml \
  --output runs/research_debug/selfplay.jsonl \
  --train \
  --checkpoint-out runs/research_debug/checkpoints/selfplay.pt

python -m eval.evaluate \
  --config configs/research_debug.yaml \
  --checkpoint runs/research_debug/checkpoints/selfplay.pt \
  --opponent random \
  --output-dir runs/research_debug/evaluation-random
