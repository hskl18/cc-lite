#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

rm -rf evidence/debug-v1 runs/evidence-debug-v1
mkdir -p runs/evidence-debug-v1/checkpoints

python -m train.self_play \
  --config configs/research_debug.yaml \
  --output runs/evidence-debug-v1/selfplay.jsonl \
  --train \
  --checkpoint-out runs/evidence-debug-v1/checkpoints/selfplay.pt

for opponent in random material; do
  python -m eval.evaluate \
    --config configs/research_debug.yaml \
    --checkpoint runs/evidence-debug-v1/checkpoints/selfplay.pt \
    --opponent "$opponent" \
    --games 2 \
    --simulations 4 \
    --max-plies 20 \
    --output-dir "evidence/debug-v1/$opponent"

  python -m eval.evidence validate --run-dir "evidence/debug-v1/$opponent"
done
