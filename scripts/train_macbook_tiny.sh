#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
mkdir -p runs/macbook_tiny/checkpoints

python -m train.self_play \
  --config configs/macbook_tiny.yaml \
  --output runs/macbook_tiny/selfplay.jsonl \
  --train \
  --checkpoint-out runs/macbook_tiny/checkpoints/latest.pt

python -m eval.evaluate \
  --config configs/macbook_tiny.yaml \
  --checkpoint runs/macbook_tiny/checkpoints/latest.pt \
  --opponent random \
  --output-dir runs/macbook_tiny/evaluation-random
