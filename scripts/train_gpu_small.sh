#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
mkdir -p runs/gpu_small/checkpoints

python -m train.self_play \
  --config configs/gpu_small.yaml \
  --output runs/gpu_small/selfplay.jsonl \
  --train \
  --checkpoint-out runs/gpu_small/checkpoints/latest.pt

python -m eval.evaluate \
  --config configs/gpu_small.yaml \
  --checkpoint runs/gpu_small/checkpoints/latest.pt \
  --opponent material

