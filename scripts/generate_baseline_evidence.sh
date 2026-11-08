#!/usr/bin/env bash
# Regenerates evidence/baseline-v1: the committed debug checkpoint with 50-simulation MCTS
# against the random and material baselines, 20 seeded games each on CPU.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

rm -rf evidence/baseline-v1

for opponent in random material; do
  python -m eval.evaluate \
    --config configs/baseline_cpu.yaml \
    --checkpoint evidence/debug-v1/checkpoint/selfplay.pt \
    --opponent "$opponent" \
    --output-dir "evidence/baseline-v1/$opponent"

  python -m eval.evidence validate --run-dir "evidence/baseline-v1/$opponent"
done
