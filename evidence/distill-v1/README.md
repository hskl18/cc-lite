# distill-v1: the first distilled checkpoint

`train/checkpoint.pt` is a 128-channel, 8-block policy-value network (3,860,347 parameters) trained by supervised distillation from Fairy-Stockfish labels.
It was produced on 2026-11-08 on one machine: AMD Ryzen 7 9800X3D, 62 GB RAM, NVIDIA GeForce RTX 5070 Ti (16 GB), Ubuntu 24.04 under WSL2, PyTorch 2.11.0 with CUDA 12.8.
This directory records how it was made and holds its three evaluation bundles.

## Labels come from Fairy-Stockfish, not Pikafish

The Pikafish NNUE license forbids commercial use without permission of its weights "and the weights file further derived from them", and a distilled checkpoint could count as derived, while this repository is Apache-2.0.
The same license states that the xiangqi networks the Pikafish developers train for Fairy-Stockfish are CC0.
The labels come from the Fairy-Stockfish build with the built-in network `xiangqi-c07e94a5c7cb`, from the `xiangqi-c07e94a5c7cb` release of `fairy-stockfish/Fairy-Stockfish-NNUE`, binary `fairy-stockfish_x86-64-avx512`, SHA-256 `086cc74c515599b1a421c57502ec05a12447230023088ca130731ecf30f47780`.
The engine is GPL-3.0 and is not distributed here; only its outputs are used.

## How it was made

1. Positions: Fairy-Stockfish played 12,000 games against itself at depth 2, with 15 percent of moves drawn uniformly at random from the legal moves, a 160-ply cap, and seed 1.
   That gave 1,012,264 positions, 848,307 of them unique after removing duplicates and the eight `paired-v1` opening positions, with 0 engine moves that were illegal under Profile 1.
   Record: [`data/positions.manifest.json`](data/positions.manifest.json).
2. Labels: every position was searched at depth 10, one thread, 16 MB hash, with the engine state cleared before each position, so a label depends only on its position.
   The best move is the policy target, and the score from the side to move, divided by 1,000 and clipped to [-1, 1], is the value target.
   Record: [`data/labels.manifest.json`](data/labels.manifest.json).
3. Training: `train.experiment` with [`configs/distill_gpu.yaml`](../../configs/distill_gpu.yaml), 6 epochs at batch 512 in bf16, 660 s on the GPU, from a clean checkout of `ed86761`.
   Record: [`train/manifest.json`](train/manifest.json) and [`train/metrics.json`](train/metrics.json).

```bash
export PYTHONPATH=src
FAIRY=~/engines/fairy/fairy-stockfish_x86-64-avx512
python -m train.engine_positions --engine "$FAIRY" --engine-flavor fairy --games 12000 --depth 2 \
  --random-move-rate 0.15 --max-plies 160 --seed 1 --workers 14 \
  --exclude-opening-suite configs/openings/paired-v1.json --output ~/runs/distill-v1/positions.jsonl
python -m train.distill_engine --engine "$FAIRY" --engine-flavor fairy --positions ~/runs/distill-v1/positions.jsonl \
  --output ~/runs/distill-v1/labels.jsonl --depth 10 --workers 14
python -m train.experiment run --config configs/distill_gpu.yaml --data ~/runs/distill-v1/labels.jsonl \
  --output-dir ~/runs/distill-v1/train
```

## Reading the numbers

The metrics in `train/metrics.json` are averages over all six training epochs on the training data, not held-out results.
Averaged that way, the network's top move matched the engine's in 24.9 percent of positions, and its unmasked top move was legal in 90.3 percent; MCTS masks illegal moves, so play never uses one.

The value targets are coarse.
Random moves make many games lopsided, and 374,457 of the 848,307 labels (44 percent) sit at the clip of plus or minus 1, so the value head learns little about the size of a large advantage.

## What is and is not here

The checkpoint, the training manifest and metrics, and the two data manifests are committed.
The positions and labels files (56 MB and 90 MB) are not in Git; their SHA-256 digests are in the manifests, and xz-compressed copies are attached to the [v0.3.0 release](https://github.com/hskl18/cc-lite/releases/tag/v0.3.0) as `positions.jsonl.xz` and `labels.jsonl.xz`.
The training bundle passed `train.experiment validate` on the training machine, before publication.
Before publication, the home directory in recorded paths was rewritten to `~/`; nothing else in these records was altered.
The validator reads the recorded paths literally and needs both the labels file and the config, so re-validating requires a local copy of `train/manifest.json` with the data and config paths pointing at those files.

## Evaluation

All three bundles were produced from a clean checkout of `989a423`, which added this checkpoint, and validate with `python -m eval.evidence validate`.

| Bundle | Settings | Result |
| --- | --- | --- |
| [`random`](random) | `configs/baseline_cpu.yaml`: CPU, 50 simulations, 20 games, colors alternating, 400-ply cap | 20 wins, 0 draws, 0 losses, 0 capped |
| [`material`](material) | the same | 19 wins, 0 draws, 1 loss, 0 capped |
| [`vs-debug`](vs-debug) | `configs/paired_eval.yaml`: GPU, 200 simulations, 6 sampled plies, seeds 1, 7, 19, `paired-v1` suite, up to 24 pairs | 43 wins, 1 draw, 0 losses over 22 complete pairs; score rate 0.989, 95 percent paired-bootstrap interval 0.966 to 1.0; stopped as superior |

The settings for `random` and `material` match `evidence/baseline-v1`, which evaluated the debug checkpoint on an Apple M3 Pro; these ran on the Ryzen CPU above, and floating-point ties can differ between machines.
The one loss against the material mover came as Black at ply 363, with no legal moves; the one draw against the debug checkpoint was a threefold repetition at ply 271.
All 44 paired games differ from one another, so the three seeds did not replay the same games.
The paired report also gives an Elo difference of 657 (533 to 780) by `logistic-elo-with-beta-half-continuity-correction`, a small-sample method; with a debug checkpoint as the opponent, that number says nothing about absolute strength.

```bash
export PYTHONPATH=src
CK=evidence/distill-v1/train/checkpoint.pt
python -m eval.evaluate --config configs/baseline_cpu.yaml --checkpoint $CK --opponent random --output-dir evidence/distill-v1/random
python -m eval.evaluate --config configs/baseline_cpu.yaml --checkpoint $CK --opponent material --output-dir evidence/distill-v1/material
python -m eval.head_to_head --config configs/paired_eval.yaml --checkpoint-a $CK \
  --checkpoint-b evidence/debug-v1/checkpoint/selfplay.pt --opening-suite configs/openings/paired-v1.json \
  --seeds 1,7,19 --min-pairs 8 --max-pairs 24 --output-dir evidence/distill-v1/vs-debug
```

