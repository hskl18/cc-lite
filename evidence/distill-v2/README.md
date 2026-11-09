# distill-v2: fewer random moves and smoother value targets

`train/checkpoint.pt` has the same architecture as [`distill-v1`](../distill-v1) (128 channels, 8 blocks) and is trained the same way, on new data that changes two things.
It did not turn out stronger; see [Evaluation](#evaluation).

| | distill-v1 | distill-v2 |
| --- | --- | --- |
| Engine games | 12,000, seed 1 | 20,000, seed 11 |
| Random-move rate | 15 percent | 5 percent |
| Unique positions | 848,307 | 883,077 |
| Value target | score / 1,000, clipped | tanh(score / 800) |
| Labels at plus or minus 0.99 or beyond | 374,457 (44 percent) | 108,913 (12 percent), 99,981 of them mates |

Fewer random moves leave fewer lopsided positions, and the tanh mapping keeps large leads apart instead of clipping them.
The scale of 800 centipawns was chosen by looking at the 22,314 held-out labels that are not mate scores, whose median absolute score is 853 and 90th percentile 1,487.
The 783,096 non-mate training labels give 856 and 1,491, so the training data would have led to the same scale, but held-out value results are not an independent test of that choice.
Both changes landed together, so a difference between the two checkpoints cannot be credited to either one alone.

## How it was made

The same tools and engine build as `distill-v1`, on the RTX 5070 Ti machine described there.
Positions took 402 s; labels took 4,200 s with 14 engine processes, slower than `distill-v1` partly because other jobs shared the CPU; training took 392 s from a clean checkout of `98877f0`.

```bash
export PYTHONPATH=src
FAIRY=~/engines/fairy/fairy-stockfish_x86-64-avx512
python -m train.engine_positions --engine "$FAIRY" --engine-flavor fairy --games 20000 --depth 2 \
  --random-move-rate 0.05 --max-plies 160 --seed 11 --workers 14 \
  --exclude-opening-suite configs/openings/paired-v1.json --output ~/runs/distill-v2/positions.jsonl
python -m train.distill_engine --engine "$FAIRY" --engine-flavor fairy --positions ~/runs/distill-v2/positions.jsonl \
  --output ~/runs/distill-v2/labels.jsonl --depth 10 --workers 14 --value-mapping tanh --value-scale 800
python -m train.experiment run --config configs/distill_gpu.yaml --data ~/runs/distill-v2/labels.jsonl \
  --output-dir ~/runs/distill-v2/train
```

Records: [`data/positions.manifest.json`](data/positions.manifest.json), [`data/labels.manifest.json`](data/labels.manifest.json), [`train/manifest.json`](train/manifest.json), and [`train/metrics.json`](train/metrics.json).
The training metrics are six-epoch averages on the training data (top move matched the engine's 24.6 percent of the time), not held-out results.
The positions and labels files are not in Git; their digests are in the manifests, xz-compressed copies are attached to the [v0.4.0 release](https://github.com/hskl18/cc-lite/releases/tag/v0.4.0) as `distill-v2-positions.jsonl.xz` and `distill-v2-labels.jsonl.xz`, and the home directory in recorded paths was rewritten to `~/` before publication.
## Evaluation

The `random`, `material`, and `vs-v1` bundles were produced from a clean checkout of `070c0fb`, which added this checkpoint, and the Pikafish bundles from `321c289`; all six validate with `python -m eval.evidence validate`.

| Bundle | Settings | Result |
| --- | --- | --- |
| [`random`](random) | `configs/baseline_cpu.yaml`: CPU, 50 simulations, 20 games, colors alternating, 400-ply cap | 20 wins, 0 draws, 0 losses, 0 capped |
| [`material`](material) | the same | 20 wins, 0 draws, 0 losses, 0 capped |
| [`vs-v1`](vs-v1) | `configs/paired_eval.yaml`: GPU, 200 simulations, 6 sampled plies, seeds 1, 7, 19, `paired-v1` suite, up to 24 pairs, against `distill-v1` | 14 wins, 1 draw, 33 losses over 24 complete pairs; score rate 0.302, 95 percent paired-bootstrap interval 0.167 to 0.448; sequential rule inconclusive at the 24-pair limit |
| [`pikafish-depth-1`](pikafish-depth-1) | `configs/engine_eval.yaml`: GPU, 200 simulations, 6 sampled plies, 20 games, colors alternating, 400-ply cap, Pikafish at depth 1 | 4 wins, 2 draws, 14 losses, 0 capped |
| [`pikafish-depth-2`](pikafish-depth-2) | the same at depth 2 | 0 wins, 0 draws, 20 losses, 0 capped |
| [`pikafish-depth-4`](pikafish-depth-4) | the same at depth 4 | 0 wins, 1 draw, 19 losses, 0 capped |

`distill-v2` did not beat `distill-v1`.
It lost the paired comparison: the bootstrap interval lies below 0.5, while the anytime-valid sequential rule, which is much more conservative, reached no decision within 24 pairs.
It has the better record against Pikafish at depth 1, but two of those four wins are perpetual-check adjudications against the engine, and 20 games cannot separate 5 of 20 points from `distill-v1`'s 1 of 20.
The Pikafish build is the one named in [`evidence/distill-v1`](../distill-v1).

```bash
export PYTHONPATH=src
CK=evidence/distill-v2/train/checkpoint.pt
python -m eval.evaluate --config configs/baseline_cpu.yaml --checkpoint $CK --opponent random --output-dir evidence/distill-v2/random
python -m eval.evaluate --config configs/baseline_cpu.yaml --checkpoint $CK --opponent material --output-dir evidence/distill-v2/material
python -m eval.head_to_head --config configs/paired_eval.yaml --checkpoint-a $CK \
  --checkpoint-b evidence/distill-v1/train/checkpoint.pt --opening-suite configs/openings/paired-v1.json \
  --seeds 1,7,19 --min-pairs 8 --max-pairs 24 --output-dir evidence/distill-v2/vs-v1
python -m eval.evaluate --config configs/engine_eval.yaml --checkpoint $CK --opponent engine --engine-command ./pikafish \
  --engine-flavor pikafish --engine-eval-file ./pikafish.nnue --engine-depth 1 --output-dir evidence/distill-v2/pikafish-depth-1
```
