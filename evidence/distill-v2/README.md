# distill-v2: fewer random moves and smoother value targets

`train/checkpoint.pt` has the same architecture as [`distill-v1`](../distill-v1) (128 channels, 8 blocks) and is trained the same way, on new data that changes two things.

| | distill-v1 | distill-v2 |
| --- | --- | --- |
| Engine games | 12,000, seed 1 | 20,000, seed 11 |
| Random-move rate | 15 percent | 5 percent |
| Unique positions | 848,307 | 883,077 |
| Value target | score / 1,000, clipped | tanh(score / 800) |
| Labels at plus or minus 0.99 or beyond | 374,457 (44 percent) | 108,913 (12 percent), 99,981 of them mates |

Fewer random moves leave fewer lopsided positions, and the tanh mapping keeps large leads apart instead of clipping them.
The scale of 800 centipawns was chosen from the 22,314 held-out labels that are not mate scores, whose median absolute score is 853 and 90th percentile 1,487.
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
The positions and labels files are not in Git; their digests are in the manifests, and the home directory in recorded paths was rewritten to `~/` before publication.
Evaluation results are added in their own bundles.
