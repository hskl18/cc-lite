# distill-v1-long: `distill-v1`'s data, trained three times as long

`train/checkpoint.pt` has the same architecture as [`distill-v1`](../distill-v1) (128 channels, 8 blocks) and was trained on the same 848,307 labels with the same settings, except for 18 epochs instead of 6.
It tests whether more epochs on that data help.
Extending this configuration from 6 to 18 epochs did not improve held-out top-1 agreement or the paired result: the longer run fit its training data much more closely, agreed with the engine no more often on held-out positions, and lost the paired comparison with `distill-v1`.

| | distill-v1 | distill-v1-long |
| --- | --- | --- |
| Epochs | 6 | 18 |
| Top move matched the engine's on the training data, averaged over all epochs | 24.9 percent | 39.7 percent |
| Top-1 agreement on the 24,728 [held-out](../heldout-v1) positions | 25.8 percent | 25.6 percent |
| Value sign agreement on the held-out positions | 93.2 percent | 93.6 percent |

Training agreement rose by 15 points while held-out agreement did not move, which is consistent with fitting those 848,307 positions rather than learning more that carries over to new ones.

## How it was made

[`configs/distill_gpu_long.yaml`](../../configs/distill_gpu_long.yaml) is `configs/distill_gpu.yaml` with `epochs: 18`.
Training ran from a clean checkout of `b665e96` on the RTX 5070 Ti machine described in `distill-v1`.
The run read a copy of the config outside the repository with the same SHA-256, `a905f707…`, which is the path in the manifest.
The labels are `distill-v1`'s, attached to the [v0.3.0 release](https://github.com/hskl18/cc-lite/releases/tag/v0.3.0) as `labels.jsonl.xz`.

```bash
export PYTHONPATH=src
python -m train.experiment run --config configs/distill_gpu_long.yaml --data ~/runs/distill-v1/labels.jsonl \
  --output-dir ~/runs/distill-v1-long/train
```

Records: [`train/manifest.json`](train/manifest.json), [`train/metrics.json`](train/metrics.json), and the held-out report [`../heldout-v1/agreement/distill-v1-long.json`](../heldout-v1/agreement/distill-v1-long.json).
The home directory in the training manifest's paths was rewritten to `~/` before publication, as for `distill-v1`.

## Evaluation

The `random` and `material` bundles and the held-out report were produced on the RTX 5070 Ti machine, and the paired and Pikafish bundles on a Kaggle machine with an NVIDIA T4, all from clean checkouts of `b665e96`; all six validate with `python -m eval.evidence validate`.
The Kaggle runs used the Pikafish build named in [`evidence/distill-v1`](../distill-v1), checked against its SHA-256 before use, so their manifests record it under `/root/engines/pikafish/`.
The manifests record each run's platform and library versions but not the GPU model, so the machines named here are as reported by whoever ran them; games on different hardware can break floating-point ties differently.

| Bundle | Settings | Result |
| --- | --- | --- |
| [`random`](random) | `configs/baseline_cpu.yaml`: CPU, 50 simulations, 20 games, colors alternating, 400-ply cap | 20 wins, 0 draws, 0 losses, 0 capped |
| [`material`](material) | the same | 20 wins, 0 draws, 0 losses, 0 capped |
| [`vs-v1`](vs-v1) | `configs/paired_eval.yaml`: GPU, 200 simulations, 6 sampled plies, seeds 1, 7, 19, `paired-v1` suite, up to 24 pairs, against `distill-v1` | 17 wins, 2 draws, 29 losses over 24 complete pairs; score rate 0.375, 95 percent paired-bootstrap interval 0.271 to 0.479; sequential rule inconclusive at the 24-pair limit |
| [`pikafish-depth-1`](pikafish-depth-1) | `configs/engine_eval.yaml`: GPU, 200 simulations, 6 sampled plies, 20 games, colors alternating, 400-ply cap, Pikafish at depth 1 | 0 wins, 2 draws, 18 losses, 0 capped |
| [`pikafish-depth-2`](pikafish-depth-2) | the same at depth 2 | 0 wins, 0 draws, 20 losses, 0 capped |
| [`pikafish-depth-4`](pikafish-depth-4) | the same at depth 4 | 0 wins, 0 draws, 20 losses, 0 capped |

`distill-v1-long` did not beat `distill-v1`.
Like `distill-v2`, it lost the paired comparison by the bootstrap interval, which lies below 0.5, while the more conservative anytime-valid sequential rule reached no decision within 24 pairs.
Against Pikafish it scored 1 of 60 points, the same as `distill-v1`.

```bash
export PYTHONPATH=src
CK=evidence/distill-v1-long/train/checkpoint.pt
python -m eval.evaluate --config configs/baseline_cpu.yaml --checkpoint $CK --opponent random --output-dir evidence/distill-v1-long/random
python -m eval.evaluate --config configs/baseline_cpu.yaml --checkpoint $CK --opponent material --output-dir evidence/distill-v1-long/material
python -m eval.head_to_head --config configs/paired_eval.yaml --checkpoint-a $CK \
  --checkpoint-b evidence/distill-v1/train/checkpoint.pt --opening-suite configs/openings/paired-v1.json \
  --seeds 1,7,19 --min-pairs 8 --max-pairs 24 --output-dir evidence/distill-v1-long/vs-v1
python -m eval.evaluate --config configs/engine_eval.yaml --checkpoint $CK --opponent engine --engine-command ./pikafish \
  --engine-flavor pikafish --engine-eval-file ./pikafish.nnue --engine-depth 1 --output-dir evidence/distill-v1-long/pikafish-depth-1
```
