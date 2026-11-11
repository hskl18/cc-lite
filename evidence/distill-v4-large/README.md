# distill-v4-large: `distill-v4`'s data in a larger network

`train/checkpoint.pt` is trained as [`distill-v4`](../distill-v4) was, on the same 3,320,673 positions with the same MultiPV 4 soft targets for 6 epochs and 38,916 optimizer steps, but its network has 192 channels and 12 residual blocks instead of 128 and 8.
Against Pikafish it scored 14 points in 60 games, against 5.5 for `distill-v4`, whose one game capped at 400 plies scores nothing, and it agreed with the engine's best move on 36.7 percent of the held-out positions, against 33.6.
Its paired games against `distill-v4` did not separate the two.

| | distill-v4 | distill-v4-large |
| --- | --- | --- |
| Channels and residual blocks | 128 and 8 | 192 and 12 |
| Parameters | 3,860,347 | 9,484,219 |
| Training policy loss and value loss, averaged over all steps | 2.721 and 0.0553 | 2.628 and 0.0517 |
| Top-1 agreement on the 24,728 [held-out](../heldout-v1) positions | 33.6 percent | 36.7 percent |
| Unmasked top move legal on those positions | 98.2 percent | 98.9 percent |
| Value sign agreement on those positions | 90.9 percent | 96.5 percent |
| Against Pikafish at depths 1, 2, and 4 | 1-2-16 with 1 capped, 3-1-16, 0-0-20 | 7-4-9, 3-3-14, 0-1-19 |
| Points against Pikafish in 60 games, a capped game scoring nothing | 5.5 | 14 |

## What the larger network changed

The training data is byte for byte `distill-v4`'s, and the config differs from `distill-v4`'s only in `channels` and `blocks`, with the same seed, GPU, and precision, and `distill-v4`'s commit `0b9a2b6` and this one's `4eee344` differ in no file under `src/` or `configs/`, so the network size is the one change.

[`vs-v4`](vs-v4) scored 27 wins, 3 draws, and 18 losses over 24 complete pairs against `distill-v4`, a score rate of 0.594 with a 95 percent paired-bootstrap interval of 0.458 to 0.719, and the sequential rule was inconclusive at the 24-pair limit.
The paired report gives an Elo difference of 65 (-28 to 159) by `logistic-elo-with-beta-half-continuity-correction`.

Each Pikafish game has a counterpart in `distill-v4`'s bundles with the same seed, color, and depth.
`distill-v4`'s one game capped at 400 plies has no result, so its pair, seed 18 at depth 1, is left out.
Over the other 59 matched games, `distill-v4-large` scored more in 14, less in 4, and the same in 41.
A two-sided sign test on the 18 that differ gives p = 0.031, but it treats the games as independent, and each seed and color plays at all three depths; summing each seed and color's games instead leaves 9 better, 3 worse, and 8 even, with p = 0.146.
Both are exploratory checks computed from the committed games, not gates in [`docs/evaluation-protocol.md`](../../docs/evaluation-protocol.md), and the two networks' games diverge after their first different move, so the match is by seed and color only.

The larger network did better on held-out agreement and scored more against Pikafish, which suggests that network size limits `distill-v4`, but neither its paired games against `distill-v4` nor the per-seed check above separates the two, and it still lost 42 of its 60 games to Pikafish.
Its value sign agreement does not show `distill-v4`'s drop, although both were trained before [#24](https://github.com/hskl18/cc-lite/pull/24) added the BatchNorm recomputation that [#21](https://github.com/hskl18/cc-lite/issues/21) describes.

## How it was made

The training file was rebuilt by concatenating `distill-v3`'s two label files from the v0.5.0 release and `extra-v2-labels.jsonl.xz` from the v0.6.0 release, in `distill-v4`'s order, and its SHA-256 matched `distill-v4`'s `92977e13…` before training.
The config is `configs/distill_gpu.yaml` with `precision: fp16`, `num_workers: 4`, `channels: 192`, and `blocks: 12` (SHA-256 `72844416…`).
Training ran from a clean checkout of `4eee344` on the RTX 5070 Ti, where the training command took 35 minutes.
Records: [`train/manifest.json`](train/manifest.json), [`train/metrics.json`](train/metrics.json), and the held-out report [`../heldout-v1/agreement/distill-v4-large.json`](../heldout-v1/agreement/distill-v4-large.json).

```bash
export PYTHONPATH=src
mkdir -p ~/runs/v4-large
xz -dc soft-v1-labels.jsonl.xz extra-v1-labels.jsonl.xz extra-v2-labels.jsonl.xz > ~/runs/v4-large/soft-all-v4.jsonl
sed -e 's/precision: bf16/precision: fp16/' -e 's/num_workers: 12/num_workers: 4/' \
    -e 's/channels: 128/channels: 192/' -e 's/blocks: 8/blocks: 12/' configs/distill_gpu.yaml > ~/runs/v4-large/distill_large.yaml
python -m train.experiment run --config ~/runs/v4-large/distill_large.yaml --data ~/runs/v4-large/soft-all-v4.jsonl \
  --max-records 4000000 --output-dir ~/runs/v4-large/train
```

## Evaluation

The game bundles ran on the RTX 5070 Ti machine from a clean checkout of `4eee344`, the paired and Pikafish games sharing the GPU, and the held-out report ran on an Apple M3 Pro from a clean checkout of `3ce0b8f`.
The game bundles validate with `python -m eval.evidence validate`.
The runs read the checkpoint from `~/runs/v4-large/train/`, so their `argv` keeps that path; `checkpoint.path` was rewritten to the committed file, whose SHA-256 is the recorded one, and the PC's home directory was rewritten to `~/`.
The Pikafish build is the one named in [`evidence/distill-v1`](../distill-v1), and each manifest pins its binary and network by SHA-256.

| Bundle | Settings | Result |
| --- | --- | --- |
| [`random`](random) | `configs/baseline_cpu.yaml`: CPU, 50 simulations, 20 games, colors alternating, 400-ply cap | 20 wins, 0 draws, 0 losses, 0 capped |
| [`material`](material) | the same | 20 wins, 0 draws, 0 losses, 0 capped |
| [`vs-v4`](vs-v4) | `configs/paired_eval.yaml`: GPU, 200 simulations, 6 sampled plies, seeds 1, 7, 19, `paired-v1` suite, up to 24 pairs, against `distill-v4` | 27 wins, 3 draws, 18 losses over 24 complete pairs; score rate 0.594, interval 0.458 to 0.719; sequential rule inconclusive at the 24-pair limit |
| [`pikafish-depth-1`](pikafish-depth-1) | `configs/engine_eval.yaml`: GPU, 200 simulations, 6 sampled plies, 20 games, colors alternating, 400-ply cap, Pikafish at depth 1 | 7 wins, 4 draws, 9 losses, 0 capped |
| [`pikafish-depth-2`](pikafish-depth-2) | the same at depth 2 | 3 wins, 3 draws, 14 losses, 0 capped |
| [`pikafish-depth-4`](pikafish-depth-4) | the same at depth 4 | 0 wins, 1 draw, 19 losses, 0 capped |

Against Pikafish at depth 1, six wins left the engine with no legal move, at plies 29 to 89, and one was an adjudication against the engine for perpetual check at ply 102; one loss was an adjudication against the network for perpetual check, and all four draws were threefold repetitions.
At depth 2 the three wins left the engine with no legal move, and the draws at depths 2 and 4 were threefold repetitions.

```bash
export PYTHONPATH=src
CK=evidence/distill-v4-large/train/checkpoint.pt
python -m eval.evaluate --config configs/baseline_cpu.yaml --checkpoint $CK --opponent random --output-dir evidence/distill-v4-large/random
python -m eval.evaluate --config configs/baseline_cpu.yaml --checkpoint $CK --opponent material --output-dir evidence/distill-v4-large/material
python -m eval.head_to_head --config configs/paired_eval.yaml --checkpoint-a $CK \
  --checkpoint-b evidence/distill-v4/train/checkpoint.pt --opening-suite configs/openings/paired-v1.json \
  --seeds 1,7,19 --min-pairs 8 --max-pairs 24 --output-dir evidence/distill-v4-large/vs-v4
python -m eval.evaluate --config configs/engine_eval.yaml --checkpoint $CK --opponent engine --engine-command ./pikafish \
  --engine-flavor pikafish --engine-eval-file ./pikafish.nnue --engine-depth 1 --output-dir evidence/distill-v4-large/pikafish-depth-1
```
