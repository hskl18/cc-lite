# distill-v3: twice the positions, with MultiPV soft targets

`train/checkpoint.pt` has `distill-v1`'s architecture (128 channels, 8 blocks) and training settings, apart from fp16 instead of bf16 on a T4, and changes two things: the policy target spreads over the engine's four best moves instead of one, and the data adds 824,481 new positions to `distill-v1`'s 848,307.
It is the first checkpoint here to beat `distill-v1` in a paired comparison: 38 wins, 1 draw, and 9 losses over 24 complete pairs, a score rate of 0.802 with a 95 percent paired-bootstrap interval of 0.677 to 0.906.
Two ablations in [`distill-v3-half-data`](../distill-v3-half-data) and [`distill-v3-half-epochs`](../distill-v3-half-epochs) separate the two changes: the soft targets alone showed no gain, and adding the new positions at about the same number of optimizer steps did.

| | distill-v1 | distill-v3 |
| --- | --- | --- |
| Positions | 848,307 | 1,672,788 |
| Policy target | best move | softmax over up to four lines at 100 centipawns |
| Optimizer steps | 9,942 | 19,608 |
| Top-1 agreement on the 24,728 [held-out](../heldout-v1) positions | 25.8 percent | 29.6 percent |
| Unmasked top move legal on those positions | 92.0 percent | 96.2 percent |
| Value sign agreement on those positions | 93.2 percent | 94.0 percent |

## What made the difference

`distill-v3` doubles both the positions and the optimizer steps, so the two ablations hold one of them fixed.
Both were trained on Kaggle from the same commit and config, and both paired comparisons ran under the same settings as `vs-v1` below.

| Checkpoint | Positions | Policy target | Optimizer steps | Held-out top-1 | Paired result |
| --- | --- | --- | --- | --- | --- |
| [`distill-v3-half-data`](../distill-v3-half-data) | 848,307 | soft | 9,942 | 26.0 percent | 0.458 (0.333 to 0.583) against `distill-v1` |
| [`distill-v3-half-epochs`](../distill-v3-half-epochs) | 1,672,788 | soft | 9,804 | 27.4 percent | 0.656 (0.552 to 0.771) against `distill-v3-half-data`, 0.646 (0.531 to 0.760) against `distill-v1` |
| `distill-v3` | 1,672,788 | soft | 19,608 | 29.6 percent | 0.698 (0.615 to 0.792) against `distill-v3-half-data`, 0.802 (0.677 to 0.906) against `distill-v1` |

The soft targets on `distill-v1`'s positions did not beat `distill-v1`.
Adding the new positions at about the same number of steps beat that ablation, so this gain does not need the extra compute.
[`distill-v1-long`](../distill-v1-long) already showed that three times the epochs on `distill-v1`'s positions, with best-move targets, did not help.
`distill-v3` and `distill-v3-half-epochs` were not played against each other, so whether the second 3 epochs added strength is suggested by the table but not measured.
No run trained best-move targets on all the positions, so whether the soft targets help on the larger data is not measured either.
Every interval above is a paired-bootstrap interval, and every sequential decision was inconclusive at the 24-pair limit.

## How it was made

The engine is the Fairy-Stockfish build described in [`evidence/distill-v1`](../distill-v1), with the same depth-10 search, one thread, 16 MB hash, and engine state cleared before every position.

1. New positions: Fairy-Stockfish played 12,000 games against itself at depth 2, with 15 percent of moves drawn uniformly at random, a 160-ply cap, and seed 21, `distill-v1`'s settings with a different seed.
   Of 1,002,469 positions, 824,481 remain after removing duplicates, the eight `paired-v1` opening positions, and the 873,035 positions of `distill-v1` and [`heldout-v1`](../heldout-v1), with 0 engine moves that were illegal under Profile 1.
   The exclusion file was `distill-v1`'s positions followed by `heldout-v1`'s, and its SHA-256 in [`data/extra-v1-positions.manifest.json`](data/extra-v1-positions.manifest.json), `c9bdc65e…`, is the digest of those two files concatenated.
   `distill-v2`'s positions were not excluded, so the two sets may share positions.
2. Labels: `distill-v1`'s positions and the new ones were each searched with `--multipv 4`.
   The policy target is `exp((score - best) / 100)` over the lines, normalized, and the value target is line 1's score divided by 1,000 and clipped, as for `distill-v1`.
   1,636,042 of the 1,672,788 targets (97.8 percent) spread over two or more moves; the other 36,746 positions had one line or fell back to the best move, as the distillation section of [docs/usage.md](../../docs/usage.md#optional-pikafish--engine-distillation) describes.
   Records: [`data/soft-v1-labels.manifest.json`](data/soft-v1-labels.manifest.json) for `distill-v1`'s positions and [`data/extra-v1-labels.manifest.json`](data/extra-v1-labels.manifest.json) for the new ones, both from a clean checkout of `073596b` on the RTX 5070 Ti machine described in `distill-v1`.
3. Training: `train.experiment` on the two label files concatenated in that order, 6 epochs at batch 512, 4,045 s on an NVIDIA T4 on Kaggle, from a clean checkout of `073596b`.
   The T4 has no native bf16 and the Kaggle machine has 4 CPUs, so the config was `configs/distill_gpu.yaml` with `precision: fp16` and `num_workers: 4` (SHA-256 `464894e1…`), written outside the repository; its path and the data's are recorded as they were on that machine.
   Record: [`train/manifest.json`](train/manifest.json) and [`train/metrics.json`](train/metrics.json).

```bash
export PYTHONPATH=src
FAIRY=~/engines/fairy/fairy-stockfish_x86-64-avx512
mkdir -p ~/runs/extra-v1
{ cat ~/runs/distill-v1/positions.jsonl; xz -dc evidence/heldout-v1/positions.jsonl.xz; } > ~/runs/extra-v1/exclude.jsonl
python -m train.engine_positions --engine "$FAIRY" --engine-flavor fairy --games 12000 --depth 2 \
  --random-move-rate 0.15 --max-plies 160 --seed 21 --workers 14 \
  --exclude-opening-suite configs/openings/paired-v1.json --exclude-positions ~/runs/extra-v1/exclude.jsonl \
  --output ~/runs/extra-v1/positions.jsonl
python -m train.distill_engine --engine "$FAIRY" --engine-flavor fairy --positions ~/runs/distill-v1/positions.jsonl \
  --output ~/runs/soft-v1/labels.jsonl --depth 10 --workers 14 --multipv 4
python -m train.distill_engine --engine "$FAIRY" --engine-flavor fairy --positions ~/runs/extra-v1/positions.jsonl \
  --output ~/runs/extra-v1/labels.jsonl --depth 10 --workers 14 --multipv 4
cat ~/runs/soft-v1/labels.jsonl ~/runs/extra-v1/labels.jsonl > ~/runs/soft-all.jsonl
sed -e 's/precision: bf16/precision: fp16/' -e 's/num_workers: 12/num_workers: 4/' configs/distill_gpu.yaml > ~/runs/distill_t4.yaml
python -m train.experiment run --config ~/runs/distill_t4.yaml --data ~/runs/soft-all.jsonl --output-dir ~/runs/distill-v3/train
```

## What is and is not here

The checkpoint, the training manifest and metrics, and the three data manifests are committed.
The two label files (50 MB and 49 MB compressed) are attached to the [v0.5.0 release](https://github.com/hskl18/cc-lite/releases/tag/v0.5.0) as `soft-v1-labels.jsonl.xz` and `extra-v1-labels.jsonl.xz`; their decompressed SHA-256 digests are in the manifests.
`distill-v1`'s positions are attached to the v0.3.0 release, and the new positions file is each label's FEN in order, which reproduces the digest in its manifest:

```bash
xz -dc extra-v1-labels.jsonl.xz | python -c 'import json, sys; [print(json.dumps({"fen": json.loads(l)["fen"]})) for l in sys.stdin]' | sha256sum
```

The home directory in the data manifests' paths was rewritten to `~/` before publication, as for `distill-v1`.

## Evaluation

The `random` and `material` bundles and the held-out report ran on an Apple M3 Pro from a clean checkout of `073596b`, the paired and Pikafish bundles on Kaggle T4 machines from the same commit, and all of them validate with `python -m eval.evidence validate`.
The Kaggle runs read the checkpoints from `/tmp/soft/`, so their `argv` keeps those paths; `checkpoint.path` and `opponent_checkpoint.path` were rewritten to the committed files, whose SHA-256 digests are the recorded ones, and nothing else in these records was altered.
The Pikafish build is the one named in [`evidence/distill-v1`](../distill-v1), checked against its SHA-256 before use.

| Bundle | Settings | Result |
| --- | --- | --- |
| [`random`](random) | `configs/baseline_cpu.yaml`: CPU, 50 simulations, 20 games, colors alternating, 400-ply cap | 20 wins, 0 draws, 0 losses, 0 capped |
| [`material`](material) | the same | 20 wins, 0 draws, 0 losses, 0 capped |
| [`vs-v1`](vs-v1) | `configs/paired_eval.yaml`: GPU, 200 simulations, 6 sampled plies, seeds 1, 7, 19, `paired-v1` suite, up to 24 pairs, against `distill-v1` | 38 wins, 1 draw, 9 losses over 24 complete pairs; score rate 0.802, 95 percent paired-bootstrap interval 0.677 to 0.906; sequential rule inconclusive at the 24-pair limit |
| [`vs-half-data`](vs-half-data) | the same, against [`distill-v3-half-data`](../distill-v3-half-data) | 31 wins, 5 draws, 12 losses over 24 complete pairs; score rate 0.698, interval 0.615 to 0.792; sequential rule inconclusive at the 24-pair limit |
| [`pikafish-depth-1`](pikafish-depth-1) | `configs/engine_eval.yaml`: GPU, 200 simulations, 6 sampled plies, 20 games, colors alternating, 400-ply cap, Pikafish at depth 1 | 4 wins, 1 draw, 15 losses, 0 capped |
| [`pikafish-depth-2`](pikafish-depth-2) | the same at depth 2 | 0 wins, 4 draws, 16 losses, 0 capped |
| [`pikafish-depth-4`](pikafish-depth-4) | the same at depth 4 | 0 wins, 0 draws, 20 losses, 0 capped |

The paired interval against `distill-v1` lies entirely above 0.5, while the anytime-valid sequential rule, which pays for every look at the data, reached no decision within 24 pairs, so by [the protocol](../../docs/evaluation-protocol.md) this is a bootstrap result, not a sequential superiority decision.
The paired report also gives an Elo difference of 236 (126 to 378) by `logistic-elo-with-beta-half-continuity-correction`, a small-sample method, relative to `distill-v1` only.
Against Pikafish at depth 1, three of the four wins were adjudications against the engine for perpetual check, and in the fourth, at ply 41, Pikafish had no legal move; one of the 15 losses was such an adjudication against the network.
The four draws at depth 2 were threefold repetitions.
Over the 60 Pikafish games it scored 6.5 points, against 5.5 for `distill-v2` and 1 each for `distill-v1` and `distill-v1-long`; with 20 games per depth these differences are noise-sized, and it is still far below a shallow engine search.

```bash
export PYTHONPATH=src
CK=evidence/distill-v3/train/checkpoint.pt
python -m eval.evaluate --config configs/baseline_cpu.yaml --checkpoint $CK --opponent random --output-dir evidence/distill-v3/random
python -m eval.evaluate --config configs/baseline_cpu.yaml --checkpoint $CK --opponent material --output-dir evidence/distill-v3/material
python -m eval.head_to_head --config configs/paired_eval.yaml --checkpoint-a $CK \
  --checkpoint-b evidence/distill-v1/train/checkpoint.pt --opening-suite configs/openings/paired-v1.json \
  --seeds 1,7,19 --min-pairs 8 --max-pairs 24 --output-dir evidence/distill-v3/vs-v1
python -m eval.evaluate --config configs/engine_eval.yaml --checkpoint $CK --opponent engine --engine-command ./pikafish \
  --engine-flavor pikafish --engine-eval-file ./pikafish.nnue --engine-depth 1 --output-dir evidence/distill-v3/pikafish-depth-1
```
