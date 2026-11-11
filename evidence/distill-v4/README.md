# distill-v4: twice `distill-v3`'s positions

`train/checkpoint.pt` keeps [`distill-v3`](../distill-v3)'s architecture, MultiPV 4 soft targets, and both config files byte for byte, and doubles the data with 1,647,885 new positions, for 3,320,673 in all.
It beat `distill-v3` in a paired comparison: 38 wins, 2 draws, and 8 losses over 24 complete pairs, a score rate of 0.812 with a 95 percent paired-bootstrap interval of 0.719 to 0.906.
An ablation, [`distill-v4-half-epochs`](../distill-v4-half-epochs), trained on the same data for about `distill-v3`'s number of optimizer steps and did not separate from `distill-v3`, so this experiment cannot credit the gain to the new positions alone.

| | distill-v3 | distill-v4 |
| --- | --- | --- |
| Positions | 1,672,788 | 3,320,673 |
| Optimizer steps | 19,608 | 38,916 |
| Training GPU | NVIDIA T4 (Kaggle) | RTX 5070 Ti |
| Top-1 agreement on the 24,728 [held-out](../heldout-v1) positions | 29.6 percent | 33.6 percent |
| Unmasked top move legal on those positions | 96.2 percent | 98.2 percent |
| Value sign agreement on those positions | 94.0 percent | 90.9 percent |

## What made the difference

`distill-v4` doubles both the positions and the optimizer steps, as `distill-v3` did, and the ablation holds the steps at `distill-v3`'s.

| Checkpoint | Positions | Optimizer steps | Held-out top-1 | Paired result against `distill-v3` |
| --- | --- | --- | --- | --- |
| [`distill-v4-half-epochs`](../distill-v4-half-epochs) | 3,320,673 | 19,458 | 31.7 percent | 0.594 (0.469 to 0.719) |
| `distill-v4` | 3,320,673 | 38,916 | 33.6 percent | 0.812 (0.719 to 0.906) |

At matched steps the new positions raised held-out agreement, but the paired interval includes 0.5, unlike `distill-v3`'s matched-step ablation, which beat its baseline.
The full run's interval lies above 0.5, so twice the data with twice the steps helped, and this ablation cannot say how much of that the data alone gave.
`distill-v4` and `distill-v4-half-epochs` were not played against each other.
Every interval above is a paired-bootstrap interval, and every sequential decision was inconclusive at the 24-pair limit.

Value sign agreement on the held-out set fell from 94.0 to 90.9 percent while the policy and the paired games improved.
The value labels are distributed alike in all three label files.
An exploratory check outside these records, in [issue #21](https://github.com/hskl18/cc-lite/issues/21), traced the drop to the BatchNorm running statistics stored in the checkpoint rather than to the learned weights; re-estimating them showed no difference in play over 24 pairs, which does not rule out a small one.

## How it was made

The engine is the Fairy-Stockfish release named in [`evidence/distill-v1`](../distill-v1), with the same depth-10 search, MultiPV 4, one thread, 16 MB hash, and engine state cleared before every position.
Positions and the labels of shards 0 to 3 ran on Kaggle CPU machines, which lack AVX-512, with the release's `fairy-stockfish_x86-64-avx2` build (SHA-256 `df70c56d…`); shards 4 to 7 and the training ran on the RTX 5070 Ti machine with the `avx512` build used for `distill-v3`.

1. New positions: Fairy-Stockfish played 24,000 games against itself at depth 2, with 15 percent of moves drawn uniformly at random, a 160-ply cap, and seed 31, `distill-v3`'s settings with twice the games and another seed.
   Of 2,019,344 positions, 1,647,885 remain after removing duplicates, the eight `paired-v1` opening positions, and the 1,697,516 distinct positions of `distill-v1`, `distill-v3`'s new positions, and [`heldout-v1`](../heldout-v1), with 0 engine moves that were illegal under Profile 1.
   The exclusion file was those three position files concatenated in that order, each rebuilt from the FENs of its released labels and checked against its manifest; its SHA-256, `4e5a50ef…`, is in [`data/extra-v2-positions.manifest.json`](data/extra-v2-positions.manifest.json).
2. Labels: the positions were split in file order into eight shards of 205,986 positions, the last 205,983, recorded with their digests in [`data/extra-v2-shards.json`](data/extra-v2-shards.json), and each shard was searched as for `distill-v3`.
   The eight label manifests are in [`data/extra-v2-labels.manifests.json`](data/extra-v2-labels.manifests.json); shards 0 to 3 ran with the default 10-second per-position timeout and shards 4 to 7 with 300 seconds, after one Kaggle shard stopped on a search that took longer than 10 seconds, and the timeout only stops a search, so it changes no label.
   1,610,860 of the 1,647,885 targets (97.8 percent) spread over two or more moves, a count the released label file reproduces with the command under "What is and is not here".
   Kaggle labeled shards 4 to 7 as well, and those labels served only as a cross-check: shard 4 matched the committed digest byte for byte, and shard 5 differed in 2 of 205,986 records, which kept the same best move and score but listed a different number of lines.
   Fairy-Stockfish prints extra reports once a search runs past 3 seconds, so the likely cause is that these two searches took longer on one machine; it was not confirmed, and [issue #22](https://github.com/hskl18/cc-lite/issues/22) has the details.
3. Training: `train.experiment` on `distill-v3`'s two label files followed by the eight shards in order, with `distill-v3`'s config for 6 epochs at batch 512 (SHA-256 `464894e1…`: `configs/distill_gpu.yaml` with `precision: fp16` and `num_workers: 4`), from a clean checkout of `0b9a2b6`.
   It ran on the RTX 5070 Ti while the ablation's paired games shared the GPU; [`data/extra-v2-labels.data.json`](data/extra-v2-labels.data.json) has the digests of the concatenated new labels and of the full training file, which is the one in [`train/manifest.json`](train/manifest.json).
   Record: [`train/manifest.json`](train/manifest.json) and [`train/metrics.json`](train/metrics.json).

```bash
export PYTHONPATH=src
FAIRY=~/engines/fairy/fairy-stockfish_x86-64-avx512
mkdir -p ~/runs/v4
# soft-v1 and extra-v1 labels from the v0.5.0 release; their FENs rebuild distill-v1's and extra-v1's positions.
for name in soft-v1 extra-v1; do
  xz -dc $name-labels.jsonl.xz | python -c 'import json, sys; [print(json.dumps({"fen": json.loads(l)["fen"]})) for l in sys.stdin]'
done > ~/runs/v4/exclude.jsonl
xz -dc evidence/heldout-v1/positions.jsonl.xz >> ~/runs/v4/exclude.jsonl
python -m train.engine_positions --engine "$FAIRY" --engine-flavor fairy --games 24000 --depth 2 \
  --random-move-rate 0.15 --max-plies 160 --seed 31 --workers 14 \
  --exclude-opening-suite configs/openings/paired-v1.json --exclude-positions ~/runs/v4/exclude.jsonl \
  --output ~/runs/v4/positions.jsonl
python -m train.distill_engine --engine "$FAIRY" --engine-flavor fairy --positions ~/runs/v4/positions.jsonl \
  --output ~/runs/v4/extra-v2-labels.jsonl --depth 10 --workers 14 --multipv 4 --timeout 300
xz -dc soft-v1-labels.jsonl.xz extra-v1-labels.jsonl.xz | cat - ~/runs/v4/extra-v2-labels.jsonl > ~/runs/v4/soft-all-v4.jsonl
sed -e 's/precision: bf16/precision: fp16/' -e 's/num_workers: 12/num_workers: 4/' configs/distill_gpu.yaml > ~/runs/distill_t4.yaml
python -m train.experiment run --config ~/runs/distill_t4.yaml --data ~/runs/v4/soft-all-v4.jsonl \
  --max-records 4000000 --output-dir ~/runs/distill-v4/train
```

`--max-records` lifts the config's 2,000,000-record limit, which would otherwise cut the file short.
Labeling the positions in one run rather than eight shards gives the same records in the same order, because each position is searched from a cleared engine state, apart from the rare search-time effect above.

## What is and is not here

The checkpoint, the training manifest and metrics, and the data records are committed.
The new labels (92 MB compressed) are attached to the [v0.6.0 release](https://github.com/hskl18/cc-lite/releases/tag/v0.6.0) as `extra-v2-labels.jsonl.xz`, and `distill-v3`'s two label files are attached to the v0.5.0 release; their decompressed SHA-256 digests are in the data records.
The new positions file is each label's FEN in order, which reproduces the digest in its manifest:

```bash
xz -dc extra-v2-labels.jsonl.xz | python -c 'import json, sys; [print(json.dumps({"fen": json.loads(l)["fen"]})) for l in sys.stdin]' | sha256sum
xz -dc extra-v2-labels.jsonl.xz | python -c 'import json, sys; print(sum(len(json.loads(l)["policy"]) > 1 for l in sys.stdin))'
```

The PC's home directory in recorded paths was rewritten to `~/` before publication, and the Kaggle paths were kept as they were.

## Evaluation

The game bundles ran on the RTX 5070 Ti machine and the held-out report on an Apple M3 Pro, all from a clean checkout of `0b9a2b6`, and the game bundles validate with `python -m eval.evidence validate`.
The runs read the checkpoints from `~/runs/v4/work/`, so their `argv` keeps those paths; `checkpoint.path` was rewritten to the committed files, whose SHA-256 digests are the recorded ones, and nothing else in these records was altered beyond the home directory.
The Pikafish build is the one named in [`evidence/distill-v1`](../distill-v1), checked against its SHA-256 before use.

| Bundle | Settings | Result |
| --- | --- | --- |
| [`random`](random) | `configs/baseline_cpu.yaml`: CPU, 50 simulations, 20 games, colors alternating, 400-ply cap | 20 wins, 0 draws, 0 losses, 0 capped |
| [`material`](material) | the same | 20 wins, 0 draws, 0 losses, 0 capped |
| [`vs-v3`](vs-v3) | `configs/paired_eval.yaml`: GPU, 200 simulations, 6 sampled plies, seeds 1, 7, 19, `paired-v1` suite, up to 24 pairs, against `distill-v3` | 38 wins, 2 draws, 8 losses over 24 complete pairs; score rate 0.812, 95 percent paired-bootstrap interval 0.719 to 0.906; sequential rule inconclusive at the 24-pair limit |
| [`vs-v1`](vs-v1) | the same, against `distill-v1` | 43 wins, 3 draws, 2 losses over 24 complete pairs; score rate 0.927, interval 0.865 to 0.979; sequential rule inconclusive at the 24-pair limit |
| [`pikafish-depth-1`](pikafish-depth-1) | `configs/engine_eval.yaml`: GPU, 200 simulations, 6 sampled plies, 20 games, colors alternating, 400-ply cap, Pikafish at depth 1 | 1 win, 2 draws, 16 losses, 1 capped |
| [`pikafish-depth-2`](pikafish-depth-2) | the same at depth 2 | 3 wins, 1 draw, 16 losses, 0 capped |
| [`pikafish-depth-4`](pikafish-depth-4) | the same at depth 4 | 0 wins, 0 draws, 20 losses, 0 capped |

The paired report gives Elo differences of 248 (159 to 378) against `distill-v3` and 420 (311 to 600) against `distill-v1` by `logistic-elo-with-beta-half-continuity-correction`, a small-sample method, each relative to that opponent only.
Against Pikafish at depth 1 the win was an adjudication against the engine for perpetual check at ply 94, the two draws were threefold repetitions, and the capped game reached 400 plies with the network ahead on material.
At depth 2 Pikafish had no legal move in all three wins, at plies 35, 43, and 83, and the draw was a threefold repetition.
Over the 60 Pikafish games it scored 5.5 points, against 6.5 for `distill-v3`; with 20 games per depth these differences are noise-sized, and it is still far below a shallow engine search.

```bash
export PYTHONPATH=src
CK=evidence/distill-v4/train/checkpoint.pt
python -m eval.evaluate --config configs/baseline_cpu.yaml --checkpoint $CK --opponent random --output-dir evidence/distill-v4/random
python -m eval.evaluate --config configs/baseline_cpu.yaml --checkpoint $CK --opponent material --output-dir evidence/distill-v4/material
python -m eval.head_to_head --config configs/paired_eval.yaml --checkpoint-a $CK \
  --checkpoint-b evidence/distill-v3/train/checkpoint.pt --opening-suite configs/openings/paired-v1.json \
  --seeds 1,7,19 --min-pairs 8 --max-pairs 24 --output-dir evidence/distill-v4/vs-v3
python -m eval.evaluate --config configs/engine_eval.yaml --checkpoint $CK --opponent engine --engine-command ./pikafish \
  --engine-flavor pikafish --engine-eval-file ./pikafish.nnue --engine-depth 1 --output-dir evidence/distill-v4/pikafish-depth-1
```
