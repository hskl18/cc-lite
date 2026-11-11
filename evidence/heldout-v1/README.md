# heldout-v1: positions no distilled checkpoint was trained on

A fixed set of 24,728 labeled positions for comparing checkpoints with `python -m eval.label_agreement`.

## How it was made

1. Positions: Fairy-Stockfish played 400 games against itself at depth 2, with 10 percent of moves drawn uniformly at random, a 160-ply cap, and seed 2.
   Of the 33,749 positions, 24,728 remain after removing duplicates, the eight `paired-v1` opening positions, and all 1,713,216 distinct training positions of `distill-v1` and `distill-v2`.
   Record: [`positions.manifest.json`](positions.manifest.json).
   The exclusion file was `distill-v1`'s positions followed by `distill-v2`'s, and its SHA-256 in that manifest, `e7dcc776d2731659…`, is the digest of those two files concatenated.
   `distill-v1`'s positions are attached to the v0.3.0 release and `distill-v2`'s to the v0.4.0 release, so that digest can be recomputed from them.
2. Labels: every position was searched at depth 10, one thread, with the engine state cleared first; the best move is the policy target, the side-to-move score is kept as `score_cp`, and the value is that score divided by 1,000 and clipped.
   Record: [`labels.manifest.json`](labels.manifest.json).

The engine is the Fairy-Stockfish build described in [`evidence/distill-v1`](../distill-v1).
Both data files are committed xz-compressed; their decompressed SHA-256 digests are the ones in the manifests, whose home-directory paths were rewritten to `~/` before publication.

## Results

| Checkpoint | Top-1 agreement, illegal moves masked | Unmasked top move legal | Value sign agreement |
| --- | ---: | ---: | ---: |
| `distill-v1` | 25.8 percent | 92.0 percent | 93.2 percent |
| `distill-v2` | 25.8 percent | 92.6 percent | 93.3 percent |
| `distill-v1-long` | 25.6 percent | 91.6 percent | 93.6 percent |
| `distill-v3-half-data` | 26.0 percent | 94.4 percent | 93.3 percent |
| `distill-v3-half-epochs` | 27.4 percent | 94.7 percent | 93.8 percent |
| `distill-v3` | 29.6 percent | 96.2 percent | 94.0 percent |
| `distill-v4-half-epochs` | 31.7 percent | 97.2 percent | 91.1 percent |
| `distill-v4` | 33.6 percent | 98.2 percent | 90.9 percent |

The labels are Fairy-Stockfish's depth-10 best move and score.
Sign agreement counts the 22,011 positions whose label is at least 0.1 from zero.
The reports are in [`agreement/`](agreement), produced with the committed data from clean checkouts of `070c0fb` for `distill-v1` and `distill-v2`, of `b665e96` for `distill-v1-long`, of `073596b` for the three `distill-v3` checkpoints, and of `0b9a2b6` for the two `distill-v4` checkpoints; they replay no games, unlike the game bundles, and hash the checkpoint, the labels, and the labels manifest.
The first three checkpoints agree with the engine about equally often, and the paired games in [`evidence/distill-v2`](../distill-v2) and [`evidence/distill-v1-long`](../distill-v1-long) separate the other two from `distill-v1`, so agreement on this set does not rank those three by playing strength.
The `distill-v3` and `distill-v4` checkpoints did not train on these positions either, because their new positions excluded them, and their top-1 agreement moved with their paired results, as [`evidence/distill-v3`](../distill-v3) and [`evidence/distill-v4`](../distill-v4) show.
The `distill-v4` checkpoints' lower value sign agreement comes from their stored BatchNorm statistics rather than their weights, by an exploratory check in [#21](https://github.com/hskl18/cc-lite/issues/21), so value agreement here does not track playing strength either.
Training recomputes those statistics at its end since [#24](https://github.com/hskl18/cc-lite/pull/24), and every checkpoint here was trained before that.

## Use

```bash
xz -dk evidence/heldout-v1/labels.jsonl.xz
python -m eval.label_agreement --checkpoint path/to/checkpoint.pt \
  --labels evidence/heldout-v1/labels.jsonl --labels-manifest evidence/heldout-v1/labels.manifest.json \
  --output report.json
```

Top-1 agreement and value sign agreement compare checkpoints trained on different value mappings; value MSE compares only checkpoints trained on the same linear mapping as these labels.
The set reuses the training data's generator with a different seed and random-move rate, so it measures generalization to unseen positions from the same kind of games, not to human play.
