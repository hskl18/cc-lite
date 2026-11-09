# heldout-v1: positions no distilled checkpoint was trained on

A fixed set of 24,728 labeled positions for comparing checkpoints with `python -m eval.label_agreement`.

## How it was made

1. Positions: Fairy-Stockfish played 400 games against itself at depth 2, with 10 percent of moves drawn uniformly at random, a 160-ply cap, and seed 2.
   Of the 33,749 positions, 24,728 remain after removing duplicates, the eight `paired-v1` opening positions, and all 1,713,216 distinct training positions of `distill-v1` and `distill-v2`.
   Record: [`positions.manifest.json`](positions.manifest.json).
   The exclusion file was `distill-v1`'s positions followed by `distill-v2`'s, and its SHA-256 in that manifest, `e7dcc776d2731659…`, is the digest of those two files concatenated.
   `distill-v1`'s positions are attached to the v0.3.0 release; `distill-v2`'s are not published yet, so until they are, its part of the disjointness rests on this record.
2. Labels: every position was searched at depth 10, one thread, with the engine state cleared first; the best move is the policy target, the side-to-move score is kept as `score_cp`, and the value is that score divided by 1,000 and clipped.
   Record: [`labels.manifest.json`](labels.manifest.json).

The engine is the Fairy-Stockfish build described in [`evidence/distill-v1`](../distill-v1).
Both data files are committed xz-compressed; their decompressed SHA-256 digests are the ones in the manifests, whose home-directory paths were rewritten to `~/` before publication.

## Use

```bash
xz -dk evidence/heldout-v1/labels.jsonl.xz
python -m eval.label_agreement --checkpoint path/to/checkpoint.pt \
  --labels evidence/heldout-v1/labels.jsonl --labels-manifest evidence/heldout-v1/labels.manifest.json \
  --output report.json
```

Top-1 agreement and value sign agreement compare checkpoints trained on different value mappings; value MSE compares only checkpoints trained on the same linear mapping as these labels.
The set reuses the training data's generator with a different seed and random-move rate, so it measures generalization to unseen positions from the same kind of games, not to human play.
