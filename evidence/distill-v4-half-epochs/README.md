# distill-v4-half-epochs: all of `distill-v4`'s data at `distill-v3`'s compute

An ablation of [`distill-v4`](../distill-v4): the same 3,320,673 positions, MultiPV 4 soft targets, and config, trained for 3 epochs instead of 6.
That is 19,458 optimizer steps, within 1 percent of the 19,608 of [`distill-v3`](../distill-v3), so the two differ in the positions they saw and hardly at all in training compute.
They differ in GPU as well: the config file is byte for byte `distill-v3`'s 3-epoch variant (SHA-256 `cdba1c71…`), but this run used the RTX 5070 Ti and `distill-v3` a Kaggle T4.
Its training data is the file in [`train/manifest.json`](train/manifest.json), built as described in `distill-v4`, and training ran from a clean checkout of `0b9a2b6`.
Records: [`train/manifest.json`](train/manifest.json), [`train/metrics.json`](train/metrics.json), and the held-out report [`../heldout-v1/agreement/distill-v4-half-epochs.json`](../heldout-v1/agreement/distill-v4-half-epochs.json).

| | distill-v3 | distill-v4-half-epochs | distill-v4 |
| --- | --- | --- | --- |
| Positions | 1,672,788 | 3,320,673 | 3,320,673 |
| Epochs | 6 | 3 | 6 |
| Optimizer steps | 19,608 | 19,458 | 38,916 |
| Top-1 agreement on the 24,728 [held-out](../heldout-v1) positions | 29.6 percent | 31.7 percent | 33.6 percent |
| Value sign agreement on those positions | 94.0 percent | 91.1 percent | 90.9 percent |

[`vs-v3`](vs-v3) uses `configs/paired_eval.yaml` (GPU, 200 simulations, 6 sampled plies, seeds 1, 7, 19, `paired-v1` suite, up to 24 pairs) against `distill-v3`: 26 wins, 5 draws, and 17 losses over 24 complete pairs, a score rate of 0.594 with a 95 percent paired-bootstrap interval of 0.469 to 0.719, and the sequential rule was inconclusive at the 24-pair limit.

At about `distill-v3`'s number of optimizer steps, twice the positions raised held-out policy agreement, but the paired games did not separate it from `distill-v3`.
`distill-v4`, with twice the steps, did separate, so on this evidence its gain cannot be credited to the positions alone; telling whether they help at matched steps would need more pairs.
The value sign agreement is lower than `distill-v3`'s for a reason [`distill-v4`](../distill-v4) describes.
`distill-v4` has the data commands and how the `~/runs/v4/work/` checkpoint paths in these records were rewritten.

```bash
sed -e 's/precision: bf16/precision: fp16/' -e 's/num_workers: 12/num_workers: 4/' -e 's/epochs: 6/epochs: 3/' \
  configs/distill_gpu.yaml > ~/runs/distill_t4_3ep.yaml
python -m train.experiment run --config ~/runs/distill_t4_3ep.yaml --data ~/runs/v4/soft-all-v4.jsonl \
  --max-records 4000000 --output-dir ~/runs/distill-v4-half-epochs/train
python -m eval.head_to_head --config configs/paired_eval.yaml \
  --checkpoint-a evidence/distill-v4-half-epochs/train/checkpoint.pt \
  --checkpoint-b evidence/distill-v3/train/checkpoint.pt --opening-suite configs/openings/paired-v1.json \
  --seeds 1,7,19 --min-pairs 8 --max-pairs 24 --output-dir evidence/distill-v4-half-epochs/vs-v3
```
