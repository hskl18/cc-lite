# distill-v3-half-epochs: all of `distill-v3`'s data for half the epochs

An ablation of [`distill-v3`](../distill-v3): the same 1,672,788 positions, MultiPV 4 soft targets, and config, trained for 3 epochs instead of 6.
That is 9,804 optimizer steps, within 2 percent of the 9,942 of [`distill-v3-half-data`](../distill-v3-half-data), so the two differ in the positions they saw and hardly at all in training compute.
Its training data is the two label files from the v0.5.0 release concatenated as for `distill-v3`, with the same SHA-256 in [`train/manifest.json`](train/manifest.json); training took 1,738 s on a Kaggle T4 from a clean checkout of `073596b`.
Records: [`train/manifest.json`](train/manifest.json), [`train/metrics.json`](train/metrics.json), and the held-out report [`../heldout-v1/agreement/distill-v3-half-epochs.json`](../heldout-v1/agreement/distill-v3-half-epochs.json).

| | distill-v3-half-data | distill-v3-half-epochs | distill-v3 |
| --- | --- | --- | --- |
| Positions | 848,307 | 1,672,788 | 1,672,788 |
| Epochs | 6 | 3 | 6 |
| Optimizer steps | 9,942 | 9,804 | 19,608 |
| Top-1 agreement on the 24,728 [held-out](../heldout-v1) positions | 26.0 percent | 27.4 percent | 29.6 percent |
| Value sign agreement on those positions | 93.3 percent | 93.8 percent | 94.0 percent |

Both bundles use `configs/paired_eval.yaml` (GPU, 200 simulations, 6 sampled plies, seeds 1, 7, 19, `paired-v1` suite, up to 24 pairs), and in both the sequential rule was inconclusive at the 24-pair limit.

| Bundle | Opponent | Result over 24 complete pairs | Score rate, 95 percent paired-bootstrap interval |
| --- | --- | --- | --- |
| [`vs-half-data`](vs-half-data) | `distill-v3-half-data` | 29 wins, 5 draws, 14 losses | 0.656, 0.552 to 0.771 |
| [`vs-v1`](vs-v1) | `distill-v1` | 30 wins, 2 draws, 16 losses | 0.646, 0.531 to 0.760 |

At about the same number of optimizer steps, the added positions gave a gain over training on `distill-v1`'s positions alone.
This checkpoint was not played against `distill-v3`; that `distill-v3` scores higher against `distill-v1` and on the held-out set suggests its second 3 epochs also helped, but no paired comparison here measures that.
[`evidence/distill-v3`](../distill-v3) has the data commands and how the `/tmp/soft/` checkpoint paths in these records were rewritten.

```bash
sed -e 's/precision: bf16/precision: fp16/' -e 's/num_workers: 12/num_workers: 4/' -e 's/epochs: 6/epochs: 3/' \
  configs/distill_gpu.yaml > distill_t4_3ep.yaml
python -m train.experiment run --config distill_t4_3ep.yaml --data ~/runs/soft-all.jsonl \
  --output-dir ~/runs/distill-v3-half-epochs/train
python -m eval.head_to_head --config configs/paired_eval.yaml \
  --checkpoint-a evidence/distill-v3-half-epochs/train/checkpoint.pt \
  --checkpoint-b evidence/distill-v3-half-data/train/checkpoint.pt --opening-suite configs/openings/paired-v1.json \
  --seeds 1,7,19 --min-pairs 8 --max-pairs 24 --output-dir evidence/distill-v3-half-epochs/vs-half-data
```
