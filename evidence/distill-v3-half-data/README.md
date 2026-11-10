# distill-v3-half-data: soft targets on `distill-v1`'s positions only

An ablation of [`distill-v3`](../distill-v3): the same MultiPV 4 soft targets, config, and 6 epochs, trained only on the 848,307 positions `distill-v1` used, so its training differs from `distill-v1`'s in the policy target and, like every `distill-v3` checkpoint, in running fp16 on a Kaggle T4 instead of bf16 on the RTX 5070 Ti.
Its training data is `soft-v1-labels.jsonl` from the v0.5.0 release, recorded in [`../distill-v3/data/soft-v1-labels.manifest.json`](../distill-v3/data/soft-v1-labels.manifest.json); training took 3,241 s on a Kaggle T4 from a clean checkout of `073596b`.
Records: [`train/manifest.json`](train/manifest.json), [`train/metrics.json`](train/metrics.json), and the held-out report [`../heldout-v1/agreement/distill-v3-half-data.json`](../heldout-v1/agreement/distill-v3-half-data.json).

| | distill-v1 | distill-v3-half-data |
| --- | --- | --- |
| Policy target | best move | softmax over up to four lines at 100 centipawns |
| Top-1 agreement on the 24,728 [held-out](../heldout-v1) positions | 25.8 percent | 26.0 percent |
| Value sign agreement on those positions | 93.2 percent | 93.3 percent |

[`vs-v1`](vs-v1) is the paired comparison with `distill-v1` under `configs/paired_eval.yaml` (GPU, 200 simulations, 6 sampled plies, seeds 1, 7, 19, `paired-v1` suite, up to 24 pairs): 20 wins, 4 draws, and 24 losses over 24 complete pairs, a score rate of 0.458 with a 95 percent paired-bootstrap interval of 0.333 to 0.583, and the sequential rule was inconclusive at the 24-pair limit.
On these positions the soft targets did not show a gain over the best move alone.
[`evidence/distill-v3`](../distill-v3) has the commands, the T4 config, and how the `/tmp/soft/` checkpoint paths in these records were rewritten.

```bash
python -m eval.head_to_head --config configs/paired_eval.yaml \
  --checkpoint-a evidence/distill-v3-half-data/train/checkpoint.pt \
  --checkpoint-b evidence/distill-v1/train/checkpoint.pt --opening-suite configs/openings/paired-v1.json \
  --seeds 1,7,19 --min-pairs 8 --max-pairs 24 --output-dir evidence/distill-v3-half-data/vs-v1
```
