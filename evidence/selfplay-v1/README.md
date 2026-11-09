# selfplay-v1: four self-play generations from `distill-v1`

Four generations of self-play did not demonstrate an improvement over the checkpoint they started from.

## How it was made

`train.self_play_loop` with [`configs/selfplay_gpu.yaml`](../../configs/selfplay_gpu.yaml), from a clean checkout of `070c0fb`, on the RTX 5070 Ti machine described in [`evidence/distill-v1`](../distill-v1), started from `distill-v1` because it had just won its paired comparison with `distill-v2`.
Each generation played 448 games with 14 CPU workers (100 simulations, 200-ply cap, temperature 1 for the first 20 plies, 0.25 Dirichlet noise), then trained one epoch at a learning rate of 0.0003 on the last two generations of self-play positions plus an equal number of `distill-v1` labels.
The record is [`manifest.json`](manifest.json), with its home directory rewritten to `~/`.

| Generation | Positions | Capped games | Average plies | Average Red score | Self-play seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45,651 | 47 | 101.9 | 0.040 | 991 |
| 2 | 44,401 | 35 | 99.1 | 0.069 | 973 |
| 3 | 46,632 | 61 | 104.1 | 0.067 | 1,034 |
| 4 | 42,163 | 50 | 94.1 | 0.297 | 679 |

Training took 11 to 32 seconds per generation, so playing the games cost far more than training on them.

## Evaluation of generation 4

| Test | Generation 4 | `distill-v1` |
| --- | --- | --- |
| Paired comparison with `distill-v1`, `configs/paired_eval.yaml`, seeds 1, 7, 19, 24 pairs | 18 wins, 4 draws, 26 losses; score rate 0.417, 95 percent paired-bootstrap interval 0.302 to 0.521; sequential rule inconclusive | the opponent |
| Pikafish at depth 1, 20 games, `configs/engine_eval.yaml` | 1 win, 0 draws, 19 losses | 1 win, 0 draws, 19 losses |
| Pikafish at depth 2, 20 games | 1 win, 0 draws, 19 losses | 0 wins, 0 draws, 20 losses |
| [`heldout-v1`](../heldout-v1) top-1 agreement, illegal moves masked | 27.2 percent | 25.8 percent |
| [`heldout-v1`](../heldout-v1) value sign agreement | 92.2 percent | 93.2 percent |

The interval includes 0.5, so generation 4 cannot be called weaker than `distill-v1`, but nothing here shows that it is stronger.
These games came from a clean checkout of `321c289`, which sends the engine the game's move history.

## What is and is not here

Only the generation record is in Git, because this checkpoint did not earn a place there.
Generation 4's checkpoint, its three evaluation bundles, and its held-out report are in `selfplay-v1-gen-04.tar.xz` on the [v0.4.0 release](https://github.com/hskl18/cc-lite/releases/tag/v0.4.0), SHA-256 `580e6b14b72cf2e48374cc8edbef16f799d4deaee15b6abdef304a4b15e046c1`.
The bundles passed `python -m eval.evidence validate` on the training machine before publication; CI does not run them, because the validator reads the recorded checkpoint path literally and that file is not in the repository.
The other generations' checkpoints and self-play data were not kept.
