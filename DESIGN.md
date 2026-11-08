# Design notes

This document explains the parts of cc-lite that are easy to get wrong, the decisions behind them, and the failures that shaped them.
The README covers setup and commands; [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) and [`docs/rules-profile.md`](docs/rules-profile.md) hold the full specifications.

## Pipeline

```text
Board (rules profile 1) -> encode_board (16 x 10 x 9 planes) -> PolicyValueNet
        ^                                                        |
        |                                  8100 logits + value in [-1, 1]
        |                                                        v
   push legal move <- MCTS (PUCT, legal masking, terminal checks) <-
        |
        v
self-play samples -> training -> checkpoint -> evaluation -> evidence bundle -> validator
```

## Fixed move space with legal masking

The policy head always emits `90 x 90 = 8100` logits, one per from-square and to-square pair.
Most indices are never legal, so the network never chooses a move directly.
`TorchEvaluator` takes a softmax over only the current position's legal moves, and MCTS expands only those children.
The fixed space keeps the network shape independent of the position, at the cost of spending most of the output layer on impossible moves.

## One pinned rules profile

Xiangqi rulebooks disagree on long checks, long chases, repetition thresholds, and move counters.
cc-lite implements one deterministic profile instead of claiming tournament compliance: three-fold repetition, a single perpetual checker loses, 120 plies without a capture or pawn move is a draw, and a side with no legal move loses.
Every result is labeled as a CC-Lite Research Adjudication Profile 1 result.
The randomized tests in `tests/test_rules_profile.py` push and pop up to eight sampled legal moves at each position of eight seeded random games, checking that the board, repetition history, and no-progress counter return to the same state.

## Terminal results versus capped games

Self-play and evaluation stop at an experiment-level `max_plies` cap.
A game stopped there has no winner under the rules, so its `terminal_score` stays null and it is reported under truncation, never under wins, draws, or losses.
The material balance at the cap is kept as a diagnostic field only.

### Failure found and fixed: material at the cap counted as results

Version 0.1 did not keep that boundary.
The README at commit `1002db3` reported `2W / 1D / 1L` for the debug checkpoint against the random mover with an average of 40.0 plies under a 40-ply cap.
Every one of those games hit the cap, so the "wins" were material adjudications rather than games anyone won.
The same version labeled capped self-play games as wins or losses whenever one side was ahead by 20 or more material points, so training targets contained outcomes that never happened.

PR #1 (`678726a`) fixed both.
Capped evaluation games now carry a null `terminal_score`, capped self-play games now carry a value target of 0, and the validator rejects any bundle that attaches a terminal score to a `max_plies` record.
The hand-recorded table was deleted because it had no raw games, checkpoint hash, or exact command behind it.
`tests/test_training.py::test_cutoff_does_not_turn_material_advantage_into_self_play_win` guards the self-play half of the fix.

## Evidence bundles that fail closed

Every evaluation writes `games.jsonl`, `summary.json`, and `manifest.json`.
The manifest records the checkpoint and config SHA-256 hashes, the exact command, the seed, the source commit and dirty flag, and the runtime environment.
`python -m eval.evidence validate` replays every move from the starting FEN, checks the final FEN and terminal reason, recomputes the material diagnostics, rebuilds the summary from the raw rows, and compares every hash.
Any mismatch fails validation, and CI validates every committed bundle on every push.

### Failure found and fixed: a summary that validated on only one Python version

The 20-game `baseline-v1/random` bundle validated on Python 3.12 and failed CI on 3.11.
Python 3.12 changed the builtin `sum()` over floats to compensated summation, so the mean of 20 nodes-per-second values differed in its last bit between versions, and the validator compares summaries exactly.
PR #6 switched the float means to `math.fsum`, which is correctly rounded on every version and reproduces every committed summary bit for bit.
CI validates every bundle on both 3.11 and 3.12, which is the check that caught it.

## Paired evaluation and stopping

Checkpoint-versus-checkpoint comparisons use a pinned suite of mirrored openings.
Each opening and seed is one pair: checkpoint A plays Red, then Black, from the same position with the same seed.
A pair counts only when both legs reach a terminal result, so a capped leg cannot tilt the color balance.
Score intervals come from a percentile bootstrap over complete pairs, and the sequential stop uses a Hoeffding confidence sequence with a union bound over repeated looks, so stopping early does not inflate the error rate.
No checkpoint pair in this repository has yet produced enough terminal pairs for a result.

## Baseline floor: what an almost untrained agent does

[`evidence/baseline-v1`](evidence/baseline-v1) evaluates the committed debug checkpoint, trained on one 8-ply self-play game, with 50 MCTS simulations per move.
It plays 20 seeded games against each scripted opponent from the standard position, alternating colors, with a 400-ply cap, on CPU.

| Opponent | Wins | Draws | Losses | Capped, agent behind on material |
| --- | ---: | ---: | ---: | ---: |
| Uniform random mover | 11 | 0 | 1 | 8 of 8 |
| One-ply greedy material mover | 0 | 2 | 15 | 3 of 3 |

All 11 wins ended with the random side out of legal moves, which a search that checks terminal states can find even with a near-uniform prior.
The agent has no sense of material: it was behind in every capped game against both opponents, and the greedy mover, which always takes the largest capture, won 15 games and lost none.
That makes the capped column part of the result.
Counting only finished games would report 11 of 12 against the random mover and hide eight games the agent was losing on material.

The result is specific to this checkpoint and the 50-simulation budget.
A short uncommitted timing probe at a different budget gave different outcomes for the same seeds, so do not extrapolate across budgets.
With about 30 legal moves per position in these games, a 50-simulation budget rarely searches the opponent's reply, so the agent cannot see a mate coming.

## Security decision: weights-only checkpoint loading

`load_checkpoint` used `torch.load(..., weights_only=False)`, which unpickles arbitrary objects, so a crafted `.pt` file could run code when evaluated.
Every checkpoint writer in `src/train` stores only tensors and plain Python values, so loading now uses `weights_only=True`.

## Known limits

- The repetition rule does not classify long chases, so some tournament outcomes differ.
- The only committed checkpoint is a debug artifact; there is no trained model of useful strength.
- Torch inference on different hardware or library versions can change floating-point ties, so a rerun elsewhere may choose different moves; the committed raw games remain the source of truth.
