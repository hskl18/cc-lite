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
`python -m eval.evidence validate` replays every move from the starting FEN, checks the final FEN and terminal reason, recomputes the material diagnostics, rebuilds the summary from the raw rows, and compares the hashes of the artifacts, checkpoints, config, and opening suite.
Engine binaries are not distributed here, so their recorded hashes are checked for presence only.
Any mismatch fails validation, and CI validates every committed evaluation bundle on every push.

### Failure found and fixed: a summary that validated on only one Python version

The 20-game `baseline-v1/random` bundle validated on Python 3.12 and failed CI on 3.11.
Python 3.12 changed the builtin `sum()` over floats to compensated summation, so the mean of 20 nodes-per-second values differed in its last bit between versions, and the validator then compared summaries exactly.
PR #6 switched the float means to `math.fsum`, whose exact partial sums do not depend on the version's builtin `sum()`.
CI validates every committed evaluation bundle on both 3.11 and 3.12, which is the check that caught it.

### Failure found and fixed: a summary that validated on only some CPUs

`distill-v1-long/vs-v1`, recorded on a Kaggle machine, passed every CI job on its pull request and then failed one Python 3.12 job on `main` with the same package versions.
Its Elo interval comes from numpy's `log10` over 10,000 bootstrap samples, and numpy's math functions dispatch on CPU features, so the same version can return different last bits on different processors.
Rebuilt with the same numpy on an emulated x86 CPU without AVX-512, the interval's upper bound came out as -14.18892726488757 instead of the recorded -14.188927264887571.
The validator now accepts summary floats within a relative or absolute tolerance of 1e-12 and still compares integers, booleans, and strings exactly.
`tests/test_evidence.py::test_validator_accepts_last_bit_float_differences_but_not_edits` checks both sides.

## Paired evaluation and stopping

Checkpoint-versus-checkpoint comparisons use a pinned suite of mirrored openings.
Each opening and seed is one pair: checkpoint A plays Red, then Black, from the same position with the same seed.
A pair counts only when both legs reach a terminal result, so a capped leg cannot tilt the color balance.
Score intervals come from a percentile bootstrap over complete pairs, and the sequential stop uses a Hoeffding confidence sequence with a union bound over repeated looks, so stopping early does not inflate the error rate.
The first result is `evidence/distill-v1/vs-debug`: the distilled checkpoint against the debug checkpoint, 43 wins, 1 draw, and 0 losses over 22 complete pairs, stopped as superior.

### Failure found and fixed: seeds that changed nothing

Before the first real comparison, a read of `eval/head_to_head.py` showed that evaluation moves are greedy (temperature 0) with no root noise, so `set_seed` had nothing to act on.
Seeds 1, 7, and 19 would have replayed the same games from each opening, and the bootstrap would have counted three identical pairs as three independent ones, narrowing the interval for free.
`eval.sample_plies` now draws each game's first moves from MCTS visit counts, and `head_to_head` refuses a multi-seed run with `sample_plies` of 0.
The sampling stream is seeded from the pair identifier rather than the bare seed; with the bare seed, an opening and its mirror consumed identical random draws, which would have coupled pairs that the statistics treat as independent.
Tests check that two seeds play different games and that moves after the sampled plies are greedy.

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

## Distillation from engine labels

v0.3.0 trains the first distilled checkpoint: an engine picks the move and scores the position, and the network learns both.

**Which engine.** The Pikafish NNUE license forbids commercial use without permission of its weights and of weights derived from them, and a distilled checkpoint could count as derived, while this repository is Apache-2.0.
The same license states that the xiangqi networks the Pikafish developers train for Fairy-Stockfish are CC0, so labels come from Fairy-Stockfish with one of those networks built in, and Pikafish remains usable as an evaluation opponent.
Fairy-Stockfish numbers ranks 1 to 10, with a two-character rank 10, and plays chess until its xiangqi variant is selected; the adapter's `fairy` flavor handles both and refuses the UCCI protocol, where the engine switches rank base.

**Positions.** The engine plays itself at depth 2, and 15 percent of moves are uniformly random, so the data covers more than one line; the eight evaluation openings are excluded, so the paired comparison starts from unseen positions.
12,000 games gave 848,307 unique positions in four minutes on 14 processes.

**Labels.** Each position is searched at depth 10 with the engine state cleared first.
Without the clear, a process's earlier searches leave hash and history state behind, so a label could depend on which positions shared a process; with it, 1 and 14 workers produce byte-identical labels.
The value target is the side-to-move score divided by 1,000 and clipped.

**What it bought.** After eleven minutes of training on an RTX 5070 Ti, the distilled checkpoint won 19 of 20 games against the one-ply material grabber, which had beaten the debug checkpoint in 15 of 20, and the paired comparison against the debug checkpoint stopped as superior after 22 pairs.
The two networks differ in size and training data, so this compares checkpoints, not training alone.

**What it did not.** The random moves make many games lopsided, so 44 percent of value targets sit at the clip and the value head learns little about the size of a large advantage.
The training metrics are averaged over the training data, not measured on held-out positions.
Against Pikafish, with the engine sent the game's move history, this checkpoint lost 59 of 60 games at depths 1, 2, and 4.
Its one win was a perpetual-check adjudication against the engine.
Distillation works end to end, and the network still plays far below a shallow engine search.

**What has not shown a gain.** `distill-v2` changed two things at once, fewer random moves and `tanh(score / 800)` value targets, and lost the paired comparison with `distill-v1` 14-1-33.
Four self-play generations from `distill-v1` scored 18-4-26 against it, with an interval of 0.302 to 0.521.
`distill-v1-long` trained on `distill-v1`'s data for 18 epochs instead of 6: its top move matched the engine's on 39.7 percent of training positions averaged over its epochs, against 24.9 percent over `distill-v1`'s, held-out agreement went from 25.8 to 25.6 percent, and it lost the paired comparison 17-2-29, so these extra epochs did not help.
`distill-v2` and `distill-v1-long` matched `distill-v1` on held-out agreement and still lost their games, so agreement on that set did not predict playing strength.

## Search throughput for self-play

Self-play is search-bound, and profiling one batched worker on the RTX 5070 Ti machine put 300 of 328 seconds inside `Board.is_in_check`.
It generated every opponent move to test one king square, and legal-move filtering called it once per candidate move.
`is_in_check` now looks outward from the king, with one branch per piece that mirrors that piece's move generator, including the flying-general direction.
On all 848,307 `distill-v1` positions, check status for both colors and every legal-move list (25.2 million moves) matched the original definition with no mismatch; the original stays in the tests as the reference.
The first two `baseline-v1` games replay move for move on the machine that recorded them and run three times faster end to end.

MCTS is now a generator that yields each board it needs evaluated, so `train.self_play` can run many games in lockstep and send one leaf per game to the GPU in a single batch, and `train.self_play_loop` runs generations across worker processes, each anchored to a sample of engine labels.

## Known limits

- The repetition rule does not classify long chases, so some tournament outcomes differ.
- No checkpoint here has scored above 25 percent against Pikafish, even at depth 1, and the engine matches use 20 games per setting, so differences between checkpoints at this level are noise-sized.
- Torch inference on different hardware or library versions can change floating-point ties, so a rerun elsewhere may choose different moves; the committed raw games remain the source of truth.
