# Industry Benchmark Roadmap

`cc-lite` is currently a local-first AlphaZero-style research baseline. It is not a
Pikafish-class engine today. The credible path toward industry-level Xiangqi strength is staged:
first make measurement reliable, then bootstrap from stronger data, then scale search and training.

## Reference Targets

- Pikafish: a strong open-source Xiangqi engine derived from Stockfish.
  <https://github.com/official-pikafish/Pikafish>
- Px0: an open-source neural-network Xiangqi engine direction under the Pikafish organization.
  <https://github.com/official-pikafish/px0>
- Fairy-Stockfish: a Stockfish-derived variant engine that includes Xiangqi support.
  <https://fairy-stockfish.github.io/>

These systems are the practical reference class. `cc-lite` should not claim comparable strength
until it can show repeatable match results against fixed-depth versions of these engines.

## Stage 0: Honest Baseline

Status: implemented.

- Legal move generation smoke tests.
- Fixed `90 x 90` from-square/to-square policy space.
- Compact policy-value network.
- Model-guided MCTS.
- Random and material-count evaluation.
- Optional UCI/UCCI engine labels for distillation.
- CI-backed test suite.

Exit criterion:

- Debug self-play, training, checkpointing, and evaluation run on a fresh clone.

## Stage 1: Measurement Against Strong Engines

Status: implemented for depths 1, 2, and 4, published in v0.4.0; depth 6 has not been run.

- Add optional UCI/UCCI engine benchmark command.
- Evaluate checkpoints against low-depth Pikafish instead of only random/material baselines.
- Save match metadata: engine command, depth, model checkpoint, config, seed, game count, result.
- Keep engine binaries and generated labels outside git, except small fixed evaluation sets such as `evidence/heldout-v1`.

Exit criterion:

- A trained checkpoint can report reproducible results against low-depth Pikafish, for example
  depth 2, 4, and 6, without manual bookkeeping.

## Stage 2: Practical Bootstrap

Status: started. v0.3.0 distills Fairy-Stockfish labels into a checkpoint that beats both scripted baselines, and v0.4.0 adds low-depth Pikafish evaluation, where its two distilled checkpoints lost 112 of 120 games; a third, `distill-v1-long`, lost 58 of 60.
v0.5.0's `distill-v3`, trained on twice the positions with MultiPV soft targets, beat `distill-v1` 38-1-9 in paired games but lost 51 of 60 to Pikafish, and record converters remain.

- Add converters for public Xiangqi records into the normalized JSONL format.
- Cache engine labels for policy and value distillation, from Fairy-Stockfish with a CC0 network, because Pikafish's weight license would bind derived checkpoints.
- Train supervised/distilled checkpoints before self-play.
- Track legal move accuracy, target move accuracy, value loss, and engine-match result together.

Exit criterion:

- Distilled model beats random and material baselines consistently and survives low-depth engine
  evaluation without obvious illegal-move or search failures.

## Stage 3: Search And Throughput

Status: started. Batched network evaluation across lockstep games, multiprocess self-play workers (`train.self_play_loop`), a temperature schedule, and a check test that no longer generates every opponent move are in place; tree reuse, a position cache, and resignation are still planned.

- Batched neural-network evaluation inside MCTS.
- Multiprocess self-play workers.
- Tree reuse between moves.
- Transposition table or position cache.
- Temperature schedule and resignation rules.

Exit criterion:

- Self-play throughput is high enough to generate useful replay data overnight on a consumer GPU.

## Stage 4: Rules And Evaluation Hardening

Status: started. v0.2.0 adds repetition and perpetual-check adjudication, a mirrored opening suite, and paired-color
bootstrap intervals; long-capture adjudication and a rating ladder are still planned.

- Repetition, long-check, and long-capture adjudication.
- Stronger FEN/engine protocol compatibility tests.
- Elo-style rating ladder with fixed opponents and confidence intervals.
- Match suites from opening positions instead of only the starting position.

Exit criterion:

- Reported strength changes are stable across seeds, openings, and side assignment.

## Stage 5: Larger Models And Ablations

Status: planned.

- Larger residual networks after data and throughput justify them.
- Policy vocabulary alternatives if `90 x 90` becomes a bottleneck.
- Value target variants: scalar, WDL, or mixed targets.
- Distillation ablations: depth, node budget, policy-only vs policy+value.

Exit criterion:

- Improvements are backed by ablation tables and engine-match results, not subjective gameplay.
