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

Status: started.

- Add optional UCI/UCCI engine benchmark command.
- Evaluate checkpoints against low-depth Pikafish instead of only random/material baselines.
- Save match metadata: engine command, depth, model checkpoint, config, seed, game count, result.
- Keep engine binaries and generated labels outside git.

Exit criterion:

- A trained checkpoint can report reproducible results against low-depth Pikafish, for example
  depth 2, 4, and 6, without manual bookkeeping.

## Stage 2: Practical Bootstrap

Status: planned.

- Add converters for public Xiangqi records into the normalized JSONL format.
- Cache Pikafish labels for policy and value distillation.
- Train supervised/distilled checkpoints before self-play.
- Track legal move accuracy, target move accuracy, value loss, and engine-match result together.

Exit criterion:

- Distilled model beats random and material baselines consistently and survives low-depth engine
  evaluation without obvious illegal-move or search failures.

## Stage 3: Search And Throughput

Status: planned.

- Batched neural-network evaluation inside MCTS.
- Multiprocess self-play workers.
- Tree reuse between moves.
- Transposition table or position cache.
- Temperature schedule and resignation rules.

Exit criterion:

- Self-play throughput is high enough to generate useful replay data overnight on a consumer GPU.

## Stage 4: Rules And Evaluation Hardening

Status: planned.

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
