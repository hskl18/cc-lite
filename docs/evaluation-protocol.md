# Evaluation protocol

This document defines the reproducible checkpoint comparison protocol introduced in cc-lite v0.2.0.
It is an experiment protocol, not evidence that any released checkpoint is stronger than another engine or checkpoint.

## Pinned opening suite

The canonical local suite is `configs/openings/paired-v1.json`.
The suite is versioned, validated before play, and hashed into an evidence manifest.
Every position was produced by legal moves from the cc-lite starting position.
Every base position has exactly one position reflected across the central file.
The loader rejects malformed positions, missing kings, terminal positions, duplicate identifiers, missing mirrors, extra mirrors, and mirrors that do not exactly match the declared transformation.

The suite is intentionally small enough for local regression work.
It does not represent the breadth of competitive Xiangqi opening theory.
A result from this suite must name the suite identifier, version, and SHA-256 hash.

## Paired colors and seeds

Each opening and seed defines one pair of games.
Checkpoint A plays Red in leg one and Black in leg two from the identical FEN with the identical random seed.
The pair is the statistical resampling unit.
The canonical default seeds are `1`, `7`, and `19`.
A schedule runs every declared seed for one opening before advancing to the next opening, so early stopping prefixes remain balanced across seeds.
A publishable comparison must use at least two distinct seeds, while a local smoke run may use one seed if it is labeled as a smoke run.

Evaluation moves are greedy and add no root noise, so a seed changes a game only through `eval.sample_plies`.
The first `sample_plies` moves of every game are drawn in proportion to MCTS visit counts, and every later move is greedy.
The sampling stream is seeded from the pair identifier, which names the suite, opening, and seed, so both legs of a pair share it and no two pairs do.
`head_to_head` refuses a run with more than one seed and `sample_plies` of 0, because every seed would replay the same games and the bootstrap would count the copies as independent pairs.

The following command runs the pinned protocol and writes raw evidence:

```bash
python -m eval.head_to_head \
  --config configs/paired_eval.yaml \
  --checkpoint-a path/to/candidate.pt \
  --checkpoint-b path/to/reference.pt \
  --opening-suite configs/openings/paired-v1.json \
  --seeds 1,7,19 \
  --min-pairs 8 \
  --max-pairs 24 \
  --output-dir evidence/my-paired-run
```

`configs/paired_eval.yaml` sets 200 simulations, a 400-ply cap, and 6 sampled plies.
`configs/research_debug.yaml` and its low simulation count are suitable for one-seed pipeline verification only, not for a strength claim.

## Result accounting

Terminal win, draw, and loss counts use `terminal_score` from checkpoint A's perspective.
Rules-profile adjudications, including repetition and no-progress outcomes, are terminal results.
The evaluator's `max_plies` limit is an experiment resource cap and is not a game result.
A capped game records `terminal_reason` as `max_plies`, leaves `terminal_score` null, and records material only as a separate diagnostic.
Material diagnostics never enter win, draw, loss, score-rate, Elo, bootstrap, or sequential-stop calculations.

A color pair is complete only when both legs have terminal results, one with checkpoint A as Red and one with checkpoint A as Black.
If either leg is capped or missing, the pair remains visible as incomplete and is excluded from paired inference.
This conservative rule prevents a favorable material balance at the cap from becoming a win and prevents a partial color pair from biasing the comparison.

## Statistical report

The score rate awards one point for a win, one half point for a draw, and zero points for a loss.
The point estimate is computed across the terminal games in complete color pairs.
The confidence interval is a percentile bootstrap that resamples complete color pairs with replacement.
The implementation uses 10,000 bootstrap samples, a fixed bootstrap seed of `0`, and a default confidence level of 95 percent.
The fixed analysis seed makes summary reconstruction deterministic.

The Elo difference uses the logistic Elo transform of score rate.
A Beta one-half continuity correction keeps the estimate finite when every completed game has the same result.
The Elo interval transforms each paired-bootstrap score sample with the same correction.
The report must retain the method label because this small-sample estimate is not interchangeable with a maximum-likelihood Elo estimate from a large tournament.

Draw rate is reported separately from truncation rate.
Per-seed score rates and complete-pair counts are included so seed concentration is visible.

## Sequential stopping

The sequential rule is evaluated only after both legs of a scheduled pair have been written.
It requires the configured minimum number of complete pairs and minimum number of distinct seeds before declaring superiority or inferiority.
It uses bounded pair scores and a Hoeffding confidence sequence with a union-bound allocation across repeated looks.
Checkpoint A is called superior only when the lower bound is above `0.5 + margin`.
Checkpoint A is called inferior only when the upper bound is below `0.5 - margin`.

Reaching the configured pair cap without crossing either boundary stops the run as inconclusive.
Truncated pairs count toward the resource cap but not toward completed-pair evidence.
The protocol has no rule that declares a capped game or capped run a win.

## Engine opponents

`python -m eval.evaluate --opponent engine` plays a checkpoint against a UCI engine at a fixed depth, with [`configs/engine_eval.yaml`](../configs/engine_eval.yaml) as the reference setting.
The engine is deterministic and the model's later moves are greedy, so `eval.sample_plies` must be above 0 for more than two games; it counts the model's own moves, so Black is sampled too, and the evaluator refuses otherwise and clears the engine's search state before every game.
Before every engine move the adapter sends the whole game (`position fen <start> moves ...`), because an engine given only the current position cannot see a repetition or a perpetual check and loses games to adjudication that it would otherwise avoid.
The manifest's `opponent_engine` records the command, the executable's path and SHA-256, the flavor, the depth, the protocol, and `sees_game_history`.
Pikafish loads its network from a separate file, so a Pikafish opponent must be given `--engine-eval-file`, and the record pins that file's path and SHA-256 too.
Engine binaries are not distributed with this repository, so the validator checks that the record is complete but cannot recompute the hash.

## Held-out label agreement

`python -m eval.label_agreement` compares a checkpoint with engine labels on positions it was not trained on, built with `train.engine_positions --exclude-positions` and labeled by `train.distill_engine`.
It reports top-1 agreement with illegal moves masked, as in MCTS, and the share of positions whose unmasked top move is legal.
Value sign agreement counts only labels at least 0.1 from zero; a model's sign does not depend on the mapping it was trained with, so checkpoints trained on different mappings can be compared on it, as long as both are measured on the same held-out labels with the same threshold.
Value MSE depends on the mapping and compares only checkpoints trained on the same one.

## Evidence validation and claims

Validate a written bundle with:

```bash
python -m eval.evidence validate --run-dir evidence/my-paired-run
```

Every raw game must include a typed `start_fen`, `moves`, `plies`, and `final_fen` replay chain.
Validation replays every move, requires `plies` to equal the raw move count, verifies the final FEN, and confirms the declared terminal or truncation state.
Unpaired records must declare a valid `model_color` perspective, while paired records must declare a valid `candidate_color` perspective.
If both color fields are present they must agree, and missing, null, or invalid perspective fields fail validation.
For a `max_plies` record, validation recomputes `material_delta` from the replayed board and derives `material_adjudication` from the shared adjudication rule.
Published material fields must match those reconstructed values exactly.
For paired evidence, validation rebuilds the canonical schedule from the pinned opening suite and manifest seeds, then requires the raw games to be an exact complete-pair prefix within `paired_protocol.max_pairs`.
The presence of any paired record metadata requires a complete paired protocol, a pinned opening suite, and manifest seeds.
Partially paired records cannot fall back to unpaired validation.
The positional comparison covers deterministic game and pair identifiers, pair legs, opening identifiers, starting FENs, seeds, and candidate colors, so skipped, reordered, duplicated, or extra schedule slots fail closed.
An early-stopped bundle must end on the first complete-pair boundary where the reconstructed sequential rule stops, and its summary must match the reconstructed stop decision and attempt counts.
Validation also reconstructs the remaining summary fields, verifies artifact hashes and provenance hashes, checks pair invariants, and rejects a terminal score attached to a `max_plies` record.
The evidence manifest records both checkpoint hashes, the config hash, the opening-suite hash, all evaluation seeds, the source commit, the command line, and the runtime environment.

Do not publish an Elo or strength claim from an incomplete run, a one-seed run, a debug simulation budget, or an invalid evidence bundle.
When an external reference engine or checkpoint is unavailable, report the local protocol validation separately and name the missing resource as a blocker.
