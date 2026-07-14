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

The following command runs the pinned protocol and writes raw evidence:

```bash
python -m eval.head_to_head \
  --config configs/research_debug.yaml \
  --checkpoint-a path/to/candidate.pt \
  --checkpoint-b path/to/reference.pt \
  --opening-suite configs/openings/paired-v1.json \
  --seeds 1,7,19 \
  --min-pairs 8 \
  --max-pairs 24 \
  --output-dir evidence/my-paired-run
```

The debug configuration and its low simulation count are suitable for pipeline verification only.
They are not suitable for a strength claim.

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
