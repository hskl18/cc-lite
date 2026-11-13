# Results in detail

The later checkpoints, the attempts that did not help, every evidence bundle, and what is left to try.
The [README](../README.md) has the headline result and the chart.

## Later checkpoints

Three attempts to improve on `distill-v1` did not demonstrate a gain, and the next two did.
[`distill-v2`](../evidence/distill-v2), with fewer random moves and smoother value targets, lost their paired comparison 14-1-33.
[`distill-v1-long`](../evidence/distill-v1-long), the same data trained for 18 epochs instead of 6, lost its paired comparison 17-2-29.
[Four generations of self-play](../evidence/selfplay-v1) from `distill-v1` scored 18-4-26 against it, an interval of 0.302 to 0.521 that does not show an improvement.
[`distill-v3`](../evidence/distill-v3) added 824,481 new engine-labeled positions and spread the policy target over the engine's four best moves, and won its paired comparison 38-1-9, a score rate of 0.802 with a 95 percent paired-bootstrap interval of 0.677 to 0.906; the sequential rule was inconclusive at the 24-pair limit.
In two ablations, the soft targets on `distill-v1`'s positions alone showed no gain, while adding the new positions at about the same number of training steps beat that ablation.
What the soft targets add on the larger data, and what the extra epochs add, was not measured.
[`distill-v4`](../evidence/distill-v4) doubled the positions again, to 3,320,673, and beat `distill-v3` 38-2-8, a score rate of 0.812 with an interval of 0.719 to 0.906, again inconclusive under the sequential rule.
Its ablation at `distill-v3`'s number of training steps scored 0.594 against `distill-v3`, an interval of 0.469 to 0.719 that does not show a gain, so this time the new positions alone have not been shown to help at matched compute.
[`distill-v4-large`](../evidence/distill-v4-large) trained a 192-channel, 12-block network on `distill-v4`'s data for the same steps, with the network size as the one change.
It scored 14 points in 60 games against Pikafish where `distill-v4` scored 5.5 with one game capped, and agreed with the engine on 36.7 percent of held-out positions against 33.6.
Its paired games against `distill-v4` scored 0.594, an interval of 0.458 to 0.719 that does not show a gain.

The validator rebuilds each count from the raw games, with the same command for every bundle in the `evidence/distill-*` directories; the generation-4 bundles are a release asset described in [`evidence/selfplay-v1`](../evidence/selfplay-v1):

```bash
python -m eval.evidence validate --run-dir evidence/distill-v1/random
python -m eval.evidence validate --run-dir evidence/distill-v1/material
python -m eval.evidence validate --run-dir evidence/distill-v1/vs-debug
```

## Evidence bundles

Each bundle holds every move, seed, color, start and final FEN, termination reason, the config and checkpoint hashes, the exact command, the source commit, and the environment.
Terminal wins, draws, and losses are kept apart from games truncated at `max_plies`; material at a cutoff is diagnostic metadata and never becomes a result or a self-play label.
CI validates every committed evaluation bundle on each push.

- [`evidence/distill-v1`](../evidence/distill-v1): the distilled checkpoint, the records of its data and training, and its six evaluation bundles; its [README](../evidence/distill-v1/README.md) has the commands, and the 4 MB and 8 MB compressed positions and labels are attached to the v0.3.0 release.
- [`evidence/distill-v2`](../evidence/distill-v2): a second distilled checkpoint with six evaluation bundles; its data is attached to the v0.4.0 release.
- [`evidence/distill-v1-long`](../evidence/distill-v1-long): `distill-v1`'s data trained for 18 epochs instead of 6, with six evaluation bundles; its paired and Pikafish games ran on a Kaggle T4.
- [`evidence/distill-v3`](../evidence/distill-v3): twice `distill-v1`'s positions with MultiPV soft targets, the first checkpoint to beat `distill-v1`, with seven evaluation bundles; its two label files are attached to the v0.5.0 release.
- [`evidence/distill-v3-half-data`](../evidence/distill-v3-half-data) and [`evidence/distill-v3-half-epochs`](../evidence/distill-v3-half-epochs): the two ablations that separate `distill-v3`'s changes, with their paired bundles.
- [`evidence/distill-v4`](../evidence/distill-v4): twice `distill-v3`'s positions with its config, the first checkpoint to beat `distill-v3`, with seven evaluation bundles; its new label file is attached to the v0.6.0 release.
- [`evidence/distill-v4-half-epochs`](../evidence/distill-v4-half-epochs): `distill-v4`'s data at `distill-v3`'s number of training steps, with its paired bundle.
- [`evidence/distill-v4-large`](../evidence/distill-v4-large): `distill-v4`'s data and steps in a 192-channel, 12-block network, with six evaluation bundles.
- [`evidence/selfplay-v1`](../evidence/selfplay-v1): the record of four self-play generations; the final checkpoint and its evaluation games are attached to the v0.4.0 release, so CI does not validate those.
- [`evidence/heldout-v1`](../evidence/heldout-v1): 24,728 labeled positions no distilled checkpoint trained on, with each checkpoint's agreement report; those reports replay no games.
- [`evidence/baseline-v1`](../evidence/baseline-v1): the debug checkpoint's floor in the table above, regenerated by `./scripts/generate_baseline_evidence.sh` with [`configs/baseline_cpu.yaml`](../configs/baseline_cpu.yaml).
  A rerun of the first two games on the same machine reproduced the recorded moves exactly; other hardware or library versions can break floating-point ties differently, so the committed games stay the source of truth.
- [`evidence/debug-v1`](../evidence/debug-v1): a two-game-per-opponent pipeline smoke artifact with a 20-ply cap, regenerated by `./scripts/generate_debug_evidence.sh`.
  All four games hit the cap, so it supports no result.
  It also versions the 6.8 MB debug checkpoint that both bundles evaluate, so a fresh clone can verify the exact bytes.

[`evidence/distill-v1/vs-debug`](../evidence/distill-v1/vs-debug) is the first committed checkpoint-versus-checkpoint comparison; any further one must pass every gate in [`docs/evaluation-protocol.md`](evaluation-protocol.md).

## Known Limitations

- The supported adjudication profile is intentionally narrower than full tournament long-check and long-chase rules.
- The ingestion pipeline supports a documented JSONL ICCS/UCCI move-record envelope, not binary XQF, CBL, or every historical notation.
- Engine compatibility still requires per-engine validation even though executable and option provenance is recorded.
- Pure random-initialized self-play remains a research baseline, not the practical strength path.
- The paired opening suite is small and designed for reproducible local comparison, not comprehensive opening coverage.
- The committed debug evidence has two games per opponent and supports no strength conclusion.
- The baseline evidence starts every game from the standard position, uses one checkpoint and one simulation budget, and comes from 20 games per opponent; it is a floor, not a strength estimate.
- Throughput values are environment-specific diagnostics and are not gated or presented as stable benchmarks.

## Next Research Steps

1. Scale engine-labeled positions, training steps, and network size, the changes behind the gains so far, and find where each flattens; at matched steps `distill-v4`'s doubling did not separate from `distill-v3`, and the larger network has not yet separated from `distill-v4` in paired games.
2. Add separately tested XQF and CBL converters into the validated JSONL ingestion boundary.
3. Build a larger replay buffer with checkpoint metadata and train/eval dashboards.
4. Find out why four self-play generations showed no gain over `distill-v1`, and train on more games per generation; distillation labels come from Fairy-Stockfish because of Pikafish's weight license (see `DESIGN.md`).
5. Add repetition and check-state history planes for models that need access to adjudication context.
6. Run ablations over channels, blocks, simulations, replay size, and data source, changing one thing at a time; `distill-v2` changed two, and `distill-v3`'s two changes needed two ablations to separate.
7. Speed up self-play generation, which takes far longer than training on its output.
