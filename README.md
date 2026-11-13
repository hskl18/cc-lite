# cc-lite

[![tests](https://github.com/hskl18/cc-lite/actions/workflows/tests.yml/badge.svg)](https://github.com/hskl18/cc-lite/actions/workflows/tests.yml)

`cc-lite` is a compact AlphaZero-style research workbench for Chinese Chess (Xiangqi): rules, a PyTorch policy-value network, PUCT MCTS, self-play, training, and an evaluation harness.
It exists because small game-playing experiments are easy to over-read, so every reported game result comes from raw games that a validator replays move by move.

**Status: research baseline, v0.7.0.**
The first distilled checkpoint clearly beats the scripted baselines and the debug checkpoint, [`distill-v3`](evidence/distill-v3) beats it in paired games, [`distill-v4`](evidence/distill-v4) beats `distill-v3`, and [`distill-v4-large`](evidence/distill-v4-large), the same data in a larger network, scores more against Pikafish, but the six distilled checkpoints lost 315 of 360 games to Pikafish at depths 1 to 4, and the repository makes no engine-strength or absolute Elo claim.

## In plain terms

- A neural network learns Chinese Chess by imitating a strong engine: Fairy-Stockfish rates millions of positions, and the network learns which moves the engine prefers and who is ahead.
- To pick a move, a tree search looks ahead with the network, the design AlphaZero uses, at a far smaller scale: the largest network has 9.5 million parameters and trained in 35 minutes on one consumer GPU.
- Most of the work is measurement: every game is saved move by move, a validator replays it and recomputes the score, and two versions are compared on the same openings with colors swapped.
- Doubling the engine-labeled data and the training steps together, twice, the first time with richer targets, gave a version that beat the one before in paired games each time; a larger network then scored more against the Pikafish engine, though its games against the smaller one were inconclusive.
- Self-play did not show a gain, and each of the six networks tested against Pikafish at depths 1 to 4 lost most of its games, so this is a careful research setup, not a strong engine.

## Headline result

The distilled checkpoint is a 128-channel, 8-block network trained in 11 minutes on one RTX 5070 Ti from 848,307 Fairy-Stockfish depth-10 labels; the debug checkpoint is a 64-channel, 4-block network trained on one 8-ply self-play game.
Each played the two scripted opponents with the same evaluation config (50-simulation MCTS, 20 seeded games, colors alternating, 400-ply cap, CPU), on different machines: the distilled one on an AMD Ryzen 7 9800X3D, the debug one on an Apple M3 Pro.

| Opponent | Distilled (`distill-v1`) | Debug checkpoint (`baseline-v1`) |
| --- | --- | --- |
| Uniform random mover | 20 wins, 0 draws, 0 losses, 0 capped | 11 wins, 0 draws, 1 loss, 8 capped |
| One-ply greedy material mover | 19 wins, 0 draws, 1 loss, 0 capped | 0 wins, 2 draws, 15 losses, 3 capped |

In a separate paired comparison on the pinned opening suite (seeds 1, 7, and 19, 200 simulations on the GPU, both colors from every opening), the distilled checkpoint scored 43 wins, 1 draw, and 0 losses against the debug checkpoint over 22 complete pairs.
Its score rate was 0.989, with a 95 percent paired-bootstrap interval of 0.966 to 1.0, and the sequential rule stopped at 22 pairs with a superiority decision.
The references are scripted movers and an almost untrained network, so these results show that distillation works end to end, not that the network plays well.

Against a real engine it does not.

![Score on finished games for distill-v1, distill-v2, distill-v1-long, distill-v3, distill-v4, and distill-v4-large: all near 100 percent against the scripted movers, at most 45 percent against Pikafish at depth 1, at most 22.5 percent at depth 2, and at most 2.5 percent at depth 4.](docs/media/results.png)

*Wins-draws-losses from the committed bundles. The scripted movers used 50 simulations on CPU and Pikafish games used 200 on the GPU; every score is over 20 games except `distill-v4`'s at Pikafish depth 1, over the 19 that ended before the 400-ply cap. `uv run scripts/plot_results.py` redraws it.*

Later checkpoints beat the one before them twice, and self-play did not show a gain; [docs/results.md](docs/results.md) has every later comparison, the evidence bundles, and how to validate them.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && ./scripts/generate_baseline_evidence.sh
```

The script replays all 40 baseline games on CPU and validates both bundles; it took about 11 minutes on an Apple M3 Pro.
`pytest` runs the full test suite in under half a minute.

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.svg">
  <img alt="Picking a move: the board under rules profile 1 is encoded as 16 planes of 10 by 9 for a residual policy-value network, whose 8100 move logits and value guide a PUCT search that masks illegal moves and plays the chosen move. Training and evaluation: engine labels from Fairy-Stockfish or Pikafish and validated ICCS or UCCI game records feed supervised training, and self-play games feed self-play training; the checkpoint is evaluated against the scripted baselines, Pikafish, and other checkpoints on paired openings, and every evaluation writes an evidence bundle of every move, seed, and hash that a validator replays game by game. The move-picking loop plays the self-play and evaluation games." src="docs/architecture-light.svg">
</picture>

The network always scores all 8100 from-to moves, and the search masks the illegal ones, so it never plays an illegal move even when the network prefers one.
Every evaluation game lands in an evidence bundle that the validator replays before any score counts.
[docs/usage.md](docs/usage.md) maps the source folders.

## Non-goals

- A strong engine: use [Pikafish](https://github.com/official-pikafish/Pikafish), [Px0](https://github.com/official-pikafish/px0), or [Fairy-Stockfish](https://fairy-stockfish.github.io/) for analysis.
- Strength or Elo claims from the committed debug checkpoint, from capped games, or from any bundle that fails `python -m eval.evidence validate`.
- Tournament-rule compliance: results follow the narrower [CC-Lite Research Adjudication Profile 1](docs/rules-profile.md).
- Large-scale training: the configs target one laptop or one consumer GPU.

The practical path to strength is distillation from engine labels first, then AlphaZero-lite self-play, which v0.3.0 to v0.7.0 have tried without yet scoring more than 22.5 percent against Pikafish at depth 2, as laid out in [`docs/industry-roadmap.md`](docs/industry-roadmap.md).
The hard parts, the decisions behind them, and the failures that shaped them are in [DESIGN.md](DESIGN.md).

## Documentation

- [Results in detail](docs/results.md): every later checkpoint comparison, the evidence bundles, known limitations, and next steps.
- [Usage](docs/usage.md): setup on each machine, the move encoding and model, training and distillation, evaluation, metrics, and tests.
- [Design notes](DESIGN.md): the pipeline, the decisions, and the failures found and fixed.
- [Evaluation protocol](docs/evaluation-protocol.md), [data ingestion](docs/data-ingestion.md), [rules profile](docs/rules-profile.md), and the [roadmap](docs/industry-roadmap.md).

## License

[Apache-2.0](LICENSE)
