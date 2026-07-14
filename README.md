# cc-lite

`cc-lite` is a small but serious AlphaZero-style research baseline for Chinese Chess / Xiangqi.
It is designed to run on a normal MacBook CPU/MPS setup or one modest consumer NVIDIA GPU.

Version 0.2.0 is a rules-pinned, reproducible Xiangqi research workbench with validated record ingestion, bounded supervised bootstrap experiments, paired opening evaluation, and statistical reporting.
It is not a DeepMind-scale reproduction and it does not claim engine strength.

## Project Status

`cc-lite` is a research baseline, not a current industry-strength Xiangqi engine.
For competitive analysis strength, use established engines such as Pikafish:

- Pikafish: <https://github.com/official-pikafish/Pikafish>
- Px0: <https://github.com/official-pikafish/px0>
- Fairy-Stockfish: <https://fairy-stockfish.github.io/>

The intended path is practical: supervised data or Pikafish distillation first, then AlphaZero-lite self-play.
The roadmap toward a stronger benchmark-oriented system is documented in [`docs/industry-roadmap.md`](docs/industry-roadmap.md).

## Evidence Status

No engine-strength result is currently claimed.
The earlier hand-recorded debug table was removed because it had no committed raw games, checkpoint hash, exact command manifest, or engine binary provenance.
Its capped games were also material-adjudicated and therefore were not terminal wins and losses.

The committed [`evidence/debug-v1`](evidence/debug-v1) bundle is a pipeline smoke artifact only.
It records every move, seed, color, start and final FEN, termination reason, config and checkpoint hashes, exact command arguments, source commit, environment, and raw-derived summary.
The 6.8 MB debug checkpoint is versioned with the bundle so a fresh clone can verify the exact bytes used by both matches.
Terminal W/D/L is kept separate from games truncated at `max_plies`.
Material state at a cutoff is diagnostic metadata and never becomes a terminal result or a self-play win label.

Regenerate the same evidence structure with:

```bash
./scripts/generate_debug_evidence.sh
```

Validate each bundle independently from its raw `games.jsonl`:

```bash
python -m eval.evidence validate --run-dir evidence/debug-v1/random
python -m eval.evidence validate --run-dir evidence/debug-v1/material
```

This two-game-per-opponent debug protocol is intentionally too small for a strength inference.
Version 0.2.0 supplies the protocol and tooling needed for a stronger result, but no qualifying paired multi-seed result is committed yet.
Any future publishable claim must still pass every evidence gate documented in [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md).

## Architecture

- `src/xiangqi/`: board representation, legal move generation, FEN parsing, board encoding, and pinned adjudication behavior.
- `src/data/`: validated ICCS/UCCI record ingestion, provenance, deduplication, split fingerprints, and rejection reports.
- `src/model/`: compact residual CNN policy-value network and checkpoint helpers.
- `src/mcts/`: PUCT-style MCTS using legal policy masking and value backup.
- `src/train/`: bounded supervised experiments, optional engine distillation labels, and self-play generation.
- `src/eval/`: random and material baselines, checkpoint matches, paired opening evaluation, statistical reporting, and FEN analysis.
- `configs/`: reproducible presets for MacBook, GPU, and quick debug runs.
- `scripts/`: common smoke, training, and evaluation commands.
- `tests/`: rules, encoding, MCTS, and training smoke coverage.

## Move Encoding

The policy vocabulary is fixed at `90 x 90 = 8100` indices:

```text
policy_index = from_square * 90 + to_square
```

Squares are row-major from Black's home rank to Red's home rank.
Files are `a-i`, rows are `0-9`, so Black's left rook starts on `a0` and Red's left rook starts on `a9`.

The model always emits 8100 logits.
During MCTS and training metrics, illegal moves are masked by the current board's legal move list.

## Model

The baseline model is a small residual CNN:

- Input: 16 planes over a `10 x 9` board.
- Piece planes: 7 red piece types and 7 black piece types.
- Extra planes: side to move and current-side-in-check.
- Policy head: logits over 8100 from-to moves.
- Value head: scalar in `[-1, 1]` from the side-to-move perspective.

Presets:

- `tiny`: 4 residual blocks, 64 channels.
- `small`: 6 residual blocks, 96 channels by default.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If PyTorch is already installed for your platform, the editable install is usually enough.
For CUDA machines, install the PyTorch wheel matching your driver/CUDA version first, then run the editable install.

## Supported Machines

`cc-lite` is meant to be portable across normal development machines:

- macOS MacBook: uses Apple MPS when PyTorch exposes it, otherwise CPU.
- Linux PC: works on CPU and uses CUDA automatically when a compatible NVIDIA GPU and CUDA PyTorch build are installed.
- Windows PC: works on CPU or CUDA through PyTorch.
  The shell scripts in `scripts/` are Bash scripts, so run them from Git Bash or WSL, or use the explicit `python -m ...` commands below from PowerShell.
- Consumer NVIDIA GPU: use `configs/gpu_small.yaml` as the first non-debug profile.

The `research_debug` config is the safest first run on any machine.
It is intentionally tiny and only proves that rules, MCTS, training, checkpointing, and evaluation are wired correctly.

## Smoke Training On MacBook

Runs one very short self-play game, trains one checkpoint, and evaluates against random:

```bash
./scripts/smoke_macbook.sh
```

Equivalent explicit commands:

```bash
export PYTHONPATH=src
python -m train.self_play \
  --config configs/research_debug.yaml \
  --output runs/research_debug/selfplay.jsonl \
  --train \
  --checkpoint-out runs/research_debug/checkpoints/selfplay.pt

python -m eval.evaluate \
  --config configs/research_debug.yaml \
  --checkpoint runs/research_debug/checkpoints/selfplay.pt \
  --opponent random \
  --output-dir runs/research_debug/evaluation-random
```

Expected runtime is seconds to a couple of minutes depending on CPU/MPS behavior.
Memory use should stay well under normal laptop limits because the debug run uses very few plies and simulations.

## Smoke Training On Windows Or Linux PC

From PowerShell on Windows, use the module commands directly:

```powershell
$env:PYTHONPATH = "src"
python -m train.self_play `
  --config configs/research_debug.yaml `
  --output runs/research_debug/selfplay.jsonl `
  --train `
  --checkpoint-out runs/research_debug/checkpoints/selfplay.pt

python -m eval.evaluate `
  --config configs/research_debug.yaml `
  --checkpoint runs/research_debug/checkpoints/selfplay.pt `
  --opponent random `
  --output-dir runs/research_debug/evaluation-random
```

From Linux, WSL, or Git Bash, either run `./scripts/smoke_macbook.sh` or the Bash commands shown in the MacBook smoke section.
The script name says MacBook because it uses the tiny debug profile, not because it is macOS-only.

## MacBook Tiny Run

```bash
./scripts/train_macbook_tiny.sh
```

This uses the tiny 4-block model, 50 MCTS simulations, small batches, and `fp32`.
Expect this to be slow compared with real engines because it is meant for reproducibility and pipeline validation.

## NVIDIA GPU Small Run

```bash
./scripts/train_gpu_small.sh
```

This uses the small model, 200 simulations, batch size 64, and CUDA mixed precision when CUDA is available.
On an RTX-class GPU, memory should remain modest because the network and replay set are small.
Increase games, replay size, and simulations only after the debug run is stable.

Windows users can run the same workflow from Git Bash/WSL, or translate it to PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m train.self_play `
  --config configs/gpu_small.yaml `
  --output runs/gpu_small/selfplay.jsonl `
  --train `
  --checkpoint-out runs/gpu_small/checkpoints/latest.pt

python -m eval.evaluate `
  --config configs/gpu_small.yaml `
  --checkpoint runs/gpu_small/checkpoints/latest.pt `
  --opponent material `
  --output-dir runs/gpu_small/evaluation-material
```

## Validated Record Ingestion

The practical path is to bootstrap from game records before self-play.
The ingestion CLI accepts documented ICCS/UCCI JSONL records and requires a separate provenance file with source and license information.

```bash
python -m data.ingest \
  --input path/to/games.jsonl \
  --provenance path/to/provenance.json \
  --output-dir runs/ingested/my-dataset \
  --validation-fraction 0.1 \
  --split-seed research-v1
```

The output contains normalized games, deterministic training and validation splits, a rejection report, and a hash-linked manifest.
See [`docs/data-ingestion.md`](docs/data-ingestion.md) for the exact input schema and license boundary.

## Bounded Supervised Bootstrap

The `bootstrap_debug` profile caps the number of records and writes a checkpoint, raw metrics, and a manifest with exact data, config, and checkpoint hashes.
The bundle is published to its final output directory only after its evidence validator passes.

```bash
python -m train.experiment run \
  --config configs/bootstrap_debug.yaml \
  --data runs/ingested/my-dataset/train.jsonl \
  --output-dir runs/bootstrap/debug

python -m train.experiment validate \
  --output-dir runs/bootstrap/debug
```

## Optional Pikafish / Engine Distillation

If you have a UCI/UCCI-compatible engine such as Pikafish, cache labels first:

```bash
export PYTHONPATH=src
python -m train.distill_engine \
  --engine ./pikafish \
  --positions data/raw/positions.jsonl \
  --output runs/distill_labels.jsonl \
  --depth 4 \
  --threads 1 \
  --hash-mb 16
```

The distillation manifest records the exact engine executable hash, any file arguments, positions hash, protocol, depth, thread count, hash size, timeout, command, and generated-label hash.
Engine labels are cached so experiments are reproducible and do not depend on querying the engine during training.

## Evaluation And Analysis

Evaluate against random or material-count baselines:

```bash
./scripts/evaluate_checkpoint.sh configs/macbook_tiny.yaml runs/macbook_tiny/checkpoints/latest.pt random
./scripts/evaluate_checkpoint.sh configs/macbook_tiny.yaml runs/macbook_tiny/checkpoints/latest.pt material
```

Evaluate against a low-depth UCI/UCCI engine such as Pikafish:

```bash
./scripts/evaluate_engine.sh \
  configs/macbook_tiny.yaml \
  runs/macbook_tiny/checkpoints/latest.pt \
  ./pikafish \
  4
```

Equivalent explicit command:

```bash
export PYTHONPATH=src
python -m eval.evaluate \
  --config configs/macbook_tiny.yaml \
  --checkpoint runs/macbook_tiny/checkpoints/latest.pt \
  --opponent engine \
  --engine-command ./pikafish \
  --engine-depth 4 \
  --output-dir runs/macbook_tiny/evaluation-pikafish-depth-4
```

Engine binaries and generated engine-label datasets should stay outside Git.
The `.gitignore` covers `runs/`, checkpoint files, and common ML artifacts.

Run the pinned opening suite with paired colors and multiple seeds:

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

The paired report includes per-seed results, paired bootstrap intervals, an uncertainty-bounded Elo difference estimate, draw and truncation rates, and a conservative sequential decision.
See [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) before interpreting any result.

Analyze a position:

```bash
export PYTHONPATH=src
python -m eval.analyze \
  --config configs/research_debug.yaml \
  --checkpoint runs/research_debug/checkpoints/selfplay.pt \
  --fen "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR r"
```

## Metrics

The current scripts log:

- policy loss
- value loss
- legal move accuracy
- target move accuracy
- result score vs random/material baseline
- MCTS nodes/s
- checkpoint metadata with config and train stats
- per-game moves, seeds, colors, FENs, and termination reasons
- terminal results separately from max-plies truncation diagnostics
- evidence manifests with config, checkpoint, source, command, and environment provenance
- paired-color score and draw intervals from complete opening pairs
- Elo difference estimates with uncertainty and explicit method labels
- sequential stopping decisions that exclude capped and incomplete pairs

## Tests

```bash
pytest
```

The test suite covers the pinned rules profile, randomized legal-game reversibility, ingestion, move encoding, MCTS, bounded training, paired evaluation, statistics, evidence reconstruction, and package metadata.

## Known Limitations

- The supported adjudication profile is intentionally narrower than full tournament long-check and long-chase rules.
- The ingestion pipeline supports a documented JSONL ICCS/UCCI move-record envelope, not binary XQF, CBL, or every historical notation.
- Engine compatibility still requires per-engine validation even though executable and option provenance is recorded.
- Pure random-initialized self-play remains a research baseline, not the practical strength path.
- The paired opening suite is small and designed for reproducible local comparison, not comprehensive opening coverage.
- The committed debug evidence has two games per opponent and supports no strength conclusion.
- Throughput values are environment-specific diagnostics and are not gated or presented as stable benchmarks.

## Next Research Steps

1. Add separately tested XQF and CBL converters into the validated JSONL ingestion boundary.
2. Build a larger replay buffer with checkpoint metadata and train/eval dashboards.
3. Distill from low-depth Pikafish, then continue with AlphaZero-lite self-play.
4. Add repetition and check-state history planes for models that need access to adjudication context.
5. Run ablations over channels, blocks, simulations, replay size, and data source.
6. Complete a qualifying paired multi-seed comparison when a reference checkpoint or engine is available.
