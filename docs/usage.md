# Using cc-lite

Setup on each kind of machine, the move encoding and model, training and distillation, evaluation, metrics, and tests.
The [README](../README.md) has the quick start.

## Source layout

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
MCTS masks illegal moves with the current board's legal move list.
The training metrics do not mask, so `legal_move_accuracy` reports how often the network's unmasked top move is legal.

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

Training ends with one more pass over its data that recomputes BatchNorm's running statistics with the final weights.
Checkpoints trained before [#24](https://github.com/hskl18/cc-lite/pull/24) keep the statistics gathered during training, which make evaluation mode under-report their value head ([#21](https://github.com/hskl18/cc-lite/issues/21)).

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
See [`docs/data-ingestion.md`](data-ingestion.md) for the exact input schema and license boundary.

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

Pikafish's NNUE license forbids commercial use of its weights, and of weights derived from them, without permission, so the published checkpoints use Fairy-Stockfish labels; [DESIGN.md](../DESIGN.md#distillation-from-engine-labels) explains the choice.

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

The distillation manifest records the exact engine executable hash, any file arguments, positions hash, protocol, depth, thread count, hash size, timeout, value mapping, policy target, command, and generated-label hash.
Engine labels are cached so experiments are reproducible and do not depend on querying the engine during training.

By default the policy target is the engine's best move alone.
With `--multipv K` above 1, the engine reports up to K best moves, and the policy target becomes a softmax over their scores, `exp((score - best) / T)`, where `--policy-temperature` sets `T` in centipawns (default 100).
The engine reports its lines in blocks, and every line comes from one block, the last whose line 1 has an exact score; reports marked `lowerbound` or `upperbound` count as missing.
Once a search runs past 3 seconds, Fairy-Stockfish also prints a block after each line of an iteration, where lines not yet searched again keep older scores, so taking every line from one block gives a slow machine the same lines as a fast one.
If that block is inconsistent, because line 1 is not the move the engine played, a move appears twice, or a move is illegal under the rules profile, the target falls back to the best move alone.
An illegal best move still stops the run, as before.
Mate scores count as plus or minus 100,000 centipawns, so two mating moves get equal weight whatever their distance.
Each line's move and score are kept under `multipv`, so another temperature needs no new search.
The value target comes from line 1's exact score, and is 0 if the engine reported none.

## Evaluation And Analysis

Evaluate against random or material-count baselines:

```bash
./scripts/evaluate_checkpoint.sh configs/macbook_tiny.yaml runs/macbook_tiny/checkpoints/latest.pt random
./scripts/evaluate_checkpoint.sh configs/macbook_tiny.yaml runs/macbook_tiny/checkpoints/latest.pt material
```

Evaluate against a low-depth UCI/UCCI engine such as Pikafish, pinning its separate network file so the evidence records exactly which one played:

```bash
./scripts/evaluate_engine.sh \
  configs/engine_eval.yaml \
  runs/macbook_tiny/checkpoints/latest.pt \
  ./pikafish \
  ./pikafish.nnue \
  4
```

Equivalent explicit command:

```bash
export PYTHONPATH=src
python -m eval.evaluate \
  --config configs/engine_eval.yaml \
  --checkpoint runs/macbook_tiny/checkpoints/latest.pt \
  --opponent engine \
  --engine-command ./pikafish \
  --engine-eval-file ./pikafish.nnue \
  --engine-depth 4 \
  --output-dir runs/macbook_tiny/evaluation-pikafish-depth-4
```

Engine binaries and generated engine-label datasets should stay outside Git; the exception is a small fixed evaluation set committed as evidence, such as [`evidence/heldout-v1`](../evidence/heldout-v1).
The `.gitignore` covers `runs/`, checkpoint files, and common ML artifacts.

Run the pinned opening suite with paired colors and multiple seeds:

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

The paired report includes per-seed results, paired bootstrap intervals, an uncertainty-bounded Elo difference estimate, draw and truncation rates, and a conservative sequential decision.
See [`docs/evaluation-protocol.md`](evaluation-protocol.md) before interpreting any result.

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
