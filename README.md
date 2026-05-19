# cc-lite

`cc-lite` is a small but serious AlphaZero-style research baseline for Chinese Chess / Xiangqi.
It is designed to run on a normal MacBook CPU/MPS setup or one modest consumer NVIDIA GPU.

This is not a DeepMind-scale reproduction and it does not claim engine strength. The first
milestone is a correct-enough Xiangqi rules engine, a compact PyTorch policy-value model,
model-guided MCTS, self-play data generation, checkpoint training, and baseline evaluation.

## Project Status

`cc-lite` is a research baseline, not a current industry-strength Xiangqi engine. For competitive
analysis strength, use established engines such as Pikafish:

- Pikafish: <https://github.com/official-pikafish/Pikafish>
- Px0: <https://github.com/official-pikafish/px0>
- Fairy-Stockfish: <https://fairy-stockfish.github.io/>

The intended path is practical: supervised data or Pikafish distillation first, then AlphaZero-lite
self-play. The roadmap toward a stronger benchmark-oriented system is documented in
[`docs/industry-roadmap.md`](docs/industry-roadmap.md).

## Architecture

- `src/xiangqi/`: board representation, legal move generation, FEN parsing, board encoding.
- `src/model/`: compact residual CNN policy-value network and checkpoint helpers.
- `src/mcts/`: PUCT-style MCTS using legal policy masking and value backup.
- `src/train/`: supervised training, optional engine distillation labels, self-play generation.
- `src/eval/`: random/material baselines, checkpoint-vs-checkpoint match, FEN analysis.
- `configs/`: reproducible presets for MacBook, GPU, and quick debug runs.
- `scripts/`: common smoke, training, and evaluation commands.
- `tests/`: rules, encoding, MCTS, and training smoke coverage.

## Move Encoding

The policy vocabulary is fixed at `90 x 90 = 8100` indices:

```text
policy_index = from_square * 90 + to_square
```

Squares are row-major from Black's home rank to Red's home rank. Files are `a-i`, rows are
`0-9`, so Black's left rook starts on `a0` and Red's left rook starts on `a9`.

The model always emits 8100 logits. During MCTS and training metrics, illegal moves are masked
by the current board's legal move list.

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
For CUDA machines, install the PyTorch wheel matching your driver/CUDA version first, then run
the editable install.

## Supported Machines

`cc-lite` is meant to be portable across normal development machines:

- macOS MacBook: uses Apple MPS when PyTorch exposes it, otherwise CPU.
- Linux PC: works on CPU; uses CUDA automatically when a compatible NVIDIA GPU and CUDA PyTorch
  build are installed.
- Windows PC: works on CPU or CUDA through PyTorch. The shell scripts in `scripts/` are Bash
  scripts, so run them from Git Bash or WSL, or use the explicit `python -m ...` commands below
  from PowerShell.
- Consumer NVIDIA GPU: use `configs/gpu_small.yaml` as the first non-debug profile.

The `research_debug` config is the safest first run on any machine. It is intentionally tiny and
only proves that rules, MCTS, training, checkpointing, and evaluation are wired correctly.

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
  --opponent random
```

Expected runtime: seconds to a couple of minutes depending on CPU/MPS behavior. Memory use should
stay well under normal laptop limits because the debug run uses very few plies and simulations.

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
  --opponent random
```

From Linux, WSL, or Git Bash, either run `./scripts/smoke_macbook.sh` or the Bash commands shown
in the MacBook smoke section. The script name says MacBook because it uses the tiny debug profile,
not because it is macOS-only.

## MacBook Tiny Run

```bash
./scripts/train_macbook_tiny.sh
```

This uses the tiny 4-block model, 50 MCTS simulations, small batches, and `fp32`. Expect this to
be slow compared with real engines; it is meant for reproducibility and pipeline validation.

## NVIDIA GPU Small Run

```bash
./scripts/train_gpu_small.sh
```

This uses the small model, 200 simulations, batch size 64, and CUDA mixed precision when CUDA is
available. On an RTX-class GPU, memory should remain modest because the network and replay set are
small. Increase games, replay size, and simulations only after the debug run is stable.

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
  --opponent material
```

## Supervised Bootstrap

The practical path is to bootstrap from game records before self-play. The baseline loader accepts
JSONL rows:

```json
{"fen": "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR r", "move": "a6a5", "value": 0}
```

Train from records:

```bash
export PYTHONPATH=src
python -m train.supervised \
  --config configs/macbook_tiny.yaml \
  --data data/raw/games.jsonl \
  --checkpoint-out runs/macbook_tiny/checkpoints/supervised.pt
```

Historical game ingestion is intentionally thin in this first milestone because public Xiangqi
record formats vary. Add converters into `src/data/` and emit this JSONL format.

## Optional Pikafish / Engine Distillation

If you have a UCCI-compatible engine such as Pikafish, cache labels first:

```bash
export PYTHONPATH=src
python -m train.distill_engine \
  --engine ./pikafish \
  --positions data/raw/positions.jsonl \
  --output runs/distill_labels.jsonl \
  --depth 4
```

Then train with the supervised command using `runs/distill_labels.jsonl`. Engine labels are cached
so experiments are reproducible and do not depend on querying the engine during training.

## Evaluation And Analysis

Evaluate against random or material-count baselines:

```bash
./scripts/evaluate_checkpoint.sh configs/macbook_tiny.yaml runs/macbook_tiny/checkpoints/latest.pt random
./scripts/evaluate_checkpoint.sh configs/macbook_tiny.yaml runs/macbook_tiny/checkpoints/latest.pt material
```

Evaluate against a low-depth UCCI engine such as Pikafish:

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
  --engine-depth 4
```

Engine binaries and generated engine-label datasets should stay outside git. The `.gitignore`
covers `runs/`, checkpoint files, and common ML artifacts.

Evaluate a new checkpoint against an older one:

```bash
export PYTHONPATH=src
python -m eval.head_to_head \
  --config configs/macbook_tiny.yaml \
  --checkpoint-a runs/new.pt \
  --checkpoint-b runs/old.pt
```

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

## Tests

```bash
pytest
```

The test suite covers legal move generation, move encoding, MCTS smoke behavior, and a tiny
training pass.

## Known Limitations

- Rules do not yet implement full repetition adjudication or advanced tournament draw rules.
- The supervised loader expects normalized JSONL, not every public Xiangqi format.
- The engine distillation helper assumes UCCI-style `ucci`, `position fen`, and `go depth`.
- Pure random-initialized self-play is supported as a research baseline, but it is not the
  practical strength path.
- Current evaluation is intentionally basic: random, material-count, low-depth engine if added,
  and checkpoint-vs-checkpoint matches.

## Next Research Steps

1. Add robust PGN/XQF/CBL converters into the JSONL supervised format.
2. Build a larger replay buffer with checkpoint metadata and train/eval dashboards.
3. Distill from low-depth Pikafish, then continue with AlphaZero-lite self-play.
4. Add repetition/check-state history planes and stronger draw adjudication.
5. Run ablations over channels, blocks, simulations, replay size, and data source.
6. Evaluate against low-depth Pikafish when an engine binary is configured.
