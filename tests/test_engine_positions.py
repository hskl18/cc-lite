from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from eval.engine import UcciEngine
from train.engine_positions import GameSettings, play_games, unique_positions
from xiangqi.board import START_FEN

# Answers like Fairy-Stockfish: chess until the xiangqi variant is selected, then
# a central cannon move in one-based ranks.
FAKE_FAIRY = """
import sys
variant = "chess"
for line in sys.stdin:
    command = line.strip()
    if command == "uci":
        print("uciok", flush=True)
    elif command.startswith("setoption name UCI_Variant value "):
        variant = command.rsplit(" ", 1)[1]
    elif command == "isready":
        print("readyok", flush=True)
    elif command.startswith("go"):
        move = "b3e3" if variant == "xiangqi" else "e2e4"
        print("info depth 1 score cp 50 pv " + move)
        print("bestmove " + move, flush=True)
    elif command == "ucinewgame":
        pass
    elif command == "quit":
        break
"""


def fake_engine(tmp_path: Path) -> str:
    script = tmp_path / "fake_fairy.py"
    script.write_text(FAKE_FAIRY, encoding="utf-8")
    return f'{sys.executable} "{script}"'


def test_fairy_flavor_selects_xiangqi_and_converts_ranks(tmp_path) -> None:
    engine = UcciEngine(fake_engine(tmp_path), timeout=5.0, flavor="fairy")
    try:
        result = engine.best_move(START_FEN, 1)
    finally:
        engine.close()
    assert result.move.uci() == "b7e7"
    assert result.score_cp == 50


def test_parallel_labels_match_a_single_worker_in_order(tmp_path) -> None:
    positions = tmp_path / "positions.jsonl"
    positions.write_text("".join(json.dumps({"fen": START_FEN}) + "\n" for _ in range(9)), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    outputs = []
    for workers in ("1", "3"):
        output = tmp_path / f"labels-{workers}.jsonl"
        subprocess.run(
            [sys.executable, "-m", "train.distill_engine", "--engine", fake_engine(tmp_path),
             "--engine-flavor", "fairy", "--positions", str(positions), "--output", str(output),
             "--depth", "1", "--workers", workers],
            check=True, env=env, capture_output=True,
        )
        outputs.append(output.read_text(encoding="utf-8"))
        manifest = json.loads(Path(f"{output}.manifest.json").read_text(encoding="utf-8"))
        assert manifest["engine"]["flavor"] == "fairy"
        assert manifest["engine"]["workers"] == int(workers)
    assert outputs[0] == outputs[1]
    assert len(outputs[0].splitlines()) == 9


def test_engine_games_are_seeded_and_fall_back_to_legal_moves(tmp_path) -> None:
    settings = GameSettings(fake_engine(tmp_path), 5.0, "fairy", 1, 0.0, 12, 7)
    first, illegal = play_games(settings, [0, 1])
    again, _ = play_games(settings, [0, 1])
    assert first == again
    assert first[0] == START_FEN
    # The fake engine only knows Red's cannon move, so Black's turns fall back to legal random moves.
    assert illegal > 0
    assert len(first) == 24


def test_unique_positions_keep_first_occurrence_and_drop_excluded() -> None:
    assert unique_positions(["a", "b", "a", "c", "d"], {"c"}) == ["a", "b", "d"]


def test_fairy_flavor_refuses_the_ucci_protocol(tmp_path) -> None:
    try:
        UcciEngine(fake_engine(tmp_path), timeout=5.0, init_command="ucci", flavor="fairy")
    except ValueError:
        return
    raise AssertionError("fairy with ucci must be rejected before the engine starts")
