from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import eval.evaluate as evaluate_module
from eval.engine import UcciEngine, cc_move_to_engine
from eval.evaluate import evaluate_checkpoint
from eval.evidence import validate_evidence_bundle, write_evidence_bundle
from model.checkpoint import save_checkpoint
from model.network import ModelConfig, PolicyValueNet

SRC = Path(__file__).resolve().parents[1] / "src"

# A Pikafish-style engine that answers with the first legal move under cc-lite's own rules.
LEGAL_ENGINE = f"""
import sys
sys.path.insert(0, {str(SRC)!r})
from xiangqi.board import Board, Move
board = None
for line in sys.stdin:
    command = line.strip()
    if command == "uci":
        print("uciok", flush=True)
    elif command == "isready":
        print("readyok", flush=True)
    elif command.startswith("position fen "):
        position, _, played = command[len("position fen "):].partition(" moves ")
        placement, turn = position.split()[:2]
        board = Board.from_fen(placement.translate(str.maketrans("NnBb", "HhEe")) + (" r" if turn == "w" else " b"))
        for move in played.split():
            board.push(Move.from_uci(move[0] + str(9 - int(move[1])) + move[2] + str(9 - int(move[3]))))
    elif command.startswith("go"):
        move = sorted(board.legal_moves(), key=lambda m: m.uci())[0].uci()
        print("info depth 1 score cp 0")
        print("bestmove " + move[0] + str(9 - int(move[1])) + move[2] + str(9 - int(move[3])), flush=True)
    elif command == "quit":
        break
"""


@pytest.fixture
def setup(tmp_path):
    script = tmp_path / "legal_engine.py"
    script.write_text(LEGAL_ENGINE, encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, PolicyValueNet(ModelConfig(channels=8, blocks=1)))
    config = tmp_path / "config.yaml"
    config.write_text("seed: 3\ndevice: cpu\n", encoding="utf-8")
    network = tmp_path / "pikafish.nnue"
    network.write_bytes(b"weights")
    return f'{sys.executable} "{script}"', checkpoint, config, network


def _evaluate(engine, checkpoint, sample_plies, games=4, eval_file=None):
    return evaluate_checkpoint(
        str(checkpoint),
        config={"seed": 3, "device": "cpu"},
        opponent="engine",
        games=games,
        simulations=16,
        max_plies=6,
        engine_command=engine,
        engine_depth=1,
        engine_eval_file=eval_file,
        sample_plies=sample_plies,
    )


def test_greedy_play_against_an_engine_is_refused(setup) -> None:
    engine, checkpoint, _, network = setup
    with pytest.raises(ValueError, match="sample_plies"):
        _evaluate(engine, checkpoint, sample_plies=0, eval_file=network)


def test_a_pikafish_opponent_needs_its_network_pinned(setup) -> None:
    engine, checkpoint, _, _ = setup
    with pytest.raises(ValueError, match="engine-eval-file"):
        _evaluate(engine, checkpoint, sample_plies=2)


def test_sampled_moves_are_counted_per_model_move_for_both_colors(setup, monkeypatch) -> None:
    engine, checkpoint, _, network = setup
    temperatures: dict[str, list[float]] = {"red": [], "black": []}

    class RecordingSearch:
        def __init__(self, evaluator, config) -> None:
            self.temperature = config.temperature
            self.last_nodes_per_second = 1.0

        def run(self, board):
            temperatures[board.turn].append(self.temperature)
            return board.legal_moves()[0], None

    monkeypatch.setattr(evaluate_module, "MCTS", RecordingSearch)
    _evaluate(engine, checkpoint, sample_plies=1, games=2, eval_file=network)
    # Black first moves at ply 1, so a ply-based count would never sample it.
    assert temperatures["red"] == [1.0, 0.0, 0.0]
    assert temperatures["black"] == [1.0, 0.0, 0.0]


def test_the_engine_is_sent_every_earlier_move_of_the_game(setup, monkeypatch) -> None:
    engine, checkpoint, _, network = setup
    positions: list[str] = []
    send = UcciEngine.send

    def recording_send(self, command: str) -> None:
        if command.startswith("position"):
            positions.append(command)
        send(self, command)

    monkeypatch.setattr(UcciEngine, "send", recording_send)
    result = _evaluate(engine, checkpoint, sample_plies=1, games=2, eval_file=network)

    # Red moves at even plies, so the engine moves at even plies exactly when the model is Black.
    expected = [
        " ".join(cc_move_to_engine(move) for move in game["moves"][:ply])
        for game in result["games"]
        for ply in range(len(game["moves"]))
        if (ply % 2 == 0) == (game["model_color"] == "black")
    ]
    assert [command.partition(" moves ")[2] for command in positions] == expected
    assert result["opponent_engine"]["sees_game_history"] is True


def test_engine_games_record_the_engine_and_validate(setup, tmp_path) -> None:
    engine, checkpoint, config, network = setup
    result = _evaluate(engine, checkpoint, sample_plies=4, eval_file=network)
    assert result["opponent_engine"]["eval_file"]["path"] == str(network.resolve())
    assert result["opponent_engine"]["depth"] == 1
    assert result["opponent_engine"]["flavor"] == "pikafish"
    assert len({tuple(record["moves"]) for record in result["games"]}) > 2

    run_dir = tmp_path / "run"
    write_evidence_bundle(
        run_dir,
        records=result["games"],
        checkpoint_path=checkpoint,
        config_path=config,
        seed=3,
        argv=["python", "-m", "eval.evaluate"],
        device="cpu",
        opponent_engine=result["opponent_engine"],
    )
    assert validate_evidence_bundle(run_dir)["valid"]

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["opponent_engine"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_evidence_bundle(run_dir)
    assert not report["valid"]
    assert "engine games require opponent_engine provenance" in report["errors"]


def test_a_network_pin_over_ucci_is_refused(setup) -> None:
    engine, _, _, network = setup
    with pytest.raises(ValueError, match="only over the UCI protocol"):
        UcciEngine(engine, init_command="ucci", eval_file=str(network))
