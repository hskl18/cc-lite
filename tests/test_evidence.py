from __future__ import annotations

import hashlib
import json
from pathlib import Path

import eval.evaluate as evaluate_module
from eval.evidence import summarize_game_records, validate_evidence_bundle, write_evidence_bundle
from eval.openings import load_opening_suite
from xiangqi.board import Board


def test_summary_keeps_terminal_results_separate_from_truncation_adjudication() -> None:
    records = [
        {
            "terminal_reason": "opponent_king_captured",
            "terminal_score": 1.0,
            "material_adjudication": None,
        },
        {
            "terminal_reason": "max_plies",
            "terminal_score": None,
            "material_adjudication": 1.0,
        },
    ]

    summary = summarize_game_records(records)

    assert summary["terminal"] == {
        "games": 1,
        "wins": 1,
        "draws": 0,
        "losses": 0,
    }
    assert summary["truncation"] == {
        "games": 1,
        "material_ahead": 1,
        "material_equal": 0,
        "material_behind": 0,
        "material_score_mean": 1.0,
    }


def test_evidence_bundle_records_provenance_and_validates_raw_reconstruction(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    config = tmp_path / "config.yaml"
    config.write_bytes(b"seed: 17\n")
    run_dir = tmp_path / "run"
    records = [
        {
            "game_id": "random-0001",
            "terminal_reason": "max_plies",
            "terminal_score": None,
            "material_adjudication": -1.0,
        }
    ]
    argv = ["python", "-m", "eval.evaluate", "--games", "1"]

    write_evidence_bundle(
        run_dir,
        records=records,
        checkpoint_path=checkpoint,
        config_path=config,
        seed=17,
        argv=argv,
        device="cpu",
    )

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["checkpoint"]["sha256"] == hashlib.sha256(b"checkpoint bytes").hexdigest()
    assert manifest["config"]["sha256"] == hashlib.sha256(b"seed: 17\n").hexdigest()
    assert manifest["seed"] == 17
    assert manifest["argv"] == argv
    assert manifest["environment"]["device"] == "cpu"
    assert manifest["environment"]["python"]
    assert manifest["environment"]["platform"]
    assert manifest["environment"]["pytorch"]
    assert len(manifest["source"]["git_commit"]) == 40
    assert isinstance(manifest["source"]["dirty"], bool)
    assert validate_evidence_bundle(run_dir) == {"valid": True, "errors": []}


def test_validator_rejects_summary_that_does_not_match_raw_games(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_bytes(b"seed: 1\n")
    run_dir = tmp_path / "run"
    write_evidence_bundle(
        run_dir,
        records=[
            {
                "terminal_reason": "opponent_king_captured",
                "terminal_score": 1.0,
                "material_adjudication": None,
            }
        ],
        checkpoint_path=checkpoint,
        config_path=config,
        seed=1,
        argv=["python", "-m", "eval.evaluate"],
        device="cpu",
    )
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["terminal"]["wins"] = 0
    summary_path.write_text(json.dumps(summary))

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "summary does not match games.jsonl reconstruction" in result["errors"]


def test_validator_rejects_terminal_score_on_max_plies_record(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_bytes(b"seed: 1\n")
    run_dir = tmp_path / "run"
    write_evidence_bundle(
        run_dir,
        records=[
            {
                "terminal_reason": "max_plies",
                "terminal_score": 1.0,
                "material_adjudication": 1.0,
            }
        ],
        checkpoint_path=checkpoint,
        config_path=config,
        seed=1,
        argv=["python", "-m", "eval.evaluate"],
        device="cpu",
    )

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "game 1: max_plies must not have terminal_score" in result["errors"]


def test_validator_rejects_missing_checkpoint_provenance(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_bytes(b"seed: 1\n")
    run_dir = tmp_path / "run"
    write_evidence_bundle(
        run_dir,
        records=[
            {
                "terminal_reason": "max_plies",
                "terminal_score": None,
                "material_adjudication": 0.0,
            }
        ],
        checkpoint_path=checkpoint,
        config_path=config,
        seed=1,
        argv=["python", "-m", "eval.evaluate"],
        device="cpu",
    )
    checkpoint.unlink()

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "missing checkpoint provenance" in result["errors"]


def test_evaluation_records_max_plies_without_counting_terminal_results(monkeypatch) -> None:
    class FakeSearch:
        def __init__(self, evaluator, config) -> None:
            self.last_nodes_per_second = 125.0

        def run(self, board):
            return board.legal_moves()[0], None

    monkeypatch.setattr(evaluate_module, "load_checkpoint", lambda *args, **kwargs: (object(), {}))
    monkeypatch.setattr(evaluate_module, "TorchEvaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluate_module, "MCTS", FakeSearch)

    result = evaluate_module.evaluate_checkpoint(
        "unused.pt",
        config={"device": "cpu"},
        opponent="random",
        games=2,
        simulations=1,
        max_plies=1,
    )

    assert len(result["games"]) == 2
    assert all(record["terminal_reason"] == "max_plies" for record in result["games"])
    assert all(record["terminal_score"] is None for record in result["games"])
    assert result["summary"]["terminal"]["games"] == 0
    assert result["summary"]["truncation"]["games"] == 2


def test_evaluation_uses_rules_profile_draw_adjudication() -> None:
    board = Board.start()
    board.no_progress_plies = 120

    assert evaluate_module._terminal_result(board, "red") == (
        "no_progress_120_plies",
        0.0,
    )


def test_evaluate_cli_writes_bundle_with_exact_argv(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 23\ndevice: cpu\n")
    run_dir = tmp_path / "run"
    record = {
        "game_id": "random-0001",
        "plies": 1,
        "terminal_reason": "max_plies",
        "terminal_score": None,
        "material_adjudication": 0.0,
        "avg_nodes_per_second": 10.0,
    }
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_checkpoint",
        lambda *args, **kwargs: {
            "summary": summarize_game_records([record]),
            "games": [record],
            "device": "cpu",
        },
    )
    argv = [
        "eval.evaluate",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(run_dir),
    ]
    monkeypatch.setattr("sys.argv", argv)

    evaluate_module.main()

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["argv"] == ["python", "-m", "eval.evaluate", *argv[1:]]
    assert validate_evidence_bundle(run_dir)["valid"] is True


def test_paired_bundle_records_both_checkpoints_suite_and_seeds(tmp_path) -> None:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    opponent = tmp_path / "opponent.pt"
    opponent.write_bytes(b"opponent")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n")
    opening_suite = tmp_path / "openings.json"
    opening_suite.write_text("{}\n")
    run_dir = tmp_path / "paired"
    records = [
        {
            "pair_id": "opening:seed-1",
            "pair_leg": 1,
            "candidate_color": "red",
            "opening_id": "opening",
            "start_fen": "fixture",
            "seed": 1,
            "terminal_reason": "max_plies",
            "terminal_score": None,
            "material_adjudication": 1.0,
        },
        {
            "pair_id": "opening:seed-1",
            "pair_leg": 2,
            "candidate_color": "black",
            "opening_id": "opening",
            "start_fen": "fixture",
            "seed": 1,
            "terminal_reason": "max_plies",
            "terminal_score": None,
            "material_adjudication": -1.0,
        },
    ]

    write_evidence_bundle(
        run_dir,
        records=records,
        checkpoint_path=candidate,
        opponent_checkpoint_path=opponent,
        opening_suite_path=opening_suite,
        config_path=config,
        seed=1,
        seeds=[1, 7, 19],
        argv=["python", "-m", "eval.head_to_head"],
        device="cpu",
    )

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["opponent_checkpoint"]["sha256"] == hashlib.sha256(b"opponent").hexdigest()
    assert manifest["opening_suite"]["sha256"] == hashlib.sha256(b"{}\n").hexdigest()
    assert manifest["seeds"] == [1, 7, 19]
    assert validate_evidence_bundle(run_dir) == {"valid": True, "errors": []}


def test_paired_bundle_reconstructs_protocol_summary_from_raw_games(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    opponent = tmp_path / "opponent.pt"
    opponent.write_bytes(b"opponent")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    opening_suite = Path("configs/openings/paired-v1.json").resolve()
    opening = load_opening_suite(opening_suite).positions[0]
    run_dir = tmp_path / "paired"
    records = [
        {
            "pair_id": "opening:seed-1",
            "pair_leg": leg,
            "candidate_color": color,
            "opening_id": opening.opening_id,
            "start_fen": opening.fen,
            "seed": 1,
            "terminal_reason": "no_legal_moves",
            "terminal_score": 0.0,
            "material_adjudication": None,
        }
        for leg, color in ((1, "red"), (2, "black"))
    ]
    protocol = {
        "min_pairs": 1,
        "max_pairs": 1,
        "min_seeds": 1,
        "confidence": 0.95,
        "superiority_margin": 0.0,
    }

    write_evidence_bundle(
        run_dir,
        records=records,
        checkpoint_path=checkpoint,
        opponent_checkpoint_path=opponent,
        opening_suite_path=opening_suite,
        config_path=config,
        seed=1,
        seeds=[1],
        paired_protocol=protocol,
        argv=["python", "-m", "eval.head_to_head"],
        device="cpu",
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["opening_suite"]["suite_id"] == "cc-lite-paired-v1"
    assert summary["sequential_stop"]["decision"] == "inconclusive"
    assert validate_evidence_bundle(run_dir) == {"valid": True, "errors": []}


def test_validator_rejects_same_candidate_color_in_both_pair_legs(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n")
    run_dir = tmp_path / "run"
    records = [
        {
            "pair_id": "opening:seed-1",
            "pair_leg": 1,
            "candidate_color": "red",
            "opening_id": "opening",
            "start_fen": "fixture",
            "seed": 1,
            "terminal_reason": "no_legal_moves",
            "terminal_score": 1.0,
            "material_adjudication": None,
        },
        {
            "pair_id": "opening:seed-1",
            "pair_leg": 2,
            "candidate_color": "red",
            "opening_id": "opening",
            "start_fen": "fixture",
            "seed": 1,
            "terminal_reason": "no_legal_moves",
            "terminal_score": -1.0,
            "material_adjudication": None,
        },
    ]
    write_evidence_bundle(
        run_dir,
        records=records,
        checkpoint_path=checkpoint,
        config_path=config,
        seed=1,
        argv=["python", "-m", "eval.head_to_head"],
        device="cpu",
    )

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "pair opening:seed-1: candidate colors must be unique" in result["errors"]
