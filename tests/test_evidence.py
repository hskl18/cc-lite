from __future__ import annotations

import hashlib
import json

import eval.evaluate as evaluate_module
from eval.evidence import summarize_game_records, validate_evidence_bundle, write_evidence_bundle


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
