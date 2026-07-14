from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

import eval.evaluate as evaluate_module
from eval.evidence import summarize_game_records, validate_evidence_bundle, write_evidence_bundle
from eval.openings import load_opening_suite, opening_suite_summary
from eval.paired import build_paired_schedule, sequential_stop_decision
from xiangqi.board import START_FEN, Board

OPENING_SUITE = Path(__file__).resolve().parents[1] / "configs/openings/paired-v1.json"


def _write_untrusted_bundle(
    run_dir: Path,
    *,
    records: list[dict],
    checkpoint: Path,
    config: Path,
    opponent: Path | None = None,
    seeds: list[int] | None = None,
    protocol: dict | None = None,
    summary_override: dict | None = None,
) -> dict:
    run_dir.mkdir()
    games_path = run_dir / "games.jsonl"
    games_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = summary_override if summary_override is not None else summarize_game_records(records)
    if protocol is not None:
        summary["sequential_stop"] = sequential_stop_decision(
            records,
            min_pairs=protocol["min_pairs"],
            max_pairs=protocol["max_pairs"],
            min_seeds=protocol["min_seeds"],
            confidence=protocol["confidence"],
            superiority_margin=protocol["superiority_margin"],
        )
        summary["opening_suite"] = opening_suite_summary(load_opening_suite(OPENING_SUITE))
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")

    def file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schema_version": 1,
        "checkpoint": {"path": str(checkpoint), "sha256": file_hash(checkpoint)},
        "config": {"path": str(config), "sha256": file_hash(config)},
        "artifacts": {
            "games.jsonl": file_hash(games_path),
            "summary.json": file_hash(summary_path),
        },
    }
    if opponent is not None:
        manifest["opponent_checkpoint"] = {
            "path": str(opponent),
            "sha256": file_hash(opponent),
        }
    if seeds is not None:
        manifest["opening_suite"] = {
            "path": str(OPENING_SUITE),
            "sha256": file_hash(OPENING_SUITE),
        }
        manifest["seeds"] = seeds
    if protocol is not None:
        manifest["paired_protocol"] = protocol
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


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
            "start_fen": START_FEN,
            "moves": [],
            "plies": 0,
            "final_fen": START_FEN,
            "model_color": "red",
            "terminal_reason": "max_plies",
            "terminal_score": None,
            "material_delta": 0,
            "material_adjudication": 0.0,
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
                "start_fen": START_FEN,
                "moves": [],
                "plies": 0,
                "final_fen": START_FEN,
                "model_color": "red",
                "terminal_reason": "max_plies",
                "terminal_score": None,
                "material_delta": 0,
                "material_adjudication": 0.0,
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
    summary["truncation"]["material_equal"] = 0
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
                "start_fen": START_FEN,
                "moves": [],
                "plies": 0,
                "final_fen": START_FEN,
                "model_color": "red",
                "terminal_reason": "max_plies",
                "terminal_score": None,
                "material_delta": 0,
                "material_adjudication": 0.0,
            }
        ],
        checkpoint_path=checkpoint,
        config_path=config,
        seed=1,
        argv=["python", "-m", "eval.evaluate"],
        device="cpu",
    )
    games_path = run_dir / "games.jsonl"
    record = json.loads(games_path.read_text(encoding="utf-8"))
    record["terminal_score"] = 1.0
    games_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "game 1: max_plies must not have terminal_score" in result["errors"]


def test_writer_and_validator_reject_plies_that_do_not_match_raw_moves(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_bytes(b"seed: 1\n")
    record = {
        "start_fen": START_FEN,
        "moves": [],
        "plies": 1,
        "final_fen": START_FEN,
        "model_color": "red",
        "terminal_reason": "max_plies",
        "terminal_score": None,
        "material_delta": 0,
        "material_adjudication": 0.0,
    }

    with pytest.raises(ValueError, match="plies must equal the number of raw moves"):
        write_evidence_bundle(
            tmp_path / "rejected",
            records=[record],
            checkpoint_path=checkpoint,
            config_path=config,
            seed=1,
            argv=["python", "-m", "eval.evaluate"],
            device="cpu",
        )

    run_dir = tmp_path / "forged"
    write_evidence_bundle(
        run_dir,
        records=[{**record, "plies": 0}],
        checkpoint_path=checkpoint,
        config_path=config,
        seed=1,
        argv=["python", "-m", "eval.evaluate"],
        device="cpu",
    )
    games_path = run_dir / "games.jsonl"
    games_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "game 1: plies must equal the number of raw moves" in result["errors"]


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
                "start_fen": START_FEN,
                "moves": [],
                "plies": 0,
                "final_fen": START_FEN,
                "model_color": "red",
                "terminal_reason": "max_plies",
                "terminal_score": None,
                "material_delta": 0,
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


def test_validator_rejects_stripped_manifest_provenance(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    write_evidence_bundle(
        run_dir,
        records=[
            {
                "start_fen": START_FEN,
                "moves": [],
                "plies": 0,
                "final_fen": START_FEN,
                "model_color": "red",
                "terminal_reason": "max_plies",
                "terminal_score": None,
                "material_delta": 0,
                "material_adjudication": 0.0,
            }
        ],
        checkpoint_path=checkpoint,
        config_path=config,
        seed=1,
        argv=["python", "-m", "eval.evaluate"],
        device="cpu",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1}) + "\n", encoding="utf-8"
    )

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    for field in ("checkpoint", "config", "source", "seed", "argv", "environment", "artifacts"):
        assert f"manifest requires {field}" in result["errors"]
    assert "manifest artifacts must include games.jsonl and summary.json" in result["errors"]


def test_validator_fails_closed_on_invalid_nested_provenance_types(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    write_evidence_bundle(
        run_dir,
        records=[
            {
                "start_fen": START_FEN,
                "moves": [],
                "plies": 0,
                "final_fen": START_FEN,
                "model_color": "red",
                "terminal_reason": "max_plies",
                "terminal_score": None,
                "material_delta": 0,
                "material_adjudication": 0.0,
            }
        ],
        checkpoint_path=checkpoint,
        config_path=config,
        seed=1,
        argv=["python", "-m", "eval.evaluate"],
        device="cpu",
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint"]["path"] = []
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "manifest checkpoint requires string path and sha256" in result["errors"]


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
        "start_fen": START_FEN,
        "moves": [],
        "plies": 0,
        "final_fen": START_FEN,
        "model_color": "red",
        "terminal_reason": "max_plies",
        "terminal_score": None,
        "material_delta": 0,
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
    opening_suite = OPENING_SUITE
    run_dir = tmp_path / "paired"
    schedule = build_paired_schedule(load_opening_suite(opening_suite), [1])[:2]
    records = [
        {
            **asdict(scheduled),
            "moves": [],
            "plies": 0,
            "final_fen": scheduled.start_fen,
            "terminal_reason": "max_plies",
            "terminal_score": None,
            "material_delta": 0,
            "material_adjudication": 0.0,
        }
        for scheduled in schedule
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
        checkpoint_path=candidate,
        opponent_checkpoint_path=opponent,
        opening_suite_path=opening_suite,
        config_path=config,
        seed=1,
        seeds=[1],
        paired_protocol=protocol,
        argv=["python", "-m", "eval.head_to_head"],
        device="cpu",
    )

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["opponent_checkpoint"]["sha256"] == hashlib.sha256(b"opponent").hexdigest()
    assert manifest["opening_suite"]["sha256"] == hashlib.sha256(
        OPENING_SUITE.read_bytes()
    ).hexdigest()
    assert manifest["seeds"] == [1]
    assert validate_evidence_bundle(run_dir) == {"valid": True, "errors": []}


def test_paired_bundle_reconstructs_protocol_summary_from_raw_games(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    opponent = tmp_path / "opponent.pt"
    opponent.write_bytes(b"opponent")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    opening_suite = (
        Path(__file__).resolve().parents[1] / "configs/openings/paired-v1.json"
    )
    schedule = build_paired_schedule(load_opening_suite(opening_suite), [1])[:2]
    run_dir = tmp_path / "paired"
    records = [
        {
            **asdict(scheduled),
            "moves": [],
            "plies": 0,
            "final_fen": scheduled.start_fen,
            "terminal_reason": "max_plies",
            "terminal_score": None,
            "material_delta": 0,
            "material_adjudication": 0.0,
        }
        for scheduled in schedule
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


def test_writer_and_validator_require_opponent_provenance_for_paired_evidence(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    schedule = build_paired_schedule(load_opening_suite(OPENING_SUITE), [1])[:2]
    records = [
        {
            **asdict(scheduled),
            "moves": [],
            "plies": 0,
            "final_fen": scheduled.start_fen,
            "terminal_reason": "max_plies",
            "terminal_score": None,
            "material_delta": 0,
            "material_adjudication": 0.0,
        }
        for scheduled in schedule
    ]
    protocol = {
        "min_pairs": 1,
        "max_pairs": 1,
        "min_seeds": 1,
        "confidence": 0.95,
        "superiority_margin": 0.0,
    }

    with pytest.raises(ValueError, match="paired evidence requires opponent_checkpoint_path"):
        write_evidence_bundle(
            tmp_path / "rejected",
            records=records,
            checkpoint_path=checkpoint,
            config_path=config,
            opening_suite_path=OPENING_SUITE,
            seeds=[1],
            paired_protocol=protocol,
            seed=1,
            argv=["python", "-m", "eval.head_to_head"],
            device="cpu",
        )

    run_dir = tmp_path / "forged-no-opponent"
    _write_untrusted_bundle(
        run_dir,
        records=records,
        checkpoint=checkpoint,
        config=config,
        seeds=[1],
        protocol=protocol,
    )

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "paired evidence requires opponent_checkpoint provenance" in result["errors"]


def test_validator_rejects_paired_records_without_complete_replay_fields(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    opponent = tmp_path / "opponent.pt"
    opponent.write_bytes(b"opponent")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    seeds = list(range(1, 9))
    schedule = build_paired_schedule(load_opening_suite(OPENING_SUITE), seeds)[:120]
    records = [
        {
            **asdict(scheduled),
            "terminal_reason": "no_legal_moves",
            "terminal_score": 1.0,
            "material_adjudication": None,
        }
        for scheduled in schedule
    ]
    run_dir = tmp_path / "missing-replay"
    protocol = {
        "min_pairs": 60,
        "max_pairs": 60,
        "min_seeds": 2,
        "confidence": 0.95,
        "superiority_margin": 0.0,
    }

    with pytest.raises(ValueError, match="moves must be a list of move strings"):
        write_evidence_bundle(
            run_dir,
            records=records,
            checkpoint_path=checkpoint,
            opponent_checkpoint_path=opponent,
            opening_suite_path=OPENING_SUITE,
            config_path=config,
            seed=1,
            seeds=seeds,
            paired_protocol=protocol,
            argv=["python", "-m", "eval.head_to_head"],
            device="cpu",
        )

    summary = _write_untrusted_bundle(
        run_dir,
        records=records,
        checkpoint=checkpoint,
        opponent=opponent,
        config=config,
        seeds=seeds,
        protocol=protocol,
    )

    assert summary["sequential_stop"]["decision"] == "superior"
    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "game 1: moves must be a list of move strings" in result["errors"]
    assert "game 1: final_fen must be a string" in result["errors"]
    assert "game 1: plies must be an integer" in result["errors"]


def test_validator_rejects_replayed_pairs_outside_canonical_schedule_prefix(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    opponent = tmp_path / "opponent.pt"
    opponent.write_bytes(b"opponent")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    canonical_slots = build_paired_schedule(load_opening_suite(OPENING_SUITE), [1, 7])[:4]
    repetition_moves = ["c7b9", "c2b0", "b9c7", "b0c2"] * 2
    records = []
    for forged_pair_number in range(1, 7):
        slot_start = 0 if forged_pair_number % 2 else 2
        for scheduled in canonical_slots[slot_start : slot_start + 2]:
            records.append(
                {
                    **asdict(scheduled),
                    "game_id": f"forged-{forged_pair_number}-leg-{scheduled.pair_leg}",
                    "pair_id": f"forged-pair-{forged_pair_number}",
                    "moves": repetition_moves,
                    "plies": len(repetition_moves),
                    "final_fen": scheduled.start_fen,
                    "terminal_reason": "threefold_repetition",
                    "terminal_score": 0.0,
                    "material_adjudication": None,
                }
            )
    run_dir = tmp_path / "forged-schedule"
    protocol = {
        "min_pairs": 1,
        "max_pairs": 6,
        "min_seeds": 2,
        "confidence": 0.95,
        "superiority_margin": 0.0,
    }

    with pytest.raises(ValueError, match="does not match canonical paired schedule"):
        write_evidence_bundle(
            run_dir,
            records=records,
            checkpoint_path=checkpoint,
            opponent_checkpoint_path=opponent,
            opening_suite_path=OPENING_SUITE,
            config_path=config,
            seed=1,
            seeds=[1, 7],
            paired_protocol=protocol,
            argv=["python", "-m", "eval.head_to_head"],
            device="cpu",
        )

    summary = _write_untrusted_bundle(
        run_dir,
        records=records,
        checkpoint=checkpoint,
        opponent=opponent,
        config=config,
        seeds=[1, 7],
        protocol=protocol,
    )

    assert summary["paired"]["complete_pairs"] == 6
    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert any("does not match canonical paired schedule" in error for error in result["errors"])


def test_writer_and_validator_require_protocol_for_any_paired_record(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    opponent = tmp_path / "opponent.pt"
    opponent.write_bytes(b"opponent")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    canonical_pair = build_paired_schedule(load_opening_suite(OPENING_SUITE), [1])[:2]
    repetition_moves = ["c7b9", "c2b0", "b9c7", "b0c2"] * 2
    records = []
    for forged_pair_number in range(1, 7):
        for scheduled in canonical_pair:
            records.append(
                {
                    **asdict(scheduled),
                    "game_id": f"forged-{forged_pair_number}-leg-{scheduled.pair_leg}",
                    "pair_id": f"forged-pair-{forged_pair_number}",
                    "model_color": scheduled.candidate_color,
                    "moves": repetition_moves,
                    "plies": len(repetition_moves),
                    "final_fen": scheduled.start_fen,
                    "terminal_reason": "threefold_repetition",
                    "terminal_score": 0.0,
                    "material_delta": None,
                    "material_adjudication": None,
                }
            )

    with pytest.raises(ValueError, match="paired records require paired_protocol"):
        write_evidence_bundle(
            tmp_path / "rejected",
            records=records,
            checkpoint_path=checkpoint,
            opponent_checkpoint_path=opponent,
            opening_suite_path=OPENING_SUITE,
            config_path=config,
            seed=1,
            seeds=[1],
            argv=["python", "-m", "eval.head_to_head"],
            device="cpu",
        )

    run_dir = tmp_path / "forged-without-protocol"
    summary = _write_untrusted_bundle(
        run_dir,
        records=records,
        checkpoint=checkpoint,
        opponent=opponent,
        config=config,
        seeds=[1],
    )

    assert summary["paired"]["complete_pairs"] == 6
    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "paired records require paired_protocol" in result["errors"]


def test_writer_and_validator_reject_null_candidate_color_shadowing_model_color(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    terminal_fen = "9/9/9/9/9/9/9/9/9/4K4 r"
    record = {
        "model_color": "black",
        "candidate_color": None,
        "start_fen": terminal_fen,
        "moves": [],
        "plies": 0,
        "final_fen": terminal_fen,
        "terminal_reason": "king_capture",
        "terminal_score": 1.0,
        "material_delta": None,
        "material_adjudication": None,
    }

    with pytest.raises(ValueError, match="candidate_color must be red or black"):
        write_evidence_bundle(
            tmp_path / "rejected",
            records=[record],
            checkpoint_path=checkpoint,
            config_path=config,
            seed=1,
            argv=["python", "-m", "eval.evaluate"],
            device="cpu",
        )

    run_dir = tmp_path / "forged-color"
    _write_untrusted_bundle(
        run_dir,
        records=[record],
        checkpoint=checkpoint,
        config=config,
    )

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "game 1: candidate_color must be red or black when present" in result["errors"]


def test_writer_and_validator_reconstruct_truncation_material_result(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    record = {
        "model_color": "red",
        "start_fen": START_FEN,
        "moves": [],
        "plies": 0,
        "final_fen": START_FEN,
        "terminal_reason": "max_plies",
        "terminal_score": None,
        "material_delta": 0,
        "material_adjudication": 1.0,
    }

    with pytest.raises(ValueError, match="material_adjudication does not match replay"):
        write_evidence_bundle(
            tmp_path / "rejected",
            records=[record],
            checkpoint_path=checkpoint,
            config_path=config,
            seed=1,
            argv=["python", "-m", "eval.evaluate"],
            device="cpu",
        )

    run_dir = tmp_path / "forged-material"
    _write_untrusted_bundle(
        run_dir,
        records=[record],
        checkpoint=checkpoint,
        config=config,
    )

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "game 1: material_adjudication does not match replay" in result["errors"]


def test_writer_and_validator_reject_empty_evidence(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="evidence must contain at least one game"):
        write_evidence_bundle(
            tmp_path / "rejected-empty",
            records=[],
            checkpoint_path=checkpoint,
            config_path=config,
            seed=1,
            argv=["python", "-m", "eval.evaluate"],
            device="cpu",
        )

    run_dir = tmp_path / "forged-empty"
    _write_untrusted_bundle(
        run_dir,
        records=[],
        checkpoint=checkpoint,
        config=config,
    )

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "evidence must contain at least one game" in result["errors"]


def test_writer_and_validator_reject_boolean_scores(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    record = {
        "start_fen": START_FEN,
        "moves": [],
        "plies": 0,
        "final_fen": START_FEN,
        "model_color": "red",
        "terminal_reason": "max_plies",
        "terminal_score": None,
        "material_delta": 0,
        "material_adjudication": False,
    }

    with pytest.raises(ValueError, match="material_adjudication must be numeric"):
        write_evidence_bundle(
            tmp_path / "rejected-boolean",
            records=[record],
            checkpoint_path=checkpoint,
            config_path=config,
            seed=1,
            argv=["python", "-m", "eval.evaluate"],
            device="cpu",
        )

    run_dir = tmp_path / "forged-boolean"
    _write_untrusted_bundle(
        run_dir,
        records=[record],
        checkpoint=checkpoint,
        config=config,
    )

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "game 1: material_adjudication must be numeric, not boolean" in result["errors"]


def test_validator_fails_closed_on_invalid_score_types(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    record = {
        "start_fen": START_FEN,
        "moves": [],
        "plies": 0,
        "final_fen": START_FEN,
        "model_color": "red",
        "terminal_reason": "max_plies",
        "terminal_score": None,
        "material_delta": 0,
        "material_adjudication": [],
    }
    run_dir = tmp_path / "forged-invalid-score"
    _write_untrusted_bundle(
        run_dir,
        records=[record],
        checkpoint=checkpoint,
        config=config,
        summary_override={},
    )

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "game 1: material_adjudication must be numeric, not boolean" in result["errors"]
    assert any(error.startswith("cannot reconstruct evidence summary:") for error in result["errors"])


def test_writer_and_validator_reject_canonical_prefix_before_sequential_stop(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    opponent = tmp_path / "opponent.pt"
    opponent.write_bytes(b"opponent")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    schedule = build_paired_schedule(load_opening_suite(OPENING_SUITE), [1])[:2]
    records = [
        {
            **asdict(scheduled),
            "moves": [],
            "plies": 0,
            "final_fen": scheduled.start_fen,
            "terminal_reason": "max_plies",
            "terminal_score": None,
            "material_delta": 0,
            "material_adjudication": 0.0,
        }
        for scheduled in schedule
    ]
    protocol = {
        "min_pairs": 1,
        "max_pairs": 2,
        "min_seeds": 1,
        "confidence": 0.95,
        "superiority_margin": 0.0,
    }

    with pytest.raises(ValueError, match="ends before a sequential stop decision"):
        write_evidence_bundle(
            tmp_path / "rejected",
            records=records,
            checkpoint_path=checkpoint,
            opponent_checkpoint_path=opponent,
            opening_suite_path=OPENING_SUITE,
            config_path=config,
            seed=1,
            seeds=[1],
            paired_protocol=protocol,
            argv=["python", "-m", "eval.head_to_head"],
            device="cpu",
        )

    run_dir = tmp_path / "forged-prefix"
    _write_untrusted_bundle(
        run_dir,
        records=records,
        checkpoint=checkpoint,
        opponent=opponent,
        config=config,
        seeds=[1],
        protocol=protocol,
    )

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "paired evidence ends before a sequential stop decision" in result["errors"]


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
            "start_fen": START_FEN,
            "seed": 1,
            "moves": [],
            "plies": 0,
            "final_fen": START_FEN,
            "terminal_reason": "max_plies",
            "terminal_score": None,
            "material_delta": 0,
            "material_adjudication": 0.0,
        },
        {
            "pair_id": "opening:seed-1",
            "pair_leg": 2,
            "candidate_color": "red",
            "opening_id": "opening",
            "start_fen": START_FEN,
            "seed": 1,
            "moves": [],
            "plies": 0,
            "final_fen": START_FEN,
            "terminal_reason": "max_plies",
            "terminal_score": None,
            "material_delta": 0,
            "material_adjudication": 0.0,
        },
    ]
    _write_untrusted_bundle(
        run_dir,
        records=records,
        checkpoint=checkpoint,
        config=config,
    )

    result = validate_evidence_bundle(run_dir)

    assert result["valid"] is False
    assert "pair opening:seed-1: candidate colors must be unique" in result["errors"]
