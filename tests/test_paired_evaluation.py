from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import eval.head_to_head as head_to_head
from eval.evidence import summarize_game_records
from eval.openings import load_opening_suite, mirror_fen
from eval.paired import build_paired_schedule, sequential_stop_decision
from xiangqi.board import BLACK, RED

OPENING_SUITE = Path("configs/openings/paired-v1.json")


def _record(
    pair_id: str,
    leg: int,
    color: str,
    score: float | None,
    *,
    seed: int,
) -> dict[str, object]:
    truncated = score is None
    return {
        "game_id": f"{pair_id}:{leg}",
        "pair_id": pair_id,
        "pair_leg": leg,
        "seed": seed,
        "opening_id": pair_id.split(":")[0],
        "candidate_color": color,
        "start_fen": "fixture-fen",
        "plies": 10,
        "terminal_reason": "max_plies" if truncated else "no_legal_moves",
        "terminal_score": score,
        "material_adjudication": 1.0 if truncated else None,
    }


def test_pinned_opening_suite_contains_exact_file_mirrors() -> None:
    suite = load_opening_suite(OPENING_SUITE)

    assert suite.suite_id == "cc-lite-paired-v1"
    assert suite.version == 1
    assert len(suite.positions) == 8
    by_id = {position.opening_id: position for position in suite.positions}
    for position in suite.positions:
        if position.mirror_of is not None:
            assert position.fen == mirror_fen(by_id[position.mirror_of].fen)


def test_opening_suite_rejects_fen_that_does_not_match_move_provenance(tmp_path) -> None:
    payload = json.loads(OPENING_SUITE.read_text(encoding="utf-8"))
    payload["positions"][0]["moves_from_start"] = ["a6a5"]
    tampered = tmp_path / "openings.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match moves_from_start"):
        load_opening_suite(tampered)


def test_schedule_pairs_both_candidate_colors_for_every_opening_and_seed() -> None:
    suite = load_opening_suite(OPENING_SUITE)
    schedule = build_paired_schedule(suite, [3, 11])

    assert len(schedule) == len(suite.positions) * 2 * 2
    pair_ids = {game.pair_id for game in schedule}
    for pair_id in pair_ids:
        pair = [game for game in schedule if game.pair_id == pair_id]
        assert {game.pair_leg for game in pair} == {1, 2}
        assert {game.candidate_color for game in pair} == {RED, BLACK}
        assert len({game.start_fen for game in pair}) == 1
        assert len({game.seed for game in pair}) == 1


def test_schedule_rejects_duplicate_seeds() -> None:
    suite = load_opening_suite(OPENING_SUITE)

    with pytest.raises(ValueError, match="unique"):
        build_paired_schedule(suite, [7, 7])


def test_paired_summary_bootstraps_complete_pairs_and_excludes_capped_pair() -> None:
    records = [
        _record("horse:1", 1, RED, 1.0, seed=1),
        _record("horse:1", 2, BLACK, -1.0, seed=1),
        _record("cannon:7", 1, RED, 1.0, seed=7),
        _record("cannon:7", 2, BLACK, 1.0, seed=7),
        _record("pawn:7", 1, RED, None, seed=7),
        _record("pawn:7", 2, BLACK, 1.0, seed=7),
    ]

    summary = summarize_game_records(records)

    assert summary["terminal"] == {"games": 5, "wins": 4, "draws": 0, "losses": 1}
    assert summary["truncation"]["games"] == 1
    paired = summary["paired"]
    assert paired["observed_pairs"] == 3
    assert paired["complete_pairs"] == 2
    assert paired["incomplete_pairs"] == 1
    assert paired["score_rate"] == 0.75
    assert paired["distinct_seeds"] == 2
    assert math.isfinite(paired["elo_difference"]["estimate"])
    assert paired["bootstrap"] == {
        "method": "paired-cluster-percentile",
        "samples": 10_000,
        "seed": 0,
        "confidence": 0.95,
    }


def test_sequential_rule_can_stop_on_decisive_complete_pairs() -> None:
    records = []
    for index in range(60):
        seed = 1 if index % 2 == 0 else 7
        pair_id = f"opening-{index}:{seed}"
        records.extend(
            [
                _record(pair_id, 1, RED, 1.0, seed=seed),
                _record(pair_id, 2, BLACK, 1.0, seed=seed),
            ]
        )

    decision = sequential_stop_decision(records, min_pairs=20, max_pairs=100)

    assert decision["stop"] is True
    assert decision["decision"] == "superior"
    assert decision["reason"] == "lower_bound_above_margin"
    assert decision["confidence_sequence"]["low"] > 0.5


def test_sequential_pair_cap_never_turns_truncations_into_a_result() -> None:
    records = []
    for index in range(5):
        pair_id = f"opening-{index}:1"
        records.extend(
            [
                _record(pair_id, 1, RED, None, seed=1),
                _record(pair_id, 2, BLACK, None, seed=1),
            ]
        )

    decision = sequential_stop_decision(records, min_pairs=2, max_pairs=5)
    summary = summarize_game_records(records)

    assert decision["stop"] is True
    assert decision["decision"] == "inconclusive"
    assert decision["completed_pairs"] == 0
    assert summary["terminal"]["games"] == 0
    assert summary["terminal"]["wins"] == 0
    assert summary["paired"]["score_rate"] is None
    assert summary["paired"]["elo_difference"] is None


def test_paired_match_records_real_max_plies_as_truncation(monkeypatch) -> None:
    class FakeSearch:
        def __init__(self, evaluator, config) -> None:
            self.last_nodes_per_second = 100.0

        def run(self, board):
            return board.legal_moves()[0], None

    monkeypatch.setattr(head_to_head, "load_checkpoint", lambda *args, **kwargs: (object(), {}))
    monkeypatch.setattr(head_to_head, "TorchEvaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(head_to_head, "MCTS", FakeSearch)

    result = head_to_head.play_paired_match(
        "candidate.pt",
        "reference.pt",
        config={"device": "cpu"},
        opening_suite=OPENING_SUITE,
        seeds=[1],
        simulations=1,
        max_plies=1,
        min_pairs=1,
        max_pairs=1,
        min_seeds=1,
    )

    assert len(result["games"]) == 2
    assert {record["candidate_color"] for record in result["games"]} == {RED, BLACK}
    assert all(record["terminal_reason"] == "max_plies" for record in result["games"])
    assert all(record["terminal_score"] is None for record in result["games"])
    assert result["summary"]["terminal"]["wins"] == 0
    assert result["summary"]["paired"]["complete_pairs"] == 0
    assert result["summary"]["sequential_stop"]["decision"] == "inconclusive"
