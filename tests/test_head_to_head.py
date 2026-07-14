from pathlib import Path

import pytest

import eval.head_to_head as head_to_head
from eval.head_to_head import _terminal_score
from xiangqi.board import Board

OPENING_SUITE = Path(__file__).resolve().parents[1] / "configs/openings/paired-v1.json"


def test_in_progress_position_has_no_terminal_head_to_head_score():
    board = Board.from_fen("4k4/9/9/9/9/4A4/9/9/9/4K4 r")

    assert board.legal_moves()
    assert _terminal_score(board, "red") is None


def test_rules_profile_draw_is_a_draw_for_both_checkpoints():
    board = Board.start()
    board.no_progress_plies = 120

    assert _terminal_score(board, "red") == 0.0
    assert _terminal_score(board, "black") == 0.0


@pytest.mark.parametrize("min_pairs", [0, -1])
def test_paired_match_rejects_nonpositive_min_pairs_before_loading_models(
    monkeypatch, min_pairs: int
):
    def fail_if_called(*args, **kwargs):
        pytest.fail("checkpoint loader must not be called for an invalid min_pairs")

    monkeypatch.setattr(head_to_head, "load_checkpoint", fail_if_called)

    with pytest.raises(ValueError, match="min_pairs must be at least 1"):
        head_to_head.play_paired_match(
            "candidate.pt",
            "reference.pt",
            config={"device": "cpu"},
            opening_suite=OPENING_SUITE,
            seeds=[1],
            simulations=1,
            max_plies=1,
            min_pairs=min_pairs,
            max_pairs=0,
            min_seeds=1,
        )
