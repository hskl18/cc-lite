from eval.head_to_head import _terminal_score
from xiangqi.board import Board


def test_in_progress_position_has_no_terminal_head_to_head_score():
    board = Board.from_fen("4k4/9/9/9/9/4A4/9/9/9/4K4 r")

    assert board.legal_moves()
    assert _terminal_score(board, "red") is None


def test_rules_profile_draw_is_a_draw_for_both_checkpoints():
    board = Board.start()
    board.no_progress_plies = 120

    assert _terminal_score(board, "red") == 0.0
    assert _terminal_score(board, "black") == 0.0
