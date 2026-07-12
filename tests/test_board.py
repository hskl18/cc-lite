import pytest

from xiangqi.board import Board, Move, parse_square


def test_start_position_has_expected_legal_move_count():
    board = Board.start()
    assert len(board.legal_moves()) == 44


def test_horse_leg_block_rule():
    board = Board.from_fen("4k4/9/9/9/9/9/9/9/4P4/3KH4 r")
    moves = {m.uci() for m in board.legal_moves()}
    assert "e9d7" not in moves
    assert "e9f7" not in moves


def test_flying_general_capture_exists_when_file_is_open():
    board = Board.from_fen("4k4/9/9/9/9/9/9/9/9/4K4 r")
    assert Move(parse_square("e9"), parse_square("e0")) in board.legal_moves()


def test_result_for_awards_no_legal_moves_to_the_other_side():
    board = Board.from_fen("4k4/9/9/9/9/4p4/9/9/3r1r3/4K4 r")

    assert board.legal_moves() == []
    assert board.result_for("red") == -1.0
    assert board.result_for("black") == 1.0


@pytest.mark.parametrize(
    "fen",
    [
        "99//9/9/9/9/9/9/9/9 r",
        "4k4/9/9/9/9/9/9/9/9/4K4 white",
        "4k4/9/9/9/9/9/9/9/9/4K4 r extra",
    ],
)
def test_fen_parser_rejects_malformed_ranks_and_side_tokens(fen):
    with pytest.raises(ValueError):
        Board.from_fen(fen)
