from __future__ import annotations

import random

from xiangqi.board import BLACK, RED, START_FEN, Board, opposite


def reference_in_check(board: Board, color: str) -> bool:
    """The original definition: some opponent pseudo-legal move lands on the king."""

    king = board.king_square(color)
    if king is None:
        return True
    return any(move.to_sq == king for move in board.pseudo_legal_moves(opposite(color)))


def _random_positions(games: int, plies: int, seed: int) -> list[Board]:
    rng = random.Random(seed)
    positions = []
    for _ in range(games):
        board = Board.from_fen(START_FEN)
        for _ in range(plies):
            moves = board.legal_moves()
            if not moves:
                break
            board.push(rng.choice(moves))
            positions.append(Board.from_fen(board.to_fen()))
    return positions


def test_fast_check_detection_matches_the_reference_on_played_positions():
    positions = _random_positions(games=40, plies=120, seed=11)
    checks = 0
    for board in positions:
        for color in (RED, BLACK):
            expected = reference_in_check(board, color)
            assert board.is_in_check(color) == expected, (board.to_fen(), color)
            checks += expected
    assert len(positions) > 3000
    assert checks > 50


def test_fast_check_detection_matches_on_unusual_placements():
    # Kings, advisors, elephants, and pawns placed where only the move generators' own rules decide.
    fens = [
        "4k4/9/9/9/9/4A4/9/9/9/4K4 r",
        "4k4/9/9/9/9/9/9/9/9/4K4 r",
        "4k4/4p4/9/9/9/9/9/9/9/4K4 r",
        "3k5/9/9/9/9/9/9/9/9/4K4 b",
        "4k4/9/4C4/9/4p4/9/9/9/9/3K5 b",
        "3ak4/4A4/9/9/9/9/9/9/9/4K4 b",
        "4k4/9/9/9/9/9/9/2E6/9/4K4 r",
        "4k4/9/3H5/9/9/9/9/9/9/4K4 b",
        "4k4/9/9/9/9/9/9/9/4p4/3K5 r",
        "9/4k4/3P5/9/9/9/9/9/9/4K4 b",
    ]
    for fen in fens:
        board = Board.from_fen(fen)
        for color in (RED, BLACK):
            assert board.is_in_check(color) == reference_in_check(board, color), (fen, color)
