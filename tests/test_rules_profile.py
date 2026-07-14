import random

import pytest

from xiangqi.board import (
    BLACK,
    BOARD_SIZE,
    NO_PROGRESS_PLIES,
    RED,
    Adjudication,
    Board,
    Move,
    in_bounds,
    opposite,
    parse_square,
    sq_to_rc,
)


def _board_with(*pieces: tuple[str, str], turn: str = RED) -> Board:
    squares: list[str | None] = [None] * BOARD_SIZE
    for name, piece in pieces:
        squares[parse_square(name)] = piece
    return Board(squares, turn)


def _destinations(board: Board, square: str, color: str) -> set[tuple[int, int]]:
    source = parse_square(square)
    return {
        sq_to_rc(move.to_sq)
        for move in board.pseudo_legal_moves(color)
        if move.from_sq == source
    }


@pytest.mark.parametrize("color", [RED, BLACK])
def test_every_king_move_stays_in_its_palace(color: str) -> None:
    rows = range(7, 10) if color == RED else range(0, 3)
    piece = "K" if color == RED else "k"
    other_king = ("e0", "k") if color == RED else ("e9", "K")
    blocker_piece = "P" if color == RED else "p"
    blockers = (("d5", blocker_piece), ("e5", blocker_piece), ("f5", blocker_piece))

    for row in rows:
        for col in range(3, 6):
            name = f"{chr(ord('a') + col)}{row}"
            board = _board_with((name, piece), other_king, *blockers, turn=color)
            destinations = _destinations(board, name, color)
            expected = {
                (next_row, next_col)
                for next_row, next_col in (
                    (row + 1, col),
                    (row - 1, col),
                    (row, col + 1),
                    (row, col - 1),
                )
                if next_row in rows and 3 <= next_col <= 5
            }
            assert destinations == expected


@pytest.mark.parametrize("color", [RED, BLACK])
def test_every_advisor_move_is_one_diagonal_step_inside_its_palace(color: str) -> None:
    rows = range(7, 10) if color == RED else range(0, 3)
    piece = "A" if color == RED else "a"
    king = ("e9", "K") if color == RED else ("e0", "k")
    other_king = ("d0", "k") if color == RED else ("d9", "K")

    for row in rows:
        for col in range(3, 6):
            name = f"{chr(ord('a') + col)}{row}"
            if name == king[0]:
                continue
            board = _board_with(king, other_king, (name, piece), turn=color)
            destinations = _destinations(board, name, color)
            expected = {
                (row + dr, col + dc)
                for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                if row + dr in rows
                and 3 <= col + dc <= 5
                and f"{chr(ord('a') + col + dc)}{row + dr}" != king[0]
            }
            assert destinations == expected


@pytest.mark.parametrize("color", [RED, BLACK])
def test_elephants_never_cross_the_river_from_any_own_side_square(color: str) -> None:
    rows = range(5, 10) if color == RED else range(0, 5)
    piece = "E" if color == RED else "e"
    kings = (("e9", "K"), ("d0", "k"))

    for row in rows:
        for col in range(9):
            name = f"{chr(ord('a') + col)}{row}"
            if name in {king[0] for king in kings}:
                continue
            board = _board_with(*kings, (name, piece), turn=color)
            destinations = _destinations(board, name, color)
            assert all(next_row in rows for next_row, _ in destinations)
            assert all(abs(next_row - row) == abs(next_col - col) == 2 for next_row, next_col in destinations)


@pytest.mark.parametrize("color", [RED, BLACK])
def test_pawn_sideways_moves_start_only_after_crossing_the_river(color: str) -> None:
    piece = "P" if color == RED else "p"
    forward = -1 if color == RED else 1
    kings = (("e9", "K"), ("d0", "k"))

    for row in range(10):
        for col in range(9):
            name = f"{chr(ord('a') + col)}{row}"
            if name in {king[0] for king in kings}:
                continue
            board = _board_with(*kings, (name, piece), turn=color)
            destinations = _destinations(board, name, color)
            expected = set()
            if in_bounds(row + forward, col):
                expected.add((row + forward, col))
            crossed = row <= 4 if color == RED else row >= 5
            if crossed:
                if col > 0:
                    expected.add((row, col - 1))
                if col < 8:
                    expected.add((row, col + 1))
            assert destinations == expected


def test_flying_general_capture_requires_an_open_file() -> None:
    open_board = Board.from_fen("4k4/9/9/9/9/9/9/9/9/4K4 r")
    assert "e9e0" in {move.uci() for move in open_board.legal_moves()}

    for blocker_row in range(1, 9):
        ranks = ["9"] * 10
        ranks[0] = "4k4"
        ranks[blocker_row] = "4P4"
        ranks[9] = "4K4"
        board = Board.from_fen("/".join(ranks) + " r")
        assert "e9e0" not in {move.uci() for move in board.legal_moves()}


def test_a_pinned_piece_cannot_expose_the_generals() -> None:
    board = Board.from_fen("4k4/9/9/9/9/4R4/9/9/9/4K4 r")
    legal = {move.uci() for move in board.legal_moves()}

    assert "e5d5" not in legal
    assert "e5f5" not in legal
    assert "e5e4" in legal
    assert "e5e6" in legal


def test_check_evasion_excludes_every_move_that_leaves_the_king_attacked() -> None:
    board = Board.from_fen("4k4/4r4/9/9/9/9/9/9/9/R3K4 r")
    legal = board.legal_moves()

    assert {move.uci() for move in legal} == {"e9d9", "e9f9"}
    for move in legal:
        captured = board.push(move)
        assert not board.is_in_check(RED)
        board.pop(move, captured)


def test_quiet_threefold_repetition_is_a_draw() -> None:
    board = Board.from_fen("r3k4/9/9/9/9/4P4/9/9/9/R3K4 r")
    cycle = ["a9b9", "a0b0", "b9a9", "b0a0"]

    for move_text in cycle * 2:
        board.push(Move.from_uci(move_text))

    assert board.repetition_count() == 3
    assert board.adjudication() == Adjudication("threefold_repetition", None)
    assert board.legal_moves() == []


def test_single_perpetual_checker_loses_on_threefold_repetition() -> None:
    board = Board.from_fen("4k4/9/3R5/9/9/4P4/9/9/9/4K4 r")
    cycle = ["d2e2", "e0d0", "e2d2", "d0e0"]

    for move_text in cycle * 2:
        board.push(Move.from_uci(move_text))

    assert board.adjudication() == Adjudication("perpetual_check", BLACK)
    assert board.result_for(RED) == -1.0
    assert board.result_for(BLACK) == 1.0


def test_copy_preserves_history_without_sharing_future_moves() -> None:
    board = Board.from_fen("r3k4/9/9/9/9/4P4/9/9/9/R3K4 r")
    cycle = ["a9b9", "a0b0", "b9a9", "b0a0"]
    for move_text in cycle:
        board.push(Move.from_uci(move_text))

    copied = board.copy()
    for move_text in cycle:
        copied.push(Move.from_uci(move_text))

    assert copied.adjudication() == Adjudication("threefold_repetition", None)
    assert board.adjudication() is None
    assert board.repetition_count() == 2


def test_120_ply_no_progress_limit_and_resets() -> None:
    quiet = Board.from_fen("r3k4/9/9/9/9/4P4/9/9/9/R3K4 r")
    quiet.no_progress_plies = NO_PROGRESS_PLIES - 1
    quiet.push(Move.from_uci("a9b9"))
    assert quiet.adjudication() == Adjudication("no_progress_120_plies", None)

    capture = Board.from_fen("4k4/9/9/9/9/4P4/9/9/h8/R3K4 r")
    capture.no_progress_plies = NO_PROGRESS_PLIES - 1
    capture.push(Move.from_uci("a9a8"))
    assert capture.no_progress_plies == 0
    assert capture.adjudication() is None

    pawn = Board.from_fen("4k4/9/9/9/9/4P4/P8/9/9/R3K4 r")
    pawn.no_progress_plies = NO_PROGRESS_PLIES - 1
    pawn.push(Move.from_uci("a6a5"))
    assert pawn.no_progress_plies == 0
    assert pawn.adjudication() is None


@pytest.mark.parametrize(
    "fen, message",
    [
        ("9/9/9/9/9/9/9/9/9/9 r", "at least one king"),
        ("4k4/9/9/9/9/9/9/9/4K4/4K4 r", "multiple kings"),
        ("k8/9/9/9/9/9/9/9/9/4K4 r", "inside its palace"),
        ("4k4/9/9/9/9/9/9/9/9/AAAK5 r", "too many red A"),
        ("4k4/9/9/9/9/9/9/9/9/04K4 r", "between 1 and 9"),
    ],
)
def test_illegal_fen_states_are_rejected(fen: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Board.from_fen(fen)


def test_illegal_moves_and_mismatched_undo_are_rejected() -> None:
    board = Board.start()
    with pytest.raises(ValueError, match="Illegal move"):
        board.push(Move.from_uci("a0a1"))
    with pytest.raises(ValueError, match="Illegal move"):
        board.push(Move.from_uci("a9b8"))

    move = Move.from_uci("a6a5")
    captured = board.push(move)
    with pytest.raises(ValueError, match="reverse push order"):
        board.pop(Move.from_uci("c6c5"), captured)
    board.pop(move, captured)


def test_king_capture_and_no_legal_moves_are_consistent_terminal_states() -> None:
    capture = Board.from_fen("4k4/9/9/9/9/9/9/9/9/4K4 r")
    capture.push(Move.from_uci("e9e0"))
    restored_capture = Board.from_fen(capture.to_fen())
    trapped = Board.from_fen("4k4/9/9/9/9/4p4/9/9/3r1r3/4K4 r")

    for board, expected in (
        (capture, Adjudication("king_capture", RED)),
        (restored_capture, Adjudication("king_capture", RED)),
        (trapped, Adjudication("no_legal_moves", BLACK)),
    ):
        assert board.adjudication() == expected
        assert board.is_game_over()
        assert board.legal_moves() == []
        assert board.result_for(RED) == -board.result_for(BLACK)


def test_random_legal_positions_round_trip_and_every_move_is_reversible() -> None:
    randomizer = random.Random(20260714)

    for _game in range(8):
        board = Board.start()
        for _ply in range(36):
            fen = board.to_fen()
            restored = Board.from_fen(fen)
            assert restored.to_fen() == fen
            assert set(restored.legal_moves()) == set(board.legal_moves())

            legal = board.legal_moves()
            if not legal:
                assert board.adjudication() is not None
                break
            sample = legal if len(legal) <= 8 else randomizer.sample(legal, 8)
            for move in sample:
                state = (
                    board.to_fen(),
                    board.no_progress_plies,
                    board.repetition_count(),
                    board.adjudication(),
                )
                captured = board.push(move)
                assert not board.is_in_check(opposite(board.turn))
                board.pop(move, captured)
                assert (
                    board.to_fen(),
                    board.no_progress_plies,
                    board.repetition_count(),
                    board.adjudication(),
                ) == state
            board.push(randomizer.choice(legal))


def test_random_terminal_results_are_symmetric_and_have_no_legal_moves() -> None:
    randomizer = random.Random(8675309)

    for _game in range(8):
        board = Board.start()
        for _ply in range(100):
            legal = board.legal_moves()
            if not legal:
                break
            board.push(randomizer.choice(legal))
        result = board.adjudication()
        if result is not None:
            assert board.is_game_over()
            assert board.legal_moves() == []
            assert board.result_for(RED) == -board.result_for(BLACK)
            if result.winner is not None:
                assert result.winner in {RED, BLACK}
                assert result.winner == opposite(opposite(result.winner))
