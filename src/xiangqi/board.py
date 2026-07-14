from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
RED = "red"
BLACK = "black"
START_FEN = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR r"

PIECE_KINDS = ("K", "A", "E", "H", "R", "C", "P")
PIECE_VALUES = {"K": 0, "A": 20, "E": 20, "H": 45, "R": 90, "C": 50, "P": 10}
REPETITION_COUNT = 3
NO_PROGRESS_PLIES = 120


@dataclass(frozen=True, slots=True)
class Adjudication:
    """A terminal decision under the cc-lite research rules profile."""

    reason: str
    winner: str | None

    def result_for(self, color: str) -> float:
        if color not in {RED, BLACK}:
            raise ValueError(f"Unknown color {color!r}")
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == color else -1.0


@dataclass(frozen=True, slots=True)
class Move:
    from_sq: int
    to_sq: int

    def uci(self) -> str:
        return square_name(self.from_sq) + square_name(self.to_sq)

    @classmethod
    def from_uci(cls, text: str) -> "Move":
        if len(text) != 4:
            raise ValueError(f"Expected four-character move, got {text!r}")
        return cls(parse_square(text[:2]), parse_square(text[2:]))


@dataclass(frozen=True, slots=True)
class _MoveRecord:
    mover: str
    gave_check: bool


@dataclass(frozen=True, slots=True)
class _UndoState:
    move: Move
    captured: str | None
    prior_no_progress_plies: int


def rc_to_sq(row: int, col: int) -> int:
    return row * BOARD_COLS + col


def sq_to_rc(square: int) -> tuple[int, int]:
    return divmod(square, BOARD_COLS)


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS


def square_name(square: int) -> str:
    row, col = sq_to_rc(square)
    return f"{chr(ord('a') + col)}{row}"


def parse_square(text: str) -> int:
    if len(text) != 2:
        raise ValueError(f"Bad square name {text!r}")
    col = ord(text[0].lower()) - ord("a")
    row = int(text[1])
    if not in_bounds(row, col):
        raise ValueError(f"Square out of bounds: {text!r}")
    return rc_to_sq(row, col)


def piece_color(piece: str) -> str:
    return RED if piece.isupper() else BLACK


def piece_kind(piece: str) -> str:
    return piece.upper()


def opposite(color: str) -> str:
    if color not in {RED, BLACK}:
        raise ValueError(f"Unknown color {color!r}")
    return BLACK if color == RED else RED


class Board:
    """Mutable Xiangqi board with legal move generation.

    Coordinates are row-major from Black's home rank at row 0 to Red's home
    rank at row 9. Files are a-i, so the starting red rook on the left is a9.
    """

    def __init__(self, squares: list[str | None] | None = None, turn: str = RED):
        if turn not in {RED, BLACK}:
            raise ValueError(f"Unknown color {turn!r}")
        if squares is not None and len(squares) != BOARD_SIZE:
            raise ValueError(f"Expected {BOARD_SIZE} squares, got {len(squares)}")
        self.squares = squares.copy() if squares is not None else [None] * BOARD_SIZE
        self.turn = turn
        self.no_progress_plies = 0
        self._position_history = [self._position_key()]
        self._move_history: list[_MoveRecord] = []
        self._undo_stack: list[_UndoState] = []

    @classmethod
    def start(cls) -> "Board":
        return cls.from_fen(START_FEN)

    @classmethod
    def from_fen(cls, fen: str) -> "Board":
        parts = fen.strip().split()
        if not 1 <= len(parts) <= 2:
            raise ValueError(f"Expected placement and optional side token, got {fen!r}")
        placement = parts[0]
        side = "r" if len(parts) == 1 else parts[1].lower()
        if side not in {"r", "b"}:
            raise ValueError(f"Expected side token 'r' or 'b', got {side!r}")
        turn = RED if side == "r" else BLACK
        squares: list[str | None] = []
        ranks = placement.split("/")
        if len(ranks) != BOARD_ROWS:
            raise ValueError(f"Expected {BOARD_ROWS} ranks, got {len(ranks)}")
        for rank_index, rank in enumerate(ranks):
            rank_squares: list[str | None] = []
            for ch in rank:
                if ch.isdigit():
                    if ch == "0":
                        raise ValueError("FEN empty-square counts must be between 1 and 9")
                    rank_squares.extend([None] * int(ch))
                elif ch.upper() in PIECE_KINDS:
                    rank_squares.append(ch)
                else:
                    raise ValueError(f"Unknown FEN piece {ch!r}")
            if len(rank_squares) != BOARD_COLS:
                raise ValueError(
                    f"Expected {BOARD_COLS} squares in rank {rank_index}, got {len(rank_squares)}"
                )
            squares.extend(rank_squares)
        if squares.count("K") > 1 or squares.count("k") > 1:
            raise ValueError("A position cannot contain multiple kings for one side")
        if squares.count("K") + squares.count("k") == 0:
            raise ValueError("A position must contain at least one king")
        if squares.count("K") == squares.count("k") == 1:
            for color, king in ((RED, "K"), (BLACK, "k")):
                row, col = sq_to_rc(squares.index(king))
                if not cls._palace_contains(color, row, col):
                    raise ValueError(f"The {color} king must be inside its palace")
        maxima = {"K": 1, "A": 2, "E": 2, "H": 2, "R": 2, "C": 2, "P": 5}
        for color, is_color in ((RED, str.isupper), (BLACK, str.islower)):
            for kind, maximum in maxima.items():
                count = sum(
                    piece is not None and is_color(piece) and piece.upper() == kind
                    for piece in squares
                )
                if count > maximum:
                    raise ValueError(f"A position contains too many {color} {kind} pieces")
        return cls(squares, turn)

    def copy(self) -> "Board":
        board = Board(self.squares, self.turn)
        board.no_progress_plies = self.no_progress_plies
        board._position_history = self._position_history.copy()
        board._move_history = self._move_history.copy()
        board._undo_stack = self._undo_stack.copy()
        return board

    def to_fen(self) -> str:
        ranks = []
        for row in range(BOARD_ROWS):
            empty = 0
            chars = []
            for col in range(BOARD_COLS):
                piece = self.squares[rc_to_sq(row, col)]
                if piece is None:
                    empty += 1
                else:
                    if empty:
                        chars.append(str(empty))
                        empty = 0
                    chars.append(piece)
            if empty:
                chars.append(str(empty))
            ranks.append("".join(chars))
        return "/".join(ranks) + (" r" if self.turn == RED else " b")

    def piece_at(self, square: int) -> str | None:
        if not 0 <= square < BOARD_SIZE:
            raise ValueError(f"Square out of bounds: {square}")
        return self.squares[square]

    def push(self, move: Move) -> str | None:
        if not isinstance(move, Move):
            raise TypeError(f"Expected Move, got {type(move).__name__}")
        if (
            self.king_square(RED) is None
            or self.king_square(BLACK) is None
            or self._history_adjudication() is not None
        ):
            raise ValueError("Cannot move after the game has ended")
        legal_moves = set(self._legal_moves_unadjudicated())
        if not legal_moves:
            raise ValueError("Cannot move after the game has ended")
        if move not in legal_moves:
            raise ValueError(f"Illegal move {move.uci()} for {self.to_fen()}")
        piece = self.squares[move.from_sq]
        assert piece is not None
        prior_no_progress_plies = self.no_progress_plies
        captured = self._push_unchecked(move)
        if captured is not None or piece_kind(piece) == "P":
            self.no_progress_plies = 0
        else:
            self.no_progress_plies += 1
        self._move_history.append(
            _MoveRecord(mover=piece_color(piece), gave_check=self.is_in_check(self.turn))
        )
        self._position_history.append(self._position_key())
        self._undo_stack.append(_UndoState(move, captured, prior_no_progress_plies))
        return captured

    def pop(self, move: Move, captured: str | None) -> None:
        if not self._undo_stack:
            raise ValueError("Cannot pop without a matching push")
        undo = self._undo_stack[-1]
        if undo.move != move or undo.captured != captured:
            raise ValueError("Moves must be popped in reverse push order with the matching capture")
        self._undo_stack.pop()
        self._position_history.pop()
        self._move_history.pop()
        self.no_progress_plies = undo.prior_no_progress_plies
        self._pop_unchecked(move, captured)

    def legal_moves(self) -> list[Move]:
        if self._history_adjudication() is not None:
            return []
        if self.king_square(RED) is None or self.king_square(BLACK) is None:
            return []
        return self._legal_moves_unadjudicated()

    def _legal_moves_unadjudicated(self) -> list[Move]:
        color = self.turn
        legal: list[Move] = []
        for move in self.pseudo_legal_moves(color):
            captured = self._push_unchecked(move)
            if not self.is_in_check(color):
                legal.append(move)
            self._pop_unchecked(move, captured)
        return legal

    def pseudo_legal_moves(self, color: str | None = None) -> Iterable[Move]:
        color = self.turn if color is None else color
        for square, piece in enumerate(self.squares):
            if piece is None or piece_color(piece) != color:
                continue
            yield from self._piece_moves(square, piece)

    def is_game_over(self) -> bool:
        return self.adjudication() is not None

    def adjudication(self) -> Adjudication | None:
        red_king = self.king_square(RED)
        black_king = self.king_square(BLACK)
        if red_king is None:
            return Adjudication("king_capture", BLACK)
        if black_king is None:
            return Adjudication("king_capture", RED)
        history_result = self._history_adjudication()
        if history_result is not None:
            return history_result
        if not self._legal_moves_unadjudicated():
            return Adjudication("no_legal_moves", opposite(self.turn))
        return None

    def result_for(self, color: str) -> float:
        if color not in {RED, BLACK}:
            raise ValueError(f"Unknown color {color!r}")
        result = self.adjudication()
        if result is not None:
            return result.result_for(color)
        return 0.0

    def repetition_count(self) -> int:
        key = self._position_key()
        return sum(previous == key for previous in self._position_history)

    def _history_adjudication(self) -> Adjudication | None:
        occurrence_indices = [
            index
            for index, key in enumerate(self._position_history)
            if key == self._position_history[-1]
        ]
        if len(occurrence_indices) >= REPETITION_COUNT:
            cycle_start = occurrence_indices[-REPETITION_COUNT]
            cycle_records = self._move_history[cycle_start:]
            perpetual_checkers = []
            for color in (RED, BLACK):
                records = [record for record in cycle_records if record.mover == color]
                if records and all(record.gave_check for record in records):
                    perpetual_checkers.append(color)
            if len(perpetual_checkers) == 1:
                return Adjudication("perpetual_check", opposite(perpetual_checkers[0]))
            return Adjudication("threefold_repetition", None)
        if self.no_progress_plies >= NO_PROGRESS_PLIES:
            return Adjudication("no_progress_120_plies", None)
        return None

    def _position_key(self) -> tuple[tuple[str | None, ...], str]:
        return tuple(self.squares), self.turn

    def _push_unchecked(self, move: Move) -> str | None:
        piece = self.squares[move.from_sq]
        captured = self.squares[move.to_sq]
        self.squares[move.to_sq] = piece
        self.squares[move.from_sq] = None
        self.turn = opposite(self.turn)
        return captured

    def _pop_unchecked(self, move: Move, captured: str | None) -> None:
        piece = self.squares[move.to_sq]
        self.squares[move.from_sq] = piece
        self.squares[move.to_sq] = captured
        self.turn = opposite(self.turn)

    def is_in_check(self, color: str) -> bool:
        king = self.king_square(color)
        if king is None:
            return True
        for move in self.pseudo_legal_moves(opposite(color)):
            if move.to_sq == king:
                return True
        return False

    def king_square(self, color: str) -> int | None:
        target = "K" if color == RED else "k"
        try:
            return self.squares.index(target)
        except ValueError:
            return None

    def material_score(self, color: str) -> int:
        score = 0
        for piece in self.squares:
            if piece is None:
                continue
            value = PIECE_VALUES[piece_kind(piece)]
            score += value if piece_color(piece) == color else -value
        return score

    def _piece_moves(self, square: int, piece: str) -> Iterable[Move]:
        kind = piece_kind(piece)
        if kind == "K":
            yield from self._king_moves(square, piece)
        elif kind == "A":
            yield from self._advisor_moves(square, piece)
        elif kind == "E":
            yield from self._elephant_moves(square, piece)
        elif kind == "H":
            yield from self._horse_moves(square, piece)
        elif kind == "R":
            yield from self._rook_moves(square, piece)
        elif kind == "C":
            yield from self._cannon_moves(square, piece)
        elif kind == "P":
            yield from self._pawn_moves(square, piece)

    def _can_land(self, color: str, row: int, col: int) -> bool:
        if not in_bounds(row, col):
            return False
        target = self.squares[rc_to_sq(row, col)]
        return target is None or piece_color(target) != color

    @staticmethod
    def _palace_contains(color: str, row: int, col: int) -> bool:
        if not 3 <= col <= 5:
            return False
        return 7 <= row <= 9 if color == RED else 0 <= row <= 2

    def _king_moves(self, square: int, piece: str) -> Iterable[Move]:
        color = piece_color(piece)
        row, col = sq_to_rc(square)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if self._palace_contains(color, nr, nc) and self._can_land(color, nr, nc):
                yield Move(square, rc_to_sq(nr, nc))
        step = -1 if color == RED else 1
        nr = row + step
        while in_bounds(nr, col):
            target = self.squares[rc_to_sq(nr, col)]
            if target is not None:
                if piece_kind(target) == "K" and piece_color(target) != color:
                    yield Move(square, rc_to_sq(nr, col))
                break
            nr += step

    def _advisor_moves(self, square: int, piece: str) -> Iterable[Move]:
        color = piece_color(piece)
        row, col = sq_to_rc(square)
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = row + dr, col + dc
            if self._palace_contains(color, nr, nc) and self._can_land(color, nr, nc):
                yield Move(square, rc_to_sq(nr, nc))

    def _elephant_moves(self, square: int, piece: str) -> Iterable[Move]:
        color = piece_color(piece)
        row, col = sq_to_rc(square)
        for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
            nr, nc = row + dr, col + dc
            eye = rc_to_sq(row + dr // 2, col + dc // 2)
            if not in_bounds(nr, nc) or self.squares[eye] is not None:
                continue
            if color == RED and nr < 5:
                continue
            if color == BLACK and nr > 4:
                continue
            if self._can_land(color, nr, nc):
                yield Move(square, rc_to_sq(nr, nc))

    def _horse_moves(self, square: int, piece: str) -> Iterable[Move]:
        color = piece_color(piece)
        row, col = sq_to_rc(square)
        specs = (
            (-2, -1, -1, 0),
            (-2, 1, -1, 0),
            (2, -1, 1, 0),
            (2, 1, 1, 0),
            (-1, -2, 0, -1),
            (1, -2, 0, -1),
            (-1, 2, 0, 1),
            (1, 2, 0, 1),
        )
        for dr, dc, lr, lc in specs:
            nr, nc = row + dr, col + dc
            leg_r, leg_c = row + lr, col + lc
            if in_bounds(nr, nc) and self.squares[rc_to_sq(leg_r, leg_c)] is None:
                if self._can_land(color, nr, nc):
                    yield Move(square, rc_to_sq(nr, nc))

    def _rook_moves(self, square: int, piece: str) -> Iterable[Move]:
        yield from self._ray_moves(square, piece, cannon=False)

    def _cannon_moves(self, square: int, piece: str) -> Iterable[Move]:
        yield from self._ray_moves(square, piece, cannon=True)

    def _ray_moves(self, square: int, piece: str, cannon: bool) -> Iterable[Move]:
        color = piece_color(piece)
        row, col = sq_to_rc(square)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            jumped = False
            while in_bounds(nr, nc):
                target_sq = rc_to_sq(nr, nc)
                target = self.squares[target_sq]
                if not cannon:
                    if target is None:
                        yield Move(square, target_sq)
                    else:
                        if piece_color(target) != color:
                            yield Move(square, target_sq)
                        break
                else:
                    if not jumped:
                        if target is None:
                            yield Move(square, target_sq)
                        else:
                            jumped = True
                    elif target is not None:
                        if piece_color(target) != color:
                            yield Move(square, target_sq)
                        break
                nr += dr
                nc += dc

    def _pawn_moves(self, square: int, piece: str) -> Iterable[Move]:
        color = piece_color(piece)
        row, col = sq_to_rc(square)
        forward = -1 if color == RED else 1
        candidates = [(row + forward, col)]
        crossed = row <= 4 if color == RED else row >= 5
        if crossed:
            candidates.extend([(row, col - 1), (row, col + 1)])
        for nr, nc in candidates:
            if self._can_land(color, nr, nc):
                yield Move(square, rc_to_sq(nr, nc))
