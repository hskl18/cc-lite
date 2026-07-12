from __future__ import annotations

import numpy as np

from xiangqi.board import BLACK, BOARD_SIZE, RED, Board, Move, piece_color, piece_kind

PIECE_PLANES = {
    (RED, "K"): 0,
    (RED, "A"): 1,
    (RED, "E"): 2,
    (RED, "H"): 3,
    (RED, "R"): 4,
    (RED, "C"): 5,
    (RED, "P"): 6,
    (BLACK, "K"): 7,
    (BLACK, "A"): 8,
    (BLACK, "E"): 9,
    (BLACK, "H"): 10,
    (BLACK, "R"): 11,
    (BLACK, "C"): 12,
    (BLACK, "P"): 13,
}
INPUT_PLANES = 16
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE


class MoveCodec:
    """Fixed from-square/to-square policy vocabulary.

    Index = from_square * 90 + to_square. Only legal moves should receive
    non-zero probability; callers mask illegal indices before sampling.
    """

    policy_size = POLICY_SIZE

    @staticmethod
    def encode(move: Move) -> int:
        return move.from_sq * BOARD_SIZE + move.to_sq

    @staticmethod
    def decode(index: int) -> Move:
        if not 0 <= index < POLICY_SIZE:
            raise ValueError(f"Policy index out of range: {index}")
        return Move(index // BOARD_SIZE, index % BOARD_SIZE)

    @staticmethod
    def legal_indices(board: Board) -> list[int]:
        return [MoveCodec.encode(move) for move in board.legal_moves()]

    @staticmethod
    def legal_mask(board: Board) -> np.ndarray:
        mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
        for index in MoveCodec.legal_indices(board):
            mask[index] = True
        return mask


def encode_board(board: Board) -> np.ndarray:
    planes = np.zeros((INPUT_PLANES, 10, 9), dtype=np.float32)
    for square, piece in enumerate(board.squares):
        if piece is None:
            continue
        row, col = divmod(square, 9)
        planes[PIECE_PLANES[(piece_color(piece), piece_kind(piece))], row, col] = 1.0
    planes[14, :, :] = 1.0 if board.turn == RED else 0.0
    planes[15, :, :] = 1.0 if board.is_in_check(board.turn) else 0.0
    return planes

