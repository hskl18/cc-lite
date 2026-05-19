from __future__ import annotations

import random

from xiangqi.board import Board, Move


def random_move(board: Board) -> Move:
    return random.choice(board.legal_moves())


def material_move(board: Board) -> Move:
    color = board.turn
    best_score = None
    best_moves: list[Move] = []
    for move in board.legal_moves():
        captured = board.push(move)
        score = board.material_score(color)
        board.pop(move, captured)
        if best_score is None or score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)
    return random.choice(best_moves)

