from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xiangqi.board import BOARD_COLS, BOARD_ROWS, START_FEN, Board, Move, rc_to_sq


@dataclass(frozen=True, slots=True)
class OpeningPosition:
    opening_id: str
    fen: str
    mirror_of: str | None
    moves_from_start: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class OpeningSuite:
    suite_id: str
    version: int
    positions: tuple[OpeningPosition, ...]
    path: Path
    sha256: str


def mirror_fen(fen: str) -> str:
    """Reflect a Xiangqi position across the central file."""
    board = Board.from_fen(fen)
    mirrored = board.copy()
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            mirrored.squares[rc_to_sq(row, col)] = board.squares[
                rc_to_sq(row, BOARD_COLS - 1 - col)
            ]
    return mirrored.to_fen()


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def load_opening_suite(path: str | Path) -> OpeningSuite:
    source = Path(path)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Opening suite is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Opening suite must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("Opening suite schema_version must be 1")
    suite_id = _require_string(payload.get("suite_id"), "suite_id")
    version = payload.get("version")
    if not isinstance(version, int) or version < 1:
        raise ValueError("version must be a positive integer")
    raw_positions = payload.get("positions")
    if not isinstance(raw_positions, list) or not raw_positions:
        raise ValueError("positions must be a non-empty list")

    positions: list[OpeningPosition] = []
    seen_ids: set[str] = set()
    by_id: dict[str, OpeningPosition] = {}
    for index, item in enumerate(raw_positions):
        if not isinstance(item, dict):
            raise ValueError(f"positions[{index}] must be an object")
        opening_id = _require_string(item.get("id"), f"positions[{index}].id")
        if opening_id in seen_ids:
            raise ValueError(f"Duplicate opening id {opening_id!r}")
        fen = _require_string(item.get("fen"), f"positions[{index}].fen")
        board = Board.from_fen(fen)
        if board.king_square("red") is None or board.king_square("black") is None:
            raise ValueError(f"Opening {opening_id!r} must contain both kings")
        if not board.legal_moves():
            raise ValueError(f"Opening {opening_id!r} must have at least one legal move")
        canonical_fen = board.to_fen()
        mirror_of = item.get("mirror_of")
        if mirror_of is not None:
            mirror_of = _require_string(mirror_of, f"positions[{index}].mirror_of")
        raw_moves = item.get("moves_from_start")
        if raw_moves is None:
            moves_from_start = None
        elif not isinstance(raw_moves, list) or not raw_moves or not all(
            isinstance(move, str) and move for move in raw_moves
        ):
            raise ValueError(f"positions[{index}].moves_from_start must be a non-empty string list")
        else:
            moves_from_start = tuple(raw_moves)
        if mirror_of is None and moves_from_start is None:
            raise ValueError(f"Base opening {opening_id!r} requires moves_from_start provenance")
        if mirror_of is not None and moves_from_start is not None:
            raise ValueError(f"Mirrored opening {opening_id!r} must not declare moves_from_start")
        if moves_from_start is not None:
            replay = Board.from_fen(START_FEN)
            for ply, move_text in enumerate(moves_from_start):
                try:
                    move = Move.from_uci(move_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Opening {opening_id!r} has invalid provenance move at ply {ply}"
                    ) from exc
                if move not in set(replay.legal_moves()):
                    raise ValueError(
                        f"Opening {opening_id!r} has illegal provenance move {move_text!r} at ply {ply}"
                    )
                replay.push(move)
            if replay.to_fen() != canonical_fen:
                raise ValueError(
                    f"Opening {opening_id!r} FEN does not match moves_from_start provenance"
                )
        position = OpeningPosition(opening_id, canonical_fen, mirror_of, moves_from_start)
        positions.append(position)
        by_id[opening_id] = position
        seen_ids.add(opening_id)

    bases = [position for position in positions if position.mirror_of is None]
    mirrors = [position for position in positions if position.mirror_of is not None]
    if not bases or len(bases) != len(mirrors):
        raise ValueError("Opening suite must contain one mirror for every base position")
    mirrored_base_ids: set[str] = set()
    for position in mirrors:
        base = by_id.get(position.mirror_of or "")
        if base is None or base.mirror_of is not None:
            raise ValueError(f"Opening {position.opening_id!r} references an invalid mirror base")
        if base.opening_id in mirrored_base_ids:
            raise ValueError(f"Opening {base.opening_id!r} has more than one mirror")
        if position.fen != mirror_fen(base.fen):
            raise ValueError(f"Opening {position.opening_id!r} is not the file mirror of its base")
        mirrored_base_ids.add(base.opening_id)
    if mirrored_base_ids != {position.opening_id for position in bases}:
        raise ValueError("Every base opening must have exactly one mirror")

    return OpeningSuite(
        suite_id=suite_id,
        version=version,
        positions=tuple(positions),
        path=source,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
