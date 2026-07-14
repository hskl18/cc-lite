from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from xiangqi.board import BLACK, RED, START_FEN, Board, Move
from xiangqi.encoding import MoveCodec

SCHEMA_VERSION = 1
RESULT_VALUES = {
    "1-0": {RED: 1.0, BLACK: -1.0},
    "0-1": {RED: -1.0, BLACK: 1.0},
    "1/2-1/2": {RED: 0.0, BLACK: 0.0},
}
REQUIRED_PROVENANCE_FIELDS = (
    "dataset_name",
    "source",
    "license",
    "license_notes",
    "attribution",
)
ARTIFACT_NAMES = (
    "games.jsonl",
    "train.jsonl",
    "validation.jsonl",
    "rejections.jsonl",
    "manifest.json",
)


class IngestionError(ValueError):
    pass


class RecordError(IngestionError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.details = details


@dataclass(frozen=True)
class Candidate:
    line_number: int
    input_sha256: str
    record_id: str
    notation: str
    initial_fen: str
    moves: tuple[str, ...]
    result: str
    game_hash: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_provenance(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IngestionError(f"Could not read provenance JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise IngestionError("Provenance must be a JSON object")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise IngestionError(f"Provenance schema_version must be {SCHEMA_VERSION}")
    for field in REQUIRED_PROVENANCE_FIELDS:
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise IngestionError(f"Provenance field {field!r} must be a non-empty string")
    for field in ("source_url", "license_url"):
        if field in raw and raw[field] is not None and not isinstance(raw[field], str):
            raise IngestionError(f"Optional provenance field {field!r} must be a string or null")
    if "redistribution_permitted" in raw and not isinstance(
        raw["redistribution_permitted"], bool
    ):
        raise IngestionError("Optional provenance field 'redistribution_permitted' must be boolean")
    return raw


def _rejection(
    line_number: int,
    input_sha256: str,
    code: str,
    message: str,
    *,
    record_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rejection: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "line_number": line_number,
        "input_sha256": input_sha256,
        "code": code,
        "message": message,
    }
    if record_id is not None:
        rejection["record_id"] = record_id
    if details:
        rejection["details"] = details
    return rejection


def _record_id(raw: Any) -> str | None:
    if isinstance(raw, dict) and isinstance(raw.get("record_id"), str):
        value = raw["record_id"].strip()
        return value or None
    return None


def _normalize_moves(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = value.split()
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        tokens = value
    else:
        raise RecordError(
            "invalid_moves",
            "moves must be a whitespace-separated string or an array of strings",
        )
    moves = tuple(token.strip().lower() for token in tokens)
    if not moves or any(not move for move in moves):
        raise RecordError(
            "invalid_moves", "moves must contain at least one non-empty move token"
        )
    return moves


def _terminal_result(board: Board) -> str | None:
    adjudication = board.adjudication()
    if adjudication is None:
        return None
    if adjudication.winner is None:
        return "1/2-1/2"
    return "1-0" if adjudication.winner == RED else "0-1"


def _parse_candidate(
    raw: Any,
    *,
    line_number: int,
    input_sha256: str,
) -> Candidate:
    if not isinstance(raw, dict):
        raise RecordError("invalid_record_object", "record must be a JSON object")
    record_id = _record_id(raw)
    if record_id is None:
        raise RecordError("missing_record_id", "record_id must be a non-empty string")
    notation = raw.get("notation", "iccs")
    if not isinstance(notation, str) or notation.lower() not in {"iccs", "ucci"}:
        raise RecordError("unsupported_notation", "notation must be 'iccs' or 'ucci'")
    notation = notation.lower()
    result = raw.get("result")
    if result not in RESULT_VALUES:
        raise RecordError(
            "invalid_result", "result must be '1-0', '0-1', or '1/2-1/2'"
        )
    initial_fen = raw.get("initial_fen", START_FEN)
    if not isinstance(initial_fen, str):
        raise RecordError("invalid_fen", "initial_fen must be a string")
    try:
        board = Board.from_fen(initial_fen)
    except (TypeError, ValueError) as exc:
        raise RecordError("invalid_fen", f"invalid initial_fen: {exc}") from exc
    if board.king_square(RED) is None or board.king_square(BLACK) is None:
        raise RecordError(
            "invalid_position", "initial_fen must contain exactly one king for each side"
        )
    initial_fen = board.to_fen()
    moves = _normalize_moves(raw.get("moves"))
    normalized_moves: list[str] = []
    for ply, move_text in enumerate(moves):
        try:
            move = Move.from_uci(move_text)
        except (TypeError, ValueError) as exc:
            raise RecordError(
                "invalid_move",
                f"invalid move at ply {ply}: {exc}",
                details={"move": move_text, "ply": ply},
            ) from exc
        normalized_move = move.uci()
        legal_moves = set(board.legal_moves())
        if move not in legal_moves:
            raise RecordError(
                "illegal_move",
                f"illegal move at ply {ply}: {normalized_move} is not legal from {board.to_fen()}",
                details={"fen": board.to_fen(), "move": normalized_move, "ply": ply},
            )
        normalized_moves.append(normalized_move)
        board.push(move)
    terminal_result = _terminal_result(board)
    if terminal_result is not None and result != terminal_result:
        raise RecordError(
            "terminal_result_conflict",
            f"result {result!r} conflicts with terminal board result {terminal_result!r}",
            details={"declared_result": result, "terminal_result": terminal_result},
        )
    game_identity = {"initial_fen": initial_fen, "moves": normalized_moves}
    game_hash = _sha256_json(game_identity)
    return Candidate(
        line_number=line_number,
        input_sha256=input_sha256,
        record_id=record_id,
        notation=notation,
        initial_fen=initial_fen,
        moves=tuple(normalized_moves),
        result=result,
        game_hash=game_hash,
    )


def _split_for(game_hash: str, split_seed: str, validation_fraction: float) -> str:
    split_hash = hashlib.sha256(f"{split_seed}:{game_hash}".encode()).digest()
    unit_interval = int.from_bytes(split_hash[:8], "big") / 2**64
    return "validation" if unit_interval < validation_fraction else "train"


def _fingerprint(hashes: list[str]) -> str:
    payload = "".join(f"{item}\n" for item in sorted(hashes)).encode("ascii")
    return _sha256_bytes(payload)


def _samples_for(
    candidates: Iterable[Candidate], provenance_fingerprint: str
) -> Iterable[dict[str, Any]]:
    for candidate in candidates:
        board = Board.from_fen(candidate.initial_fen)
        for ply, move_text in enumerate(candidate.moves):
            move = Move.from_uci(move_text)
            yield {
                "fen": board.to_fen(),
                "move": move_text,
                "policy": {str(MoveCodec.encode(move)): 1.0},
                "ply": ply,
                "value": RESULT_VALUES[candidate.result][board.turn],
                "game_hash": candidate.game_hash,
                "record_id": candidate.record_id,
                "provenance_fingerprint": provenance_fingerprint,
            }
            board.push(move)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    try:
        with temp_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    return count


def _atomic_write(path: Path, content: str) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _stage_output_directory(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".staging",
            dir=output_dir.parent,
        )
    )
    if output_dir.exists():
        shutil.copytree(output_dir, staging_dir, dirs_exist_ok=True, symlinks=True)
    return staging_dir


def _publish_output_directory(staging_dir: Path, output_dir: Path) -> None:
    if not output_dir.exists():
        staging_dir.replace(output_dir)
        return

    backup_dir = staging_dir.with_name(f"{staging_dir.name}.previous")
    output_dir.replace(backup_dir)
    try:
        staging_dir.replace(output_dir)
    except BaseException:
        backup_dir.replace(output_dir)
        raise
    shutil.rmtree(backup_dir)


def ingest_records(
    input_path: str | Path,
    provenance_path: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.1,
    split_seed: str = "0",
    force: bool = False,
) -> dict[str, Any]:
    if not 0.0 <= validation_fraction <= 1.0:
        raise IngestionError("validation_fraction must be between 0 and 1 inclusive")
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    provenance = load_provenance(provenance_path)
    if not input_path.is_file():
        raise IngestionError(f"Input path is not a file: {input_path}")
    existing = [name for name in ARTIFACT_NAMES if (output_dir / name).exists()]
    if existing and not force:
        raise IngestionError(
            f"Output artifacts already exist: {', '.join(existing)}. Pass --force to replace them."
        )

    candidates: list[Candidate] = []
    rejections: list[dict[str, Any]] = []
    input_records = 0
    with input_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            input_records += 1
            input_sha256 = _sha256_bytes(line.encode("utf-8"))
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                rejections.append(
                    _rejection(
                        line_number,
                        input_sha256,
                        "invalid_json",
                        f"Invalid JSON: {exc.msg}",
                    )
                )
                continue
            try:
                candidates.append(
                    _parse_candidate(raw, line_number=line_number, input_sha256=input_sha256)
                )
            except RecordError as exc:
                rejections.append(
                    _rejection(
                        line_number,
                        input_sha256,
                        exc.code,
                        str(exc),
                        record_id=_record_id(raw),
                        details=exc.details,
                    )
                )

    conflicted_lines: set[int] = set()
    by_record_id: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_record_id[candidate.record_id].append(candidate)
    for record_id, records in by_record_id.items():
        game_hashes = {record.game_hash for record in records}
        if len(game_hashes) <= 1:
            continue
        for record in records:
            conflicted_lines.add(record.line_number)
            rejections.append(
                _rejection(
                    record.line_number,
                    record.input_sha256,
                    "record_id_conflict",
                    "record_id refers to multiple distinct move sequences",
                    record_id=record_id,
                    details={"game_hashes": sorted(game_hashes)},
                )
            )

    by_game_hash: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.line_number not in conflicted_lines:
            by_game_hash[candidate.game_hash].append(candidate)

    accepted: list[Candidate] = []
    for game_hash, records in by_game_hash.items():
        results = {record.result for record in records}
        if len(results) > 1:
            for record in records:
                rejections.append(
                    _rejection(
                        record.line_number,
                        record.input_sha256,
                        "conflicting_duplicate",
                        "identical move sequence has conflicting results",
                        record_id=record.record_id,
                        details={"game_hash": game_hash, "results": sorted(results)},
                    )
                )
            continue
        records.sort(key=lambda item: (item.record_id, item.input_sha256, item.line_number))
        winner = records[0]
        accepted.append(winner)
        for duplicate in records[1:]:
            rejections.append(
                _rejection(
                    duplicate.line_number,
                    duplicate.input_sha256,
                    "duplicate_game",
                    "identical normalized move sequence was already accepted",
                    record_id=duplicate.record_id,
                    details={
                        "game_hash": game_hash,
                        "accepted_record_id": winner.record_id,
                    },
                )
            )

    accepted.sort(key=lambda item: item.game_hash)
    provenance_fingerprint = _sha256_json(provenance)
    split_candidates: dict[str, list[Candidate]] = {"train": [], "validation": []}
    for candidate in accepted:
        split = _split_for(candidate.game_hash, split_seed, validation_fraction)
        split_candidates[split].append(candidate)
    split_by_hash = {
        candidate.game_hash: split
        for split, candidates_for_split in split_candidates.items()
        for candidate in candidates_for_split
    }

    def game_records() -> Iterable[dict[str, Any]]:
        for candidate in accepted:
            yield {
                "schema_version": SCHEMA_VERSION,
                "record_id": candidate.record_id,
                "notation": candidate.notation,
                "initial_fen": candidate.initial_fen,
                "moves": list(candidate.moves),
                "result": candidate.result,
                "game_hash": candidate.game_hash,
                "split": split_by_hash[candidate.game_hash],
                "provenance_fingerprint": provenance_fingerprint,
            }

    rejections.sort(key=lambda item: (item["line_number"], item["code"]))
    staging_dir = _stage_output_directory(output_dir)
    try:
        _write_jsonl(staging_dir / "games.jsonl", game_records())
        train_samples = _write_jsonl(
            staging_dir / "train.jsonl",
            _samples_for(split_candidates["train"], provenance_fingerprint),
        )
        validation_samples = _write_jsonl(
            staging_dir / "validation.jsonl",
            _samples_for(split_candidates["validation"], provenance_fingerprint),
        )
        _write_jsonl(staging_dir / "rejections.jsonl", rejections)
        artifact_hashes = {
            name: _file_sha256(staging_dir / name)
            for name in ARTIFACT_NAMES
            if name != "manifest.json"
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "format": "cc-lite-iccs-ucci-jsonl",
            "input": {
                "filename": input_path.name,
                "sha256": _file_sha256(input_path),
                "records": input_records,
            },
            "provenance": provenance,
            "provenance_fingerprint": provenance_fingerprint,
            "deduplication": {
                "identity": "sha256 of canonical initial_fen and normalized moves",
                "accepted_games": len(accepted),
                "rejected_records": len(rejections),
            },
            "split": {
                "algorithm": "sha256(seed + ':' + game_hash), first 64 bits",
                "seed": split_seed,
                "validation_fraction": validation_fraction,
                "train": {
                    "games": len(split_candidates["train"]),
                    "samples": train_samples,
                    "fingerprint": _fingerprint(
                        [candidate.game_hash for candidate in split_candidates["train"]]
                    ),
                },
                "validation": {
                    "games": len(split_candidates["validation"]),
                    "samples": validation_samples,
                    "fingerprint": _fingerprint(
                        [candidate.game_hash for candidate in split_candidates["validation"]]
                    ),
                },
            },
            "artifacts": artifact_hashes,
        }
        _atomic_write(
            staging_dir / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        _publish_output_directory(staging_dir, output_dir)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and split ICCS or UCCI Xiangqi move-record JSONL."
    )
    parser.add_argument("--input", required=True, help="Input move-record JSONL file.")
    parser.add_argument("--provenance", required=True, help="Dataset provenance JSON file.")
    parser.add_argument("--output-dir", required=True, help="Directory for normalized artifacts.")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", default="0")
    parser.add_argument("--force", action="store_true", help="Replace known output artifacts.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        manifest = ingest_records(
            args.input,
            args.provenance,
            args.output_dir,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
            force=args.force,
        )
    except IngestionError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
