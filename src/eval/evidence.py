from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from eval.adjudication import material_adjudication
from eval.openings import load_opening_suite, opening_suite_summary
from eval.paired import (
    build_paired_schedule,
    paired_record_errors,
    sequential_stop_decision,
    summarize_paired_records,
)
from xiangqi.board import Board, Move

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
PAIRED_RECORD_FIELDS = (
    "pair_id",
    "pair_leg",
    "opening_id",
    "candidate_color",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_game_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    terminal_scores = [
        float(record["terminal_score"])
        for record in records
        if record.get("terminal_reason") != "max_plies"
        and record.get("terminal_score") is not None
    ]
    truncation_scores = [
        float(record["material_adjudication"])
        for record in records
        if record.get("terminal_reason") == "max_plies"
        and record.get("material_adjudication") is not None
    ]
    plies = [int(record.get("plies", 0)) for record in records]
    nodes_per_second = [
        float(record["avg_nodes_per_second"])
        for record in records
        if record.get("avg_nodes_per_second") is not None
    ]
    summary = {
        "games": len(records),
        "avg_plies": sum(plies) / len(plies) if plies else 0.0,
        "avg_nodes_per_second": (
            sum(nodes_per_second) / len(nodes_per_second) if nodes_per_second else 0.0
        ),
        "terminal": {
            "games": len(terminal_scores),
            "wins": sum(score > 0 for score in terminal_scores),
            "draws": sum(score == 0 for score in terminal_scores),
            "losses": sum(score < 0 for score in terminal_scores),
        },
        "truncation": {
            "games": len(truncation_scores),
            "material_ahead": sum(score > 0 for score in truncation_scores),
            "material_equal": sum(score == 0 for score in truncation_scores),
            "material_behind": sum(score < 0 for score in truncation_scores),
            "material_score_mean": (
                sum(truncation_scores) / len(truncation_scores) if truncation_scores else 0.0
            ),
        },
    }
    if any(record.get("pair_id") for record in records):
        summary["paired"] = summarize_paired_records(records)
    return summary


def _add_paired_protocol_summary(
    summary: dict[str, Any],
    records: Sequence[dict[str, Any]],
    opening_suite_path: str | Path,
    protocol: dict[str, Any],
) -> None:
    suite = load_opening_suite(opening_suite_path)
    openings = {opening.opening_id: opening.fen for opening in suite.positions}
    for index, record in enumerate(records, start=1):
        opening_id = record.get("opening_id")
        if opening_id not in openings:
            raise ValueError(f"game {index}: opening_id is not in the pinned suite")
        if record.get("start_fen") != openings[opening_id]:
            raise ValueError(f"game {index}: start_fen does not match the pinned opening")
    summary["sequential_stop"] = sequential_stop_decision(
        records,
        min_pairs=int(protocol["min_pairs"]),
        max_pairs=int(protocol["max_pairs"]),
        min_seeds=int(protocol["min_seeds"]),
        confidence=float(protocol["confidence"]),
        superiority_margin=float(protocol["superiority_margin"]),
    )
    summary["opening_suite"] = opening_suite_summary(suite)


def _record_uses_paired_metadata(record: dict[str, Any]) -> bool:
    return any(field in record for field in PAIRED_RECORD_FIELDS)


def _record_perspective(
    record: dict[str, Any], index: int, *, paired: bool
) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    candidate_present = "candidate_color" in record
    model_present = "model_color" in record
    candidate_color = record.get("candidate_color")
    model_color = record.get("model_color")

    if candidate_present and candidate_color not in ("red", "black"):
        errors.append(f"game {index}: candidate_color must be red or black when present")
    if model_present and model_color not in ("red", "black"):
        errors.append(f"game {index}: model_color must be red or black when present")
    if (
        candidate_present
        and model_present
        and candidate_color in ("red", "black")
        and model_color in ("red", "black")
        and candidate_color != model_color
    ):
        errors.append(f"game {index}: candidate_color conflicts with model_color")

    if paired:
        if not candidate_present:
            errors.append(f"game {index}: paired record requires candidate_color")
        perspective = candidate_color if candidate_color in ("red", "black") else None
    else:
        if not model_present:
            errors.append(f"game {index}: unpaired record requires model_color")
        perspective = model_color if model_color in ("red", "black") else None
    return perspective, errors


def _replay_game_record(
    record: dict[str, Any], index: int, *, perspective: str | None
) -> list[str]:
    errors: list[str] = []
    start_fen = record.get("start_fen")
    moves = record.get("moves")
    final_fen = record.get("final_fen")
    plies = record.get("plies")
    if not isinstance(start_fen, str) or not start_fen:
        errors.append(f"game {index}: start_fen must be a string")
    if not isinstance(moves, list) or any(not isinstance(move, str) for move in moves):
        errors.append(f"game {index}: moves must be a list of move strings")
    if not isinstance(final_fen, str) or not final_fen:
        errors.append(f"game {index}: final_fen must be a string")
    if not isinstance(plies, int) or isinstance(plies, bool):
        errors.append(f"game {index}: plies must be an integer")
    elif isinstance(moves, list) and plies != len(moves):
        errors.append(f"game {index}: plies must equal the number of raw moves")
    if errors:
        return errors

    try:
        board = Board.from_fen(start_fen)
        for move_text in moves:
            board.push(Move.from_uci(move_text))
    except (TypeError, ValueError) as exc:
        return [f"game {index}: cannot replay raw moves: {exc}"]
    if board.to_fen() != final_fen:
        errors.append(f"game {index}: final_fen does not match raw moves")
    adjudication = board.adjudication()
    if record.get("terminal_reason") == "max_plies":
        if adjudication is not None:
            errors.append(f"game {index}: max_plies position is already terminal")
        if perspective is not None:
            expected_delta = board.material_score(perspective)
            material_delta = record.get("material_delta")
            if (
                not isinstance(material_delta, int)
                or isinstance(material_delta, bool)
                or material_delta != expected_delta
            ):
                errors.append(f"game {index}: material_delta does not match replay")
            expected_adjudication = material_adjudication(expected_delta)
            if record.get("material_adjudication") != expected_adjudication:
                errors.append(
                    f"game {index}: material_adjudication does not match replay"
                )
    elif adjudication is None:
        errors.append(f"game {index}: terminal record does not replay to a terminal position")
    else:
        if record.get("terminal_reason") != adjudication.reason:
            errors.append(f"game {index}: terminal_reason does not match replay")
        if perspective is not None and record.get("terminal_score") != adjudication.result_for(
            perspective
        ):
            errors.append(f"game {index}: terminal_score does not match replay")
    return errors


def _canonical_paired_schedule_errors(
    records: Sequence[dict[str, Any]],
    *,
    opening_suite_path: str | Path,
    seeds: object,
    protocol: object,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(protocol, dict):
        return ["paired_protocol must be an object"]
    if not isinstance(seeds, list) or not seeds:
        return ["manifest evaluation seeds must be a non-empty list"]
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        return ["manifest evaluation seeds must be integers"]
    if len(seeds) != len(set(seeds)):
        return ["manifest evaluation seeds must be unique"]

    max_pairs = protocol.get("max_pairs")
    if not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or max_pairs < 1:
        return ["paired_protocol.max_pairs must be a positive integer"]
    min_pairs = protocol.get("min_pairs")
    min_seeds = protocol.get("min_seeds")
    confidence = protocol.get("confidence")
    superiority_margin = protocol.get("superiority_margin")
    if not isinstance(min_pairs, int) or isinstance(min_pairs, bool):
        return ["paired_protocol.min_pairs must be an integer"]
    if not isinstance(min_seeds, int) or isinstance(min_seeds, bool):
        return ["paired_protocol.min_seeds must be an integer"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return ["paired_protocol.confidence must be numeric"]
    if not isinstance(superiority_margin, (int, float)) or isinstance(
        superiority_margin, bool
    ):
        return ["paired_protocol.superiority_margin must be numeric"]
    try:
        sequential_stop_decision(
            [],
            min_pairs=min_pairs,
            max_pairs=max_pairs,
            min_seeds=min_seeds,
            confidence=float(confidence),
            superiority_margin=float(superiority_margin),
        )
    except ValueError as exc:
        return [f"invalid paired protocol: {exc}"]
    try:
        suite = load_opening_suite(opening_suite_path)
        schedule = build_paired_schedule(suite, seeds)
    except (TypeError, ValueError, OSError) as exc:
        return [f"cannot build canonical paired schedule: {exc}"]

    available_pairs = len(schedule) // 2
    if max_pairs > available_pairs:
        errors.append("paired_protocol.max_pairs exceeds the canonical paired schedule")
    if not records:
        errors.append("paired evidence must contain at least one complete pair")
    if len(records) % 2:
        errors.append("paired evidence must end on a complete pair boundary")
    if len(records) > max_pairs * 2:
        errors.append("paired evidence exceeds paired_protocol.max_pairs")

    expected_prefix = schedule[: min(len(records), max_pairs * 2)]
    schedule_fields = (
        "game_id",
        "pair_id",
        "pair_leg",
        "opening_id",
        "start_fen",
        "seed",
        "candidate_color",
    )
    for index, (record, expected) in enumerate(
        zip(records, expected_prefix, strict=False), start=1
    ):
        for field in schedule_fields:
            actual_value = record.get(field)
            expected_value = getattr(expected, field)
            if type(actual_value) is not type(expected_value) or actual_value != expected_value:
                errors.append(
                    f"game {index}: {field} does not match canonical paired schedule"
                )

    if len(records) % 2 == 0 and records:
        observed_pairs = len(records) // 2
        first_stop_pair: int | None = None
        final_stop: dict[str, Any] | None = None
        try:
            for pair_count in range(1, observed_pairs + 1):
                final_stop = sequential_stop_decision(
                    records[: pair_count * 2],
                    min_pairs=min_pairs,
                    max_pairs=max_pairs,
                    min_seeds=min_seeds,
                    confidence=float(confidence),
                    superiority_margin=float(superiority_margin),
                )
                if final_stop["stop"]:
                    first_stop_pair = pair_count
                    break
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid paired protocol: {exc}")
        else:
            if final_stop is not None and final_stop["attempted_pairs"] != observed_pairs:
                errors.append("sequential stop attempted_pairs does not match raw pair count")
            if first_stop_pair is not None and first_stop_pair != observed_pairs:
                errors.append("paired evidence continues after the sequential stop decision")
            if first_stop_pair is None and observed_pairs < max_pairs:
                errors.append("paired evidence ends before a sequential stop decision")
    return errors


def _game_record_errors(record: dict[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    perspective, perspective_errors = _record_perspective(
        record, index, paired=_record_uses_paired_metadata(record)
    )
    errors.extend(perspective_errors)
    if record.get("terminal_reason") == "max_plies":
        if record.get("terminal_score") is not None:
            errors.append(f"game {index}: max_plies must not have terminal_score")
        material_score = record.get("material_adjudication")
        if material_score is None:
            errors.append(f"game {index}: max_plies requires material_adjudication")
        elif isinstance(material_score, bool) or not isinstance(material_score, (int, float)):
            errors.append(
                f"game {index}: material_adjudication must be numeric, not boolean"
            )
        elif material_score not in (-1.0, 0.0, 1.0):
            errors.append(f"game {index}: material_adjudication must be -1, 0, or 1")
    else:
        terminal_score = record.get("terminal_score")
        if terminal_score is None:
            errors.append(f"game {index}: terminal game requires terminal_score")
        elif isinstance(terminal_score, bool) or not isinstance(terminal_score, (int, float)):
            errors.append(f"game {index}: terminal_score must be numeric, not boolean")
        elif terminal_score not in (-1.0, 0.0, 1.0):
            errors.append(f"game {index}: terminal_score must be -1, 0, or 1")
        if record.get("material_adjudication") is not None:
            errors.append(f"game {index}: terminal game must not have material_adjudication")
        if record.get("material_delta") is not None:
            errors.append(f"game {index}: terminal game must not have material_delta")
    errors.extend(_replay_game_record(record, index, perspective=perspective))
    return errors


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_values_match(expected: Any, actual: Any) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            _json_values_match(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(
            _json_values_match(left, right) for left, right in zip(expected, actual, strict=True)
        )
    return bool(expected == actual)


def _manifest_errors(manifest: dict[str, Any], *, paired: bool) -> list[str]:
    errors: list[str] = []
    required_fields = (
        "checkpoint",
        "config",
        "source",
        "seed",
        "argv",
        "environment",
        "artifacts",
    )
    for field in required_fields:
        if field not in manifest:
            errors.append(f"manifest requires {field}")

    for field in ("checkpoint", "config"):
        if field in manifest and not isinstance(manifest[field], dict):
            errors.append(f"manifest {field} must be an object")
    for field in ("checkpoint", "config", "opponent_checkpoint", "opening_suite"):
        provenance = manifest.get(field)
        if isinstance(provenance, dict) and (
            not isinstance(provenance.get("path"), str)
            or not provenance.get("path")
            or not isinstance(provenance.get("sha256"), str)
            or not provenance.get("sha256")
        ):
            errors.append(f"manifest {field} requires string path and sha256")
    source = manifest.get("source")
    if source is not None:
        if not isinstance(source, dict):
            errors.append("manifest source must be an object")
        elif (
            not isinstance(source.get("git_commit"), str)
            or not source.get("git_commit", "").strip()
            or type(source.get("dirty")) is not bool
        ):
            errors.append("manifest source requires git_commit and dirty")
    seed = manifest.get("seed")
    if "seed" in manifest and (not isinstance(seed, int) or isinstance(seed, bool)):
        errors.append("manifest seed must be an integer")
    argv = manifest.get("argv")
    if "argv" in manifest and (
        not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv)
    ):
        errors.append("manifest argv must be a non-empty list of strings")
    environment = manifest.get("environment")
    if environment is not None:
        if not isinstance(environment, dict):
            errors.append("manifest environment must be an object")
        elif any(
            not isinstance(environment.get(field), str) or not environment.get(field)
            for field in ("python", "platform", "pytorch", "device")
        ):
            errors.append("manifest environment is incomplete")

    artifacts = manifest.get("artifacts")
    required_artifacts = {"games.jsonl", "summary.json"}
    if not isinstance(artifacts, dict) or not required_artifacts.issubset(artifacts):
        errors.append("manifest artifacts must include games.jsonl and summary.json")
    elif any(
        not isinstance(artifacts[name], str) or not artifacts[name]
        for name in required_artifacts
    ):
        errors.append("manifest artifact hashes must be non-empty strings")

    if paired:
        for field in ("opponent_checkpoint", "opening_suite", "seeds", "paired_protocol"):
            if field not in manifest:
                errors.append(f"paired evidence requires {field} provenance")
        for field in ("opponent_checkpoint", "opening_suite"):
            if field in manifest and not isinstance(manifest[field], dict):
                errors.append(f"manifest {field} must be an object")
    return errors


def _git_source() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", ".", ":(exclude)evidence/debug-v1"],
            cwd=ROOT,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "dirty": True}
    if status.returncode not in {0, 1}:
        return {"git_commit": commit, "dirty": True}
    return {"git_commit": commit, "dirty": status.returncode == 1}


def write_evidence_bundle(
    run_dir: str | Path,
    *,
    records: Sequence[dict[str, Any]],
    checkpoint_path: str | Path,
    config_path: str | Path,
    seed: int,
    argv: Sequence[str],
    device: str,
    opponent_checkpoint_path: str | Path | None = None,
    opening_suite_path: str | Path | None = None,
    seeds: Sequence[int] | None = None,
    paired_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    has_paired_records = any(_record_uses_paired_metadata(record) for record in records)
    preflight_errors = [] if records else ["evidence must contain at least one game"]
    preflight_errors.extend(
        [
            error
            for index, record in enumerate(records, start=1)
            for error in _game_record_errors(record, index)
        ]
    )
    if has_paired_records and paired_protocol is None:
        preflight_errors.append("paired records require paired_protocol")
    if paired_protocol is not None:
        if opponent_checkpoint_path is None:
            raise ValueError("paired evidence requires opponent_checkpoint_path")
        if opening_suite_path is None:
            raise ValueError("paired_protocol requires opening_suite_path")
        if seeds is None:
            raise ValueError("paired_protocol requires evaluation seeds")
        preflight_errors.extend(paired_record_errors(records))
        preflight_errors.extend(
            _canonical_paired_schedule_errors(
                records,
                opening_suite_path=opening_suite_path,
                seeds=list(seeds),
                protocol=paired_protocol,
            )
        )
    if preflight_errors:
        raise ValueError("invalid evidence: " + "; ".join(preflight_errors))

    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    games_path = destination / "games.jsonl"
    summary_path = destination / "summary.json"
    games_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = summarize_game_records(records)
    if paired_protocol is not None:
        _add_paired_protocol_summary(summary, records, opening_suite_path, paired_protocol)
    _write_json(summary_path, summary)

    checkpoint = Path(checkpoint_path)
    config = Path(config_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "config": {"path": str(config), "sha256": sha256_file(config)},
        "source": _git_source(),
        "seed": int(seed),
        "argv": list(argv),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "device": device,
        },
        "artifacts": {
            "games.jsonl": sha256_file(games_path),
            "summary.json": sha256_file(summary_path),
        },
    }
    if opponent_checkpoint_path is not None:
        opponent_checkpoint = Path(opponent_checkpoint_path)
        manifest["opponent_checkpoint"] = {
            "path": str(opponent_checkpoint),
            "sha256": sha256_file(opponent_checkpoint),
        }
    if opening_suite_path is not None:
        opening_suite = Path(opening_suite_path)
        manifest["opening_suite"] = {
            "path": str(opening_suite),
            "sha256": sha256_file(opening_suite),
        }
    if seeds is not None:
        manifest["seeds"] = [int(value) for value in seeds]
    if paired_protocol is not None:
        manifest["paired_protocol"] = paired_protocol
    _write_json(destination / "manifest.json", manifest)
    return summary


def validate_evidence_bundle(run_dir: str | Path) -> dict[str, Any]:
    source = Path(run_dir)
    errors: list[str] = []
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (source / "games.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"cannot read evidence bundle: {exc}"]}

    if not isinstance(manifest, dict):
        return {"valid": False, "errors": ["manifest must be an object"]}
    if not isinstance(summary, dict):
        return {"valid": False, "errors": ["summary must be an object"]}
    if any(not isinstance(record, dict) for record in records):
        return {"valid": False, "errors": ["every game record must be an object"]}

    if type(manifest.get("schema_version")) is not int or manifest.get(
        "schema_version"
    ) != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if not records:
        errors.append("evidence must contain at least one game")
    for index, record in enumerate(records, start=1):
        errors.extend(_game_record_errors(record, index))
    errors.extend(paired_record_errors(records))
    try:
        reconstructed = summarize_game_records(records)
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append(f"cannot reconstruct evidence summary: {exc}")
        reconstructed = None
    has_paired_records = any(_record_uses_paired_metadata(record) for record in records)
    paired_manifest_fields = ("opponent_checkpoint", "opening_suite", "seeds", "paired_protocol")
    paired_manifest = any(field in manifest for field in paired_manifest_fields)
    errors.extend(_manifest_errors(manifest, paired=has_paired_records or paired_manifest))
    paired_protocol = manifest.get("paired_protocol")
    if has_paired_records and paired_protocol is None:
        errors.append("paired records require paired_protocol")
    if paired_protocol is not None:
        manifest_seeds = manifest.get("seeds", [])
        opening_provenance = manifest.get("opening_suite", {})
        opening_path_value = (
            opening_provenance.get("path") if isinstance(opening_provenance, dict) else None
        )
        if isinstance(opening_path_value, str) and opening_path_value:
            opening_path = Path(opening_path_value)
            if not opening_path.is_absolute():
                opening_path = ROOT / opening_path
            errors.extend(
                _canonical_paired_schedule_errors(
                    records,
                    opening_suite_path=opening_path,
                    seeds=manifest_seeds,
                    protocol=paired_protocol,
                )
            )
            if reconstructed is not None:
                try:
                    _add_paired_protocol_summary(
                        reconstructed,
                        records,
                        opening_path,
                        paired_protocol,
                    )
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    errors.append(f"cannot reconstruct paired protocol summary: {exc}")
    if reconstructed is not None and not _json_values_match(reconstructed, summary):
        errors.append("summary does not match games.jsonl reconstruction")
    artifacts = manifest.get("artifacts")
    for name, expected_hash in artifacts.items() if isinstance(artifacts, dict) else ():
        artifact = source / name
        if not artifact.is_file():
            errors.append(f"missing artifact: {name}")
        elif sha256_file(artifact) != expected_hash:
            errors.append(f"artifact hash mismatch: {name}")
    for label in ("checkpoint", "config", "opponent_checkpoint", "opening_suite"):
        if label not in manifest:
            continue
        provenance = manifest.get(label, {})
        if not isinstance(provenance, dict):
            continue
        path_value = provenance.get("path")
        if not isinstance(path_value, str) or not path_value:
            continue
        path = Path(path_value)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            errors.append(f"missing {label} provenance")
        elif sha256_file(path) != provenance.get("sha256"):
            errors.append(f"{label} hash mismatch")
    return {"valid": not errors, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cc-lite evaluation evidence.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="Validate an evidence run directory.")
    validate_parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = validate_evidence_bundle(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
