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

from eval.openings import load_opening_suite
from eval.paired import paired_record_errors, sequential_stop_decision, summarize_paired_records
from xiangqi.board import Board, Move

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]


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
    summary["opening_suite"] = {
        "suite_id": suite.suite_id,
        "version": suite.version,
        "sha256": suite.sha256,
        "positions": len(suite.positions),
    }


def _replay_game_record(record: dict[str, Any], index: int) -> list[str]:
    if "start_fen" not in record or "moves" not in record:
        return []
    errors: list[str] = []
    try:
        board = Board.from_fen(str(record["start_fen"]))
        for move_text in record["moves"]:
            board.push(Move.from_uci(str(move_text)))
    except (TypeError, ValueError) as exc:
        return [f"game {index}: cannot replay raw moves: {exc}"]
    if record.get("final_fen") is not None and board.to_fen() != record["final_fen"]:
        errors.append(f"game {index}: final_fen does not match raw moves")
    adjudication = board.adjudication()
    if record.get("terminal_reason") == "max_plies":
        if adjudication is not None:
            errors.append(f"game {index}: max_plies position is already terminal")
    elif adjudication is None:
        errors.append(f"game {index}: terminal record does not replay to a terminal position")
    else:
        if record.get("terminal_reason") != adjudication.reason:
            errors.append(f"game {index}: terminal_reason does not match replay")
        perspective = record.get("candidate_color", record.get("model_color"))
        if perspective in {"red", "black"} and record.get(
            "terminal_score"
        ) != adjudication.result_for(perspective):
            errors.append(f"game {index}: terminal_score does not match replay")
    return errors


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        if opening_suite_path is None:
            raise ValueError("paired_protocol requires opening_suite_path")
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

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    for index, record in enumerate(records, start=1):
        if record.get("terminal_reason") == "max_plies":
            if record.get("terminal_score") is not None:
                errors.append(f"game {index}: max_plies must not have terminal_score")
            if record.get("material_adjudication") is None:
                errors.append(f"game {index}: max_plies requires material_adjudication")
            elif record.get("material_adjudication") not in {-1.0, 0.0, 1.0}:
                errors.append(f"game {index}: material_adjudication must be -1, 0, or 1")
        else:
            if record.get("terminal_score") is None:
                errors.append(f"game {index}: terminal game requires terminal_score")
            elif record.get("terminal_score") not in {-1.0, 0.0, 1.0}:
                errors.append(f"game {index}: terminal_score must be -1, 0, or 1")
            if record.get("material_adjudication") is not None:
                errors.append(f"game {index}: terminal game must not have material_adjudication")
        errors.extend(_replay_game_record(record, index))
    errors.extend(paired_record_errors(records))
    reconstructed = summarize_game_records(records)
    paired_protocol = manifest.get("paired_protocol")
    if paired_protocol is not None:
        manifest_seeds = manifest.get("seeds", [])
        if len(manifest_seeds) != len(set(manifest_seeds)):
            errors.append("manifest evaluation seeds must be unique")
        record_seeds = {record.get("seed") for record in records}
        if not record_seeds.issubset(set(manifest_seeds)):
            errors.append("raw game seed is not declared in the manifest")
        opening_provenance = manifest.get("opening_suite", {})
        opening_path = Path(opening_provenance.get("path", ""))
        if not opening_path.is_absolute():
            opening_path = ROOT / opening_path
        try:
            _add_paired_protocol_summary(
                reconstructed,
                records,
                opening_path,
                paired_protocol,
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"cannot reconstruct paired protocol summary: {exc}")
    if reconstructed != summary:
        errors.append("summary does not match games.jsonl reconstruction")
    for name, expected_hash in manifest.get("artifacts", {}).items():
        artifact = source / name
        if not artifact.is_file():
            errors.append(f"missing artifact: {name}")
        elif sha256_file(artifact) != expected_hash:
            errors.append(f"artifact hash mismatch: {name}")
    for label in ("checkpoint", "config", "opponent_checkpoint", "opening_suite"):
        if label not in manifest:
            continue
        provenance = manifest.get(label, {})
        path = Path(provenance.get("path", ""))
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
