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
    return {
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
        else:
            if record.get("terminal_score") is None:
                errors.append(f"game {index}: terminal game requires terminal_score")
            if record.get("material_adjudication") is not None:
                errors.append(f"game {index}: terminal game must not have material_adjudication")
    reconstructed = summarize_game_records(records)
    if reconstructed != summary:
        errors.append("summary does not match games.jsonl reconstruction")
    for name, expected_hash in manifest.get("artifacts", {}).items():
        artifact = source / name
        if not artifact.is_file():
            errors.append(f"missing artifact: {name}")
        elif sha256_file(artifact) != expected_hash:
            errors.append(f"artifact hash mismatch: {name}")
    for label in ("checkpoint", "config"):
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
