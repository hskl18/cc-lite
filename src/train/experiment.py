from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from model.checkpoint import save_checkpoint
from train.config import choose_device, load_config, set_seed
from train.loop import train_samples
from train.self_play import build_model
from train.supervised import load_supervised_records

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "dirty": True}
    return {"git_commit": commit, "dirty": bool(status)}


def validate_experiment_bundle(output_dir: str | Path) -> dict[str, object]:
    root = Path(output_dir)
    errors: list[str] = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"cannot read experiment bundle: {exc}"]}
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    for label in ("data", "config", "checkpoint"):
        record = manifest.get(label, {})
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            errors.append(f"missing {label} artifact")
        elif sha256_file(path) != record.get("sha256"):
            errors.append(f"{label} hash mismatch")
    metrics_record = manifest.get("metrics", {})
    metrics_path = root / "metrics.json"
    if metrics_path.is_file() and sha256_file(metrics_path) != metrics_record.get("sha256"):
        errors.append("metrics hash mismatch")
    if int(metrics.get("samples", 0)) <= 0:
        errors.append("experiment must contain at least one sample")
    max_records = int(manifest.get("limits", {}).get("max_records", 0))
    if max_records <= 0 or int(metrics.get("samples", 0)) > max_records:
        errors.append("sample count exceeds declared max_records")
    return {"valid": not errors, "errors": errors}


def run_experiment(
    *,
    config_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    max_records: int | None = None,
    argv: list[str] | None = None,
) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    data_path = Path(data_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment bundle: {destination}")
    config = load_config(config_path)
    configured_limit = int(config.get("bootstrap", {}).get("max_records", 0))
    limit = configured_limit if max_records is None else int(max_records)
    if limit <= 0:
        raise ValueError("A positive bootstrap.max_records or --max-records is required")
    samples = load_supervised_records(data_path)
    if len(samples) > limit:
        samples = samples[:limit]
    seed = int(config.get("seed", 1))
    set_seed(seed)
    device = choose_device(str(config.get("device", "auto")))
    model = build_model(config)
    train_cfg = config.get("training", {})
    stats = train_samples(
        model,
        samples,
        device=device,
        batch_size=int(train_cfg.get("batch_size", 16)),
        epochs=int(train_cfg.get("epochs", 1)),
        lr=float(train_cfg.get("lr", 1e-3)),
        precision=str(train_cfg.get("precision", "fp32")),
    )
    metrics = {"samples": len(samples), **stats.__dict__}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        checkpoint = staging / "checkpoint.pt"
        save_checkpoint(
            checkpoint,
            model,
            step=len(samples),
            metadata={
                "config_sha256": sha256_file(config_path),
                "data_sha256": sha256_file(data_path),
                "seed": seed,
                "limits": {"max_records": limit},
                "metrics": metrics,
            },
        )
        _write_json(staging / "metrics.json", metrics)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "bounded_supervised_bootstrap",
            "data": {"path": str(data_path), "sha256": sha256_file(data_path)},
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "checkpoint": {"path": "checkpoint.pt", "sha256": sha256_file(checkpoint)},
            "metrics": {"path": "metrics.json", "sha256": sha256_file(staging / "metrics.json")},
            "limits": {"max_records": limit},
            "seed": seed,
            "argv": argv or [],
            "engine": None,
            "source": _source_state(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pytorch": torch.__version__,
                "device": str(device),
            },
        }
        _write_json(staging / "manifest.json", manifest)
        validation = validate_experiment_bundle(staging)
        if not validation["valid"]:
            raise RuntimeError(f"experiment evidence failed validation: {validation['errors']}")
        shutil.move(str(staging), str(destination))
    return {"output_dir": str(destination), "metrics": metrics, "validation": validation}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded supervised bootstrap experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Train and write a validated evidence bundle.")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--data", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--max-records", type=int)
    validate_parser = subparsers.add_parser("validate", help="Validate an existing bundle.")
    validate_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_experiment(
            config_path=args.config,
            data_path=args.data,
            output_dir=args.output_dir,
            max_records=args.max_records,
            argv=["python", "-m", "train.experiment", *sys.argv[1:]],
        )
    else:
        result = validate_experiment_bundle(args.output_dir)
        if not result["valid"]:
            print(json.dumps(result, indent=2, sort_keys=True))
            raise SystemExit(1)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
