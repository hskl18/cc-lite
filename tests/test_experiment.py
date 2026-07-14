from __future__ import annotations

import json

from train.experiment import run_experiment, validate_experiment_bundle
from xiangqi.board import Board


def test_bounded_supervised_experiment_writes_valid_hashed_bundle(tmp_path) -> None:
    board = Board.start()
    move = board.legal_moves()[0]
    data = tmp_path / "records.jsonl"
    data.write_text(
        "".join(
            json.dumps({"fen": board.to_fen(), "move": move.uci(), "value": 0.0}) + "\n"
            for _ in range(3)
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        """seed: 9
device: cpu
model:
  preset: tiny
  channels: 8
  blocks: 1
training:
  batch_size: 1
  epochs: 1
  lr: 0.001
  precision: fp32
bootstrap:
  max_records: 2
""",
        encoding="utf-8",
    )
    output = tmp_path / "experiment"

    result = run_experiment(config_path=config, data_path=data, output_dir=output)

    assert result["validation"] == {"valid": True, "errors": []}
    assert result["metrics"]["samples"] == 2
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment"] == "bounded_supervised_bootstrap"
    assert manifest["limits"] == {"max_records": 2}
    assert manifest["engine"] is None
    assert len(manifest["data"]["sha256"]) == 64
    assert len(manifest["checkpoint"]["sha256"]) == 64


def test_experiment_validator_rejects_tampered_metrics(tmp_path) -> None:
    output = tmp_path / "experiment"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data": {},
                "config": {},
                "checkpoint": {},
                "metrics": {"sha256": "not-the-real-hash"},
                "limits": {"max_records": 1},
            }
        ),
        encoding="utf-8",
    )
    (output / "metrics.json").write_text('{"samples": 1}\n', encoding="utf-8")

    result = validate_experiment_bundle(output)

    assert result["valid"] is False
    assert "metrics hash mismatch" in result["errors"]
