from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from data.ingest import IngestionError, ingest_records
from train.supervised import load_supervised_records


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _provenance() -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_name": "Synthetic ingestion fixture",
        "source": "Locally generated test moves",
        "license": "CC0-1.0",
        "license_notes": "Synthetic records created for cc-lite tests.",
        "attribution": "cc-lite contributors",
        "redistribution_permitted": True,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_ingestion_validates_deduplicates_and_emits_training_samples(tmp_path: Path):
    input_path = tmp_path / "games.jsonl"
    provenance_path = tmp_path / "provenance.json"
    output_dir = tmp_path / "dataset"
    records = [
        {
            "record_id": "game-b",
            "notation": "ucci",
            "moves": "a6a5 a3a4",
            "result": "1/2-1/2",
        },
        {
            "record_id": "game-a",
            "notation": "iccs",
            "moves": ["A6A5", "A3A4"],
            "result": "1/2-1/2",
        },
        {
            "record_id": "illegal",
            "moves": ["a9b9"],
            "result": "1-0",
        },
    ]
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    _write_json(provenance_path, _provenance())

    manifest = ingest_records(
        input_path,
        provenance_path,
        output_dir,
        validation_fraction=0.0,
        split_seed="fixture",
    )

    assert manifest["deduplication"] == {
        "identity": "sha256 of canonical initial_fen and normalized moves",
        "accepted_games": 1,
        "rejected_records": 2,
    }
    games = _read_jsonl(output_dir / "games.jsonl")
    assert games[0]["record_id"] == "game-a"
    assert games[0]["moves"] == ["a6a5", "a3a4"]
    assert games[0]["split"] == "train"
    rejections = _read_jsonl(output_dir / "rejections.jsonl")
    assert [rejection["code"] for rejection in rejections] == [
        "duplicate_game",
        "illegal_move",
    ]
    assert rejections[1]["details"]["move"] == "a9b9"
    samples = load_supervised_records(output_dir / "train.jsonl")
    assert len(samples) == 2
    assert [sample.value for sample in samples] == [0.0, 0.0]
    assert len(samples[0].policy) == 1


def test_split_and_artifact_fingerprints_are_repeatable(tmp_path: Path):
    input_path = tmp_path / "games.jsonl"
    provenance_path = tmp_path / "provenance.json"
    records = [
        {"record_id": "one", "moves": ["a6a5"], "result": "1-0"},
        {"record_id": "two", "moves": ["c6c5"], "result": "0-1"},
    ]
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    _write_json(provenance_path, _provenance())

    first = ingest_records(
        input_path,
        provenance_path,
        tmp_path / "first",
        validation_fraction=0.5,
        split_seed="stable",
    )
    second = ingest_records(
        input_path,
        provenance_path,
        tmp_path / "second",
        validation_fraction=0.5,
        split_seed="stable",
    )

    assert first == second
    for name in ("games.jsonl", "train.jsonl", "validation.jsonl", "rejections.jsonl"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()


def test_conflicting_duplicate_results_are_all_rejected(tmp_path: Path):
    input_path = tmp_path / "games.jsonl"
    provenance_path = tmp_path / "provenance.json"
    input_path.write_text(
        json.dumps({"record_id": "a", "moves": ["a6a5"], "result": "1-0"})
        + "\n"
        + json.dumps({"record_id": "b", "moves": ["a6a5"], "result": "0-1"})
        + "\n",
        encoding="utf-8",
    )
    _write_json(provenance_path, _provenance())

    manifest = ingest_records(input_path, provenance_path, tmp_path / "output")

    assert manifest["deduplication"]["accepted_games"] == 0
    assert [item["code"] for item in _read_jsonl(tmp_path / "output" / "rejections.jsonl")] == [
        "conflicting_duplicate",
        "conflicting_duplicate",
    ]


def test_record_id_conflict_rejects_every_distinct_game(tmp_path: Path):
    input_path = tmp_path / "games.jsonl"
    provenance_path = tmp_path / "provenance.json"
    input_path.write_text(
        json.dumps({"record_id": "same-id", "moves": ["a6a5"], "result": "1-0"})
        + "\n"
        + json.dumps({"record_id": "same-id", "moves": ["c6c5"], "result": "0-1"})
        + "\n",
        encoding="utf-8",
    )
    _write_json(provenance_path, _provenance())

    manifest = ingest_records(input_path, provenance_path, tmp_path / "output")

    assert manifest["deduplication"]["accepted_games"] == 0
    assert [item["code"] for item in _read_jsonl(tmp_path / "output" / "rejections.jsonl")] == [
        "record_id_conflict",
        "record_id_conflict",
    ]


def test_required_provenance_is_enforced_before_outputs_are_created(tmp_path: Path):
    input_path = tmp_path / "games.jsonl"
    provenance_path = tmp_path / "provenance.json"
    output_dir = tmp_path / "output"
    input_path.write_text("{}\n", encoding="utf-8")
    incomplete = _provenance()
    del incomplete["license_notes"]
    _write_json(provenance_path, incomplete)

    with pytest.raises(IngestionError, match="license_notes"):
        ingest_records(input_path, provenance_path, output_dir)

    assert not output_dir.exists()


def test_rejection_report_identifies_json_and_terminal_result_failures(tmp_path: Path):
    input_path = tmp_path / "games.jsonl"
    provenance_path = tmp_path / "provenance.json"
    terminal_record = {
        "record_id": "wrong-result",
        "initial_fen": "4k4/4R4/9/9/9/9/9/9/9/4K4 r",
        "moves": ["e1e0"],
        "result": "0-1",
    }
    input_path.write_text("{bad json\n" + json.dumps(terminal_record) + "\n", encoding="utf-8")
    _write_json(provenance_path, _provenance())

    ingest_records(input_path, provenance_path, tmp_path / "output")

    rejections = _read_jsonl(tmp_path / "output" / "rejections.jsonl")
    assert [item["code"] for item in rejections] == [
        "invalid_json",
        "terminal_result_conflict",
    ]
    assert rejections[1]["details"] == {
        "declared_result": "0-1",
        "terminal_result": "1-0",
    }


def test_rules_profile_repetition_accepts_a_declared_draw(tmp_path: Path):
    input_path = tmp_path / "games.jsonl"
    provenance_path = tmp_path / "provenance.json"
    input_path.write_text(
        json.dumps(
            {
                "record_id": "repetition-draw",
                "initial_fen": "r3k4/9/9/9/9/4P4/9/9/9/R3K4 r",
                "moves": ["a9b9", "a0b0", "b9a9", "b0a0"] * 2,
                "result": "1/2-1/2",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(provenance_path, _provenance())

    manifest = ingest_records(input_path, provenance_path, tmp_path / "output")

    assert manifest["deduplication"]["accepted_games"] == 1
    assert _read_jsonl(tmp_path / "output" / "rejections.jsonl") == []


def test_existing_artifacts_require_force(tmp_path: Path):
    input_path = tmp_path / "games.jsonl"
    provenance_path = tmp_path / "provenance.json"
    output_dir = tmp_path / "output"
    input_path.write_text(
        json.dumps({"record_id": "one", "moves": ["a6a5"], "result": "1-0"}) + "\n",
        encoding="utf-8",
    )
    _write_json(provenance_path, _provenance())
    ingest_records(input_path, provenance_path, output_dir)

    with pytest.raises(IngestionError, match="--force"):
        ingest_records(input_path, provenance_path, output_dir)

    ingest_records(input_path, provenance_path, output_dir, force=True)


def test_write_failure_does_not_publish_a_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    input_path = tmp_path / "games.jsonl"
    provenance_path = tmp_path / "provenance.json"
    output_dir = tmp_path / "output"
    input_path.write_text(
        json.dumps({"record_id": "one", "moves": ["a6a5"], "result": "1-0"})
        + "\n",
        encoding="utf-8",
    )
    _write_json(provenance_path, _provenance())
    original_open = Path.open

    def fail_while_writing_train(path: Path, *args, **kwargs):
        if ".train.jsonl." in path.name:
            raise OSError("injected artifact write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_while_writing_train)

    with pytest.raises(OSError, match="injected artifact write failure"):
        ingest_records(input_path, provenance_path, output_dir)

    assert not output_dir.exists()


def test_force_write_failure_preserves_the_previous_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    input_path = tmp_path / "games.jsonl"
    provenance_path = tmp_path / "provenance.json"
    output_dir = tmp_path / "output"
    input_path.write_text(
        json.dumps({"record_id": "one", "moves": ["a6a5"], "result": "1-0"})
        + "\n",
        encoding="utf-8",
    )
    _write_json(provenance_path, _provenance())
    ingest_records(input_path, provenance_path, output_dir)
    previous = {
        name: (output_dir / name).read_bytes()
        for name in (
            "games.jsonl",
            "train.jsonl",
            "validation.jsonl",
            "rejections.jsonl",
            "manifest.json",
        )
    }
    original_open = Path.open

    def fail_while_writing_train(path: Path, *args, **kwargs):
        if ".train.jsonl." in path.name:
            raise OSError("injected force write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_while_writing_train)

    with pytest.raises(OSError, match="injected force write failure"):
        ingest_records(input_path, provenance_path, output_dir, force=True)

    assert {name: (output_dir / name).read_bytes() for name in previous} == previous


def test_module_cli_runs_end_to_end(tmp_path: Path):
    input_path = tmp_path / "games.jsonl"
    provenance_path = tmp_path / "provenance.json"
    output_dir = tmp_path / "output"
    input_path.write_text(
        json.dumps({"record_id": "cli", "moves": ["a6a5"], "result": "1-0"}) + "\n",
        encoding="utf-8",
    )
    _write_json(provenance_path, _provenance())
    repository = Path(__file__).parents[1]
    environment = {**os.environ, "PYTHONPATH": str(repository / "src")}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "data.ingest",
            "--input",
            str(input_path),
            "--provenance",
            str(provenance_path),
            "--output-dir",
            str(output_dir),
            "--validation-fraction",
            "0",
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["deduplication"]["accepted_games"] == 1
    assert (output_dir / "manifest.json").is_file()
    assert len(load_supervised_records(output_dir / "train.jsonl")) == 1
