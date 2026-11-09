from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

from eval.engine import (
    ENGINE_FLAVORS,
    VALUE_MAPPINGS,
    UcciEngine,
    engine_provenance,
    score_to_value,
)
from train.experiment import sha256_file
from xiangqi.board import Board
from xiangqi.encoding import MoveCodec


class _LabelSettings(NamedTuple):
    engine: str
    timeout: float
    init_command: str
    flavor: str
    threads: int
    hash_mb: int
    depth: int
    value_mapping: str
    value_scale: float
    eval_file: str | None


def _label_chunk(settings: _LabelSettings, fens: list[str]) -> list[dict[str, object]]:
    engine = UcciEngine(
        settings.engine,
        timeout=settings.timeout,
        init_command=settings.init_command,
        flavor=settings.flavor,
        eval_file=settings.eval_file,
    )
    rows: list[dict[str, object]] = []
    try:
        engine.send(f"setoption name Threads value {settings.threads}")
        engine.send(f"setoption name Hash value {settings.hash_mb}")
        engine.send("isready")
        engine.read_until(("readyok",))
        for fen in fens:
            board = Board.from_fen(fen)
            engine.new_game()
            result = engine.best_move(board.to_fen(), settings.depth)
            move = result.move
            if move not in set(board.legal_moves()):
                raise ValueError(f"Engine returned illegal move {move.uci()} for {board.to_fen()}")
            value = score_to_value(result.score_cp, settings.value_scale, settings.value_mapping)
            # The raw score is kept, so a later run can remap values without searching again.
            rows.append(
                {
                    "fen": board.to_fen(),
                    "policy": {str(MoveCodec.encode(move)): 1.0},
                    "value": value,
                    "score_cp": result.score_cp,
                }
            )
    finally:
        engine.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache optional Pikafish UCI/UCCI labels for distillation.")
    parser.add_argument("--engine", required=True, help="Engine command, e.g. './pikafish'")
    parser.add_argument("--positions", required=True, help="JSONL with {'fen': ...} rows")
    parser.add_argument("--output", default="runs/distill_labels.jsonl")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--init-command", choices=["uci", "ucci"], default="uci")
    parser.add_argument("--engine-flavor", choices=sorted(ENGINE_FLAVORS), default="pikafish")
    parser.add_argument("--workers", type=int, default=1, help="Engine processes labeling in parallel")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--engine-eval-file", help="Network file to pin, for engines such as Pikafish")
    parser.add_argument("--value-mapping", choices=VALUE_MAPPINGS, default="linear")
    parser.add_argument("--value-scale", type=float, default=1000.0, help="Centipawns per unit before mapping")
    parser.add_argument("--manifest", help="Defaults to OUTPUT.manifest.json")
    args = parser.parse_args()

    if min(args.depth, args.threads, args.hash_mb, args.workers, args.value_scale) <= 0:
        parser.error("--depth, --threads, --hash-mb, --workers, and --value-scale must be positive")

    positions_path = Path(args.positions).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else output_path.with_name(output_path.name + ".manifest.json")
    )
    rows = [
        json.loads(line)
        for line in positions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    engine_record = engine_provenance(args.engine, args.engine_eval_file)
    fens = [row["fen"] for row in rows]
    settings = _LabelSettings(
        args.engine,
        args.timeout,
        args.init_command,
        args.engine_flavor,
        args.threads,
        args.hash_mb,
        args.depth,
        args.value_mapping,
        args.value_scale,
        args.engine_eval_file,
    )
    # Each position starts from a cleared engine and chunks keep their order,
    # so the labels do not depend on the number of workers.
    chunk_size = max(1, -(-len(fens) // (args.workers * 8)))
    chunks = [fens[start : start + chunk_size] for start in range(0, len(fens), chunk_size)]
    labels: list[dict[str, object]] = []
    if args.workers == 1:
        labels = _label_chunk(settings, fens)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for labeled in pool.map(_label_chunk, [settings] * len(chunks), chunks):
                labels.extend(labeled)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in labels), encoding="utf-8")
    count = len(labels)
    manifest = {
        "schema_version": 1,
        "kind": "engine_distillation_labels",
        "engine": {
            **engine_record,
            "protocol": args.init_command,
            "flavor": args.engine_flavor,
            "workers": args.workers,
            "depth": args.depth,
            "threads": args.threads,
            "hash_mb": args.hash_mb,
            "timeout_seconds": args.timeout,
        },
        "value_mapping": {"kind": args.value_mapping, "scale_cp": args.value_scale},
        "positions": {"path": str(positions_path), "sha256": sha256_file(positions_path)},
        "labels": {"path": str(output_path), "sha256": sha256_file(output_path), "records": count},
        "argv": ["python", "-m", "train.distill_engine", *sys.argv[1:]],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"wrote": count, "output": str(output_path), "manifest": str(manifest_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
