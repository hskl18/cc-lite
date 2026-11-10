from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
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
from xiangqi.board import Board, Move
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
    multipv: int = 1
    policy_temperature: float = 100.0


def soft_policy(lines: Sequence[tuple[Move, float]], temperature_cp: float) -> dict[int, float]:
    """Softmax over the engine's centipawn scores for its top moves, so the policy target ranks them."""

    scores = [score for _, score in lines]
    best = max(scores)
    weights = [math.exp((score - best) / temperature_cp) for score in scores]
    total = sum(weights)
    return {MoveCodec.encode(move): weight / total for (move, _), weight in zip(lines, weights, strict=True)}


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
        if settings.multipv > 1:
            engine.send(f"setoption name MultiPV value {settings.multipv}")
        engine.send("isready")
        engine.read_until(("readyok",))
        for fen in fens:
            board = Board.from_fen(fen)
            engine.new_game()
            result = engine.best_move(board.to_fen(), settings.depth)
            legal = set(board.legal_moves())
            if result.move not in legal:
                raise ValueError(f"Engine returned illegal move {result.move.uci()} for {board.to_fen()}")
            value = score_to_value(result.score_cp, settings.value_scale, settings.value_mapping)
            # The raw score is kept, so a later run can remap values without searching again.
            row: dict[str, object] = {
                "fen": board.to_fen(),
                "policy": {str(MoveCodec.encode(result.move)): 1.0},
                "value": value,
                "score_cp": result.score_cp,
            }
            # A line the rules profile rejects makes the report unusable as a soft target, like any
            # other inconsistency, so the label keeps the best move alone instead of stopping the run.
            if settings.multipv > 1 and result.lines and all(move in legal for move, _ in result.lines):
                policy = soft_policy(result.lines, settings.policy_temperature)
                row["policy"] = {str(index): probability for index, probability in policy.items()}
                # Each line's raw score is kept too, so another temperature needs no new search.
                row["multipv"] = [[move.uci(), score] for move, score in result.lines]
            rows.append(row)
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
    parser.add_argument(
        "--multipv", type=int, default=1, help="Engine lines per position; above 1 the policy target is soft"
    )
    parser.add_argument(
        "--policy-temperature", type=float, default=100.0, help="Centipawns per unit of the soft policy's softmax"
    )
    parser.add_argument("--manifest", help="Defaults to OUTPUT.manifest.json")
    args = parser.parse_args()

    scales = (args.value_scale, args.policy_temperature)
    if min(args.depth, args.threads, args.hash_mb, args.workers, args.multipv, *scales) <= 0 or not all(
        math.isfinite(scale) for scale in scales
    ):
        parser.error(
            "--depth, --threads, --hash-mb, --workers, --value-scale, --multipv, and --policy-temperature "
            "must be positive and finite"
        )

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
        args.multipv,
        args.policy_temperature,
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
        "policy_target": {"multipv": args.multipv, "temperature_cp": args.policy_temperature},
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
