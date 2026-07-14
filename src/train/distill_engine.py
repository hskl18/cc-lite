from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from data.replay import ReplaySample, write_jsonl
from eval.engine import UcciEngine, command_argv, score_to_value
from train.experiment import sha256_file
from xiangqi.board import Board
from xiangqi.encoding import MoveCodec


def _engine_provenance(command: str) -> dict[str, object]:
    argv = command_argv(command)
    executable = shutil.which(argv[0])
    if executable is None:
        candidate = Path(argv[0]).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Cannot resolve engine executable: {argv[0]}")
        executable = str(candidate)
    argument_files = []
    for argument in argv[1:]:
        candidate = Path(argument).expanduser()
        if candidate.is_file():
            resolved = candidate.resolve()
            argument_files.append({"path": str(resolved), "sha256": sha256_file(resolved)})
    return {
        "command": argv,
        "executable": {"path": str(Path(executable).resolve()), "sha256": sha256_file(executable)},
        "argument_files": argument_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache optional Pikafish UCI/UCCI labels for distillation.")
    parser.add_argument("--engine", required=True, help="Engine command, e.g. './pikafish'")
    parser.add_argument("--positions", required=True, help="JSONL with {'fen': ...} rows")
    parser.add_argument("--output", default="runs/distill_labels.jsonl")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--init-command", choices=["uci", "ucci"], default="uci")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--manifest", help="Defaults to OUTPUT.manifest.json")
    args = parser.parse_args()

    if args.depth <= 0 or args.threads <= 0 or args.hash_mb <= 0:
        parser.error("--depth, --threads, and --hash-mb must be positive")

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
    engine_provenance = _engine_provenance(args.engine)
    engine = UcciEngine(args.engine, timeout=args.timeout, init_command=args.init_command)
    samples: list[ReplaySample] = []
    try:
        engine.send(f"setoption name Threads value {args.threads}")
        engine.send(f"setoption name Hash value {args.hash_mb}")
        engine.send("isready")
        engine.read_until(("readyok",))
        for row in rows:
            board = Board.from_fen(row["fen"])
            result = engine.best_move(board.to_fen(), args.depth)
            move = result.move
            legal = set(board.legal_moves())
            if move not in legal:
                raise ValueError(f"Engine returned illegal move {move.uci()} for {board.to_fen()}")
            value = score_to_value(result.score_cp)
            samples.append(ReplaySample(fen=board.to_fen(), policy={MoveCodec.encode(move): 1.0}, value=value))
    finally:
        engine.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = write_jsonl(output_path, samples)
    manifest = {
        "schema_version": 1,
        "kind": "engine_distillation_labels",
        "engine": {
            **engine_provenance,
            "protocol": args.init_command,
            "depth": args.depth,
            "threads": args.threads,
            "hash_mb": args.hash_mb,
            "timeout_seconds": args.timeout,
        },
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
