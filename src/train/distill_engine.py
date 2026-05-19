from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.replay import ReplaySample, write_jsonl
from eval.engine import UcciEngine, score_to_value
from xiangqi.board import Board
from xiangqi.encoding import MoveCodec


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache optional Pikafish UCI/UCCI labels for distillation.")
    parser.add_argument("--engine", required=True, help="Engine command, e.g. './pikafish'")
    parser.add_argument("--positions", required=True, help="JSONL with {'fen': ...} rows")
    parser.add_argument("--output", default="runs/distill_labels.jsonl")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--init-command", choices=["uci", "ucci"], default="uci")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.positions).read_text(encoding="utf-8").splitlines() if line.strip()]
    engine = UcciEngine(args.engine, timeout=args.timeout, init_command=args.init_command)
    samples: list[ReplaySample] = []
    try:
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
    count = write_jsonl(args.output, samples)
    print(json.dumps({"wrote": count, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
