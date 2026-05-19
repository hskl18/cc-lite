from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from data.replay import ReplaySample, write_jsonl
from xiangqi.board import Board, Move
from xiangqi.encoding import MoveCodec


def _read_until(proc: subprocess.Popen[str], prefixes: tuple[str, ...], timeout: float) -> str:
    deadline = time.time() + timeout
    lines: list[str] = []
    while time.time() < deadline:
        line = proc.stdout.readline() if proc.stdout is not None else ""
        if not line:
            continue
        line = line.strip()
        lines.append(line)
        if line.startswith(prefixes):
            return "\n".join(lines)
    raise TimeoutError(f"Engine did not answer with {prefixes} within {timeout}s. Last lines: {lines[-5:]}")


class UcciEngine:
    def __init__(self, command: str, timeout: float = 10.0):
        self.proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.timeout = timeout
        self.send("ucci")
        _read_until(self.proc, ("ucciok",), timeout)

    def send(self, text: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("Engine stdin is closed")
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def best_move(self, fen: str, depth: int) -> tuple[Move, float | None]:
        placement, turn = fen.rsplit(" ", 1)
        engine_fen = placement + (" w" if turn == "r" else " b")
        self.send(f"position fen {engine_fen}")
        self.send(f"go depth {depth}")
        output = _read_until(self.proc, ("bestmove",), self.timeout)
        best = None
        score = None
        for line in output.splitlines():
            parts = line.split()
            if "score" in parts:
                idx = parts.index("score")
                if idx + 2 < len(parts) and parts[idx + 1] in {"cp", "mate"}:
                    try:
                        score = float(parts[idx + 2])
                    except ValueError:
                        score = None
            if parts and parts[0] == "bestmove" and len(parts) >= 2:
                best = parts[1]
        if best is None:
            raise RuntimeError(f"No bestmove in engine output: {output}")
        return Move.from_uci(best), score

    def close(self) -> None:
        try:
            self.send("quit")
        finally:
            self.proc.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache optional Pikafish/UCCI labels for distillation.")
    parser.add_argument("--engine", required=True, help="Engine command, e.g. './pikafish'")
    parser.add_argument("--positions", required=True, help="JSONL with {'fen': ...} rows")
    parser.add_argument("--output", default="runs/distill_labels.jsonl")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.positions).read_text(encoding="utf-8").splitlines() if line.strip()]
    engine = UcciEngine(args.engine, timeout=args.timeout)
    samples: list[ReplaySample] = []
    try:
        for row in rows:
            board = Board.from_fen(row["fen"])
            move, score = engine.best_move(board.to_fen(), args.depth)
            legal = set(board.legal_moves())
            if move not in legal:
                raise ValueError(f"Engine returned illegal move {move.uci()} for {board.to_fen()}")
            value = 0.0 if score is None else max(-1.0, min(1.0, score / 1000.0))
            samples.append(ReplaySample(fen=board.to_fen(), policy={MoveCodec.encode(move): 1.0}, value=value))
    finally:
        engine.close()
    count = write_jsonl(args.output, samples)
    print(json.dumps({"wrote": count, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()

