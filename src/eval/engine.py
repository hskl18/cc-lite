from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass

from xiangqi.board import Move


@dataclass(frozen=True)
class EngineResult:
    move: Move
    score_cp: float | None
    raw_output: str


def score_to_value(score_cp: float | None, scale: float = 1000.0) -> float:
    if score_cp is None:
        return 0.0
    return max(-1.0, min(1.0, score_cp / scale))


def parse_bestmove_output(output: str) -> EngineResult:
    best = None
    score_cp = None
    for line in output.splitlines():
        parts = line.split()
        if not parts:
            continue
        if "score" in parts:
            idx = parts.index("score")
            if idx + 2 < len(parts):
                kind = parts[idx + 1]
                try:
                    raw_score = float(parts[idx + 2])
                except ValueError:
                    raw_score = None
                if raw_score is not None:
                    if kind == "cp":
                        score_cp = raw_score
                    elif kind == "mate":
                        score_cp = 100000.0 if raw_score > 0 else -100000.0
        if parts[0] == "bestmove" and len(parts) >= 2:
            best = parts[1]
    if best is None:
        raise RuntimeError(f"No bestmove in engine output: {output}")
    return EngineResult(move=Move.from_uci(best), score_cp=score_cp, raw_output=output)


class UcciEngine:
    """Minimal line-oriented UCCI/UCI engine adapter.

    This is intentionally small and dependency-free. It is meant for optional
    Pikafish-style benchmarking and label caching, not for high-throughput
    self-play orchestration.
    """

    def __init__(self, command: str, timeout: float = 10.0, init_command: str = "ucci"):
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
        self._lines: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self.send(init_command)
        ok_prefix = "ucciok" if init_command == "ucci" else "uciok"
        self.read_until((ok_prefix,), timeout)

    def _read_stdout(self) -> None:
        if self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            self._lines.put(line.strip())

    def send(self, text: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("Engine stdin is closed")
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefixes: tuple[str, ...], timeout: float | None = None) -> str:
        deadline = time.time() + (self.timeout if timeout is None else timeout)
        lines: list[str] = []
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            try:
                line = self._lines.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            lines.append(line)
            if line.startswith(prefixes):
                return "\n".join(lines)
        raise TimeoutError(f"Engine did not answer with {prefixes}. Last lines: {lines[-8:]}")

    def best_move(self, fen: str, depth: int) -> EngineResult:
        placement, turn = fen.rsplit(" ", 1)
        engine_fen = placement + (" w" if turn == "r" else " b")
        self.send(f"position fen {engine_fen}")
        self.send(f"go depth {depth}")
        return parse_bestmove_output(self.read_until(("bestmove",)))

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self.send("quit")
        finally:
            self.proc.terminate()

