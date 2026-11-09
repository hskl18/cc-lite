from __future__ import annotations

import queue
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass

from xiangqi.board import BLACK, RED, Move

ENGINE_PIECE_TRANSLATION = str.maketrans({"H": "N", "h": "n", "E": "B", "e": "b"})
# Pikafish numbers ranks 0-9 from Red's side; Fairy-Stockfish numbers them 1-10
# and plays chess until its xiangqi variant is selected.
ENGINE_FLAVORS = {"pikafish": 0, "fairy": 1}
_ENGINE_MOVE = re.compile(r"([a-i])(\d{1,2})([a-i])(\d{1,2})")


@dataclass(frozen=True)
class EngineResult:
    move: Move
    score_cp: float | None
    raw_output: str


def command_argv(command: str) -> list[str]:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("Engine command must not be empty")
    return argv


def score_to_value(score_cp: float | None, scale: float = 1000.0) -> float:
    if score_cp is None:
        return 0.0
    return max(-1.0, min(1.0, score_cp / scale))


def parse_bestmove_output(output: str, rank_base: int = 0) -> EngineResult:
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
    return EngineResult(
        move=Move.from_uci(engine_move_to_cc(best, rank_base)), score_cp=score_cp, raw_output=output
    )


def engine_move_to_cc(text: str, rank_base: int = 0) -> str:
    match = _ENGINE_MOVE.fullmatch(text)
    if match is None:
        raise ValueError(f"Expected an engine move such as h2e2, got {text!r}")
    from_file, from_rank, to_file, to_rank = match.groups()
    ranks = [int(from_rank) - rank_base, int(to_rank) - rank_base]
    if not all(0 <= rank <= 9 for rank in ranks):
        raise ValueError(f"Engine move {text!r} is off the board for rank base {rank_base}")
    return f"{from_file}{9 - ranks[0]}{to_file}{9 - ranks[1]}"


class UcciEngine:
    """Minimal line-oriented UCCI/UCI engine adapter.

    This is intentionally small and dependency-free. It is meant for optional
    Pikafish-style benchmarking and label caching, not for high-throughput
    self-play orchestration.
    """

    def __init__(
        self,
        command: str,
        timeout: float = 10.0,
        init_command: str = "uci",
        flavor: str = "pikafish",
    ):
        if flavor not in ENGINE_FLAVORS:
            raise ValueError(f"Unknown engine flavor {flavor!r}; expected one of {sorted(ENGINE_FLAVORS)}")
        self.rank_base = ENGINE_FLAVORS[flavor]
        self.proc = subprocess.Popen(
            command_argv(command),
            shell=False,
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
        if flavor == "fairy":
            self.send("setoption name UCI_Variant value xiangqi")
            self.send("isready")
            self.read_until(("readyok",), timeout)

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
        engine_fen = to_engine_fen(fen)
        self.send(f"position fen {engine_fen}")
        self.send(f"go depth {depth}")
        return parse_bestmove_output(self.read_until(("bestmove",)), self.rank_base)

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self.send("quit")
        finally:
            self.proc.terminate()


def to_engine_fen(fen: str) -> str:
    """Convert cc-lite FEN to Pikafish/Fairy-Stockfish-style Xiangqi FEN."""

    placement, turn = fen.rsplit(" ", 1)
    translated = placement.translate(ENGINE_PIECE_TRANSLATION)
    engine_turn = "w" if turn == "r" or turn == RED else "b"
    if turn == BLACK:
        engine_turn = "b"
    return f"{translated} {engine_turn}"
