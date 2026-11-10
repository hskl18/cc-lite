from __future__ import annotations

import math
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from eval.evidence import sha256_file
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
    # With MultiPV, one (first move, exact score) per line from the final depth, best first;
    # empty for a single line or when that report is incomplete or inconsistent.
    lines: tuple[tuple[Move, float], ...] = ()


def command_argv(command: str) -> list[str]:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("Engine command must not be empty")
    return argv


VALUE_MAPPINGS = ("linear", "tanh")


def score_to_value(score_cp: float | None, scale: float = 1000.0, mapping: str = "linear") -> float:
    """Map a side-to-move centipawn score to [-1, 1]: clipped linear, or tanh, which keeps big leads apart."""

    if score_cp is None:
        return 0.0
    if mapping == "tanh":
        return math.tanh(score_cp / scale)
    if mapping != "linear":
        raise ValueError(f"Unknown value mapping {mapping!r}; expected one of {VALUE_MAPPINGS}")
    return max(-1.0, min(1.0, score_cp / scale))


def engine_provenance(command: str, eval_file: str | None = None) -> dict[str, object]:
    """Resolve and hash the engine executable, any file arguments, and a pinned network file."""

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
    record: dict[str, object] = {
        "command": argv,
        "executable": {"path": str(Path(executable).resolve()), "sha256": sha256_file(executable)},
        "argument_files": argument_files,
    }
    if eval_file is not None:
        network = Path(eval_file).expanduser().resolve()
        record["eval_file"] = {"path": str(network), "sha256": sha256_file(network)}
    return record


def _parse_score(parts: list[str]) -> float | None:
    if "score" not in parts:
        return None
    idx = parts.index("score")
    if idx + 2 >= len(parts):
        return None
    try:
        raw_score = float(parts[idx + 2])
    except ValueError:
        return None
    if parts[idx + 1] == "cp":
        return raw_score
    if parts[idx + 1] == "mate":
        return 100000.0 if raw_score > 0 else -100000.0
    return None


def _after(parts: list[str], token: str) -> str | None:
    if token in parts and parts.index(token) + 1 < len(parts):
        return parts[parts.index(token) + 1]
    return None


def parse_bestmove_output(output: str, rank_base: int = 0) -> EngineResult:
    best = None
    score_cp = None
    line_one_score = None
    several_lines = False
    # (depth, line number, first move, score) for every report with an exact score; a bound score
    # comes from a re-search that a later report supersedes.
    exact: list[tuple[str | None, int, str, float]] = []
    for line in output.splitlines():
        parts = line.split()
        if not parts:
            continue
        score = _parse_score(parts)
        if score is not None:
            score_cp = score
        index = _after(parts, "multipv")
        if parts[0] == "info" and index is not None and index.isdigit():
            several_lines = several_lines or int(index) > 1
            move = _after(parts, "pv")
            if score is not None and not {"lowerbound", "upperbound"} & set(parts):
                if int(index) == 1:
                    line_one_score = score
                if move is not None:
                    exact.append((_after(parts, "depth"), int(index), move, score))
        if parts[0] == "bestmove" and len(parts) >= 2:
            best = parts[1]
    if best is None:
        raise RuntimeError(f"No bestmove in engine output: {output}")
    lines: tuple[tuple[Move, float], ...] = ()
    if several_lines:
        # The last score printed can belong to a worse move, so the value is line 1's exact score.
        score_cp = line_one_score
        first = [report for report in exact if report[1] == 1]
        # Use the last depth with an exact line 1, and only if it is consistent: line 1 is the
        # move the engine played and no move appears twice. Otherwise callers keep the best move alone.
        final = {number: (move, score) for depth, number, move, score in exact if first and depth == first[-1][0]}
        moves = [move for move, _ in final.values()]
        if first and final[1][0] == best and len(set(moves)) == len(moves):
            lines = tuple(
                (Move.from_uci(engine_move_to_cc(move, rank_base)), score) for _, (move, score) in sorted(final.items())
            )
    return EngineResult(
        move=Move.from_uci(engine_move_to_cc(best, rank_base)), score_cp=score_cp, raw_output=output, lines=lines
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


def cc_move_to_engine(text: str, rank_base: int = 0) -> str:
    """The inverse of engine_move_to_cc: cc-lite ranks 0-9 top to bottom to engine ranks from Red's side."""

    move = Move.from_uci(text).uci()
    return f"{move[0]}{9 - int(move[1]) + rank_base}{move[2]}{9 - int(move[3]) + rank_base}"


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
        eval_file: str | None = None,
    ):
        if flavor not in ENGINE_FLAVORS:
            raise ValueError(f"Unknown engine flavor {flavor!r}; expected one of {sorted(ENGINE_FLAVORS)}")
        if flavor == "fairy" and init_command != "uci":
            # In UCCI mode Fairy-Stockfish switches to zero-based ranks and other option syntax.
            raise ValueError("The fairy flavor supports only the UCI protocol")
        if eval_file is not None and init_command != "uci":
            # UCCI has its own option syntax and no standard network option, so the pin could be ignored.
            raise ValueError("Pinning a network file is supported only over the UCI protocol")
        self.init_command = init_command
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
        if eval_file is not None:
            # Pin the network explicitly; otherwise the engine loads whatever file sits beside it.
            self.send(f"setoption name EvalFile value {Path(eval_file).expanduser().resolve()}")
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

    def new_game(self) -> None:
        """Clear search state, so a result depends on the position and not on what came before."""

        self.send("ucinewgame" if self.init_command == "uci" else "setoption newgame")
        self.send("isready")
        self.read_until(("readyok",))

    def best_move(self, fen: str, depth: int, moves: Sequence[str] = ()) -> EngineResult:
        """Search `fen` after `moves` (cc-lite notation), so the engine sees repetitions and perpetual checks."""

        engine_fen = to_engine_fen(fen)
        played = " ".join(cc_move_to_engine(move, self.rank_base) for move in moves)
        self.send(f"position fen {engine_fen} moves {played}" if played else f"position fen {engine_fen}")
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
