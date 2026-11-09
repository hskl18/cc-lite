"""Generate training positions from engine games with seeded random deviations.

The engine plays itself at a shallow depth, and a seeded fraction of moves is chosen
uniformly from the legal moves, so the positions cover more than one opening line.
Labels come later from `train.distill_engine` at a deeper search.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

from eval.engine import ENGINE_FLAVORS, UcciEngine
from eval.openings import load_opening_suite, mirror_fen
from train.distill_engine import _engine_provenance
from train.experiment import sha256_file
from xiangqi.board import Board


class GameSettings(NamedTuple):
    engine: str
    timeout: float
    flavor: str
    depth: int
    random_move_rate: float
    max_plies: int
    seed: int


def play_games(settings: GameSettings, game_ids: list[int]) -> tuple[list[str], int]:
    """Return the FEN before every move of each game, and how many engine moves were illegal here."""

    engine = UcciEngine(settings.engine, timeout=settings.timeout, flavor=settings.flavor)
    fens: list[str] = []
    illegal_engine_moves = 0
    try:
        for game_id in game_ids:
            rng = random.Random(f"{settings.seed}:{game_id}")
            engine.new_game()
            board = Board.start()
            for _ in range(settings.max_plies):
                moves = board.legal_moves()
                if not moves:
                    break
                fen = board.to_fen()
                fens.append(fen)
                if rng.random() < settings.random_move_rate:
                    move = rng.choice(moves)
                else:
                    move = engine.best_move(fen, settings.depth).move
                    if move not in moves:
                        # Engine rules differ from Profile 1 at the edges (for example repetition);
                        # count it and keep the game legal under this repository's rules.
                        illegal_engine_moves += 1
                        move = rng.choice(moves)
                board.push(move)
    finally:
        engine.close()
    return fens, illegal_engine_moves


def unique_positions(fens: list[str], excluded: set[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for fen in fens:
        if fen in seen or fen in excluded:
            continue
        seen.add(fen)
        unique.append(fen)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate positions from seeded engine games for distillation.")
    parser.add_argument("--engine", required=True, help="Engine command, e.g. './fairy-stockfish'")
    parser.add_argument("--engine-flavor", choices=sorted(ENGINE_FLAVORS), default="pikafish")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--depth", type=int, default=2, help="Search depth for the engine's own moves")
    parser.add_argument("--random-move-rate", type=float, default=0.15)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--exclude-opening-suite",
        help="Leave out this suite's positions and their mirrors, so evaluation openings stay unseen",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", help="Defaults to OUTPUT.manifest.json")
    args = parser.parse_args()

    if min(args.games, args.depth, args.max_plies, args.workers) <= 0:
        parser.error("--games, --depth, --max-plies, and --workers must be positive")
    if not 0.0 <= args.random_move_rate <= 1.0:
        parser.error("--random-move-rate must be between 0 and 1")

    output_path = Path(args.output).resolve()
    manifest_path = (
        Path(args.manifest).resolve() if args.manifest else output_path.with_name(output_path.name + ".manifest.json")
    )
    excluded: set[str] = set()
    suite_record = None
    if args.exclude_opening_suite:
        suite_path = Path(args.exclude_opening_suite).resolve()
        suite = load_opening_suite(suite_path)
        for position in suite.positions:
            excluded.update({position.fen, mirror_fen(position.fen)})
        suite_record = {"path": str(suite_path), "sha256": sha256_file(suite_path), "excluded_fens": len(excluded)}

    settings = GameSettings(
        args.engine, args.timeout, args.engine_flavor, args.depth, args.random_move_rate, args.max_plies, args.seed
    )
    game_ids = list(range(args.games))
    chunk_size = max(1, -(-len(game_ids) // (args.workers * 8)))
    chunks = [game_ids[start : start + chunk_size] for start in range(0, len(game_ids), chunk_size)]
    all_fens: list[str] = []
    illegal_engine_moves = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fens, illegal in pool.map(play_games, [settings] * len(chunks), chunks):
            all_fens.extend(fens)
            illegal_engine_moves += illegal

    positions = unique_positions(all_fens, excluded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps({"fen": fen}) + "\n" for fen in positions), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "kind": "engine_game_positions",
        "engine": {**_engine_provenance(args.engine), "flavor": args.engine_flavor, "depth": args.depth},
        "games": args.games,
        "seed": args.seed,
        "random_move_rate": args.random_move_rate,
        "max_plies": args.max_plies,
        "positions_seen": len(all_fens),
        "illegal_engine_moves": illegal_engine_moves,
        "excluded_opening_suite": suite_record,
        "positions": {"path": str(output_path), "sha256": sha256_file(output_path), "records": len(positions)},
        "argv": ["python", "-m", "train.engine_positions", *sys.argv[1:]],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"positions": len(positions), "seen": len(all_fens), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
