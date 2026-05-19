from __future__ import annotations

import argparse
import json

import numpy as np

from eval.baselines import material_move, random_move
from eval.engine import UcciEngine
from mcts.search import MCTS, SearchConfig, TorchEvaluator
from model.checkpoint import load_checkpoint
from train.config import choose_device, load_config, set_seed
from xiangqi.board import BLACK, RED, START_FEN, Board


def _score(board: Board, model_color: str, max_plies_reached: bool) -> float:
    if board.king_square(model_color) is None:
        return -1.0
    opponent = BLACK if model_color == RED else RED
    if board.king_square(opponent) is None:
        return 1.0
    if not board.legal_moves():
        return -1.0 if board.turn == model_color else 1.0
    if max_plies_reached:
        material = board.material_score(model_color)
        if abs(material) < 20:
            return 0.0
        return float(np.sign(material))
    return 0.0


def evaluate_checkpoint(
    checkpoint: str,
    *,
    config: dict,
    opponent: str,
    games: int,
    simulations: int,
    max_plies: int,
    engine_command: str | None = None,
    engine_depth: int = 4,
    engine_timeout: float = 10.0,
    engine_init_command: str = "uci",
) -> dict[str, float]:
    device = choose_device(config.get("device", "auto"))
    model, _ = load_checkpoint(checkpoint, map_location=device)
    evaluator = TorchEvaluator(model, str(device))
    scores: list[float] = []
    plies: list[int] = []
    nodes: list[float] = []
    engine = None
    if opponent == "engine":
        if not engine_command:
            raise ValueError("--engine-command is required when --opponent engine")
        engine = UcciEngine(engine_command, timeout=engine_timeout, init_command=engine_init_command)
    try:
        for game in range(games):
            model_color = RED if game % 2 == 0 else BLACK
            board = Board.from_fen(START_FEN)
            ply = 0
            while ply < max_plies and board.legal_moves():
                if board.turn == model_color:
                    search = MCTS(evaluator, SearchConfig(simulations=simulations, temperature=0.0))
                    move, _ = search.run(board)
                    nodes.append(search.last_nodes_per_second)
                elif opponent == "random":
                    move = random_move(board)
                elif opponent == "material":
                    move = material_move(board)
                elif opponent == "engine" and engine is not None:
                    result = engine.best_move(board.to_fen(), engine_depth)
                    move = result.move
                    if move not in set(board.legal_moves()):
                        raise ValueError(f"Engine returned illegal move {move.uci()} for {board.to_fen()}")
                else:
                    raise ValueError(f"Unknown opponent {opponent!r}")
                board.push(move)
                ply += 1
                if board.king_square(RED) is None or board.king_square(BLACK) is None:
                    break
            scores.append(_score(board, model_color, ply >= max_plies))
            plies.append(ply)
    finally:
        if engine is not None:
            engine.close()
    return {
        "games": games,
        "opponent": opponent,
        "engine_depth": float(engine_depth) if opponent == "engine" else 0.0,
        "score_mean": float(np.mean(scores)),
        "wins": float(sum(1 for s in scores if s > 0)),
        "draws": float(sum(1 for s in scores if s == 0)),
        "losses": float(sum(1 for s in scores if s < 0)),
        "avg_plies": float(np.mean(plies)),
        "avg_nodes_per_second": float(np.mean(nodes)) if nodes else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint against simple baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", choices=["random", "material", "engine"], default="random")
    parser.add_argument("--engine-command", help="UCI/UCCI engine command, e.g. './pikafish'")
    parser.add_argument("--engine-depth", type=int, default=4)
    parser.add_argument("--engine-timeout", type=float, default=10.0)
    parser.add_argument("--engine-init-command", choices=["uci", "ucci"], default="uci")
    parser.add_argument("--games", type=int)
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--max-plies", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 1)))
    eval_cfg = config.get("eval", {})
    metrics = evaluate_checkpoint(
        args.checkpoint,
        config=config,
        opponent=args.opponent,
        games=args.games or int(eval_cfg.get("games", 2)),
        simulations=args.simulations or int(eval_cfg.get("simulations", 25)),
        max_plies=args.max_plies or int(eval_cfg.get("max_plies", 120)),
        engine_command=args.engine_command,
        engine_depth=args.engine_depth,
        engine_timeout=args.engine_timeout,
        engine_init_command=args.engine_init_command,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
