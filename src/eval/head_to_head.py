from __future__ import annotations

import argparse
import json

import numpy as np

from mcts.search import MCTS, SearchConfig, TorchEvaluator
from model.checkpoint import load_checkpoint
from train.config import choose_device, load_config, set_seed
from xiangqi.board import BLACK, RED, START_FEN, Board


def play_match(
    checkpoint_a: str,
    checkpoint_b: str,
    *,
    config: dict,
    games: int,
    simulations: int,
    max_plies: int,
) -> dict[str, float]:
    device = choose_device(config.get("device", "auto"))
    model_a, _ = load_checkpoint(checkpoint_a, map_location=device)
    model_b, _ = load_checkpoint(checkpoint_b, map_location=device)
    eval_a = TorchEvaluator(model_a, str(device))
    eval_b = TorchEvaluator(model_b, str(device))
    scores: list[float] = []
    for game in range(games):
        a_color = RED if game % 2 == 0 else BLACK
        board = Board.from_fen(START_FEN)
        ply = 0
        while ply < max_plies and board.legal_moves():
            evaluator = eval_a if board.turn == a_color else eval_b
            search = MCTS(evaluator, SearchConfig(simulations=simulations, temperature=0.0))
            move, _ = search.run(board)
            board.push(move)
            ply += 1
            if board.king_square(RED) is None or board.king_square(BLACK) is None:
                break
        if board.king_square(a_color) is None:
            scores.append(-1.0)
        elif board.king_square(BLACK if a_color == RED else RED) is None:
            scores.append(1.0)
        elif not board.legal_moves():
            scores.append(-1.0 if board.turn == a_color else 1.0)
        else:
            material = board.material_score(a_color)
            scores.append(0.0 if abs(material) < 20 else float(np.sign(material)))
    return {
        "games": games,
        "a_score_mean": float(np.mean(scores)),
        "a_wins": float(sum(1 for s in scores if s > 0)),
        "draws": float(sum(1 for s in scores if s == 0)),
        "a_losses": float(sum(1 for s in scores if s < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a new checkpoint against a previous checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--games", type=int)
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--max-plies", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 1)))
    eval_cfg = config.get("eval", {})
    metrics = play_match(
        args.checkpoint_a,
        args.checkpoint_b,
        config=config,
        games=args.games or int(eval_cfg.get("games", 2)),
        simulations=args.simulations or int(eval_cfg.get("simulations", 25)),
        max_plies=args.max_plies or int(eval_cfg.get("max_plies", 120)),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

