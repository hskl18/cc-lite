from __future__ import annotations

import argparse
import json

from mcts.search import MCTS, SearchConfig, TorchEvaluator, UniformEvaluator
from model.checkpoint import load_checkpoint
from train.config import choose_device, load_config, set_seed
from xiangqi.board import START_FEN, Board


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a Xiangqi position with MCTS.")
    parser.add_argument("--config", default="configs/research_debug.yaml")
    parser.add_argument("--checkpoint", help="Optional model checkpoint. Uses uniform priors if omitted.")
    parser.add_argument("--fen", default=START_FEN)
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 1)))
    board = Board.from_fen(args.fen)
    simulations = args.simulations or int(config.get("eval", {}).get("simulations", 25))
    if args.checkpoint:
        device = choose_device(config.get("device", "auto"))
        model, _ = load_checkpoint(args.checkpoint, map_location=device)
        evaluator = TorchEvaluator(model, str(device))
    else:
        evaluator = UniformEvaluator()
    search = MCTS(evaluator, SearchConfig(simulations=simulations, temperature=0.0))
    best, policy = search.run(board)
    top = sorted(policy.items(), key=lambda item: item[1], reverse=True)[: args.top_k]
    print(
        json.dumps(
            {
                "fen": board.to_fen(),
                "best_move": best.uci(),
                "top_moves": [{"move": move.uci(), "prob": prob} for move, prob in top],
                "nodes_per_second": search.last_nodes_per_second,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

