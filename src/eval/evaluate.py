from __future__ import annotations

import argparse
import json
import sys

from eval.baselines import material_move, random_move
from eval.engine import UcciEngine
from eval.evidence import summarize_game_records, write_evidence_bundle
from mcts.search import MCTS, SearchConfig, TorchEvaluator
from model.checkpoint import load_checkpoint
from train.config import choose_device, load_config, set_seed
from xiangqi.board import BLACK, RED, START_FEN, Board


def _terminal_result(board: Board, model_color: str) -> tuple[str, float] | None:
    if board.king_square(model_color) is None:
        return "model_king_captured", -1.0
    opponent = BLACK if model_color == RED else RED
    if board.king_square(opponent) is None:
        return "opponent_king_captured", 1.0
    if not board.legal_moves():
        return "no_legal_moves", -1.0 if board.turn == model_color else 1.0
    return None


def _material_adjudication(material_delta: int) -> float:
    if abs(material_delta) < 20:
        return 0.0
    return 1.0 if material_delta > 0 else -1.0


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
) -> dict[str, object]:
    device = choose_device(config.get("device", "auto"))
    model, _ = load_checkpoint(checkpoint, map_location=device)
    evaluator = TorchEvaluator(model, str(device))
    records: list[dict[str, object]] = []
    base_seed = int(config.get("seed", 1))
    engine = None
    if opponent == "engine":
        if not engine_command:
            raise ValueError("--engine-command is required when --opponent engine")
        engine = UcciEngine(engine_command, timeout=engine_timeout, init_command=engine_init_command)
    try:
        for game in range(games):
            game_seed = base_seed + game
            set_seed(game_seed)
            model_color = RED if game % 2 == 0 else BLACK
            board = Board.from_fen(START_FEN)
            ply = 0
            moves: list[str] = []
            game_nodes: list[float] = []
            while ply < max_plies and board.legal_moves():
                if board.turn == model_color:
                    search = MCTS(evaluator, SearchConfig(simulations=simulations, temperature=0.0))
                    move, _ = search.run(board)
                    game_nodes.append(search.last_nodes_per_second)
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
                moves.append(move.uci())
                board.push(move)
                ply += 1
                if board.king_square(RED) is None or board.king_square(BLACK) is None:
                    break
            terminal = _terminal_result(board, model_color)
            if terminal is None and ply >= max_plies:
                terminal_reason = "max_plies"
                terminal_score = None
                material_delta = board.material_score(model_color)
                material_adjudication = _material_adjudication(material_delta)
            elif terminal is not None:
                terminal_reason, terminal_score = terminal
                material_delta = None
                material_adjudication = None
            else:
                raise RuntimeError("Evaluation ended without a terminal result or max-plies truncation")
            records.append(
                {
                    "game_id": f"{opponent}-{game + 1:04d}",
                    "seed": game_seed,
                    "model_color": model_color,
                    "opponent": opponent,
                    "start_fen": START_FEN,
                    "final_fen": board.to_fen(),
                    "moves": moves,
                    "plies": ply,
                    "terminal_reason": terminal_reason,
                    "terminal_score": terminal_score,
                    "material_delta": material_delta,
                    "material_adjudication": material_adjudication,
                    "illegal_move_count": 0,
                    "avg_nodes_per_second": (
                        sum(game_nodes) / len(game_nodes) if game_nodes else None
                    ),
                }
            )
    finally:
        if engine is not None:
            engine.close()
    return {"summary": summarize_game_records(records), "games": records, "device": str(device)}


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
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 1)))
    eval_cfg = config.get("eval", {})
    result = evaluate_checkpoint(
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
    write_evidence_bundle(
        args.output_dir,
        records=result["games"],
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        seed=int(config.get("seed", 1)),
        argv=["python", "-m", "eval.evaluate", *sys.argv[1:]],
        device=str(result["device"]),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
