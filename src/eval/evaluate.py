from __future__ import annotations

import argparse
import json
import sys

from eval.adjudication import material_adjudication
from eval.baselines import material_move, random_move
from eval.engine import ENGINE_FLAVORS, UcciEngine, engine_provenance
from eval.evidence import summarize_game_records, write_evidence_bundle
from mcts.search import MCTS, SearchConfig, TorchEvaluator
from model.checkpoint import load_checkpoint
from train.config import choose_device, load_config, set_seed
from xiangqi.board import BLACK, RED, START_FEN, Board


def _terminal_result(board: Board, model_color: str) -> tuple[str, float] | None:
    adjudication = board.adjudication()
    if adjudication is None:
        return None
    return adjudication.reason, adjudication.result_for(model_color)


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
    engine_flavor: str = "pikafish",
    engine_eval_file: str | None = None,
    sample_plies: int = 0,
) -> dict[str, object]:
    if sample_plies < 0:
        raise ValueError("sample_plies must not be negative")
    if opponent == "engine" and sample_plies == 0 and games > 2:
        raise ValueError(
            "A greedy model against a fixed-depth engine replays one game per color; "
            "set eval.sample_plies above 0 so each seed plays a different game"
        )
    if opponent == "engine" and engine_flavor == "pikafish" and engine_eval_file is None:
        raise ValueError("Pikafish loads its network from a separate file; pass --engine-eval-file to pin it")
    device = choose_device(config.get("device", "auto"))
    model, _ = load_checkpoint(checkpoint, map_location=device)
    evaluator = TorchEvaluator(model, str(device))
    records: list[dict[str, object]] = []
    base_seed = int(config.get("seed", 1))
    engine = None
    opponent_engine = None
    if opponent == "engine":
        if not engine_command:
            raise ValueError("--engine-command is required when --opponent engine")
        opponent_engine = {
            **engine_provenance(engine_command, engine_eval_file),
            "flavor": engine_flavor,
            "depth": engine_depth,
            "protocol": engine_init_command,
        }
        engine = UcciEngine(
            engine_command,
            timeout=engine_timeout,
            init_command=engine_init_command,
            flavor=engine_flavor,
            eval_file=engine_eval_file,
        )
    try:
        for game in range(games):
            game_seed = base_seed + game
            set_seed(game_seed)
            model_color = RED if game % 2 == 0 else BLACK
            board = Board.from_fen(START_FEN)
            if engine is not None:
                engine.new_game()
            ply = 0
            model_moves = 0
            moves: list[str] = []
            game_nodes: list[float] = []
            while ply < max_plies and board.adjudication() is None:
                if board.turn == model_color:
                    # The model's first sample_plies moves are drawn from visit counts with this game's seed,
                    # counted per model move so that Black, which first moves at ply 1, is sampled too.
                    temperature = 1.0 if model_moves < sample_plies else 0.0
                    search = MCTS(evaluator, SearchConfig(simulations=simulations, temperature=temperature))
                    move, _ = search.run(board)
                    model_moves += 1
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
            terminal = _terminal_result(board, model_color)
            if terminal is None and ply >= max_plies:
                terminal_reason = "max_plies"
                terminal_score = None
                material_delta = board.material_score(model_color)
                material_adjudication_value = material_adjudication(material_delta)
            elif terminal is not None:
                terminal_reason, terminal_score = terminal
                material_delta = None
                material_adjudication_value = None
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
                    "material_adjudication": material_adjudication_value,
                    "illegal_move_count": 0,
                    "avg_nodes_per_second": (
                        sum(game_nodes) / len(game_nodes) if game_nodes else None
                    ),
                }
            )
    finally:
        if engine is not None:
            engine.close()
    return {
        "summary": summarize_game_records(records),
        "games": records,
        "device": str(device),
        "opponent_engine": opponent_engine,
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
    parser.add_argument("--engine-flavor", choices=sorted(ENGINE_FLAVORS), default="pikafish")
    parser.add_argument("--engine-eval-file", help="Network file to pin, required for Pikafish")
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
        engine_flavor=args.engine_flavor,
        engine_eval_file=args.engine_eval_file,
        sample_plies=int(eval_cfg.get("sample_plies", 0)),
    )
    write_evidence_bundle(
        args.output_dir,
        records=result["games"],
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        seed=int(config.get("seed", 1)),
        argv=["python", "-m", "eval.evaluate", *sys.argv[1:]],
        device=str(result["device"]),
        opponent_engine=result.get("opponent_engine"),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
