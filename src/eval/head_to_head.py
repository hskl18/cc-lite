from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from eval.adjudication import material_adjudication
from eval.evidence import summarize_game_records, write_evidence_bundle
from eval.openings import OpeningSuite, load_opening_suite, opening_suite_summary
from eval.paired import build_paired_schedule, sequential_stop_decision
from mcts.search import MCTS, SearchConfig, TorchEvaluator
from model.checkpoint import load_checkpoint
from train.config import choose_device, load_config, set_seed
from xiangqi.board import BLACK, RED, START_FEN, Board


def _terminal_score(board: Board, a_color: str) -> float | None:
    adjudication = board.adjudication()
    return None if adjudication is None else adjudication.result_for(a_color)


def play_match(
    checkpoint_a: str,
    checkpoint_b: str,
    *,
    config: dict,
    games: int,
    simulations: int,
    max_plies: int,
) -> dict[str, float | None]:
    device = choose_device(config.get("device", "auto"))
    model_a, _ = load_checkpoint(checkpoint_a, map_location=device)
    model_b, _ = load_checkpoint(checkpoint_b, map_location=device)
    eval_a = TorchEvaluator(model_a, str(device))
    eval_b = TorchEvaluator(model_b, str(device))
    scores: list[float] = []
    truncated_games = 0
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
        score = _terminal_score(board, a_color)
        if score is None:
            truncated_games += 1
        else:
            scores.append(score)
    return {
        "games": games,
        "terminal_games": len(scores),
        "truncated_games": truncated_games,
        "a_score_mean": float(np.mean(scores)) if scores else None,
        "a_wins": float(sum(1 for s in scores if s > 0)),
        "draws": float(sum(1 for s in scores if s == 0)),
        "a_losses": float(sum(1 for s in scores if s < 0)),
    }


def _play_scheduled_pair(
    scheduled_games: Sequence[Any],
    *,
    eval_a: TorchEvaluator,
    eval_b: TorchEvaluator,
    simulations: int,
    max_plies: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scheduled in scheduled_games:
        set_seed(scheduled.seed)
        board = Board.from_fen(scheduled.start_fen)
        moves: list[str] = []
        nodes_per_second: list[float] = []
        while len(moves) < max_plies and board.adjudication() is None:
            evaluator = eval_a if board.turn == scheduled.candidate_color else eval_b
            search = MCTS(evaluator, SearchConfig(simulations=simulations, temperature=0.0))
            move, _ = search.run(board)
            nodes_per_second.append(search.last_nodes_per_second)
            moves.append(move.uci())
            board.push(move)

        adjudication = board.adjudication()
        if adjudication is None and len(moves) >= max_plies:
            terminal_reason = "max_plies"
            terminal_score = None
            material_delta = board.material_score(scheduled.candidate_color)
            material_adjudication_value = material_adjudication(material_delta)
        elif adjudication is not None:
            terminal_reason = adjudication.reason
            terminal_score = adjudication.result_for(scheduled.candidate_color)
            material_delta = None
            material_adjudication_value = None
        else:
            raise RuntimeError("Paired game ended without adjudication or max-plies truncation")
        records.append(
            {
                "game_id": scheduled.game_id,
                "pair_id": scheduled.pair_id,
                "pair_leg": scheduled.pair_leg,
                "seed": scheduled.seed,
                "opening_id": scheduled.opening_id,
                "candidate_color": scheduled.candidate_color,
                "model_color": scheduled.candidate_color,
                "opponent": "checkpoint",
                "start_fen": scheduled.start_fen,
                "final_fen": board.to_fen(),
                "moves": moves,
                "plies": len(moves),
                "terminal_reason": terminal_reason,
                "terminal_score": terminal_score,
                "material_delta": material_delta,
                "material_adjudication": material_adjudication_value,
                "illegal_move_count": 0,
                "avg_nodes_per_second": (
                    sum(nodes_per_second) / len(nodes_per_second) if nodes_per_second else None
                ),
            }
        )
    return records


def play_paired_match(
    checkpoint_a: str,
    checkpoint_b: str,
    *,
    config: dict,
    opening_suite: str | Path | OpeningSuite,
    seeds: Sequence[int],
    simulations: int,
    max_plies: int,
    min_pairs: int,
    max_pairs: int | None = None,
    min_seeds: int = 2,
    confidence: float = 0.95,
    superiority_margin: float = 0.0,
) -> dict[str, Any]:
    suite = (
        opening_suite
        if isinstance(opening_suite, OpeningSuite)
        else load_opening_suite(opening_suite)
    )
    schedule = build_paired_schedule(suite, seeds)
    available_pairs = len(schedule) // 2
    pair_cap = available_pairs if max_pairs is None else min(max_pairs, available_pairs)
    if min_pairs < 1:
        raise ValueError("min_pairs must be at least 1")
    if min_pairs > pair_cap:
        raise ValueError("min_pairs cannot exceed the available or configured pair cap")

    device = choose_device(config.get("device", "auto"))
    model_a, _ = load_checkpoint(checkpoint_a, map_location=device)
    model_b, _ = load_checkpoint(checkpoint_b, map_location=device)
    eval_a = TorchEvaluator(model_a, str(device))
    eval_b = TorchEvaluator(model_b, str(device))
    records: list[dict[str, Any]] = []
    stop: dict[str, Any] | None = None
    for pair_start in range(0, pair_cap * 2, 2):
        records.extend(
            _play_scheduled_pair(
                schedule[pair_start : pair_start + 2],
                eval_a=eval_a,
                eval_b=eval_b,
                simulations=simulations,
                max_plies=max_plies,
            )
        )
        stop = sequential_stop_decision(
            records,
            min_pairs=min_pairs,
            max_pairs=pair_cap,
            min_seeds=min_seeds,
            confidence=confidence,
            superiority_margin=superiority_margin,
        )
        if stop["stop"]:
            break
    summary = summarize_game_records(records)
    summary["sequential_stop"] = stop
    summary["opening_suite"] = opening_suite_summary(suite)
    paired_protocol = {
        "min_pairs": min_pairs,
        "max_pairs": pair_cap,
        "min_seeds": min_seeds,
        "confidence": confidence,
        "superiority_margin": superiority_margin,
    }
    return {
        "summary": summary,
        "games": records,
        "device": str(device),
        "suite": suite,
        "paired_protocol": paired_protocol,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a new checkpoint against a previous checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--games", type=int)
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--max-plies", type=int)
    parser.add_argument("--opening-suite")
    parser.add_argument("--seeds", help="Comma-separated evaluation seeds for paired mode.")
    parser.add_argument("--min-pairs", type=int, default=8)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--superiority-margin", type=float, default=0.0)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 1)))
    eval_cfg = config.get("eval", {})
    simulations = args.simulations or int(eval_cfg.get("simulations", 25))
    max_plies = args.max_plies or int(eval_cfg.get("max_plies", 120))
    if args.opening_suite:
        configured_seeds = eval_cfg.get("seeds", [1, 7, 19])
        seeds = (
            [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
            if args.seeds
            else [int(value) for value in configured_seeds]
        )
        result = play_paired_match(
            args.checkpoint_a,
            args.checkpoint_b,
            config=config,
            opening_suite=args.opening_suite,
            seeds=seeds,
            simulations=simulations,
            max_plies=max_plies,
            min_pairs=args.min_pairs,
            max_pairs=args.max_pairs,
            min_seeds=args.min_seeds,
            confidence=args.confidence,
            superiority_margin=args.superiority_margin,
        )
        if args.output_dir:
            write_evidence_bundle(
                args.output_dir,
                records=result["games"],
                checkpoint_path=args.checkpoint_a,
                opponent_checkpoint_path=args.checkpoint_b,
                opening_suite_path=args.opening_suite,
                config_path=args.config,
                seed=seeds[0],
                seeds=seeds,
                paired_protocol=result["paired_protocol"],
                argv=["python", "-m", "eval.head_to_head", *sys.argv[1:]],
                device=str(result["device"]),
            )
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
    else:
        if args.output_dir:
            parser.error("--output-dir requires --opening-suite")
        metrics = play_match(
            args.checkpoint_a,
            args.checkpoint_b,
            config=config,
            games=args.games or int(eval_cfg.get("games", 2)),
            simulations=simulations,
            max_plies=max_plies,
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
