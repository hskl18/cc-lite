from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from data.replay import ReplaySample, write_jsonl
from mcts.search import MCTS, SearchConfig, TorchEvaluator, UniformEvaluator
from model.checkpoint import load_checkpoint, save_checkpoint
from model.network import ModelConfig, PolicyValueNet
from train.config import choose_device, load_config, set_seed
from train.loop import train_samples
from xiangqi.board import BLACK, RED, START_FEN, Board
from xiangqi.encoding import MoveCodec


def _game_outcome(board: Board, max_plies_reached: bool) -> dict[str, float]:
    adjudication = board.adjudication()
    if adjudication is not None:
        return {RED: adjudication.result_for(RED), BLACK: adjudication.result_for(BLACK)}
    if max_plies_reached:
        return {RED: 0.0, BLACK: 0.0}
    return {RED: 0.0, BLACK: 0.0}


def generate_self_play(
    model: PolicyValueNet | None,
    *,
    device: torch.device,
    games: int,
    simulations: int,
    max_plies: int,
    seed: int,
    start_fen: str = START_FEN,
    temperature: float = 1.0,
) -> tuple[list[ReplaySample], dict[str, float]]:
    set_seed(seed)
    evaluator = TorchEvaluator(model, str(device)) if model is not None else UniformEvaluator()
    samples: list[ReplaySample] = []
    lengths: list[int] = []
    nodes_per_second: list[float] = []
    red_scores: list[float] = []
    for _game_id in range(games):
        board = Board.from_fen(start_fen)
        trajectory: list[tuple[str, str, dict[int, float]]] = []
        for _ in range(max_plies):
            if not board.legal_moves():
                break
            search = MCTS(
                evaluator,
                SearchConfig(
                    simulations=simulations,
                    temperature=temperature,
                    dirichlet_frac=0.25 if model is not None else 0.0,
                ),
            )
            move, policy = search.run(board)
            nodes_per_second.append(search.last_nodes_per_second)
            trajectory.append(
                (
                    board.to_fen(),
                    board.turn,
                    {MoveCodec.encode(m): float(p) for m, p in policy.items()},
                )
            )
            board.push(move)
            if board.adjudication() is not None:
                break
        outcome = _game_outcome(board, len(trajectory) >= max_plies)
        red_scores.append(outcome[RED])
        lengths.append(len(trajectory))
        samples.extend(ReplaySample(fen=fen, policy=policy, value=outcome[color]) for fen, color, policy in trajectory)
    metrics = {
        "games": games,
        "positions": len(samples),
        "avg_game_plies": float(np.mean(lengths)) if lengths else 0.0,
        "avg_red_score": float(np.mean(red_scores)) if red_scores else 0.0,
        "avg_nodes_per_second": float(np.mean(nodes_per_second)) if nodes_per_second else 0.0,
    }
    return samples, metrics


def build_model(config: dict) -> PolicyValueNet:
    model_cfg = config.get("model", {})
    preset = model_cfg.get("preset", "tiny")
    defaults = {"tiny": (64, 4), "small": (96, 6)}
    if preset not in defaults:
        raise ValueError(f"Unknown model preset {preset!r}")
    default_channels, default_blocks = defaults[preset]
    return PolicyValueNet(
        ModelConfig(
            channels=int(model_cfg.get("channels", default_channels)),
            blocks=int(model_cfg.get("blocks", default_blocks)),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AlphaZero-lite self-play data and optionally train.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="runs/selfplay.jsonl")
    parser.add_argument("--train", action="store_true", help="Train one compact checkpoint from generated data.")
    parser.add_argument("--checkpoint-out", default="runs/checkpoints/selfplay.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 1))
    set_seed(seed)
    device = choose_device(config.get("device", "auto"))
    if args.checkpoint:
        model, _ = load_checkpoint(args.checkpoint, map_location=device)
    else:
        model = build_model(config)
    sp = config.get("self_play", {})
    samples, metrics = generate_self_play(
        model,
        device=device,
        games=int(sp.get("games", 2)),
        simulations=int(sp.get("simulations", 25)),
        max_plies=int(sp.get("max_plies", 80)),
        seed=seed,
        temperature=float(sp.get("temperature", 1.0)),
    )
    count = write_jsonl(args.output, samples)
    print(json.dumps({"wrote": count, "output": args.output, **metrics}, indent=2, sort_keys=True))
    if args.train:
        train_cfg = config.get("training", {})
        stats = train_samples(
            model,
            samples,
            device=device,
            batch_size=int(train_cfg.get("batch_size", 16)),
            epochs=int(train_cfg.get("epochs", 1)),
            lr=float(train_cfg.get("lr", 1e-3)),
            precision=str(train_cfg.get("precision", "fp32")),
        )
        save_checkpoint(
            args.checkpoint_out,
            model,
            step=count,
            metadata={"config": config, "self_play_metrics": metrics, "train_stats": stats.__dict__},
        )
        print(json.dumps({"checkpoint": args.checkpoint_out, "train": stats.__dict__}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
