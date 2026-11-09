"""AlphaZero-lite generations from a starting checkpoint, anchored to engine labels.

Each generation, worker processes play batched self-play games with the current checkpoint;
the network then trains on the last `loop.replay_generations` of self-play data plus a seeded
sample of engine labels `loop.anchor_ratio` times that size, which keeps it near what it was
distilled on, and the result becomes the next generation. Every checkpoint, the per-generation
metrics, and the hashes of everything used go into `manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import random
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import torch

from data.replay import ReplaySample, read_jsonl, write_jsonl
from eval.evidence import _git_source, sha256_file
from model.checkpoint import load_checkpoint, save_checkpoint
from train.config import choose_device, load_config, set_seed
from train.loop import train_samples
from train.self_play import generate_self_play_batched


class WorkerJob(NamedTuple):
    checkpoint: str
    device: str
    games: int
    simulations: int
    max_plies: int
    seed: int
    temperature: float
    temperature_plies: int | None
    dirichlet_frac: float


def play_share(job: WorkerJob) -> tuple[list[ReplaySample], dict[str, float]]:
    device = torch.device(job.device)
    torch.set_num_threads(1)
    model, _ = load_checkpoint(job.checkpoint, map_location=device)
    return generate_self_play_batched(
        model,
        device=device,
        games=job.games,
        simulations=job.simulations,
        max_plies=job.max_plies,
        seed=job.seed,
        temperature=job.temperature,
        temperature_plies=job.temperature_plies,
        dirichlet_frac=job.dirichlet_frac,
    )


def split_games(games: int, workers: int) -> list[int]:
    base, extra = divmod(games, workers)
    return [base + (1 if worker < extra else 0) for worker in range(workers) if base or worker < extra]


def merge_metrics(parts: list[dict[str, float]]) -> dict[str, float]:
    games = sum(part["games"] for part in parts)
    return {
        "games": games,
        "positions": sum(part["positions"] for part in parts),
        "capped_games": sum(part["capped_games"] for part in parts),
        "avg_game_plies": sum(part["avg_game_plies"] * part["games"] for part in parts) / games,
        "avg_red_score": sum(part["avg_red_score"] * part["games"] for part in parts) / games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run self-play generations from a checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Starting checkpoint")
    parser.add_argument("--anchor-labels", required=True, help="Engine labels mixed into every generation")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 1))
    device = choose_device(str(config.get("device", "auto")))
    sp = config.get("self_play", {})
    loop_cfg = config.get("loop", {})
    train_cfg = config.get("training", {})
    generations = int(loop_cfg.get("generations", 1))
    workers = int(sp.get("workers", 1))
    games = int(sp.get("games", 2))
    temperature_plies = sp.get("temperature_plies")
    if min(generations, workers, games, int(loop_cfg.get("replay_generations", 1))) <= 0:
        raise ValueError("generations, workers, games, and replay_generations must be positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    anchor = read_jsonl(args.anchor_labels)
    replay: deque[list[ReplaySample]] = deque(maxlen=int(loop_cfg.get("replay_generations", 1)))
    current = Path(args.checkpoint)
    records = []
    context = multiprocessing.get_context("spawn")  # CUDA cannot be forked
    for generation in range(1, generations + 1):
        started = time.perf_counter()
        jobs = [
            WorkerJob(
                str(current),
                str(device),
                share,
                int(sp.get("simulations", 25)),
                int(sp.get("max_plies", 200)),
                seed * 1_000_000 + generation * 1_000 + worker,
                float(sp.get("temperature", 1.0)),
                None if temperature_plies is None else int(temperature_plies),
                float(sp.get("dirichlet_frac", 0.25)),
            )
            for worker, share in enumerate(split_games(games, workers))
        ]
        if len(jobs) == 1:
            results = [play_share(jobs[0])]
        else:
            with ProcessPoolExecutor(max_workers=len(jobs), mp_context=context) as pool:
                results = list(pool.map(play_share, jobs))
        samples = [sample for part, _ in results for sample in part]
        self_play_metrics = merge_metrics([metrics for _, metrics in results])
        self_play_seconds = time.perf_counter() - started

        directory = output / f"gen-{generation:02d}"
        write_jsonl(directory / "selfplay.jsonl", samples)
        replay.append(samples)
        buffer = [sample for part in replay for sample in part]
        anchor_count = min(len(anchor), int(float(loop_cfg.get("anchor_ratio", 1.0)) * len(buffer)))
        training_data = buffer + random.Random(seed * 1_000 + generation).sample(anchor, anchor_count)

        set_seed(seed + generation)
        model, _ = load_checkpoint(current, map_location=device)
        stats = train_samples(
            model,
            training_data,
            device=device,
            batch_size=int(train_cfg.get("batch_size", 16)),
            epochs=int(train_cfg.get("epochs", 1)),
            lr=float(train_cfg.get("lr", 1e-3)),
            precision=str(train_cfg.get("precision", "fp32")),
            num_workers=int(train_cfg.get("num_workers", 0)),
        )
        checkpoint = directory / "checkpoint.pt"
        save_checkpoint(
            checkpoint,
            model,
            step=len(training_data),
            metadata={"generation": generation, "parent_sha256": sha256_file(current), "train_stats": stats.__dict__},
        )
        records.append(
            {
                "generation": generation,
                "parent": {"path": str(current), "sha256": sha256_file(current)},
                "self_play": {**self_play_metrics, "seconds": round(self_play_seconds, 1)},
                "self_play_data": {"path": str(directory / "selfplay.jsonl"), "sha256": sha256_file(directory / "selfplay.jsonl")},
                "training": {
                    "self_play_positions": len(buffer),
                    "anchor_positions": anchor_count,
                    "seconds": round(time.perf_counter() - started - self_play_seconds, 1),
                    **stats.__dict__,
                },
                "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            }
        )
        print(json.dumps(records[-1], sort_keys=True), flush=True)
        current = checkpoint

    manifest = {
        "schema_version": 1,
        "kind": "self_play_generations",
        "config": {"path": args.config, "sha256": sha256_file(args.config)},
        "start_checkpoint": {"path": args.checkpoint, "sha256": sha256_file(args.checkpoint)},
        "anchor_labels": {"path": args.anchor_labels, "sha256": sha256_file(args.anchor_labels)},
        "generations": records,
        "source": _git_source(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "device": str(device),
        },
        "argv": ["python", "-m", "train.self_play_loop", *sys.argv[1:]],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
