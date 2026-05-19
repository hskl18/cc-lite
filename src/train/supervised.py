from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.replay import ReplaySample, read_jsonl
from model.checkpoint import save_checkpoint
from train.config import choose_device, load_config, set_seed
from train.loop import train_samples
from train.self_play import build_model
from xiangqi.board import Board, Move
from xiangqi.encoding import MoveCodec


def load_supervised_records(path: str | Path) -> list[ReplaySample]:
    path = Path(path)
    if path.suffix == ".jsonl":
        # Native format: {"fen": "...", "move": "a0a1", "value": 1}
        samples: list[ReplaySample] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            item = json.loads(raw)
            if "policy" in item:
                samples.append(
                    ReplaySample(
                        fen=item["fen"],
                        policy={int(k): float(v) for k, v in item["policy"].items()},
                        value=float(item.get("value", 0.0)),
                    )
                )
            else:
                move = Move.from_uci(item["move"])
                samples.append(
                    ReplaySample(
                        fen=item["fen"],
                        policy={MoveCodec.encode(move): 1.0},
                        value=float(item.get("value", 0.0)),
                    )
                )
        return samples
    raise ValueError("Only .jsonl supervised records are supported in the baseline loader")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train policy/value net from Xiangqi game records.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="JSONL with fen/move/value or native replay records.")
    parser.add_argument("--checkpoint-out", default="runs/checkpoints/supervised.pt")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 1))
    set_seed(seed)
    device = choose_device(config.get("device", "auto"))
    model = build_model(config)
    samples = load_supervised_records(args.data)
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
        step=len(samples),
        metadata={"config": config, "data": str(args.data), "train_stats": stats.__dict__},
    )
    print(json.dumps({"checkpoint": args.checkpoint_out, "samples": len(samples), **stats.__dict__}, indent=2))


if __name__ == "__main__":
    main()

