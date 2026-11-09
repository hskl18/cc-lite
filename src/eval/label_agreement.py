"""Measure a checkpoint against engine labels on held-out positions.

Top-1 agreement masks illegal moves, as MCTS does. Value sign agreement only counts labels
at least `--sign-threshold` from zero; compare checkpoints trained on different value mappings
on it only when both are measured on the same labels with the same threshold.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.replay import read_jsonl
from eval.evidence import _git_source, sha256_file
from model.checkpoint import load_checkpoint
from train.config import choose_device
from train.dataset import ReplayDataset


@torch.no_grad()
def measure_agreement(
    model: torch.nn.Module,
    dataset: ReplayDataset,
    *,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 0,
    sign_threshold: float = 0.1,
) -> dict[str, float | int]:
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    positions = agree = unmasked_legal = sign_positions = sign_agree = 0
    squared_error = 0.0
    for x, target_policy, target_value, legal in loader:
        x, legal = x.to(device), legal.to(device)
        target_policy, target_value = target_policy.to(device), target_value.to(device)
        logits, value = model(x)
        rows = torch.arange(x.shape[0], device=device)
        masked = logits.masked_fill(~legal, float("-inf"))
        agree += int((masked.argmax(dim=1) == target_policy.argmax(dim=1)).sum())
        unmasked_legal += int(legal[rows, logits.argmax(dim=1)].sum())
        squared_error += float(((value - target_value) ** 2).sum())
        decided = target_value.abs() >= sign_threshold
        sign_positions += int(decided.sum())
        sign_agree += int((torch.sign(value[decided]) == torch.sign(target_value[decided])).sum())
        positions += x.shape[0]
    if positions == 0:
        raise ValueError("No labeled positions to measure")
    return {
        "positions": positions,
        "top1_agreement": agree / positions,
        "unmasked_top1_legal": unmasked_legal / positions,
        "value_mse": squared_error / positions,
        "value_sign_positions": sign_positions,
        "value_sign_agreement": sign_agree / sign_positions if sign_positions else None,
        "sign_threshold": sign_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a checkpoint with held-out engine labels.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels", required=True, help="Held-out labels from train.distill_engine")
    parser.add_argument("--labels-manifest", help="Its manifest, hashed into the report")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sign-threshold", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = choose_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, map_location=device)
    metrics = measure_agreement(
        model,
        ReplayDataset(read_jsonl(args.labels)),
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sign_threshold=args.sign_threshold,
    )
    report = {
        "schema_version": 1,
        "kind": "held_out_label_agreement",
        "metrics": metrics,
        "checkpoint": {"path": args.checkpoint, "sha256": sha256_file(args.checkpoint)},
        "labels": {"path": args.labels, "sha256": sha256_file(args.labels)},
        "labels_manifest": (
            {"path": args.labels_manifest, "sha256": sha256_file(args.labels_manifest)}
            if args.labels_manifest
            else None
        ),
        "source": _git_source(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "device": str(device),
        },
        "argv": ["python", "-m", "eval.label_agreement", *sys.argv[1:]],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
