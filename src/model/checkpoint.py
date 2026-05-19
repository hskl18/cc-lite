from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model.network import ModelConfig, PolicyValueNet


def save_checkpoint(
    path: str | Path,
    model: PolicyValueNet,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": model.config.__dict__,
        "step": step,
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[PolicyValueNet, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = ModelConfig(**payload["model_config"])
    model = PolicyValueNet(config)
    model.load_state_dict(payload["model_state"])
    return model, payload

