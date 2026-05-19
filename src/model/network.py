from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from xiangqi.encoding import INPUT_PLANES, POLICY_SIZE


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 64
    blocks: int = 4
    input_planes: int = INPUT_PLANES
    policy_size: int = POLICY_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class PolicyValueNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        c = config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(config.input_planes, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(c) for _ in range(config.blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 10 * 9, config.policy_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(10 * 9, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.blocks(self.stem(x))
        policy_logits = self.policy_head(trunk)
        value = self.value_head(trunk).squeeze(-1)
        return policy_logits, value


def create_model(preset: str = "tiny", channels: int | None = None, blocks: int | None = None) -> PolicyValueNet:
    if preset == "tiny":
        cfg = ModelConfig(channels=channels or 64, blocks=blocks or 4)
    elif preset == "small":
        cfg = ModelConfig(channels=channels or 96, blocks=blocks or 6)
    else:
        raise ValueError(f"Unknown model preset {preset!r}")
    return PolicyValueNet(cfg)

