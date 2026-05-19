from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data.replay import ReplaySample
from model.network import PolicyValueNet
from train.dataset import ReplayDataset


@dataclass
class TrainStats:
    policy_loss: float
    value_loss: float
    legal_move_accuracy: float
    target_accuracy: float


def train_samples(
    model: PolicyValueNet,
    samples: list[ReplaySample],
    *,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
    precision: str = "fp32",
) -> TrainStats:
    if not samples:
        raise ValueError("No samples provided")
    dataset = ReplayDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    model.train()
    total_policy = total_value = total_legal = total_target = total_seen = 0.0
    use_amp = precision in {"fp16", "bf16"} and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and device.type == "cuda")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    for _ in range(epochs):
        for x, target_policy, target_value, legal_mask in loader:
            x = x.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.to(device)
            legal_mask = legal_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, value = model(x)
                log_probs = F.log_softmax(logits, dim=1)
                policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(value, target_value)
                loss = policy_loss + value_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch = x.shape[0]
            pred = logits.detach().argmax(dim=1)
            target = target_policy.detach().argmax(dim=1)
            total_policy += float(policy_loss.detach().cpu()) * batch
            total_value += float(value_loss.detach().cpu()) * batch
            total_legal += float(legal_mask[torch.arange(batch, device=device), pred].float().sum().cpu())
            total_target += float((pred == target).float().sum().cpu())
            total_seen += batch
    return TrainStats(
        policy_loss=total_policy / total_seen,
        value_loss=total_value / total_seen,
        legal_move_accuracy=total_legal / total_seen,
        target_accuracy=total_target / total_seen,
    )


def split_train_val(samples: list[ReplaySample], val_fraction: float = 0.1, seed: int = 1):
    dataset = ReplayDataset(samples)
    val_size = max(1, int(len(samples) * val_fraction)) if len(samples) > 1 else 0
    train_size = len(samples) - val_size
    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

