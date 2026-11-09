from __future__ import annotations

import torch

from data.replay import ReplaySample
from eval.label_agreement import measure_agreement
from model.network import ModelConfig, PolicyValueNet
from train.dataset import ReplayDataset
from xiangqi.board import START_FEN, Board
from xiangqi.encoding import POLICY_SIZE, MoveCodec


def _samples() -> list[ReplaySample]:
    moves = Board.from_fen(START_FEN).legal_moves()[:3]
    return [
        ReplaySample(fen=START_FEN, policy={MoveCodec.encode(move): 1.0}, value=value)
        for move, value in zip(moves, (0.5, -0.5, 0.0), strict=True)
    ]


class _Oracle(torch.nn.Module):
    """Puts its top logit on an illegal index and the label just below it, with the label's value sign."""

    def __init__(self, samples: list[ReplaySample]) -> None:
        super().__init__()
        self.samples = samples
        self.calls = 0

    def forward(self, x: torch.Tensor):
        logits = torch.zeros(x.shape[0], POLICY_SIZE)
        values = torch.zeros(x.shape[0])
        for row in range(x.shape[0]):
            sample = self.samples[self.calls + row]
            logits[row, 0] = 10.0  # index 0 is not a legal move from the start position
            logits[row, next(iter(sample.policy))] = 5.0
            values[row] = 0.9 if sample.value > 0 else -0.9
        self.calls += x.shape[0]
        return logits, values


def test_masked_agreement_ignores_illegal_top_moves_and_counts_value_signs() -> None:
    samples = _samples()
    assert MoveCodec.encode(Board.from_fen(START_FEN).legal_moves()[0]) != 0
    metrics = measure_agreement(_Oracle(samples), ReplayDataset(samples), device=torch.device("cpu"), batch_size=2)
    assert metrics["positions"] == 3
    assert metrics["top1_agreement"] == 1.0
    assert metrics["unmasked_top1_legal"] == 0.0
    assert metrics["value_sign_positions"] == 2
    assert metrics["value_sign_agreement"] == 1.0


def test_agreement_runs_on_a_real_network() -> None:
    samples = _samples()
    model = PolicyValueNet(ModelConfig(channels=8, blocks=1))
    metrics = measure_agreement(model, ReplayDataset(samples), device=torch.device("cpu"))
    assert metrics["positions"] == 3
    assert 0.0 <= metrics["top1_agreement"] <= 1.0
    assert metrics["value_mse"] >= 0.0
