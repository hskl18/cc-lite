from __future__ import annotations

import torch
from torch.utils.data import Dataset

from data.replay import ReplaySample
from xiangqi.board import Board
from xiangqi.encoding import POLICY_SIZE, MoveCodec, encode_board


class ReplayDataset(Dataset):
    def __init__(self, samples: list[ReplaySample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        board = Board.from_fen(sample.fen)
        x = torch.from_numpy(encode_board(board))
        policy = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        for move_index, prob in sample.policy.items():
            policy[move_index] = prob
        legal = torch.zeros(POLICY_SIZE, dtype=torch.bool)
        for move in board.legal_moves():
            legal[MoveCodec.encode(move)] = True
        value = torch.tensor(sample.value, dtype=torch.float32)
        return x, policy, value, legal

