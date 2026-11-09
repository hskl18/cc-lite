from __future__ import annotations

import math
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from xiangqi.board import Board, Move
from xiangqi.encoding import MoveCodec, encode_board


class Evaluator(Protocol):
    def evaluate(self, board: Board) -> tuple[dict[Move, float], float]:
        """Return legal move priors and value from board.turn perspective."""


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 50
    c_puct: float = 1.5
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.0


@dataclass
class Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[Move, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return bool(self.children)


class TorchEvaluator:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, board: Board) -> tuple[dict[Move, float], float]:
        encoded = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        logits, value = self.model(encoded)
        legal_moves = board.legal_moves()
        if not legal_moves:
            return {}, -1.0
        indices = torch.tensor([MoveCodec.encode(m) for m in legal_moves], device=self.device)
        legal_logits = logits[0, indices]
        probs = torch.softmax(legal_logits, dim=0).detach().cpu().numpy()
        priors = {move: float(prob) for move, prob in zip(legal_moves, probs, strict=True)}
        return priors, float(value.item())


class TorchBatchEvaluator(TorchEvaluator):
    """Evaluates many boards in one forward pass, for searches that run in lockstep."""

    @torch.no_grad()
    def evaluate_many(self, boards: list[Board]) -> list[tuple[dict[Move, float], float]]:
        encoded = torch.from_numpy(np.stack([encode_board(board) for board in boards])).to(self.device)
        logits, values = self.model(encoded)
        logits, values = logits.float().cpu(), values.float().cpu()
        results: list[tuple[dict[Move, float], float]] = []
        for row, board in enumerate(boards):
            legal_moves = board.legal_moves()
            if not legal_moves:
                results.append(({}, -1.0))
                continue
            indices = torch.tensor([MoveCodec.encode(move) for move in legal_moves])
            probs = torch.softmax(logits[row, indices], dim=0).numpy()
            priors = {move: float(prob) for move, prob in zip(legal_moves, probs, strict=True)}
            results.append((priors, float(values[row])))
        return results


Evaluation = tuple[dict[Move, float], float]


class UniformEvaluator:
    def evaluate(self, board: Board) -> tuple[dict[Move, float], float]:
        legal_moves = board.legal_moves()
        if not legal_moves:
            return {}, -1.0
        prob = 1.0 / len(legal_moves)
        return {move: prob for move in legal_moves}, 0.0


class MCTS:
    def __init__(self, evaluator: Evaluator, config: SearchConfig):
        self.evaluator = evaluator
        self.config = config
        self.last_nodes_per_second = 0.0

    def run(self, board: Board) -> tuple[Move, dict[Move, float]]:
        steps = self.search(board)
        request = next(steps)
        while True:
            try:
                request = steps.send(self.evaluator.evaluate(request))
            except StopIteration as done:
                return done.value

    def search(self, board: Board) -> Generator[Board, Evaluation, tuple[Move, dict[Move, float]]]:
        """One search that yields each board needing a network evaluation and receives (priors, value).

        Evaluators return priors over exactly the legal moves, so expansion reuses them instead of
        generating the moves again; a driver can batch the yielded boards of many searches.
        """

        root = Node(prior=1.0)
        priors, _ = yield board
        self._expand(root, priors)
        if not root.children:
            raise ValueError("No legal moves available")
        self._add_root_noise(root)
        started = time.perf_counter()
        for _ in range(self.config.simulations):
            scratch = board.copy()
            node = root
            search_path = [node]
            while node.expanded():
                move, node = self._select_child(node)
                scratch._push_legal(move)
                search_path.append(node)
            value = self._terminal_value(scratch)
            if value is None:
                priors, value = yield scratch
                self._expand(node, priors)
            self._backpropagate(search_path, value)
        elapsed = max(time.perf_counter() - started, 1e-9)
        self.last_nodes_per_second = self.config.simulations / elapsed
        policy = self._visit_policy(root)
        move = self._sample_move(policy, self.config.temperature)
        return move, policy

    @staticmethod
    def _terminal_value(board: Board) -> float | None:
        """A rules-profile result from the side to move, or None while the game goes on."""

        adjudication = board.adjudication()
        return None if adjudication is None else adjudication.result_for(board.turn)

    @staticmethod
    def _expand(node: Node, priors: dict[Move, float]) -> None:
        for move, prior in priors.items():
            node.children[move] = Node(prior=prior)

    def _select_child(self, node: Node) -> tuple[Move, Node]:
        total_visits = max(1, node.visit_count)

        def score(child: Node) -> float:
            pb_c = self.config.c_puct * child.prior * math.sqrt(total_visits) / (child.visit_count + 1)
            return -child.value + pb_c

        return max(node.children.items(), key=lambda item: score(item[1]))

    def _backpropagate(self, search_path: list[Node], value: float) -> None:
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def _visit_policy(self, root: Node) -> dict[Move, float]:
        total = sum(child.visit_count for child in root.children.values())
        if total <= 0:
            prob = 1.0 / len(root.children)
            return {move: prob for move in root.children}
        return {move: child.visit_count / total for move, child in root.children.items()}

    def _sample_move(self, policy: dict[Move, float], temperature: float) -> Move:
        moves = list(policy)
        visits = np.array([policy[m] for m in moves], dtype=np.float64)
        if temperature <= 1e-6:
            return moves[int(np.argmax(visits))]
        scaled = visits ** (1.0 / temperature)
        scaled = scaled / scaled.sum()
        return moves[int(np.random.choice(len(moves), p=scaled))]

    def _add_root_noise(self, root: Node) -> None:
        frac = self.config.dirichlet_frac
        if frac <= 0.0 or not root.children:
            return
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(root.children))
        for child, eps in zip(root.children.values(), noise, strict=True):
            child.prior = child.prior * (1.0 - frac) + float(eps) * frac
