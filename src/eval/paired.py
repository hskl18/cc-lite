from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from eval.openings import OpeningSuite
from xiangqi.board import BLACK, RED


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    game_id: str
    pair_id: str
    pair_leg: int
    opening_id: str
    start_fen: str
    seed: int
    candidate_color: str


def build_paired_schedule(suite: OpeningSuite, seeds: Sequence[int]) -> list[ScheduledGame]:
    normalized_seeds = [int(seed) for seed in seeds]
    if not normalized_seeds:
        raise ValueError("At least one evaluation seed is required")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("Evaluation seeds must be unique")
    games: list[ScheduledGame] = []
    for opening in suite.positions:
        for seed in normalized_seeds:
            pair_id = f"{suite.suite_id}:{opening.opening_id}:seed-{seed}"
            for leg, color in ((1, RED), (2, BLACK)):
                games.append(
                    ScheduledGame(
                        game_id=f"{pair_id}:leg-{leg}",
                        pair_id=pair_id,
                        pair_leg=leg,
                        opening_id=opening.opening_id,
                        start_fen=opening.fen,
                        seed=seed,
                        candidate_color=color,
                    )
                )
    return games


def _pair_groups(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pair_id = record.get("pair_id")
        if isinstance(pair_id, str) and pair_id:
            groups[pair_id].append(record)
    return dict(groups)


def paired_record_errors(records: Sequence[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for pair_id, pair_records in _pair_groups(records).items():
        if len(pair_records) > 2:
            errors.append(f"pair {pair_id}: must not contain more than two games")
        legs = [record.get("pair_leg") for record in pair_records]
        if any(leg not in (1, 2) for leg in legs):
            errors.append(f"pair {pair_id}: pair_leg must be 1 or 2")
        if len(legs) == 2 and legs[0] == legs[1]:
            errors.append(f"pair {pair_id}: pair legs must be unique")
        colors = [record.get("candidate_color") for record in pair_records]
        if any(color not in (RED, BLACK) for color in colors):
            errors.append(f"pair {pair_id}: candidate_color must be red or black")
        if len(colors) == 2 and colors[0] == colors[1]:
            errors.append(f"pair {pair_id}: candidate colors must be unique")
        for field in ("opening_id", "start_fen", "seed"):
            values = [record.get(field) for record in pair_records]
            if len(values) == 2 and values[0] != values[1]:
                errors.append(f"pair {pair_id}: {field} must match across both legs")
    return errors


def _complete_pairs(
    records: Sequence[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    complete: list[tuple[str, list[dict[str, Any]]]] = []
    for pair_id, pair_records in _pair_groups(records).items():
        if len(pair_records) != 2:
            continue
        legs = [record.get("pair_leg") for record in pair_records]
        if any(leg not in (1, 2) for leg in legs) or legs[0] == legs[1]:
            continue
        colors = [record.get("candidate_color") for record in pair_records]
        if any(color not in (RED, BLACK) for color in colors) or colors[0] == colors[1]:
            continue
        seeds = [record.get("seed") for record in pair_records]
        if any(not isinstance(seed, int) for seed in seeds) or seeds[0] != seeds[1]:
            continue
        if any(record.get("terminal_reason") == "max_plies" for record in pair_records):
            continue
        if any(record.get("terminal_score") not in (-1.0, 0.0, 1.0) for record in pair_records):
            continue
        complete.append((pair_id, pair_records))
    return complete


def _percentile_interval(
    values: np.ndarray, confidence: float
) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def _elo_from_score(score_rate: np.ndarray | float, games: int) -> np.ndarray | float:
    adjusted = (np.asarray(score_rate) * games + 0.5) / (games + 1.0)
    elo = 400.0 * np.log10(adjusted / (1.0 - adjusted))
    if np.ndim(elo) == 0:
        return float(elo)
    return elo


def summarize_paired_records(
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    groups = _pair_groups(records)
    complete = _complete_pairs(records)
    pair_scores = np.asarray(
        [
            np.mean([(float(record["terminal_score"]) + 1.0) / 2.0 for record in pair])
            for _, pair in complete
        ],
        dtype=np.float64,
    )
    pair_draw_rates = np.asarray(
        [np.mean([float(record["terminal_score"]) == 0.0 for record in pair]) for _, pair in complete],
        dtype=np.float64,
    )
    seeds: dict[str, dict[str, float | int | None]] = {}
    for seed in sorted({int(pair[0].get("seed", 0)) for _, pair in complete}):
        values = [
            pair_scores[index]
            for index, (_, pair) in enumerate(complete)
            if int(pair[0].get("seed", 0)) == seed
        ]
        seeds[str(seed)] = {
            "complete_pairs": len(values),
            "score_rate": float(np.mean(values)) if values else None,
        }

    result: dict[str, Any] = {
        "observed_pairs": len(groups),
        "complete_pairs": len(complete),
        "incomplete_pairs": len(groups) - len(complete),
        "complete_games": len(complete) * 2,
        "distinct_seeds": len(seeds),
        "by_seed": seeds,
        "bootstrap": {
            "method": "paired-cluster-percentile",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": confidence,
        },
        "score_rate": None,
        "score_rate_interval": None,
        "draw_rate": None,
        "draw_rate_interval": None,
        "elo_difference": None,
    }
    if not complete:
        return result

    rng = np.random.default_rng(bootstrap_seed)
    sample_indices = rng.integers(0, len(complete), size=(bootstrap_samples, len(complete)))
    bootstrap_scores = pair_scores[sample_indices].mean(axis=1)
    bootstrap_draws = pair_draw_rates[sample_indices].mean(axis=1)
    score_low, score_high = _percentile_interval(bootstrap_scores, confidence)
    draw_low, draw_high = _percentile_interval(bootstrap_draws, confidence)
    games = len(complete) * 2
    bootstrap_elo = np.asarray(_elo_from_score(bootstrap_scores, games))
    elo_low, elo_high = _percentile_interval(bootstrap_elo, confidence)
    result.update(
        {
            "score_rate": float(pair_scores.mean()),
            "score_rate_interval": {"low": score_low, "high": score_high},
            "draw_rate": float(pair_draw_rates.mean()),
            "draw_rate_interval": {"low": draw_low, "high": draw_high},
            "elo_difference": {
                "estimate": _elo_from_score(float(pair_scores.mean()), games),
                "low": elo_low,
                "high": elo_high,
                "method": "logistic-elo-with-beta-half-continuity-correction",
            },
        }
    )
    return result


def sequential_stop_decision(
    records: Sequence[dict[str, Any]],
    *,
    min_pairs: int,
    max_pairs: int,
    min_seeds: int = 2,
    confidence: float = 0.95,
    superiority_margin: float = 0.0,
) -> dict[str, Any]:
    if min_pairs < 1 or max_pairs < min_pairs:
        raise ValueError("Require 1 <= min_pairs <= max_pairs")
    if min_seeds < 1:
        raise ValueError("min_seeds must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if not 0.0 <= superiority_margin < 0.5:
        raise ValueError("superiority_margin must be in [0, 0.5)")

    groups = _pair_groups(records)
    complete = _complete_pairs(records)
    pair_scores = [
        sum((float(record["terminal_score"]) + 1.0) / 2.0 for record in pair) / 2.0
        for _, pair in complete
    ]
    seeds = {int(pair[0].get("seed", 0)) for _, pair in complete}
    attempted_pairs = len(groups)
    completed_pairs = len(complete)
    result: dict[str, Any] = {
        "stop": False,
        "decision": "continue",
        "reason": "minimum_evidence_not_met",
        "attempted_pairs": attempted_pairs,
        "completed_pairs": completed_pairs,
        "distinct_seeds": len(seeds),
        "score_rate": float(np.mean(pair_scores)) if pair_scores else None,
        "confidence_sequence": None,
    }

    evidence_ready = completed_pairs >= min_pairs and len(seeds) >= min_seeds
    if evidence_ready:
        alpha = 1.0 - confidence
        look_alpha = alpha * 6.0 / (math.pi**2 * completed_pairs**2)
        half_width = math.sqrt(math.log(2.0 / look_alpha) / (2.0 * completed_pairs))
        score_rate = float(np.mean(pair_scores))
        low = max(0.0, score_rate - half_width)
        high = min(1.0, score_rate + half_width)
        result["confidence_sequence"] = {
            "low": low,
            "high": high,
            "confidence": confidence,
            "method": "hoeffding-union-bound",
        }
        if low > 0.5 + superiority_margin:
            result.update(stop=True, decision="superior", reason="lower_bound_above_margin")
            return result
        if high < 0.5 - superiority_margin:
            result.update(stop=True, decision="inferior", reason="upper_bound_below_margin")
            return result
        result["reason"] = "bounds_overlap_margin"

    if attempted_pairs >= max_pairs:
        result.update(stop=True, decision="inconclusive", reason="maximum_pairs_reached")
    return result
