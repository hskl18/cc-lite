from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ReplaySample:
    fen: str
    policy: dict[int, float]
    value: float


def write_jsonl(path: str | Path, samples: Iterable[ReplaySample], append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path, limit: int | None = None) -> list[ReplaySample]:
    samples: list[ReplaySample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            samples.append(
                ReplaySample(
                    fen=raw["fen"],
                    policy={int(k): float(v) for k, v in raw["policy"].items()},
                    value=float(raw["value"]),
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples

