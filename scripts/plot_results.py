#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Draw the README's results chart from the committed evaluation bundles.

Run `uv run scripts/plot_results.py` to rewrite docs/media/results.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/media/results.png"
OPPONENTS = (
    ("random", "Random\nmover"),
    ("material", "Material\nmover"),
    ("pikafish-depth-1", "Pikafish\ndepth 1"),
    ("pikafish-depth-2", "Pikafish\ndepth 2"),
    ("pikafish-depth-4", "Pikafish\ndepth 4"),
)
CHECKPOINTS = (("distill-v1", "#4c78a8"), ("distill-v2", "#f58518"))
WIDTH = 0.38

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
figure, axis = plt.subplots(figsize=(8.4, 3.4), facecolor="white")
for index, (checkpoint, color) in enumerate(CHECKPOINTS):
    results = [
        json.loads((ROOT / "evidence" / checkpoint / run / "summary.json").read_text())["terminal"]
        for run, _ in OPPONENTS
    ]
    offsets = [position + (index - 0.5) * WIDTH for position in range(len(OPPONENTS))]
    scores = [100 * (result["wins"] + result["draws"] / 2) / result["games"] for result in results]
    bars = axis.bar(offsets, scores, width=WIDTH, label=checkpoint, color=color)
    axis.bar_label(
        bars, labels=[f"{r['wins']}-{r['draws']}-{r['losses']}" for r in results], fontsize=9
    )
axis.set_xticks(range(len(OPPONENTS)), [label for _, label in OPPONENTS])
axis.set_ylabel("Score over 20 games")
axis.set_ylim(0, 112)
axis.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
figure.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(OUTPUT, dpi=160, metadata={"Software": None})
