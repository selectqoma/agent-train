import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_scores(csv_path: str) -> list[float]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [float(row["score"]) for row in csv.DictReader(f)]


def save_plot(scores: list[float], path: str) -> None:
    if not scores:
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    x = list(range(1, len(scores) + 1))
    ax.scatter(x, scores, color="#c8c8c8", s=22, zorder=2, alpha=0.8, linewidths=0)

    window = min(15, len(scores))
    if len(scores) >= window:
        rolling = np.convolve(scores, np.ones(window) / window, mode="valid")
        rolling_x = list(range(window, len(scores) + 1))
        ax.plot(
            rolling_x,
            rolling,
            color="#111111",
            linewidth=2.2,
            zorder=3,
            label=f"Rolling avg ({window})",
        )

    ax.set_xlabel("Iteration", fontsize=11, color="#444")
    ax.set_ylabel("Score / 10", fontsize=11, color="#444")
    ax.set_title("Agent voice alignment over training iterations", fontsize=13, fontweight="bold", color="#111", pad=14)
    ax.set_ylim(0, 10.8)
    ax.set_xlim(0, len(scores) + 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(axis="y", alpha=0.25, color="#bbbbbb")
    ax.grid(axis="y", which="minor", alpha=0.1, color="#bbbbbb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dddddd")
    ax.spines["bottom"].set_color("#dddddd")
    ax.tick_params(colors="#666")
    ax.legend(fontsize=10, framealpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
