#!/usr/bin/env python
"""Draw the figures the report needs, from the measured artifacts.

    python scripts/make_figures.py

The script reads results/sweep.json, results/calibration.txt and
results/summary.json and writes PNG files into docs/figures, which GitHub
Pages serves. Nothing is simulated: every point comes from the run that wrote
those files, so rerunning the sweep and then this script keeps the figures and
the repository in agreement.

matplotlib is an optional dependency (``pip install -e '.[report]'``) because
the figures are documentation, not part of the measurement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"

INK = "#1f2933"
ACCENT = "#1f6f8b"
MUTED = "#8a94a6"
GRID = "#d7dde5"


@dataclass(frozen=True)
class Distribution:
    """The similarity percentiles of one group of questions."""

    label: str
    count: int
    values: tuple[float, ...]


def _read_sweep() -> list[dict]:
    return json.loads((RESULTS / "sweep.json").read_text(encoding="utf-8"))


def _read_summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))


def _read_distributions(text: str) -> list[Distribution]:
    """Pull the two percentile rows out of the calibration report."""

    line_pattern = re.compile(
        r"^(paraphrase|negative)\s+n=\s*(\d+)\s+(?:p\d+=[\d.]+\s+)*"
        r"p25=([\d.]+)\s+p50=([\d.]+)\s+p75=([\d.]+)\s+p90=([\d.]+)\s+p100=([\d.]+)"
    )
    labels = {"paraphrase": "paraphrases", "negative": "negatives"}
    rows: list[Distribution] = []
    for line in text.splitlines():
        match = line_pattern.match(line.strip())
        if match:
            rows.append(
                Distribution(
                    label=labels[match.group(1)],
                    count=int(match.group(2)),
                    values=tuple(float(value) for value in match.groups()[2:]),
                )
            )
    return rows


def _style(axis) -> None:
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK, labelsize=10)


def tradeoff_figure(sweep: list[dict], summary: dict) -> None:
    """Hit rate and average latency, one point per threshold."""

    thresholds = [row["threshold"] for row in sweep]
    hit_rates = [row["hit_rate"] * 100.0 for row in sweep]
    false_hits = [row["false_hits"] for row in sweep]
    latencies = [row["avg_latency_ms"] for row in sweep]
    chosen = summary["threshold"]
    baseline = summary["baseline_avg_latency_ms"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))

    left.plot(thresholds, hit_rates, color=ACCENT, linewidth=2, marker="o", markersize=5)
    for threshold, rate, false in zip(thresholds, hit_rates, false_hits, strict=True):
        if false == 0:
            left.plot(
                threshold,
                rate,
                marker="o",
                markersize=11,
                markerfacecolor="none",
                markeredgecolor=ACCENT,
                markeredgewidth=1.2,
            )
    left.axvline(chosen, color=MUTED, linewidth=1, linestyle="--")
    left.annotate(
        f"operating point {chosen:.2f}",
        xy=(chosen, max(hit_rates)),
        xytext=(6, -16),
        textcoords="offset points",
        color=MUTED,
        fontsize=9,
    )
    left.set_xlabel("similarity threshold")
    left.set_ylabel("questions answered from the cache (%)")
    left.set_title("Hit rate", color=INK, fontsize=12, loc="left")
    left.set_ylim(bottom=0)
    _style(left)

    right.bar([f"{number:.2f}" for number in thresholds], latencies, color=ACCENT, width=0.55)
    right.axhline(baseline, color=INK, linewidth=1.2)
    right.annotate(
        f"baseline {baseline:.0f} ms",
        xy=(0, baseline),
        xytext=(4, 5),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )
    right.set_xlabel("similarity threshold")
    right.set_ylabel("average latency (ms)")
    right.set_title("Latency", color=INK, fontsize=12, loc="left")
    right.set_ylim(bottom=0)
    _style(right)

    figure.text(
        0.01,
        -0.02,
        "100 test questions against 30 FAQ entries, one machine, threshold sweep from "
        "results/sweep.json",
        color=MUTED,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "threshold_tradeoff.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def similarity_figure(distributions: list[Distribution], threshold: float) -> None:
    """Where the paraphrase and negative similarity distributions separate."""

    figure, axis = plt.subplots(figsize=(8, 3.8))
    positions = range(len(distributions))
    for position, row in zip(positions, distributions, strict=True):
        colour = ACCENT if row.label == "paraphrases" else MUTED
        axis.plot(row.values, [position] * len(row.values), color=colour, linewidth=2, alpha=0.55)
        axis.plot(row.values, [position] * len(row.values), "o", color=colour, markersize=5)
        axis.annotate(
            f"median {row.values[1]:.3f}",
            xy=(row.values[1], position),
            xytext=(-6, 12),
            textcoords="offset points",
            color=colour,
            fontsize=9,
            ha="right",
        )
    axis.axvline(threshold, color=INK, linewidth=1.2, linestyle="--")
    axis.annotate(
        f"threshold {threshold:.2f}",
        xy=(threshold, len(distributions) - 0.5),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK,
        fontsize=10,
    )
    axis.set_yticks(list(positions))
    axis.set_yticklabels(
        [f"{row.label} (n={row.count})" for row in distributions], color=INK
    )
    axis.set_xlabel("similarity to the FAQ a question is measured against")
    axis.set_xlim(0, 1.0)
    axis.set_ylim(-0.6, len(distributions) - 0.3)
    _style(axis)
    figure.text(
        0.01,
        -0.02,
        "percentiles from results/calibration.txt; the strictest threshold with no "
        "false hit sits at 0.70",
        color=MUTED,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "similarity_separation.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    sweep = _read_sweep()
    summary = _read_summary()
    distributions = _read_distributions((RESULTS / "calibration.txt").read_text(encoding="utf-8"))
    if len(distributions) != 2:
        print("results/calibration.txt does not carry both percentile rows")
        return 1

    FIGURES.mkdir(parents=True, exist_ok=True)
    tradeoff_figure(sweep, summary)
    similarity_figure(distributions, summary["threshold"])
    print(f"wrote {FIGURES / 'threshold_tradeoff.png'}")
    print(f"wrote {FIGURES / 'similarity_separation.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
