"""Charts for the README, drawn from results/benchmarks.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=20, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def _label_end(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)


def decode_latency(results: dict, path: Path) -> Path:
    rows = results["decode_step"]
    x = [r["context"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for key, label, color in (("no_cache_ms", "No cache", SERIES[1]), ("kv_cache_ms", "KV cache", SERIES[0])):
        y = [r[key] for r in rows]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, label=label)
        _label_end(ax, x[-1], y[-1], f"{label}: {y[-1]:.0f} ms")
    _style(ax, "Time to generate one token as the context grows", "Tokens already in context", "Milliseconds per token")
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=x[-1] * 1.25)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def batching(results: dict, path: Path) -> Path:
    rows = results["batching"]
    labels = [str(r["batch"]) for r in rows]
    y = [r["tok_s"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, y, color=SERIES[0], width=0.6, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, y):
        ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    _style(ax, "Total throughput when decoding requests together", "Batch size (requests)", "Tokens per second (all requests)")
    ax.set_ylim(0, max(y) * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def speculative(results: dict, path: Path) -> Path:
    spec = results["speculative"]
    alphas = spec["alphas"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for color, (k, ys) in zip(SERIES, spec["expected"].items()):
        ax.plot(alphas, ys, color=color, linewidth=2, label=f"k = {k} draft tokens")
        _label_end(ax, alphas[-1], ys[-1], f"k={k}: x{ys[-1]:.1f}")
    for row in spec.get("measured", []):
        ax.plot(row["acceptance_rate"], row["speedup"], marker="o", markersize=8,
                color=INK, markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none")
        _label_end(ax, row["acceptance_rate"], row["speedup"], f"measured k={row['k']}")
    ax.axhline(1.0, color=INK_2, linewidth=1, linestyle="--")
    ax.annotate("no speedup", (alphas[0], 1.0), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK_2)
    _style(ax, "Speculative decoding: expected speedup", "Share of draft tokens the target accepts",
           "Speedup over normal decoding")
    ax.text(0, 1.01, f"{spec['draft']} drafting for {spec['target']}; draft step costs "
            f"{spec['cost_ratio']:.0%} of a target step",
            transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
    ax.set_xlim(0, 1.12)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_all(results: dict, out: Path) -> list[Path]:
    out = Path(out)
    return [
        decode_latency(results, out / "decode_latency.png"),
        batching(results, out / "batching.png"),
        speculative(results, out / "speculative_speedup.png"),
    ]


if __name__ == "__main__":
    import sys

    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    for p in make_all(json.loads((folder / "benchmarks.json").read_text()), folder):
        print(p)
