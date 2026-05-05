#!/usr/bin/env python3
"""Generate a PNG bar chart comparing BC vs BC+ACT success (pick-and-place test tube)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path("test_tube_pick_place_success.png"),
        help="Output PNG path",
    )
    p.add_argument("--bc-pct", type=float, default=10.0, help="BC success rate (%)")
    p.add_argument(
        "--bc-act-pct",
        type=float,
        default=10.0,
        help="BC w/ ACT success rate (%)",
    )
    args = p.parse_args()

    bc = max(0.0, min(100.0, args.bc_pct))
    bc_act = max(0.0, min(100.0, args.bc_act_pct))

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["BC", "BC w/ ACT"]
    values = [bc, bc_act]
    colors = ["#1565c0", "#2e7d32"]

    bars = ax.bar(
        labels, values, color=colors, edgecolor="black", linewidth=1.0, width=0.5
    )
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Pick-and-place: test tube\nPolicy comparison",
        fontsize=13,
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{v:.0f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="medium",
        )

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
