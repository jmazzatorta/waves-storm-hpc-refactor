import matplotlib.pyplot as plt
import numpy as np
from data import COLORS, setup_style, base_1m, p32_1m, p64_1m, p128_1m, c_1m

setup_style()

# Original sequential time from report (Section 2.3 insight box)
SEQ_1M = 278.8


def main():
    labels = [
        "Seq.\n(orig.)",
        "Baseline\n(optim.)",
        "MPI\n1 node",
        "MPI\n2 nodes",
        "MPI\n4 nodes",
        "CUDA",
    ]
    times = [
        SEQ_1M,
        base_1m,
        p32_1m[0],   # median
        p64_1m[0],
        p128_1m[0],
        c_1m[0],
    ]
    colors = [
        COLORS["PaleInk"],    # sequential
        COLORS["LightBlue"],  # baseline
        COLORS["Blue"],       # MPI 1 node
        COLORS["Navy"],       # MPI 2 nodes
        COLORS["DarkNavy"],   # MPI 4 nodes
        COLORS["Purple"],     # CUDA
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    bars = ax.bar(labels, times, color=colors, edgecolor="white", linewidth=0.4)

    ax.set_ylabel("Execution time (s)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2, axis="y", linestyle="-")
    ax.tick_params(axis="x", labelsize=5.5)

    # Value labels on top of each bar
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.2f}" if t < 10 else f"{t:.0f}",
                ha="center", va="bottom", fontsize=5.5, color=COLORS["MidInk"])

    plt.tight_layout()
    plt.savefig("time_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("time_comparison.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("✓ time_comparison.pdf / .png saved")


if __name__ == "__main__":
    main()
