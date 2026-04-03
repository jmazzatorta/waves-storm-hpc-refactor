import matplotlib.pyplot as plt
import numpy as np
from data import COLORS, setup_style, base_1m, p32_1m, p64_1m, p128_1m, c_1m

setup_style()

SEQ_1M = 278.8


def main():
    labels = [
        "Baseline\n(optim.)",
        "MPI\n1 node\n(32 cores)",
        "MPI\n2 nodes\n(64 cores)",
        "MPI\n4 nodes\n(128 cores)",
        "CUDA\n(1 GPU)",
    ]
    medians = [base_1m, p32_1m[0], p64_1m[0], p128_1m[0], c_1m[0]]
    speedups = [SEQ_1M / m for m in medians]

    colors = [
        COLORS["LightBlue"],
        COLORS["Blue"],
        COLORS["Navy"],
        COLORS["DarkNavy"],
        COLORS["Purple"],
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    bars = ax.bar(labels, speedups, color=colors, edgecolor="white", linewidth=0.3)

    ax.set_ylabel("Speedup over sequential")
    ax.grid(True, alpha=0.2, axis="y", linestyle="-")
    ax.tick_params(axis="x", labelsize=5.5)

    # Value labels
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{s:.0f}×", ha="center", va="bottom",
                fontsize=5.5, fontweight="bold", color=COLORS["MidInk"])

    plt.tight_layout()
    plt.savefig("speedup_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("speedup_comparison.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("✓ speedup_comparison.pdf / .png saved")


if __name__ == "__main__":
    main()
