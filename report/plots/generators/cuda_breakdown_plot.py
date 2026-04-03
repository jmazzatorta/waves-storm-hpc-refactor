import matplotlib.pyplot as plt
import numpy as np
from data import COLORS, setup_style, cuda_breakdown_data

setup_style()


def main():
    labels = ["30K", "1M"]
    d30k = cuda_breakdown_data["30K"]
    d1m = cuda_breakdown_data["1M"]
    kernels = cuda_breakdown_data["labels"]

    # Colours from data.py palette
    slice_colors = [
        COLORS["Navy"],      # storm_kernel
        COLORS["Red"],       # reduce_max
        COLORS["Blue"],      # stencil
        COLORS["Amber"],     # H2D
        COLORS["Green"],     # D2H
        COLORS["PaleInk"],   # Memset
    ]

    x = np.arange(len(labels))
    width = 0.45

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    # Stacked bars
    bottoms = np.zeros(len(labels))
    bars_all = []
    for i, kernel in enumerate(kernels):
        values = np.array([d30k[i], d1m[i]])
        b = ax.bar(x, values, width, bottom=bottoms,
                   label=kernel, color=slice_colors[i],
                   edgecolor="white", linewidth=0.3, zorder=3)
        bars_all.append((kernel, values, bottoms.copy()))
        bottoms += values

    ax.set_ylabel("Execution time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.2, axis="y", linestyle="-")

    # Percentage annotation on storm_kernel bar
    tot_30k = sum(d30k)
    tot_1m = sum(d1m)
    for j, (tot, val) in enumerate([(tot_30k, d30k[0]), (tot_1m, d1m[0])]):
        pct = val / tot * 100
        ax.text(x[j], val / 2, f"{pct:.1f}%",
                ha="center", va="center", fontsize=6.5,
                color="white", fontweight="bold")

    # Total time on top
    for j, tot in enumerate([tot_30k, tot_1m]):
        ax.text(x[j], tot, f"{tot:.1f} ms",
                ha="center", va="bottom", fontsize=5.5,
                color=COLORS["MidInk"])

    ax.legend(frameon=True, fancybox=False, edgecolor=COLORS["Mist"],
              fontsize=5.5, loc="upper left", ncol=2)

    plt.tight_layout()
    plt.savefig("cuda_breakdown.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("cuda_breakdown.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("✓ cuda_breakdown.pdf / .png saved")


if __name__ == "__main__":
    main()
