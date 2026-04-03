import matplotlib.pyplot as plt
import numpy as np
from data import COLORS, setup_style, perf_comparison_data

setup_style()


def main():
    metrics  = perf_comparison_data["metrics"]
    baseline = perf_comparison_data["baseline"]
    parallel = perf_comparison_data["parallel"]

    y = np.arange(len(metrics))
    h = 0.32

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    bars_b = ax.barh(y + h / 2, baseline, h,
                     label="Baseline", color=COLORS["LightBlue"],
                     edgecolor="white", linewidth=0.3, zorder=3)
    bars_p = ax.barh(y - h / 2, parallel, h,
                     label="Parallel (32 cores)", color=COLORS["Navy"],
                     edgecolor="white", linewidth=0.3, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=6.5)
    ax.set_xlabel("Value")
    ax.set_xlim(0, max(max(baseline), max(parallel)) * 1.25)
    ax.grid(True, alpha=0.2, axis="x", linestyle="-")

    # Value annotations
    def annotate_bars(bars, values):
        for bar, v in zip(bars, values):
            fmt = f"{v:.2f}" if v < 10 else f"{v:.1f}"
            ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                    fmt, va="center", fontsize=5.5, color=COLORS["MidInk"])

    annotate_bars(bars_b, baseline)
    annotate_bars(bars_p, parallel)

    ax.legend(frameon=True, fancybox=False, edgecolor=COLORS["Mist"],
              fontsize=6.5, loc="lower right")

    plt.tight_layout()
    plt.savefig("perf_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("perf_comparison.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("✓ perf_comparison.pdf / .png saved")


if __name__ == "__main__":
    main()
