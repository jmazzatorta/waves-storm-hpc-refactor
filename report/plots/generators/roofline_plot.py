import matplotlib.pyplot as plt
import numpy as np
from data import COLORS, setup_style, roofline_data

setup_style()


def main():
    peak_flops = roofline_data["peak_flops"]   # 16300 GFLOP/s
    peak_bw = roofline_data["peak_bw"]         # 384 GB/s
    pts = roofline_data["points"]

    # Bandwidths
    dram_bw = peak_bw
    l2_bw = roofline_data["l2_bw"]

    # Ridge points (where memory-bound meets compute-bound)
    ridge_dram = peak_flops / dram_bw    # ~42.4 FLOP/byte
    ridge_l2   = peak_flops / l2_bw      # ~8.15 FLOP/byte

    # Roofline curves
    ai = np.logspace(-1, 5, 1000)
    perf_dram = np.minimum(ai * dram_bw, peak_flops)
    perf_l2   = np.minimum(ai * l2_bw,   peak_flops)

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    # L2 ceiling (upper envelope)
    ax.plot(ai, perf_l2, color=COLORS["Blue"], linewidth=0.8,
            linestyle="-", zorder=2, label=f"L2 cache (~{l2_bw} GB/s)")

    # DRAM ceiling (lower)
    ax.plot(ai, perf_dram, color=COLORS["Navy"], linewidth=0.8,
            linestyle="--", zorder=2, label=f"DRAM ({dram_bw} GB/s)")

    # Compute ceiling label
    ax.text(ridge_l2 * 3, peak_flops * 0.72,
            f"{peak_flops / 1000:.1f} TFLOP/s", color=COLORS["Navy"],
            fontsize=6, fontstyle="italic")

    # Kernel points (analytical model, no errorbar on AI)
    ax.plot(pts["30K"]["ai"], pts["30K"]["perf"], "o",
            color=COLORS["Red"], markersize=4, zorder=4,
            label=f'30 K  (AI\u2248{pts["30K"]["ai"]:.0f},  {pts["30K"]["perf"]:.0f} GF/s)')

    ax.plot(pts["1M"]["ai"], pts["1M"]["perf"], "^",
            color=COLORS["Purple"], markersize=4, zorder=4,
            label=f'1 M   (AI\u2248{pts["1M"]["ai"]:.0f},  {pts["1M"]["perf"]:.0f} GF/s)')

    # Annotate % of peak
    for key, va_off in [("30K", -1), ("1M", 1)]:
        pct = pts[key]["perf"] / peak_flops * 100
        ax.annotate(f"{pct:.1f}% peak",
                    xy=(pts[key]["ai"], pts[key]["perf"]),
                    xytext=(0, 12 if va_off > 0 else -14),
                    textcoords="offset points",
                    fontsize=5.5, color=COLORS["MidInk"],
                    ha="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 50000)
    ax.set_ylim(50, 25000)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOP/s)")

    ax.legend(frameon=True, fancybox=False, edgecolor=COLORS["Mist"],
              fontsize=5, loc="lower right")

    plt.tight_layout()
    plt.savefig("roofline.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("roofline.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("\u2713 roofline.pdf / .png saved")


if __name__ == "__main__":
    main()
