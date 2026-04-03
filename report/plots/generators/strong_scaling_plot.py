import matplotlib.pyplot as plt
import numpy as np
from data import COLORS, setup_style, strong_data

setup_style()


def main():
    cores = np.array(strong_data["cores"])

    # Series: key, color, marker, label, dodge factor at 128, annotation offset
    # Dodge: multiplicative shift on x at the last point (log scale)
    series = [
        ("30K",  COLORS["Blue"],  "o", "30 K",  1.03),
        ("120K", COLORS["Navy"],  "s", "120 K", 1.00),
        ("1M",   COLORS["Red"],   "D", "1 M",   0.97),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    # Ideal line
    ax.plot(cores, cores, "--", color=COLORS["PaleInk"], linewidth=0.6,
            label="Ideal", zorder=1)

    for key, col, mk, lbl, dodge in series:
        meds = [s[0] for s in strong_data[key]]
        errs_lo = [s[1][0] for s in strong_data[key]]
        errs_hi = [s[1][1] for s in strong_data[key]]

        # Apply horizontal dodge only at the last point (128 cores)
        x_pts = cores.astype(float).copy()
        x_pts[-1] = cores[-1] * dodge

        ax.errorbar(x_pts, meds,
                    yerr=[errs_lo, errs_hi],
                    fmt=f"{mk}-", color=col, markeredgecolor=col,
                    markerfacecolor=col, markersize=3, linewidth=0.9,
                    capsize=2, capthick=0.5, elinewidth=0.5,
                    label=lbl, zorder=3)

    ax.set_xlabel("Number of cores")
    ax.set_ylabel("Speedup  ($S_p = T_s / T_p$)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(cores)
    ax.set_xticklabels([str(c) for c in cores])
    ax.set_yticks([4, 8, 16, 32, 64, 128])
    ax.set_yticklabels(["4", "8", "16", "32", "64", "128"])
    ax.set_xlim(24, 180)
    ax.set_ylim(3, 180)

    ax.legend(frameon=True, fancybox=False, edgecolor=COLORS["Mist"],
              fontsize=7, loc="upper left")

    # Annotate Amdahl f — right margin, aligned to each series' y
    for key, col, mk, lbl, dodge in series:
        s128 = strong_data[key][2][0]
        f_pct = strong_data["amdahl_f"][key]
        ax.text(1.02, s128, f"$f$={f_pct:.1f}%",
                transform=ax.get_yaxis_transform(),
                fontsize=5.5, color=col, va="center", clip_on=False)

    plt.tight_layout()
    plt.savefig("strong_scaling.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("strong_scaling.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("✓ strong_scaling.pdf / .png saved")


if __name__ == "__main__":
    main()
