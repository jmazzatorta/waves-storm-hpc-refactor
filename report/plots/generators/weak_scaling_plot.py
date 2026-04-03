import matplotlib.pyplot as plt
import numpy as np
from data import COLORS, setup_style, weak_data

setup_style()


def main():
    nodes = np.array(weak_data["nodes"])

    # Series: key prefix, color, marker, label, dodge
    series = [
        ("prog1", COLORS["Blue"], "o", "Prog. 1  (30 K → 120 K)", 0.97),
        ("prog2", COLORS["Red"],  "s", "Prog. 2  (1 M → 4 M)",   1.03),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    for prefix, col, mk, lbl, dodge in series:
        meds = weak_data[f"{prefix}_meds"]
        errs = weak_data[f"{prefix}_errs"]

        x_pts = nodes.astype(float).copy()
        # Dodge only at multi-point positions (2 and 4 nodes)
        x_pts[1] = nodes[1] * dodge
        x_pts[2] = nodes[2] * dodge

        ax.errorbar(x_pts, meds,
                    yerr=errs,
                    fmt=f"{mk}-", color=col, markeredgecolor=col,
                    markerfacecolor=col, markersize=3, linewidth=0.9,
                    capsize=2, capthick=0.5, elinewidth=0.5,
                    label=lbl, zorder=3)

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Execution time (s)")

    ax.set_xticks(nodes)
    ax.set_xticklabels([str(n) for n in nodes])
    ax.set_xlim(0.5, 4.8)

    ax.legend(frameon=True, fancybox=False, edgecolor=COLORS["Mist"],
              fontsize=6.5, loc="upper right")

    plt.tight_layout()
    plt.savefig("weak_scaling.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("weak_scaling.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("✓ weak_scaling.pdf / .png saved")


if __name__ == "__main__":
    main()
