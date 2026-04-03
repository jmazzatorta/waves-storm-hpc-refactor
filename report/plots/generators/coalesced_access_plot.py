import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data import COLORS, setup_style

setup_style()


def main():
    # Thread colours — from data.py palette
    thread_palette = [COLORS["Red"], COLORS["Blue"], COLORS["Green"], COLORS["Amber"]]
    thread_light   = ["#f5b7b1", "#aed6f1", "#a9dfbf", "#f9e79f"]

    n_cells = 16
    n_threads = 4
    stride = n_threads

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-3.2, 11)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Title ──
    ax.text(8, 10.5, "Warp — 4 threads,  stride = 4",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=COLORS["Ink"], fontfamily="serif")

    # ── Thread boxes ──
    thread_y, thread_h, thread_w = 8.2, 1.3, 3.0
    thread_gap = 0.7
    thread_total_w = n_threads * thread_w + (n_threads - 1) * thread_gap
    thread_x0 = (n_cells - thread_total_w) / 2

    for t in range(n_threads):
        x = thread_x0 + t * (thread_w + thread_gap)
        rect = patches.FancyBboxPatch(
            (x, thread_y), thread_w, thread_h,
            boxstyle="round,pad=0.10", linewidth=0.9,
            edgecolor=thread_palette[t], facecolor=thread_palette[t], alpha=0.92,
        )
        ax.add_patch(rect)
        ax.text(x + thread_w / 2, thread_y + thread_h / 2,
                f"Thread {t}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", fontfamily="serif")

    # ── Memory cells ──
    cell_y, cell_h, cell_w, cell_gap = 2.5, 1.1, 0.9, 0.1

    for i in range(n_cells):
        x = i * (cell_w + cell_gap)
        owner = i % stride
        rect = patches.FancyBboxPatch(
            (x, cell_y), cell_w, cell_h,
            boxstyle="round,pad=0.04", linewidth=0.7,
            edgecolor=thread_palette[owner], facecolor=thread_light[owner],
        )
        ax.add_patch(rect)
        ax.text(x + cell_w / 2, cell_y + cell_h / 2, str(i),
                ha="center", va="center", fontsize=7, fontweight="bold",
                color=COLORS["Ink"], fontfamily="serif")

    # ── Arrows: thread → owned cells ──
    arrow_from_y = thread_y - 0.05
    arrow_to_y = cell_y + cell_h + 0.05

    for t in range(n_threads):
        tcx = thread_x0 + t * (thread_w + thread_gap) + thread_w / 2
        for ci in range(t, n_cells, stride):
            ccx = ci * (cell_w + cell_gap) + cell_w / 2
            ax.annotate("",
                        xy=(ccx, arrow_to_y), xytext=(tcx, arrow_from_y),
                        arrowprops=dict(arrowstyle="-|>", color=thread_palette[t],
                                        lw=0.7, alpha=0.55, mutation_scale=7))

    # ── Coalesced-transaction brackets ──
    bracket_y = cell_y - 0.15
    bw = 4 * (cell_w + cell_gap) - cell_gap

    for blk in range(4):
        x_start = blk * 4 * (cell_w + cell_gap)
        ax.annotate("", xy=(x_start, bracket_y), xytext=(x_start + bw, bracket_y),
                    arrowprops=dict(arrowstyle="|-|", color=COLORS["MidInk"],
                                    lw=0.7, mutation_scale=4))
        ax.text(x_start + bw / 2, bracket_y - 0.45,
                "1 coalesced transaction", ha="center", fontsize=5.5,
                color=COLORS["MidInk"], style="italic", fontfamily="serif")

    # ── Global label below all brackets ──
    ax.text(n_cells * (cell_w + cell_gap) / 2, bracket_y - 1.2,
            "Global Memory Addresses", ha="center", fontsize=7,
            color=COLORS["LightInk"], style="italic", fontfamily="serif")

    plt.tight_layout()
    plt.savefig("cuda_coalesced.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("cuda_coalesced.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("✓ cuda_coalesced.pdf / .png saved")


if __name__ == "__main__":
    main()
