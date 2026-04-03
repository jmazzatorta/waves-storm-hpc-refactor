import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data import COLORS, setup_style

setup_style()

# ── Palette (from data.py + tex accents) ──
NAVY      = COLORS["Navy"]
BLUE      = COLORS["Blue"]
LIGHTBLUE = COLORS["LightBlue"]
DARKNAVY  = COLORS["DarkNavy"]
RED       = COLORS["Red"]
GREEN     = COLORS["Green"]
AMBER     = COLORS["Amber"]
PURPLE    = COLORS["Purple"]
INK       = COLORS["Ink"]
MIDK      = COLORS["MidInk"]
LIGHTINK  = COLORS["LightInk"]
PALEINK   = COLORS["PaleInk"]
MIST      = COLORS["Mist"]
SNOW      = COLORS["Snow"]
GRIDGRAY  = COLORS["GridGray"]

# Rank colours: four distinct hues from the palette
rank_colors = [NAVY, BLUE, GREEN, AMBER]

# Lighter tints for OMP threads (per rank)
omp_tints = [
    ["#c8dae8", "#a4c4da", "#7faecb"],   # Navy tints
    ["#bdd8eb", "#94c2de", "#6babd1"],   # Blue tints
    ["#c3e6c4", "#9fd4a0", "#7cc27d"],   # Green tints
    ["#f0ddb8", "#e6c88e", "#dbb364"],   # Amber tints
]

# ── Figure ──
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.set_xlim(-1, 101)
ax.set_ylim(-1.5, 16)
ax.axis("off")
fig.patch.set_facecolor("white")

# ══════════════════════════════════════════
# Global Layer bar
# ══════════════════════════════════════════
layer_y, layer_h = 13.8, 1.0
rect_global = patches.FancyBboxPatch(
    (0, layer_y), 100, layer_h,
    boxstyle="round,pad=0.12",
    linewidth=0.8, edgecolor=LIGHTINK, facecolor=MIST,
)
ax.add_patch(rect_global)
ax.text(50, layer_y + layer_h / 2,
        "Global Layer  [0 … N−1]",
        ha="center", va="center", fontsize=8, fontweight="bold",
        color=INK, fontfamily="serif")

# ══════════════════════════════════════════
# Node boundary (dashed)
# ══════════════════════════════════════════
node_y, node_h = 0.6, 11.5
node_rect = patches.FancyBboxPatch(
    (-0.3, node_y), 100.6, node_h,
    boxstyle="round,pad=0.25",
    linewidth=0.7, edgecolor=PALEINK, facecolor="none",
    linestyle="--",
)
ax.add_patch(node_rect)
ax.text(50, node_y + node_h - 0.35,
        "Node  (2 sockets · 4 NUMA domains · 32 cores)",
        ha="center", va="center", fontsize=7,
        color=LIGHTINK, style="italic", fontfamily="serif")

# ══════════════════════════════════════════
# MPI Rank boxes
# ══════════════════════════════════════════
rank_y, rank_h = 7.2, 2.2
rank_w = 23.5
rank_gap = 1.0
numa_labels = ["NUMA 0", "NUMA 1", "NUMA 2", "NUMA 3"]

for i in range(4):
    x0 = i * (rank_w + rank_gap) + 0.5
    col = rank_colors[i]
    rect = patches.FancyBboxPatch(
        (x0, rank_y), rank_w, rank_h,
        boxstyle="round,pad=0.15",
        linewidth=1.2, edgecolor=col, facecolor=col, alpha=0.92,
    )
    ax.add_patch(rect)
    ax.text(x0 + rank_w / 2, rank_y + rank_h / 2 + 0.35,
            f"MPI Rank {i}", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white", fontfamily="serif")
    ax.text(x0 + rank_w / 2, rank_y + rank_h / 2 - 0.45,
            numa_labels[i], ha="center", va="center",
            fontsize=6.5, color="#ffffffbb", fontfamily="serif")

    # Arrow: rank → global layer
    ax.annotate("",
                xy=(x0 + rank_w / 2, layer_y - 0.05),
                xytext=(x0 + rank_w / 2, rank_y + rank_h + 0.05),
                arrowprops=dict(arrowstyle="-|>", color=PALEINK,
                                lw=0.8, mutation_scale=8))

# ══════════════════════════════════════════
# OMP Thread boxes
# ══════════════════════════════════════════
thread_y, thread_h = 1.5, 3.2
n_threads = 3

for ri in range(4):
    rx0 = ri * (rank_w + rank_gap) + 0.5
    tw = (rank_w - (n_threads - 1) * 0.5) / n_threads

    for t in range(n_threads):
        x0 = rx0 + t * (tw + 0.5)
        fc = omp_tints[ri][t]
        ec = rank_colors[ri]

        rect = patches.FancyBboxPatch(
            (x0, thread_y), tw, thread_h,
            boxstyle="round,pad=0.08",
            linewidth=0.7, edgecolor=ec, facecolor=fc, alpha=0.88,
        )
        ax.add_patch(rect)
        ax.text(x0 + tw / 2, thread_y + thread_h / 2 + 0.35,
                f"T{t}", ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=INK, fontfamily="serif")
        ax.text(x0 + tw / 2, thread_y + thread_h / 2 - 0.55,
                f"[{t}C .. {t+1}C)",
                ha="center", va="center",
                fontsize=5.5, color=MIDK, fontfamily="monospace")

    # Arrow: threads block → rank
    ax.annotate("",
                xy=(rx0 + rank_w / 2, rank_y - 0.05),
                xytext=(rx0 + rank_w / 2, thread_y + thread_h + 0.05),
                arrowprops=dict(arrowstyle="-|>", color=PALEINK,
                                lw=0.8, mutation_scale=8))

# ── Footer note ──
ax.text(50, -0.7,
        "C = chunk size per thread  (aligned to 64-byte cache-line boundaries)",
        ha="center", va="center", fontsize=6.5,
        color=LIGHTINK, style="italic", fontfamily="serif")

plt.tight_layout()
plt.savefig("domain_decomposition.pdf", bbox_inches="tight", dpi=300)
plt.savefig("domain_decomposition.png", bbox_inches="tight", dpi=300)
plt.close()
print("✓ domain_decomposition.pdf / .png saved")
