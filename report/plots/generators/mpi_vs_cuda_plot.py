import matplotlib.pyplot as plt
import numpy as np
from data import COLORS, setup_style, mpi_vs_cuda_data

setup_style()


def main():
    sizes = mpi_vs_cuda_data["sizes"]
    mpi_meds = [d[0] for d in mpi_vs_cuda_data["mpi_best"]]
    mpi_errs = [[d[1][0] for d in mpi_vs_cuda_data["mpi_best"]],
                [d[1][1] for d in mpi_vs_cuda_data["mpi_best"]]]
    cuda_meds = [d[0] for d in mpi_vs_cuda_data["cuda_best"]]
    cuda_errs = [[d[1][0] for d in mpi_vs_cuda_data["cuda_best"]],
                 [d[1][1] for d in mpi_vs_cuda_data["cuda_best"]]]

    x = np.arange(len(sizes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    ax.bar(x - w / 2, mpi_meds, w, yerr=mpi_errs,
           color=COLORS["Navy"], edgecolor="white", linewidth=0.3,
           capsize=2, error_kw=dict(lw=0.5, capthick=0.5),
           label="MPI+OMP (best)", zorder=3)

    ax.bar(x + w / 2, cuda_meds, w, yerr=cuda_errs,
           color=COLORS["Purple"], edgecolor="white", linewidth=0.3,
           capsize=2, error_kw=dict(lw=0.5, capthick=0.5),
           label="CUDA", zorder=3)

    ax.set_ylabel("Execution time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=7)
    ax.set_xlabel("Problem size")
    ax.grid(True, alpha=0.2, axis="y", linestyle="-")

    ax.legend(frameon=True, fancybox=False, edgecolor=COLORS["Mist"],
              fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig("mpi_vs_cuda.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("mpi_vs_cuda.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("✓ mpi_vs_cuda.pdf / .png saved")


if __name__ == "__main__":
    main()
