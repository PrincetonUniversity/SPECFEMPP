"""
Plot seismograms from 2D and 3D Tromp_2005 benchmarks side by side.

Run from the 3D build directory:
    python plot_seismograms.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
D3 = HERE / "OUTPUT_FILES/results"
D2 = HERE / "../../dim2/Tromp_2005/OUTPUT_FILES/results"


def load(directory, name):
    return np.loadtxt(directory / name)


def main():
    bxx2 = load(D2, "AA.S0001.S2.BXX.semd")
    bxz2 = load(D2, "AA.S0001.S2.BXZ.semd")
    bxx3 = load(D3, "AA.S0001.S3.BXX.semd")
    bxz3 = load(D3, "AA.S0001.S3.BXZ.semd")
    bxy3 = load(D3, "AA.S0001.S3.BXY.semd")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        "Tromp 2005: 2D vs 3D Seismograms\n"
        "(source: fx=−1, x=50 km z=40 km  →  receiver x=150 km z=40 km)",
        fontsize=13,
    )

    panels = [
        (axes[0, 0], bxx2, "2D BXX", "b"),
        (axes[0, 1], bxx3, "3D BXX", "r"),
        (axes[1, 0], bxz2, "2D BXZ", "b"),
        (axes[1, 1], bxz3, "3D BXZ", "r"),
        (axes[2, 0], None, "2D BXY\n(not simulated)", None),
        (axes[2, 1], bxy3, "3D BXY", "r"),
    ]

    for ax, data, label, color in panels:
        if data is None:
            ax.text(
                0.5,
                0.5,
                "(not in 2D)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="gray",
                fontsize=12,
            )
        else:
            t, u = data[:, 0], data[:, 1]
            ax.plot(t, u, color=color, lw=0.9)
            ax.axhline(0, color="k", lw=0.3)
        ax.set_title(label)
        ax.set_xlim(-5, 40)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for row in range(3):
        axes[row, 0].set_ylabel("Displacement (m)")

    plt.tight_layout()
    out = HERE / "OUTPUT_FILES/seismogram_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
