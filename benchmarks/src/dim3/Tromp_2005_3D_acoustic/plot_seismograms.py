"""
Plot pressure seismogram from 3D Tromp_2005 acoustic benchmark.

Run from the 3D acoustic build directory:
    python plot_seismograms.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
D3 = HERE / "OUTPUT_FILES/results"


def load(directory, name):
    return np.loadtxt(directory / name)


def main():
    bxp3 = load(D3, "AA.S0001.S3.BXP.semp")

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.suptitle(
        "Tromp 2005 Acoustic 3D: Pressure Seismogram\n"
        "(source: fx=−1, x=50 km z=40 km  →  receiver x=150 km z=40 km)",
        fontsize=13,
    )

    t, p = bxp3[:, 0], bxp3[:, 1]
    ax.plot(t, p, color="b", lw=0.9)
    ax.axhline(0, color="k", lw=0.3)
    ax.set_title("3D BXP (pressure)")
    ax.set_xlim(-5, 40)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (Pa)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = HERE / "OUTPUT_FILES/seismogram_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
