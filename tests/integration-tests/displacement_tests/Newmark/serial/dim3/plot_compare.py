#!/usr/bin/env python
"""Overlay SPECFEM++ seismograms against SPECFEM3D reference traces.

Usage:
    python plot_compare.py <fixture_dir> [<specfempp_dir>]

<fixture_dir>   fixture directory containing traces/ (SPECFEM3D reference)
<specfempp_dir> directory with SPECFEM++ .sem* output (default:
                <fixture_dir>/seismograms)

Writes <fixture_dir>/compare.png and prints per-channel amplitude ratios,
correlations, and residuals.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def main():
    fixture = sys.argv[1]
    pp_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(fixture, "seismograms")
    tr_dir = os.path.join(fixture, "traces")

    # Discover stations and channels from the reference traces.
    names = sorted(f for f in os.listdir(tr_dir) if ".sem" in f)
    stations = sorted({n.split(".")[1] for n in names})
    channels = sorted({".".join(n.split(".")[3:]) for n in names})
    network = names[0].split(".")[0]
    loc = names[0].split(".")[2]

    fig, axes = plt.subplots(
        len(stations),
        len(channels),
        figsize=(4.0 * len(channels), 2.4 * len(stations)),
        squeeze=False,
    )

    print("st chan        |3D|max      |++|max     ratio    corr    resid_max")
    for i, st in enumerate(stations):
        for j, ch in enumerate(channels):
            ax = axes[i][j]
            fn = f"{network}.{st}.{loc}.{ch}"
            p3 = os.path.join(tr_dir, fn)
            pp = os.path.join(pp_dir, fn)
            if os.path.exists(p3):
                t, v = load(p3)
                ax.plot(t, v, "k-", lw=1.3, label="SPECFEM3D")
            if os.path.exists(pp):
                t, v = load(pp)
                ax.plot(t, v, "r--", lw=1.0, label="SPECFEM++")
            if i == 0:
                ax.set_title(ch, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{network}.{st}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.axhline(0, color="gray", lw=0.4)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
            if os.path.exists(p3) and os.path.exists(pp):
                _, v3 = load(p3)
                _, vp = load(pp)
                n = min(len(v3), len(vp))
                v3, vp = v3[:n], vp[:n]
                corr = np.dot(v3, vp) / (
                    np.linalg.norm(v3) * np.linalg.norm(vp) + 1e-30
                )
                a3, ap = np.abs(v3).max(), np.abs(vp).max()
                print(
                    f"{st} {ch:11s} {a3:10.3e} {ap:10.3e} "
                    f"{a3 / (ap + 1e-30):7.2f}  {corr:+.4f} "
                    f"{np.abs(vp - v3).max():10.3e}"
                )

    fig.suptitle(
        f"{os.path.basename(os.path.normpath(fixture))}: "
        "SPECFEM++ (red dashed) vs SPECFEM3D (black)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(fixture, "compare.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
