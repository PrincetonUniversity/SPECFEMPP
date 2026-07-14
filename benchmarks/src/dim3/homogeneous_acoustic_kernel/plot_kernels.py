"""
Plot acoustic sensitivity kernels on the source-receiver plane (X-Z at Y=center).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def load_kernels(kernels_dir):
    subdir = os.path.join(kernels_dir, "Kernels", "acoustic_isotropic")
    if not os.path.isdir(subdir):
        kernels_root = os.path.join(kernels_dir, "Kernels")
        proc_dirs = sorted(
            os.path.join(kernels_root, entry, "acoustic_isotropic")
            for entry in os.listdir(kernels_root)
            if entry.startswith("proc_")
            and os.path.isdir(os.path.join(kernels_root, entry, "acoustic_isotropic"))
        )
        if not proc_dirs:
            raise FileNotFoundError(f"No acoustic kernel files found in {kernels_root}")

        arrays = {}
        for name in ("X", "Y", "Z", "kappa", "alpha", "rhop"):
            arrays[name] = np.concatenate(
                [
                    np.load(os.path.join(proc_dir, f"{name}.npy")).flatten()
                    for proc_dir in proc_dirs
                ]
            )
        return (
            arrays["X"],
            arrays["Y"],
            arrays["Z"],
            {
                "kappa": arrays["kappa"],
                "alpha": arrays["alpha"],
                "rhop": arrays["rhop"],
            },
        )

    X = np.load(os.path.join(subdir, "X.npy")).flatten()
    Y = np.load(os.path.join(subdir, "Y.npy")).flatten()
    Z = np.load(os.path.join(subdir, "Z.npy")).flatten()
    kappa = np.load(os.path.join(subdir, "kappa.npy")).flatten()
    alpha = np.load(os.path.join(subdir, "alpha.npy")).flatten()
    rhop = np.load(os.path.join(subdir, "rhop.npy")).flatten()
    return X, Y, Z, {"kappa": kappa, "alpha": alpha, "rhop": rhop}


def slice_xz(X, Y, Z, values, y_center, half_width=3500.0):
    mask = np.abs(Y - y_center) < half_width
    return X[mask], Z[mask], {k: v[mask] for k, v in values.items()}


def plot_kernel_slice(ax, x, z, values, label, source, station):
    xi = np.linspace(x.min(), x.max(), 200)
    zi = np.linspace(z.min(), z.max(), 200)
    Xi, Zi = np.meshgrid(xi, zi)

    vi = griddata((x, z), values, (Xi, Zi), method="linear")

    vmax = np.nanpercentile(np.abs(vi), 98)
    im = ax.pcolormesh(
        Xi / 1000, Zi / 1000, vi, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
    )
    plt.colorbar(im, ax=ax, label=label)

    ax.plot(source[0] / 1000, source[2] / 1000, "r*", markersize=12, label="Source")
    ax.plot(station[0] / 1000, station[2] / 1000, "rv", markersize=10, label="Station")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Z (km)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <kernels_dir> <output_png>")
        sys.exit(1)

    kernels_dir = sys.argv[1]
    output_png = sys.argv[2]

    source = (110000.0, 50000.0, -50000.0)
    station = (40000.0, 50000.0, -50000.0)
    y_center = source[1]

    X, Y, Z, kernels = load_kernels(kernels_dir)
    x, z, ks = slice_xz(X, Y, Z, kernels, y_center)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    labels = {"kappa": "κ kernel", "alpha": "α kernel", "rhop": "ρ' kernel"}

    for ax, (name, label) in zip(axes, labels.items()):
        plot_kernel_slice(ax, x, z, ks[name], label, source, station)
        ax.set_title(label)

    fig.suptitle(
        f"Acoustic kernels — X–Z plane at Y = {y_center / 1000:.0f} km", y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_png}")


if __name__ == "__main__":
    main()
