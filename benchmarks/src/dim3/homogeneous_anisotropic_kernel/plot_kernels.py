"""Plot 3D anisotropic sensitivity kernels on the source-receiver plane."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


KERNEL_NAMES = (
    "rho",
    "c11",
    "c12",
    "c13",
    "c14",
    "c15",
    "c16",
    "c22",
    "c23",
    "c24",
    "c25",
    "c26",
    "c33",
    "c34",
    "c35",
    "c36",
    "c44",
    "c45",
    "c46",
    "c55",
    "c56",
    "c66",
)


def load_kernel_directory(directory):
    arrays = {}
    for name in ("X", "Y", "Z", *KERNEL_NAMES):
        path = os.path.join(directory, f"{name}.npy")
        arrays[name] = np.load(path).flatten()
    return arrays


def load_kernels(kernels_dir):
    kernels_root = os.path.join(kernels_dir, "Kernels")
    serial_dir = os.path.join(kernels_root, "elastic_anisotropic")

    if os.path.isdir(serial_dir):
        arrays = load_kernel_directory(serial_dir)
    else:
        process_dirs = sorted(
            os.path.join(kernels_root, entry, "elastic_anisotropic")
            for entry in os.listdir(kernels_root)
            if entry.startswith("proc_")
            and os.path.isdir(os.path.join(kernels_root, entry, "elastic_anisotropic"))
        )
        if not process_dirs:
            raise FileNotFoundError(
                f"No elastic anisotropic kernels found in {kernels_root}"
            )

        process_arrays = [load_kernel_directory(path) for path in process_dirs]
        arrays = {
            name: np.concatenate([values[name] for values in process_arrays])
            for name in ("X", "Y", "Z", *KERNEL_NAMES)
        }

    coordinates = arrays["X"], arrays["Y"], arrays["Z"]
    kernels = {name: arrays[name] for name in KERNEL_NAMES}
    return *coordinates, kernels


def slice_xz(x, y, z, kernels, y_center, half_width=3500.0):
    mask = np.abs(y - y_center) < half_width
    sliced = {name: values[mask] for name, values in kernels.items()}
    return x[mask], z[mask], sliced


def plot_kernel_slice(axis, x, z, values, label, source, station):
    xi = np.linspace(x.min(), x.max(), 160)
    zi = np.linspace(z.min(), z.max(), 160)
    xi_grid, zi_grid = np.meshgrid(xi, zi)
    interpolated = griddata((x, z), values, (xi_grid, zi_grid), method="linear")

    limit = np.nanpercentile(np.abs(interpolated), 98)
    if not np.isfinite(limit) or limit == 0.0:
        limit = 1.0
    image = axis.pcolormesh(
        xi_grid / 1000,
        zi_grid / 1000,
        interpolated,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        shading="auto",
    )
    plt.colorbar(image, ax=axis, shrink=0.8)

    axis.plot(source[0] / 1000, source[2] / 1000, "r*", markersize=8)
    axis.plot(station[0] / 1000, station[2] / 1000, "rv", markersize=6)
    axis.set_title(label)
    axis.set_xlabel("X (km)")
    axis.set_ylabel("Z (km)")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <kernels_dir> <output_png>")
        sys.exit(1)

    source = (110000.0, 50000.0, -50000.0)
    station = (40000.0, 50000.0, -50000.0)

    x, y, z, kernels = load_kernels(sys.argv[1])
    x, z, kernels = slice_xz(x, y, z, kernels, source[1])

    figure, axes = plt.subplots(4, 6, figsize=(24, 15), constrained_layout=True)
    for axis, name in zip(axes.flat, KERNEL_NAMES):
        plot_kernel_slice(axis, x, z, kernels[name], name, source, station)
    for axis in axes.flat[len(KERNEL_NAMES) :]:
        axis.set_visible(False)

    figure.suptitle("3D anisotropic kernels — X-Z source-receiver plane")
    figure.savefig(sys.argv[2], dpi=150)
    print(f"Saved: {sys.argv[2]}")


if __name__ == "__main__":
    main()
