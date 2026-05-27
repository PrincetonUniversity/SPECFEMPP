"""
Plot acoustic sensitivity kernels for the homogeneous halfspace benchmark.

Usage (from Snakemake):
    from plot import plot_kernels
    plot_kernels(input_directory="...", output="...")

Or from the command line:
    python plot.py <kernel_dir> <output_png>
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_kernels(input_directory: str, output: str) -> None:
    """
    Load acoustic isotropic kernels and plot vertical cross-sections.

    Parameters
    ----------
    input_directory : str
        Path to the Kernels output directory (contains the
        ``acoustic_isotropic/`` subdirectory).
    output : str
        Output PNG file path.
    """
    kernel_dir = input_directory + "/acoustic_isotropic"

    X = np.load(kernel_dir + "/X.npy")  # (nelem, ngllz, nglly, ngllx)
    Y = np.load(kernel_dir + "/Y.npy")
    Z = np.load(kernel_dir + "/Z.npy")
    alpha = np.load(kernel_dir + "/alpha.npy")
    rhop = np.load(kernel_dir + "/rhop.npy")

    # Flatten all GLL points across elements
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = Z.ravel()
    alpha_flat = alpha.ravel()
    rhop_flat = rhop.ravel()

    # Vertical cross-section near the source X-coordinate (67 222 m)
    x_center = 67222.0
    x_half_width = 10000.0
    mask = np.abs(X_flat - x_center) < x_half_width

    def _sym_vmax(values):
        """Return a symmetric color limit at the 99th percentile."""
        if values.size == 0:
            return 1.0
        vmax = np.percentile(np.abs(values), 99)
        return vmax if vmax > 0 else 1.0

    kernels = [alpha_flat, rhop_flat]
    labels = [r"$K_\alpha$  (alpha kernel)", r"$K_{\rho'}$  (rhop kernel)"]
    titles = [
        r"$\alpha$ kernel — vertical cross-section at $x \approx 67\,\mathrm{km}$",
        r"$\rho'$ kernel — vertical cross-section at $x \approx 67\,\mathrm{km}$",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "3-D Acoustic Sensitivity Kernels\n"
        "Homogeneous halfspace, Ricker source at (67, 67, -30) km",
        fontsize=11,
    )

    for ax, kernel, label, title in zip(axes, kernels, labels, titles):
        vals = kernel[mask]
        vmax = _sym_vmax(vals)
        sc = ax.scatter(
            Y_flat[mask] / 1e3,
            Z_flat[mask] / 1e3,
            c=vals,
            cmap="RdBu_r",
            s=1,
            vmin=-vmax,
            vmax=vmax,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label=label, pad=0.02)
        ax.set_xlabel("Y  (km)")
        ax.set_ylabel("Z  (km)")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal", adjustable="box")

        # Mark source and receiver positions (projected onto the cross-section)
        ax.axhline(-30, color="k", lw=0.5, ls="--", label="source depth")
        ax.plot(22.732, -0.05, "v", color="forestgreen", ms=8, label="rcvr X20")
        ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Kernel plot saved to {output}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot.py <kernel_dir> <output_png>")
        sys.exit(1)
    plot_kernels(input_directory=sys.argv[1], output=sys.argv[2])
