import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def load_data(kernel_file):
    base = kernel_file + "/acoustic_isotropic"
    X = np.load(base + "/X.npy")
    Y = np.load(base + "/Y.npy")
    Z = np.load(base + "/Z.npy")
    rho = np.load(base + "/rho.npy")
    kappa = np.load(base + "/kappa.npy")
    rhop = np.load(base + "/rhop.npy")
    alpha = np.load(base + "/alpha.npy")

    return X, Y, Z, rho, kappa, rhop, alpha


def infer_y_axis(Y):
    spans = []
    for axis in range(Y.ndim):
        span = np.max(Y, axis=axis) - np.min(Y, axis=axis)
        spans.append(span.mean())
    return int(np.argmax(spans))


def collapse_y_dimension(array, reference_shape, y_axis):
    if array.shape != reference_shape:
        array = array.reshape(reference_shape)
    return array.mean(axis=y_axis)


def preprocess_data(X, Z, **kwargs):
    xi = np.linspace(X.min(), X.max(), 100)
    zi = np.linspace(Z.min(), Z.max(), 100)

    X_grid, Z_grid = np.meshgrid(xi, zi)

    data = {}
    for key, value in kwargs.items():
        data[key] = griddata((X, Z), value, (X_grid, Z_grid), method="cubic")

    return X_grid, Z_grid, data


def plot_data(ax, X, Z, data, title, cmap):
    _ = plt.colorbar(
        ax.contourf(X, Z, data, cmap=cmap, levels=1000, vmin=-1.5e-8, vmax=1.5e-8),
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Z (km)")

    ax.set_xticks(np.linspace(X.min(), X.max(), 5))
    ax.set_yticks(np.linspace(Z.min(), Z.max(), 5))
    ax.set_xticklabels(["{:.0f}".format(x / 1000) for x in ax.get_xticks()])
    ax.set_yticklabels(["{:.0f}".format(z / 1000) for z in ax.get_yticks()])


def plot_kernels(input_directory, output):
    X, Y, Z, rho, kappa, rhop, alpha = load_data(input_directory)
    y_axis = infer_y_axis(Y)
    reference_shape = X.shape

    X = collapse_y_dimension(X, reference_shape, y_axis).ravel()
    Z = (collapse_y_dimension(Z, reference_shape, y_axis) + 80000.0).ravel()
    rho = collapse_y_dimension(rho, reference_shape, y_axis).ravel()
    kappa = collapse_y_dimension(kappa, reference_shape, y_axis).ravel()
    rhop = collapse_y_dimension(rhop, reference_shape, y_axis).ravel()
    alpha = collapse_y_dimension(alpha, reference_shape, y_axis).ravel()

    X_grid, Z_grid, data = preprocess_data(
        X, Z, rho=rho, kappa=kappa, rhop=rhop, alpha=alpha
    )

    _, ax = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.5)
    cmap = plt.get_cmap("RdYlGn")

    plot_data(ax[0, 0], X_grid, Z_grid, data["kappa"], r"$\kappa_\kappa$", cmap)
    plot_data(ax[0, 1], X_grid, Z_grid, data["alpha"], r"$\kappa_\alpha$", cmap)
    plot_data(ax[1, 0], X_grid, Z_grid, data["rho"], r"$\kappa_\rho$", cmap)
    plot_data(ax[1, 1], X_grid, Z_grid, data["rhop"], r"$\kappa_\rho'$", cmap)

    plt.savefig(output, dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_kernels("OUTPUT_FILES/Kernels", "Kernels_out.png")
