import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# Load the kernels
def load_data(kernel_file):
    X = np.load(kernel_file + "/elastic_psv_isotropic/X.npy")
    Z = np.load(kernel_file + "/elastic_psv_isotropic/Z.npy")

    rho = np.load(kernel_file + "/elastic_psv_isotropic/rho.npy")
    mu = np.load(kernel_file + "/elastic_psv_isotropic/mu.npy")
    kappa = np.load(kernel_file + "/elastic_psv_isotropic/kappa.npy")
    rhop = np.load(kernel_file + "/elastic_psv_isotropic/rhop.npy")
    alpha = np.load(kernel_file + "/elastic_psv_isotropic/alpha.npy")
    beta = np.load(kernel_file + "/elastic_psv_isotropic/beta.npy")

    return X, Z, rho, kappa, mu, rhop, alpha, beta


# Preprocess the data into a 2D grid for plotting
def preprocess_data(X, Z, **kwargs):
    xi = np.linspace(X.min(), X.max(), 100)
    zi = np.linspace(Z.min(), Z.max(), 100)

    X_grid, Z_grid = np.meshgrid(xi, zi)

    data = {}
    for key, value in kwargs.items():
        data[key] = griddata((X, Z), value, (X_grid, Z_grid), method="cubic")

    return X_grid, Z_grid, data


# Plot the data
def plot_data(ax, X, Z, data, title, cmap):
    # ax.contourf(X, Z, data, cmap = cmap, levels=1000, vmin = -1.5e-8, vmax = 1.5e-8)
    # bar levels
    _ = plt.colorbar(
        ax.contourf(X, Z, data, cmap=cmap, levels=1000, vmin=-1.5e-8, vmax=1.5e-8),
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Z (km)")

    ## set 5 ticks
    ax.set_xticks(np.linspace(X.min(), X.max(), 5))
    ax.set_yticks(np.linspace(Z.min(), Z.max(), 5))

    ## devide the ticks by 1000
    ax.set_xticklabels(["{:.0f}".format(x / 1000) for x in ax.get_xticks()])
    ax.set_yticklabels(["{:.0f}".format(x / 1000) for x in ax.get_yticks()])

    return


def plot_kernels(input_directory, output):
    # Load the kernels
    X, Z, rho, kappa, mu, rhop, alpha, beta = load_data(input_directory)

    # Preprocess the data
    X_grid, Z_grid, data = preprocess_data(
        X, Z, rho=rho, kappa=kappa, mu=mu, rhop=rhop, alpha=alpha, beta=beta
    )

    # Unpack the data
    rho_grid = data["rho"]
    kappa_grid = data["kappa"]
    mu_grid = data["mu"]
    rhop_grid = data["rhop"]
    alpha_grid = data["alpha"]
    beta_grid = data["beta"]

    # Plot the data
    _, ax = plt.subplots(3, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.5)
    cmap = plt.get_cmap("RdYlGn")

    ## plot mu
    plot_data(ax[0, 0], X_grid, Z_grid, mu_grid, r"$\kappa_\mu$", cmap)

    # ## plot beta
    plot_data(ax[0, 1], X_grid, Z_grid, beta_grid, r"$\kappa_\beta$", cmap)

    # ## plot kappa
    plot_data(ax[1, 0], X_grid, Z_grid, kappa_grid, r"$\kappa_\kappa$", cmap)

    # ## plot alpha
    plot_data(ax[1, 1], X_grid, Z_grid, alpha_grid, r"$\kappa_\alpha$", cmap)

    # ## plot rho
    plot_data(ax[2, 0], X_grid, Z_grid, rho_grid, r"$\kappa_\rho$", cmap)

    # ## plot rhop
    plot_data(ax[2, 1], X_grid, Z_grid, rhop_grid, r"$\kappa_\rho'$", cmap)

    plt.savefig(output, dpi=300)

    return


if __name__ == "__main__":
    plot_kernels("OUTPUT_FILES/Kernels", "Kernels_out.png")
