import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import yaml


def plot_wavefield(coordinates, field, ax=None):
    """
    Plot the angle points of a given field.

    Parameters
    ----------
    coordinates : np.ndarray
        The coordinates of the points.
    field : np.ndarray
        The field values at the points.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """

    # Instead of using the ll value compute the line  coordinates as done above
    # from the angle

    # interpolate the data using griddata
    X, Z = np.meshgrid(
        np.linspace(np.min(coordinates[:, 0]), np.max(coordinates[:, 0]), 100),
        np.linspace(np.min(coordinates[:, 1]), np.max(coordinates[:, 1]), 100),
    )
    Ux = griddata(
        (coordinates[:, 0], coordinates[:, 1]), field[:, 0], (X, Z), method="linear"
    )
    Uz = griddata(
        (coordinates[:, 0], coordinates[:, 1]), field[:, 1], (X, Z), method="linear"
    )

    dx = X[0, 1] - X[0, 0]
    dz = Z[1, 0] - Z[0, 0]
    print("   dx,dz", dx, dz)
    dUx_z = np.gradient(Ux, dz, axis=1)
    dUz_x = np.gradient(Uz, dx, axis=0)

    # Curl
    cU = 0.5 * (dUz_x - dUx_z)

    # print min max curl
    print(f"    Min curl: {np.min(cU)}")
    print(f"    Max curl: {np.max(cU)}")

    fig = plt.figure(figsize=(30, 5.0))

    ax_ux = fig.add_subplot(131)

    ax_ux.set_title("$U_x$")
    ax_ux.set_xlabel("x (m)")
    ax_ux.set_ylabel("z (m)")
    ax_ux.set_aspect("equal", adjustable="box")
    ux_scale = np.max(np.abs(Ux))
    sc_ux = ax_ux.contourf(
        X,
        Z,
        Ux,
        levels=np.linspace(-ux_scale, ux_scale, 20, endpoint=True),
        cmap="seismic",
        alpha=1.0,
        antialiased=True,
    )
    plt.colorbar(sc_ux, ax=ax_ux)

    ax_uz = fig.add_subplot(132)
    ax_uz.set_title("$U_z$")
    ax_uz.set_xlabel("x (m)")
    ax_uz.set_ylabel("z (m)")
    ax_uz.set_aspect("equal", adjustable="box")
    uz_scale = np.max(np.abs(Uz))
    sc_uz = ax_uz.contourf(
        X,
        Z,
        Uz,
        levels=np.linspace(-uz_scale, uz_scale, 20, endpoint=True),
        cmap="seismic",
        alpha=1.0,
        antialiased=True,
    )
    plt.colorbar(sc_uz, ax=ax_uz)

    # Plot curl
    ax_curl = fig.add_subplot(133)
    ax_curl.set_title(r"$\frac{1}{2}\mathbf{\nabla} \times \mathbf{s}$")
    ax_curl.set_xlabel("x (m)")
    ax_curl.set_ylabel("z (m)")
    ax_curl.set_aspect("equal", adjustable="box")
    curl_scale = np.max(np.abs(cU))
    sc_curl = ax_curl.contourf(
        X,
        Z,
        cU,
        levels=np.linspace(-curl_scale, curl_scale, 20, endpoint=True),
        cmap="seismic",
        alpha=1.0,
        antialiased=True,
    )
    plt.colorbar(sc_curl, ax=ax_curl)

    return fig


# Get user and filename
filename = "OUTPUT_FILES/wavefield/ForwardWavefield.h5"
output_dir = "OUTPUT_FILES/wavefield/wavefield_plots"

if not os.path.exists(filename):
    print(f"File {filename} does not exist. Please check the path.")
    exit()

# Ensure output directory exists
os.makedirs("OUTPUT_FILES/wavefield/wavefield_plots", exist_ok=True)

glob_files = glob.glob("OUTPUT_FILES/wavefield/wavefield_plots/*")
for f in glob_files:
    if os.path.isfile(f):
        os.remove(f)

# Read coordinates
with h5py.File(filename, "r") as f:
    x = f["/Coordinates/elastic_psv/X"][:].flatten()
    z = f["/Coordinates/elastic_psv/Z"][:].flatten()

    # Get steps from list of keys
    keys = list(f["/"].keys())
    keys.remove("Coordinates")
    steps = [int(x[4:]) for x in list(keys)]

    steps.sort()
    it = steps[0]
    ft = steps[-1]
    delta_timestep = steps[1] - steps[0]

print("Timing information")
print("------------------")
print(f"  First timestep: {it}")
print(f"  Last timestep: {ft}")
print(f"  Delta timestep: {delta_timestep}")
print(f"  Number of timesteps to plot: {len(steps)}")

# Print extent
print("Extent of the model")
print("-------------------")
print(f"  X min: {min(x)}, X max: {max(x)}")
print(f"  Z min: {min(z)}, Z max: {max(z)}")
# Print number of points
print(f"  Number of points: {len(x)}\n")

# get dt from yaml file
configfile = "specfem_config.yaml"
with open(configfile, "r") as f:
    config = yaml.safe_load(f)
    dt = config["parameters"]["simulation-setup"]["solver"]["time-marching"][
        "time-scheme"
    ]["dt"]
    print(f"  Time step: {dt}")


# Loop through timesteps
for it in range(it, ft, delta_timestep):
    plt.figure(figsize=(12, 12))

    print(f"Processing timestep {it:0>5d} of {ft:0>5d}")
    print("---------------------------------")
    print("  Reading ...")

    # Read displacement data
    path = f"/Step{it:06d}/elastic_psv/Displacement"
    with h5py.File(filename, "r") as f:
        u = f[path][:, :]

    print("  Interpolating ...")
    shape = u.shape

    u = u.flatten().reshape(shape[1], shape[0]).T

    ux = u[:, 0].flatten()
    uz = u[:, 1].flatten()

    # Interpolate ux to grid
    # Ux = griddata((x, z), ux, (X, Z), method="linear")
    # Uz = griddata((x, z), uz, (X, Z), method="linear")
    # Ur = griddata((x, z), ur, (X, Z), method="linear")

    cx_scale = np.max(np.abs(ux))
    cz_scale = np.max(np.abs(uz))

    print(f"    Max ux: {cx_scale}")
    print(f"    Max uz: {cz_scale}")

    print("  Plotting ...")

    plot_wavefield(np.column_stack((x.flatten(), z.flatten())), u)
    # plt.title(f"PSV-T Wavefield at t = {it * dt:.2f} s")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.suptitle(f"P-SV Wavefield at t = {it * dt:.2f} s")
    # Save figure
    figname = f"OUTPUT_FILES/wavefield/wavefield_plots/psv{it:06d}.pdf"
    plt.savefig(figname, dpi=300)
    plt.close()

    print(f"  Saved {figname}\n")
