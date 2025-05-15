import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import yaml


def plot_angle_points(coordinates, field, ax=None):
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

    # Plot the angle points
    # sc = ax.scatter(coordinates[:, 0], coordinates[:, 1], c= field[:, 0], cmap='seismic')

    # Make manual cross for each point as line collection where each
    # line is separated by a np.nan value
    # ll = 10
    # lw = 0.001
    # x = coordinates[:, 0]
    # y = coordinates[:, 1]
    # angle = field[:, 2] * 1e4

    # lxx___0 = ll * np.cos(angle)
    # lxy___0 = ll * np.sin(angle)
    # lyx___0 = -ll * np.sin(angle)
    # lyy___0 = ll * np.cos(angle)
    # lxx_180 = ll * np.cos(angle + np.pi)
    # lxy_180 = ll * np.sin(angle + np.pi)
    # lyx_180 = -ll * np.sin(angle + np.pi)
    # lyy_180 = ll * np.cos(angle + np.pi)
    # xp = np.column_stack(
    #     (
    #         x + lxx___0,
    #         x + lxx_180,
    #         np.nan * np.ones_like(x),
    #         x + lyx___0,
    #         x + lyx_180,
    #         np.nan * np.ones_like(x),
    #     )
    # ).flatten()
    # yp = np.column_stack(
    #     (
    #         y + lxy___0,
    #         y + lxy_180,
    #         np.nan * np.ones_like(y),
    #         y + lyy___0,
    #         y + lyy_180,
    #         np.nan * np.ones_like(y),
    #     )
    # ).flatten()

    # angle = field[:, 2]

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
    Ur = griddata(
        (coordinates[:, 0], coordinates[:, 1]), field[:, 2], (X, Z), method="linear"
    )

    dx = X[0, 1] - X[0, 0]
    dz = Z[1, 0] - Z[0, 0]
    print("   dx,dz", dx, dz)
    dUx_z = np.gradient(Ux, dz, axis=1)
    dUz_x = np.gradient(Uz, dx, axis=0)

    # Curl
    cU = 0.5 * (dUz_x - dUx_z)

    diff = Ur - cU

    # print min max curl
    print(f"    Min curl: {np.min(cU)}")
    print(f"    Max curl: {np.max(cU)}")
    print(f"    Min Ur: {np.min(Ur)}")
    print(f"    Max Ur: {np.max(Ur)}")

    fig = plt.figure(figsize=(30, 16))

    ax_ux = fig.add_subplot(231)

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

    ax_uz = fig.add_subplot(232)
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

    ax_ur = fig.add_subplot(233)
    ax_ur.set_title("$U_r$")
    ax_ur.set_xlabel("x (m)")
    ax_ur.set_ylabel("z (m)")
    ax_ur.set_aspect("equal", adjustable="box")
    ur_scale = np.max(np.abs(Ur))
    sc_ur = ax_ur.contourf(
        X,
        Z,
        Ur,
        levels=np.linspace(-ur_scale, ur_scale, 20, endpoint=True),
        cmap="seismic",
        alpha=1.0,
        antialiased=True,
    )
    plt.colorbar(sc_ur, ax=ax_ur)

    # Plot curl
    ax_curl = fig.add_subplot(235)
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

    # Plot the difference
    ax_diff = fig.add_subplot(236)
    ax_diff.set_title(r"$U_r - \frac{1}{2}\mathbf{\nabla} \times \mathbf{s}$")
    ax_diff.set_xlabel("x (m)")
    ax_diff.set_ylabel("z (m)")
    ax_diff.set_aspect("equal", adjustable="box")
    diff_scale = np.max(np.abs(diff))
    sc_diff = ax_diff.contourf(
        X,
        Z,
        diff,
        levels=np.linspace(-diff_scale, diff_scale, 20, endpoint=True),
        cmap="seismic",
        alpha=1.0,
        antialiased=True,
    )
    plt.colorbar(sc_diff, ax=ax_diff)

    return fig, ax


# Get user and filename
user = os.environ.get("USER")
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
    x = f["/Coordinates/elastic_psv_t/X"][:].flatten()
    z = f["/Coordinates/elastic_psv_t/Z"][:].flatten()

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
    path = f"/Step{it:06d}/elastic_psv_t/Displacement"
    with h5py.File(filename, "r") as f:
        u = f[path][:, :]

    print("  Interpolating ...")
    shape = u.shape

    u = u.flatten().reshape(shape[1], shape[0]).T

    ux = u[:, 0].flatten()
    uz = u[:, 1].flatten()
    ur = u[:, 2].flatten()

    # Interpolate ux to grid
    # Ux = griddata((x, z), ux, (X, Z), method="linear")
    # Uz = griddata((x, z), uz, (X, Z), method="linear")
    # Ur = griddata((x, z), ur, (X, Z), method="linear")

    cx_scale = np.max(np.abs(ux))
    cz_scale = np.max(np.abs(uz))
    cr_scale = np.max(np.abs(ur))

    print(f"    Max ux: {cx_scale}")
    print(f"    Max uz: {cz_scale}")
    print(f"    Max ur: {cr_scale}")

    print("  Plotting ...")

    plot_angle_points(
        np.column_stack((x.flatten(), z.flatten())),
        u,
    )
    # plt.title(f"PSV-T Wavefield at t = {it * dt:.2f} s")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.suptitle(f"PSV-T Wavefield at t = {it * dt:.2f} s")
    # Save figure
    figname = f"OUTPUT_FILES/wavefield/wavefield_plots/psvt{it:06d}.pdf"
    plt.savefig(figname, dpi=300)
    plt.close()

    print(f"  Saved {figname}\n")
