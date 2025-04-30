import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors as colors

# Get user and filename
user = os.environ.get("USER")
filename = "OUTPUT_FILES/wavefield/ForwardWavefield.h5"

# Read coordinates
with h5py.File(filename, "r") as f:
    x = f["/Coordinates/elastic_psv_t/X"][:].flatten()
    z = f["/Coordinates/elastic_psv_t/Z"][:].flatten()

# Print extent
print(f"X min: {min(x)}, X max: {max(x)}")
print(f"Z min: {min(z)}, Z max: {max(z)}")
# Print number of points
print(f"Number of points: {len(x)}")

# Create grid for interpolation
NEX = 500
NEZ = int(NEX * (max(z) - min(z)) / (max(x) - min(x)))
X, Z = np.meshgrid(
    np.linspace(min(x), max(x), NEX + 1), np.linspace(min(z), max(z), NEZ + 1)
)

# Create custom colormap
n = 256
red = np.column_stack(
    [np.linspace(1, 1, n // 2), np.linspace(1, 0, n // 2), np.linspace(1, 0, n // 2)]
)
blue = np.column_stack(
    [np.linspace(0, 1, n // 2), np.linspace(0, 1, n // 2), np.linspace(1, 1, n // 2)]
)
cmap_data = np.vstack([np.flipud(red), np.flipud(blue)])
custom_cmap = colors.ListedColormap(cmap_data)

# Ensure output directory exists
os.makedirs("OUTPUT_FILES/wavefield/wavefield_plots", exist_ok=True)

# Loop through timesteps
for it in range(200, 1601, 200):
    plt.figure(figsize=(12, 5))

    # Read displacement data
    path = f"/Step{it:06d}/elastic_psv_t/Displacement"
    with h5py.File(filename, "r") as f:
        u = f[path][:, :]

    shape = u.shape

    u = u.flatten().reshape(shape[1], shape[0]).T

    print(f"Displacement shape: {u.shape}")

    ux = u[:, 0].flatten()
    uz = u[:, 1].flatten()
    ur = u[:, 2].flatten()

    # Interpolate ux to grid
    Ux = griddata((x, z), ux, (X, Z), method="linear")
    Uz = griddata((x, z), uz, (X, Z), method="linear")
    Ur = griddata((x, z), ur, (X, Z), method="linear")

    cx_scale = np.max(np.abs(ux))
    cz_scale = np.max(np.abs(uz))
    cr_scale = np.max(np.abs(ur))

    print(f"Max ux: {cx_scale}")
    print(f"Max uz: {cz_scale}")
    print(f"Max ur: {cr_scale}")

    # Plot ux
    plt.subplot(1, 3, 1)

    im1 = plt.pcolormesh(X, Z, Ux, cmap="seismic")
    plt.clim(-cx_scale, cx_scale)
    plt.colorbar()
    # plt.axis('equal')
    # plt.axis('off')
    # plt.plot([0, 1], [0.2, 0.2], 'k')
    plt.title("P-SV-T U_x")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")

    # # Plot uz
    plt.subplot(1, 3, 2)
    im2 = plt.pcolormesh(X, Z, Uz, cmap="seismic")
    plt.clim(-cz_scale, cz_scale)
    plt.title("P-SV-T U_z")
    plt.xlabel("x (m)")
    plt.colorbar()
    # # plt.axis('equal')
    # # plt.axis('off')
    # # plt.plot([0, 1], [0.2, 0.2], 'k')
    # # plt.ylabel('z (m)')
    # plt.gca().set_aspect('equal', adjustable='box')
    # # plt.xlim([min(x), max(x)])
    # # plt.ylim([min(z), max(z)])

    # # Plot ur
    plt.subplot(1, 3, 3)
    im3 = plt.pcolormesh(X, Z, Ur, cmap="seismic")
    plt.clim(-cr_scale, cr_scale)
    plt.colorbar()
    # # plt.axis('equal')
    # # plt.axis('off')
    # # plt.plot([0, 1], [0.2, 0.2], 'k')
    plt.title("P-SV-T U_r")

    # Save figure
    figname = f"OUTPUT_FILES/wavefield/wavefield_plots/psvt{it:06d}.jpg"
    plt.savefig(figname)
    plt.close()

    print(f"Saved {figname}")
