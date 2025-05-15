#!/usr/bin/env python

import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy.interpolate


def mesh2grid(v, x, z):
    """Interpolates from an unstructured coordinates (mesh) to a structured
    coordinates (grid)
    """
    lx = x.max() - x.min()
    lz = z.max() - z.min()
    nn = v.size
    mesh = _stack(x, z)

    nx = int(np.around(np.sqrt(nn * lx / lz)))
    nz = int(np.around(np.sqrt(nn * lz / lx)))

    # construct structured grid
    x = np.linspace(x.min(), x.max(), nx)
    z = np.linspace(z.min(), z.max(), nz)
    X, Z = np.meshgrid(x, z)
    grid = _stack(X.flatten(), Z.flatten())

    # interpolate to structured grid
    V = scipy.interpolate.griddata(mesh, v, grid, "linear")

    # workaround edge issues
    if np.any(np.isnan(V)):
        W = scipy.interpolate.griddata(mesh, v, grid, "nearest")
        for i in np.where(np.isnan(V)):
            V[i] = W[i]

    return np.reshape(V, (int(nz), int(nx))), x, z


def _stack(*args):
    return np.column_stack(args)


if __name__ == "__main__":
    f = h5py.File("OUTPUT_FILES/results/ForwardWavefield.h5", "r")

    x = np.array(f["/Coordinates/elastic_psv/X"])
    z = np.array(f["/Coordinates/elastic_psv/Z"])

    for key in f.keys():
        if key[0] != "S":
            continue

        dset = np.array(f[key]["elastic_psv"]["Displacement"]).reshape([2, x.shape[0]])

        for i, comp in enumerate(["vx", "vz"]):
            V, X, Z = mesh2grid(dset[i, :], x, z)
            plt.figure(figsize=(8, 6))

            amax = abs(V).max()
            plt.pcolor(X, Z, V, vmin=amax * -1, vmax=amax)
            locs = np.arange(X.min(), X.max() + 1, (X.max() - X.min()) / 5)
            plt.xticks(locs)
            plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            locs = np.arange(Z.min(), Z.max() + 1, (Z.max() - Z.min()) / 5)
            plt.yticks(locs)
            plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("z")
            plt.title(comp)
            plt.gca().invert_yaxis()
            plt.set_cmap("seismic")

            plt.savefig("OUTPUT_FILES/" + key + "_" + comp + ".png")
