import os
import sys
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable


def vertex_map(mapping, x, z, verbose=False):
    """
    This function creates vertices and field map for quadrilateral from the
    ibool mapping and the coordinates x and z. The input mapping is a 3D array
    of shape (nspec, ngll, ngll) and the coordinates are 1D arrays of length
    nglob. The function returns a 3D array of shape (ntotal, 4, 2) for the
    vertices and a 2D array of shape (ntotal, 4) for the field map.
    The vertices are the coordinates of the corners of the quadrilaterals to be
    plotted and the field map is the mapping from iglob to the vertices.

    The idea is that you now can take the field you want to plot map the field
    to the vertices, average their values and assign the color to the
    facecolor of the quadrilateral.

    Note: There may be a better way of doing this. If you consider
    the field of each element you could probably create a map of the all iglob
    and weights needed for the center of each quadrilateral.

    Graphical Explanation:

    Looking at a single element, we take the local to global mapping to create
    quadrilateral for each subrectangle of the element. Starting with the ix=0, iz=0
    corner moving counterclockwise for each subquad, indicated by the numbers
    coinciding with the GLL points. Then we move in ix direction for each quad
    indicated by the number on the face of each quad.


    .. code-block::

        •----•-----•-----4----3
        | 12 |  13 |  14 | 15 |
        •----•-----•-----1----2
        |  8 |   9 |  10 | 11 |
        •----•-----•-----•----•
        |  4 |   5 |   6 |  7 |
        4----3-----•-----•----•
        |  0 |   1 |   2 |  3 |
        1----2-----•-----•----•

    In code this will look like this (expanded of course to all elements to all quads):

    .. code-block:: python

        # vert0 = [[x1, z1], [x2, z2], [x3, z3], [x4, z4]]

        i1 = mapping[i, iz, ix]
        i2 = mapping[i, iz, ix + 1]
        i3 = mapping[i, iz + 1, ix + 1]
        i4 = mapping[i, iz + 1, ix]

        # vertices
        vert0 = [[x[i1], z[i1]], [x[i2], z[i2]], [x[i3], z[i3]], [x[i4], z[i4]]]

        # map
        map = [[i1, i2, i3, i4],]


    This leaves us with nspec * (ngll - 1) ** 2 quads to plot. And

    """
    nspec = np.shape(mapping)[0]
    ngll = np.shape(mapping)[1]

    # Total number of quads to be plotted
    ntotal = nspec * (ngll - 1) ** 2

    vertices = np.zeros((ntotal, 4, 2))
    field_map = np.zeros((ntotal, 4), dtype=int)

    for i in range(nspec):
        for iz in range(ngll - 1):
            for ix in range(ngll - 1):
                face_index = i * (ngll - 1) ** 2 + iz * (ngll - 1) + ix

                if verbose:
                    print(i, iz, ix, "->", face_index)

                iglob1 = mapping[i, iz, ix]
                iglob2 = mapping[i, iz, ix + 1]
                iglob3 = mapping[i, iz + 1, ix + 1]
                iglob4 = mapping[i, iz + 1, ix]

                x1 = x[iglob1]
                x2 = x[iglob2]
                x3 = x[iglob3]
                x4 = x[iglob4]
                z1 = z[iglob1]
                z2 = z[iglob2]
                z3 = z[iglob3]
                z4 = z[iglob4]

                if verbose:
                    print((x1, z1), (x2, z2), (x3, z3), (x4, z4))

                field_map[face_index, 0] = iglob1
                field_map[face_index, 1] = iglob2
                field_map[face_index, 2] = iglob3
                field_map[face_index, 3] = iglob4

                vertices[face_index, 0, :] = [x1, z1]
                vertices[face_index, 1, :] = [x2, z2]
                vertices[face_index, 2, :] = [x3, z3]
                vertices[face_index, 3, :] = [x4, z4]

    return vertices, field_map


def get_plot_dict(filename: str, verbose=False):
    """
    This function reads the hdf5 file and returns a dictionary of things to plot.
    The dictionary contains the coordinates and the steps to plot.

    Note that this function is not used to load the actual field to plot but
    to generate all the metadata, such as vertices and field maps to plot
    timesteps. afterwards.
    """

    with h5py.File(filename, "r") as f:
        # Create plotdict and store filename
        plotdict = dict(wavefield_file=os.path.abspath(filename))

        # Store filename and potential media to plot
        media = list(f["Coordinates"].keys())

        # Empty list to store media if plot required
        plotdict["media"] = []

        nfields = 0

        # Loop over media
        for medium in media:
            # if medium == "elastic_psv":
            #     continue

            if verbose:
                print(f"  {medium}")

            plotdict[medium] = dict()

            # Get the coordinates
            x = f["Coordinates"][medium]["X"][:]
            z = f["Coordinates"][medium]["Z"][:]

            mapping = f["Coordinates"][medium]["mapping"][:]
            mapping = mapping.flatten().reshape(
                (mapping.shape[2], mapping.shape[1], mapping.shape[0])
            )
            mapping = mapping.T

            if len(x) == 0:
                if verbose:
                    print(f"  No coordinates for {medium}")
                continue
            else:
                plotdict["media"].append(medium)

            # Compute vertices and field map
            plotdict[medium]["vertices"], plotdict[medium]["field_map"] = vertex_map(
                mapping, x, z, verbose=False
            )

            # Compute steps in the file
            keys = list(f["/"].keys())
            keys.remove("Coordinates")
            steps = [int(x[4:]) for x in list(keys)]
            steps.sort()

            plotdict["steps"] = dict()
            plotdict["steps"]["start"] = steps[0]
            plotdict["steps"]["end"] = steps[-1]
            plotdict["steps"]["delta"] = steps[1] - steps[0]

            # We need to know how many field we are plotting
            if medium == "acoustic":
                nfields += 1
            else:
                stepstr = f"Step{steps[0]:06d}"
                ncomp = f[stepstr][medium]["Displacement"].shape[1]
                if verbose:
                    print(f"  {medium} has {ncomp} components")
                nfields += ncomp

    plotdict["nfields"] = nfields

    return plotdict


def field_helper(medium: str, field_type: str):
    """
    This function returns the field type to plot. The field type is either
    "Displacement" or "Velocity". The function also returns the field type
    to use in the plot.
    """

    if medium == "acoustic":
        if field_type == "Displacement":
            field_type = "Potential"
        elif field_type == "Velocity":
            field_type = "PotentialDot"
        elif field_type == "Acceleration":
            field_type = "PotentialDotDot"
        else:
            raise ValueError(f"Unknown field type {field_type}")
    else:
        pass

    return field_type


def field_modifier(medium: str, field: np.ndarray, combined=True, component=0):
    """
    This function modifies the field to plot. The field is either
    "Displacement" or "Velocity". The function also returns the field type
    to use in the plot.
    """

    if medium == "acoustic":
        return np.abs(field[:, 0]) / 1e5

    elif medium == "elastic_psv":
        if combined:
            field = np.sqrt(field[:, 0] ** 2 + field[:, 1] ** 2)
        else:
            field = field[:, component]

    elif medium == "elastic_sh":
        return field[:, 0]

    return field


def load_field(
    filename: str,
    medium: str,
    field_type: str,
    stepstr: str,
    combined=True,
    component=0,
):
    with h5py.File(filename, "r") as f:
        # Get the field type depending on the medium
        get_field = field_helper(medium, field_type)

        # Load the field and transform it (matlab order)
        field = f[stepstr][medium][get_field][:]
        field = field.flatten().reshape(field.shape[1], field.shape[0]).T

        print(f"  {medium}: {field.shape}")
        print(f"            Min: {np.min(field)}, Max: {np.max(field)}")

        # Modify the field for the plot
        field = field_modifier(medium, field, combined=combined, component=component)

    return field


def compute_scaling(facefields):
    """
    This function computes the scaling for the color map. The scaling is
    computed as the maximum absolute value of the field.
    """

    fmin = 999999999
    fmax = -999999999

    for medium, field in facefields.items():
        fmin = min(fmin, np.min(field))
        fmax = max(fmax, np.max(field))

    fabs = np.max(np.abs([fmin, fmax]))
    return fabs


def compute_extent(plotdict, verbose=False):
    """
    This function computes the extent of the plot. The extent is computed
    as the minimum and maximum x and z coordinates of the vertices.
    """

    x_min = 999999999
    x_max = -999999999
    z_min = 999999999
    z_max = -999999999

    if verbose:
        print("Extent of the model")
        print("-------------------")

    for medium in plotdict["media"]:
        mxmin = np.min(plotdict[medium]["vertices"][:, :, 0])
        mxmax = np.max(plotdict[medium]["vertices"][:, :, 0])
        mzmin = np.min(plotdict[medium]["vertices"][:, :, 1])
        mzmax = np.max(plotdict[medium]["vertices"][:, :, 1])

        if verbose:
            print(f"  {medium}:")
            print(f"    X min: {mxmin}, X max: {mxmax}")
            print(f"    Z min: {mzmin}, Z max: {mzmax}")

        x_min = min(x_min, mxmin)
        x_max = max(x_max, mxmax)
        z_min = min(z_min, mzmin)
        z_max = max(z_max, mzmax)

    if verbose:
        print("  Overall extent of the model")
        print(f"    X min: {x_min}, X max: {x_max}")
        print(f"    Z min: {z_min}, Z max: {z_max}")

    return (x_min, x_max, z_min, z_max)


def plot_wavefield(
    plotdict,
    title,
    combined=True,
    component=0,
    cmap=None,
    field_type="Displacement",
    antialiased=False,
    output_dir=None,
    dpi=300,
    output_type="png",
    edgecolors="none",
    verbose=False,
):
    print("Processing Metadata")
    print("--------------------")
    print("  Reading ...")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title)

    # Get extent of the plot
    extent = compute_extent(plotdict, verbose=True)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal", adjustable="box")

    # Create the poly collections and plot a 0 time step (white)
    polys = dict()
    facefields = dict()

    # Plot the vertices
    if cmap is None:
        cmap = plt.get_cmap("Grays")
    else:
        cmap = plt.get_cmap(cmap)

    for medium in plotdict["media"]:
        polys[medium] = PolyCollection(
            plotdict[medium]["vertices"], antialiased=antialiased
        )
        ax.add_collection(polys[medium])

        # Get the field at the first time step
        stepstr = f"Step{plotdict['steps']['start']:06d}"

        # Get the field
        fields = load_field(
            plotdict["wavefield_file"],
            medium,
            field_type,
            stepstr,
            combined=combined,
            component=component,
        )

        if verbose:
            print(f"  {medium}: {fields.shape}")
            print(f"            Min: {np.min(fields)}, Max: {np.max(fields)}")

        # Assign to field dict
        facefields[medium] = fields[plotdict[medium]["field_map"]].sum(axis=1) / 4.0

    # Compute scaling across all media
    fabs = compute_scaling(facefields)
    if combined:
        norm = plt.Normalize(vmin=-fabs, vmax=fabs)
    else:
        norm = plt.Normalize(vmin=0, vmax=fabs)

    # Set the facecolors to white
    for medium in plotdict["media"]:
        # Set the facecolors
        polys[medium].set_facecolors(cmap(norm(np.zeros(len(facefields[medium])))))
        polys[medium].set_edgecolors(edgecolors)

    sc = ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(sc, ax=ax)

    plt.show(block=False)
    plt.pause(0.1)

    if output_dir is not None:
        print("  Saving ...")

        glob_files = glob.glob(f"{output_dir}/*.{output_type.lower()}")
        for f in glob_files:
            if os.path.isfile(f):
                os.remove(f)

        f"Step{0:06d}"
        plt.savefig(f"{output_dir}/{stepstr}.{output_type.lower()}", dpi=dpi)

    for it in range(
        plotdict["steps"]["start"],
        plotdict["steps"]["end"] + 1,
        plotdict["steps"]["delta"],
    ):
        print(f"Processing timestep {it:0>5d} of {plotdict['steps']['end']:0>5d}")
        print("---------------------------------")
        print("  Reading ...")

        # Loop over the media
        for medium in plotdict["media"]:
            # Get the field at the first time step
            stepstr = f"Step{it:06d}"

            # Get the field
            field = load_field(
                plotdict["wavefield_file"],
                medium,
                field_type,
                stepstr,
                combined=combined,
                component=component,
            )

            # Compute the facefield
            facefields[medium] = field[plotdict[medium]["field_map"]].sum(axis=1) / 4.0

        # Compute scaling across all media
        fabs = compute_scaling(facefields)
        if combined:
            norm = plt.Normalize(vmin=-fabs, vmax=fabs)
            sc.set_clim(vmin=-fabs, vmax=fabs)
        else:
            norm = plt.Normalize(vmin=0, vmax=fabs)
            sc.set_clim(vmin=-fabs, vmax=fabs)

        # Set the facefolor
        # Set the facecolors to white
        for medium in plotdict["media"]:
            # Set the facecolors
            polys[medium].set_facecolors(cmap(norm(facefields[medium])))
            polys[medium].set_edgecolors(edgecolors)

        if output_dir is not None:
            print("  Saving ...")
            plt.savefig(f"{output_dir}/{stepstr}.{output_type.lower()}", dpi=dpi)

        plt.pause(0.1)

    return fig, ax, polys


if __name__ == "__main__":
    # get command line arguments
    if len(sys.argv) > 1:
        wavefield_file = sys.argv[1]
    else:
        wavefield_file = (
            "examples/fluid-solid-interface/OUTPUT_FILES/wavefield/ForwardWavefield.h5"
        )
        # sys.exit("Please provide the wavefield file to plot, e.g. python plot_wavefield.py wavefield_file.h5")

    # Get the output directory if provided
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    # Get the plot dictionary
    plotdict = get_plot_dict(wavefield_file, verbose=True)

    # Compute the
    fig, ax, polys = plot_wavefield(
        plotdict,
        "Comboplot",
        combined=True,
        component=0,
        cmap=None,
        field_type="Displacement",
        output_dir=output_dir,
    )

    plt.show(block=True)
