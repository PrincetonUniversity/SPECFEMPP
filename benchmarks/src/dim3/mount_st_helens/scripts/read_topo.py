"""Load the Mount St. Helens topography and build an elevation interpolator.

This is the Gmsh-workflow replacement for the CUBIT ``read_topo.py`` that came
with the original SPECFEM3D example.  Instead of skinning a CUBIT surface from
the topography points, we expose a simple ``topo_elevation(x, y)`` callable that
``create_mesh.py`` uses to warp a structured box mesh onto the terrain.

The topography file ``ptopo.mean.utm`` holds one ``X Y Z`` triplet per line in
UTM zone 10 coordinates (easting, northing, elevation in metres).
"""

import os

import numpy as np
from scipy.interpolate import NearestNDInterpolator, CloughTocher2DInterpolator

TOPO_FILE = "ptopo.mean.utm"


def load_topography(path=None):
    """Read the topography point cloud.

    Returns
    -------
    points : (N, 2) ndarray
        UTM easting/northing of each topography sample.
    elevation : (N,) ndarray
        Elevation (m) of each sample.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TOPO_FILE)
    data = np.loadtxt(path)
    return data[:, :2], data[:, 2]


def topography_interpolator(path=None):
    """Build a ``topo_elevation(x, y)`` callable over the topography samples.

    Linear interpolation is used inside the convex hull of the samples; nodes
    that fall outside it (the meshed box footprint slightly overhangs the data)
    fall back to nearest-neighbour so every node receives a finite elevation.
    """
    points, elevation = load_topography(path)
    linear = CloughTocher2DInterpolator(points, elevation)
    nearest = NearestNDInterpolator(points, elevation)

    def topo_elevation(x, y):
        z = linear(x, y)
        outside = np.isnan(z)
        if np.any(outside):
            z = np.where(outside, nearest(x, y), z)
        return z

    return topo_elevation


if __name__ == "__main__":
    pts, elev = load_topography()
    print(f"Loaded {len(elev)} topography points from {TOPO_FILE}")
    print(f"  easting  range: {pts[:, 0].min():.1f} .. {pts[:, 0].max():.1f} m")
    print(f"  northing range: {pts[:, 1].min():.1f} .. {pts[:, 1].max():.1f} m")
    print(f"  elevation range: {elev.min():.1f} .. {elev.max():.1f} m")
