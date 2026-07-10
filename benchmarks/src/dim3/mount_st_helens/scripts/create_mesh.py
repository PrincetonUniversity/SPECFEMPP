"""Build a topography-conforming hexahedral mesh of Mount St. Helens with Gmsh.

This is the open-source replacement for the CUBIT ``mesh_mount.py`` shipped with
the original SPECFEM3D example.  The recipe is the same: take a rectangular box
under the volcano and deform its top onto the measured topography.  Here we

  1. build a structured (transfinite) hexahedral box at 500 m resolution, then
  2. warp every node vertically so the top face lands on the terrain while the
     flat bottom stays put (a "sigma"-style stretch).

The box footprint and depth match the CUBIT brick: 15 km x 22 km x 20 km centred
at UTM zone 10 (561738, 5116370).  The resulting ``mount_sthelens.msh`` is then
converted to the SPECFEM++ text mesh format by ``export_mesh.py``.
"""

import sys
import gmsh
import numpy as np

# read_topo.py is copied next to this script (see CMakeLists `file(COPY scripts)`)
# and Snakemake runs us as `python .../scripts/create_mesh.py`, so the script's
# own directory is on sys.path -- import it as a sibling module.
from read_topo import topography_interpolator

# --- Box geometry (UTM zone 10, metres) -------------------------------------
# Centred at (561738, 5116370); brick 15 km x 22 km, bottom 10 km below sea
# level.  The top (Z1) is a reference plane that the warp maps onto topography.
X0, X1 = 554238.0, 569238.0  # easting  (15 km)
Y0, Y1 = 5105370.0, 5127370.0  # northing (22 km)
# Bottom at -10 km; reference top set near the summit elevation (~2.4 km) so the
# box is ~12 km tall. That makes the transfinite 500 m spacing give ~24 vertical
# layers -- matching the reference CUBIT mesh once the top is warped onto the
# topography (otherwise a 10 km box yields only 20 layers, i.e. ~580 m elements).
Z0, Z1 = -10000.0, 2000.0  # bottom, reference top
ELEMENT_SIZE = 500.0  # target edge length (m)


def build_box():
    """Create the box, tag boundary surfaces, and mesh it with structured hexes."""
    gmsh.model.add("mount_sthelens")

    box = gmsh.model.occ.addBox(X0, Y0, Z0, X1 - X0, Y1 - Y0, Z1 - Z0)
    gmsh.model.occ.synchronize()

    # Volume group for the elastic material region.
    gmsh.model.addPhysicalGroup(3, [box], name="material_1")

    # Tag each outer face from its bounding box (robust to Gmsh's tag ordering).
    eps = 1.0
    for dim, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        if xmax < X0 + eps:
            gmsh.model.addPhysicalGroup(2, [tag], name="abs_xmin")
        elif xmin > X1 - eps:
            gmsh.model.addPhysicalGroup(2, [tag], name="abs_xmax")
        elif ymax < Y0 + eps:
            gmsh.model.addPhysicalGroup(2, [tag], name="abs_ymin")
        elif ymin > Y1 - eps:
            gmsh.model.addPhysicalGroup(2, [tag], name="abs_ymax")
        elif zmax < Z0 + eps:
            gmsh.model.addPhysicalGroup(2, [tag], name="abs_bottom")
        else:  # the z = Z1 face becomes the free surface after warping
            gmsh.model.addPhysicalGroup(2, [tag], name="free_or_abs_zmax")

    # Structured transfinite hex mesh at ELEMENT_SIZE.
    for dim, tag in gmsh.model.getEntities(1):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        length = max(abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin))
        n = round(length / ELEMENT_SIZE) + 1
        gmsh.model.mesh.setTransfiniteCurve(tag, n)

    for dim, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(dim, tag)

    gmsh.model.mesh.setTransfiniteVolume(box)
    gmsh.model.mesh.generate(3)


def warp_to_topography(path=None):
    """Move every node's z so the top face follows the terrain.

    A node originally at height ``z in [Z0, Z1]`` is mapped to
    ``Z0 + (z - Z0)/(Z1 - Z0) * (topo(x, y) - Z0)`` so the bottom (z = Z0) is
    fixed and the top (z = Z1) lands exactly on the topography.  Only
    coordinates change -- the connectivity is untouched, so the boundary-face
    resolution in ``export_gmsh3d.py`` is unaffected.
    """
    topo_elevation = topography_interpolator(path)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords).reshape(-1, 3)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    topo_z = topo_elevation(x, y)
    frac = (z - Z0) / (Z1 - Z0)
    z_new = Z0 + frac * (topo_z - Z0)

    for tag, xi, yi, zi in zip(node_tags, x, y, z_new):
        gmsh.model.mesh.setNode(int(tag), [xi, yi, zi], [])

    return topo_z.min(), topo_z.max()


def main():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    if len(sys.argv) != 3:
        print("Usage: python create_mesh.py <output.msh> <topography.utm>")
        sys.exit(1)

    MSH_FILE = sys.argv[1]
    TOPOGRAPHY_FILE = sys.argv[2]

    try:
        build_box()
        zmin, zmax = warp_to_topography(TOPOGRAPHY_FILE)
        gmsh.write(MSH_FILE)
        n_hex = len(gmsh.model.mesh.getElements(3)[1][0])
        print(f"Mesh written to {MSH_FILE}")
        print(f"  {n_hex} hexahedral elements")
        print(f"  topography elevation range: {zmin:.1f} .. {zmax:.1f} m")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
