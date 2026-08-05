"""Generate a small structured hex cube mesh for the 8-partition MPI fixture.

A 4 x 4 x 4 element cube (64 elements). When decomposed with METIS into 8
partitions (NPROC=8, Par_file) the optimal cut is the 2x2x2 octant split, which
produces the *general* set of MPI interfaces that the internal mesher never
emits: top/bottom faces, horizontal edges, and single-node corner connections
(the central node shared by all 8 octants). This is what the
``assembly_mpi_dim3_8proc`` regression test exercises.

Run from this directory (``provenance/``):

    uv run --group scripts python create_mesh.py
"""

import os

import gmsh

HERE = os.path.dirname(os.path.abspath(__file__))

## Cube: x,y,z in [0, 4000]; free surface on top (z = 4000), absorbing elsewhere.
L = 4000.0
ELEM = 1000.0  ## -> 4 elements per edge

gmsh.initialize()
gmsh.model.add("elastic_cube_2x2x2")

box = gmsh.model.occ.addBox(0, 0, 0, L, L, L)
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(3, [box], name="material_1")

## Label each bounding face by its position so export can emit boundary files.
eps = 1.0
for dim, tag in gmsh.model.getEntities(2):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    if xmax < eps:
        gmsh.model.addPhysicalGroup(2, [tag], name="abs_xmin")
    elif xmin > L - eps:
        gmsh.model.addPhysicalGroup(2, [tag], name="abs_xmax")
    elif ymax < eps:
        gmsh.model.addPhysicalGroup(2, [tag], name="abs_ymin")
    elif ymin > L - eps:
        gmsh.model.addPhysicalGroup(2, [tag], name="abs_ymax")
    elif zmax < eps:
        gmsh.model.addPhysicalGroup(2, [tag], name="abs_bottom")
    else:  ## z = L: free surface
        gmsh.model.addPhysicalGroup(2, [tag], name="free_or_abs_zmax")

## Structured transfinite hex mesh.
for dim, tag in gmsh.model.getEntities(1):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    length = max(abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin))
    n = round(length / ELEM) + 1
    gmsh.model.mesh.setTransfiniteCurve(tag, n)

for dim, tag in gmsh.model.getEntities(2):
    gmsh.model.mesh.setTransfiniteSurface(tag)
    gmsh.model.mesh.setRecombine(dim, tag)

gmsh.model.mesh.setTransfiniteVolume(box)

gmsh.model.mesh.generate(3)
gmsh.write(os.path.join(HERE, "cube.msh"))
gmsh.finalize()
print("Mesh written to cube.msh")
