import gmsh

gmsh.initialize()
gmsh.model.add("halfspace_3d")

## Create a 10 km x 10 km x 5 km box: x:[0,10000], y:[0,10000], z:[-5000,0]
box = gmsh.model.occ.addBox(0, 0, -5000, 10000, 10000, 5000)
gmsh.model.occ.synchronize()

## Volume group for the elastic material
gmsh.model.addPhysicalGroup(3, [box], name="material_1")

## Use the bounding box of each surface to identify and label each face
eps = 1.0
for dim, tag in gmsh.model.getEntities(2):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    if xmax < eps:
        gmsh.model.addPhysicalGroup(2, [tag], name="abs_xmin")
    elif xmin > 10000 - eps:
        gmsh.model.addPhysicalGroup(2, [tag], name="abs_xmax")
    elif ymax < eps:
        gmsh.model.addPhysicalGroup(2, [tag], name="abs_ymin")
    elif ymin > 10000 - eps:
        gmsh.model.addPhysicalGroup(2, [tag], name="abs_ymax")
    elif zmax < -5000 + eps:
        gmsh.model.addPhysicalGroup(2, [tag], name="abs_bottom")
    else:  ## z = 0: free surface
        gmsh.model.addPhysicalGroup(2, [tag], name="free_or_abs_zmax")

## Structured transfinite hex mesh at 500 m element size
for dim, tag in gmsh.model.getEntities(1):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    length = max(abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin))
    n = round(length / 500) + 1
    gmsh.model.mesh.setTransfiniteCurve(tag, n)

for dim, tag in gmsh.model.getEntities(2):
    gmsh.model.mesh.setTransfiniteSurface(tag)
    gmsh.model.mesh.setRecombine(dim, tag)

gmsh.model.mesh.setTransfiniteVolume(box)

gmsh.model.mesh.generate(3)
gmsh.write("halfspace.msh")
gmsh.finalize()
print("Mesh written to halfspace.msh")
