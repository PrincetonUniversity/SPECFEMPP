# `gmsh2meshfem`

`gmsh2meshfem` is a library that reads and exports a mesh generated in `gmsh`.

## Model Builder

Given a generated mesh in a running `gmsh` instance, a `Model` can be created by the surface tags. This model can then be passed into an `Exporter2D` instance to generate the files.

`gmsh` differentiates *entities* (model-defining structures, pre-meshing, see [`gmsh/model` namespace](https://gmsh.info/doc/texinfo/gmsh.html#Namespace-gmsh_002fmodel)) and *elements* (mesh-side objects, can be points, lines, surfaces, or volumes; a list can be found in "[MSH file format](https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format)", see [`gmsh/model/mesh` namespace](https://gmsh.info/doc/texinfo/gmsh.html#Namespace-gmsh_002fmodel_002fmesh)). We use that convention here.

- `Model.from_meshed_surface` generates a `Model` instance for each surface entity tag, passed as a list or a single value.
  - For now, only 9-node quadrilateral elements are supported (`MSH_QUA_9`). Node locations and elements' node tags are taken directly from `gmsh`.
  - Material IDs are chosen by the layer (1 is the lowest, then 2, etc.)
  - boundaries store the pairs (element ID, edge) for all edges on the boundary.
  - conforming interfaces are computed by matching interior nodes.
  - nonconforming interfaces are found using a recursive subdivision algorithm for intersection detection.

- `Exporter2D` takes the completed model, and writes out the external mesh files
  - Currently, no free surface boundaries are exported, so the resultant simulation will be with full natural (Neumann) boundaries.

### `Model` Object

The `Model` object should not be confused with the `gmsh` model.

The desire of this object is to store the data relevant to mesh exporting without needing access to an active `gmsh` environment. After reading in a mesh from `gmsh` with `Model.from_meshed_surface`, one should be able to close `gmsh` ([`gmsh.finalize()`](https://gmsh.info/doc/texinfo/gmsh.html#index-gmsh_002ffinalize)) and still have access to the mesh and be able to read, process, and ultimately export its data. Theoretically, this would mean that `Model` can be used for a general external mesher, say `cubit`, as an intermediary.
