# `gmshlayerbuilder`

Converts a topography file (used by the internal mesher) into a set of output files to be read in by `meshfem2D` (for `gmshlayerbuilder 2d`) or `meshfem3D` (for `gmshlayerbuilder 3d`).
A list of options is provided with the `-h` flag.

> All commands are assumed to be run in the `scripts` directory.

This script can convert a topography file `topography.dat` into external mesh files in a directory `outputs` using

```sh
python gmshlayerbuilder 2d topography.dat outputs/
```

More information can be found using

```sh
python gmshlayerbuilder -h
python gmshlayerbuilder 2d -h
python gmshlayerbuilder 3d -h
```

A test file has been placed in the `gmshlayerbuilder` directory for a demo. Running the following command generates the files read in by `meshfem`, which are placed in the `results` directory.

```sh
python gmshlayerbuilder 2d gmshlayerbuilder/test_topo.dat ../results/gmsh_demo/
```

## Plotting

The output can be viewed with the inclusion of a `--plot` flag. `matplotlib` must be installed to do so.

```sh
python gmshlayerbuilder 2d gmshlayerbuilder/test_topo.dat ../results/gmsh_demo/ --plot
```

## Setting boundaries

Boundary conditions can be set on the top, bottom, left, and right sides of the domain using `--top`, `--bottom`, `--left`, and `--right` respectively. The available conditions are `neumann`, `acoustic_free_surface`, and `absorbing`. Default is `neumann` for all sides.

```sh
python gmshlayerbuilder 2d gmshlayerbuilder/test_topo.dat ../results/gmsh_demo/ --top acoustic_free_surface
```

Note that `gmshlayerbuilder` is not aware of the material by default. Currently, there is only one way of telling which material type each layer is:

- `--materials [string-code]`: `string-code` is a case-insensitive string of material-type characters, which must have a length equal to the number of layers.
  - `S`: solid
  - `F`: fluid

  Each layer's material type is set according to this string, from bottom to top. For example, `--materials SF` places a fluid layer on top of a solid layer.

If materials are not specified, then for the purposes of setting the `acoustic_free_surface` condition, all layers are treated as fluid and given the `acoustic_free_surface` tag when relevant to the top, bottom, left, and right sides of the domain.

## Development Notes

The code is split into two core parts. If / when we separate the "[Model Builder](#model-builder)" section into its own script (say, to replace `gmsh2meshfem`), the section below for that code can be moved with it.

### Topography Reader

A topography file is read in `topo_reader.py`, recovering a set of layer boundaries and the number of cells in the vertical direction between them. A `LayeredBuilder` object is created, which stores the boundaries as piecewise linear. Each layer is assigned a horizontal cell resolution. The number of cells `nx` in the horizontal direction is chosen to have the cell aspect ratios closest to one. The left and right walls are chosen to be the minimum and maxiumum x-values given in the topography file. Calling `builder.create_model()` creates the mesh in `gmsh` and calls the routine of the second part.

- `topo_reader.builder_from_topo_file` generates the `LayerBuilder`
  - each interface in the topography file gives a set of points. These points are read into `LayerBoundary` objects.
  - `xlow` and `xhigh` specify the left and right walls. These are set by the min and max values among the points.
  - `Layer` objects store the cell resolution. `nz` is given by the topography file. `nx` is chosen to approximate an aspect ratio of 1 for the cells of each layer.
- `create_model` takes the `LayerBuilder` and generates the mesh before calling the model builder routine and returning the result.
  - Each `LayerBoundary` creates a `BuildResult` that stores the `gmsh` entity tags of the created geometry.
  - `Layer` generates the left and right walls, then takes the boundary geometry to create a "surface" geometry.
  - The list of surface tags are passed to the routine below.

### Model Builder

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

#### `Model` Object

The `Model` object should not be confused with the `gmsh` model.

The desire of this object is to store the data relevant to mesh exporting without needing access to an active `gmsh` environment. After reading in a mesh from `gmsh` with `Model.from_meshed_surface`, one should be able to close `gmsh` ([`gmsh.finalize()`](https://gmsh.info/doc/texinfo/gmsh.html#index-gmsh_002ffinalize)) and still have access to the mesh and be able to read, process, and ultimately export its data. Theoretically, this would mean that `Model` can be used for a general external mesher, say `cubit`, as an intermediary.
