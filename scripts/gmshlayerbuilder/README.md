# `gmshlayerbuilder`

Converts a topography file (used by the internal mesher) into a set of output files to be read in by `meshfem2D`.

> All commands are assumed to be run in the `scripts` directory.

This script can convert a topography file `topography.dat` into external mesh files in a directory `outputs` using

```sh
python gmshlayerbuilder topography.dat outputs/
```

More information can be found using

```sh
python gmshlayerbuilder -h
```

A test file has been placed in the `gmshlayerbuilder` directory for a demo. Running the following command generates the files read in by `meshfem`, which are placed in the `results` directory.

```sh
python gmshlayerbuilder gmshlayerbuilder/test_topo.dat ../results/gmsh_demo/
```

## Plotting

The output can be viewed with the inclusion of a `--plot` flag. `matplotlib` must be installed to do so.

```sh
python gmshlayerbuilder gmshlayerbuilder/test_topo.dat ../results/gmsh_demo/ --plot
```

## Development Notes

The code is split into two core parts.

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

- `Model.from_meshed_surface` generates a `Model` instance for each surface, passed as a list or a single value.
  - For now, only 9-node quadrilateral elements are supported (`MSH_QUA_9`). Node locations and elements' node tags are taken directly from `gmsh`.
  - Material IDs are chosen by the layer (1 is the lowest, then 2, etc.)
  - boundaries store the pairs (element ID, edge) for all edges on the boundary.
  - conforming interfaces are computed by matching interior nodes.
  - nonconforming interfaces are found using a recursive subdivision algorithm for intersection detection.

- `Exporter2D` takes the completed model, and writes out the external mesh files
  - Currently, no free surface boundaries are exported, so the resultant simulation will be with full natural (Neumann) boundaries.
