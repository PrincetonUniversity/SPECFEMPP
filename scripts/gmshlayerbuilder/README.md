# `gmshlayerbuilder`

Converts a topography file (used by the internal mesher) into a set of output files to be read in by `meshfem2D` (for `gmshlayerbuilder 2d`) or `xdecompose_mesh` (for `gmshlayerbuilder 3d`).
A list of options is provided with the `-h` flag.

> All commands are assumed to be run in the `scripts` directory.

This script can convert a topography file `topography.dat` (for 2D) into external mesh files in a directory `outputs` using

```sh
python gmshlayerbuilder 2d topography.dat outputs/
```

or an interfaces file `interfaces.txt` (for 3D) into external mesh files in a directory `outputs` using

```sh
python gmshlayerbuilder 3d interfaces.txt outputs/
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

In 3D, the flags `--front` and `--back` are permitted for setting the front (-y) and back (+y) boundary conditions respectively. However, it should be noted that `xdecompose_mesh` is the ultimate decider on what the boundary conditions are. Unlike in 2D, the 3D `gmshlayerbuilder` outputs the faces into the corresponding boundary files (top, bottom, etc.) so long as the condition is not `neumann`. The decision on being an absorbing boundary or free surface is handled by `xdecompose_mesh`.

Note that `gmshlayerbuilder` is not aware of the material by default. Currently, there is only one way of telling which material type each layer is:

- `--materials [string-code]`: `string-code` is a case-insensitive string of material-type characters, which must have a length equal to the number of layers.
  - `S`: solid
  - `F`: fluid

  Each layer's material type is set according to this string, from bottom to top. For example, `--materials SF` places a fluid layer on top of a solid layer.

If materials are not specified, then for the purposes of setting the `acoustic_free_surface` condition, all layers are treated as fluid and given the `acoustic_free_surface` tag when relevant to the top, bottom, left, and right sides of the domain.

## Development Notes

This script reads a topography and uses `gmsh2meshfem` to create a model.

For the time being, due to difficulties in implementation (namely a lack of being able to enforce how elements are placed across a nonconforming interface), the 3D code avoids `gmsh`.

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
