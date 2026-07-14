## Nonconforming-marked Conforming mesh: Flat

`Nonconforming-marked Conforming mesh`es are test meshes that are conforming, but are labelled as nonconforming for the purposes of the database and `specfem`. These meshes help compare conforming and nonconforming kernels, especially when they should be equivalent.

### Flat

A 4 x 4 grid of elements with the bottom two rows as elastic and the top two as acoustic.

To regenerate the database, convert the topography file into the files in `MESH` through `gmsh`, then run meshfem over `Par_file`:

```bash
python scripts/gmshlayerbuilder 2d simple_dg_topo.dat MESH
xmeshfem2D -p Par_file
```

Relative to `provenance`:

```bash
python ../../../../../../scripts/gmshlayerbuilder 2d simple_dg_topo.dat MESH
../../../../../../bin/xmeshfem2D -p Par_file
```
