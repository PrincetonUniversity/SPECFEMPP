## Nonconforming-marked Conforming mesh: Curved

`Nonconforming-marked Conforming mesh`es are test meshes that are conforming, but are labelled as nonconforming for the purposes of the database and `specfem`. These meshes help compare conforming and nonconforming kernels, especially when they should be equivalent.

### Curved

An 8 x 8 grid of elements with the bottom half as elastic and the top half as acoustic. This mesh has a sinusoidal surface of 1 wavelength and amplitude of 20% of the domain.

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
