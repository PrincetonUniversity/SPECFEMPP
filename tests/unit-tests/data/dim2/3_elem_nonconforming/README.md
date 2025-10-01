# `3_elem_nonconforming`

Two elements (50 x 50) are placed above a larger element (100 x 100), as below:

```none
┌────┬────┐
│ 2  │  3 │
├────┴────┤
│         │
│    1    │
└─────────┘
```

To regenerate the database, convert the topography file into the files in `MESH` through `gmsh`, then run meshfem over `Par_file`:

```bash
python scripts/gmshlayerbuilder simple_dg_topo.dat MESH
xmeshfem2D -p Par_file
```
