# `interfaces_only/*`

Each case in `interfaces_only` features a mesh with a large surface-area:volume ratio,
used to test couplings.

## acoustic_elastic_M-N

A flat, square in `x`-`y`, thin in `z` block of NxNx2 acoustic elements are laid on top of a block
of MxMx2 elastic elements, all with aspect ratio 1. These blocks fill $[-1000,1000]^2$ in
$x$ and $y$, with the bottom of the elastic block starting at $z = 0$.

Two elements (50 x 50) are placed above a larger element (100 x 100), as below:

```none
┌──────────┐ - z = 2*elem_size_elastic + 2*elem_size_acoustic
│          │
│ acoustic │
├──────────┤ - z = 2*elem_size_elastic
│ elastic  │
│          │
└──────────┘ - z = 0
```

To regenerate the database, convert the interfaces file into the files in `provenance` through `gmsh`,
then run xdecompose_mesh over `xdecompose_par_file`:

```bash
python scripts/gmshlayerbuilder 3d --depth_block_km 0 "${PROVENANCE_DIR}/interface_files/interfaces.txt" "${PROVENANCE_DIR}/meshfiles"
xmeshfem2D -p ${PROVENANCE_DIR}/xdecompose_par_file
```

Relative to `interfaces_only/acoustic_elastic_M-N/provenance`:

```bash
python ../../../../../../../../SPECFEMPP/scripts/gmshlayerbuilder 3d --depth_block_km 0 "interface_files/interfaces.txt" "meshfiles"
../../../../../../../bin/xdecompose_mesh -p xdecompose_par_file
```
