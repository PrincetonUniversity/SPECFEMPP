# InjectionStaceyBox

A small homogeneous elastic box for unit-testing FK wavefield injection at the
Stacey boundary GLL points (issue #2123).

## Geometry & material

- Domain: 4000 m (x) × 4000 m (y) × 3000 m (z), `Z_MAX = 0` (free surface),
  `Z_MIN = -3000`.
- Mesh: 4 × 4 × 3 = 48 spectral elements (1000 m edges).
- Boundaries: free surface on top; Stacey absorbing on the four lateral faces
  and the bottom.
- Material: a single elastic Poisson solid, `rho = 2700`, `vp = 3000`,
  `vs = 1732.051`. This matches a one-layer elastic half-space `LayeredModel`,
  so the injection model-vs-mesh consistency check passes everywhere.

## Generation

Produced by the SPECFEM++ internal mesher from the inputs in
`provenance/specfempp/` (`Mesh_Par_file`, `interfaces.txt`, `interface1.txt`).
To regenerate:

```
SPECFEMPP_BINDIR=<build>/bin snakemake -c1
```

See `Snakefile`. No Fortran reference database is generated: the unit test
validates the FK extractor and provider against a self-consistent FK solve.
