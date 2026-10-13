# InjectionAcousticElasticBox

A small acoustic-over-elastic box for unit-testing FK wavefield injection at the
Stacey boundary GLL points of a fluid-solid model (issue #2123, Phase 2).

## Geometry & material

- Domain: 4000 m (x) × 4000 m (y) × 4000 m (z), `Z_MAX = 0` (free surface),
  `Z_MIN = -4000`.
- Mesh: 4 × 4 × 4 = 64 spectral elements (1000 m edges).
- Layering: a 2000 m acoustic (fluid) column on top (`Z` in `[-2000, 0]`) over a
  2000 m elastic block (`Z` in `[-4000, -2000]`); a single fluid layer, so the
  model is immune to the multi-fluid propagator issue (#2139).
- Boundaries: free surface on top; Stacey absorbing on the four lateral faces
  and the bottom.
- Materials:
  - acoustic: `rho = 1020`, `vp = 1500` (seawater), domain id 1;
  - elastic: `rho = 2700`, `vp = 3000`, `vs = 1732.051`, domain id 2.

These match a `LayeredModel` with one acoustic layer (`thickness = 2000`) over
an elastic half-space, so the injection model-vs-mesh consistency check passes.

## Generation

Produced by the SPECFEM++ internal mesher from the inputs in
`provenance/specfempp/`. To regenerate:

```
SPECFEMPP_BINDIR=<build>/bin snakemake -c1
```

See `Snakefile`.
