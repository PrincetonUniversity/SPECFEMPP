# FK injection through two elastic layers

A plane P wave enters a box from below through the FK injection plugin and travels
up through a half-space into a slower 30 km layer. The traces are compared with
SPECFEM3D's FK coupling on the same mesh.

| | |
|---|---|
| Box | 300 km × 200 km × 120 km, 12 × 8 × 8 elements |
| Layer | rho 2500, vp 2500, vs 1250, down to −30 km |
| Half-space | rho 2700, vp 3000, vs 1732.051 |
| Wave | P, take-off 20°, back-azimuth 30°, half-duration 5 s |
| Run | GLL4, dt 0.3 s, 600 steps, t0 = −12 s |
| Boundaries | Stacey and injection on the sides and bottom, free surface on top |

The force in `source.yaml` has zero amplitude and only sets t0.

## Running it

Configure SPECFEM++ with `-DSPECFEM_BUILD_BENCHMARKS=ON` and build `specfem` and
`xmeshfem3D`, then in the benchmark's build directory:

```bash
uv run snakemake -j 1
```

This builds the plugin against the installed SDK, meshes, runs the solver, and writes
`OUTPUT_FILES/comparison.txt` and `OUTPUT_FILES/comparison.png`.

## References

`reference_seismograms/` holds SPECFEM3D's traces. `provenance/fortran/DATA/` holds the
inputs that produced them, with `COUPLE_WITH_INJECTION_TECHNIQUE` on and
`INJECTION_TECHNIQUE_TYPE = 3`.

## Results

Measured on Apple silicon with the release preset; the time loop takes about 2.4 s. The
error is the Newmark displacement tests' number: one per station, over all components
and types. Correlation and lag are over all traces.

| `fk.yaml` | Worst error | Lowest correlation | Lag |
|---|---|---|---|
| As given | 7.2e-3 | 0.99986 | −0.032 to −0.022 s |
| `reproduce-specfem3d-reference: true` | 3.6e-3 | 0.99993 | −0.006 to 0.004 s |

The lag is SPECFEM3D's N/(N − 1) stretch of its delays, which the plugin leaves out by
default.
