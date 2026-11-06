# FK injection into an ocean over crust and mantle

A plane P wave enters a box from below through the FK injection plugin, crosses the
mantle and the oceanic crust and reverberates in a 4 km ocean, where a hydrophone at
1500 m depth records pressure, as a MERMAID float would. This is the 3D, injection-based
counterpart of the 2D `fluid-solid-bathymetry` benchmark, where the incident plane wave
is built from a line of sources. The pressure trace is compared with SPECFEM3D's FK
coupling on the same mesh.

| | |
|---|---|
| Box | 10 km × 10 km × 20 km, 15 × 15 × 30 elements (about 670 m) |
| Ocean (acoustic) | rho 1020, vp 1500, 0 to −4 km |
| Crust | rho 2900, vp 6800, vs 3900, −4 to −11 km |
| Mantle half-space | rho 3380, vp 8100, vs 4500, below −11 km |
| Wave | P, take-off 15°, back-azimuth 30°, half-duration 0.5 s (`frequency-max: 2`) |
| Hydrophone | `H` at the box centre, z = −1500 m, pressure |
| Run | GLL4, dt 0.005 s, 6000 steps (30 s) |
| Boundaries | Stacey and injection on the sides and bottom, free surface on top |

The force in `source.yaml` has zero amplitude and only sets t0. The water column
reverberates with a period of 2h/c ≈ 5.3 s, which the trace shows after the direct
arrival.

## Running it

Configure SPECFEM++ with `-DSPECFEM_BUILD_BENCHMARKS=ON` and build `specfem`,
`xmeshfem3D` and the `fk_injection` plugin, then in the benchmark's build directory:

```bash
uv run snakemake -j 1
```

This meshes, runs the solver with the plugin, and writes
`OUTPUT_FILES/comparison.txt` and `OUTPUT_FILES/comparison.png`.

## References

`reference_seismograms/` holds SPECFEM3D's pressure trace. `provenance/fortran/DATA/`
holds the inputs that produced it, with `COUPLE_WITH_INJECTION_TECHNIQUE` on and
`INJECTION_TECHNIQUE_TYPE = 3`; the fluid layer is the `LAYER` with vs = 0 in
`FKmodel`. SPECFEM3D's acoustic FK coupling adds
(u_FK · n + χ̇_FK / (ρ v_p)) dS on the acoustic absorbing faces, the term SPECFEM++'s
`stacey` injection method applies at acoustic points.

## Results

Measured on Apple silicon with the release preset; the time loop takes about 4 minutes.
The error is the Newmark displacement tests' number, over the pressure trace. The FK
series are stored at the simulation rate (`frequency-sampling: 200`, `FREQUENCY_SAMPLING
200.` in `FKmodel`), so SPECFEM3D's resampling rate is 1 and neither its inverse-FFT
normalization nor its N/(N − 1) delay stretch shows; `reproduce-specfem3d-reference: true`
therefore gives the same numbers.

| `fk.yaml` | Error | Correlation | Ratio | Lag |
|---|---|---|---|---|
| As given | 3.7e-3 | 0.99999 | 0.9996 | 0.000 s |

The mesh resolves periods down to about 0.56 s (SPECFEM3D's mesher estimate), so the
0.5 s half-duration, whose Gaussian is below 1% above 1.4 Hz, is close to what it
allows. A shorter pulse needs a finer mesh. The origin SPECFEM3D derives depends on the
pulse length: changing `FREQUENCY_MAX` changes the `origin` to copy into `fk.yaml`.

With a coarser FK sampling (`frequency-sampling: 10`) SPECFEM3D's amplitudes come out
larger by its resampling rate (20 here), as described in the plugin's README.
