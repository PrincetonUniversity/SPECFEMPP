# GlobalSphereMesh fixture

Six-chunk, six-partition spherical PREM fixture for issue #2052. Uses
`NEX_XI = NEX_ETA = 32`, `NPROC_XI = NPROC_ETA = 1`, with ellipticity,
topography, attenuation, gravity and rotation disabled. Each database is one
90-degree chunk. The original one-chunk elliptic fixture remains in
`../GlobalSmallMesh`.

`provenance/DATA/` contains the mesher inputs. `provenance/mesher.yaml` records
the independent `finalize_mesher` volume, mass, solver DT, and element count;
`provenance/finalize_mesher.txt` retains the corresponding report. The mesher
reports volume on the unit sphere and mass in kg. Its element count includes
fictitious central-cube elements; the thin databases exclude these duplicates. Tests dimensionalize volume
using the database's `PlanetConstants`, and compare region radii against the
model's discontinuity radii.

Regenerate with an MPI-enabled `xmeshfem3D_globe`:

```bash
SPECFEMPP_BINDIR=/path/to/bin uv run --group scripts snakemake -c6
```

The six-rank `globe_info_mpi_tests` test checks quadrature volume against the
mesher and sphere, integrated mass against the mesher and Earth's mass, the
suggested CFL time step against the mesher's DT, region statistics, and exact
rank agreement for every reported value and limiting location. Volume allows
1e-5 relative quadrature/roundoff error against the mesher (which uses single
precision), and 0.1% against the sphere; mass allows 0.1%.

The 27-node geometry approximates spherical interfaces between anchors. GLL
radius ranges therefore allow 0.2% interpolation error. A separate 1 m check
compares the mesher's exact per-region shell context with `PlanetConstants`,
so region tagging is checked independently of geometric interpolation.
