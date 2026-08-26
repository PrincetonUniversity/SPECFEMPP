# SPECFEM3D_GLOBE mesh reader fixture

This unit-test fixture creates a minimal six-chunk global mesh with
`xmeshfem3D_globe` and reads all six partitions with SPECFEM++'s production
globe-mesh reader.

It is intentionally mesher-only:

- `NCHUNKS = 6`
- `NEX_XI = NEX_ETA = 32`
- `NPROC_XI = NPROC_ETA = 1` (so the run uses 6 MPI ranks, one per chunk)
- `MODEL = 1D_isotropic_prem`
- ellipticity is enabled to exercise reference-anchor output; surface topography
  remains disabled because its external ETOPO dataset is not part of this fixture
- gravity, rotation, and attenuation are disabled

`SPECFEMPP_DATABASE = .true.` is set, so the mesher writes only the thin SPECFEM++
mesh database (`DATABASES_MPI/proc??????_specfempp_database.bin`) and skips its
native full-mesh databases. The unit test validates record framing, model
configuration, node and element connectivity, local adjacency, boundary data,
and cross-rank MPI interface matching.

Because attenuation is disabled here, the run does not exercise the validator's
checks on the attenuation period band; those need a Par_file with
`ATTENUATION = .true.`.

Generate or refresh the thin databases from this directory:

```bash
SPECFEMPP_BINDIR=/path/to/specfempp/bin \
  uv run --group scripts snakemake --cores 1
```

Build and run the MPI unit test with:

```bash
cmake --build build/release-mpi --target io_mesh_dim3_globe_tests
ctest --test-dir build/release-mpi/tests/run \
  -R ReadGlobeMeshTests.GlobalSmallMesh --output-on-failure
```
