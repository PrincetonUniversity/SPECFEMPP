# global_small_6rank fixture

Six-rank globe fixture for the MPI interfaces of `specfem::assembly::mpi<dim3>`
(issue #2051): one rank per cubed-sphere chunk. The configuration is the
`benchmarks/src/dim3_globe/global_small_mesh` benchmark (PR #2011):

- `NCHUNKS = 6`, `NPROC_XI = NPROC_ETA = 1` -- six ranks
- `NEX_XI = NEX_ETA = 32`, the smallest six-chunk mesh the mesher accepts
  (`NEX = 16` stops with "NEX_XI must be greater to cut the sphere into slices
  with positive Jacobian")
- `MODEL = 1D_isotropic_prem`, ellipticity on; topography, gravity, rotation and
  attenuation off
- `SPECFEMPP_DATABASE = .true.`, so only the thin SPECFEM++ databases are written

Only the inputs are committed, under `provenance/DATA/`. The six databases are
~47 MB in total, so they are generated rather than committed; the mesher takes a
few seconds.

## How the tests get the databases

In an MPI build, `tests/unit-tests/mpi.cmake` registers the CTest fixture
`globe_small_6rank_mesh`. It runs `xmeshfem3D_globe` on six ranks under the same
launcher as the MPI tests (so at test time, never during the build) in

```text
<SPECFEM_TEST_OUTPUT_DIR>/generated/dim3_globe/global_small_6rank/
```

and CTest runs it before `assembly_mpi_dim3_globe_tests`, which reads
`generated/dim3_globe/global_small_6rank/DATABASES_MPI/proc??????_specfempp_database.bin`.
Running `ctest -R assembly_mpi_dim3_globe` is enough; the fixture is pulled in
automatically.

## Regenerate by hand

From this directory, with `xmeshfem3D_globe` built under `SPECFEMPP_BINDIR` and
an MPI launcher on `PATH` (override with `MPIEXEC`):

```bash
SPECFEMPP_BINDIR=/path/to/specfempp/bin uv run --group scripts snakemake -c6
```

The outputs (`DATABASES_MPI/`, `OUTPUT_FILES/`, ...) are git-ignored.
