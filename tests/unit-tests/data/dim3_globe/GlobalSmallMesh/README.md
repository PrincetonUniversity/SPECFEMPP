# GlobalSmallMesh fixture

Six-partition MPI test fixture for SPECFEM3D_GLOBE mesh-reader coverage. The
fixture is generated with `xmeshfem3D_globe` and read by SPECFEM++'s production
globe-mesh reader.

## Configuration

This is intentionally mesher-only:

- `NCHUNKS = 6`
- `NEX_XI = NEX_ETA = 32`
- `NPROC_XI = NPROC_ETA = 1` so the run uses 6 MPI ranks, one per chunk
- `MODEL = 1D_isotropic_prem`
- ellipticity is enabled to exercise reference-anchor output
- surface topography is disabled because its external ETOPO dataset is not part
  of this fixture
- gravity, rotation, and attenuation are disabled

`SPECFEMPP_DATABASE = .true.` is set, so the mesher writes only the thin
SPECFEM++ mesh database and skips its native full-mesh databases. The committed
fixture is:

```text
DATABASES_MPI/proc000000_specfempp_database.bin
...
DATABASES_MPI/proc000005_specfempp_database.bin
```

The generation inputs are kept under `provenance/DATA/`.

## Regenerate

From this directory, with `xmeshfem3D_globe` built under `SPECFEMPP_BINDIR`:

```bash
SPECFEMPP_BINDIR=/path/to/specfempp/bin uv run --group scripts snakemake -c1
```

The workflow copies `provenance/DATA/` to a temporary local `DATA/` directory
because `xmeshfem3D_globe` reads that path from its working directory. `DATA/`,
`OUTPUT_FILES/`, `.snakemake/`, and `executable_checked.txt` are regenerable
intermediates and are not committed.

The corresponding unit test reads these files through the standard unit-test
runtime data path:

```text
data/dim3_globe/GlobalSmallMesh/DATABASES_MPI/proc??????_specfempp_database.bin
```
