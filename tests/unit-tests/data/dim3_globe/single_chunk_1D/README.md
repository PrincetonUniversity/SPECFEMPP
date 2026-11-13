# single_chunk_1D fixture

Single-partition serial **spherical** globe fixture for SPECFEM++ source/receiver
placement and `locate_point` coverage. Generated with `xmeshfem3D_globe` and read
by SPECFEM++'s production globe-mesh reader.

## Configuration

A single 90°×90° chunk of a 1D PREM Earth, centred at latitude 40°, longitude
10°:

- `NCHUNKS = 1`
- `NEX_XI = NEX_ETA = 32`
- `NPROC_XI = NPROC_ETA = 1` so the run uses one process without MPI
- `MODEL = 1D_isotropic_prem`
- **`ELLIPTICITY = .false.`** — the figure is a perfect sphere, so a source or
  receiver resolves through the spherical geographic→geocentric path. (The
  elliptical surface is issue #2058; the assembly coordinate resolution throws
  on an elliptical globe until then, which is why this placement fixture is
  spherical.)
- surface topography, oceans, gravity, rotation, and attenuation are disabled

`SPECFEMPP_DATABASE = .true.`, so the mesher writes only the thin SPECFEM++ mesh
database and skips its native full-mesh databases. The committed fixture is:

```text
DATABASES_MPI/proc000000_specfempp_database.bin
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

The globe placement/locate tests read the database through the standard
unit-test runtime data path:

```text
data/dim3_globe/single_chunk_1D/DATABASES_MPI/proc??????_specfempp_database.bin
```
