# GlobalSmallMeshAttenuation fixture

Single-partition serial fixture for SPECFEM3D_GLOBE attenuation coverage. The
fixture is generated with `xmeshfem3D_globe` and is used to build a full globe
assembly with oracle-sampled per-GLL Q.

## Configuration

Identical to `../GlobalSmallMesh` except:

- `ATTENUATION = .true.`
- `NEX_XI = NEX_ETA = 16` to keep the database small and the assembly fast

The remaining settings are:

- `NCHUNKS = 1`
- `NPROC_XI = NPROC_ETA = 1` so the run uses one process without MPI
- `MODEL = 1D_isotropic_prem` (no transversely isotropic elements, so every
  elastic element carries a `constant_isotropic` attenuation container)
- ellipticity is enabled; surface topography, gravity and rotation are disabled

The mesher selects the attenuation period band from the mesh resolution; for
this configuration it is `[277.987305, 15632.3760] s`. Tests read the band
back from the database rather than hard-coding it.

`SPECFEMPP_DATABASE = .true.` is set, so the mesher writes only the thin
SPECFEM++ mesh database. The committed fixture is:

```text
DATABASES_MPI/proc000000_specfempp_database.bin
```

The generation inputs are kept under `provenance/DATA/`.

## Regenerate

From this directory, with `xmeshfem3D_globe` built under `SPECFEMPP_BINDIR`:

```bash
SPECFEMPP_BINDIR=/path/to/specfempp/bin uv run --group scripts snakemake -c1
```

`DATA/`, `OUTPUT_FILES/`, `.snakemake/`, and `executable_checked.txt` are
regenerable intermediates and are not committed.
