# GlobalSmallMesh fixture

Single-partition serial fixture for SPECFEM3D_GLOBE mesh-reader coverage. The
fixture is generated with `xmeshfem3D_globe` and read by SPECFEM++'s production
globe-mesh reader.

## Configuration

This is intentionally mesher-only:

- `NCHUNKS = 1`
- `NEX_XI = NEX_ETA = 32`
- `NPROC_XI = NPROC_ETA = 1` so the run uses one process without MPI
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

## Spherical coordinate reference

`spherical_coordinates.txt` contains six corner GLL samples as
`raw_element_index radius_over_r_planet colatitude longitude` (zero-based
indices, radians). These were generated with the unmodified
`xyz_2_rthetaphi_dble` and `reduce` routines from SPECFEM3D_GLOBE commit
`9c312cb2c991b47484a7f302775f4f01ed9470f8`, using the final, deformed anchors
of this database. They exercise the same conversion as `prepare_timerun`'s
`rstore`, followed by the angular normalization used by elastic setup.
They are conversion reference values, not a full solver-run `rstore` dump:
this thin-database fixture does not emit the native solver databases.

To regenerate with gfortran available, run from this directory:

```bash
python3 provenance/dump_spherical_coordinates.py /path/to/specfem3d_globe
```

The test converts the cache's SI radius back using the database's planet radius
and compares all three components to `1e-6`. The fixture replaces the older
`single_chunk_1D` fixture named in issue #2041. Separate tests check every GLL
point's Cartesian round trip at `1e-9` relative to radius, including when the
solver stores Cartesian coordinates in single precision.
