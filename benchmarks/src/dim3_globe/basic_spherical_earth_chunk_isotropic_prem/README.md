# Bolivia 1994 deep earthquake on a single-chunk isotropic PREM Earth

## Purpose

Code-to-code benchmark of the SPECFEM++ globe solver against SPECFEM3D_GLOBE on the
smallest spherical setup that still has real Earth structure: one 90 degree chunk of
1D isotropic PREM, cut off at 770 km depth so that only the upper mantle and the top of the
lower mantle are meshed. The source is the 1994-06-09 Northern Bolivia event (CMT `060994A`,
moment tensor and location of the catalogue, half duration 40 s), moved to 100 km depth,
recorded at 22 stations inside the chunk. The setup is sized so that the whole pipeline
runs in minutes.

## Settings

| Parameter | Value | Reason |
| --- | --- | --- |
| `NCHUNKS`, `NPROC_XI`, `NPROC_ETA` | 1, 1, 1 | smallest setup, one rank |
| `ANGULAR_WIDTH_XI/ETA_IN_DEGREES` | 90 | covers the event and the stations |
| `CENTER_LATITUDE/LONGITUDE_IN_DEGREES` | -12.41375, -65.8018 | chunk centre shifted by half an element (90/32/2 = 1.40625 degrees) north and east of the epicentre, so the source does not sit on an element corner (it is at xi = eta = 0.5 of its element) |
| `GAMMA_ROTATION_AZIMUTH` | 0 | chunk axes along north and east |
| `NEX_XI`, `NEX_ETA` | 32 | coarse; 5888 elements; the mesher reports a 139.0 s empirical minimum period |
| `REGIONAL_MESH_CUTOFF`, `_DEPTH`, `_ADD_2ND_DOUBLING` | `.true.`, 770 km, `.false.` | mesh only the crust and mantle above 770 km: no outer or inner core, no CMB/ICB fluid-solid interfaces, small element count |
| Source depth | 100 km (CMTSOLUTION) | shallow source inside the cut-off mesh; hdur 40 s, time shift 0, lat -13.82, lon -67.25 and the Bolivia moment tensor are unchanged |
| `MODEL` | `1D_isotropic_prem` | |
| `RECORD_LENGTH_IN_MINUTES`, `DT`, `NSTEP` | 30, 0.10 s, 18000 | pinned so both codes step identically; the mesher's own choice (0.19 s) violates the SPECFEM++ CFL guard (CFL 0.80 > Cmax = 0.5, suggested dt 0.119 s), so both use 0.10 s (0.9 x suggested, rounded down to 2 decimals) |
| `NTSTEP_BETWEEN_OUTPUT_SAMPLE` | 10 | 1 s sampling, 1800 samples per trace |
| `ELLIPTICITY` | `.false.` | SPECFEM++ coordinate resolution on an elliptical globe is not yet implemented (issue #2058) |
| `TOPOGRAPHY` | `.false.` | external ETOPO data is not shipped |
| `ABSORBING_CONDITIONS` | `.false.` | the thin SPECFEM++ database carries no absorbing faces, so reflecting chunk sides are the only setup both codes share |
| `GRAVITY`, `FULL_GRAVITY`, `ROTATION`, `ATTENUATION`, `OCEANS` | `.false.` | not yet implemented in the SPECFEM++ globe path |
| `SPECFEMPP_DATABASE` | `.true.` | mesher writes only the thin SPECFEM++ database |
| Seismograms | SAC binary | read by obspy on both sides |

## Stations

22 stations from the SPECFEM3D_GLOBE examples `benchmarks/prem_ani_noocean_Q_Cowling_Bolivia_9s`
and `small_benchmark_run_to_test_very_simple_Earth`: every station of those two lists that
lies inside the chunk with |xi|, |eta| < 40 degrees. HOPE and DWPF were excluded as edge
stations and SAML (in both lists) was de-duplicated. Burial is 0.

## Status

The full pipeline runs: `xmeshfem3D_globe` (serial, about 2 s), `check_database.py`,
`specfem globe` on the thin database, and the comparison. On an Apple Silicon laptop
(serial build) the SPECFEM++ solver takes about 8 min for the 18000 steps; SPECFEM3D_GLOBE
(1 rank) takes about 6.7 min for mesher and solver together (time loop 404 s).

`check_database.py` passes on the cut-off database: 5888 elements, 1024 free-surface
faces and no CMB or ICB faces. The thin-database writer leaves the mesh bottom of a
regional cut-off mesh unrecorded, like the chunk sides.

Comparison (2026-10-21, SPECFEM++ serial `release` preset vs SPECFEM3D_GLOBE `688967d8b`),
normalised L2 misfit `||u - r|| / ||r||` and zero-lag correlation over the common window of
each trace:

| | min | mean | max |
| --- | --- | --- | --- |
| misfit (66 traces) | 0.00012 | 0.00033 | 0.00099 |
| correlation | 1.0000 (at 5 digits) | | 1.0000 |

The largest misfits are on the E component of the closest stations (LVC, NNA, LCO).

The source sits at xi = 0.4999926, eta = 0.4986403, gamma = 0.4286212 of its element in
SPECFEM3D_GLOBE; all 22 stations are located with errors below 1e-11 km.

SPECFEM++ names its channels `BX?` (from the 0.1 s time step), SPECFEM3D_GLOBE `LX?`
(from the 1 s sampling); `plot_seismograms.py` matches either.

MPI runs are not covered: they need #2051 and a reconciliation of the per-rank database
file naming (`proc%06d_specfempp_database.bin` from the Fortran writer vs
`<stem>/proc_N.bin` expected by SPECFEM++).

Note on the CFL guard: SPECFEM++ compares `dt` against the element-wise suggested time
step; the mesher's own 0.19 s is rejected on this mesh (suggested 0.119 s), hence the
pinned 0.10 s.

## Running

```bash
cmake --preset release
cmake --build build/release --target specfem xmeshfem3D_globe
cd benchmarks/build/release/dim3_globe/basic_spherical_earth_chunk_isotropic_prem
uv run snakemake -j 1                # mesh, database check, SPECFEM++ solver, comparison
```

Alternatively, run the CMake target `benchmark_dim3_globe_basic_spherical_earth_chunk_isotropic_prem`
(serial mesher followed by `check_database.py`).

## Reference seismograms

Generated with a built SPECFEM3D_GLOBE checkout on 1 rank:

```bash
export SPECFEM3D_GLOBE_ROOT=/path/to/specfem3d_globe
uv run snakemake -j 1 generate_reference
```

The rule fails if the variable is unset. It writes the SAC binary files
`NET.STA.LXE.sem.sac` (and `LXN`, `LXZ`) into `reference_seismograms/`, which is a symlink
into the source tree, plus `reference_seismograms/provenance/` with the Par_file,
CMTSOLUTION, STATIONS, mesher and solver output, and `VERSION.txt` (commit, describe,
configure line, date, ranks).

- Fortran commit: `688967d8b` (v8.1.0-438, built with `./configure FC=h5fc CC=h5cc MPIFC=h5pfc --with-mpi --with-hdf5`)
- wall-clock: about 6.7 min for mesher and solver on one rank (Apple Silicon laptop); solver time loop 404 s
- elements: 5888 (crust/mantle above 770 km only); DT 0.10 s, 18000 steps, 1800 samples per trace at 1 s, channel code `LX?`

## Time conventions

t = 0 at the centroid in both codes. SPECFEM3D_GLOBE starts at -1.5 hdur = -60 s,
SPECFEM++ at -2 hdur = -80 s, so with 1800 samples at 1 s the traces cover -60 s to 1739 s and
-80 s to 1719 s respectively. The Fortran onset truncation is
0.5 (1 + erf(-2.44)) ~ 3e-4 of the moment, which is negligible. Both use the erf/Gaussian
Heaviside source time function with hdur / 1.628.

## Comparison

`plot_seismograms.py` reads both sides with `obspy.read`, aligns them on the SAC `b`
header, and writes `OUTPUT_FILES/comparison.txt` with the normalised L2 misfit and
correlation per trace, plus record sections per component
(`OUTPUT_FILES/{E,N,Z}_record_section.png`) and `OUTPUT_FILES/geometry.png`. It works in
reference-only mode when no SPECFEM++ traces exist.

## Cleaning up

```bash
uv run snakemake -j 1 clean
```
