# Bolivia 1994 deep earthquake on a single-chunk isotropic PREM Earth

## Purpose

Code-to-code benchmark of the SPECFEM++ globe solver against SPECFEM3D_GLOBE on the
smallest spherical setup that still has real Earth structure: one 90 degree chunk of
1D isotropic PREM. The source is the 1994-06-09 Northern Bolivia event (CMT `060994A`,
647 km depth, half duration 40 s), recorded at 22 stations inside the chunk.

## Settings

| Parameter | Value | Reason |
| --- | --- | --- |
| `NCHUNKS`, `NPROC_XI`, `NPROC_ETA` | 1, 1, 1 | smallest setup, one rank |
| `ANGULAR_WIDTH_XI/ETA_IN_DEGREES` | 90 | covers the event and the stations |
| `CENTER_LATITUDE/LONGITUDE_IN_DEGREES` | -13.82, -67.25 | chunk centred on the epicentre |
| `GAMMA_ROTATION_AZIMUTH` | 0 | chunk axes along north and east |
| `NEX_XI`, `NEX_ETA` | 64 | coarse but with real crustal layering; the mesher reports a 69.5 s empirical minimum period |
| `MODEL` | `1D_isotropic_prem` | |
| `RECORD_LENGTH_IN_MINUTES`, `DT`, `NSTEP` | 30, 0.11 s, 17500 | pinned so both codes step identically; the mesher's own choice (0.19 s) violates the SPECFEM++ CFL guard (Cmax = 0.5, suggested dt 0.119 s), so both use 0.11 s |
| `NTSTEP_BETWEEN_OUTPUT_SAMPLE` | 9 | 0.99 s sampling |
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

The full pipeline runs: `xmeshfem3D_globe` (serial), `check_database.py`, `specfem globe`
on the thin database, and the comparison. On an Apple Silicon laptop the SPECFEM++ solver
takes about 51 min for the 17500 steps (serial build), SPECFEM3D_GLOBE about 23 min.

First comparison (2026-10-20, SPECFEM++ serial `release` preset vs SPECFEM3D_GLOBE
`688967d8b`), normalised L2 misfit `||u - r|| / ||r||` and zero-lag correlation over the
common window of each trace:

| | min | mean | max |
| --- | --- | --- | --- |
| misfit (66 traces) | 0.009 | 0.053 | 0.104 |
| correlation | 0.994 | | 0.9997 |

The residual is not yet attributed. Candidates to rule out, in order: the start-time
truncation of the source (see "Time conventions"), the sampling of the 1D model at GLL
points versus the mesher's element-wise model, the mass-matrix treatment at the CMB/ICB
fluid-solid interfaces, and the different surface stations' depth resolution. The
SPECFEM++ globe assembly is still being completed (issues #2047, #2048, #2052, #2053), so
expect these numbers to move; the benchmark exists to track them.

MPI runs are not covered: they need #2051 and a reconciliation of the per-rank database
file naming (`proc%06d_specfempp_database.bin` from the Fortran writer vs
`<stem>/proc_N.bin` expected by SPECFEM++).

Note on the CFL guard: SPECFEM++ compares `dt` against the element-wise suggested time
step; the mesher's own 0.19 s is rejected on this mesh (suggested 0.119 s), hence the
pinned 0.11 s.

## Running

```bash
cmake --preset release
cmake --build build/release --target specfem xmeshfem3D_globe
cd benchmarks/build/release/dim3_globe/basic_spherical_earth_isotropic_prem
uv run snakemake -j 1                # mesh, database check, SPECFEM++ solver, comparison
```

Alternatively, run the CMake target `benchmark_dim3_globe_basic_spherical_earth_isotropic_prem`
(serial mesher followed by `check_database.py`).

## Reference seismograms

Generated with a built SPECFEM3D_GLOBE checkout on 1 rank:

```bash
export SPECFEM3D_GLOBE_ROOT=/path/to/specfem3d_globe
uv run snakemake -j 1 generate_reference
```

The rule fails if the variable is unset. It writes the SAC binary files
`NET.STA.MXE.sem.sac` (and `MXN`, `MXZ`) into `reference_seismograms/`, which is a symlink
into the source tree, plus `reference_seismograms/provenance/` with the Par_file,
CMTSOLUTION, STATIONS, mesher and solver output, and `VERSION.txt` (commit, describe,
configure line, date, ranks).

- Fortran commit: `688967d8b` (v8.1.0-438, built with `./configure FC=h5fc CC=h5cc MPIFC=h5pfc --with-mpi --with-hdf5`)
- wall-clock: 23 min 18 s for the solver on one rank (Apple Silicon laptop), mesher a few seconds
- elements: 37440 in the thin database (34560 crust/mantle + outer core and inner core); DT 0.11 s, 17500 steps, 1945 samples per trace at 0.99 s

## Time conventions

t = 0 at the centroid in both codes. SPECFEM3D_GLOBE starts at -1.5 hdur = -60 s,
SPECFEM++ at -2 hdur = -80 s, so with 17500 steps the traces end at about 1865 s and
1845 s respectively. The Fortran onset truncation is
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
