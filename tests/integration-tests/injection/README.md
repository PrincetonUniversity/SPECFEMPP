# End-to-end injection tests

`specfem 3d` with the plane-wave example plugin (`examples/plugins/plane_wave`)
and `injection: {method: stacey}` on a homogeneous box with Stacey on all six
faces, all of them injection faces. With nothing to reflect, the exact field
inside the box is the incident plane wave alone.

| Case | Ranks | Check |
|---|---|---|
| `zero_field` | 1 | A force source plus injection at amplitude 0 gives seismograms bit-identical to the run without injection |
| `p_wave`, `sv_wave` | 1 | Relative L2 error against the analytic traces ≤ tolerance, and \|lag\| < 0.25 dt |
| `p_wave_mpi2`, `p_wave_mpi4`, `sv_wave_mpi4` | 2, 4 | Relative L2 error against the serial traces ≤ 1e-4, and \|lag\| < 0.25 dt |

The cases are listed in `tests.yaml` and run by `injection_tests.cpp`. They
need a built plugin, so CMake registers them (label `injection_e2e`) only when
`SPECFEM_INJECTION_TEST_PLUGIN` names one:

```bash
cmake --install build --component plugin_sdk --prefix build/plugin_sdk
cmake -S examples/plugins/plane_wave -B build/plane_wave -DCMAKE_PREFIX_PATH=$PWD/build/plugin_sdk
cmake --build build/plane_wave
cmake build -DSPECFEM_INJECTION_TEST_PLUGIN=$PWD/build/plane_wave/libplane_wave.so
cmake --build build && ctest --test-dir build/tests -L injection_e2e
```

The Jenkins pipeline `.jenkins/plugin_checks.gvy` does this on the CPU (serial,
OpenMP), CUDA and MPI runners.

## Fixture

- Box: 10 km × 10 km × 10 km, 10 × 10 × 10 elements (GLL4), ρ = 2300,
  vp = 2800, vs = 1500. `TOP_FREE_SURFACE = .false.` makes the top face
  absorbing; `STACEY_AS_INJECTION_BOUNDARY = .true.` makes every Stacey face an
  injection face.
- Waves (`p.yaml`, `sv.yaml`): Ricker f0 = 0.4 Hz, P at take-off 30° and
  azimuth 40°, SV at 25° and 120°, both crossing the box centre inside the
  run. They start and end outside the box.
- dt = 0.01 s; the run starts at t0 = −3 s, set by the sources' Ricker
  (−1.2 / f0).
- Stations: 8 interior points, 3 km from the faces (`STATIONS`).

## Provenance

```bash
SPECFEMPP_BINDIR=<MPI build>/bin PLANE_WAVE_PLUGIN=<.../libplane_wave.so> snakemake --cores 4
```

- `mesh/{serial,mpi2,mpi4}/database*`: the SPECFEM++ mesher
  (`injection_mesher.smk`), partitioned for 1, 2 and 4 ranks.
- `traces/<case>/`: the analytic velocity at the stations, from
  `injection_cases.py`, which writes the plane wave out independently of the
  plugin's C++.
- `traces_serial/<case>/`: a serial `specfem 3d` run with the plugin, the
  reference of the MPI cases.

## Tolerances

Refinement study (P and SV, worst station and component):

| Elements per side | dt (s) | P L2 error | P lag | SV L2 error | SV lag |
|---|---|---|---|---|---|
| 6 | 0.0167 | 1.87e-2 | 1.6 ms | 3.40e-2 | 5.2 ms |
| 6 | 0.0083 | 1.84e-2 | 1.5 ms | 3.24e-2 | 4.5 ms |
| 8 | 0.0125 | 1.33e-2 | 1.1 ms | 1.40e-2 | 2.9 ms |
| 8 | 0.0063 | 1.30e-2 | 1.1 ms | 1.29e-2 | 2.5 ms |
| **10** | **0.01** | **1.03e-2** | **0.96 ms** | **7.20e-3** | **1.1 ms** |
| 10 | 0.005 | 1.02e-2 | 0.91 ms | 7.14e-3 | 0.9 ms |
| 14 | 0.0071 | 7.16e-3 | 0.66 ms | 4.98e-3 | 0.69 ms |
| 14 | 0.0036 | 7.09e-3 | 0.63 ms | 4.96e-3 | 0.57 ms |

- The error falls with the element size and barely changes with dt: it is
  the spatial discretization's.
- The lag does not scale with dt either: it is numerical dispersion, not a
  time offset. A half-step offset between the injected field and the Newmark
  update would be 0.5 dt = 5 ms at the fixture's dt; the bound 0.25 dt sits
  between the two. Shifting the plugin's time by dt / 2 gives a lag of
  0.6 dt and fails.
- The tolerances are 1.5 × the error of the fixture (bold row): 1.6e-2 for P,
  1.1e-2 for SV.
- Partitioning changes the order of single-precision sums: 2 and 4 ranks
  differ from the serial run by 4.1e-5 relative L2, the same with and without
  injection, hence the MPI tolerance of 1e-4.
