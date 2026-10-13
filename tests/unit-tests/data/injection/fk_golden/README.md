# FK Golden Files (FKG1 format)

These files are the numerical oracle for the FK injection solver.
They are produced by the `DISABLED_GenerateAll` test in
`tests/unit-tests/injection/fk_golden_generator.cpp`.

## FKG1 binary layout

All integers are `int32_t`, all real values are `double` (8 bytes).
Byte order: little-endian (x86 / arm64 assumed).

### Fixed header

| Field            | Type     | Description |
|------------------|----------|-------------|
| magic            | char[4]  | "FKG1" |
| version          | int32    | 1 |
| num_points       | int32    | Number of evaluation points |
| num_components   | int32    | 7 |
| num_steps        | int32    | TimeWindow.nstep |
| np_resamp        | int32    | FkResult.resampling_rate() |
| nf_stored        | int32    | FkSizes.number_of_stored_frequencies |
| nfft             | int32    | FkSizes.number_of_fft_samples |
| dt_sem           | double   | TimeWindow.dt (s) |
| dt_min           | double   | FkResult.resampled_dt() (s) |
| origin_time      | double   | IncidentWave.origin_time (s) |
| reference_time   | double   | FkResult.reference_time() (s) |

### Per-point record (repeated num_points times)

| Field  | Type   | Description |
|--------|--------|-------------|
| medium | int32  | 0=elastic, 1=acoustic |
| tau    | double | FkResult.time_delays()(point) (s) |
| z      | double | depth in m |
| x      | double | x-coordinate in m |
| y      | double | y-coordinate in m |

### Value block

```
double[num_points][num_components][num_steps]  (C order)
```

Component order: 0=ux, 1=uy, 2=uz, 3=tx, 4=ty, 5=tz, 6=pressure.

**Reconstruction formula:**
```
abscissa = istep / np_resamp
value(p, c, istep) = cubic_bspline::evaluate(&coef(p, c, 0), ncoef, abscissa)
```

## Scenarios

| ID | Model | Wave | theta | dt (s) | nstep | #pts | traction | derivative |
|----|-------|------|-------|--------|-------|------|----------|------------|
| E1 | elastic (3-layer) | P  | 20° | 0.1  | 300 |  9 | yes | displacement |
| E2 | elastic (3-layer) | SV | 20° | 0.05 | 600 |  5 | yes | displacement |
| E3 | elastic (3-layer) | P  | 20° | 0.05 | 600 |  5 | yes | displacement |
| E4 | elastic (3-layer) | P  | 20° | 0.1  | 300 |  3 | yes | velocity |
| A1 | acoustic+elastic  | P  | 15° | 0.1  | 300 | 10 | yes | displacement |

## Regeneration

```sh
FK_GOLDEN_OUT=<abs_path_to_output_dir> \
  build/release/tests/unit-tests/injection_golden_generator \
  --gtest_also_run_disabled_tests \
  --gtest_filter='*DISABLED_GenerateAll*'
```

GENERATOR_SHA: implementation e7d43bfe5 (drift-merged injection-1.1-pr7-docs); generator source on branch scratch/fk-golden-capture (old pre-rework API, not rebuildable against the reworked API)
BUILD: cmake preset `release`; Kokkos Serial backend; AppleClang 17.0.0; type_real = float (default precision)
