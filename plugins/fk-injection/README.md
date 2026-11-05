# FK injection plugin

An injection plugin that serves the frequency-wavenumber (FK) field of a plane P
or SV wave incident on a horizontally layered model: a solid half-space below
any number of solid layers, optionally with a fluid block on top. Use it to
inject a teleseismic wavefield through the sides and bottom of a regional mesh.

It is built with SPECFEM++ against the in-tree plugin SDK. It does not link any
SPECFEM++ library; it uses SPECFEM++'s Kokkos for `Kokkos::mdspan` only.

## Build

The plugin is built with SPECFEM++ on Linux and macOS, into
`build/<preset>/lib/libfk_injection.so`. Its tests run with the unit tests:

```bash
cmake --preset release -DSPECFEM_BUILD_TESTS=ON
cmake --build build/release
ctest --test-dir build/release/tests -L fk_injection --output-on-failure
```

pocketfft is downloaded with `FetchContent` at configure time, and pocketfft and
SPECFEM++'s yaml-cpp are compiled in, so the plugin has no runtime dependencies. Only the entry point
`specfem_injection_plugin` is exported, which the build checks.

| CMake option | Default | Effect |
|---|---|---|
| `FK_INJECTION_DOUBLE` | `OFF` | Compute in double precision; independent of how SPECFEM++ was built |
| `FK_INJECTION_ENABLE_OPENMP` | `OFF` | Sample the field in parallel; needs a compiler with OpenMP |

## Run

```bash
specfem 3d -p parameters.yaml \
  --injection-plugin build/release/lib/libfk_injection.so \
  --injection-config plugins/fk-injection/fk.yaml
```

with `injection: {method: stacey}` in `parameters.yaml`. The mesh must declare
its Stacey faces as injection faces (`STACEY_AS_INJECTION_BOUNDARY`).

The field is defined by the layered model, so the mesh must match it. Setup
compares each injection point's rho, vp and vs with the layer at its depth and
stops the run, naming the first point that differs. A point on an interface may
carry the material of either layer.

## Configuration

`fk.yaml` is a working example. Unknown keys are rejected. Give exactly one of
`layers` and `model-file`.

### Model

| Key | Meaning |
|---|---|
| `layers` | List from the top down, each `acoustic: {rho, vp, thickness}` or `elastic: {rho, vp, vs, thickness}`, in kg/m³, m/s and m |
| `model-file` | SPECFEM3D `FKmodel` file; provides the model and, as defaults, the wave and time window |

The last layer is the half-space, and its `thickness` is ignored. Every other
layer needs a positive thickness. Acoustic layers must form one block above the
elastic layers.

### `incidence`

Required with `layers`. With `model-file` every key is optional and overrides the
file's value.

| Key | Meaning |
|---|---|
| `type` | `P` or `SV`; required |
| `back-azimuth` or `azimuth` | Degrees clockwise from +y (north); give one, required. φ = −`back-azimuth` − 90° or φ = 90° − `azimuth` |
| `take-off` | Degrees from vertical; required |
| `origin` | `[x, y, z]` in the mesh frame, in m; required |
| `origin-time` | Simulation time in s at which the wavefront crosses `origin`; default: the first step |
| `amplitude` | Scale of the field; default 1 |
| `half-duration` | Gaussian half-duration Tg in s; default 1 / `frequency-max` |

### `time-window`

Required with `layers`. With `model-file` every key is optional and overrides the
file's value.

| Key | Meaning |
|---|---|
| `frequency-max` | Maximum frequency of interest in Hz; default 1 / `half-duration` |
| `frequency-sampling` | Sampling frequency of the stored series in Hz; default 10 / `half-duration` |
| `length` | FK time window in s; required with `layers`. Setup fails if it can't cover the run and every delay, or if `origin-time` is too far from the run for one FFT period |

Give at least one of `frequency-max` and `half-duration`. A `half-duration` that
leaves more than 40% of the Gaussian's energy above `frequency-max`,
erfc(√2 π `frequency-max` Tg) > 0.4, is rejected, including 0. So a delta source
isn't available.

### Other keys

| Key | Meaning |
|---|---|
| `material-tolerance` | Relative tolerance of the material check; default 1e-3 |
| `reproduce-specfem3d-reference` | `true` reproduces SPECFEM3D's FK, see below; default `false` |

## Conventions

- z points up. Depth is `zmax − z`, with `zmax` the top of the global mesh, so the
  top of the model is at the top of the mesh wherever that is.
- The wave travels toward (sin θ cos φ, sin θ sin φ, cos θ), with θ the take-off
  angle. A back-azimuth of 0 therefore travels toward −y.
- Plugin step `s` is simulation time `t0 + s·dt`.
- The source spectrum is exp(−(ω Tg / 2)²), a Gaussian in time of unit area.
- Traction is σ n̂ on the outward normal of the injection face.
- Elastic points serve velocity and traction. Acoustic points serve displacement
  and potential rate.

## Comparing with SPECFEM3D

SPECFEM3D counts FK time from the first step and ignores the simulation's t0, so
leave `origin-time` out, which does the same. A non-zero `ORIGIN_TIME` in a model
file is rejected, because there it means the FK time at the first step, not the
time the wavefront crosses `origin`; set `incidence.origin-time` instead.

The plugin differs from SPECFEM3D's FK in two ways that
`reproduce-specfem3d-reference: true` turns off:

- SPECFEM3D normalizes the inverse FFT by the simulation step rather than the
  resampled step, which makes amplitudes larger by the resampling rate.
- SPECFEM3D caps the FK window near twice the run length. The computed field
  repeats every FFT period, and a fluid layer's reverberation decays slowly (by
  the ocean-bottom reflection coefficient, about 0.7, per round trip), so on a
  short run the reverberation outlasts the period and wraps onto the start of
  the run. The plugin keeps the cap but lengthens the window until the
  reverberation has decayed to 1e-4 of its start. For 30 km of water at 20° that
  is about 950 s; for 4 km it is about 120 s. The estimate covers the fluid
  block over the top solid layer; reverberations in solid layers aren't
  estimated.
- SPECFEM3D's frequency grid stretches times by N/(N − 1), with N the FFT length.
  It also truncates `ORIGIN_TIME` to a whole number of resampled steps, where the
  plugin keeps the fraction.

The incident SV bottom vector differs in both modes: see
[the bug report](docs/specfem3d-sv-fk-bug-report.md).

## Limits

- Layering is an optional fluid block over solid layers. A fluid between solids or
  a fluid half-space isn't supported.
- Only isotropic elastic and acoustic points are served; `elastic_spin` and
  anisotropic points are rejected.
- The `stacey` method in SPECFEM++ doesn't request acoustic quantities yet, so a
  run with injection faces in a fluid stops before reaching the plugin. The plugin
  side is covered by its unit tests.
- Against SPECFEM3D on fine meshes there is a residual of about 3e-3, suspected to
  come from edge and corner points using a single normal.

## Layout

| File | Contents |
|---|---|
| `include/fk_injection/` | The FK solve, depending on the SDK's medium enum and pocketfft only |
| `src/plugin_config.*` | Reading `fk.yaml` |
| `src/fk_injection_plugin.*` | The plugin class: `supports`, `setup`, `fill` |
| `src/plugin_entry.cpp` | `SPECFEM_DEFINE_INJECTION_PLUGIN` |
| `src/impl/` | Layer propagators, depth series, spline sampling |
| `tests/` | Unit tests, and the module driven through the ABI with dlopen |
