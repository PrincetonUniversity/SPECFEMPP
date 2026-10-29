# Plane-wave injection plugin

An injection plugin that serves an analytic P or SV plane wave in a
homogeneous, isotropic elastic whole space. It is the worked example for
plugin authors and the source of exact reference solutions for validation.

It is a standalone CMake project that uses only the installed plugin SDK; it
does not link any SPECFEM++ library.

## Build

```bash
# 1. Install the SDK from a configured SPECFEM++ build tree
cmake --preset release
cmake --install build/release --component plugin_sdk --prefix <sdk>

# 2. Build the plugin and run its tests
cmake -S examples/plugins/plane_wave -B build/plane_wave -DCMAKE_PREFIX_PATH=<sdk>
cmake --build build/plane_wave
ctest --test-dir build/plane_wave --output-on-failure
```

yaml-cpp is downloaded with `FetchContent` and linked statically; only the
entry point `specfem_injection_plugin` is exported, which the build checks.
`fill` runs in parallel with OpenMP when the compiler provides it.

## Run

```bash
specfem 3d -p parameters.yaml \
  --injection-plugin build/plane_wave/libplane_wave.so \
  --injection-config examples/plugins/plane_wave/plane_wave.yaml
```

with `injection: {method: stacey}` in `parameters.yaml`. The mesh must declare
its Stacey faces as injection faces (`STACEY_AS_INJECTION_BOUNDARY`), and its
material must match the configuration: setup fails, naming the first
mismatched point, otherwise.

## Configuration

See `plane_wave.yaml`:

| Key | Meaning |
|---|---|
| `material.rho`, `.vp`, `.vs` | Material of the whole space; checked against every point |
| `wave.type` | `P` or `SV` |
| `wave.take-off` | Angle θ of the propagation direction from +z, in degrees |
| `wave.azimuth` | Angle φ from +x toward +y, in degrees |
| `wave.origin`, `wave.origin-time` | The wavefront crosses `origin` at `origin-time` |
| `wave.amplitude` | Displacement amplitude A |
| `source-time-function` | `{type: ricker, f0}` or `{type: gaussian, half-duration}` |
| `material-tolerance` | Relative tolerance of the material check |

## Conventions

- z points up. Plugin step `s` is simulation time `t0 + s·dt`.
- Direction k̂ = (sin θ cos φ, sin θ sin φ, cos θ).
- Polarization d̂: k̂ for P; ∂k̂/∂θ = (cos θ cos φ, cos θ sin φ, −sin θ) for SV.
- τ = t − t_origin − k̂·(x − x₀)/c, with c = vp for P and vs for SV.
- f is the displacement waveform:
  Gaussian f = exp(−(τ/h)²); Ricker f = (1 − 2π²f₀²τ²) exp(−π²f₀²τ²).
- u = A d̂ f(τ), v = A d̂ f′(τ), a = A d̂ f″(τ).
- ε = −(A/c) f′(τ) sym(d̂ ⊗ k̂), σ = λ tr(ε) I + 2μ ε, traction = σ n̂ on the
  point's normal (out of the total-field region).

## Layout

| File | Contents |
|---|---|
| `include/plane_wave/plane_wave.hpp` | The physics, depending only on the standard library |
| `src/config.*` | Reading `plane_wave.yaml` |
| `src/plane_wave_plugin.*` | The plugin class: `supports`, `setup`, `fill` |
| `src/plugin_entry.cpp` | `SPECFEM_DEFINE_INJECTION_PLUGIN` |
| `tests/plane_wave_tests.cpp` | Finite-difference checks and plugin tests |
