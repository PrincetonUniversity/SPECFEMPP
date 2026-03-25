# Solver and Time Scheme

## Solver (`specfem::solver`)

**Files:** `core/specfem/solver/`

The abstract base class `solver::solver` has a single virtual method `run()`. The concrete implementation is `solver::time_marching<Simulation, DimensionTag, NGLL>`, templated on:

- `Simulation` — `specfem::simulation::type::forward` or `::combined`
- `DimensionTag` — `dim2` or `dim3`
- `NGLL` — number of GLL points per element per dimension (currently fixed at 5)

### Forward Solver (`simulation::type::forward`)

Each timestep follows a **predictor-corrector** pattern. Media are processed in a specific order to ensure correct multi-physics coupling at interfaces:

1. **Predictor phase** (all media) — extrapolate displacement/velocity to half-step.
2. **Acoustic wavefield computation** → corrector phase.
3. **Elastic wavefield computation** (elastic_psv, elastic_sh, elastic_psv_t) → corrector phase.
4. **Poroelastic wavefield computation** → corrector phase.

> **Important:** The media processing order (acoustic → elastic → poroelastic) is fixed and critical for correct coupling at multi-physics interfaces.

### Combined Solver (`simulation::type::combined`)

Used for **adjoint simulations** to compute Fréchet kernels:

- Runs the **adjoint wavefield** forward in time (driven by adjoint sources at receiver locations).
- Simultaneously runs the **backward wavefield** in reverse-time (reconstructed from saved checkpoints).
- At each step, correlates adjoint and backward fields to accumulate **Fréchet derivative kernels**.

---

## Time Scheme (`specfem::time_scheme`)

**Files:** `core/specfem/timescheme/`

`time_scheme` is the abstract base class for time integration. The current implementation is `newmark` — the classic **Newmark-beta predictor-corrector** scheme.

Key interface:

```cpp
for (const auto [istep, dt] : ts.iterate_forward()) {
    ts.apply_predictor_phase_forward(medium_tag);
    // ... compute accelerations ...
    ts.apply_corrector_phase_forward(medium_tag);
}
```

The `iterate_forward()` / `iterate_backward()` helper ranges make the time loop direction-agnostic and clean.

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
