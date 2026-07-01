# Simulation Workflow

The following is the canonical execution sequence for a forward simulation (both 2D and 3D follow the same pattern):

```
1. Parse YAML config
        │
        ▼
2. Instantiate GLL quadrature
        │
        ▼
3. Read mesh database (Fortran binary)
        │
        ▼
4. Read sources (YAML) → compute t₀ from source STF
        │
        ▼
5. Read receivers (stations file or YAML)
        │
        ▼
6. Build assembly<DimensionTag>
   ├── Locate sources and receivers in mesh elements
   ├── Compute Jacobians at all GLL points
   ├── Compute and assemble mass matrix
   ├── Initialize material properties at GLL points
   ├── Set up boundary condition data
   └── Allocate forward (+ adjoint/backward) field arrays
        │
        ▼
7. [Optional] Load saved material properties from disk
        │
        ▼
8. [Optional] Early exit: write material properties to disk
        │
        ▼
9. Instantiate time scheme (Newmark)
        │
        ▼
10. Register periodic tasks
    ├── wavefield_reader (if adjoint)
    ├── wavefield_writer (if saving checkpoints)
    ├── wavefield_plotter (if visualization enabled)
    └── check_signal
        │
        ▼
11. Instantiate solver (time_marching<Forward|Combined, Dim, NGLL>)
        │
        ▼
12. solver.run()  ← main time loop
    for each timestep:
        ├── time_scheme.apply_predictor(all media)
        ├── compute_derivatives + update_medium (acoustic)
        ├── time_scheme.apply_corrector(acoustic)
        ├── compute_derivatives + update_medium (elastic)
        ├── time_scheme.apply_corrector(elastic)
        ├── compute_derivatives + update_medium (poroelastic)
        ├── time_scheme.apply_corrector(poroelastic)
        ├── apply boundary conditions
        ├── accumulate seismograms (every nstep_between_samples)
        └── run periodic tasks
        │
        ▼
13. Write seismograms to disk
        │
        ▼
14. [Optional] Write sensitivity kernels to disk
```

## Key Invariants

- The media processing order **acoustic → elastic → poroelastic** is fixed and must not be changed — it ensures correct flux exchange at multi-physics interfaces.
- Periodic tasks (step 10) are registered before the solver is instantiated and are executed inside the time loop at the configured step intervals.
- Steps 7–8 allow the simulation to function as a property inspector/converter without running the time loop.

---

← [Back to Simulation](modes.md) | [Back to Index](../index.md)
