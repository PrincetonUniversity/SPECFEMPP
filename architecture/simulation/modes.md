# Simulation Modes

SPECFEM++ supports two simulation types, controlled by the `simulation-mode` YAML section.

## Forward (`simulation::type::forward`)

Standard forward wave propagation. Sources inject energy, receivers record synthetic seismograms. Optionally saves:

- Seismograms (displacement, velocity, acceleration, pressure)
- Wavefield snapshots at configurable intervals
- Material property files

---

## Combined (`simulation::type::combined`)

Adjoint + backward simulation for computing **Fréchet sensitivity kernels** used in seismic tomography.

### How It Works

1. The **adjoint wavefield** is propagated forward with time-reversed seismogram residuals injected at receiver locations.
2. The **backward wavefield** reconstructs the original forward wavefield from stored boundary values.
3. The two fields are **cross-correlated** at each timestep to accumulate Fréchet kernels (∂χ/∂m for each material parameter m).

### Prerequisites

This mode requires a prior forward run with `wavefield-writer` configured to save boundary checkpoints. The checkpoint files are then read back by `wavefield-reader` during the combined run.

### Output

Sensitivity kernels are written to disk after the time loop (step 14 of the [workflow](workflow.md)).

---

← [Back to Workflow](workflow.md) | [Back to Index](../index.md)
