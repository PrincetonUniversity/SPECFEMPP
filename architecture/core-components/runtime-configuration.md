# Runtime Configuration

**Files:** `core/specfem/runtime_configuration/`

The `setup` class parses a YAML parameter file and holds all configuration as typed C++ objects. It acts as a **factory** for simulation components:

| Method | Returns |
|---|---|
| `instantiate_quadrature()` | `specfem::quadrature::quadratures` |
| `instantiate_timescheme(fields)` | `shared_ptr<time_scheme>` |
| `instantiate_solver<NGLL, Dim>(...)` | `shared_ptr<solver::solver>` |
| `instantiate_seismogram_writer()` | `shared_ptr<io::writer>` |
| `instantiate_wavefield_writer<Dim>()` | `shared_ptr<periodic_task>` |
| `instantiate_kernel_writer()` | `shared_ptr<io::writer>` |
| `instantiate_property_reader/writer()` | `shared_ptr<io::reader/writer>` |

## YAML to C++ Mapping

| YAML section | C++ class |
|---|---|
| `header` | `runtime_configuration::header` |
| `simulation-setup.quadrature` | `runtime_configuration::quadrature` |
| `simulation-setup.solver` | `runtime_configuration::solver` |
| `simulation-setup.solver.time-scheme` | `runtime_configuration::time_scheme` |
| `simulation-mode` | Determines `specfem::simulation::type` |
| `receivers` | `runtime_configuration::receivers` |
| `sources` | `runtime_configuration::sources` |
| `databases` | `runtime_configuration::database_configuration` |

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
