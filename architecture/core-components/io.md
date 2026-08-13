# I/O Subsystem and Periodic Tasks

## I/O Subsystem (`specfem::io`)

**Files:** `core/specfem/io/`

The I/O layer is organized around abstract `reader` and `writer` base classes with multiple format backends.

### Mesh Readers

| Function | Format |
|---|---|
| `io::read_2d_mesh` | Fortran binary (MESHFEM2D output) |
| `io::read_3d_mesh` | Fortran binary (MESHFEM3D output) |

### Format Backends

| Backend | Description |
|---|---|
| `io::ASCII` | Plain text (default for seismograms) |
| `io::HDF5` | HDF5 binary (requires `SPECFEM_ENABLE_HDF5=ON`) |
| `io::ADIOS2` | ADIOS2 streaming/file I/O (requires `SPECFEM_ENABLE_ADIOS2=ON`) |
| `io::NPY` | NumPy `.npy` single-array binary |
| `io::NPZ` | NumPy `.npz` archive (requires `SPECFEM_ENABLE_NPZ=ON`) |

### Data-Type Writers/Readers

| Data type | Description |
|---|---|
| `io::seismogram::writer/reader` | Synthetic seismograms at receivers |
| `io::wavefield::writer/reader` | Full wavefield snapshots |
| `io::property::writer/reader` | Material property fields |
| `io::kernel::writer` | Fréchet sensitivity kernels |

---

## Periodic Tasks (`specfem::periodic_tasks`)

**Files:** `core/specfem/periodic_tasks/`

`periodic_task<DimensionTag>` is an abstract base class for work that must be executed at regular intervals **during the time-stepping loop** (e.g., every N steps). Concrete tasks:

| Task | Description |
|---|---|
| `wavefield_checkpoint` | Define fixed-stride and subdivided checkpoint replay windows |
| `wavefield_writer` | Write wavefield snapshots to disk at configured intervals |
| `wavefield_reader` | Read pre-computed wavefield snapshots (adjoint setup) |
| `plot_wavefield` | Real-time or file-based wavefield visualization (VTK/PNG/JPG) |
| `check_signal` | Catch `SIGINT`/`SIGTERM` for graceful shutdown |

Tasks are collected in a `std::vector<shared_ptr<periodic_task>>` and executed by the solver at each configured step interval.

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
