# Mesh

**Files:** `core/specfem/mesh/`

The `mesh<DimensionTag>` struct stores everything read from the **external mesher database** (a Fortran binary file produced by MESHFEM2D/MESHFEM3D). It is a passive data container — no computation happens here. The goal is to have an interface that is as close as possible to the original Fortran data structures, to have a clear representation of the mesh as it is written by the mesher. This helps us to isolate the Fortran binary reading logic in `io::read_2d_mesh` / `io::read_3d_mesh` and keep the rest of the codebase independent of the Fortran data layout.

## Sub-structs

| Sub-struct | Contents |
|---|---|
| `mesh::parameters` | Global mesh parameters (nspec, nglob, …) |
| `mesh::coordinates` | GLL-point coordinates |
| `mesh::control_nodes` | Corner node coordinates |
| `mesh::mapping` | Local ↔ global DOF mapping |
| `mesh::materials` | Per-element material assignments |
| `mesh::tags` | Per-element medium/property/boundary tags |
| `mesh::boundaries` | Absorbing, free-surface, forcing boundary lists |
| `mesh::coupled_interfaces` | Fluid-solid / solid-solid coupling lists |
| `mesh::coloring` | Graph-coloring for race-free parallel updates |
| `mesh::inner_outer` | MPI inner/outer element classification |
| `mesh::adjacency` | Element adjacency graph |
| `mesh::mpi` | Shared DOFs between MPI ranks |

The mesh is **read once** from disk by `io::read_2d_mesh` / `io::read_3d_mesh` (Fortran binary format) and then consumed by the [`assembly`](assembly.md) constructor.

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
