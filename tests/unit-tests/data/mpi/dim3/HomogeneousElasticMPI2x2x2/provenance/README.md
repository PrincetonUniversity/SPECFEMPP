# HomogeneousElasticMPI2x2x2 fixture

8-partition MPI test fixture for 3D spectral-element assembly. A 4×4×4-element
homogeneous elastic cube (64 elements, 125 nodes) generated with gmsh and
decomposed with **METIS into 8 partitions** (`NPROC = 8`) via `xdecompose_mesh`.

## Why this fixture exists

The internal mesher (`xmeshfem3D`) and the other committed MPI fixtures use a
structured processor grid and only ever produce a restricted set of MPI
interfaces: lateral faces and vertical edges, all of a single orientation per
neighbor. A general (METIS / `xdecompose_mesh`) decomposition instead produces
**top/bottom faces, horizontal edges, and single-node corner connections**, and
mixes orientations on a single neighbor pair.

That distinction surfaced a real bug: the packer/unpacker paired interface
connections by list index without a rank-consistent ordering, so the receiver's
per-index orientation only matched the sender for uniform-orientation
(structured) interfaces. It manifested as a slow MPI instability on
gmsh/xdecompose meshes while serial and internal-mesher MPI runs were fine. The
fix adds a rank-symmetric canonical sort of the MPI connections in the
`specfem::assembly::mpi<dim3>` constructor.

This fixture exercises the general interface set so the `CommunicationPattern`
round-trip test (run on 8 ranks by `assembly_mpi_dim3_8proc_tests`) guards
against a regression. Without the fix the round-trip coordinate check fails on
the mixed-orientation interfaces; with the fix it passes.

## Regenerate

From this directory, with `xdecompose_mesh` built and the HDF5/OpenMPI runtime
on `LD_LIBRARY_PATH`:

```bash
uv run --group scripts snakemake -c1
# or manually:
uv run --group scripts python create_mesh.py     # -> cube.msh
uv run --group scripts python export_mesh.py      # -> MESH/
xdecompose_mesh -p Par_file                        # -> ../Database/proc_*.bin
```

`cube.msh` and `MESH/` are regenerable intermediates and are not committed; the
committed fixture is `../Database/proc_0.bin … proc_7.bin`.
