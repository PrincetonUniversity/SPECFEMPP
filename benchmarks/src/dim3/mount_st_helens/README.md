# Mount St. Helens — topographic external mesh (Gmsh)

This benchmark ports the classic SPECFEM3D *Mount St. Helens* real-world example
to SPECFEM++. It exercises the external-mesh workflow on real topography: a
structured box under the volcano is meshed with Gmsh and warped onto the
measured terrain, exported via `export_gmsh3d.py`, partitioned with
`xdecompose_mesh`, and simulated with `specfem3d`.

The original mesh was built with CUBIT (proprietary). Here `read_topo.py` +
`create_mesh.py` reproduce it with [Gmsh](https://gmsh.info), so the same recipe
works for **any** topography file `X Y Z` in UTM coordinates.

## Physics

- Domain: 15 km × 22 km box, 10 km deep, top warped onto Mount St. Helens
  topography (UTM zone 10), ~26k hexahedral elements at 500 m resolution.
- Homogeneous isotropic elastic medium: ρ = 2300 kg/m³, vp = 2800 m/s,
  vs = 1500 m/s, Qμ = 150.
- Isotropic explosion source (CMTSOLUTION) near the summit, 1.5 s half duration.
- Topographic free surface at the top; Stacey absorbing boundaries on the sides
  and bottom.
- 12.5 s simulation (dt = 0.005 s, 2500 steps).
- Eight surface receivers (lat/lon) around the edifice.

Sources and receivers are given in geographic coordinates and projected onto the
UTM mesh by the solver (`UTM_PROJECTION_ZONE = 10`,
`SUPPRESS_UTM_PROJECTION = .false.` in the `Par_file`).

## Running the benchmark

This benchmark is **MPI-only** (the mesh is too large for a serial run), so it is
configured only when SPECFEM++ is built with MPI enabled.

Install [uv](https://docs.astral.sh/uv/getting-started/installation), then from
the build directory for this benchmark:

```bash
uv sync --group examples
uv run snakemake -j 1
```

To run on a Slurm cluster:

```bash
uv run snakemake --executor slurm -j 1
```

## Cleaning up

```bash
uv run snakemake clean -j 1
```

⚠️ **Warning:** The reference traces are from the Fortran code, but the
                receivers in the fortran code are not exactly at the surface
                since the Fortran code is not computing the exact element
                topography. Therefore, the traces will not match exactly. The
                traces are only for reference.

## Workflow stages

1. **create_mesh** — runs `create_mesh.py` to build `mount_sthelens.msh`: a
   transfinite hex box whose top nodes are warped onto the topography read from
   `ptopo.mean.utm` by `read_topo.py`.
2. **export_mesh** — runs `export_mesh.py` to convert the `.msh` into the
   SPECFEM++ text mesh format under `MESH/`.
3. **decompose_mesh** — runs `xdecompose_mesh` to partition the mesh into
   `DATABASES_MPI/proc_*.bin` (4 ranks).
4. **run_solver** — runs `mpirun -n 4 specfem 3d` to produce seismograms in
   `OUTPUT_FILES/results/`.
5. **plot_seismogram** — produces `OUTPUT_FILES/seismogram_plot.png`.
