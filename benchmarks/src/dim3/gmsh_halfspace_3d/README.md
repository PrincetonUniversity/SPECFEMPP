# Wave propagation through homogeneous elastic halfspace (Gmsh external mesh)

This benchmark exercises the Gmsh external-mesh workflow: a 10 km × 10 km × 5 km
elastic box meshed with Gmsh, exported via `export_gmsh3d.py`, decomposed with
`xdecompose_mesh`, and simulated with `specfem3d`.

It is the benchmark companion to the
[Gmsh cookbook](../../../../docs/sections/cookbooks/wavepropagation/dim3/Gmsh/index.rst).

## Physics

- Homogeneous isotropic elastic halfspace
- Density: 2700 kg/m³, Vp: 6000 m/s, Vs: 3500 m/s
- Vertical force source at (5000, 5000, -500) m, 1 Hz Ricker wavelet
- Free surface at z = 0; Stacey absorbing boundaries on all other faces
- 5 s simulation (dt = 0.004 s, 1250 steps)
- Six surface receivers at 1–3 km offset from the source along the x-axis

## Running the benchmark

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

## Workflow stages

1. **create_mesh** — runs `create_mesh.py` to generate `halfspace.msh` via Gmsh
2. **export_mesh** — runs `export_mesh.py` to convert the `.msh` file into the
   SPECFEM++ text mesh format under `MESH/`
3. **decompose_mesh** — runs `xdecompose_mesh` to partition the mesh into
   `DATABASES_MPI/Database.bin`
4. **run_solver** — runs `specfem 3d` to produce seismograms in
   `OUTPUT_FILES/results/`
5. **plot_seismogram** — produces `OUTPUT_FILES/seismogram_plot.png`
