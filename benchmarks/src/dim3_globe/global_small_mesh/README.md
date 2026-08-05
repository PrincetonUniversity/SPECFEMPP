# Simple global mesh

This benchmark creates a minimal six-chunk global mesh with `xmeshfem3D_globe`.

It is intentionally mesher-only:

- `NCHUNKS = 6`
- `NEX_XI = NEX_ETA = 32`
- `NPROC_XI = NPROC_ETA = 1` (so the run uses 6 MPI ranks, one per chunk)
- `MODEL = 1D_isotropic_prem`
- ellipticity is enabled to exercise reference-anchor output; surface topography
  remains disabled because its external ETOPO dataset is not part of this benchmark
- gravity, rotation, and attenuation are disabled

`SPECFEMPP_DATABASE = .true.` is set, so the mesher also writes the thin SPECFEM++
mesh database (`DATABASES_MPI/proc??????_specfempp_database.bin`) alongside its own
solver databases. `check_database.py` validates those files: record framing,
node/element consistency, CSR adjacency symmetry, CMB/ICB node welding, boundary face
counts, and cross-rank agreement on the MPI interfaces. Both the snakemake workflow
and the CMake target run it.

Configure SPECFEM++ with the globe mesher enabled:

```bash
cmake -S . -B build/dim3_globe \
  -DSPECFEM_ENABLE_MPI=ON \
  -DSPECFEM_BUILD_MESHFEM3D_GLOBE=ON
cmake --build build/dim3_globe --target xmeshfem3D_globe
```

Then run the generated benchmark from the configured benchmark directory:

```bash
cd benchmarks/build/dim3_globe/global_small_mesh
uv run snakemake -j 1
```

Alternatively, run the CMake target:

```bash
cmake --build build/dim3_globe --target benchmark_dim3_globe_global_small_mesh
```
