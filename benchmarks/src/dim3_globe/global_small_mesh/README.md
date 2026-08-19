# Simple global mesh

This benchmark creates a minimal six-chunk global mesh with `xmeshfem3D_globe`.

It is intentionally mesher-only:

- `NCHUNKS = 6`
- `NEX_XI = NEX_ETA = 16`
- `NPROC_XI = NPROC_ETA = 1`
- `MODEL = 1D_isotropic_prem`
- gravity, rotation, and attenuation are disabled

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
