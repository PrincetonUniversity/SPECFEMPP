# Homogeneous 3D anisotropic kernels

This benchmark computes the full anisotropic elastic sensitivity kernel for a
small homogeneous 3D model. It follows SPECFEM3D's `ANISOTROPIC_KL = .true.`
workflow: the forward and adjoint wavefields propagate through an isotropic
elastic material, while their interaction is accumulated as the density kernel
and the 21 independent entries of the symmetric stiffness kernel.

The geometry and source-receiver pair are intentionally shared with the
`homogeneous_elastic_kernel` benchmark so that the isotropic and anisotropic
parameterizations can be compared directly.

## Run

Configure and build SPECFEM++ with benchmarks enabled, then run from the
configured benchmark directory:

```bash
uv run snakemake -j 1
```

For an MPI build, the configured Snakefile runs the mesher and solver with six
ranks. A cluster executor can be selected in the usual Snakemake manner.

The workflow creates:

- `OUTPUT_FILES/kernels/Kernels/elastic_anisotropic/{rho,c11,...,c66}.npy`
- `OUTPUT_FILES/kernels.png`, showing an X-Z slice through all 22 kernels
- forward and adjoint logs plus the saved forward wavefield

Clean generated results with:

```bash
uv run snakemake clean -j 1
```

The legacy `Par_File` generated beside the Snakefile documents the equivalent
SPECFEM3D setting, including `ANISOTROPIC_KL = .true.`.

## Integration status

This benchmark is the end-to-end acceptance workflow for the 3D anisotropic
kernel implementation. On the prerequisite kernel branch, its CMake generation,
serial/MPI DAGs, and output tooling are testable. Running the adjoint rule also
requires the subsequent 3D anisotropic property and dispatch integration from
issue 2044.
