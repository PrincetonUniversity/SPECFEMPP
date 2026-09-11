# Homogeneous 3D anisotropic kernels

This benchmark computes the full anisotropic elastic sensitivity kernel for a
small homogeneous 3D model. The material is declared with `anisotropy_flag = 1` in the
`Mesh_Par_file`, so SPECFEM++ reconstructs its 21 stiffnesses with its port of
SPECFEM3D's `model_aniso` and propagates genuinely anisotropic waves. The
forward and adjoint wavefields interact to give the density kernel and the 21
independent entries of the symmetric stiffness kernel.

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

`Reference_Kernels/Kernels.png` holds the SPECFEM3D panel for the same
anisotropic model; `Reference_Kernels/README.md` records the exact SPECFEM3D
configuration and how to regenerate it.

Clean generated results with:

```bash
uv run snakemake clean -j 1
```

The legacy `Par_File` generated beside the Snakefile documents the equivalent
SPECFEM3D setting, including `ANISOTROPIC_KL = .true.`.

## Integration status

This benchmark is the end-to-end acceptance workflow for the 3D anisotropic
kernel implementation and the property/container dispatch integrated in issue
2044.

There is no kernel-parameterization switch: the kernel parameterization always
follows the element's property tag. `anisotropy_flag = 1` makes SPECFEM++ tag
these elements `anisotropic`, so they get `c_ij` kernels as a consequence, not
as a separate request.

Because the meshfem3D database carries only the flag alongside
`(rho, vp, vs)`, the 21 stiffnesses are reconstructed by SPECFEM++'s port of
`model_aniso` (`core/specfem/io/mesh/impl/fortran/dim3/read_materials.cpp`),
matching what SPECFEM3D's `xgenerate_databases` does. With SPECFEM3D's default
perturbation factors, flag 1 adds the one-zeta P term
`c14 = c24 = c34 = -0.4 rho vp^2` to the isotropic-equivalent matrix.

The reference has been regenerated from SPECFEM3D with the same
`anisotropy_flag = 1` model (see `Reference_Kernels/README.md`).

**Forward physics is validated.** Against that reference the forward
wavefields agree to six digits -- correlation 1.000000 / 0.999994 / 0.999999 on
the X / Y / Z components, amplitude ratios within 0.3% -- including the
transverse motion that exists only because of the anisotropy. The adjoint
sources agree to the same precision.

**Kernels agree away from the source and receiver, except the out-of-plane
shear terms.** Both points are kernel singularities carrying 55-62% of the
total kernel energy in under 0.1% of the grid points, so whole-slice statistics
are dominated by them; the isotropic benchmark looks equally "wrong" by that
measure and is in fact exact. Excluding a 10 km ball around each point, rho,
c11, c12, c13, c15, c22, c23, c25, c33, c35 and c55 correlate at 0.89-0.99 with
amplitude ratios near 1.0, while c14, c16, c24, c26, c34, c44, c45, c46, c56
and c66 sit at 0.26-0.80. The split is exactly Voigt indices 4 (yz) and 6 (xy)
versus 1, 2, 3, 5 -- the out-of-plane shear kernels. See
`Reference_Kernels/README.md` for the full table and what it rules out.
