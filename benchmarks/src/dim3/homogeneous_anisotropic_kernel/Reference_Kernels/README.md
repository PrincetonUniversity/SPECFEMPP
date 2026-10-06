# SPECFEM3D reference kernels

`Kernels.png` is the SPECFEM3D panel this benchmark is compared against.

## How it was generated

From a SPECFEM3D checkout, in an example directory configured to match this
benchmark (domain 150 x 100 x 100 km, `NEX_XI=18`, `NEX_ETA=15`, `NZ=15`,
rho/vp/vs = 2300/2800/1500, force source at (110, 50, -50) km with Ricker
`f0 = 0.05`, station X40 at (40, 50, -50) km, `DT = 0.08`, `NSTEP = 800`,
cross-correlation traveltime adjoint source on the X component):

`DATA/meshfem3D_files/Mesh_Par_file`

```
NMATERIALS = 1
#material_id  #rho  #vp  #vs  #Q_Kappa  #Q_mu  #anisotropy_flag  #domain_id
1   2300.0   2800.0   1500.0  9999  9999  1  2

STACEY_ABSORBING_CONDITIONS    = .true.
TOP_FREE_SURFACE               = .true.
BOTTOM_FREE_SURFACE            = .false.
```

`DATA/Par_file`

```
ANISOTROPY                      = .true.    # applies model_aniso to the flag
ANISOTROPIC_KL                  = .true.    # write the 21 c_ij kernels
STACEY_ABSORBING_CONDITIONS     = .true.
STACEY_INSTEAD_OF_FREE_SURFACE  = .false.   # must match TOP_FREE_SURFACE above
SAVE_TRANSVERSE_KL              = .false.
```

Then run mesher, database generation, forward (`SIMULATION_TYPE=1`,
`SAVE_FORWARD=.true.`), adjoint source generation, and the kernel run
(`SIMULATION_TYPE=3`). The panel is plotted from
`OUTPUT_FILES/DATABASES/proc*_{rho,c11..c66}_kernel.bin` with the same layout,
slice (X-Z at y = 50 km, half-width 3.5 km) and 98th-percentile colour limits
as this benchmark's `plot_kernels.py`.

> `STACEY_INSTEAD_OF_FREE_SURFACE` is the easy one to get wrong. It defaults to
> `.true.` in the shipped SPECFEM3D examples, which makes the top boundary
> absorbing and silently removes every free-surface reflection. With it left at
> `.true.` the reference has different physics from this benchmark, and the two
> disagree on the Z component by six orders of magnitude.

## Agreement status

Forward wavefields agree to six digits (correlation 1.000000 on X, 0.999994 on
Y, 0.999999 on Z; amplitude ratios within 0.3%), including the transverse
components that exist only because of the anisotropy. The adjoint sources agree
to the same precision.

Kernel agreement has to be quoted with the source and receiver excluded. Both
points are kernel singularities holding 55-62% of the total kernel energy in
under 0.1% of the grid points, so a whole-slice correlation is dominated by
them and understates the agreement badly.

Excluding a 10 km ball around each of the source and receiver:

- The isotropic `homogeneous_elastic_kernel` benchmark agrees essentially
  exactly -- every kernel at correlation 1.00 and amplitude ratio 1.00-1.04
  (against 0.66-0.80 at ratio 1.6-2.2 with the singularities included).
- Here, rho, c11, c12, c13, c15, c22, c23, c25, c33, c35 and c55 agree at
  correlation 0.89-0.99 with amplitude ratios near 1.0.
- c14, c16, c24, c26, c34, c44, c45, c46, c56 and c66 retain a real
  disagreement, correlation 0.26-0.80 at amplitude ratio ~0.95.

The split is by Voigt index: every kernel carrying index 4 (yz) or 6 (xy)
disagrees, while those built only from 1, 2, 3 and 5 (xx, yy, zz, xz) agree.
Since x-z is the source-receiver plane, the affected kernels are exactly the
out-of-plane shear ones. It is not the Voigt scaling (c55 agrees at 0.95 while
c44 sits at 0.61 with the same factor of four) and not amplitude (ratios are
~0.95). The forward u_y matches SPECFEM3D to six digits, so the forward field
is sound; the lead points at the eps_yz / eps_xy strain terms on the kernel
side. These are also the lowest-amplitude kernels -- c44 rms is ~25x below c11
-- so some of the lost correlation is genuine low signal-to-noise.
