# Attenuation

Constant-Q attenuation with `N_SLS` standard linear solids, for 2D, 3D
Cartesian and 3D globe meshes, isotropic and anisotropic elastic media.

## Layers

Setup is split into three layers, each unaware of the one above it.

| Layer | Where | Knows about |
|---|---|---|
| **Fit**: Q → `tau_epsilon`, modulus defects, dispersion scales | `core/specfem/attenuation/` | Q, band, `tau_sigma` |
| **Shift**: reference → unrelaxed moduli at one GLL point | `assembly/attenuation/impl/attenuation_medium/dim3/modulus_shift.hpp`, `unrelaxed_tensor.hpp` | property tag, radial frame |
| **Container**: loop over GLL points, memory-variable rates | `assembly/attenuation/impl/attenuation_medium/dim{2,3}/constant_isotropic.tpp` | dimension, element ranges, I/O |

### Fit

- One formulation for every mesh: the 1/L-normalized Maxwell solid of
  Chapter 5 of Tromp's attenuation notes (§5.4–5.6), in `specfem::attenuation`.
  See `docs/sections/api/specfem/attenuation/index.rst` for the equations.
- `compute_tau_eps` is the only Q search: the Fortran codes' objective and
  fminsearch constants on `specfem::optimization::NelderMeadSimplex`, in
  double precision, throwing on an unconverged fit. It does not clip Q.
- `fit_quality_factor` returns a `QualityFactorFit`: `tau_eps`, the defects
  `beta` and `unrelaxed_ratio = M_U/M_R`, and `ModulusScales`
  (`relaxed = M_R/M_0`, `ratio = M_U/M_R`).
- `specfem::attenuation::QualityFactorTable` quantizes Q to buckets of
  `1 / constants::Q_RESOLUTION` and memoizes one fit per bucket.
  `QualityFactorFitTables` (bulk + shear) clips Q to `constants::Q_MAXIMUM`
  by default (3D); 2D passes 0 and uses Q as given.

### Shift

- `IsotropicModulusShift` scales `kappa` and `mu` by their multipliers.
- `AnisotropicModulusShift` shifts:
  - shear: globe's Love-parameter update in the point's `RadialFrame`
    (Cartesian z, geocentric, or inner-core hexagonal), in two stages
    (`relaxed`, then `ratio`);
  - bulk: an isotropic Voigt-kappa increment, applied when bulk attenuation
    is on.
- These go beyond the isotropic theory of the notes; the assumptions are
  listed under "Assumptions beyond the isotropic theory" in the API docs.
- Each returns the unrelaxed moduli that drive the relaxation rates.

### Container

`recompute()` is the same for every property tag:

1. Look up both Q values in the tables (passed in by the caller, or private
   to the call).
2. Call the shift.
3. Call `set_relaxation_rates()`, which uses
   `AttenuationPropertyValues::relaxation_rate()` (`M_R * beta_j / tau_sigma_j`,
   the `delta M_j / tau_sigma_j` of eqs. 5.64-5.65; doubled for shear).

Bulk attenuation is controlled only by the mesh's `bulk_attenuation` flag.
Globe meshes turn it off.

## Roadmap

These are follow-ups to PR #2192 (issue #2059), one PR each. Each step keeps
existing reference tests passing or states exactly which values move.

1. **One table per setup (issue #2060, done).** `QualityFactorTables`
   (bulk + shear) is shared by every container a setup recomputes: the globe
   property setup and the dim3 property reader pass one instance to the
   isotropic and anisotropic `recompute()`. Each `recompute()` logs its fit
   count at debug level. The tables hold a `std::map`, so they are owned by
   the setup code, not by `assembly::attenuation`, which is copied into
   device kernels. Materials still canonicalize Q with their own instance;
   that goes away with step 2, when materials stop fitting.
2. **One lifecycle.** Materials keep physical moduli everywhere. Isotropic
   materials currently hand out unrelaxed moduli from `get_properties()` and
   fit Q in `materials::apply_attenuation`. Make the container shift
   isotropic and anisotropic properties the same way, once, after the
   properties are filled. This removes the fit from the container
   constructor.
3. **2D anisotropic attenuation** through the same shift layer, with a 2D
   radial frame. The container loop is shared once the shift is
   dimension-generic.
4. **Per-GLL scales in 2D.** The dim2 container stores per-element modulus
   scales and keeps the last GLL point's value for GLL-varying Q. Store them
   per GLL point as dim3 does, so a write/read round trip is exact.
