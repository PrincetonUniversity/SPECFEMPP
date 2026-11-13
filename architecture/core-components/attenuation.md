# Attenuation

Constant-Q attenuation with `N_SLS` standard linear solids, for 2D, 3D
Cartesian and 3D globe meshes, isotropic and anisotropic elastic media.

## Layers

Setup is split into three layers, each unaware of the one above it.

| Layer | Where | Knows about |
|---|---|---|
| **Fit**: Q → `tau_epsilon`, modulus defects, dispersion scales | `core/specfem/attenuation/` | Q, band, `tau_sigma`, formulation |
| **Shift**: reference → unrelaxed moduli at one GLL point | `assembly/attenuation/impl/attenuation_medium/dim3/modulus_shift.hpp`, `unrelaxed_tensor.hpp` | property tag, radial frame |
| **Container**: loop over GLL points, memory-variable rates | `assembly/attenuation/impl/attenuation_medium/dim{2,3}/constant_isotropic.tpp` | dimension, element ranges, I/O |

### Fit

- Two formulations exist today:
  - 2D: `specfem::attenuation` (1/L-normalized Maxwell model).
  - 3D: `specfem::attenuation::globe` (port of SPECFEM3D_GLOBE).
- Both fit with `specfem::optimization::NelderMeadSimplex`, which follows
  fminsearch. A caller that must reproduce a Fortran run passes its constants
  through `NelderMeadOptions`; the optimizer has no formulation switches.
- `specfem::attenuation::QualityFactorTable` quantizes Q to buckets of
  `1 / constants::Q_RESOLUTION` and memoizes one fit per bucket. The fit
  function is supplied by the caller, so the table works with either
  formulation. Globe additionally clips Q to `constants::Q_MAXIMUM`.

### Shift

- `IsotropicModulusShift` scales `kappa` and `mu` by their multipliers.
- `AnisotropicModulusShift` shifts:
  - shear: globe's Love-parameter update in the point's `RadialFrame`
    (Cartesian z, geocentric, or inner-core hexagonal), in two stages;
  - bulk: an isotropic Voigt-kappa increment, applied when bulk attenuation
    is on.
- Each returns the unrelaxed moduli that drive the relaxation rates.

### Container

`recompute()` is the same for every property tag:

1. Look up both Q values in the tables (passed in by the caller, or private
   to the call).
2. Call the shift.
3. Call `set_relaxation_rates()`.

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
3. **Formulation as a tag.** Replace the
   `if constexpr (DimensionTag == dim3)` switch in
   `medium/dim2/elastic/isotropic/material.hpp` with a formulation selected by
   type: 1/L for SPECFEM2D parity, globe elsewhere. Decide whether 2D keeps
   its own formulation or moves to globe's, against the 2D benchmark
   references.
4. **2D anisotropic attenuation** through the same shift layer, with a 2D
   radial frame. The container loop is shared once the shift is
   dimension-generic.
5. **Per-GLL scales in 2D.** The dim2 container stores per-element modulus
   scales and keeps the last GLL point's value for GLL-varying Q. Store them
   per GLL point as dim3 does, so a write/read round trip is exact.
