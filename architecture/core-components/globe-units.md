# Globe Units and Planet Constants

SPECFEM++ has one units boundary for globe models:

- The thin globe database stores geometry and scales in SI units (metres and
  kilograms per cubic metre).
- The raw mesh and the assembled simulation state remain in SI units.
- The SPECFEM3D_GLOBE model catalog works internally with length divided by
  `R_PLANET` and density divided by `RHOAV`.
- Conversion to and from those non-dimensional values is confined to the C++
  globe model evaluator. Code outside that evaluator must not call
  `globe::nondimensionalize` or `globe::dimensionalize`.

`globe::PlanetConstants` is populated from the resolved values in the mesh
database. This preserves the exact scales and planet metadata used by the
mesher, including model-specific overrides. After replaying `MODEL_CONFIG`, the
globe model evaluator cross-checks its `R_PLANET` and `RHOAV` against the
database.

The `specfem::globe` component owns this metadata, the replayable model
configuration, unit conversion at the Fortran boundary, and the model evaluator
adapter. The mesh layer owns the globe-specific raw mesh payload, the I/O layer
only deserializes that payload, and assembly consumes it while populating GLL
properties. This keeps mesh data independent of I/O implementation types and
avoids retaining globe-only state in Cartesian assemblies.

Discontinuity radii are model values stored beside the planet metadata in the
database. `globe::ModelEvaluator` independently derives them after replaying
`MODEL_CONFIG` and checks them against the stored values. Population validates
`0 < r_icb < r_cmb < r_moho < r_planet`.

The evaluator also compares the database's opaque model codes and flags with the
values derived by the linked Fortran catalog. C++ does not interpret those raw
values; they only detect catalog/database version skew. Every catalog call is
serialized because upstream routines retain module and `save` scratch state.
File-backed models are rejected before Fortran initialization when the runtime
`DATA/` directory is absent, avoiding an unrecoverable Fortran `STOP`.

Reference-model consumers use dedicated evaluator accessors rather than the 3-D
element path. `reference_density()` exposes the pure planet reference profile in
SI for gravity setup, while `ellipticity_spline()` returns the exact
Clairaut/Radau spline constructed by the mesher catalog. The density integration
and rotation-rate physics therefore remain on the Fortran side of the units
boundary.

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
