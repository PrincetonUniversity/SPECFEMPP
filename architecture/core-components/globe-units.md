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

`globe::PlanetConstants` temporarily owns the database's opaque,
schema-versioned planet-value array. After replaying `MODEL_CONFIG`, the globe
model evaluator derives the selected planet's values and compares the complete
array with the database. The reader then discards `PlanetConstants`; neither the
raw mesh nor assembly retains a second source of planet data.

The `specfem::globe` component owns the planet schema, replayable model
configuration, unit conversion at the Fortran boundary, and model evaluator
adapter. The mesh layer owns only the globe-specific payload needed at assembly
time, while the I/O layer deserializes and validates database records. This keeps
mesh data independent of I/O implementation types and avoids retaining
verification-only state in Cartesian assemblies.

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
