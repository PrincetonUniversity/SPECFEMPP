# Globe Units and Planet Constants

SPECFEM++ has one units boundary for globe models:

- The thin globe database stores geometry and scales in SI units (metres and
  kilograms per cubic metre).
- The raw mesh and the assembled simulation state remain in SI units.
- The SPECFEM3D_GLOBE model catalog works internally with length divided by
  `R_PLANET` and density divided by `RHOAV`.
- Conversion to and from those non-dimensional values is a private
  implementation detail of `globe::ModelEvaluator`. Code outside the evaluator
  remains in SI units; no public catalog-unit conversion API is provided.

The mesh database is authoritative for the fixed planet values resolved by the
mesher: `R_PLANET`, `RHOAV`, `ONE_MINUS_F_SQUARED`, `HOURS_PER_DAY`,
`SECONDS_PER_HOUR`, and `TOPO_MAXIMUM`. This preserves model-specific scale
overrides without maintaining a second Earth/Mars/Moon table in C++. The raw
mesh retains `globe::PlanetConstants` until assembly, when the configured model
evaluator cross-checks its selected planet, `R_PLANET`, and `RHOAV` against the
database. The assembled simulation does not retain a second source of planet
data.

The `specfem::globe` component owns the planet schema, replayable model
configuration, unit conversion at the Fortran boundary, and model evaluator
adapter. The mesh layer owns only the globe-specific payload needed at assembly
time, while the I/O layer deserializes and validates database records. This keeps
mesh data independent of I/O implementation types and avoids retaining
verification-only state in Cartesian assemblies.

The raw mesh also retains the database's opaque model codes and flags until
assembly. The evaluator compares them with values derived by the linked Fortran
catalog; C++ does not interpret those values because they exist only to detect
catalog/database version skew. Validation uses the same evaluator instance that
populates material properties, so the process-global Fortran catalog is
initialized only once. Every catalog call is serialized because upstream
routines retain module and `save` scratch state. File-backed models are rejected
before Fortran initialization when the runtime `DATA/` directory is absent,
avoiding an unrecoverable Fortran `STOP`.

Reference-model consumers use dedicated evaluator accessors rather than the 3-D
element path. `reference_density()` exposes the pure planet reference profile in
SI for gravity setup, while `ellipticity_spline()` returns the exact
Clairaut/Radau spline constructed by the mesher catalog. The density integration
and rotation-rate physics therefore remain on the Fortran side of the units
boundary.

Discontinuity radii belong exclusively to the selected reference model and are
not stored in the mesh database or assembly. During its single initialization,
`globe::ModelEvaluator` queries the model oracle and validates the radii locally
before using that same evaluator instance to populate material properties. The
radii are then discarded. Validation requires every radius to be finite and
positive, with `r_icb < r_cmb < r_moho < R_PLANET` from the model catalog.

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
