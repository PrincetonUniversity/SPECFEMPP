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

The mesh database is authoritative for the fixed planet values resolved by the
mesher: `R_PLANET`, `RHOAV`, `ONE_MINUS_F_SQUARED`, `HOURS_PER_DAY`,
`SECONDS_PER_HOUR`, and `TOPO_MAXIMUM`. This preserves model-specific scale
overrides without maintaining a second Earth/Mars/Moon table in C++. After
replaying `MODEL_CONFIG`, the globe model evaluator cross-checks its `R_PLANET`
and `RHOAV` against the database.

The `specfem::globe` component owns this metadata, the replayable model
configuration, unit conversion at the Fortran boundary, and the model evaluator
adapter. The mesh layer owns the globe-specific raw mesh payload, the I/O layer
only deserializes that payload, and assembly consumes it while populating GLL
properties. This keeps mesh data independent of I/O implementation types and
avoids retaining globe-only state in Cartesian assemblies.

Discontinuity radii belong exclusively to the selected reference model and are
not stored in the mesh database or assembly. During its single initialization,
`globe::ModelEvaluator` queries the model oracle and validates the radii locally
before using that same evaluator instance to populate material properties. The
radii are then discarded. Validation requires every radius to be finite and
positive, with `r_icb < r_cmb < r_moho < R_PLANET` from the model catalog.

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
