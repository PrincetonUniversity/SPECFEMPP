# Globe Units and Planet Constants

SPECFEM++ has one units boundary for globe models:

- The thin globe database stores geometry and scales in SI units (metres and
  kilograms per cubic metre).
- The raw mesh and the assembled simulation state remain in SI units.
- The SPECFEM3D_GLOBE model catalog works internally with length divided by
  `R_PLANET` and density divided by `RHOAV`.
- Conversion to and from those non-dimensional values is confined to the C++
  globe-model oracle wrapper. Code outside that wrapper must not call
  `utilities::nondimensionalize` or `utilities::dimensionalize`.

`globe::PlanetConstants` is populated from the resolved values in the mesh
database. This preserves the exact scales and planet metadata used by the
mesher, including model-specific overrides. After replaying `MODEL_CONFIG`, the
globe-model oracle cross-checks its `R_PLANET` and `RHOAV` against the database.

Discontinuity radii are model values stored beside the planet metadata in the
database. `io::globe_model` independently derives them after replaying
`MODEL_CONFIG` and checks them against the stored values. Population validates
`0 < r_icb < r_cmb < r_moho < r_planet`.

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
