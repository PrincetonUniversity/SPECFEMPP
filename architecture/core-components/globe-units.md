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

`constants::PlanetConstants` is selected from the database `PLANET_TYPE`. Its
planet-wide values come from the immutable `constants/globe.hpp` table. The
reader cross-checks the redundant database `R_PLANET` and `RHOAV` fields against
that table, catching a database associated with the wrong planet.

Discontinuity radii are model values, not planet constants and not database
fields. `io::globe_model` derives them after replaying `MODEL_CONFIG`; the
assembly then stores their SI values in its copy of `PlanetConstants`.
Consumers can check `has_radii()` before using them; the guarded `radii()`
accessor throws if they are unavailable. Population also validates
`0 < r_icb < r_cmb < r_moho < r_planet`.

---

← [Back to Core Components](index.md) | [Back to Index](../index.md)
