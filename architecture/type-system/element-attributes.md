# Element Attributes

**Files:** `core/specfem/element/`

`element::attributes<DimensionTag, MediumTag>` is a **traits class** providing compile-time information about each medium:

```cpp
// Example: 2D elastic PSV
using attrs = element::attributes<dim2, elastic_psv>;
static_assert(attrs::components == 2);      // u_x, u_z
static_assert(attrs::dimension == 2);
static_assert(attrs::has_cosserat_stress == false);
```

## Available Static Members

| Member | Type | Description |
|---|---|---|
| `components` | `int` | Number of DOF components per GLL point |
| `dimension` | `int` | Spatial dimension (2 or 3) |
| `has_cosserat_stress` | `bool` | Whether Cosserat couple-stress is active |

## Per-Medium Components

| Medium tag | `components` | DOFs |
|---|---|---|
| `elastic_psv` | 2 | u_x, u_z |
| `elastic_sh` | 1 | u_y |
| `elastic_psv_t` | 3 | u_x, u_z, ω_y |
| `acoustic` | 1 | φ (potential) |
| `elastic` (3D) | 3 | u_x, u_y, u_z |

This zero-cost abstraction allows the compiler to eliminate dead code and fully specialize physics for each medium type.

---

← [Back to Type System](index.md) | [Back to Index](../index.md)
