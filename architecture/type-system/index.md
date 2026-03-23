# Element Type System

SPECFEM++ uses a **tag-based** compile-time type system to represent element physics without runtime branching. All major data structures and physics functions are templated on combinations of these tags.

## How It Works

Tags are `enum class` values used as template parameters. The compiler generates separate, fully optimized code paths for each combination — there is no runtime `if` or `switch` over physics types.

```
element::attributes<DimensionTag, MediumTag>
         │
         ├── components  (number of DOF per GLL point)
         ├── dimension   (2 or 3)
         └── has_cosserat_stress  (bool)
```

## Tags

| Tag type | Selects |
|---|---|
| [`dimension_tag`](tags.md#dimension-tags) | 2D vs. 3D code paths |
| [`medium_tag`](tags.md#medium-tags) | Wave physics (acoustic, elastic, poroelastic, …) |
| [`property_tag`](tags.md#property-tags) | Material symmetry (isotropic, anisotropic, Cosserat) |
| [`boundary_tag`](tags.md#boundary-tags) | Boundary condition type |
| [`attenuation_tag`](tags.md#attenuation-tags) | Attenuation model |

## Pages

| Page | Description |
|---|---|
| [Tags](tags.md) | All enum definitions and their values |
| [Element Attributes](element-attributes.md) | `element::attributes` traits class |
| [Template Patterns](template-patterns.md) | How templates are used throughout the codebase |

---

← [Back to Index](../index.md)
