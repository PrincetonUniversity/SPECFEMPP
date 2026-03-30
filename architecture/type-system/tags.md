# Tags

SPECFEM++ uses `enum class` tags as template parameters to select physics at compile time with zero runtime overhead.

## Dimension Tags

```cpp
enum class dimension_tag { dim2, dim3 };
```

All major data structures and algorithms are templated on `dimension_tag`. The compiler generates separate, optimized code paths for 2D and 3D.

---

## Medium Tags

```cpp
enum class medium_tag {
    elastic_psv,        // 2D P + SV waves (2 DOF: u_x, u_z)
    elastic_sh,         // 2D SH waves (1 DOF: u_y)
    elastic_psv_t,      // 2D PSV + Cosserat spin (3 DOF: u_x, u_z, ω_y)
    acoustic,           // Pressure waves (1 DOF: φ)
    poroelastic,        // Biot poroelastic (fluid+solid DOFs)
    electromagnetic_te, // 2D TE electromagnetic mode
    elastic,            // 3D elastic (3 DOF: u_x, u_y, u_z)
    elastic_spin,       // 3D elastic with spin
    electromagnetic,    // 3D electromagnetic
};
```

---

## Property Tags

```cpp
enum class property_tag {
    isotropic,          // Scalar λ, μ, ρ
    anisotropic,        // Full elastic tensor Cijkl
    isotropic_cosserat  // Isotropic + micropolar constants
};
```

---

## Boundary Tags

```cpp
enum class boundary_tag {
    none,
    acoustic_free_surface,
    stacey,
    composite_stacey_dirichlet
};
```

---

## Attenuation Tags

```cpp
enum class attenuation_tag {
    none,
    constant_isotropic  // Constant Q over frequency band
};
```

---

← [Back to Type System](index.md) | [Back to Index](../index.md)
