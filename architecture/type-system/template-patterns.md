# Template Design Patterns

SPECFEM++ makes extensive use of C++17 templates to achieve zero-cost abstraction. The following patterns appear throughout the codebase.

## 1. Tag Dispatch

Dimension, medium, property, and boundary tags are template parameters that route to correct physics at compile time:

```cpp
template <specfem::element::medium_tag MediumTag>
void compute_stress(/* ... */);

// Instantiated separately for elastic_psv, acoustic, etc.
```

No `if` or `switch` over media types at runtime.

## 2. Policy-Based Design

`chunk_config` and `parallel_configuration` encode hardware policies as types; execution iterators are instantiated from these policies:

```cpp
using config = specfem::parallel_configuration::chunk_config<
    dim2, ChunkSize, TileSize, NumThreads, VectorLanes, SIMD, ExecSpace>;

ChunkedDomainIterator<config> iter(assembly);
```

## 3. Traits Classes (CRTP-style)

`element::attributes<Dim, Medium>` provides static properties without virtual dispatch:

```cpp
constexpr int ndof = element::attributes<dim2, elastic_psv>::components; // 2
```

## 4. `if constexpr`

Used extensively inside physics functions to conditionally compile Cosserat, damping, or coupling terms:

```cpp
if constexpr (attrs::has_cosserat_stress) {
    // Cosserat couple-stress computation
}
```

Dead branches are completely removed by the compiler.

## 5. Kokkos Lambda Capture

Physics functions are passed as lambdas to `Kokkos::parallel_for` / `Kokkos::parallel_reduce`, enabling transparent GPU offload with no source-level changes:

```cpp
Kokkos::parallel_for("update_wavefields", policy,
    KOKKOS_LAMBDA(const int ielement, const int igll) {
        // Runs on GPU or CPU depending on build
    });
```

---

← [Back to Type System](index.md) | [Back to Index](../index.md)
