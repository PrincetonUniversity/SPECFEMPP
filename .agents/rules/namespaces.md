# Namespace Organization

## Hierarchy

All project code lives under the `specfem` top-level namespace. Maximum depth is **3 levels**:

```
specfem::component::sub_component
```

Examples of correct nesting:
- `specfem::mesh` (level 2)
- `specfem::assembly::impl` (level 3)
- `specfem::medium_physics::impl` (level 3)

Do NOT create deeper nesting like `specfem::io::mesh::impl::fortran::dim2` (6 levels).

## `_impl` namespaces

Implementation details that should not be part of the public API go in a `_impl` suffix
namespace at the same level as the public namespace:

```cpp
// Public API
namespace specfem::io::sources {
  void read_sources(...);
}

// Implementation helpers (not public)
namespace specfem::io::sources_impl {
  std::string trim(const std::string &s);
  std::string to_lower(const std::string &s);
}
```

This is the project's alternative to anonymous namespaces (which are banned due to
unity builds).

## Utility placement

General-purpose utilities (string manipulation, math helpers, type traits) belong in
`specfem::utilities`, NOT inside specific implementation namespaces. If you write a
`trim()`, `to_lower()`, or similar general helper, place it in `specfem::utilities`.

Only put helpers in `_impl` namespaces when they are genuinely specific to one module.

## Namespace closing comments

Always annotate closing braces with the namespace name:

```cpp
namespace specfem {
namespace mesh {
// ...
} // namespace mesh
} // namespace specfem
```

## Definitions in implementation files (`.cpp`, `.tpp`)

In implementation files (`.cpp` and `.tpp`), **always** use fully-qualified names for
function definitions. Do NOT wrap definitions in namespace blocks:

```cpp
// CORRECT in .cpp and .tpp files:
void specfem::io::sources_impl::parse_format_key(...) { ... }

// WRONG in .cpp and .tpp files:
namespace specfem::io::sources_impl {
void parse_format_key(...) { ... }
} // namespace sources_impl
```

## Declarations in header files (`.hpp`)

In header files (`.hpp`), use namespace blocks for declarations:

```cpp
// CORRECT in .hpp files:
namespace specfem {
namespace mesh {

class reader {
  void load(...);
};

} // namespace mesh
} // namespace specfem
```
