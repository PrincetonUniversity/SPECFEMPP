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

## Fully-qualified names

Implementation files (`.cpp`, `.tpp`) should use fully-qualified names rather than
wrapping code in deeply-nested namespace blocks:

```cpp
// PREFERRED in implementation files:
void specfem::mesh::reader::load(...) { ... }

// ACCEPTABLE for short files:
namespace specfem::mesh {
void reader::load(...) { ... }
} // namespace specfem::mesh
```
