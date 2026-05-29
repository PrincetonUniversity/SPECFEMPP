---
paths:
  - "core/**"
  - "docs/**"
---

# Documentation Conventions

## Doxygen (C++ inline documentation)

- Use `/** ... */` block comment style for classes, functions, and namespaces.
- Required tags:
  - `@brief` -- always, one-line summary
  - `@param` -- for every parameter (describe purpose, not just type)
  - `@return` -- for non-void functions
  - `@tparam` -- for every template parameter, explaining constraints
- Math notation: `\f$ ... \f$` for inline, `\f[ ... \f]` for display equations.
- Usage examples: `@code ... @endcode` blocks.
- **Do NOT add `@file` directives** at the top of files.
- Member variables: use `///< brief description` trailing comments.
- Keep descriptions concise -- explain *why* and *what*, not implementation mechanics.

Example:
```cpp
/**
 * @brief Compute the stiffness interaction for an element
 *
 * Evaluates the weak form of the elastic wave equation on a single
 * spectral element using GLL quadrature.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @tparam MediumTag Physics type (elastic, acoustic, etc.)
 * @param element The element to compute stiffness for
 * @param field Current displacement field
 * @return Stiffness contribution at each GLL point
 */
```

## RST documentation (Sphinx)

- Documentation source lives in `docs/sections/`.
- API reference uses Breathe directives that pull from Doxygen XML output:
  ```rst
  .. doxygenclass:: specfem::mesh::reader
     :members:
  ```
- When adding or modifying a public class, ensure a corresponding RST file exists
  in `docs/sections/api/specfem/<namespace>/`.
- RST style: use reStructuredText directives, not Markdown syntax.
- Parameter documentation uses hierarchical dropdown format with defaults, types,
  and constraints.

## Documentation sync

When modifying public APIs, update both:
1. Doxygen comments in the header file
2. The corresponding RST page under `docs/sections/api/`

If removing or renaming a public API, update all RST references. Do not leave
documentation pointing to non-existent functions.
