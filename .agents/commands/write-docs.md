---
argument-hint: <file-path>
---

You are a documentation writer for SPECFEM++. Generate documentation for the code at `$ARGUMENTS`.

## Step 1: Read and understand the code

1. Read the target file completely.
2. Search for usages of the class/functions in the codebase to understand how they are used.
3. Check git log for the file to understand its evolution and purpose.
4. Read related files (parent classes, template specializations, callers).

## Step 2: Generate Doxygen comments

Follow these SPECFEM++ documentation conventions:

- Use `/** ... */` block comments
- Always include `@brief` with a concise one-line summary
- Use `@param` for every parameter (describe purpose, not just type)
- Use `@return` for non-void functions
- Use `@tparam` for every template parameter, explaining constraints and valid values
- Use `\f$ ... \f$` for inline math, `\f[ ... \f]` for block equations
- Include `@code ... @endcode` examples showing typical usage when helpful
- **Do NOT add `@file` directives**
- Use `///< description` for trailing member variable documentation
- Keep descriptions concise -- explain *why* and *what*, not implementation mechanics

Example format:
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

## Step 3: Check for existing RST API page

Search in `docs/sections/api/` for an existing RST page for this class/namespace.

- If it exists, verify it matches the current API and update if needed.
- If it does not exist, create one following this pattern. **The RST underline must
  be exactly the same length as the header text (including backticks).** Count
  the characters carefully.

  For a class `specfem::mesh::reader` the RST page would be:

  ```rst
  ``specfem::mesh::reader``
  =========================

  .. doxygenclass:: specfem::mesh::reader
     :members:
  ```

  Here, ``specfem::mesh::reader`` is 25 characters (including the 4 backticks),
  so the underline is 25 `=` characters.

Add the new page to the appropriate `index.rst` toctree in the same directory.

## Step 4: Present changes

Show the proposed documentation changes (both Doxygen and RST) before applying them.
Wait for user approval before editing files.
