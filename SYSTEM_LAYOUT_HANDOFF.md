# Handoff: `SystemLayout` refactor — needs Trilinos verification

**Branch:** `issue-1982-system-layout` (off `issue-1982`)

**Read this first:** the code on this branch has **never been compiled against
Trilinos**. It was written on a machine with no Trilinos install, where every line
of it is excluded by `#ifdef SPECFEM_ENABLE_TRILINOS`. The non-Trilinos build
passes and the guards are correct — that is all that has been proven. Your job is
to build it with Trilinos, run the suite, and fix what breaks.

---

## What this branch does

It consolidates "how the discretization maps onto Tpetra objects" into one type,
`specfem::linear_system::SystemLayout<Tags>`, replacing `DofMap` and two
open-coded graph builders.

**Before:** `DofMap` owned the numbering and maps;
`StiffnessAssembler::build_graph` built the fully-connected graph for `K`;
`DampingAssembler::assemble` built a block-diagonal graph for `C` inline,
mid-function; `assemble_mass_vector` allocated its vector directly; and four
production sites hand-rolled the `(iglob, icomp)` ↔ Tpetra-row transfer loop.

**After:** `SystemLayout` owns the numbering, maps, sparsity graphs, and every
transfer. `DofMap` is gone. `gid()` is private.

```cpp
// structure
Teuchos::RCP<crs_matrix_type> full_matrix() const;                  // K's graph, cached
Teuchos::RCP<crs_matrix_type> block_diagonal_matrix(
    const std::function<bool(int)> &mask = {}) const;               // C's graph
Teuchos::RCP<vector_type>     create_vector() const;
const std::vector<global_ordinal_type> &element_column_gids(int) const;

// transfers
using host_field_view_type =
    Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::HostSpace>;
void scatter(const host_field_view_type &src, vector_type &dst) const;
Teuchos::RCP<vector_type> scatter(const host_field_view_type &src) const;
void gather(const vector_type &src, const host_field_view_type &dst) const;
void scatter_point_block(crs_matrix_type &, int iglob,
                         const host_field_view_type &block) const;
bool has_point_block(const crs_matrix_type &, int iglob) const;
```

Two design points that matter if you have to debug:

1. **`full_matrix()` caches its graph.** `StiffnessAssembler` and
   `ImplicitNewmarkSolver::form_operator` both call it, so `A` is built on the
   *same graph object* as `K`. This is what makes `form_operator`'s
   `replaceGlobalValues`/`sumIntoGlobalValues` calls hit every entry.
2. **Both graphs come from one numbering**, so every entry of a
   `block_diagonal_matrix` is also in the `full_matrix` graph by construction.
   `form_operator` sums `C` onto `K`'s graph and relies on this — it used to be an
   argued property in a comment, now it is structural and directly tested.

The refactor is intended to be **behaviour-preserving**. It is groundwork for
making the DOF ordering a free choice (see "Follow-on" below).

---

## Your job

```bash
cmake --preset release-trilinos
cmake --build build/release-trilinos -j
ctest --test-dir build/release-trilinos/tests -R \
  "StiffnessAssembler|DampingAssembler|MassVector|ElementStiffness|SystemLayout|ImplicitSolver|ImplicitNewmark|LinearSystemTrilinos"
```

Note `ctest --test-dir` must point at `build/<preset>/tests`, not `build/<preset>`.
Tests are registered by GoogleTest name, not by target name.

**Pass condition: every test green, with no assertion weakened or deleted.**

This is the whole point. The refactor claims to preserve behaviour, so if a test
fails, the refactor is wrong — do not "fix" it by loosening a tolerance, deleting a
check, or adjusting an expected value. Fix the code. If you conclude a test itself
was wrong, say so explicitly and explain why rather than quietly editing it.

Also confirm the non-Trilinos path still builds:

```bash
cmake --preset release && cmake --build build/release -j
ctest --test-dir build/release/tests -R "ElementStiffness"    # 4 tests, must pass
```

---

## Known risks, most likely first

**1. `gather` takes `const vector_type &` and calls
`getLocalViewHost(Tpetra::Access::ReadOnly)`.**
Believed to be a const-qualified overload in Tpetra 16.1, but unverified. If it
fails to compile, take the vector by non-const reference in
`SystemLayout::gather` (`system_layout.hpp`, `system_layout.cpp`) and update the
callers (`implicit_solver.cpp` `write_state_to_fields`, and the tests). Cosmetic
fix — do not restructure around it.

**2. `has_point_block` uses `matrix.getGlobalRowCopy(...)` with
`nonconst_global_inds_host_view_type` / `nonconst_values_host_view_type` buffers
sized by `getGlobalMaxNumRowEntries()`.** Same pattern as
`implicit_solver.cpp::form_operator`, so it should be fine, but it is called on
matrices that are *not* fill-complete during
`SystemLayout3D.BlockDiagonalEntriesLieInTheFullGraph`. If Tpetra objects to
`getGlobalRowCopy` before `fillComplete`, either fill-complete the probe matrix in
that test or reimplement `has_point_block` against the graph
(`matrix.getCrsGraph()`).

**3. `DampingAssembler3D.BlockDiagonalWithEmptyInteriorRows`** is the test that
proves the mask functor reproduces the old compact graph exactly (interior rows
empty, `ncomp` entries at damping points). If this fails, the bug is in
`SystemLayout::block_diagonal_matrix`'s handling of the `std::function` mask
(`system_layout.cpp`), not in the damping physics.

**4. `DampingAssembler3D.MassPathCrossCheck`** is the most sensitive test to the
rewritten addressing — it compares three separately-obtained quantities and was
changed from `gid`-indexed `std::vector` to `(iglob, icomp)` host views. A failure
here most likely means `scatter`/`gather` transpose or misaddress; check them
against `SystemLayout3D.ScatterGatherRoundTrip` first, which isolates that.

**5. Unity builds.** Three new files were added under
`core/specfem/linear_system/` and `tests/unit-tests/linear_system/`. The library
uses unity builds (batch size 8, disabled on Apple). If you hit ODR or
redefinition errors, check that `specfem::linear_system_impl::elastic_isotropic_tags`
— which is redeclared in each `.cpp` following the module's existing pattern —
does not collide under unity. This pattern predates the branch, but the branch
adds one more copy of it in `system_layout.cpp`.

---

## Files changed

New:
- `core/specfem/linear_system/system_layout.{hpp,cpp}`
- `tests/unit-tests/linear_system/system_layout_tests.cpp` (9 tests)

Modified:
- `core/specfem/linear_system/dof_map.hpp` — `DofMap` deleted; the file survives as
  the shared Tpetra type-alias header (`scalar_type`, `map_type`, `vector_type`,
  `crs_graph_type`, `crs_matrix_type`, `global_ordinal_type`)
- `core/specfem/linear_system/tpetra_assembler.{hpp,cpp}` — `build_graph` and
  `element_column_gids` removed; `assemble()` uses `layout_.full_matrix()`
- `core/specfem/linear_system/damping_assembler.{hpp,cpp}` — inline graph build and
  block scatter removed; uses `block_diagonal_matrix(mask)` +
  `scatter_point_block`
- `core/specfem/linear_system/mass_vector.{hpp,cpp}` — takes `SystemLayout`; body
  collapses to `layout.scatter(h_mass)`
- `core/specfem/solver/implicit_solver.{hpp,cpp}` — `dof_map_` → `layout_`; vectors
  via `create_vector()`; `A` via `full_matrix()`; transfers via `scatter`/`gather`
- `tests/unit-tests/linear_system/*`, `tests/unit-tests/solver/implicit_solver_tests.cpp`,
  `tests/unit-tests/serial.cmake`
- `docs/sections/api/specfem/linear_system/index.rst`

---

## Follow-on (do not do as part of verification)

Once the suite is green, there is a cheap, high-value experiment this refactor was
built to enable. Nothing in the Trilinos stack renumbers DOFs for you — `CrsMatrix`
stores rows in row-map order, `fillComplete` only sorts columns *within* a row, and
plain `RILUK` factors in the order it is handed. With level-of-fill pinned to 0
(`implicit_solver.cpp`), the fill pattern is exactly `A`'s graph, so the DOF
ordering decides what ILU drops — and therefore GMRES iteration count.

The current ordering is component-blocked (`gid = icomp * nglob + iglob`), which
puts the three components of the *same* mesh point `nglob` apart even though they
are the most tightly coupled DOFs in the system.

**Experiment:** flip the one line in `SystemLayout::gid` (private, in
`system_layout.hpp`) to point-blocked `iglob * ncomp + icomp`, rebuild, and compare
`gmres_->getNumIters()` per step on the same fixture. Same matrix, same spectrum,
different ordering — any difference is pure preconditioner order-sensitivity. No
consumer should need editing; if one does, that is a bug in this refactor worth
reporting.

Report the iteration counts rather than acting on them.

---

## Background documents

Not in the repo — on the originating machine under `~/.claude/plans/`:
`feasibility-crs-graph-from-adjacency-2026-08-31.md` (why this direction),
`okay-lets-break-this-rosy-parnas.md` (plan 1: structure factories),
`plan2-systemlayout-scatter-gather-2026-08-31.md` (plan 2: transfers).
Ask the user if you need them.
