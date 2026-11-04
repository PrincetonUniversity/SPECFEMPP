# Pre-existing issues deferred from the Metal port

Scope decision (2026-11-04): fix issues directly caused by the Metal
implementation. Record pre-existing application or dependency defects here;
do not fix them as part of this port merely because Metal exposes them.

## Tiled NPY serialization

The NPY dataset writer dumps raw storage while declaring logical Fortran order.
DomainView storage is tiled on GPU backends. Output may interleave elements and
truncate a partial final tile. Reordering coordinates cannot recover missing
values. The corresponding raw-buffer reader has the same layout assumption.

Affected code: `core/specfem/io_backends/NPY/impl/dataset.tpp` and medium/property/
kernel output using DomainView host mirrors. The logical-index packing fix and
its partial-tile regression test were reverted before committing, as requested.
Historical validation of that removed fix was 25 passing CPU I/O tests; it does
not validate the current implementation. Saved 3D acoustic benchmark outputs
produce seven invalid-geometry records under the stricter comparison runner.

Consequence: affected output comparisons cannot close G3. Report them as known
limitations, not numerical passes. Any eventual fix belongs in separate work.

## Stiffness level-1 scratch fallback

The pre-existing fallback allocates level-1 scratch, but stiffness scratch views
use `team_scratch(0)`. Taking the fallback can corrupt memory. Metal's smaller
scratch budget exposed this in 3D attenuation; it is not caused by Metal code.

The backend-independent fix in commit `4b9106c3f` has been reverted. Metal chunk
sizes remain 2D 16 / 3D 2, with output chunk size 1 in 3D, as Metal-specific
memory-budget configuration. Those sizes do not prove safety for every order
and instantiation. Do not assume over-budget configurations now fail safely:
the original fallback is restored. Verify scratch fit before running new cases.

## Kokkos SYCL staging-lock exception safety

Kokkos can retain its staging mutex after a failed kernel launch, deadlocking
later submissions. This is a pre-existing SYCL exception-safety defect.

Removed `0010-sycl-staging-lock-raii.patch` from the port's FetchContent patch
series (introduced in `1370770e7`). The source-controlled series now contains
nine patches. Error-path recovery is not guaranteed; isolate failing cases in
separate processes. Existing fetched Kokkos trees and binaries may still contain
the removed patch: deleting a patch file does not reverse an already applied
patch. Use a fresh build directory for validation of this revised patch series.
Earlier ten-patch results remain historical evidence, not validation of the
new nine-patch stack.

## Zero 3D acoustic sensitivity kernels

The homogeneous acoustic kernel benchmark produces identically zero sensitivity
kernels on both CPU and Metal on this branch. No fix was made. Agreement between
two zero outputs is not proof of sensitivity-kernel correctness. This remains
outside the Metal port's validated scope.

## Review of the five port commits

- `5da67d3ef`: retained. Device arithmetic must avoid implicit FP64 for Metal's
  float-only JIT. These are Metal compatibility edits, not unrelated repairs.
- `4b9106c3f`: reverted; see scratch fallback above.
- `d4f51a441`: retained. Consolidates existing CUDA/HIP/SYCL checks into the
  shared GPU configuration macro without intentionally changing backend scope.
- `1370770e7`: retained except patch 0010. Build configuration, compatibility
  prelude/patches, and Metal-only chunk limits implement the new backend.
- `c44e8fb3b`: retained. Metal startup selection, verification and tests prevent
  the new backend from silently selecting the CPU.

No earlier commits before the port base `45a1c67f9` were altered. The removals
are recorded in a follow-up commit rather than rewriting existing history.
