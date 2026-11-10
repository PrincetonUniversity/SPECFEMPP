# SPECFEM++

[![Documentation Status](https://readthedocs.org/projects/specfem2d-kokkos/badge/?version=latest)](https://specfem2d-kokkos.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)<br>
![GCC (main)](https://jenkins.princeton.edu/buildStatus/icon?job=SpecFEM_KOKKOS%2FGNU_main&build=last&subject=GCC%20(main))
![IntelLLVM (main)](https://jenkins.princeton.edu/buildStatus/icon?job=SpecFEM_KOKKOS%2FIntel_main&build=last&subject=IntelLLVM%20(main))
![NVIDIA (main)](https://jenkins.princeton.edu/buildStatus/icon?job=SpecFEM_KOKKOS%2FNVIDIA_main&build=last&subject=NVIDIA%20(main))
[![Docker (main)](https://img.shields.io/github/actions/workflow/status/PrincetonUniversity/SPECFEMPP/docker.yml?label=Docker%20(main)&branch=main)](https://github.com/PrincetonUniversity/SPECFEMPP/actions/workflows/docker.yml)
[![codecov (main)](https://codecov.io/gh/PrincetonUniversity/SPECFEMPP/branch/main/graph/badge.svg)](https://app.codecov.io/gh/PrincetonUniversity/SPECFEMPP/tree/main)<br>
![GCC (devel)](https://jenkins.princeton.edu/buildStatus/icon?job=SpecFEM_KOKKOS%2FGNU_devel&build=last&subject=GCC%20(devel))
![IntelLLVM (devel)](https://jenkins.princeton.edu/buildStatus/icon?job=SpecFEM_KOKKOS%2FIntel_devel&build=last&subject=IntelLLVM%20(devel))
![NVIDIA (devel)](https://jenkins.princeton.edu/buildStatus/icon?job=SpecFEM_KOKKOS%2FNVIDIA_devel&build=last&subject=NVIDIA%20(devel))
[![Docker (devel)](https://img.shields.io/github/actions/workflow/status/PrincetonUniversity/SPECFEMPP/docker.yml?label=Docker%20(devel)&branch=devel)](https://github.com/PrincetonUniversity/SPECFEMPP/actions/workflows/docker.yml)
[![codecov (devel)](https://codecov.io/gh/PrincetonUniversity/SPECFEMPP/branch/devel/graph/badge.svg)](https://app.codecov.io/gh/PrincetonUniversity/SPECFEMPP/tree/devel)




## About

SPECFEM++ is a complete re-write of SPECFEM suite of packages (SPECFEM2D, SPECFEM3D, SPECFEM3D_GLOBE) using C++. Compared to the earlier version, SPECFEM++ code base provides:

 1. a robust and flexible code structure,
 2. modularity that allows for easy addition of new features,
 3. portability that allows the code to run on a variety of architectures (CPU, NVIDIA GPUs, Intel GPUs, AMD GPUs etc.), and
 4. a user-friendly build infrastructure that allows the code to be easily compiled and run on a variety of platforms.

<!--[specfempp-py](https://github.com/PrincetonUniversity/SPECFEMPP-py) is the official Python package for configuring and running SPECFEM++ with Python.-->

## Experimental Apple GPU implementation (`sycl-metal`)

This branch runs SPECFEM++ on Apple Silicon GPUs through Kokkos' SYCL backend
and AdaptiveCpp's generic single-pass (SSCP) compiler targeting Metal. It is
an application implementation relevant to the discussion in
[Kokkos #9750](https://github.com/kokkos/kokkos/issues/9750), with local
compatibility patches rather than a new Kokkos execution-space backend.
The implementation remains experimental and supports single precision only.

### Implementation entry points

- [Metal build integration](cmake/metal.cmake) validates the configuration,
  injects the prelude, and arranges automatic patching of fetched Kokkos 5.2.0.
- [Host/device dispatch prelude](cmake/metal/kokkos_acpp_prelude.hpp) defines
  `KOKKOS_IF_ON_HOST/DEVICE` and `DESUL_IF_ON_HOST/DEVICE` using AdaptiveCpp's
  target-specialized branches. SSCP does not define `__SYCL_DEVICE_ONLY__`.
- [Patch application](cmake/metal/apply_kokkos_patches.cmake) applies the nine
  patches below in filename order; use a fresh build directory when changing
  the series.
- [Chunk configuration](core/specfem/parallel_configuration/chunk_config.hpp)
  uses 16 elements per chunk in 2D and 2 in 3D, with a 3D field-output chunk
  of 1, to fit the tested threadgroup-memory budgets.
- [Stiffness kernels](core/specfem/compute/impl/stiffness_kernels.hpp) fix the
  Metal scratch address space to level 0 at compile time. A runtime choice
  between level-0 and level-1 pointers fails Metal address-space inference;
  see [AdaptiveCpp #2313](https://github.com/AdaptiveCpp/AdaptiveCpp/issues/2313).
- [Device check](tests/metal_device_check.cpp) supports verification that
  integration tests execute on the Metal device.

### Kokkos patch inventory

These patches are checked into
[`cmake/metal/kokkos-patches/`](cmake/metal/kokkos-patches/).
They describe the current local implementation, not a proposed upstream patch
set. Several deliberately restrict functionality or work around missing
AdaptiveCpp features and need redesign or further validation before upstreaming.

| Patch | Purpose and limits |
| --- | --- |
| [0001: compiler detection](cmake/metal/kokkos-patches/0001-cmake-accept-acpp-and-skip-dpcpp-flags.patch) | Detect AdaptiveCpp and skip DPC++ driver flags. |
| [0002: constexpr dispatch and abort](cmake/metal/kokkos-patches/0002-acpp-constexpr-dispatch-and-abort.patch) | Use the portable bit-manipulation path for constant evaluation and AdaptiveCpp's device-assert builtin for abort. Metal device abort does not provide normal host-abort semantics. |
| [0003: printf extension guard](cmake/metal/kokkos-patches/0003-printf-guard-oneapi-extension.patch) | Avoid the oneAPI printf extension; retain host printing and drop device output. |
| [0004: nan/remquo fallbacks](cmake/metal/kokkos-patches/0004-portable-nan-remquo.patch) | Supply missing math entry points. The remquo fallback is not validated across its full domain and can overflow while reconstructing quotient bits; it is not an upstream-ready implementation. |
| [0005: unsupported-feature guards](cmake/metal/kokkos-patches/0005-acpp-unsupported-feature-guards.patch) | Reject team reductions and UniqueToken uses requiring unavailable oneAPI extensions; guard desul's unsupported lock-based atomic paths. Includes a test-only no-op mode that application configuration rejects. |
| [0006: optional coarse-grained events](cmake/metal/kokkos-patches/0006-instance-optional-coarse-grained-events.patch) | Opt in for Kokkos-owned queues with `KOKKOS_ACPP_COARSE_GRAINED_EVENTS=1`; off by default and slower in the measured SPECFEM++ cases. |
| [0007: device-level kernel limits](cmake/metal/kokkos-patches/0007-acpp-device-level-kernel-limits.patch) | Replace unavailable kernel-bundle queries with device limits. These can overestimate a particular kernel's safe workgroup size. |
| [0008: platform diagnostics](cmake/metal/kokkos-patches/0008-acpp-print-platform-name.patch) | Print the platform name because AdaptiveCpp's backend enum lacks stream insertion. |
| [0009: reserved local memory](cmake/metal/kokkos-patches/0009-acpp-reserved-local-memory.patch) | Account for AdaptiveCpp Metal's additional 8 bytes per work-item when sizing teams and scratch. This encodes an implementation detail of the pinned runtime. |

The prelude is an additional required compatibility component, separate from
these nine patches. This branch does not implement the two-pass reduction/scan
or enlarged indirect-kernel buffer pool described in Kokkos #9750. Application
test passes here do not establish correctness of arbitrary Kokkos reductions
or cross-workgroup memory ordering; see
[AdaptiveCpp #2311](https://github.com/AdaptiveCpp/AdaptiveCpp/issues/2311).

### AdaptiveCpp dependencies

The tested toolchain uses AdaptiveCpp commit
[`f4ff36814e3d6fcbeb6e0a8483d3b18e16d5d508`](https://github.com/AdaptiveCpp/AdaptiveCpp/commit/f4ff36814e3d6fcbeb6e0a8483d3b18e16d5d508)
with the following seven local patches. **These AdaptiveCpp patch files and
their build script are maintained outside this repository and are not included
in this branch.** An unpatched AdaptiveCpp installation is not the tested
configuration.

| External patch filename | Purpose |
| --- | --- |
| `0001-event-store-async-handler.patch` | Retain the async handler passed to the event constructor. |
| `0002-metal-get-num-subgroups.patch` | Implement the missing Metal subgroup-count builtin; same approach as [AdaptiveCpp #2312](https://github.com/AdaptiveCpp/AdaptiveCpp/issues/2312). |
| `0003-event-keep-runtime-alive-through-node-destruction.patch` | Keep runtime/plugins alive until an event's DAG node has been destroyed. |
| `0004-metal-u32-fetch-add-memory-order.patch` | Add ordering fences around unsigned 32-bit fetch-add; this is not a complete fix for all Metal atomic ordering/visibility issues. |
| `0005-metal-scalarize-before-emission.patch` | Scalarize vector operations before Metal source emission. |
| `0006-jit-cache-recheck-after-lock.patch` | Recheck the JIT code-object cache after acquiring its mutex to avoid replacing an object another thread has already constructed. |
| `0007-metal-float-atomic-add-any-address-space.patch` | Handle float atomic add/sub for inferred device, threadgroup, and thread address spaces, using integer CAS for threadgroup memory. |

### Build configuration and validation scope

The [release-sycl-metal preset](CMakePresets.json) selects `acpp`, SYCL,
float, and the patched FetchContent Kokkos; CPU SIMD, MPI, oneDPL, VTK, and
Python bindings are disabled. It currently names the macOS 26.5 SDK explicitly.
With the patched AdaptiveCpp installation on `PATH` and that SDK available:

```sh
cmake --preset release-sycl-metal
cmake --build build/release-sycl-metal --target specfem -j4
```

Do not supply an external Kokkos installation. Before interpreting a run,
verify its startup log identifies the Apple GPU and AdaptiveCpp Metal backend;
the AdaptiveCpp host device is a separate device.

Recorded M4 validation at solver revision `62e6ebaf8` passed 32 integration
tests and six benchmark comparisons against a CPU float reference. After the
LIBDL change at `048374723`, a separate Metal build repeated those checks
successfully. These are recorded local results, not a complete Kokkos
conformance claim or Metal CI coverage.

On an Apple M4 (4 performance + 6 efficiency CPU cores, 24 GB), the recorded
3D elastic NGLL-5 solver time-loop comparisons were about 1.9x faster on Metal
at 1,620 elements and about 3.1x at 12,960 and 103,680 elements than the best
OpenMP + SIMD setting tested (4 or 10 threads, float, same machine and solver
source). These are warm-cache loop timings, not a claim of a fully tuned CPU
baseline. Detailed timing artifacts are maintained outside this repository.

Known scope limits include FP64 globe assembly (unsupported on Metal),
team-level reductions/scans and device facilities requiring missing oneAPI
extensions, unvalidated math fallback edge cases, and tiled NPY serialization.
Scratch fit must be checked for new orders/media; the tested application cases
do not establish safety for every instantiation.

## Documentation


The online documentation for SPECFEM++ is located
[here](https://specfem2d-kokkos.readthedocs.io/en/latest/index.html#)

## Getting Started with SPECFEM++

Follow the [Getting Started Guide](https://specfem2d-kokkos.readthedocs.io/en/latest/sections/getting_started/index.html) to install SPECFEM++ on your system and run the solver.

## Examples

We recommend starting with the [cookbook examples](https://specfem2d-kokkos.readthedocs.io/en/latest/sections/cookbooks/index.html) to learn how to customize the solver for your use case.

## Contributing to SPECFEM++

SPECFEM is a community project that lives by the participation of its members —
i.e., including you! It is our goal to build an inclusive and participatory
community so we are happy that you are interested in participating! Please see
[this
page](https://specfem2d-kokkos.readthedocs.io/en/latest/sections/developer_documentation/contributing.html)
for developer documentation.

In particular you should follow the git development workflow and pre-commit
style checks when contributing to SPECFEM++.

## License

SPECFEM++ is distributed under the [GPL v3 license](LICENSE)
