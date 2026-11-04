# Apple GPU (Metal) support through AdaptiveCpp's generic SSCP compiler.
#
# SPECFEM_ENABLE_METAL builds SPECFEM++ and a patched, FetchContent-provided
# Kokkos 5.2.0 with `acpp`. It
#   * validates the configuration (SYCL backend, single precision, no SIMD,
#     no atomics bypass, no external Kokkos);
#   * force-includes the Kokkos/AdaptiveCpp prelude into every C++ translation
#     unit (Kokkos and SPECFEM++ alike), so host/device dispatch macros are
#     resolved per target;
#   * exposes SPECFEM_METAL_KOKKOS_PATCH_COMMAND, used by cmake/kokkos.cmake to
#     apply the compatibility patch series in cmake/metal/kokkos-patches/.
#
# The AdaptiveCpp installation itself must carry the local patches documented
# with the Metal port; it is found through CMAKE_CXX_COMPILER=acpp.
#
# Must be included before cmake/kokkos.cmake.

option(SPECFEM_ENABLE_METAL
  "Build for Apple GPUs (Metal) with AdaptiveCpp (requires Kokkos_ENABLE_SYCL)" OFF)

if(NOT SPECFEM_ENABLE_METAL)
  return()
endif()

message(STATUS "Configuring SYCL build for Apple GPUs (AdaptiveCpp/Metal)")

set(_metal_errors "")
if(NOT Kokkos_ENABLE_SYCL)
  list(APPEND _metal_errors "Kokkos_ENABLE_SYCL must be ON")
endif()
if(SPECFEM_ENABLE_DOUBLE_PRECISION)
  list(APPEND _metal_errors
    "SPECFEM_ENABLE_DOUBLE_PRECISION must be OFF (Metal has no FP64)")
endif()
if(SPECFEM_ENABLE_SIMD)
  list(APPEND _metal_errors "SPECFEM_ENABLE_SIMD must be OFF")
endif()
if(Kokkos_ENABLE_ATOMICS_BYPASS)
  list(APPEND _metal_errors
    "Kokkos_ENABLE_ATOMICS_BYPASS must be OFF (GPU kernels need real atomics)")
endif()
if(DEFINED Kokkos_ROOT OR DEFINED ENV{Kokkos_ROOT} OR DEFINED KOKKOS_PATH)
  list(APPEND _metal_errors
    "Kokkos_ROOT/KOKKOS_PATH must not be set: Metal requires the patched "
    "Kokkos fetched by cmake/kokkos.cmake")
endif()
get_filename_component(_metal_cxx_name "${CMAKE_CXX_COMPILER}" NAME)
if(NOT _metal_cxx_name MATCHES "^acpp")
  list(APPEND _metal_errors
    "CMAKE_CXX_COMPILER must be AdaptiveCpp's acpp (got ${CMAKE_CXX_COMPILER})")
endif()
if(_metal_errors)
  list(JOIN _metal_errors "\n  - " _metal_error_text)
  message(FATAL_ERROR
    "SPECFEM_ENABLE_METAL configuration errors:\n  - ${_metal_error_text}")
endif()

# Test-only Kokkos escape hatch; it turns unsupported features into silent
# no-ops and must never reach an application build.
if(CMAKE_CXX_FLAGS MATCHES "KOKKOS_IMPL_ACPP_UNSUPPORTED_AS_NOOP")
  message(FATAL_ERROR
    "KOKKOS_IMPL_ACPP_UNSUPPORTED_AS_NOOP is for Kokkos' own tests only")
endif()

set(SPECFEM_METAL_DIR "${CMAKE_CURRENT_LIST_DIR}/metal")
set(SPECFEM_METAL_PRELUDE "${SPECFEM_METAL_DIR}/kokkos_acpp_prelude.hpp")
string(APPEND CMAKE_CXX_FLAGS " -include ${SPECFEM_METAL_PRELUDE}")

set(SPECFEM_METAL_KOKKOS_PATCH_COMMAND
  "${CMAKE_COMMAND}"
  "-DKOKKOS_SOURCE_DIR=<SOURCE_DIR>"
  "-DPATCH_DIR=${SPECFEM_METAL_DIR}/kokkos-patches"
  -P "${SPECFEM_METAL_DIR}/apply_kokkos_patches.cmake")

# oneDPL is Intel-only; libdl handling is already done in cmake/kokkos.cmake.
set(Kokkos_ENABLE_ONEDPL OFF CACHE BOOL
  "Enable oneDPL for Kokkos SYCL algorithms" FORCE)

add_compile_definitions(SPECFEM_ENABLE_METAL)
