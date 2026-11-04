# Applies the AdaptiveCpp/Metal compatibility patch series to a pristine
# Kokkos source tree. Invoked as a FetchContent PATCH_COMMAND:
#
#   cmake -DKOKKOS_SOURCE_DIR=<dir> -DPATCH_DIR=<dir> -P apply_kokkos_patches.cmake
#
# `patch -p1` is used rather than `git apply`: the fetched source lives inside
# the SPECFEM++ build tree, which is itself inside a git work tree, and
# `git apply` would resolve paths against that repository instead of the
# Kokkos source directory.
#
# Idempotent: a patch that is already applied (reverse dry run succeeds) is
# skipped; any other failure is fatal.

if(NOT KOKKOS_SOURCE_DIR OR NOT PATCH_DIR)
  message(FATAL_ERROR "KOKKOS_SOURCE_DIR and PATCH_DIR are required")
endif()

find_program(SPECFEM_PATCH_EXECUTABLE patch REQUIRED)
file(GLOB _patches "${PATCH_DIR}/*.patch")
list(SORT _patches)
if(NOT _patches)
  message(FATAL_ERROR "No Kokkos patches found in ${PATCH_DIR}")
endif()

foreach(_patch IN LISTS _patches)
  get_filename_component(_name "${_patch}" NAME)
  execute_process(
    COMMAND "${SPECFEM_PATCH_EXECUTABLE}" -p1 -N -s --dry-run -i "${_patch}"
    WORKING_DIRECTORY "${KOKKOS_SOURCE_DIR}"
    RESULT_VARIABLE _forward OUTPUT_QUIET ERROR_QUIET)
  if(_forward EQUAL 0)
    execute_process(
      COMMAND "${SPECFEM_PATCH_EXECUTABLE}" -p1 -N -s --no-backup-if-mismatch
              -i "${_patch}"
      WORKING_DIRECTORY "${KOKKOS_SOURCE_DIR}"
      RESULT_VARIABLE _applied)
    if(NOT _applied EQUAL 0)
      message(FATAL_ERROR "Failed to apply Kokkos patch ${_name}")
    endif()
    message(STATUS "Applied Kokkos patch ${_name}")
    continue()
  endif()
  execute_process(
    COMMAND "${SPECFEM_PATCH_EXECUTABLE}" -p1 -R -s --dry-run -i "${_patch}"
    WORKING_DIRECTORY "${KOKKOS_SOURCE_DIR}"
    RESULT_VARIABLE _reverse OUTPUT_QUIET ERROR_QUIET)
  if(_reverse EQUAL 0)
    message(STATUS "Kokkos patch ${_name} already applied")
  else()
    message(FATAL_ERROR
      "Kokkos patch ${_name} neither applies nor is already applied to "
      "${KOKKOS_SOURCE_DIR}. Is this the pinned Kokkos version?")
  endif()
endforeach()
