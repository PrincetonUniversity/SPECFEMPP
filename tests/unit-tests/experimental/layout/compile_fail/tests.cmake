set(_layout_compile_source
    "${CMAKE_CURRENT_LIST_DIR}/invalid_layout.cpp")
add_library(layout_compile_control OBJECT EXCLUDE_FROM_ALL "${_layout_compile_source}")
target_link_libraries(layout_compile_control PRIVATE Kokkos::kokkos)

# This source order matches LAYOUT_COMPILE_CASE in invalid_layout.cpp.
set(_layout_diagnostics
    "topology"
    "nonnegative"
    "nonnegative"
    "overflow"
    "overflow"
    "overflow"
    "constant coordinate"
    "coordinate rank"
    "overflow"
    "nonnull backing"
    "overflow"
    "constant coordinate"
    "constant coordinate"
    "overflow"
    "overflow"
    "overflow"

    "layout broadcast"
    "uniqueness not certified"
    "uniqueness not certified"
    "uniqueness not certified"
    "uniqueness not certified"
    "layout broadcast"
    "read-only|modifiable lvalue|cannot assign"
    "ordinary element references"
    "layout permutation"
    "layout permutation"
    "layout permutation"
    "layout transform path"
    "layout permutation"
    "layout group"
    "layout group"
    "layout group"
    "layout ungroup"
    "layout ungroup"
    "layout transform path"
    "constant coordinate"
    "read-only|modifiable lvalue|cannot assign"
    "one selector per top-level mode"
    "tuple selector must match a nested mode"
    "tuple selector must match a nested mode"
    "selector must be an index, all, range or tuple"
    "subview index out of range"
    "range out of order or out of range"
    "not aligned to nested mode blocks"
    "read-only|modifiable lvalue|cannot assign"
    "range bounds must be integers"
    "range out of order or out of range"
    "range out of order or out of range"
    "range bounds must be integers"
    "selector must be an index, all, range or tuple")
foreach(_mode IN ITEMS debug release)
  set(_case 0)
  foreach(_diagnostic IN LISTS _layout_diagnostics)
    math(EXPR _case "${_case} + 1")
    set(_target "layout_invalid_${_mode}_${_case}")
    add_library(${_target} OBJECT EXCLUDE_FROM_ALL "${_layout_compile_source}")
    target_link_libraries(${_target} PRIVATE Kokkos::kokkos)
    target_compile_definitions(${_target} PRIVATE
      LAYOUT_COMPILE_CASE=${_case}
      LAYOUT_COMPILE_DEBUG=$<STREQUAL:${_mode},debug>)
    set_target_properties(${_target} PROPERTIES UNITY_BUILD OFF)
    add_test(NAME HierarchicalLayoutCompile.${_mode}_${_case}
      COMMAND ${CMAKE_COMMAND}
        "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DTARGET=${_target}"
        "-DEXPECTED=${_diagnostic}"
        -P "${CMAKE_CURRENT_LIST_DIR}/check_failure.cmake")
    # Each test invokes the build tool in the same build directory.
    set_tests_properties(HierarchicalLayoutCompile.${_mode}_${_case}
      PROPERTIES RUN_SERIAL TRUE TIMEOUT 300 LABELS "layout_compile")
  endforeach()
endforeach()
