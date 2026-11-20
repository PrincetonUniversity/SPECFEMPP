# Compile-fail tests. Each test builds one object target that must fail to
# compile with an expected message. check_failure.cmake first builds a control
# target from the same source, so a broken build cannot pass as a rejection.

function(specfem_layout_compile_fail name source control expected)
  foreach(_mode IN ITEMS debug release)
    set(_target "layout_compile_${name}_${_mode}")
    add_library(${_target} OBJECT EXCLUDE_FROM_ALL "${source}")
    target_link_libraries(${_target} PRIVATE Kokkos::kokkos)
    target_compile_definitions(${_target} PRIVATE ${ARGN}
      LAYOUT_COMPILE_DEBUG=$<STREQUAL:${_mode},debug>)
    set_target_properties(${_target} PROPERTIES UNITY_BUILD OFF)
    add_test(NAME HierarchicalLayoutCompile.${name}_${_mode}
      COMMAND ${CMAKE_COMMAND}
        "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DCONTROL=${control}"
        "-DTARGET=${_target}"
        "-DEXPECTED=${expected}"
        -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/check_failure.cmake")
    # Each test invokes the build tool in the same build directory.
    set_tests_properties(HierarchicalLayoutCompile.${name}_${_mode}
      PROPERTIES RUN_SERIAL TRUE TIMEOUT 300 LABELS "layout_compile")
  endforeach()
endfunction()

# Read "#if/#elif <macro> == N" lines from source into <out>_numbers and
# <out>_expected. The expected message follows "// expect:" on the same line
# or on the next line. Duplicate numbers and missing messages fail configure.
# Editing the source re-runs configure, which re-registers its tests.
function(specfem_layout_read_cases source macro out)
  # Re-run configure when cases change, so new cases are registered.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
  file(STRINGS "${source}" _lines)
  set(_numbers "")
  set(_expected "")
  set(_pending "")
  foreach(_line IN LISTS _lines)
    if(_line MATCHES "^#(el)?if ${macro} == ([0-9]+)")
      set(_number "${CMAKE_MATCH_2}")
      if(_number IN_LIST _numbers)
        message(FATAL_ERROR "${source}: ${macro} ${_number} appears twice")
      endif()
      if(NOT _pending STREQUAL "")
        message(FATAL_ERROR "${source}: ${macro} ${_pending} has no expect line")
      endif()
      set(_pending "${_number}")
      if(_line MATCHES "// expect: (.+)$")
        list(APPEND _numbers "${_pending}")
        string(REPLACE ";" "\\;" _message "${CMAKE_MATCH_1}")
        list(APPEND _expected "${_message}")
        set(_pending "")
      endif()
    elseif(NOT _pending STREQUAL "" AND _line MATCHES "^ *// expect: (.+)$")
      list(APPEND _numbers "${_pending}")
      string(REPLACE ";" "\\;" _message "${CMAKE_MATCH_1}")
      list(APPEND _expected "${_message}")
      set(_pending "")
    endif()
  endforeach()
  if(NOT _pending STREQUAL "")
    message(FATAL_ERROR "${source}: ${macro} ${_pending} has no expect line")
  endif()
  set(${out}_numbers "${_numbers}" PARENT_SCOPE)
  set(${out}_expected "${_expected}" PARENT_SCOPE)
endfunction()

# Listed cases: one test per LAYOUT_COMPILE_CASE in invalid_layout.cpp.
set(_layout_listed "${CMAKE_CURRENT_LIST_DIR}/invalid_layout.cpp")
add_library(layout_compile_control OBJECT EXCLUDE_FROM_ALL "${_layout_listed}")
target_link_libraries(layout_compile_control PRIVATE Kokkos::kokkos)
specfem_layout_read_cases("${_layout_listed}" LAYOUT_COMPILE_CASE _listed)
foreach(_number _message IN ZIP_LISTS _listed_numbers _listed_expected)
  specfem_layout_compile_fail(case_${_number} "${_layout_listed}"
    layout_compile_control "${_message}" LAYOUT_COMPILE_CASE=${_number})
endforeach()

# Root-constructor matrix: every LAYOUT_BAD row in root_matrix.cpp against
# every LAYOUT_ENTRY constructor. Each entry has its own control build of a
# valid descriptor through the same constructor.
set(_layout_matrix "${CMAKE_CURRENT_LIST_DIR}/root_matrix.cpp")
specfem_layout_read_cases("${_layout_matrix}" LAYOUT_BAD _rows)
file(STRINGS "${_layout_matrix}" _entry_lines REGEX "^#(el)?if LAYOUT_ENTRY == [0-9]+")
set(_entries "")
foreach(_line IN LISTS _entry_lines)
  string(REGEX MATCH "== ([0-9]+)" _ "${_line}")
  list(APPEND _entries "${CMAKE_MATCH_1}")
endforeach()
foreach(_entry IN LISTS _entries)
  set(_control "layout_matrix_control_${_entry}")
  add_library(${_control} OBJECT EXCLUDE_FROM_ALL "${_layout_matrix}")
  target_link_libraries(${_control} PRIVATE Kokkos::kokkos)
  target_compile_definitions(${_control} PRIVATE LAYOUT_ENTRY=${_entry}
    LAYOUT_COMPILE_DEBUG=1)
  set_target_properties(${_control} PROPERTIES UNITY_BUILD OFF)
  foreach(_row _message IN ZIP_LISTS _rows_numbers _rows_expected)
    specfem_layout_compile_fail(matrix_${_row}_entry_${_entry}
      "${_layout_matrix}" ${_control} "${_message}"
      LAYOUT_BAD=${_row} LAYOUT_ENTRY=${_entry})
  endforeach()
endforeach()
