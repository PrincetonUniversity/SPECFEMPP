# Compile-failure tests for the requirement checks of PointwiseSurfaceMethod
# (core/specfem/injection/method/pointwise_surface_method.hpp). Each case is an
# object library left out of the default build; its ctest recompiles it with
# compile_fresh.cmake and passes when the compiler names the expected
# unsatisfied concept. VALID is the control and must compile. Skip them with `ctest -LE compile_fail`.
# Each message alternates the Clang wording with the GCC one ("required for
# the satisfaction of ‘Concept<...>’ [with Method = ...]").
# Included from tests/unit-tests/CMakeLists.txt.

set(_injection_requirement_cases
  "VALID@"
  "NO_NAME@does not satisfy 'NamedMethod'|satisfaction of [^ ]*NamedMethod<"
  "NO_COMPUTE@does not satisfy 'ComputesSomewhere'|satisfaction of [^ ]*ComputesSomewhere<"
  "QUANTITIES_WITHOUT_COMPUTE@'QuantitiesHaveCompute<[^>]*QuantitiesWithoutCompute|satisfaction of [^ ]*QuantitiesHaveCompute<[^ ]* [^ ]* .with Method = [^ ]*QuantitiesWithoutCompute"
  "WRONG_RETURN@does not satisfy 'ComputeReturnsAcceleration'|satisfaction of [^ ]*ComputeReturnsAcceleration<"
)

foreach(_case IN LISTS _injection_requirement_cases)
  string(REPLACE "@" ";" _case "${_case}")
  list(GET _case 0 _name)
  list(GET _case 1 _message)
  string(TOLOWER "injection_requirement_${_name}" _target)

  add_library(${_target} OBJECT EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_LIST_DIR}/requirement_failures.cpp)
  target_compile_definitions(${_target} PRIVATE CASE_${_name})
  set_target_properties(${_target} PROPERTIES UNITY_BUILD OFF)
  target_link_libraries(${_target} PRIVATE
    specfem::injection_methods
    Kokkos::kokkos
  )

  add_test(NAME ${_target}
    COMMAND ${CMAKE_COMMAND} -DBINARY_DIR=${CMAKE_BINARY_DIR}
            -DTARGET=${_target} "-DOBJECTS=$<TARGET_OBJECTS:${_target}>"
            -DCONFIG=$<CONFIG>
            -P ${CMAKE_CURRENT_LIST_DIR}/compile_fresh.cmake)
  set_tests_properties(${_target} PROPERTIES
    LABELS "injection;compile_fail"
    RESOURCE_LOCK injection_compile_fail
    TIMEOUT 600
  )
  if(_message)
    set_tests_properties(${_target} PROPERTIES
      PASS_REGULAR_EXPRESSION "${_message}")
  endif()
endforeach()
