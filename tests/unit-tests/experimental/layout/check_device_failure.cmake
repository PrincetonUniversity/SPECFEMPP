# Isolate Kokkos aborts, including GPU assertions, from the test runner.
execute_process(
  COMMAND "${PROGRAM}" --invalid-device-coordinate
  RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err
)
if(result STREQUAL "0")
  message(FATAL_ERROR "Invalid transformed coordinate succeeded")
endif()
if(NOT "${out}\n${err}" MATCHES "layout transformed coordinate out of bounds")
  message(FATAL_ERROR "Expected device diagnostic missing: ${out}\n${err}")
endif()
