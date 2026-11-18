# Isolate Kokkos aborts, including GPU assertions, from the test runner.
# Usage: cmake -DPROGRAM=<exe> -DARGUMENT=<flag> -DEXPECTED=<regex> -P <this>
foreach(required PROGRAM ARGUMENT EXPECTED)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "check_device_failure.cmake requires -D${required}")
  endif()
endforeach()
execute_process(
  COMMAND "${PROGRAM}" "${ARGUMENT}"
  RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err
)
if(result STREQUAL "0")
  message(FATAL_ERROR "${PROGRAM} ${ARGUMENT} succeeded")
endif()
if(NOT "${out}\n${err}" MATCHES "${EXPECTED}")
  message(FATAL_ERROR "Expected device diagnostic missing: ${out}\n${err}")
endif()
