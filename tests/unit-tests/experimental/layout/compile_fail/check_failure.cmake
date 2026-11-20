# Build the control first so missing dependencies cannot masquerade as rejection.
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${BUILD_DIR}" --target layout_compile_control
  RESULT_VARIABLE control_result OUTPUT_VARIABLE control_out ERROR_VARIABLE control_err
)
if(NOT control_result EQUAL 0)
  message(FATAL_ERROR "Layout compile control failed:\n${control_out}\n${control_err}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${BUILD_DIR}" --target "${TARGET}"
  RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err
)
if(result EQUAL 0)
  message(FATAL_ERROR "Invalid layout compiled successfully: ${TARGET}")
endif()
set(diagnostic "${out}\n${err}")
if(diagnostic MATCHES "${EXPECTED}")
  return()
endif()

# NVCC omits the message argument when reporting a constexpr precondition
# failure. Require its abort trace and the failed operation instead.
if(diagnostic MATCHES "error: expression must have a constant value" AND
   diagnostic MATCHES "cannot call non-constexpr function \"Kokkos::abort\"")
  if(EXPECTED STREQUAL "overflow" AND
     diagnostic MATCHES "(span|product|add|multiply)\\(")
    return()
  elseif(EXPECTED STREQUAL "nonnull backing" AND
         diagnostic MATCHES "require\\(data != nullptr")
    return()
  endif()
endif()
message(FATAL_ERROR "Missing expected diagnostic ${EXPECTED}:\n${diagnostic}")
