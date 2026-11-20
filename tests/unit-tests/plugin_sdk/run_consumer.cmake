# Install the plugin_sdk component into a scratch prefix, then configure and
# build the consumer project against it. Proves the installed SDK is
# self-contained (no core targets, no Kokkos).
#
# Usage:
#   cmake -DBUILD_DIR=<main build> -DSOURCE_DIR=<consumer src> \
#         -DWORK_DIR=<scratch> -DGENERATOR=<gen> -DCXX_COMPILER=<cxx> \
#         -P run_consumer.cmake

foreach(_var BUILD_DIR SOURCE_DIR WORK_DIR GENERATOR CXX_COMPILER)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "run_consumer: ${_var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
set(_prefix "${WORK_DIR}/prefix")

function(run_step description)
    execute_process(COMMAND ${ARGN}
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _output
        ERROR_VARIABLE _output)
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "run_consumer: ${description} failed:\n${_output}")
    endif()
endfunction()

run_step("installing plugin_sdk"
    "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --component plugin_sdk
    --prefix "${_prefix}")

if(NOT EXISTS "${_prefix}/include/specfem/plugin/injection_plugin.hpp")
    message(FATAL_ERROR "run_consumer: SDK headers missing from ${_prefix}")
endif()

run_step("configuring the consumer"
    "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${WORK_DIR}/build"
    -G "${GENERATOR}"
    "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
    "-DCMAKE_PREFIX_PATH=${_prefix}"
    -DCMAKE_BUILD_TYPE=Release)

# Building runs the export check as a post-build step.
run_step("building the consumer"
    "${CMAKE_COMMAND}" --build "${WORK_DIR}/build")
