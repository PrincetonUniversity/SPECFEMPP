# Builds TARGET in BINARY_DIR after deleting its OBJECTS, so a case always
# recompiles: an object left by an earlier successful build would otherwise
# count as up to date. Prints the compiler output for ctest to match.
#
#   cmake -DBINARY_DIR=<dir> -DTARGET=<target> -DOBJECTS=<objs> [-DCONFIG=<c>]
#         -P compile_fresh.cmake

file(REMOVE ${OBJECTS})
set(_config_args)
if(CONFIG)
  set(_config_args --config ${CONFIG})
endif()
execute_process(
  COMMAND ${CMAKE_COMMAND} --build ${BINARY_DIR} --target ${TARGET}
          ${_config_args}
  OUTPUT_VARIABLE _output
  ERROR_VARIABLE _output
  RESULT_VARIABLE _result
)
message("${_output}")
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "Build of ${TARGET} failed")
endif()
