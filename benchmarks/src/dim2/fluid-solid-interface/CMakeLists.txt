CMAKE_MINIMUM_REQUIRED(VERSION 3.17.5)

# This basically jsut gets the example name
get_filename_component(CURRENT_DIR_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)


if (NOT DEFINED SPECFEM_BENCHMARKS_BUILD_DIR)
    set(SPECFEM_BENCHMARKS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../build)
endif()

set(CURRENT_BENCHMARK_BUILD_DIR "${SPECFEM_BENCHMARKS_BUILD_DIR}/dim2/${CURRENT_DIR_NAME}")

# Configure files directly into the cmake benchmarks_build!
configure_file(cpu/CMakeFiles/Par_File.in ${CURRENT_BENCHMARK_BUILD_DIR}/cpu/Par_File)
configure_file(cpu/CMakeFiles/specfem_config.yaml.in ${CURRENT_BENCHMARK_BUILD_DIR}/cpu/specfem_config.yaml)
configure_file(cpu/CMakeFiles/Snakefile.in ${CURRENT_BENCHMARK_BUILD_DIR}/cpu/Snakefile)

configure_file(gpu/CMakeFiles/Par_File.in ${CURRENT_BENCHMARK_BUILD_DIR}/gpu/Par_File)
configure_file(gpu/CMakeFiles/specfem_config.yaml.in ${CURRENT_BENCHMARK_BUILD_DIR}/gpu/specfem_config.yaml)
configure_file(gpu/CMakeFiles/Snakefile.in ${CURRENT_BENCHMARK_BUILD_DIR}/gpu/Snakefile)

# Copy static files into the benchmarks.
file(
  COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/sources.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
  DESTINATION ${CURRENT_BENCHMARK_BUILD_DIR}
)

file(
  COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/topography.dat
  DESTINATION ${CURRENT_BENCHMARK_BUILD_DIR}/cpu
)

file(
  COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/topography.dat
  DESTINATION ${CURRENT_BENCHMARK_BUILD_DIR}/gpu
)
