
CMAKE_MINIMUM_REQUIRED(VERSION 3.17.5)

get_filename_component(CURRENT_DIR_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

if (NOT DEFINED SPECFEM_NIGHTLY_BENCHMARKS_BUILD_DIR)
    set(SPECFEM_NIGHTLY_BENCHMARKS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../build)
endif()

set(CURRENT_BENCHMARK_BUILD_DIR "${SPECFEM_NIGHTLY_BENCHMARKS_BUILD_DIR}/dim2/${CURRENT_DIR_NAME}")

if (SPECFEM_CPU_BUILD)
    set(TOPOGRAPHY_FILE "${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles/topography.cpu.in")
    set(PAR_FILE "${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles/Par_File.cpu.in")
    set(CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles/specfem_config.yaml.cpu.in")
endif()

if (SPECFEM_GPU_BUILD)
    set(TOPOGRAPHY_FILE "${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles/topography.gpu.in")
    set(PAR_FILE "${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles/Par_File.gpu.in")
    set(CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles/specfem_config.yaml.gpu.in")
endif()

# Configure files directly into the cmake benchmarks_build!
configure_file(${PAR_FILE} ${CURRENT_BENCHMARK_BUILD_DIR}/Par_File)
configure_file(${CONFIG_FILE} ${CURRENT_BENCHMARK_BUILD_DIR}/specfem_config.yaml)
configure_file(${TOPOGRAPHY_FILE} ${CURRENT_BENCHMARK_BUILD_DIR}/topography.dat)
configure_file(CMakeFiles/Snakefile.in ${CURRENT_BENCHMARK_BUILD_DIR}/Snakefile)

# Copy static files into the benchmarks.
file(
  COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/sources.yaml
    DESTINATION ${CURRENT_BENCHMARK_BUILD_DIR}
)
