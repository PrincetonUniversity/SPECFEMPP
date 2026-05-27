CMAKE_MINIMUM_REQUIRED(VERSION 3.17.5)

get_filename_component(CURRENT_DIR_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

if (NOT DEFINED SPECFEM_BENCHMARKS_BUILD_DIR)
    set(SPECFEM_BENCHMARKS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../build)
endif()

set(CURRENT_BENCHMARK_BUILD_DIR "${SPECFEM_BENCHMARKS_BUILD_DIR}/dim3/${CURRENT_DIR_NAME}")

# Copy static files into the benchmarks build directory.
file(
  COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    ${CMAKE_CURRENT_SOURCE_DIR}/DATA
    ${CMAKE_CURRENT_SOURCE_DIR}/forward_source.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/adjoint_source.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_adjoint_sources.py
    ${CMAKE_CURRENT_SOURCE_DIR}/plot.py
  DESTINATION ${CURRENT_BENCHMARK_BUILD_DIR}
)

# Configure files with paths substituted
configure_file(CMakeFiles/Mesh_Par_file.in ${CURRENT_BENCHMARK_BUILD_DIR}/DATA/meshfem3D_files/Mesh_Par_file)
configure_file(CMakeFiles/interfaces.txt.in ${CURRENT_BENCHMARK_BUILD_DIR}/DATA/meshfem3D_files/interfaces.txt)
configure_file(CMakeFiles/forward_config.yaml.in ${CURRENT_BENCHMARK_BUILD_DIR}/forward_config.yaml)
configure_file(CMakeFiles/adjoint_config.yaml.in ${CURRENT_BENCHMARK_BUILD_DIR}/adjoint_config.yaml)
configure_file(CMakeFiles/Snakefile.in ${CURRENT_BENCHMARK_BUILD_DIR}/Snakefile)

# Create required output directories
file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/OUTPUT_FILES/DATABASES)
file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/OUTPUT_FILES/results)
file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/adjoint_sources)
