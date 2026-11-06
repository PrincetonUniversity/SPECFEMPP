CMAKE_MINIMUM_REQUIRED(VERSION 3.17.5)

# This basically just gets the example name
get_filename_component(CURRENT_DIR_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)


if (NOT DEFINED SPECFEM_BENCHMARKS_BUILD_DIR)
    set(SPECFEM_BENCHMARKS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../build)
endif()

set(CURRENT_BENCHMARK_BUILD_DIR "${SPECFEM_BENCHMARKS_BUILD_DIR}/dim3/${CURRENT_DIR_NAME}")

# Copy static files into the benchmarks.
file(
  COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    ${CMAKE_CURRENT_SOURCE_DIR}/DATA
    ${CMAKE_CURRENT_SOURCE_DIR}/source.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/fk.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/../fk_injection_two_layer/compare_seismograms.py
  DESTINATION ${CURRENT_BENCHMARK_BUILD_DIR}
)

file(
  CREATE_LINK
    ${CMAKE_CURRENT_SOURCE_DIR}/reference_seismograms
    ${CURRENT_BENCHMARK_BUILD_DIR}/reference_seismograms
    SYMBOLIC
)

# Configure files directly into the cmake benchmarks_build!
configure_file(CMakeFiles/interfaces.txt.in ${CURRENT_BENCHMARK_BUILD_DIR}/DATA/meshfem3D_files/interfaces.txt)
configure_file(CMakeFiles/Mesh_Par_file.in ${CURRENT_BENCHMARK_BUILD_DIR}/DATA/meshfem3D_files/Mesh_Par_file)
configure_file(CMakeFiles/specfem_config.yaml.in ${CURRENT_BENCHMARK_BUILD_DIR}/specfem_config.yaml)
configure_file(CMakeFiles/Snakefile.in ${CURRENT_BENCHMARK_BUILD_DIR}/Snakefile)

file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/OUTPUT_FILES/DATABASES)
file(MAKE_DIRECTORY ${CURRENT_BENCHMARK_BUILD_DIR}/OUTPUT_FILES/results)
